"""Make-agnostic result figure — 3-manufacturer summary + trajectory grid.

Recomputes trajectories for each cell (needed for the notebook grid) and
emits a combined figure showing PINN vs physics across CALB/REPT/EVE.
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path

sys.path.insert(0, "/home/hj/Desktop/PINNs")

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from Voltaris.sciml.data       import load_all as load_calb, feature_normaliser as norm_calb, CLEAN_IDS as CALB_IDS
from Voltaris.sciml.data_rept  import load_rept, normaliser_rept
from Voltaris.sciml.data_eve   import load_eve,  normaliser_eve
from Voltaris.sciml.physics    import estimate_k_sei_from_window, physics_trajectory
from Voltaris.sciml.train_joint import (JointConfig, JointPINN, train_joint,
                                          predict_full_trajectory_joint)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 50
OUT = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/sciml_day2")

def path_b(n_scale):
    return JointConfig(K=K, epochs=10000, lr=1e-3, lam_phys=2.0, lam_mono=0.05,
                        n_norm_scale=n_scale, n_col_per_cell=400,
                        p_init=0.5, verbose_every=99999)


def train_and_predict(name, cells, mean, std):
    print(f"\n=== {name} — {len(cells)} cells ===")
    torch.manual_seed(42); np.random.seed(42)
    mean_s = mean[:-1]; std_s = std[:-1]
    n_shared = len(cells[0].x_health) - 1

    model = JointPINN(n_cells=len(cells), n_shared_features=n_shared,
                        embed_dim=8, hidden=128, n_layers=5,
                        feat_mean=mean_s, feat_std=std_s, p_init=0.5)
    cfg = path_b(float(max(c.n_total for c in cells)))
    t0 = time.time()
    train_joint(model, cells, cfg, DEVICE)
    print(f"  Trained {time.time()-t0:.1f}s")

    trajs = {}
    for i, cell in enumerate(cells):
        n = cell.n_traj.numpy(); s = cell.soh_traj.numpy()
        first_cy = float(n[0]); k_end = first_cy + K
        mask_te = n > k_end
        if mask_te.sum() < 3: continue
        soh_pred = predict_full_trajectory_joint(model, cell, i, cfg, DEVICE).numpy()
        k_L0 = estimate_k_sei_from_window(cell, K)
        n_t = torch.tensor(n, dtype=torch.float32)
        soh_phys = physics_trajectory(cell.soh_init, k_L0, n_t, first_cy).numpy()
        rmse_pinn = float(np.sqrt(np.mean((soh_pred[mask_te] - s[mask_te])**2))) * 100
        rmse_phys = float(np.sqrt(np.mean((soh_phys[mask_te] - s[mask_te])**2))) * 100
        trajs[cell.cell_id] = dict(n=n, s=s, soh_pred=soh_pred, soh_phys=soh_phys,
                                    first_cy=first_cy, k_end=k_end,
                                    rmse_pinn=rmse_pinn, rmse_phys=rmse_phys)
    return trajs


# Run 3 manufacturers
calb = [c for c in load_calb() if c.cell_id in CALB_IDS]
mean, std = norm_calb(load_calb())
trajs_calb = train_and_predict("CALB", calb, mean, std)

rept = load_rept()
mean, std = normaliser_rept(rept)
trajs_rept = train_and_predict("REPT", rept, mean, std)

eve = load_eve()
mean, std = normaliser_eve(eve)
trajs_eve = train_and_predict("EVE",  eve,  mean, std)

# Save trajectories
with open(OUT / "make_agnostic_trajectories.pkl", "wb") as f:
    pickle.dump({"CALB": trajs_calb, "REPT": trajs_rept, "EVE": trajs_eve}, f)
print(f"\nTrajectories: {OUT / 'make_agnostic_trajectories.pkl'}")

# Combined grid — one column per manufacturer, cells stacked vertically
n_calb, n_rept, n_eve = len(trajs_calb), len(trajs_rept), len(trajs_eve)
max_rows = max(n_calb, n_rept, n_eve)

fig, axs = plt.subplots(max_rows, 3, figsize=(16, 2.4*max_rows))
if max_rows == 1: axs = axs[None, :]

for col, (name, trajs) in enumerate([("CALB", trajs_calb),
                                       ("REPT", trajs_rept),
                                       ("EVE",  trajs_eve)]):
    for row, (cid, t) in enumerate(sorted(trajs.items())):
        ax = axs[row, col]
        n, s = t["n"], t["s"]
        first_cy, k_end = t["first_cy"], t["k_end"]
        ax.axvspan(k_end, n[-1], color="tab:blue", alpha=0.06)
        ax.axvspan(first_cy, k_end, color="tab:orange", alpha=0.12)
        ax.scatter(n, s*100, s=3, color="black", alpha=0.25)
        ax.plot(n, t["soh_phys"]*100, color="tab:red", lw=1.2, ls="--",
                 label=f"phys {t['rmse_phys']:.2f}pp")
        ax.plot(n, t["soh_pred"]*100, color="tab:green", lw=1.5,
                 label=f"PINN {t['rmse_pinn']:.2f}pp")
        ax.axvline(k_end, color="dimgray", ls="--", lw=0.6)
        ax.set_title(f"{name} cell {cid}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")
    for row in range(len(trajs), max_rows):
        axs[row, col].set_visible(False)

fig.suptitle("Make-agnostic joint PINN — K=50, same architecture across CALB / REPT / EVE",
              fontsize=13, y=1.005)
fig.tight_layout()
fig.savefig(OUT / "make_agnostic_grid.png", dpi=130)
print(f"Figure: {OUT / 'make_agnostic_grid.png'}")
