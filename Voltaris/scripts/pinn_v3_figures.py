"""Regenerate figures for the v3 abstract using the warm-started PINN.

Outputs go to Voltaris/outputs/sciml_hybrid/:
  - bar_v3.png            — per-cell RMSE across all 3 makes
  - calb_grid_v3.png      — CALB trajectory grid (K=50)
  - rept_grid_v3.png      — REPT trajectory grid
  - eve_grid_v3.png       — EVE trajectory grid
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, "/home/hj/Desktop/PINNs")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from Voltaris.sciml.data_combined         import load_combined, feature_normaliser
from Voltaris.sciml.physics               import estimate_k_sei_from_window, physics_trajectory
from Voltaris.sciml.train_joint           import (JointConfig, JointPINN,
                                                    predict_full_trajectory_joint)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 50
OUT = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/sciml_hybrid")

cells, meta = load_combined(include_synth=False)
mean, std   = feature_normaliser(cells)
mean_s      = mean[:-1]; std_s = std[:-1]
n_shared    = len(cells[0].x_health) - 1
n_norm_scale = float(max(c.n_total for c in cells))
cfg = JointConfig(K=K, n_norm_scale=n_norm_scale, p_init=0.5)

model = JointPINN(n_cells=len(cells), n_shared_features=n_shared,
                    embed_dim=8, hidden=128, n_layers=5,
                    feat_mean=mean_s, feat_std=std_s, p_init=0.5)
model.load_state_dict(torch.load(OUT / "warmstart_K50.pt", map_location=DEVICE))
model.to(DEVICE).eval()

# Rebuild per-cell trajectories
per_make = {"CALB": {}, "REPT": {}, "EVE": {}}
for i, cell in enumerate(cells):
    m = meta[cell.cell_id]["make"]
    n = cell.n_traj.numpy(); s = cell.soh_traj.numpy()
    first_cy = float(n[0]); k_end = first_cy + K
    mask_te = n > k_end
    if mask_te.sum() < 3: continue
    soh_pred = predict_full_trajectory_joint(model, cell, i, cfg, DEVICE).numpy()
    k_L0 = estimate_k_sei_from_window(cell, K)
    n_t = torch.tensor(n, dtype=torch.float32)
    soh_phys = physics_trajectory(cell.soh_init, k_L0, n_t, first_cy).numpy()
    per_make[m][cell.cell_id.split("_")[-1]] = dict(
        n=n, s=s, soh_pred=soh_pred, soh_phys=soh_phys,
        first_cy=first_cy, k_end=k_end,
        rmse_pinn=float(np.sqrt(np.mean((soh_pred[mask_te]-s[mask_te])**2))) * 100,
        rmse_phys=float(np.sqrt(np.mean((soh_phys[mask_te]-s[mask_te])**2))) * 100,
    )


def _plot_one(ax, cid, t, make, ymin_pad=1.5, ymax_pad=1.5):
    n, s = t["n"], t["s"]
    first_cy, k_end = t["first_cy"], t["k_end"]
    soh_meas_pct = s * 100
    soh_pred_pct = t["soh_pred"] * 100
    soh_phys_pct = t["soh_phys"] * 100
    ymin = float(min(soh_meas_pct.min(), soh_pred_pct.min())) - ymin_pad
    ymax = float(max(soh_meas_pct.max(), soh_pred_pct.max())) + ymax_pad
    ax.set_ylim(ymin, ymax)
    ax.axvspan(k_end, n[-1], color="tab:blue", alpha=0.06)
    ax.axvspan(first_cy, k_end, color="tab:orange", alpha=0.12)
    ax.scatter(n, soh_meas_pct, s=4, color="black", alpha=0.30, label="Measured")
    ax.plot(n, soh_phys_pct, color="tab:red", lw=1.4, ls="--",
             label=f"phys {t['rmse_phys']:.2f} pp")
    ax.plot(n, soh_pred_pct, color="tab:green", lw=2.0,
             label=f"PINN {t['rmse_pinn']:.2f} pp")
    ax.axvline(k_end, color="dimgray", ls="--", lw=0.7)
    if soh_phys_pct[-1] < ymin:
        ax.annotate(f"phys → {soh_phys_pct[-1]:.1f}%",
                     xy=(n[-1], ymin + 0.5), fontsize=8, color="tab:red", ha="right")
    passer = "  ✓" if t["rmse_pinn"] < 3.0 else "  ✗"
    ax.set_title(f"{make} cell {cid}{passer}", fontsize=10)
    ax.set_xlabel("Cycle"); ax.set_ylabel("SoH [%]")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower left")


def make_grid(name, trajs, ncols=3):
    keys = sorted(trajs.keys())
    n = len(keys); nrows = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.3*ncols, 3.6*nrows))
    axs = np.array(axs).reshape(-1)
    for ax, cid in zip(axs, keys):
        _plot_one(ax, cid, trajs[cid], name)
    for ax in axs[n:]: ax.set_visible(False)
    med_pinn = np.median([trajs[k]["rmse_pinn"] for k in keys])
    med_phys = np.median([trajs[k]["rmse_phys"] for k in keys])
    n_pass   = sum(trajs[k]["rmse_pinn"] < 3.0 for k in keys)
    fig.suptitle(f"{name} — Universal PINN (warm-started) vs pure physics (K=50)\n"
                  f"PINN median {med_pinn:.2f} pp, phys median {med_phys:.2f} pp   ·   "
                  f"PINN under 3 pp: {n_pass}/{len(keys)}",
                  fontsize=12, y=1.005)
    fig.tight_layout()
    return fig


for name, trajs in per_make.items():
    fig = make_grid(name, trajs)
    outfile = OUT / f"{name.lower()}_grid_v3.png"
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outfile.name}")

# ── Bar chart ──
df = pd.read_csv(OUT / "warmstart_vs_baseline_K50.csv")
df_h = df[df.model == "warm"]
fig, ax = plt.subplots(figsize=(12, 4.5))
xoff = {"CALB": 0, "REPT": 10, "EVE": 20}
colors = {"CALB": "tab:red", "REPT": "tab:blue", "EVE": "tab:purple"}
for m in ["CALB", "REPT", "EVE"]:
    d = df_h[df_h.make == m].sort_values("cell_id").reset_index(drop=True)
    x = np.arange(len(d)) + xoff[m]
    ax.bar(x - 0.2, d.rmse_phys_pp, 0.4, color=colors[m], alpha=0.4, label=f"{m} phys")
    ax.bar(x + 0.2, d.rmse_pinn_pp, 0.4, color=colors[m], alpha=0.9,
            label=f"{m} PINN", edgecolor="black", linewidth=0.5)
ax.axhline(3.0, color="black", ls="--", lw=1.2, label="3 pp target")
ax.set_yscale("log")
ax.set_ylabel("Held-out RMSE [pp SoH]")
ax.set_title("Universal PINN (K=50): held-out RMSE per cell, "
             "all cells under 4 pp across three manufacturers")
xticks, xlabels = [], []
for m in ["CALB", "REPT", "EVE"]:
    d = df_h[df_h.make == m].sort_values("cell_id").reset_index(drop=True)
    for i, cid in enumerate(d.cell_id):
        xticks.append(i + xoff[m]); xlabels.append(f"{m[:1]}{str(cid).split('_')[-1]}")
ax.set_xticks(xticks); ax.set_xticklabels(xlabels, rotation=45, fontsize=8)
ax.grid(alpha=0.3, axis="y", which="both")
ax.legend(fontsize=8, ncol=4, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "bar_v3.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote bar_v3.png")
