"""Path A — L2 SEI+LAM physics prior at K=100.

Goal: fix cell 19 (delayed acceleration) so all 7/7 clean cells cross
the 3 pp target at K=100. That gives the '4× reduction vs K=400 physics'
headline.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, "/home/hj/Desktop/PINNs")

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from Voltaris.sciml.data          import load_all, feature_normaliser, CLEAN_IDS
from Voltaris.sciml.physics       import estimate_k_sei_from_window, physics_trajectory
from Voltaris.sciml.train_joint_L2 import (L2Config, JointPINN_L2, train_joint_L2, predict_L2)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 100
OUT = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/sciml_day2")
OUT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42); np.random.seed(42)

print(f"=== Path A: L2 SEI+LAM physics prior, joint PINN, 7 clean cells, K={K} ===")
print(f"Device: {DEVICE}\n")

all_cells = load_all()
cells = [c for c in all_cells if c.cell_id in CLEAN_IDS]
mean, std = feature_normaliser(all_cells)
mean_shared = mean[:-1]; std_shared = std[:-1]
n_shared = len(cells[0].x_health) - 1

model = JointPINN_L2(n_cells=len(cells), n_shared_features=n_shared,
                      embed_dim=4, hidden=64, n_layers=4,
                      feat_mean=mean_shared, feat_std=std_shared)

cfg = L2Config(K=K, epochs=6000, lr=1e-3, lam_phys=1.0, lam_mono=0.05,
                n_norm_scale=float(max(c.n_total for c in cells)),
                n_col_per_cell=200, p_init=0.5, verbose_every=2000)

t0 = time.time()
res = train_joint_L2(model, cells, cfg, DEVICE)
print(f"\nTraining took {time.time()-t0:.1f}s")
print(f"\nLearned per-cell L2 parameters:")
for i, c in enumerate(cells):
    print(f"  cell {c.cell_id:>2}:  k_SEI={res['params']['k_sei'][i]:.2e}  "
          f"p={res['params']['p'][i]:.3f}  "
          f"k_LAM={res['params']['k_lam'][i]:.2e}  "
          f"n_c={res['params']['n_c'][i]:.0f}  "
          f"tau={res['params']['tau'][i]:.0f}")

results = []; trajectories = {}
for i, cell in enumerate(cells):
    n = cell.n_traj.numpy(); s = cell.soh_traj.numpy()
    first_cy = float(n[0])
    k_end = first_cy + K
    mask_te = n > k_end

    soh_pred = predict_L2(model, cell, i, cfg, DEVICE).numpy()
    k_L0 = estimate_k_sei_from_window(cell, K)
    n_t = torch.tensor(n, dtype=torch.float32)
    soh_phys = physics_trajectory(cell.soh_init, k_L0, n_t, first_cy).numpy()

    rmse_pinn = float(np.sqrt(np.mean((soh_pred[mask_te] - s[mask_te])**2))) * 100
    rmse_phys = float(np.sqrt(np.mean((soh_phys[mask_te] - s[mask_te])**2))) * 100
    results.append(dict(cell_id=cell.cell_id, K_train_cy=K,
                         rmse_pinn_test_pp=rmse_pinn,
                         rmse_phys_test_pp=rmse_phys,
                         k_sei=res['params']['k_sei'][i],
                         p=res['params']['p'][i],
                         k_lam=res['params']['k_lam'][i],
                         n_c=res['params']['n_c'][i],
                         tau=res['params']['tau'][i]))
    trajectories[cell.cell_id] = (n, s, soh_pred, soh_phys, first_cy, k_end)

df = pd.DataFrame(results)
df.to_csv(OUT / "pathA_L2_K100.csv", index=False)

import pickle
with open(OUT / "pathA_L2_trajectories.pkl", "wb") as f:
    pickle.dump(trajectories, f)

print(f"\n{'='*66}")
print(f"{'cell':>5}  {'L2 PINN test':>13}  {'phys test':>10}  {'winner':>8}  {'<3pp?':>6}")
print(f"{'='*66}")
for r in results:
    winner = "PINN" if r['rmse_pinn_test_pp'] < r['rmse_phys_test_pp'] else "phys"
    passer = "yes" if r['rmse_pinn_test_pp'] < 3.0 else "NO"
    print(f"{r['cell_id']:>5}  {r['rmse_pinn_test_pp']:>10.3f} pp  "
          f"{r['rmse_phys_test_pp']:>7.3f} pp  {winner:>8}  {passer:>6}")

n_pass = int((df.rmse_pinn_test_pp < 3.0).sum())
print(f"\nMedian PINN: {df.rmse_pinn_test_pp.median():.3f} pp")
print(f"Median phys: {df.rmse_phys_test_pp.median():.3f} pp")
print(f"PINN cells under 3pp: {n_pass}/7")
print(f"phys cells under 3pp: {int((df.rmse_phys_test_pp<3.0).sum())}/7")

if n_pass == 7:
    print(f"\n*** PATH A ACHIEVED: 7/7 cells under 3pp at K=100 ***")
elif n_pass == 6:
    print(f"\n>>> PATH A close: 6/7. Which cell fails? "
          f"{df[df.rmse_pinn_test_pp >= 3.0].cell_id.tolist()}")
