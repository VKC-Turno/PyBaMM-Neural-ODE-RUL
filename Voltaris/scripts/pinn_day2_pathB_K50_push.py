"""Path B — aggressive push for K=50.

Change vs Day 2 baseline at K=50:
- Larger network (hidden=128, n_layers=5)
- Longer training (10000 epochs)
- Larger embedding dim (8)
- Higher lam_phys (2.0) — enforce physics stronger with less data
- More collocation points (400 per cell)
"""
from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, "/home/hj/Desktop/PINNs")

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from Voltaris.sciml.data       import load_all, feature_normaliser, CLEAN_IDS
from Voltaris.sciml.physics    import estimate_k_sei_from_window, physics_trajectory
from Voltaris.sciml.train_joint import (JointConfig, JointPINN, train_joint,
                                          predict_full_trajectory_joint)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 50
OUT = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/sciml_day2")
OUT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42); np.random.seed(42)

print(f"=== Path B: aggressive PINN push, K={K} ===")
print(f"Larger net, longer training, more collocation")
print(f"Device: {DEVICE}\n")

all_cells = load_all()
cells = [c for c in all_cells if c.cell_id in CLEAN_IDS]
mean, std = feature_normaliser(all_cells)
mean_shared = mean[:-1]; std_shared = std[:-1]
n_shared = len(cells[0].x_health) - 1

model = JointPINN(n_cells=len(cells), n_shared_features=n_shared,
                    embed_dim=8, hidden=128, n_layers=5,
                    feat_mean=mean_shared, feat_std=std_shared,
                    p_init=0.5)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

cfg = JointConfig(K=K, epochs=10000, lr=1e-3, lam_phys=2.0, lam_mono=0.05,
                    n_norm_scale=float(max(c.n_total for c in cells)),
                    n_col_per_cell=400, p_init=0.5, verbose_every=2500)

t0 = time.time()
train_result = train_joint(model, cells, cfg, DEVICE)
print(f"\nTraining took {time.time()-t0:.1f}s\n")

results = []; trajectories = {}
for i, cell in enumerate(cells):
    n = cell.n_traj.numpy(); s = cell.soh_traj.numpy()
    first_cy = float(n[0])
    k_end = first_cy + K
    mask_te = n > k_end

    soh_pred = predict_full_trajectory_joint(model, cell, i, cfg, DEVICE).numpy()
    k_L0 = estimate_k_sei_from_window(cell, K)
    n_t = torch.tensor(n, dtype=torch.float32)
    soh_phys = physics_trajectory(cell.soh_init, k_L0, n_t, first_cy).numpy()

    rmse_pinn = float(np.sqrt(np.mean((soh_pred[mask_te] - s[mask_te])**2))) * 100
    rmse_phys = float(np.sqrt(np.mean((soh_phys[mask_te] - s[mask_te])**2))) * 100
    results.append(dict(cell_id=cell.cell_id, K_train_cy=K,
                         rmse_pinn_test_pp=rmse_pinn,
                         rmse_phys_test_pp=rmse_phys,
                         k_sei=train_result['k_sei_final'][i],
                         p=train_result['p_final'][i]))
    trajectories[cell.cell_id] = (n, s, soh_pred, soh_phys, first_cy, k_end)

df = pd.DataFrame(results)
df.to_csv(OUT / "pathB_K50_push.csv", index=False)
import pickle
with open(OUT / "pathB_K50_trajectories.pkl", "wb") as f:
    pickle.dump(trajectories, f)

print(f"\n{'='*66}")
print(f"{'cell':>5}  {'B PINN test':>13}  {'phys test':>10}  {'winner':>8}  {'<3pp?':>6}")
print(f"{'='*66}")
for r in results:
    winner = "PINN" if r['rmse_pinn_test_pp'] < r['rmse_phys_test_pp'] else "phys"
    passer = "yes" if r['rmse_pinn_test_pp'] < 3.0 else "NO"
    print(f"{r['cell_id']:>5}  {r['rmse_pinn_test_pp']:>10.3f} pp  "
          f"{r['rmse_phys_test_pp']:>7.3f} pp  {winner:>8}  {passer:>6}")

n_pass = int((df.rmse_pinn_test_pp < 3.0).sum())
print(f"\nMedian PINN: {df.rmse_pinn_test_pp.median():.3f} pp")
print(f"Median phys: {df.rmse_phys_test_pp.median():.3f} pp")
print(f"PINN cells under 3pp: {n_pass}/7   (target: 5+ for Path B win)")
