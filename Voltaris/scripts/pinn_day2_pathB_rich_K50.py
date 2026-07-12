"""Path B config with RICH 12-feature characterisation input at K=50.

Hypothesis: extra features (SoH curvature, early-life slope, DoD range,
capacity, c/d-rate) give cells 6 and 19 the signal needed to cross 3 pp
at K=50. Both currently sit at 3.79 and 3.87 pp with basic 4-feature input.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, "/home/hj/Desktop/PINNs")

import numpy as np
import pandas as pd
import torch

from Voltaris.sciml.data_rich  import load_all_rich, normaliser_rich, CLEAN_IDS, RICH_KEYS
from Voltaris.sciml.physics    import estimate_k_sei_from_window, physics_trajectory
from Voltaris.sciml.train_joint import (JointConfig, JointPINN, train_joint,
                                          predict_full_trajectory_joint)
from Voltaris.sciml.data       import CellData  # for compat


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 50
OUT = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/sciml_day2")
torch.manual_seed(42); np.random.seed(42)

print(f"=== Path B + rich features (12) at K={K} ===")
print(f"Features: {RICH_KEYS}")
print(f"Device: {DEVICE}\n")

rich_cells = load_all_rich(CLEAN_IDS, K=K)
# Adapt to CellData interface used by train_joint
cells = [CellData(cell_id=c.cell_id, is_clean=c.is_clean,
                   n_traj=c.n_traj, soh_traj=c.soh_traj,
                   x_health=c.x_health, soh_init=c.soh_init,
                   n_total=c.n_total)
         for c in rich_cells]
mean, std = normaliser_rich(rich_cells)
mean_shared = mean[:-1]; std_shared = std[:-1]
n_shared = len(rich_cells[0].x_health) - 1
print(f"n_shared_features: {n_shared}")

model = JointPINN(n_cells=len(cells), n_shared_features=n_shared,
                    embed_dim=8, hidden=128, n_layers=5,
                    feat_mean=mean_shared, feat_std=std_shared,
                    p_init=0.5)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

cfg = JointConfig(K=K, epochs=10000, lr=1e-3, lam_phys=2.0, lam_mono=0.05,
                    n_norm_scale=float(max(c.n_total for c in cells)),
                    n_col_per_cell=400, p_init=0.5, verbose_every=2500)

t0 = time.time()
tr = train_joint(model, cells, cfg, DEVICE)
print(f"\nTraining {time.time()-t0:.1f}s\n")

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
                         k_sei=tr['k_sei_final'][i],
                         p=tr['p_final'][i]))
    trajectories[cell.cell_id] = (n, s, soh_pred, soh_phys, first_cy, k_end)

df = pd.DataFrame(results)
df.to_csv(OUT / "pathB_rich_K50.csv", index=False)
import pickle
with open(OUT / "pathB_rich_K50_trajectories.pkl", "wb") as f:
    pickle.dump(trajectories, f)

print(f"{'='*70}")
print(f"{'cell':>5}  {'PINN (rich)':>12}  {'phys':>8}  {'winner':>7}  {'<3pp?':>6}")
print(f"{'='*70}")
for r in results:
    w = "PINN" if r['rmse_pinn_test_pp'] < r['rmse_phys_test_pp'] else "phys"
    p = "yes" if r['rmse_pinn_test_pp'] < 3.0 else "NO"
    print(f"{r['cell_id']:>5}  {r['rmse_pinn_test_pp']:>9.3f} pp  "
          f"{r['rmse_phys_test_pp']:>5.3f} pp  {w:>7}  {p:>6}")

n_pass = int((df.rmse_pinn_test_pp < 3.0).sum())
print(f"\nMedian PINN: {df.rmse_pinn_test_pp.median():.3f} pp")
print(f"PINN <3pp: {n_pass}/7   (Path B basic was 5/7)")
