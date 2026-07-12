"""Path B config extended to ALL 9 cells (7 clean + 2 batch-artefact).

Does the joint PINN with per-cell embedding rescue cells 24, 30 that
pure physics has to exclude via shape filter?
"""
from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, "/home/hj/Desktop/PINNs")

import numpy as np
import pandas as pd
import torch

from Voltaris.sciml.data       import load_all, feature_normaliser, ALL_IDS
from Voltaris.sciml.physics    import estimate_k_sei_from_window, physics_trajectory
from Voltaris.sciml.train_joint import (JointConfig, JointPINN, train_joint,
                                          predict_full_trajectory_joint)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 50
OUT = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/sciml_day2")

torch.manual_seed(42); np.random.seed(42)

print(f"=== Path B — all 9 CALB cells (7 clean + 2 DIRTY), K={K} ===")
print(f"Device: {DEVICE}\n")

all_cells = load_all()
mean, std = feature_normaliser(all_cells)
mean_shared = mean[:-1]; std_shared = std[:-1]
n_shared = len(all_cells[0].x_health) - 1

model = JointPINN(n_cells=len(all_cells), n_shared_features=n_shared,
                    embed_dim=8, hidden=128, n_layers=5,
                    feat_mean=mean_shared, feat_std=std_shared,
                    p_init=0.5)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

cfg = JointConfig(K=K, epochs=10000, lr=1e-3, lam_phys=2.0, lam_mono=0.05,
                    n_norm_scale=float(max(c.n_total for c in all_cells)),
                    n_col_per_cell=400, p_init=0.5, verbose_every=2500)

t0 = time.time()
train_result = train_joint(model, all_cells, cfg, DEVICE)
print(f"\nTraining took {time.time()-t0:.1f}s\n")

results = []; trajectories = {}
for i, cell in enumerate(all_cells):
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
    results.append(dict(cell_id=cell.cell_id, is_clean=cell.is_clean,
                         K_train_cy=K,
                         rmse_pinn_test_pp=rmse_pinn,
                         rmse_phys_test_pp=rmse_phys,
                         k_sei=train_result['k_sei_final'][i],
                         p=train_result['p_final'][i]))
    trajectories[cell.cell_id] = (n, s, soh_pred, soh_phys, first_cy, k_end, cell.is_clean)

df = pd.DataFrame(results)
df.to_csv(OUT / "pathB_9cell_K50.csv", index=False)
import pickle
with open(OUT / "pathB_9cell_trajectories.pkl", "wb") as f:
    pickle.dump(trajectories, f)

print(f"{'='*74}")
print(f"{'cell':>5} {'tag':>7}  {'PINN test':>10}  {'phys test':>10}  {'PINN <3pp':>10}  {'phys <3pp':>10}")
print(f"{'='*74}")
for r in results:
    tag = "clean" if r['is_clean'] else "DIRTY"
    p = "yes" if r['rmse_pinn_test_pp'] < 3.0 else "NO"
    q = "yes" if r['rmse_phys_test_pp'] < 3.0 else "NO"
    print(f"{r['cell_id']:>5} {tag:>7}  {r['rmse_pinn_test_pp']:>7.3f} pp  "
          f"{r['rmse_phys_test_pp']:>7.3f} pp  {p:>10}  {q:>10}")

n_clean = int(((df.rmse_pinn_test_pp<3.0) & df.is_clean).sum())
n_dirty = int(((df.rmse_pinn_test_pp<3.0) & ~df.is_clean).sum())
print(f"\nClean cells under 3pp:  PINN {n_clean}/7   phys {int(((df.rmse_phys_test_pp<3.0)&df.is_clean).sum())}/7")
print(f"DIRTY cells under 3pp:  PINN {n_dirty}/2   phys {int(((df.rmse_phys_test_pp<3.0)&~df.is_clean).sum())}/2")
print(f"ALL cells under 3pp:    PINN {int((df.rmse_pinn_test_pp<3.0).sum())}/9   phys {int((df.rmse_phys_test_pp<3.0).sum())}/9")
print(f"Median PINN: {df.rmse_pinn_test_pp.median():.3f} pp")
print(f"Median phys: {df.rmse_phys_test_pp.median():.3f} pp")
