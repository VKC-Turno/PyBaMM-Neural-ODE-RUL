"""Full K-sweep for joint PINN on 7 clean CALB cells.

K ∈ {50, 100, 200, 400}. Compares PINN test-RMSE against the
pure-physics baseline used by the current abstract.

Emits Voltaris/outputs/sciml_day2/ksweep_summary.csv with per-cell
per-K numbers so both notebooks can plot from the same source.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, "/home/hj/Desktop/PINNs")

import numpy as np
import pandas as pd
import torch

from Voltaris.sciml.data       import load_all, feature_normaliser, CLEAN_IDS
from Voltaris.sciml.physics    import estimate_k_sei_from_window, physics_trajectory
from Voltaris.sciml.train_joint import (JointConfig, JointPINN, train_joint,
                                          predict_full_trajectory_joint)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_VALUES = [50, 100, 200, 400]
OUT = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/sciml_day2")
OUT.mkdir(parents=True, exist_ok=True)

print(f"=== K-sweep — Joint PINN on 7 clean CALB cells ===")
print(f"K values: {K_VALUES}")
print(f"Device: {DEVICE}\n")

all_cells = load_all()
cells = [c for c in all_cells if c.cell_id in CLEAN_IDS]
mean, std = feature_normaliser(all_cells)
mean_shared = mean[:-1]; std_shared = std[:-1]
n_shared = len(cells[0].x_health) - 1

rows = []
trajectories = {}
t_top = time.time()

for K in K_VALUES:
    print(f"\n--- K = {K} ---")
    torch.manual_seed(42); np.random.seed(42)

    model = JointPINN(n_cells=len(cells), n_shared_features=n_shared,
                        embed_dim=4, hidden=64, n_layers=4,
                        feat_mean=mean_shared, feat_std=std_shared,
                        p_init=0.5)

    # More epochs for smaller K (fewer data points → needs more time)
    epochs = 6000 if K <= 100 else 4000
    cfg = JointConfig(K=K, epochs=epochs, lr=1e-3, lam_phys=1.0, lam_mono=0.05,
                        n_norm_scale=float(max(c.n_total for c in cells)),
                        n_col_per_cell=200, p_init=0.5, verbose_every=10000)

    t0 = time.time()
    train_result = train_joint(model, cells, cfg, DEVICE)
    train_secs = time.time() - t0

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
        rows.append(dict(
            cell_id=cell.cell_id, K_train_cy=K, epochs=epochs,
            n_test=int(mask_te.sum()),
            rmse_pinn_test_pp=rmse_pinn,
            rmse_phys_test_pp=rmse_phys,
            k_sei_learned=train_result['k_sei_final'][i],
            p_learned=train_result['p_final'][i],
            train_secs=train_secs,
        ))
        trajectories[(K, cell.cell_id)] = (n, s, soh_pred, soh_phys,
                                             first_cy, k_end)

    kdf = pd.DataFrame([r for r in rows if r['K_train_cy'] == K])
    print(f"  {train_secs:.1f}s.  Median PINN={kdf.rmse_pinn_test_pp.median():.3f} pp  "
          f"phys={kdf.rmse_phys_test_pp.median():.3f} pp  "
          f"PINN <3pp: {int((kdf.rmse_pinn_test_pp<3.0).sum())}/7  "
          f"phys <3pp: {int((kdf.rmse_phys_test_pp<3.0).sum())}/7")

df = pd.DataFrame(rows)
df.to_csv(OUT / "ksweep_summary.csv", index=False)

# Also save trajectories for notebook plotting
import pickle
with open(OUT / "ksweep_trajectories.pkl", "wb") as f:
    pickle.dump(trajectories, f)

print(f"\nTotal wall-time: {time.time() - t_top:.1f}s")
print(f"CSV : {OUT / 'ksweep_summary.csv'}")
print(f"Traj: {OUT / 'ksweep_trajectories.pkl'}")

# Summary table
print("\n" + "="*60)
print("Summary table")
print("="*60)
for K in K_VALUES:
    kdf = df[df.K_train_cy == K]
    print(f"K={K:>3}:  PINN median={kdf.rmse_pinn_test_pp.median():5.2f}  "
          f"phys median={kdf.rmse_phys_test_pp.median():5.2f}  "
          f"PINN <3pp={int((kdf.rmse_pinn_test_pp<3.0).sum())}/7  "
          f"phys <3pp={int((kdf.rmse_phys_test_pp<3.0).sum())}/7")
