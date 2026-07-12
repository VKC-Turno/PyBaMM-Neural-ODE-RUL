"""Day 2 B — Joint PINN training on all 7 clean CALB cells at K=100.

Uses the learnable-parameter formulation:
- Per-cell log(k_SEI) trained end-to-end alongside the NN
- Shared SoH-dependence exponent p ∈ [0.1, 0.9], initialized at 0.5
- Physics loss: MSE(dNN/dn, -k_SEI(i) · SoH^p) on collocation points

The joint model should transfer patterns from well-fitting cells (20, 25)
to hard cells (6, 7) via cell embedding + shared p.
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
K = 100
OUT = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/sciml_day2")
OUT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42); np.random.seed(42)

print(f"=== Day 2 — Joint PINN training on {len(CLEAN_IDS)} clean cells, K={K} ===")
print(f"Device: {DEVICE}\n")

all_cells = load_all()
cells = [c for c in all_cells if c.cell_id in CLEAN_IDS]
mean, std = feature_normaliser(all_cells)
mean_shared = mean[:-1]; std_shared = std[:-1]

n_shared = len(cells[0].x_health) - 1
model = JointPINN(n_cells=len(cells), n_shared_features=n_shared,
                    embed_dim=4, hidden=64, n_layers=4,
                    feat_mean=mean_shared, feat_std=std_shared,
                    p_init=0.5)
print(f"Joint PINN params: {sum(p.numel() for p in model.parameters()):,}")
print(f"  cells: {[c.cell_id for c in cells]}\n")

cfg = JointConfig(K=K, epochs=4000, lr=1e-3, lam_phys=1.0, lam_mono=0.05,
                    n_norm_scale=float(max(c.n_total for c in cells)),
                    n_col_per_cell=150, p_init=0.5, verbose_every=800)

t0 = time.time()
train_result = train_joint(model, cells, cfg, DEVICE)
train_secs = time.time() - t0

print(f"\nTraining took {train_secs:.1f}s")
print(f"Learned per-cell physics parameters:")
for i, c in enumerate(cells):
    print(f"  cell {c.cell_id:>2}:  k_SEI = {train_result['k_sei_final'][i]:.4e}   "
          f"p = {train_result['p_final'][i]:.3f}")

# Evaluate per-cell
results = []; trajectories = {}
for i, cell in enumerate(cells):
    n = cell.n_traj.numpy(); s = cell.soh_traj.numpy()
    first_cy = float(n[0])
    k_end = first_cy + K
    mask_tr = n <= k_end
    mask_te = n >  k_end

    soh_pred = predict_full_trajectory_joint(model, cell, i, cfg, DEVICE).numpy()

    # Baseline: Day-1 style linear pure-physics
    k_L0 = estimate_k_sei_from_window(cell, K)
    n_t  = torch.tensor(n, dtype=torch.float32)
    soh_phys = physics_trajectory(cell.soh_init, k_L0, n_t, first_cy).numpy()

    rmse_pinn_tr = float(np.sqrt(np.mean((soh_pred[mask_tr] - s[mask_tr])**2))) * 100
    rmse_pinn_te = float(np.sqrt(np.mean((soh_pred[mask_te] - s[mask_te])**2))) * 100
    rmse_phys_te = float(np.sqrt(np.mean((soh_phys[mask_te] - s[mask_te])**2))) * 100
    delta_end_pinn = float((soh_pred[-1] - s[-1]) * 100)
    delta_end_phys = float((soh_phys[-1] - s[-1]) * 100)

    results.append(dict(
        cell_id=cell.cell_id, K_train_cy=K, n_total=cell.n_total,
        n_train=int(mask_tr.sum()), n_test=int(mask_te.sum()),
        k_sei_learned=train_result['k_sei_final'][i],
        p_learned=train_result['p_final'][i],
        rmse_pinn_train_pp=rmse_pinn_tr,
        rmse_pinn_test_pp=rmse_pinn_te,
        rmse_phys_test_pp=rmse_phys_te,
        delta_end_pinn_pp=delta_end_pinn,
        delta_end_phys_pp=delta_end_phys,
    ))
    trajectories[cell.cell_id] = (n, s, soh_pred, soh_phys, first_cy, k_end)

df = pd.DataFrame(results)
df.to_csv(OUT / "joint_K100_summary.csv", index=False)

# ── Print head-to-head ──
print("\n" + "="*72)
print(f"{'cell':>5}  {'K':>4}  {'PINN train':>11}  {'PINN test':>10}  "
      f"{'phys test':>10}  {'winner':>7}")
print("="*72)
for r in results:
    winner = "PINN" if r['rmse_pinn_test_pp'] < r['rmse_phys_test_pp'] else "phys"
    print(f"{r['cell_id']:>5}  {r['K_train_cy']:>4}  "
          f"{r['rmse_pinn_train_pp']:>8.3f} pp  {r['rmse_pinn_test_pp']:>7.3f} pp  "
          f"{r['rmse_phys_test_pp']:>7.3f} pp  {winner:>7}")

print(f"\nMedian PINN test: {df.rmse_pinn_test_pp.median():.3f} pp")
print(f"Median phys test: {df.rmse_phys_test_pp.median():.3f} pp")
print(f"PINN wins:        {int((df.rmse_pinn_test_pp < df.rmse_phys_test_pp).sum())}/{len(df)}")
print(f"PINN under 3 pp:  {int((df.rmse_pinn_test_pp < 3.0).sum())}/{len(df)}")
print(f"phys under 3 pp:  {int((df.rmse_phys_test_pp < 3.0).sum())}/{len(df)}")

# ── Grid plot ──
fig, axs = plt.subplots(3, 3, figsize=(16, 11))
axs = axs.flatten()
for ax, (cid, (n, s, soh_pred, soh_phys, first_cy, k_end)) in zip(axs, trajectories.items()):
    r = df[df.cell_id == cid].iloc[0]
    ax.axvspan(k_end, n[-1], color="tab:blue", alpha=0.06)
    ax.axvspan(first_cy, k_end, color="tab:orange", alpha=0.08)
    ax.scatter(n, s*100, s=3, color="black", alpha=0.25, label="Measured")
    ax.plot(n, soh_phys*100, color="tab:red", lw=1.5, ls="--",
            label=f"L0 physics ({r['rmse_phys_test_pp']:.2f} pp)")
    ax.plot(n, soh_pred*100, color="tab:green", lw=1.8,
            label=f"Joint PINN ({r['rmse_pinn_test_pp']:.2f} pp)")
    ax.axvline(k_end, color="dimgray", ls="--", lw=0.7)
    ax.set_title(f"cell {cid}", fontsize=11)
    ax.set_xlabel("Cycle"); ax.set_ylabel("SoH [%]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
for ax in axs[len(trajectories):]:
    ax.set_visible(False)
fig.suptitle(f"Day 2 — Joint PINN (per-cell k_SEI + p) vs pure-physics, K={K}",
             fontsize=13, y=1.005)
fig.tight_layout()
fig.savefig(OUT / "joint_K100_grid.png", dpi=140)
print(f"\nCSV: {OUT / 'joint_K100_summary.csv'}")
print(f"Plot: {OUT / 'joint_K100_grid.png'}")
