"""Day 2 — Joint PINN on FULL 9-cell cohort (7 clean + 2 batch-artefact).

The current abstract EXCLUDES cells 24 and 30 because their canonical
cycle numbering has batch-transition discontinuities that pure physics
cannot handle. This experiment asks: can the joint PINN, with its per-cell
learned embedding, recover them?

If YES → we have a substantial story: PINN takes 5/7 physics coverage
to 8/9 (or better) at K=100, without needing a shape filter to drop cells.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, "/home/hj/Desktop/PINNs")

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from Voltaris.sciml.data       import load_all, feature_normaliser, ALL_IDS
from Voltaris.sciml.physics    import estimate_k_sei_from_window, physics_trajectory
from Voltaris.sciml.train_joint import (JointConfig, JointPINN, train_joint,
                                          predict_full_trajectory_joint)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 100
OUT = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/sciml_day2")
OUT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42); np.random.seed(42)

print(f"=== Day 2 — Joint PINN on ALL 9 CALB cells (K={K}) ===")
print(f"    Includes cells 24, 30 (batch-transition artefacts)")
print(f"Device: {DEVICE}\n")

all_cells = load_all()
mean, std = feature_normaliser(all_cells)
mean_shared = mean[:-1]; std_shared = std[:-1]

n_shared = len(all_cells[0].x_health) - 1
model = JointPINN(n_cells=len(all_cells), n_shared_features=n_shared,
                    embed_dim=8, hidden=64, n_layers=4,
                    feat_mean=mean_shared, feat_std=std_shared,
                    p_init=0.5)
print(f"Joint PINN params: {sum(p.numel() for p in model.parameters()):,}")
print(f"  cells: {[c.cell_id for c in all_cells]}")
print(f"  labels: {['clean' if c.is_clean else 'DIRTY' for c in all_cells]}\n")

cfg = JointConfig(K=K, epochs=5000, lr=1e-3, lam_phys=1.0, lam_mono=0.05,
                    n_norm_scale=float(max(c.n_total for c in all_cells)),
                    n_col_per_cell=200, p_init=0.5, verbose_every=1000)

t0 = time.time()
train_result = train_joint(model, all_cells, cfg, DEVICE)
train_secs = time.time() - t0

print(f"\nTraining took {train_secs:.1f}s\n")
print(f"Learned per-cell physics parameters:")
for i, c in enumerate(all_cells):
    tag = "clean" if c.is_clean else "DIRTY"
    print(f"  cell {c.cell_id:>2} [{tag}]:  k_SEI = {train_result['k_sei_final'][i]:.4e}   "
          f"p = {train_result['p_final'][i]:.3f}")

# Evaluate per-cell
results = []; trajectories = {}
for i, cell in enumerate(all_cells):
    n = cell.n_traj.numpy(); s = cell.soh_traj.numpy()
    first_cy = float(n[0])
    k_end = first_cy + K
    mask_tr = n <= k_end
    mask_te = n >  k_end

    soh_pred = predict_full_trajectory_joint(model, cell, i, cfg, DEVICE).numpy()

    k_L0 = estimate_k_sei_from_window(cell, K)
    n_t  = torch.tensor(n, dtype=torch.float32)
    soh_phys = physics_trajectory(cell.soh_init, k_L0, n_t, first_cy).numpy()

    rmse_pinn_te = float(np.sqrt(np.mean((soh_pred[mask_te] - s[mask_te])**2))) * 100
    rmse_phys_te = float(np.sqrt(np.mean((soh_phys[mask_te] - s[mask_te])**2))) * 100
    results.append(dict(
        cell_id=cell.cell_id, is_clean=cell.is_clean, K_train_cy=K,
        n_total=cell.n_total, n_test=int(mask_te.sum()),
        rmse_pinn_test_pp=rmse_pinn_te,
        rmse_phys_test_pp=rmse_phys_te,
        k_sei_learned=train_result['k_sei_final'][i],
        p_learned=train_result['p_final'][i],
    ))
    trajectories[cell.cell_id] = (n, s, soh_pred, soh_phys, first_cy, k_end,
                                    cell.is_clean)

df = pd.DataFrame(results)
df.to_csv(OUT / "joint_9cell_K100_summary.csv", index=False)

print(f"\n{'='*80}")
print(f"{'cell':>5} {'tag':>7}  {'PINN test':>10}  {'phys test':>10}  {'PINN <3?':>9}  {'phys <3?':>9}")
print(f"{'='*80}")
for r in results:
    tag = "clean" if r['is_clean'] else "DIRTY"
    p_pass = "yes" if r['rmse_pinn_test_pp'] < 3.0 else "NO"
    ph_pass = "yes" if r['rmse_phys_test_pp'] < 3.0 else "NO"
    print(f"{r['cell_id']:>5} {tag:>7}  {r['rmse_pinn_test_pp']:>7.3f} pp  "
          f"{r['rmse_phys_test_pp']:>7.3f} pp  {p_pass:>9}  {ph_pass:>9}")

print(f"\n{'='*80}")
print(f"Clean cells (n=7): PINN under 3pp = {int(((df.rmse_pinn_test_pp<3.0)&df.is_clean).sum())}/7   "
      f"phys under 3pp = {int(((df.rmse_phys_test_pp<3.0)&df.is_clean).sum())}/7")
print(f"DIRTY cells (n=2): PINN under 3pp = {int(((df.rmse_pinn_test_pp<3.0)&~df.is_clean).sum())}/2   "
      f"phys under 3pp = {int(((df.rmse_phys_test_pp<3.0)&~df.is_clean).sum())}/2")
print(f"ALL cells (n=9):   PINN under 3pp = {int((df.rmse_pinn_test_pp<3.0).sum())}/9   "
      f"phys under 3pp = {int((df.rmse_phys_test_pp<3.0).sum())}/9")
print(f"Median PINN test: {df.rmse_pinn_test_pp.median():.3f} pp")
print(f"Median phys test: {df.rmse_phys_test_pp.median():.3f} pp")

# ── Grid plot (3x3) ──
fig, axs = plt.subplots(3, 3, figsize=(16, 11))
axs = axs.flatten()
for ax, (cid, (n, s, soh_pred, soh_phys, first_cy, k_end, is_clean)) in \
        zip(axs, trajectories.items()):
    r = df[df.cell_id == cid].iloc[0]
    ax.axvspan(k_end, n[-1], color="tab:blue", alpha=0.06)
    ax.axvspan(first_cy, k_end, color="tab:orange", alpha=0.08)
    ax.scatter(n, s*100, s=3, color="black", alpha=0.25, label="Measured")
    ax.plot(n, soh_phys*100, color="tab:red", lw=1.4, ls="--",
            label=f"phys ({r['rmse_phys_test_pp']:.2f} pp)")
    ax.plot(n, soh_pred*100, color="tab:green", lw=1.8,
            label=f"PINN ({r['rmse_pinn_test_pp']:.2f} pp)")
    ax.axvline(k_end, color="dimgray", ls="--", lw=0.7)
    tag = "" if is_clean else " [DIRTY]"
    ax.set_title(f"cell {cid}{tag}", fontsize=11)
    ax.set_xlabel("Cycle"); ax.set_ylabel("SoH [%]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
fig.suptitle(f"Day 2 — 9-cell joint PINN vs pure-physics, K={K}",
             fontsize=13, y=1.005)
fig.tight_layout()
fig.savefig(OUT / "joint_9cell_K100_grid.png", dpi=140)
print(f"\nCSV : {OUT / 'joint_9cell_K100_summary.csv'}")
print(f"Plot: {OUT / 'joint_9cell_K100_grid.png'}")
