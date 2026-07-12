"""Day 1 smoke test: Standard PINN on cell 25, K=100.

Goal: verify the training loop runs end-to-end, physics loss is meaningful,
and hold-out RMSE is comparable (or better) than pure PyBaMM's 0.29 pp.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, "/home/hj/Desktop/PINNs")

import numpy as np
import torch
import matplotlib.pyplot as plt

from Voltaris.sciml.data    import load_all, feature_normaliser
from Voltaris.sciml.physics import estimate_k_sei_from_window, physics_trajectory
from Voltaris.sciml.models  import build
from Voltaris.sciml.train   import TrainConfig, train_one_cell, predict_full_trajectory


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CELL_ID = 25
K = 100
OUT = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/sciml_day1")
OUT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

print(f"=== Day 1 smoke test — Standard PINN, cell {CELL_ID}, K={K} ===\n")
print(f"Device: {DEVICE}")

# Load ALL cells for normalisation stats; train on just cell 25
all_cells = load_all()
cell = next(c for c in all_cells if c.cell_id == CELL_ID)
mean, std = feature_normaliser(all_cells)
print(f"Feature norm mean={mean.tolist()} std={std.tolist()}")
print(f"Cell {cell.cell_id}: N={cell.n_total} cycles, SoH {cell.soh_init:.3f} -> {float(cell.soh_traj[-1]):.3f}")

# Fit k_SEI on training window (analytical physics prior)
k_sei = estimate_k_sei_from_window(cell, K)
print(f"k_SEI (fit on 0..{K}cy): {k_sei:.6f} per cycle")

# Build model
model = build("standard", n_features=len(cell.x_health), feat_mean=mean, feat_std=std)
print(f"Standard PINN params: {sum(p.numel() for p in model.parameters()):,}")

# Train
cfg = TrainConfig(K=K, epochs=1500, lr=1e-3, lam_phys=0.3, lam_mono=0.05,
                   n_norm_scale=float(cell.n_total), verbose_every=250)
print(f"\nTraining ({cfg.epochs} epochs)...")
result = train_one_cell(model, cell, cfg, DEVICE)
print(f"\nk_SEI={result['k_sei']:.6f}")
print(f"Loss history:")
for h in result["history"]:
    print(f"  epoch {h['epoch']:>4}: L_data={h['L_data']:.4e}  "
          f"L_phys={h['L_phys']:.4e}  L_mono={h['L_mono']:.4e}")

# Predict full trajectory
soh_pred = predict_full_trajectory(model, cell, cfg, DEVICE).numpy()
soh_meas = cell.soh_traj.numpy()
n = cell.n_traj.numpy()
first_cy = float(n[0])
k_end = first_cy + K

# RMSE (in pp SoH)
mask_train = n <= k_end
mask_test  = n > k_end
rmse_train_pp = float(np.sqrt(np.mean((soh_pred[mask_train] - soh_meas[mask_train])**2))) * 100
rmse_test_pp  = float(np.sqrt(np.mean((soh_pred[mask_test]  - soh_meas[mask_test])**2))) * 100

# Pure-PyBaMM analytical prediction (for comparison against 0.29 pp reference)
n_t = torch.tensor(n, dtype=torch.float32)
soh_phys = physics_trajectory(cell.soh_init, k_sei, n_t, first_cy).numpy()
rmse_phys_test = float(np.sqrt(np.mean((soh_phys[mask_test] - soh_meas[mask_test])**2))) * 100

print(f"\n=== Results ===")
print(f"  Standard PINN RMSE_train:  {rmse_train_pp:.3f} pp SoH")
print(f"  Standard PINN RMSE_test:   {rmse_test_pp:.3f} pp SoH")
print(f"  Pure-physics RMSE_test:    {rmse_phys_test:.3f} pp SoH  (reference)")
print(f"  Δend PINN:  {(soh_pred[-1] - soh_meas[-1])*100:+.3f} pp")
print(f"  Δend phys:  {(soh_phys[-1] - soh_meas[-1])*100:+.3f} pp")

# Plot
fig, ax = plt.subplots(figsize=(9, 5))
ax.axvspan(k_end, n[-1], color="tab:blue", alpha=0.08, label="Held-out")
ax.axvspan(first_cy, k_end, color="tab:orange", alpha=0.10, label=f"Training (K={K})")
ax.scatter(n, soh_meas*100, s=6, color="black", alpha=0.25, label="Measured")
ax.plot(n, soh_phys*100, color="tab:red", lw=2, ls="--",
        label=f"Physics only (RMSE_test={rmse_phys_test:.2f} pp)")
ax.plot(n, soh_pred*100, color="tab:green", lw=2,
        label=f"Standard PINN (RMSE_test={rmse_test_pp:.2f} pp)")
ax.axvline(k_end, color="dimgray", ls="--", lw=1)
ax.set_xlabel("Cycle number")
ax.set_ylabel("SoH [%]")
ax.set_title(f"Day 1 smoke test — cell {CELL_ID}, K={K}\n"
             f"Standard PINN vs pure-physics baseline")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / f"day1_cell{CELL_ID}_K{K}.png", dpi=140)
print(f"\nPlot saved: {OUT / f'day1_cell{CELL_ID}_K{K}.png'}")
