"""Day 1 cohort run: Standard PINN on all 7 clean CALB cells at K=100.

Trains one PINN per cell (same architecture, same hyperparameters),
compares held-out RMSE to the pure-physics baseline used in the
current PyBaMM Conf abstract.

Outputs:
- Voltaris/outputs/sciml_day1/K100_cohort_summary.csv
- Voltaris/outputs/sciml_day1/K100_grid.png  (7-cell trajectory overlays)
"""
from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, "/home/hj/Desktop/PINNs")

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from Voltaris.sciml.data    import load_all, feature_normaliser, CLEAN_IDS
from Voltaris.sciml.physics import estimate_k_sei_from_window, physics_trajectory
from Voltaris.sciml.models  import build
from Voltaris.sciml.train   import TrainConfig, train_one_cell, predict_full_trajectory


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 100
OUT = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/sciml_day1")
OUT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42); np.random.seed(42)

print(f"=== Day 1 cohort: Standard PINN on {len(CLEAN_IDS)} clean cells, K={K} ===")
print(f"Device: {DEVICE}\n")

# Load all cells for cohort-wide feature normalisation
all_cells = load_all()
mean, std = feature_normaliser(all_cells)

results = []
trajectories = {}
t_top = time.time()

for cid in CLEAN_IDS:
    cell = next(c for c in all_cells if c.cell_id == cid)
    n = cell.n_traj.numpy()
    s = cell.soh_traj.numpy()
    first_cy = float(n[0])
    k_end = first_cy + K
    mask_tr = n <= k_end
    mask_te = n >  k_end

    # ── Train Standard PINN ──
    torch.manual_seed(42 + cid)   # per-cell seed for reproducibility
    model = build("standard", n_features=len(cell.x_health),
                   feat_mean=mean, feat_std=std)
    cfg = TrainConfig(K=K, epochs=1500, lr=1e-3, lam_phys=1.0, lam_mono=0.05,
                       n_norm_scale=float(cell.n_total), n_collocation=200,
                       verbose_every=1000)
    t0 = time.time()
    train_result = train_one_cell(model, cell, cfg, DEVICE)
    k_sei = train_result["k_sei"]
    train_secs = time.time() - t0

    soh_pred = predict_full_trajectory(model, cell, cfg, DEVICE).numpy()

    # ── Pure-physics baseline (matching current abstract's method) ──
    n_t = torch.tensor(n, dtype=torch.float32)
    soh_phys = physics_trajectory(cell.soh_init, k_sei, n_t, first_cy).numpy()

    # ── RMSE ──
    rmse_pinn_train = float(np.sqrt(np.mean((soh_pred[mask_tr] - s[mask_tr])**2))) * 100
    rmse_pinn_test  = float(np.sqrt(np.mean((soh_pred[mask_te] - s[mask_te])**2))) * 100
    rmse_phys_test  = float(np.sqrt(np.mean((soh_phys[mask_te] - s[mask_te])**2))) * 100
    delta_end_pinn  = float((soh_pred[-1] - s[-1]) * 100)
    delta_end_phys  = float((soh_phys[-1] - s[-1]) * 100)

    results.append(dict(
        cell_id=cid, K_train_cy=K,
        n_train=int(mask_tr.sum()), n_test=int(mask_te.sum()),
        n_total=int(cell.n_total),
        k_sei=k_sei,
        rmse_pinn_train_pp=rmse_pinn_train,
        rmse_pinn_test_pp =rmse_pinn_test,
        rmse_phys_test_pp =rmse_phys_test,
        delta_end_pinn_pp =delta_end_pinn,
        delta_end_phys_pp =delta_end_phys,
        train_secs=train_secs,
    ))
    trajectories[cid] = (n, s, soh_pred, soh_phys, first_cy, k_end)

    winner = "PINN" if rmse_pinn_test < rmse_phys_test else "phys"
    print(f"  cell {cid:>2}  K={K}  ({train_secs:5.1f}s)  "
          f"PINN test={rmse_pinn_test:5.3f} pp  "
          f"phys test={rmse_phys_test:5.3f} pp  "
          f"[{winner} wins by {abs(rmse_pinn_test-rmse_phys_test):5.3f}]")

print(f"\nTotal wall-time: {time.time()-t_top:.1f}s")

df = pd.DataFrame(results)
df.to_csv(OUT / "K100_cohort_summary.csv", index=False)
print(f"\nSummary CSV: {OUT / 'K100_cohort_summary.csv'}")

# ── Grid plot: 7 cells × (measured, PINN, physics) ──
fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=False)
axs = axs.flatten()
for ax, (cid, (n, s, soh_pred, soh_phys, first_cy, k_end)) in zip(axs, trajectories.items()):
    r = df[df.cell_id == cid].iloc[0]
    ax.axvspan(k_end, n[-1], color="tab:blue", alpha=0.08)
    ax.axvspan(first_cy, k_end, color="tab:orange", alpha=0.10)
    ax.scatter(n, s*100, s=4, color="black", alpha=0.25, label="Measured")
    ax.plot(n, soh_phys*100, color="tab:red", lw=1.5, ls="--",
            label=f"Physics ({r['rmse_phys_test_pp']:.2f} pp)")
    ax.plot(n, soh_pred*100, color="tab:green", lw=1.8,
            label=f"PINN ({r['rmse_pinn_test_pp']:.2f} pp)")
    ax.axvline(k_end, color="dimgray", ls="--", lw=0.8)
    ax.set_title(f"cell {cid}", fontsize=11)
    ax.set_xlabel("Cycle"); ax.set_ylabel("SoH [%]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
# hide unused axes
for ax in axs[len(trajectories):]:
    ax.set_visible(False)
fig.suptitle(f"Day 1 — Standard PINN vs pure-physics, K={K}", fontsize=13, y=1.005)
fig.tight_layout()
fig.savefig(OUT / "K100_grid.png", dpi=140)
print(f"Grid plot: {OUT / 'K100_grid.png'}")

# ── Headline summary ──
print("\n=== Headline ===")
print(f"Median PINN RMSE_test: {df.rmse_pinn_test_pp.median():.3f} pp")
print(f"Median phys RMSE_test: {df.rmse_phys_test_pp.median():.3f} pp")
print(f"PINN cells under 3 pp: {int((df.rmse_pinn_test_pp < 3.0).sum())}/{len(df)}")
print(f"phys cells under 3 pp: {int((df.rmse_phys_test_pp < 3.0).sum())}/{len(df)}")
print(f"PINN wins per cell:    {int((df.rmse_pinn_test_pp < df.rmse_phys_test_pp).sum())}/{len(df)}")
