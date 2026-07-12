"""K=50 hybrid-vs-baseline comparison on the 20-cell real cohort.

  Baseline: JointPINN (softplus(NN) decrement only) — Path B / Regime A
            from three_regimes.csv (CALB median 2.87 pp).
  Hybrid:   JointPINN_Hybrid (softplus(log_a)*n_norm + softplus(NN))
            with per-cell log_a warm-started from training-window linear slope.

Evaluates held-out RMSE on real cells only (CALB + REPT + EVE).
"""
from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, "/home/hj/Desktop/PINNs")

import numpy as np
import pandas as pd
import torch

from Voltaris.sciml.data_combined      import load_combined, feature_normaliser
from Voltaris.sciml.physics            import estimate_k_sei_from_window, physics_trajectory
from Voltaris.sciml.train_joint        import (JointConfig, JointPINN, train_joint,
                                                 predict_full_trajectory_joint)
from Voltaris.sciml.train_joint_hybrid import (HybridConfig, JointPINN_Hybrid,
                                                 train_hybrid, predict_hybrid)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 50
OUT = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/sciml_hybrid")
OUT.mkdir(parents=True, exist_ok=True)


def _eval_baseline(model, cells, meta, cfg):
    rows = []
    for i, cell in enumerate(cells):
        if meta[cell.cell_id]["is_synthetic"]:
            continue
        n = cell.n_traj.numpy(); s = cell.soh_traj.numpy()
        first_cy = float(n[0]); k_end = first_cy + K
        mask_te = n > k_end
        if mask_te.sum() < 3: continue
        soh_pred = predict_full_trajectory_joint(model, cell, i, cfg, DEVICE).numpy()
        k_L0 = estimate_k_sei_from_window(cell, K)
        n_t = torch.tensor(n, dtype=torch.float32)
        soh_phys = physics_trajectory(cell.soh_init, k_L0, n_t, first_cy).numpy()
        rmse_pinn = float(np.sqrt(np.mean((soh_pred[mask_te]-s[mask_te])**2))) * 100
        rmse_phys = float(np.sqrt(np.mean((soh_phys[mask_te]-s[mask_te])**2))) * 100
        rows.append(dict(
            model="baseline", make=meta[cell.cell_id]["make"],
            cell_id=cell.cell_id, n_total=cell.n_total,
            n_test=int(mask_te.sum()),
            rmse_pinn_pp=rmse_pinn, rmse_phys_pp=rmse_phys))
    return rows


def _eval_hybrid(model, cells, meta, cfg):
    rows = []
    for i, cell in enumerate(cells):
        if meta[cell.cell_id]["is_synthetic"]:
            continue
        n = cell.n_traj.numpy(); s = cell.soh_traj.numpy()
        first_cy = float(n[0]); k_end = first_cy + K
        mask_te = n > k_end
        if mask_te.sum() < 3: continue
        soh_pred = predict_hybrid(model, cell, i, cfg, DEVICE).numpy()
        k_L0 = estimate_k_sei_from_window(cell, K)
        n_t = torch.tensor(n, dtype=torch.float32)
        soh_phys = physics_trajectory(cell.soh_init, k_L0, n_t, first_cy).numpy()
        rmse_pinn = float(np.sqrt(np.mean((soh_pred[mask_te]-s[mask_te])**2))) * 100
        rmse_phys = float(np.sqrt(np.mean((soh_phys[mask_te]-s[mask_te])**2))) * 100
        rows.append(dict(
            model="hybrid", make=meta[cell.cell_id]["make"],
            cell_id=cell.cell_id, n_total=cell.n_total,
            n_test=int(mask_te.sum()),
            rmse_pinn_pp=rmse_pinn, rmse_phys_pp=rmse_phys))
    return rows


# ── Load real-only cohort ──
cells, meta = load_combined(include_synth=False)
print(f"Pool: {len(cells)} real cells "
      f"(CALB={sum(1 for c in cells if meta[c.cell_id]['make']=='CALB')} "
      f"REPT={sum(1 for c in cells if meta[c.cell_id]['make']=='REPT')} "
      f"EVE={sum(1 for c in cells if meta[c.cell_id]['make']=='EVE')})\n")

mean, std   = feature_normaliser(cells)
mean_s      = mean[:-1]; std_s = std[:-1]
n_shared    = len(cells[0].x_health) - 1
n_norm_scale = float(max(c.n_total for c in cells))

# ── Baseline: reproduce Regime A ──
print("=== BASELINE: JointPINN (softplus(NN) only) ===")
torch.manual_seed(42); np.random.seed(42)
base_cfg = JointConfig(K=K, epochs=10000, lr=1e-3, lam_phys=2.0, lam_mono=0.05,
                        n_norm_scale=n_norm_scale, n_col_per_cell=400,
                        p_init=0.5, verbose_every=99999)
base = JointPINN(n_cells=len(cells), n_shared_features=n_shared,
                    embed_dim=8, hidden=128, n_layers=5,
                    feat_mean=mean_s, feat_std=std_s, p_init=0.5)
t0 = time.time()
train_joint(base, cells, base_cfg, DEVICE)
print(f"  baseline trained in {time.time()-t0:.1f}s")
rows_base = _eval_baseline(base, cells, meta, base_cfg)

# ── Hybrid ──
print("\n=== HYBRID: softplus(log_a)*n_norm + softplus(NN) ===")
torch.manual_seed(42); np.random.seed(42)
hyb_cfg = HybridConfig(K=K, epochs=10000, lr=1e-3, lam_phys=2.0, lam_mono=0.05,
                        n_norm_scale=n_norm_scale, n_col_per_cell=400,
                        p_init=0.5, verbose_every=99999)
hyb = JointPINN_Hybrid(n_cells=len(cells), n_shared_features=n_shared,
                        embed_dim=8, hidden=128, n_layers=5,
                        feat_mean=mean_s, feat_std=std_s, p_init=0.5)
t0 = time.time()
train_hybrid(hyb, cells, hyb_cfg, DEVICE)
print(f"  hybrid trained in {time.time()-t0:.1f}s")
rows_hyb = _eval_hybrid(hyb, cells, meta, hyb_cfg)

# ── Combine + save ──
df = pd.concat([pd.DataFrame(rows_base), pd.DataFrame(rows_hyb)], ignore_index=True)
df.to_csv(OUT / "hybrid_vs_baseline_K50.csv", index=False)

# Save both model checkpoints for downstream figure regeneration
torch.save(base.state_dict(), OUT / "baseline_K50.pt")
torch.save(hyb.state_dict(),  OUT / "hybrid_K50.pt")

# ── Report ──
print("\n" + "="*72)
print(f"{'Make':>6}  {'Baseline median':>18}  {'Hybrid median':>18}  {'Δ':>10}")
print("="*72)
for m in ["CALB", "REPT", "EVE"]:
    dB = df[(df.model == "baseline") & (df.make == m)]
    dH = df[(df.model == "hybrid")   & (df.make == m)]
    if len(dB) == 0: continue
    b, h = dB.rmse_pinn_pp.median(), dH.rmse_pinn_pp.median()
    print(f"{m:>6}  {b:>13.3f} pp     {h:>13.3f} pp   {b-h:>+7.3f} pp")

print()
for label, sub in [("baseline", df[df.model == "baseline"]),
                    ("hybrid",   df[df.model == "hybrid"])]:
    under3 = int((sub.rmse_pinn_pp < 3.0).sum())
    print(f"  {label:>8}: median={sub.rmse_pinn_pp.median():5.2f} pp   "
          f"under 3pp={under3}/{len(sub)}")

print("\nWrote:")
print(f"  {OUT/'hybrid_vs_baseline_K50.csv'}")
print(f"  {OUT/'baseline_K50.pt'}")
print(f"  {OUT/'hybrid_K50.pt'}")
