"""Full 3-regime comparison at K=50.

  A: real only (20 cells)
  B: real + synth full pool, EQUAL weight (baseline synthetic augmentation)
  C: real + synth full pool, WEIGHTED — synth cells get 0.3× data-loss weight
     (uses physics constraint diversity from synth without biasing real
      predictions)

Evaluates held-out RMSE on real cells only.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, "/home/hj/Desktop/PINNs")

import numpy as np
import pandas as pd
import torch

from Voltaris.sciml.data_combined import load_combined, feature_normaliser
from Voltaris.sciml.physics       import estimate_k_sei_from_window, physics_trajectory
from Voltaris.sciml.train_joint    import (JointConfig, JointPINN, train_joint,
                                             predict_full_trajectory_joint)
from Voltaris.sciml.train_joint_weighted import WeightedConfig, train_joint_weighted


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 50
OUT = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/sciml_synth")
OUT.mkdir(parents=True, exist_ok=True)

SYNTH_FULL_PQ = "/home/hj/Desktop/PINNs/Voltaris/outputs/synthetic/synthetic_full.parquet"


def cfg_path_b(n_scale, K=K):
    return JointConfig(K=K, epochs=10000, lr=1e-3, lam_phys=2.0, lam_mono=0.05,
                        n_norm_scale=n_scale, n_col_per_cell=400,
                        p_init=0.5, verbose_every=99999)


def wcfg_path_b(n_scale, synth_weight=0.3, K=K):
    return WeightedConfig(K=K, epochs=10000, lr=1e-3, lam_phys=2.0, lam_mono=0.05,
                            n_norm_scale=n_scale, n_col_per_cell=400,
                            p_init=0.5, synth_weight=synth_weight,
                            verbose_every=99999)


def _evaluate_real(model, cells, meta, cfg):
    """Return per-real-cell RMSE rows."""
    rows = []
    for i, cell in enumerate(cells):
        if meta[cell.cell_id]["is_synthetic"]:
            continue
        n = cell.n_traj.numpy(); s = cell.soh_traj.numpy()
        first_cy = float(n[0]); k_end = first_cy + K
        mask_te = n > k_end
        if mask_te.sum() < 3:
            continue
        soh_pred = predict_full_trajectory_joint(model, cell, i, cfg, DEVICE).numpy()
        k_L0 = estimate_k_sei_from_window(cell, K)
        n_t = torch.tensor(n, dtype=torch.float32)
        soh_phys = physics_trajectory(cell.soh_init, k_L0, n_t, first_cy).numpy()
        rmse_pinn = float(np.sqrt(np.mean((soh_pred[mask_te] - s[mask_te])**2))) * 100
        rmse_phys = float(np.sqrt(np.mean((soh_phys[mask_te] - s[mask_te])**2))) * 100
        rows.append(dict(
            make=meta[cell.cell_id]["make"],
            cell_id=cell.cell_id,
            n_total=cell.n_total, n_test=int(mask_te.sum()),
            rmse_pinn_pp=rmse_pinn,
            rmse_phys_pp=rmse_phys,
        ))
    return rows


def run_regime_joint(name, cells, meta):
    torch.manual_seed(42); np.random.seed(42)
    mean, std = feature_normaliser(cells)
    mean_s = mean[:-1]; std_s = std[:-1]
    n_shared = len(cells[0].x_health) - 1
    model = JointPINN(n_cells=len(cells), n_shared_features=n_shared,
                        embed_dim=8, hidden=128, n_layers=5,
                        feat_mean=mean_s, feat_std=std_s, p_init=0.5)
    cfg = cfg_path_b(float(max(c.n_total for c in cells)))
    t0 = time.time()
    train_joint(model, cells, cfg, DEVICE)
    print(f"  {name}: trained in {time.time()-t0:.1f}s")
    return _evaluate_real(model, cells, meta, cfg)


def run_regime_weighted(name, cells, meta, synth_weight=0.3):
    torch.manual_seed(42); np.random.seed(42)
    mean, std = feature_normaliser(cells)
    mean_s = mean[:-1]; std_s = std[:-1]
    n_shared = len(cells[0].x_health) - 1
    model = JointPINN(n_cells=len(cells), n_shared_features=n_shared,
                        embed_dim=8, hidden=128, n_layers=5,
                        feat_mean=mean_s, feat_std=std_s, p_init=0.5)
    cfg = wcfg_path_b(float(max(c.n_total for c in cells)), synth_weight=synth_weight)
    weights = [1.0 if not meta[c.cell_id]["is_synthetic"] else synth_weight
                for c in cells]
    t0 = time.time()
    train_joint_weighted(model, cells, weights, cfg, DEVICE)
    print(f"  {name}: trained in {time.time()-t0:.1f}s "
          f"(n_real={weights.count(1.0)}, n_synth={weights.count(synth_weight)})")
    # Use predict_full_trajectory_joint (same interface)
    return _evaluate_real(model, cells, meta, cfg)


# ── Load pools ──
cells_A, meta_A = load_combined(include_synth=False)
cells_B, meta_B = load_combined(include_synth=True, synth_parquet=SYNTH_FULL_PQ)
n_synth = sum(1 for c in cells_B if meta_B[c.cell_id]["is_synthetic"])
print(f"Pool A (real): {len(cells_A)} cells")
print(f"Pool B/C (real + synth): {len(cells_B)} cells ({n_synth} synthetic)\n")

# ── Regime A ──
print("=== Regime A (real only) ===")
rows_A = run_regime_joint("A", cells_A, meta_A)

# ── Regime B (equal-weight synthetic augmentation) ──
print("\n=== Regime B (real + synth, equal weight) ===")
rows_B = run_regime_joint("B", cells_B, meta_B)

# ── Regime C (synth downweighted 0.3) ──
print("\n=== Regime C (real + synth, 0.3× weight) ===")
rows_C = run_regime_weighted("C", cells_B, meta_B, synth_weight=0.3)

# ── Combine ──
dfs = []
for label, rows in [("A_real_only", rows_A), ("B_equal", rows_B), ("C_weighted", rows_C)]:
    df = pd.DataFrame(rows)
    df["regime"] = label
    dfs.append(df)
df_all = pd.concat(dfs, ignore_index=True)
df_all.to_csv(OUT / "three_regimes.csv", index=False)

# ── Head-to-head ──
print("\n" + "="*72)
print(f"{'Make':>6}  {'A median':>10}  {'B median':>10}  {'C median':>10}  {'Winner':>15}")
print("="*72)
for m in ["CALB", "REPT", "EVE"]:
    dA = df_all[(df_all.regime == "A_real_only") & (df_all.make == m)]
    dB = df_all[(df_all.regime == "B_equal") & (df_all.make == m)]
    dC = df_all[(df_all.regime == "C_weighted") & (df_all.make == m)]
    if len(dA) == 0: continue
    a, b, c = dA.rmse_pinn_pp.median(), dB.rmse_pinn_pp.median(), dC.rmse_pinn_pp.median()
    winner = min([("A", a), ("B", b), ("C", c)], key=lambda x: x[1])
    print(f"{m:>6}  {a:>7.3f} pp  {b:>7.3f} pp  {c:>7.3f} pp  {winner[0]+' wins':>15}")

# Overall
dA_all = df_all[df_all.regime == "A_real_only"]
dB_all = df_all[df_all.regime == "B_equal"]
dC_all = df_all[df_all.regime == "C_weighted"]
print()
for label, d in [("A", dA_all), ("B", dB_all), ("C", dC_all)]:
    print(f"  {label}: median={d.rmse_pinn_pp.median():5.2f} pp   under 3pp={int((d.rmse_pinn_pp<3.0).sum())}/{len(d)}")
