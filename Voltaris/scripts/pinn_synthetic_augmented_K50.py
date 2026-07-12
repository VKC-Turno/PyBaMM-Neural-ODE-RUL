"""PINN training on combined real + PyBaMM-synthetic cells at K=50.

Compares two training regimes:
  A) Real cells only (7 CALB + 9 REPT + 4 EVE = 20 cells)  — baseline
  B) Real + synthetic pool (20 real + 24-100+ synthetic)   — augmented

Evaluates held-out RMSE on the 20 real cells (not synthetic — those are
training aid, not the ground truth we care about).
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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 50
OUT = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/sciml_synth")
OUT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42); np.random.seed(42)


def path_b_config(n_scale: float) -> JointConfig:
    return JointConfig(K=K, epochs=10000, lr=1e-3, lam_phys=2.0, lam_mono=0.05,
                        n_norm_scale=n_scale, n_col_per_cell=400,
                        p_init=0.5, verbose_every=99999)


def run_regime(name: str, cells, meta, out_csv: str) -> pd.DataFrame:
    """Train PINN on `cells`, evaluate on real cells only (using `meta`)."""
    print(f"\n=== {name} — {len(cells)} cells ===")
    real_cells = [(i, c) for i, c in enumerate(cells)
                    if not meta[c.cell_id]["is_synthetic"]]
    synth_cells = [c for c in cells if meta[c.cell_id]["is_synthetic"]]
    print(f"  Real: {len(real_cells)}  Synthetic: {len(synth_cells)}")

    mean, std = feature_normaliser(cells)
    mean_s = mean[:-1]; std_s = std[:-1]
    n_shared = len(cells[0].x_health) - 1

    torch.manual_seed(42); np.random.seed(42)
    model = JointPINN(n_cells=len(cells), n_shared_features=n_shared,
                        embed_dim=8, hidden=128, n_layers=5,
                        feat_mean=mean_s, feat_std=std_s, p_init=0.5)
    n_scale = float(max(c.n_total for c in cells))
    cfg = path_b_config(n_scale)

    t0 = time.time()
    train_joint(model, cells, cfg, DEVICE)
    print(f"  Trained {time.time()-t0:.1f}s")

    # Evaluate on REAL cells only
    rows = []
    for i, cell in real_cells:
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
            regime=name,
            make=meta[cell.cell_id]["make"],
            cell_id=cell.cell_id,
            n_total=cell.n_total, n_test=int(mask_te.sum()),
            rmse_pinn_pp=rmse_pinn,
            rmse_phys_pp=rmse_phys,
        ))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / out_csv, index=False)

    # Print summary
    for m in df.make.unique():
        d = df[df.make == m]
        print(f"  {m:>8}: PINN median={d.rmse_pinn_pp.median():5.2f}  "
              f"under 3pp={int((d.rmse_pinn_pp<3.0).sum())}/{len(d)}")
    print(f"  ALL   : PINN median={df.rmse_pinn_pp.median():5.2f}  "
          f"under 3pp={int((df.rmse_pinn_pp<3.0).sum())}/{len(df)}")
    return df


# Regime A: real cells only
cells_A, meta_A = load_combined(include_synth=False)
df_A = run_regime("A_real_only", cells_A, meta_A, "regime_A_real_only.csv")

# Regime B: real + synthetic
cells_B, meta_B = load_combined(include_synth=True)
if len(cells_B) > len(cells_A):
    df_B = run_regime("B_augmented", cells_B, meta_B, "regime_B_augmented.csv")
    # Combined comparison
    df_A["regime"] = "A_real_only"
    df_B["regime"] = "B_augmented"
    combined = pd.concat([df_A, df_B], ignore_index=True)
    combined.to_csv(OUT / "regimes_comparison.csv", index=False)

    print("\n" + "="*70)
    print("HEAD-TO-HEAD COMPARISON")
    print("="*70)
    for m in ["CALB", "REPT", "EVE"]:
        dA = df_A[df_A.make == m]
        dB = df_B[df_B.make == m]
        if len(dA) == 0 or len(dB) == 0: continue
        print(f"  {m}:")
        print(f"    A (real only):   median={dA.rmse_pinn_pp.median():5.2f} pp"
              f"   under 3pp={int((dA.rmse_pinn_pp<3.0).sum())}/{len(dA)}")
        print(f"    B (real+synth):  median={dB.rmse_pinn_pp.median():5.2f} pp"
              f"   under 3pp={int((dB.rmse_pinn_pp<3.0).sum())}/{len(dB)}")
        delta = dA.rmse_pinn_pp.median() - dB.rmse_pinn_pp.median()
        print(f"    Δ (A - B):       {delta:+5.2f} pp median  "
              f"({'B wins' if delta > 0 else 'A wins'})")
else:
    print("\nNo synthetic cells loaded — run synthesise.py first")
