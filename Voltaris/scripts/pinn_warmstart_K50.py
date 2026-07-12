"""K=50 warm-started PINN vs baseline (reuses saved baseline checkpoint).

  Baseline: JointPINN (softplus(NN) only) — checkpoint from previous run at
            Voltaris/outputs/sciml_hybrid/baseline_K50.pt
  Warm:     JointPINN + train_joint_warmstart — same architecture, but NN is
            pre-trained on a linear-fade target derived from the training-window
            slope before the physics-loss loop kicks in.

Only trains the warm-started variant. Evaluates both against real cells on the
K=50 held-out window.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, "/home/hj/Desktop/PINNs")

import numpy as np
import pandas as pd
import torch

from Voltaris.sciml.data_combined         import load_combined, feature_normaliser
from Voltaris.sciml.physics               import estimate_k_sei_from_window, physics_trajectory
from Voltaris.sciml.train_joint           import (JointConfig, JointPINN,
                                                    predict_full_trajectory_joint)
from Voltaris.sciml.train_joint_warmstart import WarmStartConfig, train_joint_warmstart


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 50
OUT = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/sciml_hybrid")
BASELINE_CKPT = OUT / "baseline_K50.pt"


def _eval(model, cells, meta, cfg, tag):
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
            model=tag, make=meta[cell.cell_id]["make"],
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

# ── Load baseline from checkpoint ──
print("=== BASELINE: reusing saved checkpoint ===")
base = JointPINN(n_cells=len(cells), n_shared_features=n_shared,
                    embed_dim=8, hidden=128, n_layers=5,
                    feat_mean=mean_s, feat_std=std_s, p_init=0.5)
base.load_state_dict(torch.load(BASELINE_CKPT, map_location=DEVICE))
base.to(DEVICE).eval()
base_cfg = JointConfig(K=K, epochs=10000, lr=1e-3, lam_phys=2.0, lam_mono=0.05,
                        n_norm_scale=n_norm_scale, n_col_per_cell=400,
                        p_init=0.5, verbose_every=99999)
rows_base = _eval(base, cells, meta, base_cfg, "baseline")

# ── Warm-started ──
print("\n=== WARM-STARTED: pre-train NN on linear-fade target, then main loop ===")
torch.manual_seed(42); np.random.seed(42)
warm_cfg = WarmStartConfig(K=K, epochs=10000, lr=1e-3, lam_phys=2.0, lam_mono=0.05,
                            n_norm_scale=n_norm_scale, n_col_per_cell=400,
                            p_init=0.5, verbose_every=99999,
                            warmup_epochs=800, warmup_lr=3e-3,
                            warmup_n_col=200, warmup_domain_frac=1.0)
warm = JointPINN(n_cells=len(cells), n_shared_features=n_shared,
                    embed_dim=8, hidden=128, n_layers=5,
                    feat_mean=mean_s, feat_std=std_s, p_init=0.5)
t0 = time.time()
train_joint_warmstart(warm, cells, warm_cfg, DEVICE)
print(f"  warm trained in {time.time()-t0:.1f}s")
rows_warm = _eval(warm, cells, meta, base_cfg, "warm")

# ── Save ──
df = pd.concat([pd.DataFrame(rows_base), pd.DataFrame(rows_warm)], ignore_index=True)
df.to_csv(OUT / "warmstart_vs_baseline_K50.csv", index=False)
torch.save(warm.state_dict(), OUT / "warmstart_K50.pt")

# ── Report ──
print("\n" + "="*72)
print(f"{'Make':>6}  {'Baseline median':>18}  {'Warm median':>18}  {'Δ':>10}")
print("="*72)
for m in ["CALB", "REPT", "EVE"]:
    dB = df[(df.model == "baseline") & (df.make == m)]
    dW = df[(df.model == "warm")     & (df.make == m)]
    if len(dB) == 0: continue
    b, w = dB.rmse_pinn_pp.median(), dW.rmse_pinn_pp.median()
    print(f"{m:>6}  {b:>13.3f} pp     {w:>13.3f} pp   {b-w:>+7.3f} pp")

print()
for label, sub in [("baseline", df[df.model == "baseline"]),
                    ("warm",     df[df.model == "warm"])]:
    under3 = int((sub.rmse_pinn_pp < 3.0).sum())
    print(f"  {label:>10}: median={sub.rmse_pinn_pp.median():5.2f} pp   "
          f"under 3pp={under3}/{len(sub)}")

print("\nWrote:")
print(f"  {OUT/'warmstart_vs_baseline_K50.csv'}")
print(f"  {OUT/'warmstart_K50.pt'}")
