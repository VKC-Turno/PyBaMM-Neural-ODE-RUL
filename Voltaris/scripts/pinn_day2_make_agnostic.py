"""Make-agnostic test — Path B PINN on CALB, REPT, EVE independently.

Trains one joint PINN per manufacturer using Path B config. Reports
per-cell held-out RMSE and compares to pure-physics baseline for each
manufacturer.

Story goal: same architecture handles CALB (used cells, 15-25% fade),
REPT (near-fresh, 2-3% fade), EVE (near-fresh, 1-2% fade) — validates
that the PINN framework isn't calibrated to one dataset regime.

K is picked per-manufacturer to preserve held-out signal:
- CALB: K=50 (of 1000-1500 cy = 4% of trajectory used for training)
- REPT: K=50 (of 200 cy = 25% used)
- EVE:  K=50 (of 150 cy = 33% used)
"""
from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, "/home/hj/Desktop/PINNs")

import numpy as np
import pandas as pd
import torch

from Voltaris.sciml.data       import load_all as load_calb, feature_normaliser as norm_calb, CLEAN_IDS as CALB_IDS
from Voltaris.sciml.data_rept  import load_rept, normaliser_rept, REPT_IDS
from Voltaris.sciml.data_eve   import load_eve,  normaliser_eve,  EVE_IDS
from Voltaris.sciml.physics    import estimate_k_sei_from_window, physics_trajectory
from Voltaris.sciml.train_joint import (JointConfig, JointPINN, train_joint,
                                          predict_full_trajectory_joint)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 50
OUT = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/sciml_day2")
torch.manual_seed(42); np.random.seed(42)

# Path B config used consistently across manufacturers
def path_b_config(n_scale: float) -> JointConfig:
    return JointConfig(K=K, epochs=10000, lr=1e-3, lam_phys=2.0, lam_mono=0.05,
                        n_norm_scale=n_scale, n_col_per_cell=400,
                        p_init=0.5, verbose_every=99999)

def run_manufacturer(name: str, cells, mean, std):
    print(f"\n=== {name} — {len(cells)} cells, K={K} ===")
    mean_s = mean[:-1]; std_s = std[:-1]
    n_shared = len(cells[0].x_health) - 1
    torch.manual_seed(42); np.random.seed(42)

    model = JointPINN(n_cells=len(cells), n_shared_features=n_shared,
                        embed_dim=8, hidden=128, n_layers=5,
                        feat_mean=mean_s, feat_std=std_s, p_init=0.5)
    cfg = path_b_config(float(max(c.n_total for c in cells)))
    t0 = time.time()
    tr = train_joint(model, cells, cfg, DEVICE)
    print(f"  Trained {time.time()-t0:.1f}s")

    rows = []
    for i, cell in enumerate(cells):
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
        rows.append(dict(make=name, cell_id=cell.cell_id, K=K,
                         n_total=cell.n_total, n_test=int(mask_te.sum()),
                         rmse_pinn_pp=rmse_pinn, rmse_phys_pp=rmse_phys))
    return rows

all_rows = []
# CALB
calb = load_calb()
mean, std = norm_calb(calb)
cells = [c for c in calb if c.cell_id in CALB_IDS]
all_rows += run_manufacturer("CALB", cells, mean, std)

# REPT
rept = load_rept()
mean, std = normaliser_rept(rept)
all_rows += run_manufacturer("REPT", rept, mean, std)

# EVE
eve = load_eve()
mean, std = normaliser_eve(eve)
all_rows += run_manufacturer("EVE", eve, mean, std)

df = pd.DataFrame(all_rows)
df.to_csv(OUT / "make_agnostic_K50.csv", index=False)
print(f"\nCSV: {OUT / 'make_agnostic_K50.csv'}")

# Per-make summary
print("\n" + "="*80)
print(f"{'Make':>6}  {'cells':>5}  {'median PINN':>13}  {'median phys':>13}  "
      f"{'PINN <3pp':>10}  {'wins':>6}")
print("="*80)
for m in ["CALB", "REPT", "EVE"]:
    d = df[df.make == m]
    if len(d) == 0: continue
    print(f"{m:>6}  {len(d):>5}  "
          f"{d.rmse_pinn_pp.median():>10.3f} pp  "
          f"{d.rmse_phys_pp.median():>10.3f} pp  "
          f"{int((d.rmse_pinn_pp<3.0).sum()):>4}/{len(d)}      "
          f"{int((d.rmse_pinn_pp<d.rmse_phys_pp).sum()):>3}/{len(d)}")

print("\nPer-cell detail:")
for m in ["CALB", "REPT", "EVE"]:
    d = df[df.make == m]
    if len(d) == 0: continue
    print(f"\n  {m}:")
    for _, r in d.iterrows():
        p = "yes" if r.rmse_pinn_pp < 3.0 else "NO"
        w = "PINN" if r.rmse_pinn_pp < r.rmse_phys_pp else "phys"
        print(f"    cell {r.cell_id:>4}  N={r.n_total:>4}  "
              f"PINN={r.rmse_pinn_pp:>6.3f}  phys={r.rmse_phys_pp:>6.3f}  "
              f"[{w} wins, <3pp: {p}]")
