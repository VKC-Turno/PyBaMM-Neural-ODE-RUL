"""Day 2 diagnostic — does upgrading the physics prior from L0 (linear)
to L1 (SoH-dependent) to L2 (SEI+delayed LAM) actually help?

We're asking a targeted question: before running any PINN, does the
right-shape ODE alone extrapolate cells 6 and 7 correctly? If L2 fits
the training window well AND extrapolates below 3 pp held-out RMSE for
cells 6 and 7, we've already won — the PINN just needs to match L2.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, "/home/hj/Desktop/PINNs")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Voltaris.sciml.data    import load_all, CLEAN_IDS
from Voltaris.sciml.physics import (
    estimate_k_sei_from_window, physics_trajectory,
    fit_L1, physics_trajectory_L1,
    fit_L2, physics_trajectory_L2,
)

K = 100
OUT = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/sciml_day2")
OUT.mkdir(parents=True, exist_ok=True)

print(f"=== Day 2 ODE diagnostic — K={K}, 7 clean cells ===\n")
print("For each cell we fit three physics forms on the training window")
print("then evaluate held-out RMSE across the remaining cycles:\n")
print("  L0:  dSoH/dn = -k_SEI                            (constant, Day 1 baseline)")
print("  L1:  dSoH/dn = -k_SEI * SoH^p                    (SoH-dependent rate)")
print("  L2:  dSoH/dn = -k_SEI * SoH^p                    ")
print("               - k_LAM * exp((n-n_c)/tau) * [n>n_c] (SEI + delayed LAM)\n")

cells = load_all()
results = []
trajectories = {}
t_top = time.time()

for cid in CLEAN_IDS:
    cell = next(c for c in cells if c.cell_id == cid)
    n = cell.n_traj.numpy(); s = cell.soh_traj.numpy()
    first_cy = float(n[0])
    k_end = first_cy + K
    mask_te = n > k_end

    # L0 linear
    t0 = time.time()
    k_SEI_L0 = estimate_k_sei_from_window(cell, K)
    soh_L0 = (cell.soh_init - k_SEI_L0 * (n - first_cy))
    t_L0 = time.time() - t0

    # L1 SoH-dependent
    t0 = time.time()
    p1 = fit_L1(cell, K)
    soh_L1 = physics_trajectory_L1(p1["soh_0"], p1["k_SEI"], p1["p"], n, first_cy)
    t_L1 = time.time() - t0

    # L2 SEI + LAM
    t0 = time.time()
    p2 = fit_L2(cell, K)
    soh_L2 = physics_trajectory_L2(p2["soh_0"], p2, n, first_cy)
    t_L2 = time.time() - t0

    # Held-out RMSE in pp SoH
    rmse_L0 = float(np.sqrt(np.mean((soh_L0[mask_te] - s[mask_te])**2))) * 100
    rmse_L1 = float(np.sqrt(np.mean((soh_L1[mask_te] - s[mask_te])**2))) * 100
    rmse_L2 = float(np.sqrt(np.mean((soh_L2[mask_te] - s[mask_te])**2))) * 100

    results.append(dict(
        cell_id=cid,
        rmse_L0_pp=rmse_L0, rmse_L1_pp=rmse_L1, rmse_L2_pp=rmse_L2,
        k_SEI_L0=k_SEI_L0,
        k_SEI_L1=p1["k_SEI"], p_L1=p1["p"],
        k_SEI_L2=p2["k_SEI"], p_L2=p2["p"],
        k_LAM_L2=p2["k_LAM"], n_c_L2=p2["n_c"], tau_L2=p2["tau"],
        fit_secs_L0=t_L0, fit_secs_L1=t_L1, fit_secs_L2=t_L2,
    ))
    trajectories[cid] = (n, s, soh_L0, soh_L1, soh_L2, first_cy, k_end)

    winner = min([("L0", rmse_L0), ("L1", rmse_L1), ("L2", rmse_L2)],
                  key=lambda x: x[1])
    print(f"  cell {cid:>2}  L0={rmse_L0:6.3f}  L1={rmse_L1:6.3f}  L2={rmse_L2:6.3f} pp"
          f"   winner={winner[0]}  (n_c={p2['n_c']:.0f}, tau={p2['tau']:.0f})")

df = pd.DataFrame(results)
df.to_csv(OUT / "ode_levels_K100.csv", index=False)
print(f"\nCSV: {OUT / 'ode_levels_K100.csv'}")

# ── Summary ──
print(f"\n{'='*60}")
print("Summary — held-out RMSE per ODE level")
print(f"{'='*60}")
for level, col in [("L0 linear", "rmse_L0_pp"),
                    ("L1 SoH^p",   "rmse_L1_pp"),
                    ("L2 SEI+LAM", "rmse_L2_pp")]:
    print(f"  {level:>12}  median={df[col].median():5.3f}  "
          f"max={df[col].max():5.3f}  "
          f"cells<3pp={int((df[col]<3.0).sum())}/{len(df)}")

# ── Grid plot ──
fig, axs = plt.subplots(3, 3, figsize=(16, 11))
axs = axs.flatten()
for ax, (cid, (n, s, soh_L0, soh_L1, soh_L2, first_cy, k_end)) in zip(axs, trajectories.items()):
    r = df[df.cell_id == cid].iloc[0]
    ax.axvspan(k_end, n[-1], color="tab:blue", alpha=0.06)
    ax.axvspan(first_cy, k_end, color="tab:orange", alpha=0.08)
    ax.scatter(n, s*100, s=3, color="black", alpha=0.25, label="Measured")
    ax.plot(n, soh_L0*100, color="tab:gray",   lw=1.3, ls=":",
             label=f"L0 linear ({r['rmse_L0_pp']:.2f} pp)")
    ax.plot(n, soh_L1*100, color="tab:orange", lw=1.5, ls="--",
             label=f"L1 SoH^p ({r['rmse_L1_pp']:.2f} pp)")
    ax.plot(n, soh_L2*100, color="tab:green",  lw=1.8,
             label=f"L2 SEI+LAM ({r['rmse_L2_pp']:.2f} pp)")
    ax.axvline(k_end, color="dimgray", ls="--", lw=0.7)
    ax.set_title(f"cell {cid}", fontsize=11)
    ax.set_xlabel("Cycle"); ax.set_ylabel("SoH [%]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
for ax in axs[len(trajectories):]:
    ax.set_visible(False)
fig.suptitle(f"Day 2 — ODE physics prior comparison (K={K})", fontsize=13, y=1.005)
fig.tight_layout()
fig.savefig(OUT / "ode_levels_K100_grid.png", dpi=140)
print(f"\nPlot: {OUT / 'ode_levels_K100_grid.png'}")
print(f"\nTotal wall-time: {time.time()-t_top:.1f}s")
