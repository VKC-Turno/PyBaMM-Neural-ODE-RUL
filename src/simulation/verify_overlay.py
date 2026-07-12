"""Overlay PyBaMM verification trajectory with raw CALB cell data.

Loads the verification sim trajectory (0.25 C, 25 °C, DFN + SEI(solvent-diff)
+ plating, NO LAM) and overlays the raw measured trajectories from the
CALB stitching cohort (cells 9, 6, 7, 8 — all at 0.25 C/0.25 C per the
internal stitching study).

Each measured cell starts at some used SoH (not 1.0). To compare shapes,
we shift each cell's cycle axis so its first measured point aligns with
the PyBaMM cycle at which the sim first reaches that cell's initial SoH.
That lets us read off model-vs-measured error in the region where both
overlap.

An inset zooms into the SoH 0.5-0.7 mid-life region where all four
measured cells have coverage.

Writes:
    data/synthetic/verification/overlay_with_measured.png
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, "/home/hj/Desktop/PINNs")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = Path("/home/hj/Desktop/PINNs/data/synthetic/verification/overlay_with_measured.png")

# Cells from the stitching study, at 0.25 C/0.25 C
STITCH_CELLS = ["0009", "0006", "0007", "0008"]


def load_sim():
    """PyBaMM verification trajectory."""
    df = pd.read_parquet("/home/hj/Desktop/PINNs/data/synthetic/verification/full_trajectory.parquet")
    return df


def load_measured():
    """Raw canonical CALB longterm data for the stitching-study cells."""
    df = pd.read_parquet("/home/hj/Desktop/PINNs/soh/data/canonical/calb_old.parquet")
    out = {}
    for cid in STITCH_CELLS:
        sub = df[df.cell_id == cid].sort_values("global_cycle")
        n = sub.global_cycle.values.astype(float)
        s = sub.soh.values
        # Filter: keep only positive-SoH points, monotonic
        mask = s > 0.01
        out[cid] = dict(n=n[mask], s=s[mask])
    return out


def align_to_sim(sim: pd.DataFrame, cell_n: np.ndarray, cell_s: np.ndarray):
    """Shift cell_n so cell's first measured SoH aligns with the PyBaMM
    cycle at which sim SoH first drops below that value."""
    if len(cell_s) == 0: return cell_n, np.nan
    s0 = cell_s[0]
    sim_soh = sim.SOH.values
    sim_cy  = sim.cycle_n.values.astype(float)
    below = np.where(sim_soh <= s0)[0]
    if len(below) == 0:
        return cell_n, np.nan
    align_cy = float(sim_cy[below[0]])
    # Shift measured cycles so measured cycle 0 maps to align_cy
    shifted = cell_n - cell_n[0] + align_cy
    return shifted, align_cy


def compute_pointwise_error(sim: pd.DataFrame, meas_n: np.ndarray, meas_s: np.ndarray):
    """For each measured point, find sim SoH at the same cycle. Return
    residual = sim - measured (in pp)."""
    if len(meas_n) == 0: return np.array([]), np.array([])
    sim_at_meas = np.interp(meas_n, sim.cycle_n.values, sim.SOH.values,
                              left=np.nan, right=np.nan)
    valid = ~np.isnan(sim_at_meas)
    residual = (sim_at_meas[valid] - meas_s[valid]) * 100  # pp
    return meas_n[valid], residual


def main():
    sim = load_sim()
    measured = load_measured()

    # Align each measured cell + compute error
    aligned = {}
    print("Alignment info:")
    for cid, m in measured.items():
        shifted_n, align_cy = align_to_sim(sim, m["n"], m["s"])
        _, resid = compute_pointwise_error(sim, shifted_n, m["s"])
        rmse = float(np.sqrt(np.mean(resid**2))) if len(resid) > 0 else float("nan")
        max_err = float(np.max(np.abs(resid))) if len(resid) > 0 else float("nan")
        aligned[cid] = dict(n=shifted_n, s=m["s"], resid=resid,
                              rmse=rmse, max_err=max_err, align_cy=align_cy)
        print(f"  cell {cid}: SoH_start={m['s'][0]:.3f} aligns at PyBaMM cy {align_cy:.0f}, "
              f"RMSE={rmse:.2f} pp, max|err|={max_err:.2f} pp")

    # ── Plot: main axis + zoomed inset ──
    fig, ax = plt.subplots(figsize=(12, 6.5))

    ax.plot(sim.cycle_n, sim.SOH * 100, "b-", lw=2.4,
              label="PyBaMM DFN sim (0.25 C, 25 °C, no LAM)", zorder=5)

    colors = {"0009": "tab:orange", "0006": "tab:green",
               "0007": "tab:purple",  "0008": "tab:brown"}
    for cid, a in aligned.items():
        c = colors[cid]
        ax.scatter(a["n"], a["s"] * 100, s=8, color=c, alpha=0.35,
                     label=f"Cell {cid} measured (RMSE {a['rmse']:.1f} pp)",
                     zorder=3)

    ax.axhline(80, color="tab:orange", ls="--", lw=1.0, alpha=0.6, label="EoL (SoH 0.80)")
    ax.axhline(40, color="tab:red",    ls="--", lw=1.0, alpha=0.6, label="EoSL (SoH 0.40)")
    ax.set_xlabel("Cycle (measured cells aligned to PyBaMM by SoH match)")
    ax.set_ylabel("SoH [%]")
    ax.set_title("PyBaMM DFN long-run verification: overlaid with CALB measured trajectories\n"
                  "(cells 9, 6, 7, 8 at 0.25 C/0.25 C, aligned by initial SoH)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.set_xlim(-100, 4200)
    ax.set_ylim(0, 105)

    # ── Zoomed inset: SoH 40-70 % mid-life region ──
    axins = ax.inset_axes([0.52, 0.42, 0.42, 0.44])
    # Find the sim cycle range corresponding to SoH 40-70 %
    sim_soh_pct = sim.SOH.values * 100
    mask_zoom = (sim_soh_pct >= 40) & (sim_soh_pct <= 70)
    if mask_zoom.any():
        cy_lo = sim.cycle_n.values[mask_zoom].min()
        cy_hi = sim.cycle_n.values[mask_zoom].max()
    else:
        cy_lo, cy_hi = 1000, 3700

    axins.plot(sim.cycle_n, sim_soh_pct, "b-", lw=2.0)
    for cid, a in aligned.items():
        c = colors[cid]
        axins.scatter(a["n"], a["s"] * 100, s=10, color=c, alpha=0.5)
    axins.axhline(40, color="tab:red",    ls="--", lw=0.8, alpha=0.6)
    axins.axhline(80, color="tab:orange", ls="--", lw=0.8, alpha=0.6)
    axins.set_xlim(cy_lo - 100, cy_hi + 100)
    axins.set_ylim(35, 72)
    axins.set_title("Zoomed: mid-life fade (SoH 40-70 %)", fontsize=9)
    axins.grid(alpha=0.3)
    ax.indicate_inset_zoom(axins, edgecolor="grey", alpha=0.5)

    # Aggregate RMSE across all cells
    all_resid = np.concatenate([a["resid"] for a in aligned.values() if len(a["resid"]) > 0])
    overall_rmse = float(np.sqrt(np.mean(all_resid**2))) if len(all_resid) > 0 else float("nan")
    ax.text(0.02, 0.02, f"Combined RMSE across all 4 cells: {overall_rmse:.2f} pp",
             transform=ax.transAxes, fontsize=10,
             bbox=dict(facecolor="white", edgecolor="grey", alpha=0.85))

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"\nWrote {OUT}")
    print(f"Overall RMSE across cells 9, 6, 7, 8: {overall_rmse:.2f} pp")


if __name__ == "__main__":
    main()
