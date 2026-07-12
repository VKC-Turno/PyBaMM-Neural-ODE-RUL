"""Figure 2 for PyBaMM Conf abstract: SoH trajectory overlay.

Shows measured SoH + PyBaMM prediction (calibrated on cycles 0-K) with the
held-out region shaded. Picks cell 25 as the representative "clean" case
(RMSE_test=0.29 pp at K=100, delta_end=+0.21 pp — best-in-cohort extrapolation).

Companion to holdout_rmse_vs_K.png. Output: outputs/holdout_sweep/fig2_trajectory_cell25.png
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/hj/Desktop/PINNs")
sys.path.insert(0, str(Path(__file__).parent))
from calb_clean_cells_rxnlim_sweep import (
    load_measured, load_char_for, run_long_sim, calibrate_rxnlim,
)
from calb_clean_cells_holdout_sweep import train_slope_pp_per_100cy


CELL_ID = 25
K_TRAIN = 100
OUT = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/holdout_sweep")


def main():
    meas = load_measured(CELL_ID)
    pre_age = float(meas.soh.iloc[0])
    first_cy = float(meas.global_cycle.iloc[0])
    start_y  = float(meas.soh_pct.iloc[0])
    n_cycles = int(meas.global_cycle.max())
    k_end = first_cy + K_TRAIN

    char, _ = load_char_for(CELL_ID)
    target = train_slope_pp_per_100cy(meas, K_TRAIN)
    cal = calibrate_rxnlim(char, target, pre_age)
    sim = run_long_sim(char, cal["value"], n_cycles, pre_age)
    sim_cy  = sim.cycle_n.values.astype(float) + (first_cy - sim.cycle_n.values[0])
    sim_anc = sim.SOH.values * 100.0 + (start_y - sim.SOH.values[0] * 100.0)

    # ── plot ──
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    # Held-out region (shaded)
    ax.axvspan(k_end, n_cycles, color="tab:blue", alpha=0.08,
               label="Held-out (predicted)")
    ax.axvspan(first_cy, k_end, color="tab:orange", alpha=0.10,
               label=f"Training window (K={K_TRAIN} cy)")

    # Measured — raw scatter + smoothed line
    ax.scatter(meas.global_cycle, meas.soh_pct, s=4, color="black",
               alpha=0.20, label="Measured (raw)")
    ax.plot(meas.global_cycle, meas.smoothed, color="black", lw=1.5,
            label="Measured (smoothed)")

    # PyBaMM prediction
    ax.plot(sim_cy, sim_anc, color="tab:red", lw=2.0,
            label=f"PyBaMM rxn-lim SEI (fit on 0…{K_TRAIN} cy)")

    # End-of-trajectory annotation
    end_meas = float(meas.soh_pct.iloc[-1])
    end_sim  = float(sim_anc[-1])
    delta = end_sim - end_meas
    ax.annotate(f"Δ end = {delta:+.2f} pp",
                xy=(n_cycles, end_meas),
                xytext=(n_cycles*0.62, end_meas - 3),
                fontsize=10,
                arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.8))

    # K boundary line
    ax.axvline(k_end, color="dimgray", ls="--", lw=1.0, alpha=0.7)

    ax.set_xlabel("Cycle number")
    ax.set_ylabel("SoH [%]")
    ax.set_title(f"CALB_old cell {CELL_ID}: PyBaMM rxn-lim SEI extrapolation "
                 f"from {K_TRAIN} training cycles\n"
                 f"RMSE on held-out cycles: 0.29 pp SoH  ·  "
                 f"prediction covers {n_cycles - K_TRAIN} cycles")
    ax.set_xlim(first_cy - 20, n_cycles + 40)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out_path = OUT / "fig2_trajectory_cell25.png"
    fig.savefig(out_path, dpi=180)
    print(f"Figure: {out_path}")

    # Also emit vector for LaTeX
    fig.savefig(OUT / "fig2_trajectory_cell25.pdf")
    print(f"PDF: {OUT / 'fig2_trajectory_cell25.pdf'}")


if __name__ == "__main__":
    main()
