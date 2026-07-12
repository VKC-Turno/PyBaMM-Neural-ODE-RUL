"""Joint reaction-limited SEI + negative-LAM calibration for CALB_old cell 26.

The single-mechanism fits in `calb_26_sei_comparison.py` confirmed that pure
SEI growth (either solvent-diffusion or reaction-limited) cannot reproduce
cell 26's three-regime measured shape — fast early fade, accelerating mid,
flattening late. That shape is the classical signature of LAM (loss of
active material) kicking in mid-life. This script adds LAM_neg as a second
free parameter and fits both jointly against the measured trajectory.

Free parameters:
    j_SEI    = "SEI reaction exchange current density [A.m-2]"   (rxn-lim mode)
    LAM_neg  = "Negative electrode LAM constant proportional term [s-1]"
               (stress-driven LAM, PyBaMM default 2.78e-7)

Objective:
    Minimize RMSE between PyBaMM sim and Hampel-filtered measured SoH over
    a 400-cycle calibration window (cycles 3-403, captures the early →
    accelerating-mid transition).

Method: scipy.optimize.minimize (Nelder-Mead) on a 2D log-space.

Output:
    CALB_old_26_joint_calibrated.json
    CALB_old_26_joint_sim_1608cy.parquet
    CALB_old_26_joint_compare.png  — overlay rxn-lim-only vs joint vs measured
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

sys.path.insert(0, "/home/hj/Desktop/PINNs")
from pybamm_tuning import build_pybamm_parameters, load_characterization
from pybamm_tuning.simulation import CyclingProtocol, Simulation


# ─────────────────────────── config ───────────────────────────
TAG = "CALB_old_26"
PROTOCOL = CyclingProtocol(c_rate=0.25)
TEMP_K   = 298.15
N_CYCLES_CALIBRATION = 400        # captures early + start of mid-window acceleration
N_CYCLES_LONG        = 1608       # match measured horizon
SKIP_FIRST_N = 1                   # skip the batch2 cycle-1 conditioning anomaly

CACHE_DIR = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/pybamm_cache")
OUT_DIR   = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/tuned_params")

JOINT_DFN_OPTS = {
    "SEI": "reaction limited",
    "SEI porosity change": "true",
    "lithium plating": "none",
    "loss of active material": "stress-driven",
}

KEY_J  = "SEI reaction exchange current density [A.m-2]"
KEY_LAM = "Negative electrode LAM constant proportional term [s-1]"

# Initial guess (log10 space): rxn-lim's prior calibration + PyBaMM LAM default
X0 = np.array([np.log10(1.0e-7),     # log10(j_SEI)
                np.log10(2.78e-7)])    # log10(LAM_neg)

# Bounds (we'll clip — Nelder-Mead doesn't enforce, so we add a soft penalty)
BOUNDS = [(-9.0, -5.0),    # log10(j_SEI)
          (-9.0, -4.0)]    # log10(LAM_neg)


def _drop_outliers(series: pd.Series, k: float = 3.0, window: int = 5) -> pd.Series:
    if len(series) < window:
        return pd.Series([True] * len(series), index=series.index)
    med = series.rolling(window, center=True, min_periods=1).median()
    mad = (series - med).abs().rolling(window, center=True, min_periods=1).median()
    threshold = k * 1.4826 * mad.clip(lower=1e-9)
    return (series - med).abs() <= threshold


def load_measured() -> pd.DataFrame:
    df = pd.read_parquet("soh/data/canonical/calb_old.parquet")
    c = df[df.cell_id.astype(str).str.zfill(4) == "0026"].sort_values("global_cycle")
    c = c[c.global_cycle >= SKIP_FIRST_N + 1].copy()
    c["soh_pct"] = c["soh"] * 100.0
    keep = _drop_outliers(c["soh_pct"], k=3.0, window=5)
    c = c[keep].copy()
    return c


def run_sim(char, j_sei: float, lam_neg: float, n_cycles: int,
            pre_age: float) -> pd.DataFrame:
    params = build_pybamm_parameters(
        char, base="Prada2013", temperature_K=TEMP_K,
        extra_overrides={KEY_J: j_sei, KEY_LAM: lam_neg},
        pre_age_to_soh=pre_age,
    )
    sim = Simulation(params, protocol=PROTOCOL, cache_dir=CACHE_DIR,
                      dfn_options=JOINT_DFN_OPTS)
    return sim.run(n_cycles=n_cycles)


# ─────────────────────────── main ───────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t_top = time.time()
    print(f"=== {TAG} — joint rxn-lim SEI + LAM_neg calibration ===")

    meas = load_measured()
    first_cy   = float(meas.global_cycle.iloc[0])
    start_y    = float(meas.soh_pct.iloc[0])
    pre_age    = float(meas.soh.iloc[0])

    # Measured points inside the calibration window
    cal_meas = meas[meas.global_cycle.between(first_cy, first_cy + N_CYCLES_CALIBRATION - 1)]
    print(f"  measured: {len(cal_meas)} points in calibration window "
          f"(cy {int(cal_meas.global_cycle.min())}-{int(cal_meas.global_cycle.max())}), "
          f"pre_age={pre_age:.3f}")

    char = load_characterization(cohort="CALB_old", aggregate=True)
    n_evals = [0]
    best = {"rmse": float("inf"), "x": None}

    def rmse_for(log10_pair: np.ndarray) -> float:
        # Soft clip to bounds (heavy penalty outside)
        for i, ((lo, hi), v) in enumerate(zip(BOUNDS, log10_pair)):
            if v < lo or v > hi:
                return 100.0 + abs(v - max(lo, min(hi, v))) * 100  # large
        j_sei = 10 ** log10_pair[0]
        lam   = 10 ** log10_pair[1]
        try:
            sim = run_sim(char, j_sei, lam, N_CYCLES_CALIBRATION, pre_age)
        except Exception as e:
            print(f"    ✗ sim failed (j={j_sei:.2e}, lam={lam:.2e}): {e}")
            return 50.0
        # Anchor sim to measured at cycle 3
        soh = sim.SOH.values * 100.0
        cy  = sim.cycle_n.values.astype(float) + (first_cy - sim.cycle_n.values[0])
        soh_anchored = soh + (start_y - soh[0])
        # RMSE on measured cycles inside the calibration window
        sim_at_meas = np.interp(cal_meas.global_cycle.values, cy, soh_anchored)
        rmse = float(np.sqrt(np.mean((sim_at_meas - cal_meas.soh_pct.values)**2)))
        n_evals[0] += 1
        if rmse < best["rmse"]:
            best["rmse"] = rmse
            best["x"]    = log10_pair.copy()
            marker = "  ←best"
        else:
            marker = ""
        print(f"    eval {n_evals[0]:>2}  j={j_sei:.2e}  lam={lam:.2e}  RMSE={rmse:.3f} pp{marker}")
        return rmse

    print(f"\n  Nelder-Mead from x0 = (log10 j_SEI={X0[0]:.2f}, log10 LAM={X0[1]:.2f})")
    t_opt = time.time()
    result = minimize(rmse_for, X0, method="Nelder-Mead",
                       options={"xatol": 0.05, "fatol": 0.05, "maxiter": 60})
    print(f"\n  Nelder-Mead done in {time.time()-t_opt:.1f}s, {n_evals[0]} evaluations")
    print(f"    Best RMSE = {best['rmse']:.3f} pp at "
          f"j_SEI = {10**best['x'][0]:.3e}, LAM_neg = {10**best['x'][1]:.3e}")

    # Long sim with best parameters
    j_best = float(10 ** best["x"][0])
    lam_best = float(10 ** best["x"][1])
    print(f"\n  Running {N_CYCLES_LONG}-cycle sim with best params…")
    t_long = time.time()
    long_sim = run_sim(char, j_best, lam_best, N_CYCLES_LONG, pre_age)
    long_sim.to_parquet(OUT_DIR / f"{TAG}_joint_sim_{N_CYCLES_LONG}cy.parquet")
    print(f"    wrote parquet, {time.time()-t_long:.1f}s")

    # Save calibration JSON
    long_soh = long_sim.SOH.values * 100.0
    long_cy  = long_sim.cycle_n.values.astype(float) + (first_cy - long_sim.cycle_n.values[0])
    sim_anchored = long_soh + (start_y - long_soh[0])
    full_rmse = float(np.sqrt(np.mean(
        (np.interp(meas.global_cycle.values, long_cy, sim_anchored) - meas.soh_pct.values)**2)))
    payload = {
        "cell": TAG,
        "model": "rxn-lim SEI + LAM_neg (stress-driven)",
        "calibration_window_cycles": int(N_CYCLES_CALIBRATION),
        "best_calib_rmse_pp": best["rmse"],
        "full_traj_rmse_pp": full_rmse,
        "params": {
            KEY_J:   j_best,
            KEY_LAM: lam_best,
        },
        "log10_j_SEI": float(best["x"][0]),
        "log10_LAM_neg": float(best["x"][1]),
        "n_optimizer_evals": n_evals[0],
        "pre_age_to_soh": pre_age,
        "dfn_options": {k: str(v) for k, v in JOINT_DFN_OPTS.items()},
        "char_source": "cohort-median CALB_old",
        "method": "Nelder-Mead 2D log-space",
        "skip_first_n_cycles": SKIP_FIRST_N,
        "wall_time_s": time.time() - t_top,
    }
    (OUT_DIR / f"{TAG}_joint_calibrated.json").write_text(
        json.dumps(payload, indent=2, default=str))
    print(f"\n  Full-trajectory RMSE: {full_rmse:.3f} pp "
          f"(rxn-lim-only was 6.30, solv-diff was 9.71)")

    # ── Compare plot: rxn-lim-only vs joint vs measured ──
    rxn_only = pd.read_parquet(OUT_DIR / f"{TAG}_rxnlim_sim_{N_CYCLES_LONG}cy.parquet")
    rxn_soh = rxn_only.SOH.values * 100.0
    rxn_cy  = rxn_only.cycle_n.values.astype(float) + (first_cy - rxn_only.cycle_n.values[0])
    rxn_anchored = rxn_soh + (start_y - rxn_soh[0])

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for ax, xlim, title in [
        (axs[0], None, f"{TAG} — full {N_CYCLES_LONG}-cy horizon"),
        (axs[1], (first_cy, first_cy + 400),
         f"{TAG} — calibration window (≤ {int(first_cy)+400} cy)"),
    ]:
        ax.plot(meas.global_cycle, meas.soh_pct, "o-", lw=0.6, ms=1.5,
                color="#d62728", label=f"measured ({len(meas)} pts)")
        ax.plot(rxn_cy, rxn_anchored, "--", lw=1.4, color="#2ca02c",
                label="sim — rxn-lim only (RMSE 6.30 pp)")
        ax.plot(long_cy, sim_anchored, "-", lw=1.6, color="#9467bd",
                label=f"sim — joint rxn-lim + LAM_neg (RMSE {full_rmse:.2f} pp)")
        if xlim:
            ax.set_xlim(*xlim)
            ywin = np.concatenate([
                meas[meas.global_cycle.between(*xlim)].soh_pct.values,
                rxn_anchored[(rxn_cy >= xlim[0]) & (rxn_cy <= xlim[1])],
                sim_anchored[(long_cy >= xlim[0]) & (long_cy <= xlim[1])],
            ])
            if ywin.size:
                pad = max(0.5, (ywin.max() - ywin.min()) * 0.1)
                ax.set_ylim(ywin.min() - pad, ywin.max() + pad)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Cycle"); ax.set_ylabel("SoH (%)")
        ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        f"{TAG} — joint rxn-lim SEI + LAM_neg vs rxn-lim-only "
        f"(j_SEI={j_best:.2e}, LAM={lam_best:.2e})",
        fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{TAG}_joint_compare.png", dpi=120)
    plt.close(fig)
    print(f"\n=== Total wall-time: {time.time()-t_top:.1f}s ===")


if __name__ == "__main__":
    main()
