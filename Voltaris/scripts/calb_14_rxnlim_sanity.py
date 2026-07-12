"""CALB_old cell 14 — pure reaction-limited SEI sanity check.

Cell 14 has the cleanest long-trajectory data in CALB_old (noise std 0.29 pp,
4.5× cleaner than cell 26) and a near-linear fade shape — early/mid/late
slopes all in [-0.81, -0.91] pp/100cy. If pure rxn-lim SEI can match this
shape, it suggests the joint SEI+LAM machinery from cell 26 was overkill
for the typical near-linear CALB cell.

Setup:
    - Use cell-14-specific char (q_rpt 46.32 Ah, SoH 64.3 %)
    - DFN: SEI = "reaction limited", lithium plating = "none", LAM = "none"
    - Pre-age PyBaMM to the cycle-2 measured SoH (~0.628)
    - Target slope = whole-trajectory linear slope of the cleaned measured curve
    - Calibrate j_SEI; run 1301-cycle sim; report full-trajectory RMSE
"""
from __future__ import annotations

import json, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/hj/Desktop/PINNs")
from pybamm_tuning import build_pybamm_parameters, load_characterization
from pybamm_tuning.simulation import CyclingProtocol, Simulation


# ─────────────────────────── config ───────────────────────────
CELL_ID = "14"
TAG     = f"CALB_old_{CELL_ID}"
PROTOCOL = CyclingProtocol(c_rate=0.25)
TEMP_K   = 298.15
N_CYCLES_CALIBRATION = 10
N_CYCLES_LONG = 1301
SKIP_FIRST_N = 1

CACHE_DIR = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/pybamm_cache")
OUT_DIR   = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/tuned_params")
CANON_PQ  = Path("/home/hj/Desktop/PINNs/soh/data/canonical/calb_old.parquet")

RXNLIM_OPTS = {
    "SEI": "reaction limited",
    "SEI porosity change": "true",
    "lithium plating": "none",
    "loss of active material": "none",
}
KEY_J = "SEI reaction exchange current density [A.m-2]"


def _drop_outliers(series: pd.Series, k: float = 3.0, window: int = 5) -> pd.Series:
    if len(series) < window:
        return pd.Series([True] * len(series), index=series.index)
    med = series.rolling(window, center=True, min_periods=1).median()
    mad = (series - med).abs().rolling(window, center=True, min_periods=1).median()
    threshold = k * 1.4826 * mad.clip(lower=1e-9)
    return (series - med).abs() <= threshold


def load_measured() -> pd.DataFrame:
    df = pd.read_parquet(CANON_PQ)
    c = df[df.cell_id.astype(str).str.zfill(4) == CELL_ID.zfill(4)].sort_values("global_cycle")
    c = c[(c.global_cycle >= SKIP_FIRST_N + 1) & (c.soh > 0.05)].copy()
    c["soh_pct"] = c.soh * 100.0
    keep = _drop_outliers(c["soh_pct"], k=3.0, window=5)
    return c[keep].copy()


def slope_for(char, j_sei, n_cycles, pre_age):
    params = build_pybamm_parameters(
        char, base="Prada2013", temperature_K=TEMP_K,
        extra_overrides={KEY_J: j_sei},
        pre_age_to_soh=pre_age,
    )
    sim = Simulation(params, protocol=PROTOCOL, cache_dir=CACHE_DIR,
                      dfn_options=RXNLIM_OPTS)
    df = sim.run(n_cycles=n_cycles)
    cy = df.cycle_n.to_numpy(dtype=float)
    soh = df.SOH.to_numpy(dtype=float) * 100.0
    s, _ = np.polyfit(cy[1:], soh[1:], 1)
    return float(s * 100), df


def calibrate(char, target, pre_age, bracket=(-10.0, -6.0), rtol=0.20, max_iter=12):
    n_fresh = [0]; n_evals = 0
    def f(log10_j):
        nonlocal n_evals
        slope, _ = slope_for(char, 10**log10_j, N_CYCLES_CALIBRATION, pre_age)
        n_evals += 1
        return slope
    lo, hi = bracket
    sl_lo, sl_hi = f(lo), f(hi)
    print(f"    bracket [{lo}, {hi}] → slopes [{sl_lo:+.4f}, {sl_hi:+.4f}], target {target:+.4f}")
    if (sl_lo - target) * (sl_hi - target) > 0:
        best = lo if abs(sl_lo - target) < abs(sl_hi - target) else hi
        return {"log10": best, "value": 10**best,
                "achieved": sl_lo if best==lo else sl_hi,
                "n_evals": n_evals, "note": "bracket-edge"}
    mid = 0.5 * (lo + hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        sl_mid = f(mid)
        if abs(sl_mid - target) <= rtol * max(abs(target), 0.05): break
        if (sl_lo - target) * (sl_mid - target) < 0:
            hi, sl_hi = mid, sl_mid
        else:
            lo, sl_lo = mid, sl_mid
    return {"log10": mid, "value": 10**mid, "achieved": sl_mid,
            "n_evals": n_evals, "note": "converged"}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== {TAG} — rxn-lim-only sanity check ===")
    t_top = time.time()

    meas = load_measured()
    meas["smoothed"] = meas.soh_pct.rolling(50, center=True, min_periods=10).median()
    # Whole-window target slope on the SMOOTHED trace (robust to noise)
    target = float(np.polyfit(meas.global_cycle, meas.soh_pct, 1)[0] * 100)
    pre_age = float(meas.soh.iloc[0])
    print(f"  measured: {len(meas)} cycles, target slope (whole) = {target:+.4f} pp/100cy")
    print(f"  pre-age PyBaMM to SoH = {pre_age:.3f}")

    char = load_characterization(cohort="CALB_old", cell_id=CELL_ID)
    print(f"  using cell-14-specific char: q_rpt={char.q_rpt_ah:.2f} Ah, SoH={char.soh_pct:.2f}%")

    print(f"\n  calibrating j_SEI…")
    t = time.time()
    cal = calibrate(char, target, pre_age, bracket=(-10.0, -6.0))
    print(f"    j_SEI = {cal['value']:.3e}, achieved {cal['achieved']:+.4f} pp/100cy, "
          f"{cal['n_evals']} evals, {time.time()-t:.1f}s")

    # 1301-cycle sim
    print(f"\n  running {N_CYCLES_LONG}-cycle sim…")
    t = time.time()
    _, sim = slope_for(char, cal["value"], N_CYCLES_LONG, pre_age)
    sim.to_parquet(OUT_DIR / f"{TAG}_rxnlim_sim_{N_CYCLES_LONG}cy.parquet")
    print(f"    {time.time()-t:.1f}s")

    # Anchor + RMSE
    first_cy = float(meas.global_cycle.iloc[0])
    start_y  = float(meas.soh_pct.iloc[0])
    sim_soh = sim.SOH.values * 100.0
    sim_cy  = sim.cycle_n.values.astype(float) + (first_cy - sim.cycle_n.values[0])
    sim_anc = sim_soh + (start_y - sim_soh[0])
    interp = np.interp(meas.global_cycle.values, sim_cy, sim_anc)
    rmse = float(np.sqrt(np.mean((interp - meas.soh_pct.values)**2)))
    print(f"\n  Full-trajectory RMSE: {rmse:.3f} pp")
    print(f"    (cell 26 rxn-lim-only was 6.30 pp; cell 26 joint was 8.85 full / 2.73 calib window)")
    print(f"  End-of-test:")
    print(f"    measured @ cy {int(meas.global_cycle.iloc[-1])}: {meas.soh_pct.iloc[-1]:.2f}%")
    print(f"    sim      @ cy {int(sim_cy[-1])}:               {sim_anc[-1]:.2f}%  (Δ {sim_anc[-1]-meas.soh_pct.iloc[-1]:+.2f} pp)")

    # Save
    payload = {
        "cell": TAG, "model": "rxn-lim-only",
        "target_slope_pp_per_100cy": target,
        "calibrated_param": KEY_J,
        "calibrated_value": cal["value"],
        "log10_value": cal["log10"],
        "achieved_slope_pp_per_100cy": cal["achieved"],
        "residual_pp_per_100cy": cal["achieved"] - target,
        "relative_error_pct": abs(cal["achieved"] - target)/max(abs(target), 1e-9)*100,
        "full_traj_rmse_pp": rmse,
        "sim_end_soh_pct": sim_anc[-1],
        "meas_end_soh_pct": float(meas.soh_pct.iloc[-1]),
        "delta_end_pp": sim_anc[-1] - float(meas.soh_pct.iloc[-1]),
        "n_evaluations": cal["n_evals"],
        "pre_age_to_soh": pre_age,
        "n_cycles_long": N_CYCLES_LONG,
        "char_source": "cell-14-specific",
        "skip_first_n_cycles": SKIP_FIRST_N,
    }
    (OUT_DIR / f"{TAG}_rxnlim_calibrated.json").write_text(json.dumps(payload, indent=2, default=str))
    meas[["global_cycle","soh","soh_pct"]].to_csv(OUT_DIR / f"{TAG}_measured_per_cycle.csv", index=False)

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for ax, xl, title in [
        (axs[0], None, f"{TAG} — full {N_CYCLES_LONG}-cy horizon"),
        (axs[1], (first_cy, first_cy + 300), f"{TAG} — early window (≤ {int(first_cy)+300} cy)"),
    ]:
        ax.plot(meas.global_cycle, meas.soh_pct, "o-", lw=0.6, ms=1.5,
                color="#d62728", label=f"measured ({len(meas)} pts)")
        ax.plot(sim_cy, sim_anc, "-", lw=1.4, color="#2ca02c",
                label=f"sim — rxn-lim (j_SEI={cal['value']:.1e}, RMSE {rmse:.2f} pp)")
        if xl:
            ax.set_xlim(*xl)
            sel = (meas.global_cycle >= xl[0]) & (meas.global_cycle <= xl[1])
            ywin = np.concatenate([
                meas[sel].soh_pct.values,
                sim_anc[(sim_cy >= xl[0]) & (sim_cy <= xl[1])],
            ])
            if ywin.size:
                pad = max(0.3, (ywin.max() - ywin.min()) * 0.1)
                ax.set_ylim(ywin.min() - pad, ywin.max() + pad)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Cycle"); ax.set_ylabel("SoH (%)")
        ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=9)
    fig.suptitle(f"{TAG} — pure reaction-limited SEI vs measured "
                  f"(target {target:+.3f} pp/100cy)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{TAG}_rxnlim_compare.png", dpi=120)
    plt.close(fig)
    print(f"\n=== Wall-time: {time.time()-t_top:.1f}s ===")


if __name__ == "__main__":
    main()
