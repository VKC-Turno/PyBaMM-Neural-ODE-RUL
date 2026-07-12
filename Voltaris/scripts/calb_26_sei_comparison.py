"""CALB_old cell 26 — SEI sub-model comparison over 1,608 measured cycles.

Goal: test which PyBaMM SEI sub-model (solvent-diffusion vs reaction-limited)
fits the measured concave-down fade arc better. Cell 26 is uniquely valuable
because it's the only CALB_old cell with a clean 1,608-cycle SoH trajectory
that spans ~17 pp of real aging — 10× the signal of any REPT cell we've
calibrated against.

Caveats:
    - Cell 26 is NOT in the char workbook (gap in CALB_old chars at IDs 25/26/28).
      Use cohort-median CALB_old char as the electrochemistry baseline.
    - Cycle 1 in batch2 shows a conditioning anomaly (SoH 0.81 → 0.59 in one
      cycle); skip cycle 1 and treat cycle 2 as the start of the aging arc.
    - Pre-age PyBaMM to the cycle-2 measured SoH (~0.591) so the sim begins
      from a comparable aged state.

Outputs:
    CALB_old_26_solvdiff_calibrated.json + _solvdiff_sim_1608cy.parquet
    CALB_old_26_rxnlim_calibrated.json   + _rxnlim_sim_1608cy.parquet
    CALB_old_26_sei_compare.png
    CALB_old_26_measured_per_cycle.csv
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

sys.path.insert(0, "/home/hj/Desktop/PINNs")
from pybamm_tuning import build_pybamm_parameters, load_characterization
from pybamm_tuning.simulation import CyclingProtocol, Simulation


# ─────────────────────────── config ───────────────────────────
CELL_ID = "26"
COHORT  = "CALB_old"
TAG     = f"{COHORT}_26"
PROTOCOL = CyclingProtocol(c_rate=0.25)   # matches our REPT/EVE convention
TEMP_K = 298.15
N_CYCLES_CALIBRATION = 10
N_CYCLES_LONG = 1608   # match measured horizon exactly
SKIP_FIRST_N = 1       # skip the batch2 cycle-1 conditioning anomaly

CACHE_DIR = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/pybamm_cache")
OUT_DIR   = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/tuned_params")
CANON_PQ  = Path("/home/hj/Desktop/PINNs/soh/data/canonical/calb_old.parquet")

SOLVDIFF_OPTS = {
    "SEI": "solvent-diffusion limited",
    "SEI porosity change": "true",
    "lithium plating": "none",
    "loss of active material": "none",
}
RXNLIM_OPTS = {
    "SEI": "reaction limited",
    "SEI porosity change": "true",
    "lithium plating": "none",
    "loss of active material": "none",
}


# ─────────────────────────── helpers ───────────────────────────
def _slope_pp_per_100cy(cyc: np.ndarray, soh_pct: np.ndarray) -> float:
    if cyc.size < 4:
        return float("nan")
    s, _ = np.polyfit(cyc[1:], soh_pct[1:], 1)
    return float(s * 100.0)


def _drop_outliers(series: pd.Series, k: float = 3.0, window: int = 5) -> pd.Series:
    if len(series) < window:
        return pd.Series([True] * len(series), index=series.index)
    med = series.rolling(window, center=True, min_periods=1).median()
    mad = (series - med).abs().rolling(window, center=True, min_periods=1).median()
    threshold = k * 1.4826 * mad.clip(lower=1e-9)
    return (series - med).abs() <= threshold


def measured_trajectory() -> pd.DataFrame:
    df = pd.read_parquet(CANON_PQ)
    c = df[df.cell_id.astype(str).str.zfill(4) == CELL_ID.zfill(4)].sort_values("global_cycle")
    c = c[c.global_cycle >= SKIP_FIRST_N + 1].copy()
    c["soh_pct"] = c["soh"] * 100.0
    keep = _drop_outliers(c["soh_pct"], k=3.0, window=5)
    c["kept"] = keep
    return c


def calibrate(target_slope: float, dfn_opts: dict, param_key: str,
              log10_bracket: tuple[float, float], rtol: float, char,
              pre_age_to_soh, label: str) -> dict:
    """Bisect log10(param) so PyBaMM's slope matches `target_slope`."""
    n_fresh = [0]

    def slope_for(log10_val: float) -> float:
        params = build_pybamm_parameters(
            char, base="Prada2013", temperature_K=TEMP_K,
            extra_overrides={param_key: 10 ** log10_val},
            pre_age_to_soh=pre_age_to_soh,
        )
        sim = Simulation(params, protocol=PROTOCOL, cache_dir=CACHE_DIR,
                          dfn_options=dfn_opts)
        df = sim.run(n_cycles=N_CYCLES_CALIBRATION)
        if not getattr(sim, "last_was_cached", True):
            n_fresh[0] += 1
        soh = df["SOH"].to_numpy(dtype=float) * 100.0
        cyc = df["cycle_n"].to_numpy(dtype=float)
        return _slope_pp_per_100cy(cyc, soh)

    lo, hi = log10_bracket
    slope_lo, slope_hi = slope_for(lo), slope_for(hi)
    n_evals = 2
    print(f"    {label}: bracket [{lo}, {hi}] → slope [{slope_lo:+.4f}, {slope_hi:+.4f}], "
          f"target {target_slope:+.4f}")

    if (slope_lo - target_slope) * (slope_hi - target_slope) > 0:
        # Target outside bracket — return the nearest endpoint
        if abs(slope_lo - target_slope) < abs(slope_hi - target_slope):
            best_log10, best_slope = lo, slope_lo
        else:
            best_log10, best_slope = hi, slope_hi
        return {"log10_value": best_log10, "fitted_value": 10 ** best_log10,
                "achieved_slope": best_slope, "n_evals": n_evals,
                "n_fresh_sims": n_fresh[0], "note": "bracket-edge — target outside bracket"}

    mid = 0.5 * (lo + hi)
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        slope_mid = slope_for(mid); n_evals += 1
        if abs(slope_mid - target_slope) <= rtol * max(abs(target_slope), 0.05):
            break
        if (slope_lo - target_slope) * (slope_mid - target_slope) < 0:
            hi, slope_hi = mid, slope_mid
        else:
            lo, slope_lo = mid, slope_mid

    return {"log10_value": mid, "fitted_value": 10 ** mid,
            "achieved_slope": slope_mid, "n_evals": n_evals,
            "n_fresh_sims": n_fresh[0], "note": "converged"}


def long_sim(char, dfn_opts: dict, param_key: str, fitted_value: float,
              pre_age_to_soh) -> pd.DataFrame:
    params = build_pybamm_parameters(
        char, base="Prada2013", temperature_K=TEMP_K,
        extra_overrides={param_key: fitted_value},
        pre_age_to_soh=pre_age_to_soh,
    )
    sim = Simulation(params, protocol=PROTOCOL, cache_dir=CACHE_DIR,
                      dfn_options=dfn_opts)
    return sim.run(n_cycles=N_CYCLES_LONG)


# ─────────────────────────── main ───────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== CALB_old cell 26 — SEI sub-model comparison ===")
    t_top = time.time()

    # 1) Load measured trajectory
    meas = measured_trajectory()
    meas[["global_cycle","soh","soh_pct","kept"]].to_csv(
        OUT_DIR / f"{TAG}_measured_per_cycle.csv", index=False)
    clean = meas[meas.kept]
    n_dropped = (~meas.kept).sum()
    print(f"  measured: {len(clean):,} clean cycles (cy {int(clean.global_cycle.min())}-"
          f"{int(clean.global_cycle.max())}), {n_dropped} outliers dropped")
    target_slope = float(np.polyfit(clean.global_cycle, clean.soh_pct, 1)[0] * 100)
    measured_start_soh = float(clean.soh.iloc[0])
    print(f"  measured target slope (whole window): {target_slope:+.4f} pp/100cy")
    print(f"  measured start SoH (cy {int(clean.global_cycle.iloc[0])}): "
          f"{measured_start_soh:.3f}")

    # 2) Load cohort-median CALB_old char (cell 26 not in workbook)
    char = load_characterization(cohort=COHORT, aggregate=True)
    print(f"  using cohort-median CALB_old char (q_rpt={char.q_rpt_ah:.2f} Ah, "
          f"n={char.cell_id})")
    pre_age = float(measured_start_soh)
    print(f"  pre-aging PyBaMM to SoH = {pre_age:.3f} to match measured start")

    # 3) Calibrate both sub-models
    print(f"\n  calibrating solvent-diffusion (D_SEI)…")
    t = time.time()
    sd = calibrate(target_slope, SOLVDIFF_OPTS,
                    "SEI solvent diffusivity [m2.s-1]",
                    log10_bracket=(-24.0, -19.0), rtol=0.20,
                    char=char, pre_age_to_soh=pre_age, label="solv-diff")
    print(f"    D_SEI = {sd['fitted_value']:.3e} m²/s, "
          f"achieved {sd['achieved_slope']:+.4f} pp/100cy, "
          f"{sd['n_fresh_sims']}/{sd['n_evals']} sims, {time.time()-t:.1f}s")

    print(f"\n  calibrating reaction-limited (j_SEI)…")
    t = time.time()
    rx = calibrate(target_slope, RXNLIM_OPTS,
                    "SEI reaction exchange current density [A.m-2]",
                    log10_bracket=(-10.0, -6.0), rtol=0.20,
                    char=char, pre_age_to_soh=pre_age, label="rxn-lim")
    print(f"    j_SEI = {rx['fitted_value']:.3e} A/m², "
          f"achieved {rx['achieved_slope']:+.4f} pp/100cy, "
          f"{rx['n_fresh_sims']}/{rx['n_evals']} sims, {time.time()-t:.1f}s")

    # 4) Long sims
    print(f"\n  running {N_CYCLES_LONG}-cycle sim (solv-diff)…")
    t = time.time()
    sd_sim = long_sim(char, SOLVDIFF_OPTS,
                       "SEI solvent diffusivity [m2.s-1]",
                       sd["fitted_value"], pre_age)
    sd_sim.to_parquet(OUT_DIR / f"{TAG}_solvdiff_sim_{N_CYCLES_LONG}cy.parquet")
    print(f"    wrote parquet, {time.time()-t:.1f}s")

    print(f"\n  running {N_CYCLES_LONG}-cycle sim (rxn-lim)…")
    t = time.time()
    rx_sim = long_sim(char, RXNLIM_OPTS,
                       "SEI reaction exchange current density [A.m-2]",
                       rx["fitted_value"], pre_age)
    rx_sim.to_parquet(OUT_DIR / f"{TAG}_rxnlim_sim_{N_CYCLES_LONG}cy.parquet")
    print(f"    wrote parquet, {time.time()-t:.1f}s")

    # 5) Save calibration JSONs
    for label, payload, key in [
        ("solvdiff", sd, "SEI solvent diffusivity [m2.s-1]"),
        ("rxnlim",   rx, "SEI reaction exchange current density [A.m-2]"),
    ]:
        out = {
            "cell": TAG,
            "cohort": COHORT,
            "model": label,
            "target_slope_pp_per_100cy": target_slope,
            "calibrated_param": key,
            "calibrated_value": payload["fitted_value"],
            "log10_value": payload["log10_value"],
            "achieved_slope_pp_per_100cy": payload["achieved_slope"],
            "residual_pp_per_100cy": payload["achieved_slope"] - target_slope,
            "relative_error_pct": abs(payload["achieved_slope"] - target_slope) /
                                   max(abs(target_slope), 1e-9) * 100.0,
            "n_evaluations": payload["n_evals"],
            "n_fresh_sims": payload["n_fresh_sims"],
            "note": payload["note"],
            "pre_age_to_soh": pre_age,
            "n_cycles_long": N_CYCLES_LONG,
            "char_source": "cohort-median CALB_old (cell 26 not in workbook)",
            "skip_first_n_cycles": SKIP_FIRST_N,
        }
        (OUT_DIR / f"{TAG}_{label}_calibrated.json").write_text(
            json.dumps(out, indent=2, default=str))

    # 6) Comparison plot
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    sd_soh = sd_sim["SOH"].values * 100.0
    sd_cy  = sd_sim["cycle_n"].values.astype(float)
    rx_soh = rx_sim["SOH"].values * 100.0
    rx_cy  = rx_sim["cycle_n"].values.astype(float)
    # Anchor at first kept measured cycle
    first_cy = float(clean.global_cycle.iloc[0])
    start_y  = float(clean.soh_pct.iloc[0])
    def _anchor(cy, soh):
        return cy + (first_cy - cy[0]), soh + (start_y - soh[0])
    sd_x, sd_y = _anchor(sd_cy, sd_soh)
    rx_x, rx_y = _anchor(rx_cy, rx_soh)

    for ax, xlim, title in [
        (axs[0], None, f"{TAG} — full {N_CYCLES_LONG}-cy horizon"),
        (axs[1], (first_cy, first_cy + 200),
         f"{TAG} — early window (≤ {int(first_cy)+200} cy)"),
    ]:
        ax.plot(clean.global_cycle, clean.soh_pct, "o-", lw=0.6, ms=1.5,
                 color="#d62728", label=f"measured (canonical, {len(clean)} pts)")
        ax.plot(sd_x, sd_y, "-", lw=1.2, color="#1f77b4",
                 label=f"sim — solv-diff (D_SEI={sd['fitted_value']:.1e})")
        ax.plot(rx_x, rx_y, "--", lw=1.2, color="#2ca02c",
                 label=f"sim — rxn-lim (j_SEI={rx['fitted_value']:.1e})")
        if xlim:
            ax.set_xlim(*xlim)
            ywin = np.concatenate([
                clean[clean.global_cycle.between(*xlim)].soh_pct.values,
                sd_y[(sd_x >= xlim[0]) & (sd_x <= xlim[1])],
                rx_y[(rx_x >= xlim[0]) & (rx_x <= xlim[1])],
            ])
            if ywin.size:
                pad = max(0.5, (ywin.max() - ywin.min()) * 0.1)
                ax.set_ylim(ywin.min() - pad, ywin.max() + pad)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Cycle"); ax.set_ylabel("SoH (%)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle(f"{TAG} — SEI sub-model comparison "
                  f"(target {target_slope:+.3f} pp/100cy, "
                  f"measured ends at {clean.soh_pct.iloc[-1]:.1f}%)",
                  fontsize=12, y=1.01)
    fig.tight_layout()
    out_png = OUT_DIR / f"{TAG}_sei_compare.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"\n  comparison PNG: {out_png.name}")
    print(f"\n=== Total wall-time: {time.time()-t_top:.1f}s ===")


if __name__ == "__main__":
    main()
