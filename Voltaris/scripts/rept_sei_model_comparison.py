"""SEI model comparison — solvent-diffusion vs reaction-limited.

For REPT_87 and REPT_7, calibrates the SEI knob under two different PyBaMM
SEI sub-models and runs a 2000-cycle sim with each. The output overlays
both sims against the measured longterm CSV so the SHAPES (not just slopes)
can be compared.

Why this experiment:
    The current production calibration uses "solvent-diffusion limited" SEI.
    Its `dSEI/dt ∝ 1/SEI_thickness` produces a concave-up SoH(cycle) curve
    that flattens with time. Measured cells in the REPT cohort look more
    linear over the first 150 cycles. Reaction-limited SEI grows at a roughly
    constant rate per cycle, giving a near-linear early curve. If that
    matches the measured shape better, the production calibration should
    switch sub-models — or at least add `reaction limited` as a fallback.

Outputs (per cell):
    REPT_<id>_sei_compare.png
    REPT_<id>_solvdiff_sim_2000cy.parquet  (already exists from rept_long_sim.py)
    REPT_<id>_rxnlim_sim_2000cy.parquet
    REPT_<id>_rxnlim_calibrated.json
Plus a combined 2x2 grid: REPT_sei_model_compare_grid.png
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
CELLS = [78]
PROTOCOL = CyclingProtocol(c_rate=0.25)
TEMP_K = 298.15
N_CYCLES_CALIBRATION = 10
N_CYCLES_LONG = 2000
CACHE_DIR = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/pybamm_cache")
OUT_DIR = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/tuned_params")
LONGTERM_DIR = Path("/home/hj/Desktop/PINNs/Data/Longterm")
SKIP_FIRST_N = 4

# PyBaMM DFN options for each SEI sub-model. Both isolate SEI from plating
# and LAM so the calibration sees only the SEI knob.
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
    """Linear fit slope, skipping the first cycle (warm-up)."""
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


def measured_per_cycle(cid: int, nominal: float) -> pd.DataFrame:
    csv = LONGTERM_DIR / f"REPT_Longterm_cell_{cid:04d}.csv"
    raw = pd.read_csv(csv, usecols=["cycle_no", "step_name", "capacity_ah"])
    dchg = raw[raw.step_name.astype(str).str.contains("DChg")]
    out = (dchg.groupby("cycle_no").capacity_ah
                .agg(lambda s: float(s.abs().max())).reset_index())
    out["soh"] = out.capacity_ah / nominal
    return out.sort_values("cycle_no").reset_index(drop=True)


def calibrate_sei(char, target_slope: float, dfn_opts: dict,
                   param_key: str, log10_bracket: tuple[float, float],
                   rtol: float = 0.20, n_cycles: int = N_CYCLES_CALIBRATION,
                   max_iter: int = 12, pre_age_to_soh=None) -> dict:
    """Generic bisection: find log10(param) such that the slope of a short
    sim matches `target_slope`. Works for either SEI sub-model — the caller
    provides the PyBaMM key + DFN options."""
    n_fresh = [0]

    def slope_for(log10_val: float) -> float:
        params = build_pybamm_parameters(
            char, base="Prada2013", temperature_K=TEMP_K,
            extra_overrides={param_key: 10 ** log10_val},
            pre_age_to_soh=pre_age_to_soh,
        )
        sim = Simulation(params, protocol=PROTOCOL, cache_dir=CACHE_DIR,
                          dfn_options=dfn_opts)
        df = sim.run(n_cycles=n_cycles)
        if not getattr(sim, "last_was_cached", True):
            n_fresh[0] += 1
        cyc = df["cycle_n"].to_numpy(dtype=float)
        soh = df["SOH"].to_numpy(dtype=float) * 100.0
        return _slope_pp_per_100cy(cyc, soh)

    lo, hi = log10_bracket
    slope_lo, slope_hi = slope_for(lo), slope_for(hi)
    n_evals = 2

    # If target is outside the bracket, return the nearest endpoint
    if (slope_lo - target_slope) * (slope_hi - target_slope) > 0:
        if abs(slope_lo - target_slope) < abs(slope_hi - target_slope):
            best_log10, best_slope = lo, slope_lo
        else:
            best_log10, best_slope = hi, slope_hi
        return {"log10_value": best_log10, "fitted_value": 10 ** best_log10,
                "achieved_slope": best_slope, "n_evals": n_evals,
                "n_fresh_sims": n_fresh[0],
                "note": "bracket-edge — target outside bracket"}

    mid = 0.5 * (lo + hi)
    for _ in range(max_iter):
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


def run_long_sim(char, param_key: str, fitted_value: float,
                  dfn_opts: dict, pre_age_to_soh, n_cycles: int) -> pd.DataFrame:
    params = build_pybamm_parameters(
        char, base="Prada2013", temperature_K=TEMP_K,
        extra_overrides={param_key: fitted_value},
        pre_age_to_soh=pre_age_to_soh,
    )
    sim = Simulation(params, protocol=PROTOCOL, cache_dir=CACHE_DIR,
                      dfn_options=dfn_opts)
    return sim.run(n_cycles=n_cycles)


# ─────────────────────────── per-cell experiment ───────────────────────────
def run_cell(cid: int) -> dict:
    cell_tag = f"REPT_{cid}"
    print(f"\n=== {cell_tag} — SEI model comparison ===")
    t0 = time.time()

    # 1) Load the existing solvent-diffusion calibration for the target slope
    prod_json = json.loads((OUT_DIR / f"{cell_tag}_aging_calibrated.json").read_text())
    target_slope = float(prod_json["target_slope_pp_per_100cy"])
    pre_age_to_soh = float(prod_json.get("pre_age_to_soh", 1.0))
    print(f"  target slope (workbook b1→b2): {target_slope:+.4f} pp/100cy")
    print(f"  pre_age_to_soh:                {pre_age_to_soh:.3f}")

    # 2) Load char_b1 once
    char = load_characterization(manufacturer="REPT", cell_id=str(cid), batch=1)

    # 3) Reaction-limited SEI calibration
    print(f"  calibrating reaction-limited SEI (k_SEI)…")
    t_cal = time.time()
    rxn = calibrate_sei(
        char, target_slope,
        dfn_opts=RXNLIM_OPTS,
        param_key="SEI reaction exchange current density [A.m-2]",
        log10_bracket=(-10.0, -6.0),
        pre_age_to_soh=pre_age_to_soh,
    )
    print(f"    k_SEI={rxn['fitted_value']:.3e} m/s, "
          f"achieved slope={rxn['achieved_slope']:+.4f} pp/100cy, "
          f"{rxn['n_fresh_sims']}/{rxn['n_evals']} sims, {time.time()-t_cal:.1f}s")

    # 4) Long sim with reaction-limited SEI
    print(f"  running {N_CYCLES_LONG}-cycle sim (reaction-limited)…")
    t_sim = time.time()
    rxn_sim = run_long_sim(char, "SEI reaction exchange current density [A.m-2]",
                            rxn["fitted_value"], RXNLIM_OPTS, pre_age_to_soh,
                            N_CYCLES_LONG)
    rxn_parq = OUT_DIR / f"{cell_tag}_rxnlim_sim_{N_CYCLES_LONG}cy.parquet"
    rxn_sim.to_parquet(rxn_parq)
    print(f"    wrote {rxn_parq.name}, {time.time()-t_sim:.1f}s")

    # 5) Save the reaction-limited calibration result
    rxn_payload = {
        "cell": cell_tag,
        "model": "reaction limited",
        "target_slope_pp_per_100cy": target_slope,
        "calibrated_param": "SEI reaction exchange current density [A.m-2]",
        "calibrated_value": rxn["fitted_value"],
        "log10_value": rxn["log10_value"],
        "achieved_slope_pp_per_100cy": rxn["achieved_slope"],
        "residual_pp_per_100cy": rxn["achieved_slope"] - target_slope,
        "relative_error_pct": abs(rxn["achieved_slope"] - target_slope) /
                              max(abs(target_slope), 1e-9) * 100.0,
        "n_evaluations": rxn["n_evals"],
        "n_fresh_sims": rxn["n_fresh_sims"],
        "note": rxn["note"],
        "pre_age_to_soh": pre_age_to_soh,
        "dfn_options": {k: str(v) for k, v in RXNLIM_OPTS.items()},
    }
    (OUT_DIR / f"{cell_tag}_rxnlim_calibrated.json").write_text(
        json.dumps(rxn_payload, indent=2, default=str))

    return {
        "cell": cell_tag,
        "target_slope": target_slope,
        "rxn_k_SEI": rxn["fitted_value"],
        "rxn_achieved_slope": rxn["achieved_slope"],
        "wall_time_s": time.time() - t0,
    }


# ─────────────────────────── plotting ───────────────────────────
def plot_cell_comparison(cid: int) -> Path:
    """Single-cell overlay: measured + solvent-diffusion sim + reaction-limited sim."""
    cell_tag = f"REPT_{cid}"

    prod_json = json.loads((OUT_DIR / f"{cell_tag}_aging_calibrated.json").read_text())
    char = load_characterization(manufacturer="REPT", cell_id=str(cid), batch=1)
    nominal = char.nominal_capacity_ah

    solvdiff_sim = pd.read_parquet(OUT_DIR / f"{cell_tag}_long_sim_{N_CYCLES_LONG}cy.parquet")
    rxnlim_sim   = pd.read_parquet(OUT_DIR / f"{cell_tag}_rxnlim_sim_{N_CYCLES_LONG}cy.parquet")
    meas = measured_per_cycle(cid, nominal)
    meas["soh_pct"] = meas.soh * 100.0
    m = meas[meas.cycle_no >= SKIP_FIRST_N + 1].copy()
    keep = _drop_outliers(m["soh_pct"], k=3.0, window=5)
    clean = m[keep]
    first_cy = int(clean.cycle_no.iloc[0])
    meas_start = float(clean["soh_pct"].iloc[0])

    def _anchor(sim_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        soh = sim_df["SOH"].values * 100.0
        cyc = sim_df["cycle_n"].values.astype(float)
        soh_anc = soh + (meas_start - soh[0])
        cyc_anc = cyc + (first_cy - cyc[0])
        return cyc_anc, soh_anc

    sd_cy, sd_soh = _anchor(solvdiff_sim)
    rx_cy, rx_soh = _anchor(rxnlim_sim)

    rxn_json = json.loads((OUT_DIR / f"{cell_tag}_rxnlim_calibrated.json").read_text())

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Left panel: full 2000-cycle horizon
    for ax in axs:
        ax.plot(clean.cycle_no, clean["soh_pct"], "o-", lw=0.7, ms=2.5,
                 color="#d62728", label=f"measured (longterm CSV, ends cy {int(clean.cycle_no.max())})")
        ax.plot(sd_cy, sd_soh, "-", lw=1.2, color="#1f77b4",
                 label=f"sim — solvent-diffusion (D_SEI={prod_json['calibrated_value']:.1e})")
        ax.plot(rx_cy, rx_soh, "--", lw=1.2, color="#2ca02c",
                 label=f"sim — reaction-limited (k_SEI={rxn_json['calibrated_value']:.1e})")
        ax.axhline(80, ls=":", color="black", alpha=0.5, label="EOL = 80%")
        ax.set_xlabel("Cycle"); ax.set_ylabel("SoH (%)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    axs[0].set_title(f"{cell_tag} — full {N_CYCLES_LONG}-cycle horizon")
    axs[1].set_title(f"{cell_tag} — zoomed to measured window (≤ 200 cy)")
    axs[1].set_xlim(0, 200)
    # tight y-zoom for the right panel
    ywin = np.concatenate([clean["soh_pct"].values,
                            sd_soh[sd_cy <= 200], rx_soh[rx_cy <= 200]])
    pad = max(0.05, (ywin.max() - ywin.min()) * 0.1)
    axs[1].set_ylim(ywin.min() - pad, ywin.max() + pad)

    fig.suptitle(f"{cell_tag} — SEI model comparison "
                  f"(target slope = {prod_json['target_slope_pp_per_100cy']:+.3f} pp/100cy)",
                  fontsize=12, y=1.01)
    fig.tight_layout()
    out = OUT_DIR / f"{cell_tag}_sei_compare.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_combined_grid() -> Path:
    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    for col, cid in enumerate(CELLS):
        cell_tag = f"REPT_{cid}"
        prod_json = json.loads((OUT_DIR / f"{cell_tag}_aging_calibrated.json").read_text())
        rxn_json  = json.loads((OUT_DIR / f"{cell_tag}_rxnlim_calibrated.json").read_text())
        char = load_characterization(manufacturer="REPT", cell_id=str(cid), batch=1)
        nominal = char.nominal_capacity_ah

        solvdiff_sim = pd.read_parquet(OUT_DIR / f"{cell_tag}_long_sim_{N_CYCLES_LONG}cy.parquet")
        rxnlim_sim   = pd.read_parquet(OUT_DIR / f"{cell_tag}_rxnlim_sim_{N_CYCLES_LONG}cy.parquet")
        meas = measured_per_cycle(cid, nominal)
        meas["soh_pct"] = meas.soh * 100.0
        m = meas[meas.cycle_no >= SKIP_FIRST_N + 1].copy()
        keep = _drop_outliers(m["soh_pct"], k=3.0, window=5)
        clean = m[keep]
        first_cy = int(clean.cycle_no.iloc[0])
        meas_start = float(clean["soh_pct"].iloc[0])

        def _anchor(sim_df):
            soh = sim_df["SOH"].values * 100.0
            cyc = sim_df["cycle_n"].values.astype(float)
            return cyc + (first_cy - cyc[0]), soh + (meas_start - soh[0])

        sd_cy, sd_soh = _anchor(solvdiff_sim)
        rx_cy, rx_soh = _anchor(rxnlim_sim)

        for row, (title, xlim) in enumerate([
                (f"{cell_tag} — full {N_CYCLES_LONG} cy", None),
                (f"{cell_tag} — zoomed (≤ 200 cy)",       (0, 200)),
        ]):
            ax = axs[row, col]
            ax.plot(clean.cycle_no, clean["soh_pct"], "o-", lw=0.7, ms=2,
                     color="#d62728", label="measured CSV")
            ax.plot(sd_cy, sd_soh, "-", lw=1.2, color="#1f77b4",
                     label=f"solv-diff (D_SEI={prod_json['calibrated_value']:.1e})")
            ax.plot(rx_cy, rx_soh, "--", lw=1.2, color="#2ca02c",
                     label=f"rxn-lim (k_SEI={rxn_json['calibrated_value']:.1e})")
            ax.axhline(80, ls=":", color="black", alpha=0.5)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Cycle"); ax.set_ylabel("SoH (%)")
            ax.grid(alpha=0.3)
            ax.legend(loc="best", fontsize=8)
            if xlim is not None:
                ax.set_xlim(*xlim)
                ywin = np.concatenate([clean["soh_pct"].values,
                                        sd_soh[sd_cy <= xlim[1]],
                                        rx_soh[rx_cy <= xlim[1]]])
                pad = max(0.05, (ywin.max() - ywin.min()) * 0.1)
                ax.set_ylim(ywin.min() - pad, ywin.max() + pad)

    fig.suptitle(f"REPT — SEI sub-model comparison "
                  f"(solvent-diffusion vs reaction-limited, {N_CYCLES_LONG} cy)",
                  fontsize=13, y=1.005)
    fig.tight_layout()
    out = OUT_DIR / "REPT_sei_model_compare_grid.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for cid in CELLS:
        try:
            results.append(run_cell(cid))
        except Exception as e:
            print(f"  FAIL REPT_{cid}: {type(e).__name__}: {e}")

    # Per-cell plots
    for cid in CELLS:
        p = plot_cell_comparison(cid)
        print(f"  per-cell plot: {p.name}")

    # Combined grid only useful when there's more than one cell
    if len(CELLS) > 1:
        grid = plot_combined_grid()
        print(f"\nCombined grid: {grid}")

    print("\n=== Summary ===")
    for r in results:
        print(f"  {r['cell']:<8} "
              f"target={r['target_slope']:+.4f} pp/100cy  "
              f"k_SEI={r['rxn_k_SEI']:.3e}  "
              f"achieved={r['rxn_achieved_slope']:+.4f} pp/100cy  "
              f"wall={r['wall_time_s']:.1f}s")


if __name__ == "__main__":
    main()
