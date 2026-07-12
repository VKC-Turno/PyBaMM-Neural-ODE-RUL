"""EVE cohort sweep — runs the AGENT_VOLTARIS_TUNING workflow on all 8 EVE
cells deterministically in one foreground process.

Why not the subagent fan-out? The agent layer adds variance (gate-firing
order, prompt drift, retry behaviour) on top of a workflow that is
fundamentally deterministic. Each cell's run takes ~5 s of pure pybamm_tuning
work — fan-out was overkill, and a subagent-orchestrator process exiting
mid-flight wiped the wave-1 results.

Per-cell pipeline (matches AGENT_VOLTARIS_TUNING.md, Tasks 1-7):
    1. Load char → quick sanity gates
    2. Stoichiometry fit from OCV
    3. Build pybamm parameters (saves `<cell>_pybamm_params.json`)
    4. Compute per-cycle SoH + target fade slope from the longterm CSV
    5. Calibrate SEI diffusivity
    6. Validate over N cycles → metrics + parquet
    7. Write `_aging_calibrated.json` + `_calibration_report.md` + `_sim_vs_measured.png`
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/hj/Desktop/PINNs")
from pybamm_tuning import (
    apply_r0_to_contact_resistance, build_pybamm_parameters,
    calibrate_sei_diffusivity, fit_stoichiometry_from_ocv,
    load_characterization, SEI_ONLY_DFN_OPTIONS,
)
from pybamm_tuning.simulation import CyclingProtocol, Simulation


PROTOCOL = CyclingProtocol(c_rate=0.25)
TEMP_K = 298.15
N_CYCLES_CALIBRATION = 10
N_CYCLES_VALIDATION = 20
CACHE_DIR = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/pybamm_cache")
OUT_DIR = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/tuned_params")
LONGTERM_DIR = Path("/home/hj/Desktop/PINNs/Data/Longterm")

# Per-cell override for the cycle window used to compute the target fade
# slope. Default is the whole test; specify (min_cycle, max_cycle) when a
# cell has a non-monotonic SoH trajectory (e.g. EVE_1's mid-test
# discontinuity at cycles 80–90 — see EVE_1_diagnosis.md).
PER_CELL_SLOPE_WINDOW: dict[int, tuple[int, int]] = {
    1: (100, 150),   # late-window only — skips formation + cycle-80 discontinuity
}


# ──────────────────────── per-cycle SoH from raw CSV ────────────────────────

def per_cycle_soh(longterm_csv: Path, nominal_ah: float) -> pd.DataFrame:
    """Read the raw cycler longterm CSV → per-cycle (dchg_cap_ah, soh).

    `capacity_ah` is signed (− on discharge); use abs().max() per step.
    Sums per cycle in case the protocol has multiple CC_DChg sub-steps.
    """
    df = pd.read_csv(longterm_csv,
                      usecols=["cycle_no", "step_name", "capacity_ah"])
    dchg = df[df["step_name"].astype(str).str.contains("DChg")]
    if dchg.empty:
        return pd.DataFrame(columns=["cycle_no", "dchg_cap_ah", "soh"])

    # Per (cycle_no), take the absolute peak of capacity_ah — handles signed
    # cycler exports + multi-step discharges (max is the full-discharge value).
    out = (dchg.groupby("cycle_no")["capacity_ah"]
               .agg(lambda s: float(s.abs().max()))
               .reset_index()
               .rename(columns={"capacity_ah": "dchg_cap_ah"}))
    out["soh"] = out["dchg_cap_ah"] / nominal_ah
    return out.sort_values("cycle_no").reset_index(drop=True)


# Map dod label → fraction of nominal cell capacity actually discharged per
# cycle. The cycler's `dod` field is "min_max" SoC bounds; the discharged
# fraction is (max - min) / 100. Used to scale measured cycle-1 capacity
# before comparing it to the workbook's q_rpt.
def _dod_window_fraction(longterm_csv: Path) -> float:
    df = pd.read_csv(longterm_csv, usecols=["dod"], nrows=10_000)
    val = df["dod"].dropna().iloc[0] if df["dod"].notna().any() else None
    if val is None or "_" not in str(val):
        return 1.0
    lo, hi = (int(x) for x in str(val).split("_"))
    return max(0.0, (hi - lo) / 100.0)


def fade_slope_pp_per_100cy(per_cycle: pd.DataFrame,
                              cycle_min: int = 1,
                              cycle_max: int | None = None) -> float:
    """Linear regression: SoH (pp) vs cycle → slope in pp/100cy. Negative for fade."""
    sub = per_cycle[per_cycle["cycle_no"] >= cycle_min]
    if cycle_max is not None:
        sub = sub[sub["cycle_no"] <= cycle_max]
    if len(sub) < 5:
        return float("nan")
    x = sub["cycle_no"].astype(float).values
    y = sub["soh"].astype(float).values * 100.0
    slope, _ = np.polyfit(x, y, 1)
    return float(slope * 100.0)


# ──────────────────────── per-cell workflow ────────────────────────

def run_cell(cell_id: int, batch: int = 1) -> dict:
    cell_tag = f"EVE_{cell_id}"
    print(f"\n=== {cell_tag} (batch {batch}) ===")
    t0 = time.time()

    # 1) Load char
    char = load_characterization(manufacturer="EVE", cell_id=str(cell_id),
                                  batch=batch)
    print(f"  char: q_rpt={char.q_rpt_ah:.2f} Ah, soh={char.soh_pct:.2f} %")

    # 2) OCV stoichiometry
    fit = fit_stoichiometry_from_ocv(char.ocv_soc_grid, char.ocv_v_curve)
    print(f"  ocv:  RMSE={fit.rmse_mV:.2f} mV, top_V={fit.ocv_top_v:.3f} V "
          f"(outside_band={fit.ocv_top_outside_lfp_band})")

    # 3) Gates
    gates = {
        "LOW_OCV_QUALITY":          fit.rmse_mV > 15,
        "OCV_TOP_OUTSIDE_LFP_BAND": fit.ocv_top_outside_lfp_band,
        "NO_DCIR":                  char.dcir_r0_mohm.size == 0,
        "SHORT_LONGTERM":           False,    # filled later
    }

    # 4) Longterm CSV → per-cycle SoH + target slope
    longterm_csv = LONGTERM_DIR / f"EVE_Longterm_cell_{cell_id:04d}.csv"
    if not longterm_csv.exists():
        raise FileNotFoundError(longterm_csv)
    per_cycle = per_cycle_soh(longterm_csv, nominal_ah=char.nominal_capacity_ah)
    n_cyc = int(per_cycle["cycle_no"].max()) if not per_cycle.empty else 0
    gates["SHORT_LONGTERM"] = n_cyc < 50
    window = PER_CELL_SLOPE_WINDOW.get(cell_id)
    if window is not None:
        target_slope = fade_slope_pp_per_100cy(per_cycle, *window)
        slope_note = f"cycles {window[0]}-{window[1]} only (per-cell override)"
    else:
        target_slope = fade_slope_pp_per_100cy(per_cycle)
        slope_note = "whole test"
    gates["LOW_SOH_SIGNAL"] = abs(target_slope) < 0.05
    # SEI model can only produce *negative* slopes (degradation). A positive
    # target means the SoH proxy is non-monotonic. Skip calibration outright.
    gates["INVERTED_SLOPE"] = target_slope > 0
    print(f"  long: {n_cyc} cycles, target slope = {target_slope:.4f} pp/100cy "
          f"({slope_note})")

    per_cycle.to_csv(OUT_DIR / f"{cell_tag}_longterm_per_cycle.csv", index=False)

    # 5) Pre-aging factor + workbook-vs-measured cross-check.
    # Use workbook q_rpt as the trusted SoH anchor (RPT-measured outside the
    # longterm protocol, immune to formation artifacts). Cross-check against
    # the measured peak dchg in cycles 1-20 (skips formation rise), scaling
    # by the DoD window fraction so partial-DoD cells aren't false-flagged.
    workbook_soh = float(char.q_rpt_ah) / float(char.nominal_capacity_ah)
    window_frac = _dod_window_fraction(longterm_csv)
    early = per_cycle[per_cycle["cycle_no"] <= 20]
    measured_peak_ah = float(early["dchg_cap_ah"].max()) if not early.empty else float("nan")
    # Up-scale measured peak by the DoD window so it's comparable to q_rpt
    measured_full_ah = measured_peak_ah / max(window_frac, 1e-6)
    measured_soh = measured_full_ah / char.nominal_capacity_ah
    disagree_pp = abs(workbook_soh - measured_soh) * 100.0
    gates["WORKBOOK_VS_MEASURED_DISAGREE"] = disagree_pp > 5.0
    pre_age_soh = workbook_soh   # use workbook as the anchor
    print(f"  pre-age: workbook SoH={workbook_soh:.3f}, "
          f"measured SoH={measured_soh:.3f} (dod_frac={window_frac:.2f}), "
          f"disagree={disagree_pp:.1f} pp")

    prefer_r0 = "hppc" if gates["NO_DCIR"] else "dcir"
    base_overrides = build_pybamm_parameters(char, base="Prada2013",
                                              temperature_K=TEMP_K,
                                              fit_stoichiometry=True,
                                              pre_age_to_soh=pre_age_soh)
    pybamm_params_path = OUT_DIR / f"{cell_tag}_pybamm_params.json"
    pybamm_params_path.write_text(json.dumps({
        "cell": cell_tag,
        "cohort": "EVE",
        "batch": batch,
        "soh_pct": float(char.soh_pct),
        "q_rpt_ah": float(char.q_rpt_ah),
        "nominal_capacity_ah": float(char.nominal_capacity_ah),
        "stoichiometry_fit": {
            "x_100": fit.x_100, "x_0": fit.x_0,
            "y_100": fit.y_100, "y_0": fit.y_0,
            "ocv_rmse_mV": fit.rmse_mV,
            "ocv_top_v": fit.ocv_top_v,
            "n_anchors": fit.n_anchors,
        },
        "data_quality_flags": {k: bool(v) for k, v in gates.items()},
        "r0_anchor_source": prefer_r0,
        "r0_at_50pct_mohm": char.r0_at_soc(0.5, prefer=prefer_r0),
        "pybamm_overrides": {k: (v if not callable(v) else f"<callable {k}>")
                              for k, v in base_overrides.items()},
    }, indent=2, default=str))

    # 6) Calibration — short-circuit if INVERTED_SLOPE; the SEI calibrator
    # would just bottom out at the bracket edge with rel_err ≈ 100 %.
    if gates["INVERTED_SLOPE"]:
        print(f"  cal:  SKIPPED (INVERTED_SLOPE: target {target_slope:+.3f} pp/100cy)")
        classification = "POOR"
        rel_err = float("nan")
        # Build a dummy result so the rest of the JSON write goes through
        from pybamm_tuning.calibration import CalibrationResult
        cal = CalibrationResult(
            parameter_name="SEI solvent diffusivity [m2.s-1]",
            fitted_value=float("nan"),
            achieved_slope_pp_per_100cy=float("nan"),
            target_slope_pp_per_100cy=target_slope,
            residual_pp_per_100cy=float("nan"),
            n_evaluations=0, log10_bracket_used=(-24.0, -19.0), n_fresh_sims=0,
        )
    else:
        cal = calibrate_sei_diffusivity(
            char, target_slope_pp_per_100cy=target_slope,
            protocol=PROTOCOL, temperature_K=TEMP_K,
            n_cycles=N_CYCLES_CALIBRATION,
            log10_bracket=(-24.0, -19.0), rtol=0.20,
            cache_dir=CACHE_DIR,
            pre_age_to_soh=pre_age_soh,
        )
        rel_err = abs(cal.residual_pp_per_100cy / cal.target_slope_pp_per_100cy) * 100.0
        classification = ("GOOD" if rel_err <= 25 else
                           "FAIR" if rel_err <= 50 else "POOR")
        print(f"  cal:  D_SEI={cal.fitted_value:.3e} m²/s, "
              f"rel_err={rel_err:.1f} % → {classification}  "
              f"(fresh={cal.n_fresh_sims}/{cal.n_evaluations})")

    # 7) Validation. If a slope window was used, validate against that same
    # window — otherwise sim starts at SoH=100 % while measured at cycle N
    # in the window can be at any SoH, and the absolute mid-/end-errors
    # are nonsensical. When window is set, we anchor sim to the measured
    # SoH at window-start and compare slopes + anchored deltas.
    val_start = window[0] if window is not None else 1
    val_end   = val_start + N_CYCLES_VALIDATION - 1
    meas = per_cycle[per_cycle["cycle_no"].between(val_start, val_end)].copy()
    meas["soh_pct"] = meas["soh"] * 100.0
    if not gates["INVERTED_SLOPE"]:
        val_params = build_pybamm_parameters(
            char, base="Prada2013", temperature_K=TEMP_K,
            extra_overrides={"SEI solvent diffusivity [m2.s-1]": cal.fitted_value},
            pre_age_to_soh=pre_age_soh,
        )
        sim = Simulation(val_params, protocol=PROTOCOL, cache_dir=CACHE_DIR,
                          dfn_options=SEI_ONLY_DFN_OPTIONS)
        val_df = sim.run(n_cycles=N_CYCLES_VALIDATION)
        val_df.to_parquet(OUT_DIR / f"{cell_tag}_calibrated_sim_{N_CYCLES_VALIDATION}cy.parquet")
        sim_soh_pct = val_df["SOH"].values * 100.0
        sim_cyc = val_df["cycle_n"].values.astype(float)
        sim_slope = float(np.polyfit(sim_cyc[1:], sim_soh_pct[1:], 1)[0] * 100)
        slope_mae = abs(sim_slope - target_slope)
        # Anchor sim trajectory to measured SoH at window-start so the
        # absolute mid/end errors reflect deviation FROM THE CALIBRATED
        # FADE PATTERN, not the (irrelevant) absolute SoH offset.
        meas_soh_at_start = float(meas["soh_pct"].iloc[0]) if not meas.empty else 100.0
        sim_offset = meas_soh_at_start - sim_soh_pct[0]
        sim_anchored = sim_soh_pct + sim_offset
        sim_cyc_anchored = sim_cyc + (val_start - sim_cyc[0])
        mid_cy = val_start + N_CYCLES_VALIDATION // 2
        mid_err = float(np.interp(mid_cy, sim_cyc_anchored, sim_anchored)
                         - np.interp(mid_cy, meas["cycle_no"], meas["soh_pct"]))
        end_err = float(np.interp(val_end, sim_cyc_anchored, sim_anchored)
                         - np.interp(val_end, meas["cycle_no"], meas["soh_pct"]))
    else:
        sim_cyc, sim_soh_pct = np.array([]), np.array([])
        sim_slope = slope_mae = mid_err = end_err = float("nan")
        mid_cy = N_CYCLES_VALIDATION // 2

    # 8) Plot overlay (use anchored sim when slope window was overridden)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(meas["cycle_no"], meas["soh_pct"], "o-", lw=1.2,
             label=f"measured (cycles {val_start}-{val_end})", color="#d62728")
    if sim_cyc.size:
        sim_x = sim_cyc_anchored if window is not None else sim_cyc
        sim_y = sim_anchored      if window is not None else sim_soh_pct
        anchor_note = f" — anchored to cycle {val_start}" if window is not None else ""
        ax.plot(sim_x, sim_y, "s--", lw=1.2,
                 label=f"simulated (D_SEI={cal.fitted_value:.2e}){anchor_note}",
                 color="#1f77b4")
    else:
        ax.text(0.5, 0.5, "calibration skipped\n(INVERTED_SLOPE)",
                 transform=ax.transAxes, ha="center", va="center",
                 fontsize=12, color="grey")
    ax.axhline(80, ls=":", color="grey", alpha=0.6)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("SoH (%)")
    ax.set_title(f"{cell_tag} batch {batch} — calibration overlay ({N_CYCLES_VALIDATION} cy)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{cell_tag}_sim_vs_measured.png", dpi=120)
    plt.close(fig)

    # 9) Aging-calibrated JSON
    aging_payload = {
        "cell": cell_tag,
        "cohort": "EVE",
        "batch": batch,
        "soh_pct": float(char.soh_pct),
        "pre_age_to_soh": float(pre_age_soh),
        "workbook_soh": float(workbook_soh),
        "measured_soh_dod_corrected": float(measured_soh),
        "dod_window_fraction": float(window_frac),
        "workbook_vs_measured_disagree_pp": float(disagree_pp),
        "slope_window": list(window) if window is not None else None,
        "slope_window_note": slope_note,
        "measured_target_pp_per_100cy": target_slope,
        "achieved_pp_per_100cy": cal.achieved_slope_pp_per_100cy,
        "residual_pp_per_100cy": cal.residual_pp_per_100cy,
        "relative_error_pct": rel_err,
        "calibrated_param": cal.parameter_name,
        "calibrated_value": cal.fitted_value,
        "log10_bracket_used": list(cal.log10_bracket_used),
        "n_evaluations": cal.n_evaluations,
        "n_fresh_sims": cal.n_fresh_sims,
        "classification": classification,
        "dfn_options": {str(k): str(v) for k, v in SEI_ONLY_DFN_OPTIONS.items()},
        "protocol": {"c_rate": PROTOCOL.c_rate,
                      "discharge_cut_V": getattr(PROTOCOL, "v_min", 2.5),
                      "charge_cut_V":    getattr(PROTOCOL, "v_max", 3.65)},
        "n_cycles_calibration": N_CYCLES_CALIBRATION,
        "n_cycles_validation":  N_CYCLES_VALIDATION,
        "validation": {
            "sim_slope_pp_per_100cy": sim_slope,
            "slope_mae_pp_per_100cy": slope_mae,
            "mid_life_error_pp":      mid_err,
            "end_of_window_error_pp": end_err,
        },
        "gate_audit": {k: {"tripped": bool(v)} for k, v in gates.items()},
        "fallback_strategies_invoked": [],
    }
    (OUT_DIR / f"{cell_tag}_aging_calibrated.json").write_text(
        json.dumps(aging_payload, indent=2, default=str))

    # 10) Markdown report
    window_caveat = ""
    if window is not None:
        window_caveat = (f"\n> **Slope window override**: target slope computed from "
                         f"cycles {window[0]}–{window[1]} only (whole-test slope is "
                         f"misleading — see `{cell_tag}_diagnosis.md` if present).\n")

    md = f"""# {cell_tag} (batch {batch}) — Voltaris parameter-tuning report

## TL;DR
**Classification: {classification}** — SEI solvent diffusivity calibrated to `{cal.fitted_value:.3e} m²/s`
with **relative error {rel_err:.2f} %** vs target slope `{target_slope:.4f} pp/100cy`.
Validation over {N_CYCLES_VALIDATION} cycles: slope MAE `{slope_mae:.4f} pp/100cy`,
mid-life error `{mid_err:.2f} pp`, end-of-window error `{end_err:.2f} pp`.
{window_caveat}

## Gates
| Gate | Tripped? |
|---|:---:|
""" + "\n".join(f"| `{g}` | {'✓' if v else '✗'} |" for g, v in gates.items()) + f"""

## OCV fit
- Anchors: {fit.n_anchors}
- Top V: **{fit.ocv_top_v:.3f}**
- RMSE: **{fit.rmse_mV:.2f} mV**
- Stoichiometric limits: x_100={fit.x_100:.4f}, x_0={fit.x_0:.4f}, y_100={fit.y_100:.4f}, y_0={fit.y_0:.4f}

## Calibration
- Parameter: `{cal.parameter_name}`
- Calibrated value: `{cal.fitted_value:.3e}`
- log10 bracket used: {cal.log10_bracket_used}
- n_evaluations: {cal.n_evaluations} (fresh PyBaMM solves: **{cal.n_fresh_sims}**)
- Target slope: `{target_slope:.4f} pp/100cy`
- Achieved slope: `{cal.achieved_slope_pp_per_100cy:.4f} pp/100cy`
- Residual: `{cal.residual_pp_per_100cy:.4f} pp/100cy`

## Validation ({N_CYCLES_VALIDATION} cycles)
- Sim slope: `{sim_slope:.4f} pp/100cy`
- Slope MAE: `{slope_mae:.4f} pp/100cy`
- Mid-life (cyc {mid_cy}) error: `{mid_err:.2f} pp`
- End-of-window (cyc {N_CYCLES_VALIDATION}) error: `{end_err:.2f} pp`

## Wall-time
- Total: **{time.time() - t0:.1f} s**
"""
    (OUT_DIR / f"{cell_tag}_calibration_report.md").write_text(md)

    return {
        "cell": cell_tag, "classification": classification,
        "D_SEI": cal.fitted_value, "rel_err": rel_err,
        "n_fresh_sims": cal.n_fresh_sims,
        "wall_time_s": time.time() - t0,
        "gates_tripped": [g for g, v in gates.items() if v],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for cell_id in range(1, 9):
        try:
            results.append(run_cell(cell_id))
        except Exception as e:
            print(f"  FAIL EVE_{cell_id}: {type(e).__name__}: {e}")
            results.append({"cell": f"EVE_{cell_id}", "error": str(e)})

    print("\n=== Sweep summary ===")
    for r in results:
        if "error" in r:
            print(f"  {r['cell']:<8} ERROR: {r['error']}")
        else:
            print(f"  {r['cell']:<8} {r['classification']:<5}  "
                  f"D_SEI={r['D_SEI']:.2e}  err={r['rel_err']:.1f}%  "
                  f"fresh={r['n_fresh_sims']}  t={r['wall_time_s']:.1f}s "
                  f"gates={','.join(r['gates_tripped']) or '-'}")


if __name__ == "__main__":
    main()
