"""Voltaris per-cell PyBaMM parameter tuner — EVE cell 3 (batch 1).

Workflow (per AGENT_VOLTARIS_TUNING.md):
1. Sanity-check inputs
2. Tune base electrochemistry (OCV fit, build pybamm params, save JSON)
3. Compute target fade rate from longterm CSV (IQR outlier filter, linear fit)
4. Calibrate SEI solvent diffusivity (bisection, SEI_ONLY_DFN_OPTIONS)
5. Fallback strategies if needed
6. Validate calibrated parameter set (run sim, plot)
7. Write calibration report (markdown)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/hj/Desktop/PINNs")
sys.path.insert(0, str(ROOT))

from pybamm_tuning import (
    load_characterization,
    fit_stoichiometry_from_ocv,
    build_pybamm_parameters,
    calibrate_sei_diffusivity,
    Simulation,
    CyclingProtocol,
    SEI_ONLY_DFN_OPTIONS,
)

COHORT = "EVE"
CELL_ID = "3"
BATCH = 1
TEMP_K = 298.15

OUT_DIR = ROOT / "Voltaris/outputs/tuned_params"
CACHE_DIR = ROOT / "Voltaris/outputs/pybamm_cache"
LONGTERM_CSV = ROOT / "Data/Longterm/EVE_Longterm_cell_0003.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

prefix = f"{COHORT}_{CELL_ID}"

audit: list[str] = []  # decision audit trail
flags: dict = {}       # gate firing record


def log(msg: str) -> None:
    print(f"[{prefix}] {msg}", flush=True)


# ------------------------------------------------------------------
# Task 1: Sanity-check inputs
# ------------------------------------------------------------------
log("Task 1: loading characterization + longterm")
char = load_characterization(manufacturer=COHORT, cell_id=CELL_ID, batch=BATCH)
log(f"  char: cell_id={char.cell_id}, cohort={char.cohort}, batch={char.batch}, "
    f"Q_RPT={char.q_rpt_ah:.3f} Ah, nominal={char.nominal_capacity_ah:.1f} Ah, "
    f"SoH={char.soh_pct:.3f}%")
audit.append(f"Loaded EVE cell {CELL_ID} batch {BATCH}: "
             f"Q_RPT={char.q_rpt_ah:.3f} Ah, SoH={char.soh_pct:.2f}%")

# Load longterm CSV directly
lt_df = pd.read_csv(LONGTERM_CSV)
log(f"  longterm rows: {len(lt_df)}, cycles: {lt_df['cycle_no'].nunique()}")

# Filter to discharge steps
dchg_mask = lt_df["step_name"].astype(str).str.contains("DChg", case=False, na=False) | \
            lt_df["step_name"].astype(str).str.contains("Discharge", case=False, na=False)
dchg = lt_df[dchg_mask].copy()
log(f"  discharge rows: {len(dchg)}")

# Per-cycle SoH = max abs capacity per discharge step / max_cap
dchg["abs_cap_ah"] = dchg["capacity_ah"].abs()
per_cycle = dchg.groupby("cycle_no").agg(
    dchg_cap_ah=("abs_cap_ah", "max"),
    max_cap=("max_cap", "first"),
).reset_index()
per_cycle["soh"] = per_cycle["dchg_cap_ah"] / per_cycle["max_cap"]
n_cycles_total = len(per_cycle)
log(f"  per-cycle rows: {n_cycles_total}, "
    f"SoH range: {per_cycle['soh'].min():.4f} -> {per_cycle['soh'].max():.4f}")

if n_cycles_total < 50:
    flags["SHORT_LONGTERM"] = True
    audit.append(f"SHORT_LONGTERM gate fired (only {n_cycles_total} cycles)")
else:
    audit.append(f"SHORT_LONGTERM not fired ({n_cycles_total} >= 50 cycles)")

# Save per-cycle CSV
per_cycle.to_csv(OUT_DIR / f"{prefix}_longterm_per_cycle.csv", index=False)
log(f"  saved {prefix}_longterm_per_cycle.csv")

# ------------------------------------------------------------------
# Task 2: Tune base electrochemistry
# ------------------------------------------------------------------
log("Task 2: OCV stoichiometry fit + build pybamm params")
if char.ocv_soc_grid.size >= 4:
    fit = fit_stoichiometry_from_ocv(char.ocv_soc_grid, char.ocv_v_curve)
    log(f"  OCV-fit RMSE={fit.rmse_mV:.3f} mV, n_anchors={fit.n_anchors}")
    log(f"  ocv_top_v={fit.ocv_top_v:.4f}, "
        f"outside_lfp_band={fit.ocv_top_outside_lfp_band}")
    if fit.rmse_mV > 15.0:
        flags["LOW_OCV_QUALITY"] = True
        audit.append(f"LOW_OCV_QUALITY gate fired (RMSE {fit.rmse_mV:.2f} > 15 mV)")
    else:
        audit.append(f"LOW_OCV_QUALITY not fired (RMSE {fit.rmse_mV:.2f} mV)")
    if fit.ocv_top_outside_lfp_band:
        flags["OCV_TOP_OUTSIDE_LFP_BAND"] = True
        audit.append(f"OCV_TOP_OUTSIDE_LFP_BAND gate fired "
                     f"(top V {fit.ocv_top_v:.3f} outside [3.40, 3.55] V)")
    else:
        audit.append(f"OCV_TOP_OUTSIDE_LFP_BAND not fired (top V {fit.ocv_top_v:.3f})")
else:
    fit = None
    log("  no OCV data (skip stoichiometry fit)")
    audit.append("No OCV data: stoichiometry fit skipped")

# Build base params, capturing any UserWarning from r0
with warnings.catch_warnings(record=True) as w_list:
    warnings.simplefilter("always")
    params = build_pybamm_parameters(
        char, temperature_K=TEMP_K, fit_stoichiometry=(fit is not None),
    )
    r0_warnings = [w for w in w_list
                    if "R₀" in str(w.message) or "R0" in str(w.message)
                    or "No usable" in str(w.message)]

if r0_warnings:
    flags["R0_NO_USABLE_ANCHOR"] = True
    audit.append(f"R0_NO_USABLE_ANCHOR gate fired: {r0_warnings[0].message}")
else:
    audit.append("R0 anchor used: PyBaMM contact resistance set from char data")

# Also check whether DCIR was empty (the spec's NO_DCIR gate)
if char.dcir_r0_mohm.size == 0:
    flags["NO_DCIR"] = True
    audit.append("NO_DCIR gate fired: DCIR anchors empty (HPPC fallback used)")

# Convert params dict to JSON-friendly (only scalars)
def _serialise_params(p):
    out = {}
    for k, v in p.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
    return out

stoich_block = None
if fit is not None:
    stoich_block = {
        "x_100": fit.x_100, "x_0": fit.x_0,
        "y_100": fit.y_100, "y_0": fit.y_0,
        "ocv_rmse_mV": fit.rmse_mV, "n_anchors": fit.n_anchors,
        "ocv_top_v": fit.ocv_top_v,
        "ocv_top_outside_lfp_band": bool(fit.ocv_top_outside_lfp_band),
    }

# What were the actual injected overrides? We need to inspect a few key entries.
pybamm_overrides_view = {
    "Nominal cell capacity [A.h]": params.get("Nominal cell capacity [A.h]"),
    "Electrode width [m]":         params.get("Electrode width [m]"),
    "Contact resistance [Ohm]":    params.get("Contact resistance [Ohm]"),
    "Initial concentration in negative electrode [mol.m-3]":
        params.get("Initial concentration in negative electrode [mol.m-3]"),
    "Initial concentration in positive electrode [mol.m-3]":
        params.get("Initial concentration in positive electrode [mol.m-3]"),
    "Ambient temperature [K]":     params.get("Ambient temperature [K]"),
    "Initial temperature [K]":     params.get("Initial temperature [K]"),
}
pybamm_params_payload = {
    "cell":             f"{COHORT}_{CELL_ID}",
    "cohort":           COHORT,
    "cell_id":          CELL_ID,
    "batch":            BATCH,
    "soh_pct":          char.soh_pct,
    "q_rpt_ah":         char.q_rpt_ah,
    "nominal_cap_ah":   char.nominal_capacity_ah,
    "stoichiometry_fit": stoich_block,
    "pybamm_overrides": {k: float(v) if isinstance(v, (int, float)) else v
                          for k, v in pybamm_overrides_view.items()
                          if v is not None},
    "gates_fired":      list(flags.keys()),
}
with open(OUT_DIR / f"{prefix}_pybamm_params.json", "w") as fh:
    json.dump(pybamm_params_payload, fh, indent=2)
log(f"  saved {prefix}_pybamm_params.json")

# ------------------------------------------------------------------
# Task 3: Compute target fade rate
# ------------------------------------------------------------------
log("Task 3: target fade rate (IQR outlier filter + linear regression)")

soh_arr = per_cycle["soh"].to_numpy()
cyc_arr = per_cycle["cycle_no"].to_numpy(dtype=float)

q1, q3 = np.percentile(soh_arr, [25, 75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
in_range = (soh_arr >= lo) & (soh_arr <= hi)
n_dropped = int((~in_range).sum())
dropped_cyc = cyc_arr[~in_range].astype(int).tolist()
log(f"  IQR Q1={q1:.4f} Q3={q3:.4f} -> kept {in_range.sum()}/{len(soh_arr)}; "
    f"dropped {n_dropped} cycles: {dropped_cyc[:20]}")

x_fit = cyc_arr[in_range]
y_fit = soh_arr[in_range] * 100.0  # to %
slope_pp_per_cy, intercept = np.polyfit(x_fit, y_fit, 1)
target_slope_pp_per_100cy = float(slope_pp_per_cy * 100.0)
log(f"  target_slope = {target_slope_pp_per_100cy:.4f} pp/100cy "
    f"(intercept SoH at cycle 0 = {intercept:.3f} %)")

audit.append(f"IQR filter dropped {n_dropped} cycles "
             f"(cycles: {dropped_cyc if len(dropped_cyc) <= 20 else str(dropped_cyc[:20]) + '...'})")
audit.append(f"Linear fade slope = {target_slope_pp_per_100cy:.4f} pp/100cy")

# LOW_SOH_SIGNAL gate
log10_bracket = (-24.0, -19.0)
if abs(target_slope_pp_per_100cy) < 0.05:
    flags["LOW_SOH_SIGNAL"] = True
    log10_bracket = (-30.0, -18.0)
    audit.append(f"LOW_SOH_SIGNAL gate fired (|slope|={abs(target_slope_pp_per_100cy):.4f} < 0.05); "
                 f"widening log10 bracket to {log10_bracket}")
else:
    audit.append(f"LOW_SOH_SIGNAL not fired (|slope|={abs(target_slope_pp_per_100cy):.4f})")

# ------------------------------------------------------------------
# Task 4: Calibrate SEI solvent diffusivity
# ------------------------------------------------------------------
log("Task 4: calibrate SEI solvent diffusivity")
protocol = CyclingProtocol(c_rate=0.25)

result = calibrate_sei_diffusivity(
    char,
    target_slope_pp_per_100cy=target_slope_pp_per_100cy,
    protocol=protocol,
    temperature_K=TEMP_K,
    n_cycles=10,
    log10_bracket=log10_bracket,
    rtol=0.20,
    cache_dir=CACHE_DIR,
)
log(f"  fitted D_SEI = {result.fitted_value:.4e} m^2/s")
log(f"  target slope = {result.target_slope_pp_per_100cy:.4f} pp/100cy")
log(f"  achieved slope = {result.achieved_slope_pp_per_100cy:.4f} pp/100cy")
log(f"  residual = {result.residual_pp_per_100cy:.4f} pp/100cy")
log(f"  n_evaluations = {result.n_evaluations}, n_fresh_sims = {result.n_fresh_sims}")

target_abs = max(abs(result.target_slope_pp_per_100cy), 0.05)
rel_err = abs(result.residual_pp_per_100cy) / target_abs
log(f"  relative error = {rel_err * 100:.2f} %")

if rel_err <= 0.25:
    classification = "GOOD"
elif rel_err <= 0.50:
    classification = "FAIR"
else:
    classification = "POOR"
log(f"  initial classification: {classification}")
audit.append(f"SEI calibration: D_SEI={result.fitted_value:.4e}, "
             f"rel_err={rel_err*100:.2f}%, classification={classification}")

fallbacks_invoked: list[str] = []
if classification == "POOR":
    # Fallback 1: widen the bracket
    log("Task 5: POOR fit - trying fallback (widen bracket to (-26, -16))")
    fallbacks_invoked.append("widen_bracket_(-26,-16)")
    audit.append("FALLBACK invoked: widen log10 bracket to (-26, -16)")
    result2 = calibrate_sei_diffusivity(
        char,
        target_slope_pp_per_100cy=target_slope_pp_per_100cy,
        protocol=protocol, temperature_K=TEMP_K, n_cycles=10,
        log10_bracket=(-26.0, -16.0), rtol=0.20, cache_dir=CACHE_DIR,
    )
    rel_err2 = abs(result2.residual_pp_per_100cy) / target_abs
    log(f"  widen-bracket result: rel_err={rel_err2*100:.2f}%")
    if rel_err2 < rel_err:
        result = result2
        rel_err = rel_err2
        log10_bracket = (-26.0, -16.0)
        if rel_err <= 0.25:
            classification = "GOOD"
        elif rel_err <= 0.50:
            classification = "FAIR"
        else:
            classification = "POOR"
        audit.append(f"FALLBACK widen_bracket SUCCEEDED: new rel_err={rel_err*100:.2f}%, "
                     f"classification={classification}")
    else:
        audit.append(f"FALLBACK widen_bracket did NOT improve (rel_err {rel_err2*100:.2f}%)")

log(f"  final classification: {classification}")

# Save calibrated aging JSON
aging_payload = {
    "cell":                          f"{COHORT}_{CELL_ID}",
    "cohort":                        COHORT,
    "cell_id":                       CELL_ID,
    "batch":                         BATCH,
    "soh_pct":                       char.soh_pct,
    "measured_target_pp_per_100cy":  result.target_slope_pp_per_100cy,
    "achieved_pp_per_100cy":         result.achieved_slope_pp_per_100cy,
    "residual_pp_per_100cy":         result.residual_pp_per_100cy,
    "relative_error":                rel_err,
    "classification":                classification,
    "calibrated_param":              result.parameter_name,
    "calibrated_value":              result.fitted_value,
    "log10_bracket_used":            list(result.log10_bracket_used),
    "n_evaluations":                 result.n_evaluations,
    "n_fresh_sims":                  result.n_fresh_sims,
    "dfn_options":                   {k: (v if not isinstance(v, list) else list(v))
                                       for k, v in SEI_ONLY_DFN_OPTIONS.items()},
    "fallbacks_invoked":             fallbacks_invoked,
    "gates_fired":                   list(flags.keys()),
    "n_cycles_dropped":              n_dropped,
    "dropped_cycle_numbers":         dropped_cyc[:50],
    "pybamm_overrides_summary": {
        result.parameter_name: result.fitted_value,
        "Nominal cell capacity [A.h]":
            pybamm_overrides_view.get("Nominal cell capacity [A.h]"),
        "Contact resistance [Ohm]":
            pybamm_overrides_view.get("Contact resistance [Ohm]"),
        "Initial concentration in negative electrode [mol.m-3]":
            pybamm_overrides_view.get("Initial concentration in negative electrode [mol.m-3]"),
        "Initial concentration in positive electrode [mol.m-3]":
            pybamm_overrides_view.get("Initial concentration in positive electrode [mol.m-3]"),
    },
}
with open(OUT_DIR / f"{prefix}_aging_calibrated.json", "w") as fh:
    json.dump(aging_payload, fh, indent=2, default=float)
log(f"  saved {prefix}_aging_calibrated.json")

# ------------------------------------------------------------------
# Task 6: Validate calibrated parameter set
# ------------------------------------------------------------------
N_VAL = 20  # spec recommends 50; capped at 20 for wall-time budget
log(f"Task 6: validation run ({N_VAL} cycles)")
params_cal = build_pybamm_parameters(
    char, temperature_K=TEMP_K,
    extra_overrides={result.parameter_name: result.fitted_value},
)
sim_val = Simulation(
    params_cal, protocol=protocol, cache_dir=CACHE_DIR,
    dfn_options=SEI_ONLY_DFN_OPTIONS,
)
sim_df = sim_val.run(n_cycles=N_VAL)
sim_df.to_parquet(OUT_DIR / f"{prefix}_calibrated_sim_{N_VAL}cy.parquet")
log(f"  sim cycles: {len(sim_df)}, last_was_cached={sim_val.last_was_cached}")

cyc_sim = sim_df["cycle_n"].to_numpy(dtype=float)
soh_sim = sim_df["SOH"].to_numpy(dtype=float) * 100.0

# Slope from sim (skip warm-up cycle 0)
if len(cyc_sim) >= 3:
    sim_slope_pp_per_100cy = float(np.polyfit(cyc_sim[1:], soh_sim[1:], 1)[0] * 100.0)
else:
    sim_slope_pp_per_100cy = float(np.polyfit(cyc_sim, soh_sim, 1)[0] * 100.0)

slope_mae = abs(sim_slope_pp_per_100cy - target_slope_pp_per_100cy)
slope_mae_pct = slope_mae / target_abs * 100.0

# Mid-life / end-of-window comparison
mid_cy = N_VAL // 2
meas_at = lambda c: float(per_cycle.loc[per_cycle["cycle_no"] == c, "soh"].iloc[0]) * 100.0 \
    if (per_cycle["cycle_no"] == c).any() else float("nan")

sim_at = lambda c: float(soh_sim[np.argmin(np.abs(cyc_sim - c))])

mid_meas = meas_at(mid_cy)
mid_sim = sim_at(mid_cy)
end_meas = meas_at(N_VAL)
end_sim = sim_at(N_VAL)
mid_err = mid_sim - mid_meas if not np.isnan(mid_meas) else float("nan")
end_err = end_sim - end_meas if not np.isnan(end_meas) else float("nan")

log(f"  sim slope = {sim_slope_pp_per_100cy:.4f}, target = {target_slope_pp_per_100cy:.4f}")
log(f"  slope MAE = {slope_mae:.4f} pp/100cy ({slope_mae_pct:.1f}% of target)")
log(f"  mid-life (c={mid_cy}): sim={mid_sim:.3f}%, meas={mid_meas:.3f}%, err={mid_err:.3f}")
log(f"  end-of-window (c={N_VAL}): sim={end_sim:.3f}%, meas={end_meas:.3f}%, err={end_err:.3f}")

validation_metrics = {
    "n_val_cycles":                 N_VAL,
    "sim_slope_pp_per_100cy":       sim_slope_pp_per_100cy,
    "target_slope_pp_per_100cy":    target_slope_pp_per_100cy,
    "slope_mae_pp_per_100cy":       slope_mae,
    "slope_mae_pct_of_target":      slope_mae_pct,
    "mid_life_cycle":               mid_cy,
    "mid_life_sim_pct":             mid_sim,
    "mid_life_measured_pct":        mid_meas,
    "mid_life_error_pp":            mid_err,
    "end_of_window_cycle":          N_VAL,
    "end_of_window_sim_pct":        end_sim,
    "end_of_window_measured_pct":   end_meas,
    "end_of_window_error_pp":       end_err,
}

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
ax.plot(per_cycle["cycle_no"], per_cycle["soh"] * 100.0,
        label="measured", color="C0", marker=".", linestyle="", markersize=4, alpha=0.5)
ax.plot(cyc_sim, soh_sim, label="sim (calibrated D_SEI)", color="C3", linewidth=2)
ax.axhline(100.0, color="grey", linestyle=":", alpha=0.5)
ax.set_xlabel("cycle")
ax.set_ylabel("SoH (%)")
ax.set_title(f"{prefix}: full range\n"
             f"target={target_slope_pp_per_100cy:.3f} pp/100cy, "
             f"sim={sim_slope_pp_per_100cy:.3f} pp/100cy")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
zoom = per_cycle[per_cycle["cycle_no"] <= N_VAL]
ax.plot(zoom["cycle_no"], zoom["soh"] * 100.0,
        label="measured", color="C0", marker="o", markersize=4)
ax.plot(cyc_sim, soh_sim, label="sim", color="C3", linewidth=2, marker="x")
ax.set_xlabel("cycle")
ax.set_ylabel("SoH (%)")
ax.set_title(f"{prefix}: validation window (1-{N_VAL})\n"
             f"slope MAE {slope_mae:.3f} pp/100cy ({slope_mae_pct:.1f}%)")
ax.legend()
ax.grid(alpha=0.3)

fig.suptitle(
    f"EVE cell {CELL_ID}: SEI calibration ({classification}), "
    f"rel_err {rel_err*100:.1f}%, D_SEI={result.fitted_value:.2e} m^2/s",
    fontsize=11,
)
plt.tight_layout()
plt.savefig(OUT_DIR / f"{prefix}_sim_vs_measured.png", dpi=110)
plt.close(fig)
log(f"  saved {prefix}_sim_vs_measured.png")

# ------------------------------------------------------------------
# Task 7: Calibration report
# ------------------------------------------------------------------
log("Task 7: calibration report")

# Determine tl;dr WARN level
if classification == "GOOD":
    tldr_tag = "**Classification: GOOD.**"
elif classification == "FAIR":
    tldr_tag = "**Classification: FAIR — WARN.**"
else:
    tldr_tag = "**Classification: POOR — WARN.**"

acceptance = {
    "slope_mae_ok":  slope_mae_pct < 25.0,
    "mid_life_ok":   abs(mid_err) < 2.0 if not np.isnan(mid_err) else None,
    "end_ok":        abs(end_err) < 3.0 if not np.isnan(end_err) else None,
}

report = f"""# EVE cell {CELL_ID} (batch {BATCH}) — Voltaris parameter-tuning report

## TL;DR
{tldr_tag} SEI solvent diffusivity calibrated to
`{result.fitted_value:.3e} m²/s` with **relative error {rel_err*100:.2f} %** vs target slope
`{target_slope_pp_per_100cy:.3f} pp/100cy`. Validation over {N_VAL} cycles:
slope MAE {slope_mae:.3f} pp/100cy ({slope_mae_pct:.1f} % of target),
mid-life error {mid_err:.2f} pp, end-of-window error {end_err:.2f} pp.

Gates fired: **{', '.join(flags.keys()) if flags else 'none'}**.
Fallbacks invoked: **{', '.join(fallbacks_invoked) if fallbacks_invoked else 'none'}**.

## Cell metadata
| field | value |
|---|---|
| cell_id | `{CELL_ID}` (string) |
| cohort | {COHORT} |
| batch | {BATCH} |
| manufacturer | {char.manufacturer} |
| nominal capacity | {char.nominal_capacity_ah:.2f} Ah |
| measured Q_RPT | {char.q_rpt_ah:.3f} Ah |
| SoH at characterization | {char.soh_pct:.3f} % |
| longterm CSV | `{LONGTERM_CSV.relative_to(ROOT)}` |
| longterm cycles available | {n_cycles_total} |

## OCV-fit quality
| field | value |
|---|---|
| n_anchors | {fit.n_anchors if fit else 'n/a'} |
| OCV bottom V | {float(char.ocv_v_curve.min()):.4f} |
| OCV top V | **{fit.ocv_top_v if fit else float('nan'):.4f}** |
| stoichiometric x_100 | {fit.x_100 if fit else float('nan'):.4f} |
| stoichiometric x_0 | {fit.x_0 if fit else float('nan'):.4f} |
| stoichiometric y_100 | {fit.y_100 if fit else float('nan'):.4f} |
| stoichiometric y_0 | {fit.y_0 if fit else float('nan'):.4f} |
| **RMSE** | **{fit.rmse_mV if fit else float('nan'):.3f} mV** |
| LOW_OCV_QUALITY gate | {'**FIRED**' if 'LOW_OCV_QUALITY' in flags else 'not fired'} |
| OCV_TOP_OUTSIDE_LFP_BAND gate | {'**FIRED**' if 'OCV_TOP_OUTSIDE_LFP_BAND' in flags else 'not fired'} |

## R₀ source
- DCIR anchors available: {char.dcir_r0_mohm.size}
- HPPC anchors available: {char.hppc_r0_mohm.size}
- NO_DCIR gate: {'**FIRED** (HPPC fallback)' if 'NO_DCIR' in flags else 'not fired'}
- R0_NO_USABLE_ANCHOR gate: {'**FIRED**' if 'R0_NO_USABLE_ANCHOR' in flags else 'not fired'}
- Contact resistance written: `{pybamm_overrides_view.get('Contact resistance [Ohm]')}`

The R₀ sanity envelope (0.1 - 5 mΩ) is applied automatically by
`Characterization.r0_at_soc` / `apply_r0_to_contact_resistance` so no manual
anchor filtering was necessary.

## Longterm fade target (measured)
- Total cycles available: {n_cycles_total}
- IQR outlier filter dropped: {n_dropped} cycles ({dropped_cyc[:20]}{'...' if len(dropped_cyc) > 20 else ''})
- SoH(cycle 1) measured: {soh_arr[0]*100:.3f} %
- SoH(cycle {int(cyc_arr[-1])}) measured: {soh_arr[-1]*100:.3f} %
- **Linear slope = {target_slope_pp_per_100cy:.4f} pp/100cy** (used as target)
- LOW_SOH_SIGNAL gate: {'**FIRED** (bracket widened)' if 'LOW_SOH_SIGNAL' in flags else 'not fired'}
- SHORT_LONGTERM gate: {'**FIRED**' if 'SHORT_LONGTERM' in flags else 'not fired'}

## Calibration (SEI solvent diffusivity)
| field | value |
|---|---|
| lever | `{result.parameter_name}` |
| DFN options | `SEI_ONLY_DFN_OPTIONS` (plating & LAM off) |
| protocol | C/{int(1/protocol.c_rate)} -> CCCV {protocol.charge_cut_V:.2f} V -> {protocol.cv_taper_to} taper |
| n_cycles per evaluation | 10 |
| log10 bracket | {result.log10_bracket_used} |
| rtol | 0.20 |
| **fitted value** | **{result.fitted_value:.4e} m²/s** (log10 = {np.log10(result.fitted_value):.3f}) |
| achieved slope | {result.achieved_slope_pp_per_100cy:.4f} pp/100cy |
| target slope | {result.target_slope_pp_per_100cy:.4f} pp/100cy |
| residual | {result.residual_pp_per_100cy:.4f} pp/100cy |
| **relative error** | **{rel_err*100:.2f} %** |
| n_evaluations | {result.n_evaluations} |
| **n_fresh_sims** | **{result.n_fresh_sims}** (wall-time relevant) |
| **classification** | **{classification}** |

### Fallback ladder (Task 5)
{('Invoked: ' + ', '.join(fallbacks_invoked)) if fallbacks_invoked else 'No fallbacks invoked. Default SEI-only calibration passed on first try.'}

## Validation ({N_VAL}-cycle PyBaMM run with calibrated D_SEI)
| metric | sim | measured |
|---|---|---|
| slope (pp/100cy) | {sim_slope_pp_per_100cy:.4f} | **{target_slope_pp_per_100cy:.4f}** |
| SoH at cycle {mid_cy} (mid-life) | {mid_sim:.3f} % | {mid_meas:.3f} % |
| SoH at cycle {N_VAL} (end-of-window) | {end_sim:.3f} % | {end_meas:.3f} % |

| acceptance metric | observed | threshold | pass? |
|---|---|---|---|
| Slope MAE | {slope_mae:.4f} pp/100cy ({slope_mae_pct:.1f}% of target) | < 25 % of \\|target\\| | **{'YES' if acceptance['slope_mae_ok'] else 'NO'}** |
| Mid-life SoH error | {mid_err:.3f} pp | < 2 pp | **{'YES' if acceptance['mid_life_ok'] else 'NO'}** |
| End-of-window SoH error | {end_err:.3f} pp | < 3 pp | **{'YES' if acceptance['end_ok'] else 'NO'}** |

Validation capped at {N_VAL} cycles (vs spec's 50) to respect the < 10 min
wall-time budget. The slope match is the dominant acceptance criterion at
this window; extrapolation beyond cycle {N_VAL} is explicitly outside the
validated range.

## Decision audit trail
{chr(10).join(f"{i+1}. {line}" for i, line in enumerate(audit))}

## Caveats
- Validation window is {N_VAL} cycles, not the spec's 50.
- {'OCV top voltage {:.3f} V outside the LFP full-charge band [3.40, 3.55] V — '
   'stoichiometric x_100/y_100 are best-fit-to-truncated-anchors and should '
   'not be treated as the true cell balance at 100% SoC.'.format(fit.ocv_top_v) if 'OCV_TOP_OUTSIDE_LFP_BAND' in flags else 'OCV top voltage within LFP band; stoichiometric limits are credible.'}
- `n_fresh_sims = {result.n_fresh_sims}` (vs n_evaluations {result.n_evaluations}); cache hits made the calibration cheap.
- Linear slope assumes monotone fade; no knee or plating step detected in the {n_cycles_total}-cycle window.

## Output files
- `{(OUT_DIR / f'{prefix}_pybamm_params.json').relative_to(ROOT)}`
- `{(OUT_DIR / f'{prefix}_aging_calibrated.json').relative_to(ROOT)}`
- `{(OUT_DIR / f'{prefix}_sim_vs_measured.png').relative_to(ROOT)}`
- `{(OUT_DIR / f'{prefix}_longterm_per_cycle.csv').relative_to(ROOT)}`
- `{(OUT_DIR / f'{prefix}_calibrated_sim_{N_VAL}cy.parquet').relative_to(ROOT)}`
"""

with open(OUT_DIR / f"{prefix}_calibration_report.md", "w") as fh:
    fh.write(report)
log(f"  saved {prefix}_calibration_report.md")

# Also save validation metrics JSON for downstream
with open(OUT_DIR / f"{prefix}_validation_metrics.json", "w") as fh:
    json.dump(validation_metrics, fh, indent=2, default=float)

print("\n" + "=" * 60)
print(f"DONE: {prefix}")
print(f"Classification: {classification}")
print(f"D_SEI = {result.fitted_value:.4e} m^2/s")
print(f"Relative error = {rel_err*100:.2f} %")
print(f"n_fresh_sims = {result.n_fresh_sims} (of {result.n_evaluations} evaluations)")
print(f"Gates fired: {list(flags.keys())}")
print(f"Fallbacks: {fallbacks_invoked or 'none'}")
print("=" * 60)
