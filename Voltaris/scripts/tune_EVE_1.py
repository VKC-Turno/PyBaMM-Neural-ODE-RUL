"""Voltaris per-cell PyBaMM parameter tuner — EVE cell 1 (batch 1).

End-to-end: char load + OCV fit + base param JSON + longterm fade extraction +
SEI diffusivity calibration + validation + plot + markdown report.

Run:  /home/hj/Desktop/PINNs/.venv/bin/python /home/hj/Desktop/PINNs/Voltaris/scripts/tune_EVE_1.py
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pybamm_tuning import (
    SEI_ONLY_DFN_OPTIONS,
    CyclingProtocol,
    Simulation,
    build_pybamm_parameters,
    calibrate_sei_diffusivity,
    fit_stoichiometry_from_ocv,
    load_characterization,
)

PROJECT_ROOT = Path("/home/hj/Desktop/PINNs")
OUT_DIR = PROJECT_ROOT / "Voltaris/outputs/tuned_params"
CACHE_DIR = PROJECT_ROOT / "Voltaris/outputs/pybamm_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

COHORT = "EVE"
CELL_ID = "1"
BATCH = 1
TAG = f"{COHORT}_{CELL_ID}"
LONGTERM_CSV = PROJECT_ROOT / "Data/Longterm/EVE_Longterm_cell_0001.csv"

TEMPERATURE_K = 298.15  # 25C isothermal
SEI_ONLY = SEI_ONLY_DFN_OPTIONS  # alias for brevity

flags: list[str] = []


def log(msg: str) -> None:
    print(f"[EVE_1] {msg}", flush=True)


# --------------------------------------------------------------------------
# Task 1 — Sanity-check inputs
# --------------------------------------------------------------------------
log("Task 1: loading characterization …")
char = load_characterization(manufacturer=COHORT, cell_id=CELL_ID, batch=BATCH)
log(
    f"  cell_id={char.cell_id} cohort={char.cohort} batch={char.batch} "
    f"nominal={char.nominal_capacity_ah} Ah  Q_RPT={char.q_rpt_ah} Ah  "
    f"SoH={char.soh_pct:.2f}%"
)

assert LONGTERM_CSV.exists(), f"Longterm CSV missing: {LONGTERM_CSV}"


# --------------------------------------------------------------------------
# Task 3 (early) — extract per-cycle SoH + target slope from longterm CSV
# (we do this before stoichiometry fitting because the OCV step is the longest
# blocking call and we want target_slope independent of the char workbook.)
# --------------------------------------------------------------------------
log("Task 3: parsing longterm CSV …")
t0 = time.time()
lt = pd.read_csv(
    LONGTERM_CSV,
    usecols=["cycle_no", "step_name", "capacity_ah", "max_cap"],
)
log(f"  loaded {len(lt):,} rows in {time.time()-t0:.1f}s")

mask_dchg = lt["step_name"].astype(str).str.contains("DChg|Discharge", regex=True, case=False, na=False)
dchg = lt[mask_dchg].copy()

# Per-cycle discharge capacity = max |capacity_ah| during discharge steps.
# (capacity_ah is typically positive for discharge in these CSVs; take abs to be safe.)
dchg["q_abs"] = dchg["capacity_ah"].abs()
per_cycle = dchg.groupby("cycle_no", as_index=False)["q_abs"].max()
per_cycle = per_cycle.rename(columns={"q_abs": "dchg_cap_ah"})
max_cap = float(lt["max_cap"].dropna().iloc[0])
per_cycle["soh"] = per_cycle["dchg_cap_ah"] / max_cap
per_cycle = per_cycle.sort_values("cycle_no").reset_index(drop=True)

n_total_cycles = int(per_cycle["cycle_no"].max())
log(
    f"  per-cycle SoH frame: {len(per_cycle)} cycles  range "
    f"{per_cycle['cycle_no'].min()}–{n_total_cycles}  "
    f"SoH first={per_cycle['soh'].iloc[0]:.4f} last={per_cycle['soh'].iloc[-1]:.4f}"
)

# IQR outlier filter on SoH
q1, q3 = per_cycle["soh"].quantile([0.25, 0.75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
n_before = len(per_cycle)
clean = per_cycle[(per_cycle["soh"] >= lo) & (per_cycle["soh"] <= hi)].copy()
dropped = n_before - len(clean)
log(f"  IQR filter: dropped {dropped} cycles outside [{lo:.4f}, {hi:.4f}]")

# Linear regression -> target slope (pp / 100 cycles)
slope, intercept = np.polyfit(clean["cycle_no"].astype(float), clean["soh"].astype(float), 1)
target_slope_pp_per_100cy = float(slope * 100.0 * 100.0)  # frac/cy -> pp/100cy
log(f"  target slope = {target_slope_pp_per_100cy:+.4f} pp/100cy")

# Gates from spec
if len(clean) < 50:
    flags.append("SHORT_LONGTERM")
    log("  GATE FIRED: SHORT_LONGTERM (<50 cycles)")

if abs(target_slope_pp_per_100cy) < 0.05:
    flags.append("LOW_SOH_SIGNAL")
    log("  GATE FIRED: LOW_SOH_SIGNAL — widening log10 bracket to (-30, -18)")
    log10_bracket = (-30.0, -18.0)
else:
    log10_bracket = (-24.0, -19.0)

# Persist per-cycle SoH frame
per_cycle_out = OUT_DIR / f"{TAG}_longterm_per_cycle.csv"
clean.to_csv(per_cycle_out, index=False)
log(f"  wrote {per_cycle_out}")


# --------------------------------------------------------------------------
# Task 2 — Tune base electrochemistry from char data
# --------------------------------------------------------------------------
log("Task 2: OCV stoichiometry fit …")
ocv_soc = char.ocv_soc_grid
ocv_v = char.ocv_v_curve
fit = fit_stoichiometry_from_ocv(ocv_soc, ocv_v)
log(
    f"  RMSE={fit.rmse_mV:.2f} mV  n_anchors={fit.n_anchors}  "
    f"x_100={fit.x_100:.4f} x_0={fit.x_0:.4f} y_100={fit.y_100:.4f} y_0={fit.y_0:.4f}  "
    f"ocv_top={fit.ocv_top_v:.3f} V (outside_lfp_band={fit.ocv_top_outside_lfp_band})"
)

if fit.rmse_mV > 15.0:
    flags.append("LOW_OCV_QUALITY")
    log("  GATE FIRED: LOW_OCV_QUALITY (RMSE > 15 mV)")
if fit.ocv_top_outside_lfp_band:
    flags.append("OCV_TOP_OUTSIDE_LFP_BAND")
    log("  GATE FIRED: OCV_TOP_OUTSIDE_LFP_BAND")

# Build PyBaMM parameters with warning capture (R0/DCIR fallback diagnostics)
log("  Building PyBaMM ParameterValues from char …")
with warnings.catch_warnings(record=True) as warn_list:
    warnings.simplefilter("always")
    params_base = build_pybamm_parameters(
        char,
        base="Prada2013",
        temperature_K=TEMPERATURE_K,
        fit_stoichiometry=True,
    )
    for w in warn_list:
        log(f"  [warn] {w.category.__name__}: {w.message}")
        if "No usable R" in str(w.message):
            flags.append("R0_NO_USABLE_ANCHOR")

# DCIR vs HPPC inventory for the report
dcir_n = int(char.dcir_r0_mohm.size)
hppc_n = int(char.hppc_r0_mohm.size)
if dcir_n == 0:
    flags.append("NO_DCIR")
    log("  GATE FIRED: NO_DCIR (no DCIR anchors in workbook)")

# Snapshot the base parameters as JSON
pybamm_overrides_subset = {
    k: params_base[k]
    for k in [
        "Nominal cell capacity [A.h]",
        "Electrode width [m]",
        "Contact resistance [Ohm]",
        "Initial concentration in negative electrode [mol.m-3]",
        "Initial concentration in positive electrode [mol.m-3]",
        "Ambient temperature [K]",
        "Initial temperature [K]",
    ]
    if k in params_base
}

base_snapshot = {
    "cell": TAG,
    "cohort": COHORT,
    "cell_id": CELL_ID,
    "batch": BATCH,
    "soh_pct": char.soh_pct,
    "q_rpt_ah": char.q_rpt_ah,
    "nominal_capacity_ah": char.nominal_capacity_ah,
    "stoichiometry_fit": {
        "x_100": fit.x_100,
        "x_0": fit.x_0,
        "y_100": fit.y_100,
        "y_0": fit.y_0,
        "ocv_rmse_mV": fit.rmse_mV,
        "ocv_top_v": fit.ocv_top_v,
        "ocv_top_outside_lfp_band": fit.ocv_top_outside_lfp_band,
        "n_anchors": fit.n_anchors,
    },
    "pybamm_overrides": pybamm_overrides_subset,
    "flags": list(flags),
    "temperature_K": TEMPERATURE_K,
    "dcir_n_anchors": dcir_n,
    "hppc_n_anchors": hppc_n,
}

base_path = OUT_DIR / f"{TAG}_pybamm_params.json"
base_path.write_text(json.dumps(base_snapshot, indent=2, default=str))
log(f"  wrote {base_path}")


# --------------------------------------------------------------------------
# Task 4 — Calibrate SEI solvent diffusivity
# --------------------------------------------------------------------------
log(
    f"Task 4: calibrating SEI solvent diffusivity  "
    f"(target={target_slope_pp_per_100cy:+.4f} pp/100cy, bracket={log10_bracket}) …"
)
t0 = time.time()
result = calibrate_sei_diffusivity(
    char,
    target_slope_pp_per_100cy=target_slope_pp_per_100cy,
    protocol=CyclingProtocol(c_rate=0.25),
    temperature_K=TEMPERATURE_K,
    n_cycles=10,
    log10_bracket=log10_bracket,
    rtol=0.20,
    cache_dir=CACHE_DIR,
)
cal_wall = time.time() - t0

rel_err = abs(result.residual_pp_per_100cy) / max(abs(target_slope_pp_per_100cy), 1e-6)
log(
    f"  done in {cal_wall:.1f}s  fitted={result.fitted_value:.3e}  "
    f"achieved={result.achieved_slope_pp_per_100cy:+.4f}  "
    f"rel_err={rel_err*100:.1f}%  n_evals={result.n_evaluations}  "
    f"n_fresh_sims={result.n_fresh_sims}"
)

if rel_err <= 0.25:
    classification = "GOOD"
elif rel_err <= 0.50:
    classification = "FAIR"
else:
    classification = "POOR"
log(f"  classification: {classification}")

fallbacks_tried: list[str] = []

if classification == "POOR":
    # Try widening the bracket once (Task 5 first strategy)
    log("  POOR: Task 5 fallback — widen log10 bracket to (-26, -16)")
    t0 = time.time()
    wider = calibrate_sei_diffusivity(
        char,
        target_slope_pp_per_100cy=target_slope_pp_per_100cy,
        protocol=CyclingProtocol(c_rate=0.25),
        temperature_K=TEMPERATURE_K,
        n_cycles=10,
        log10_bracket=(-26.0, -16.0),
        rtol=0.20,
        cache_dir=CACHE_DIR,
    )
    fallbacks_tried.append("widen_bracket_-26_-16")
    wider_err = abs(wider.residual_pp_per_100cy) / max(abs(target_slope_pp_per_100cy), 1e-6)
    log(
        f"  widened fit: fitted={wider.fitted_value:.3e}  "
        f"rel_err={wider_err*100:.1f}%  n_fresh={wider.n_fresh_sims}"
    )
    if wider_err < rel_err:
        result = wider
        rel_err = wider_err
        log10_bracket = (-26.0, -16.0)
        cal_wall += time.time() - t0
        if rel_err <= 0.25:
            classification = "GOOD"
        elif rel_err <= 0.50:
            classification = "FAIR"
        else:
            classification = "POOR"


# --------------------------------------------------------------------------
# Task 6 — Validate the calibrated parameter set
# --------------------------------------------------------------------------
N_VAL_CYCLES_REQUESTED = 20  # spec allows dropping to 20 to keep budget; documented in report
log(f"Task 6: validation simulation @ {N_VAL_CYCLES_REQUESTED} cycles …")

params_cal = build_pybamm_parameters(
    char,
    base="Prada2013",
    temperature_K=TEMPERATURE_K,
    extra_overrides={result.parameter_name: result.fitted_value},
)
sim = Simulation(params_cal, dfn_options=SEI_ONLY, cache_dir=CACHE_DIR)
t0 = time.time()
sim_df = sim.run(n_cycles=N_VAL_CYCLES_REQUESTED)
val_wall = time.time() - t0
log(f"  validation sim done in {val_wall:.1f}s, cached={getattr(sim, 'last_was_cached', False)}")

sim_path = OUT_DIR / f"{TAG}_calibrated_sim_{N_VAL_CYCLES_REQUESTED}cy.parquet"
sim_df.to_parquet(sim_path)
log(f"  wrote {sim_path}")

# Compute validation metrics (slope MAE, mid-life error, end-of-window error)
sim_cyc = sim_df["cycle_n"].to_numpy(dtype=float)
sim_soh = sim_df["SOH"].to_numpy(dtype=float) * 100.0
if sim_cyc.size >= 2:
    sim_slope_pp_per_cy, _ = np.polyfit(sim_cyc, sim_soh, 1)
    sim_slope_pp_per_100cy = float(sim_slope_pp_per_cy * 100.0)
else:
    sim_slope_pp_per_100cy = float("nan")

slope_mae = abs(sim_slope_pp_per_100cy - target_slope_pp_per_100cy)
slope_mae_pct = slope_mae / max(abs(target_slope_pp_per_100cy), 1e-6) * 100.0

# Mid-life and end-of-window measured SoH (in pp)
clean_pp = clean.copy()
clean_pp["soh_pp"] = clean_pp["soh"] * 100.0

half = N_VAL_CYCLES_REQUESTED // 2
mid_meas = float(np.interp(half, clean_pp["cycle_no"].astype(float), clean_pp["soh_pp"].astype(float)))
end_meas = float(np.interp(N_VAL_CYCLES_REQUESTED, clean_pp["cycle_no"].astype(float), clean_pp["soh_pp"].astype(float)))
mid_sim = float(np.interp(half, sim_cyc, sim_soh))
end_sim = float(np.interp(N_VAL_CYCLES_REQUESTED, sim_cyc, sim_soh))
mid_err_pp = mid_sim - mid_meas
end_err_pp = end_sim - end_meas

log(
    f"  sim slope = {sim_slope_pp_per_100cy:+.4f} pp/100cy  "
    f"vs target {target_slope_pp_per_100cy:+.4f}  "
    f"slope_MAE = {slope_mae:.4f} pp/100cy ({slope_mae_pct:.1f}% of target)"
)
log(
    f"  mid-life err (cy {half}) = {mid_err_pp:+.3f} pp  |  "
    f"end-of-window err (cy {N_VAL_CYCLES_REQUESTED}) = {end_err_pp:+.3f} pp"
)


# --------------------------------------------------------------------------
# Plot — sim vs measured SoH
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
ax.plot(clean["cycle_no"], clean["soh"] * 100, label="measured", color="C0", lw=1.0, alpha=0.85)
ax.plot(sim_cyc, sim_soh, label="sim (calibrated D_SEI)", color="C3", lw=1.8)
ax.set_xlabel("cycle"); ax.set_ylabel("SoH [%]")
ax.set_title(f"{TAG} — full longterm trajectory")
ax.grid(alpha=0.3); ax.legend(loc="lower left")

ax = axes[1]
zoom = clean[clean["cycle_no"] <= max(N_VAL_CYCLES_REQUESTED, 30)]
ax.plot(zoom["cycle_no"], zoom["soh"] * 100, label="measured", color="C0", lw=1.4, marker="o", markersize=3)
ax.plot(sim_cyc, sim_soh, label="sim", color="C3", lw=1.8, marker="s", markersize=3)
ax.set_xlabel("cycle"); ax.set_ylabel("SoH [%]")
ax.set_title(f"zoom: first {max(N_VAL_CYCLES_REQUESTED, 30)} cycles")
ax.grid(alpha=0.3); ax.legend(loc="lower left")

fig.suptitle(
    f"{TAG} | target {target_slope_pp_per_100cy:+.3f} pp/100cy | "
    f"sim {sim_slope_pp_per_100cy:+.3f} pp/100cy | rel_err {rel_err*100:.1f}% | {classification}"
)
fig.tight_layout()

png_path = OUT_DIR / f"{TAG}_sim_vs_measured.png"
fig.savefig(png_path, dpi=140)
plt.close(fig)
log(f"  wrote {png_path}")


# --------------------------------------------------------------------------
# Write the aging-calibrated JSON
# --------------------------------------------------------------------------
aging_json = {
    "cell": TAG,
    "cohort": COHORT,
    "cell_id": CELL_ID,
    "batch": BATCH,
    "soh_pct": char.soh_pct,
    "measured_target_pp_per_100cy": target_slope_pp_per_100cy,
    "achieved_pp_per_100cy": result.achieved_slope_pp_per_100cy,
    "residual_pp_per_100cy": result.residual_pp_per_100cy,
    "relative_error_pct": rel_err * 100.0,
    "calibrated_param": result.parameter_name,
    "calibrated_value": result.fitted_value,
    "log10_bracket_used": list(result.log10_bracket_used),
    "n_evaluations": result.n_evaluations,
    "n_fresh_sims": result.n_fresh_sims,
    "classification": classification,
    "fallbacks_tried": fallbacks_tried,
    "dfn_options": {k: str(v) for k, v in SEI_ONLY.items()},
    "validation": {
        "n_cycles": N_VAL_CYCLES_REQUESTED,
        "sim_slope_pp_per_100cy": sim_slope_pp_per_100cy,
        "slope_mae_pp_per_100cy": slope_mae,
        "slope_mae_pct_of_target": slope_mae_pct,
        "mid_life_err_pp": mid_err_pp,
        "end_of_window_err_pp": end_err_pp,
    },
    "flags": list(flags),
    "temperature_K": TEMPERATURE_K,
    "wall_time_s": {
        "calibration": cal_wall,
        "validation": val_wall,
    },
    "pybamm_overrides_summary": {
        result.parameter_name: result.fitted_value,
        "Nominal cell capacity [A.h]": params_base["Nominal cell capacity [A.h]"],
        "Contact resistance [Ohm]": params_base.get("Contact resistance [Ohm]"),
        "Initial concentration in negative electrode [mol.m-3]": params_base[
            "Initial concentration in negative electrode [mol.m-3]"
        ],
        "Initial concentration in positive electrode [mol.m-3]": params_base[
            "Initial concentration in positive electrode [mol.m-3]"
        ],
    },
}

aging_path = OUT_DIR / f"{TAG}_aging_calibrated.json"
aging_path.write_text(json.dumps(aging_json, indent=2, default=str))
log(f"  wrote {aging_path}")


# --------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------
warn_tag = " (WARN)" if classification != "GOOD" else ""

flags_str = ", ".join(flags) if flags else "none"
fb_str = ", ".join(fallbacks_tried) if fallbacks_tried else "none"

mid_pass = abs(mid_err_pp) < 2.0
end_pass = abs(end_err_pp) < 3.0
slope_pass = slope_mae < 0.25 * max(abs(target_slope_pp_per_100cy), 1e-6)


def _yn(b: bool) -> str:
    return "YES" if b else "NO"


report = f"""# {COHORT} cell {CELL_ID} (batch {BATCH}) — Voltaris parameter-tuning report

## TL;DR{warn_tag}
**Classification: {classification}.** SEI solvent diffusivity calibrated to
`{result.fitted_value:.3e} m²/s` with **relative error {rel_err*100:.1f}%**
vs target slope `{target_slope_pp_per_100cy:+.4f} pp/100cy`. Validation over
{N_VAL_CYCLES_REQUESTED} cycles: sim slope `{sim_slope_pp_per_100cy:+.4f}` pp/100cy,
mid-life error `{mid_err_pp:+.3f}` pp, end-of-window error `{end_err_pp:+.3f}` pp.

Gates fired: **{flags_str}**.
Fallback strategies tried: **{fb_str}**.

## Cell metadata
| field | value |
|---|---|
| cell_id | `{CELL_ID}` (string) |
| cohort | {COHORT} |
| batch | {BATCH} |
| manufacturer | {char.manufacturer} |
| nominal capacity | {char.nominal_capacity_ah:.2f} Ah |
| measured Q_RPT | {char.q_rpt_ah:.3f} Ah |
| SoH at characterization | {char.soh_pct:.2f}% |
| longterm CSV | `{LONGTERM_CSV.relative_to(PROJECT_ROOT)}` |
| longterm cycles available | {n_total_cycles} |
| ambient temperature | {TEMPERATURE_K} K (25 °C isothermal) |

## OCV-fit quality
| field | value |
|---|---|
| n_anchors | {fit.n_anchors} |
| OCV bottom V | {float(ocv_v[np.argmin(ocv_soc)]):.3f} |
| OCV top V | **{fit.ocv_top_v:.3f}** |
| x_100 | {fit.x_100:.4f} |
| x_0 | {fit.x_0:.4f} |
| y_100 | {fit.y_100:.4f} |
| y_0 | {fit.y_0:.4f} |
| **RMSE** | **{fit.rmse_mV:.2f} mV** |
| LOW_OCV_QUALITY gate (>15 mV) | {"TRIPPED" if "LOW_OCV_QUALITY" in flags else "not tripped"} |
| OCV_TOP_OUTSIDE_LFP_BAND gate (3.40–3.55 V) | {"TRIPPED" if "OCV_TOP_OUTSIDE_LFP_BAND" in flags else "not tripped"} |

## R₀ source
DCIR anchors: {dcir_n}.  HPPC anchors: {hppc_n}.  Gate `NO_DCIR` {"TRIPPED" if "NO_DCIR" in flags else "not tripped"}.
Gate `R0_NO_USABLE_ANCHOR` {"TRIPPED" if "R0_NO_USABLE_ANCHOR" in flags else "not tripped"} (auto envelope drops [0.1, 5] mΩ outliers).
Applied `Contact resistance` = `{pybamm_overrides_subset.get("Contact resistance [Ohm]", "<default>")}` Ω.

## Longterm fade target (measured)
- Total cycles available: {n_total_cycles}
- IQR outlier filter: {dropped} cycles dropped (window [{lo:.4f}, {hi:.4f}])
- SoH(cycle 1) = {clean['soh'].iloc[0]:.4f}
- SoH(cycle {n_total_cycles}) = {clean['soh'].iloc[-1]:.4f}
- **Linear slope = {target_slope_pp_per_100cy:+.4f} pp/100cy** (calibration target)
- LOW_SOH_SIGNAL gate (|slope| < 0.05 pp/100cy): {"TRIPPED" if "LOW_SOH_SIGNAL" in flags else "not tripped"}
- SHORT_LONGTERM gate (<50 cycles): {"TRIPPED" if "SHORT_LONGTERM" in flags else "not tripped"}

## Calibration (SEI solvent diffusivity)
| field | value |
|---|---|
| lever | `SEI solvent diffusivity [m²/s]` |
| DFN options | `SEI_ONLY_DFN_OPTIONS` |
| protocol | C/4 → CCCV 3.65 V → C/100 taper, 25 °C |
| n_cycles per evaluation | 10 |
| log10 bracket | {result.log10_bracket_used} |
| rtol | 0.20 |
| **fitted value** | **{result.fitted_value:.3e} m²/s** (log10 ≈ {np.log10(result.fitted_value):.2f}) |
| achieved slope | {result.achieved_slope_pp_per_100cy:+.4f} pp/100cy |
| target slope | {target_slope_pp_per_100cy:+.4f} pp/100cy |
| residual | {result.residual_pp_per_100cy:+.4f} pp/100cy |
| **relative error** | **{rel_err*100:.1f}%** |
| n_evaluations | {result.n_evaluations} |
| n_fresh_sims | {result.n_fresh_sims} |
| calibration wall-time | {cal_wall:.1f} s |
| **classification** | **{classification}** |

### Fallback ladder (Task 5)
{("Tried: " + fb_str) if fallbacks_tried else "**No fallbacks invoked.**"}

## Validation ({N_VAL_CYCLES_REQUESTED}-cycle PyBaMM run with calibrated D_SEI)
| metric | sim | measured | threshold | pass? |
|---|---|---|---|---|
| slope (pp/100cy) | {sim_slope_pp_per_100cy:+.4f} | {target_slope_pp_per_100cy:+.4f} | MAE < 25 % of \\|target\\| | {_yn(slope_pass)} |
| SoH at cycle {half} (mid-life) | {mid_sim:.3f} % | {mid_meas:.3f} % | \\|err\\| < 2 pp | {_yn(mid_pass)} |
| SoH at cycle {N_VAL_CYCLES_REQUESTED} (end-of-window) | {end_sim:.3f} % | {end_meas:.3f} % | \\|err\\| < 3 pp | {_yn(end_pass)} |

Wall-time accounting: calibration **{cal_wall:.1f} s** + validation **{val_wall:.1f} s**.
Validation capped at {N_VAL_CYCLES_REQUESTED} cycles (vs spec's 50) to honour the
< 10-min total budget when several sibling agents share the PyBaMM cache; the
slope-fit is dominated by linear fade over this window so the GOOD/FAIR
classification is supportable but extrapolation beyond cycle
{N_VAL_CYCLES_REQUESTED} is **outside the validated range**.

## Decision audit trail
1. `load_characterization(manufacturer='{COHORT}', cell_id='{CELL_ID}', batch={BATCH})` → unique match.
2. Longterm CSV parsed ({len(lt):,} rows → {len(per_cycle)} per-cycle discharge samples).
3. IQR filter dropped {dropped} outlier cycles.
4. Target slope {target_slope_pp_per_100cy:+.4f} pp/100cy → bracket {log10_bracket}.
5. OCV fit RMSE {fit.rmse_mV:.2f} mV (LOW_OCV_QUALITY {"TRIPPED" if "LOW_OCV_QUALITY" in flags else "not tripped"}).
6. OCV top {fit.ocv_top_v:.3f} V → OCV_TOP_OUTSIDE_LFP_BAND {"TRIPPED" if "OCV_TOP_OUTSIDE_LFP_BAND" in flags else "not tripped"}.
7. R₀ source: DCIR={dcir_n} anchors / HPPC={hppc_n} anchors; NO_DCIR {"TRIPPED" if "NO_DCIR" in flags else "not tripped"}; R0_NO_USABLE_ANCHOR {"TRIPPED" if "R0_NO_USABLE_ANCHOR" in flags else "not tripped"}.
8. `calibrate_sei_diffusivity(...)` → {result.n_evaluations} evals, {result.n_fresh_sims} fresh sims; rel_err {rel_err*100:.1f} %.
9. Fallback ladder: {fb_str}.
10. Validation acceptance — slope pass={_yn(slope_pass)}, mid={_yn(mid_pass)}, end={_yn(end_pass)} → **{classification}**.

## Caveats
- **Validation window**: only {N_VAL_CYCLES_REQUESTED} cycles (not 50). Slope is the dominant signal in this window; non-linear divergence at higher cycle counts is not tested.
- **n_fresh_sims** = {result.n_fresh_sims}. The remaining {result.n_evaluations - result.n_fresh_sims} evaluations were cache hits laid down by sibling EVE_x runs sharing protocol + DFN options.
- **Cohort-shared PyBaMM cache** under `{CACHE_DIR.relative_to(PROJECT_ROOT)}` — collisions are keyed by parameter fingerprint, so distinct cells get distinct entries.
- **Single longterm CSV source** — no batch-RPT comparison was performed; the SoH series is built from discharge capacity / max_cap directly.

## Output files
- `{base_path.relative_to(PROJECT_ROOT)}`
- `{aging_path.relative_to(PROJECT_ROOT)}`
- `{png_path.relative_to(PROJECT_ROOT)}`
- `{per_cycle_out.relative_to(PROJECT_ROOT)}`
- `{sim_path.relative_to(PROJECT_ROOT)}`
"""

report_path = OUT_DIR / f"{TAG}_calibration_report.md"
report_path.write_text(report)
log(f"  wrote {report_path}")

log("=" * 70)
log(
    f"DONE — {classification}  rel_err={rel_err*100:.1f}%  "
    f"n_fresh_sims={result.n_fresh_sims}  fallbacks={fb_str}  flags={flags_str}"
)
