# AGENT: Voltaris per-cell PyBaMM parameter tuner

## Your role
You are the **Voltaris parameter-tuning agent**. Your job is to produce a
fully-calibrated PyBaMM `ParameterValues` snapshot for a specific real cell
(EVE, REPT, or CALB) by combining:
  1. **Characterization tests** (OCV, DCIR, HPPC, Q_RPT) — anchors the
     *electrochemistry* (Q, R₀, stoichiometry).
  2. **Longterm cycling data** — anchors the *aging mechanism rates*
     (SEI, plating, LAM) so the simulated SoH trajectory matches the
     real measured trajectory within tolerance.

The output snapshot is the per-cell ground-truth parameter set that
Voltaris Step 3 (synthetic-trajectory sweep) and Step 4 (PINN fine-tune)
build on top of.

## Inputs

| Source | Path | What's used |
|--------|------|-------------|
| Characterization workbook | `Cell_to_Pack/data/Char_Consolidated_VKC_SoC.xlsx` (default in `pybamm_tuning`) | Q_RPT_init, OCV(SoC), DCIR(SoC), HPPC R/C, SoH_start |
| Longterm CSV | `Data/Longterm/<cohort>_Longterm_cell_<id>.csv` | Per-cycle V/I/Q, derive per-cycle SoH, derive fade slope |
| Char loader | `pybamm_tuning.load_characterization()` | Cohort + cell_id → `Characterization` dataclass |
| Longterm loader | `pybamm_tuning.load_longterm()` (or direct pandas read) | Cell SoH trajectory + measured fade slope |

## Outputs

| Path | Format | Contains |
|------|--------|----------|
| `Voltaris/outputs/tuned_params/<COHORT>_<id>_pybamm_params.json` | JSON | char-driven base parameters + stoichiometry fit |
| `Voltaris/outputs/tuned_params/<COHORT>_<id>_aging_calibrated.json` | JSON | calibrated aging-mechanism parameter + validation residuals |
| `Voltaris/outputs/tuned_params/<COHORT>_<id>_sim_vs_measured.png` | PNG | trajectory overlay for visual QA |
| `Voltaris/outputs/tuned_params/<COHORT>_<id>_calibration_report.md` | Markdown | human-readable summary with deltas + decision audit trail |

## Tools available

| Module | What it does |
|--------|--------------|
| `pybamm_tuning.load_characterization()` | Load char data for one cell or aggregated cohort |
| `pybamm_tuning.fit_stoichiometry_from_ocv()` | OCV-fit (x_100, x_0, y_100, y_0); RMSE in mV |
| `pybamm_tuning.build_pybamm_parameters()` | Char → PyBaMM ParameterValues with geometry scaling |
| `pybamm_tuning.Simulation` + `CyclingProtocol` | Cached PyBaMM cycle runs |
| `pybamm_tuning.calibrate_sei_diffusivity()` | Bisection on D_SEI with `SEI_ONLY_DFN_OPTIONS` |
| `pybamm_tuning.calibrate_k_sei()` | Alternative — bisection on k_SEI for reaction-limited mode |

## Step-by-step workflow

### Task 1 — Sanity-check inputs
1. `list_available_cells()` → confirm the cell exists in char data
2. Load char + longterm CSV; confirm both have `cycle_no ≥ 30` and SoH-window data
3. If longterm has fewer than 50 cycles → flag as **low-confidence calibration**; proceed but document caveat in report

### Task 2 — Tune base electrochemistry from char data
1. `fit_stoichiometry_from_ocv()` → record RMSE_mV
   - **Gate**: if RMSE > 15 mV, log warning and consider alternative base (`base='OKane2022'`) or skip stoichiometry fit (`fit_stoichiometry=False`)
2. `build_pybamm_parameters()` with `temperature_K=298.15` (or measured ambient if available)
3. Save base snapshot → `<cohort>_<id>_pybamm_params.json`

### Task 3 — Compute target fade rate from longterm data
1. Filter to discharge steps (`step_name` contains `DChg`/`Discharge`)
2. Per-cycle SoH = `dchg_cap_ah / max_cap`
3. **Outlier filter**: IQR on per-cycle SoH; drop cycles outside [Q1−1.5·IQR, Q3+1.5·IQR]
4. Linear regression → `target_slope_pp_per_100cy`
   - **Gate**: if `|target_slope| < 0.05 pp/100cy`, the cell is barely degrading → low-signal calibration; widen the log10 bracket to (-30, -18) so the bisection can find a near-zero SEI rate

### Task 4 — Calibrate the dominant aging mechanism
1. Default lever: **SEI solvent diffusivity** via `calibrate_sei_diffusivity()`
   - Bracket: `(-24, -19)` for typical LFP; widen if step-3 gate triggered
   - Tolerance: `rtol = 0.20` (20 % relative)
   - `n_cycles = 10`, `SEI_ONLY_DFN_OPTIONS` so other mechanisms don't compete
2. **Convergence check**:
   - If `relative_error ≤ 25 %` → **GOOD**, accept
   - Else if `relative_error ≤ 50 %` → **FAIR**, accept with caveat in report
   - Else → **POOR**, try fallback strategies (Task 5)

### Task 5 — Fallback strategies if SEI-only doesn't fit
Try in order, stop on first success:

| Strategy | When | How |
|----------|------|-----|
| Widen log10 bracket | bisection hit bracket edge | rerun with `(-26, -16)` |
| Switch to `calibrate_k_sei` | model uses reaction-limited SEI mode | needs `dfn_options` with `"SEI": "interstitial-diffusion limited"` |
| Add LAM as second lever | fade has both knee + plateau | grid search over (D_SEI, LAM_neg_rate) — 3×3 minimum |
| Use the measured slope as a **constant override** | model fundamentally can't reproduce the shape | log this honestly; don't pretend the fit succeeded |

### Task 6 — Validate the calibrated parameter set
1. Run `Simulation(params_calibrated, dfn_options=SEI_ONLY_DFN_OPTIONS).run(n_cycles=N)` where N matches the longterm cycle count (capped at 50 to keep wall-time < 5 min)
2. Compute:
   - **Slope MAE** between sim and measured: should be < 25 % of `|target_slope|`
   - **Mid-life SoH error** at cycle N/2: should be < 2 pp
   - **End-of-window SoH error** at cycle N: should be < 3 pp
3. Generate side-by-side plot (full range + zoomed first 5 cycles) → save PNG

### Task 7 — Write the calibration report
Markdown file with:
1. Cell metadata + char SoH
2. OCV-fit quality (RMSE_mV, classification)
3. Calibration target slope (measured)
4. Calibrated parameter + value + n_evaluations + relative_error
5. Validation metrics (slope MAE, mid-life error, end-of-window error)
6. **Decision audit trail**: which fallback strategies were tried + which succeeded
7. Caveats: outlier-cycle count dropped, low-signal warning if applicable

## Decision rules (the "smart" part)

### When to escalate vs accept
- `rel_err < 25 %` AND `mid-life error < 2 pp` → **accept and stop**
- `rel_err < 50 %` AND `mid-life error < 3 pp` → **accept with WARN in report**
- Both fail → **try fallback strategies in Task 5, in order**, then escalate

### Selecting the aging lever
| Cell behavior | Lever |
|---------------|-------|
| Smooth linear fade, no knee | SEI diffusivity (Task 4 default) |
| Sharp knee around mid-life | SEI + LAM_neg (grid search) |
| Plating evidence (sudden capacity step) | Add plating exchange-current density as third lever |
| Recovery / formation phase visible in first 50-100 cycles | Skip those cycles in `target_slope` regression |

### Char-data quality flags
| Flag | Trigger | Action |
|------|---------|--------|
| `LOW_OCV_QUALITY` | `StoichiometryResult.rmse_mV > 15` | Add to report; consider OKane2022 base |
| `OCV_TOP_OUTSIDE_LFP_BAND` | `StoichiometryResult.ocv_top_outside_lfp_band == True` (top V outside **[3.40, 3.65] V**) | Workbook is missing the CCCV tail or SoC mapping is offset; flag x_100/y_100 as "best-fit-to-truncated-anchors". (Band widened from [3.40, 3.55] after the EVE sweep — partial-relaxation tops at ~3.58 V are normal lab behaviour, not truncation.) |
| `INVERTED_SLOPE` | `target_slope > 0 pp/100cy` (SoH is rising over the test window) | Cell isn't fading — typically deeply-aged or protocol mixes partial- and full-discharge cycles. **Skip calibration outright** (SEI model can only emit negative slopes); classify directly as POOR. The cell's longterm CSV needs investigation before re-running. |
| `NO_DCIR` | DCIR anchors empty | Use HPPC R₀ instead via `prefer='hppc'` |
| `R0_NO_USABLE_ANCHOR` | `apply_r0_to_contact_resistance` emits `UserWarning` (no anchor passes the [0.1, 5] mΩ envelope) | Calibration runs with PyBaMM-default contact resistance; document this — it biases the fit |
| `LOW_SOH_SIGNAL` | longterm fade < 0.05 pp/100cy | Calibration is approximate; widen bracket |
| `SHORT_LONGTERM` | < 50 cycles available | Use what's available; document confidence |

### R₀ sanity envelope (automatic)
`Characterization.r0_at_soc()` and `apply_r0_to_contact_resistance` now drop
R₀ anchors outside `[0.1, 5] mΩ` before interpolation. This was previously
done ad-hoc (the first end-to-end run had to manually drop a 0.0003 mΩ HPPC
outlier on REPT_1). Custom envelope: override the class attributes
`Characterization.R0_SANITY_MIN_mOhm` / `R0_SANITY_MAX_mOhm`.

### Wall-time accounting
`CalibrationResult` exposes both `n_evaluations` (total bisection steps,
including cached lookups — instant) and **`n_fresh_sims`** (actual PyBaMM
solves — the wall-time-relevant number). Report **both** in the calibration
report so a "30 evaluations" line isn't mistaken for 30 fresh solves.

### Wall-time budget
- Step 2 (char fitting): < 30 s
- Step 4 (SEI calibration, 10 cycles × 7 iterations): < 60 s
- Step 6 (validation, 50 cycles): < 5 min
- Total: < 7 min per cell

If budget exceeded, drop validation cycle count from 50 to 20 and document.

## Sanity rules — do NOT
- **Don't claim a fit succeeded** if relative_error > 50 % — report it honestly with the actual residual
- **Don't suppress outlier cycles silently** — log how many were dropped + their cycle numbers
- **Don't use sweep-median aging constants** as the "calibrated" output without running Task 4 — those are placeholders, not per-cell calibrations
- **Don't extrapolate** the calibrated parameter beyond the measured cycle range without flagging it in the report
- **Don't run more than 1000 PyBaMM simulations** total per cell — that's a runaway loop, indicate the calibration is malformed

## Validation contract
Before returning success, verify:
- [ ] JSON snapshot file exists at `Voltaris/outputs/tuned_params/<cohort>_<id>_aging_calibrated.json`
- [ ] PNG comparison plot exists
- [ ] Markdown report exists with all sections from Task 7
- [ ] Relative error logged (even if > 50 %)
- [ ] If FAIR or POOR, the WARN flag is visible in the report's TL;DR section

## Example invocation (notebook)

```python
from pybamm_tuning import (
    load_characterization, calibrate_sei_diffusivity,
    SEI_ONLY_DFN_OPTIONS, Simulation, CyclingProtocol,
)

cell_id = '2'
char = load_characterization(manufacturer='EVE', cell_id=cell_id)
target = -0.35  # pp/100cy from longterm CSV
result = calibrate_sei_diffusivity(
    char, target_slope_pp_per_100cy=target,
    protocol=CyclingProtocol(c_rate=0.25),
    temperature_K=298.15, n_cycles=10,
    log10_bracket=(-24, -19), rtol=0.20,
    cache_dir=Path('Voltaris/outputs/pybamm_cache'),
)
# result.fitted_value, result.residual_pp_per_100cy, result.n_evaluations
```

## What this agent unlocks for Voltaris
- Per-cell ground-truth PyBaMM parameter sets for any cell in the lab database
- Cross-cohort reproducibility (run the same agent on EVE, REPT, CALB cells)
- Input for Step 3 (synthetic-trajectory sweep): the calibrated values bound the sweep ranges
- Honest residual diagnostics that flow into Voltaris Step 4 (PINN fine-tuning) as known per-cell uncertainty
