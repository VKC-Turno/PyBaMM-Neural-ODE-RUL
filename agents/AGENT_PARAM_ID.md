# AGENT: Parameter Identification

## Your role
You are the parameter identification agent. Your job is to extract real
electrochemical parameters from the characterisation test data and produce
`configs/identified_params.yaml` — the personalised PyBAMM parameter file
for these specific LFP cells.

## Inputs (read from data/raw/)
- `OCV_SOC/`      — OCV vs SOC curves (multiple temperatures if available)
- `GITT/`         — GITT pulse sequences (V vs t at low C-rate)
- `DCIR/`         — DC internal resistance pulses
- `HPPC/`         — Hybrid pulse power characterisation
- `SelfDischarge/` — Open-circuit voltage decay over rest periods

## Outputs (write to)
- `configs/identified_params.yaml`   — all identified parameters
- `data/processed/param_id_report.md` — fit quality summary with plots

## Step-by-step tasks

### Task 1: OCV stoichiometric fitting  →  src/param_id/ocv_fit.py
Fit the electrode half-cell OCV curves to extract:
- `x_100`, `x_0`  : graphite stoichiometric limits (lithiated/delithiated)
- `y_100`, `y_0`  : LFP stoichiometric limits
- `Q_n_init`, `Q_p_init` : initial electrode capacities [Ah]

Method:
1. Load PyBAMM's built-in LFP half-cell OCV functions
   (`pybamm.ParameterValues("Prada2013")`)
2. Use `scipy.optimize.minimize` to fit stoichiometric windows
   to your measured full-cell OCV curve
3. Report RMSE of fit; flag if > 5 mV

Key code pattern:
```python
import pybamm, numpy as np
from scipy.optimize import minimize

param = pybamm.ParameterValues("Prada2013")
# U_p = LFP cathode OCV function (vs Li/Li+)
# U_n = graphite anode OCV function (vs Li/Li+)
# Full cell OCV = U_p(y) - U_n(x) where x,y track with SOC
```

### Task 2: GITT diffusion extraction  →  src/param_id/gitt_ds.py
Compute **defensible GITT step metrics** and (optionally) an **apparent**
diffusion coefficient, with explicit assumptions.

Why this matters:
- Your GITT measurement is **full-cell voltage**. It contains contributions from
  both electrodes + ohmic/kinetic polarization, so you generally **cannot**
  uniquely identify *both* graphite and LFP solid diffusivities from it without
  additional information (e.g. half-cell curves, or a model-based fit).

What we do in this repo (scientifically defensible minimum):
1. Parse each GITT step (pulse + relaxation) and compute audit-friendly metrics:
   - pulse duration `τ`
   - steady-state relaxation change `ΔEs`
   - pulse change `ΔEτ` (approx.)
   - early-time slope `dV/d√t` and fit quality (R²)
   - SOC estimate from coulomb counting using nominal capacity
2. Compute an **apparent** diffusion coefficient only if you provide an explicit
   diffusion length `L` (meters), using the classical simplified relation:
     D_app = (4 L² / (π τ)) * (ΔEs/ΔEτ)²

If you truly need electrode-specific Dₛ(SOC):
- Fit Dₛ inside a reduced electrochemical model (SPM/SPMe) to the pulse + rest
  waveform (i.e. *model-based pulse fitting*), or use half-cell data.

### Task 3: Resistance identification  →  src/param_id/dcir_hppc.py
From DCIR pulses:
- `R0` : instantaneous resistance = ΔV/I at t=0⁺ [Ohm]

From HPPC pulses (fit first-order RC model V(t) = OCV - I*R0 - I*R1*(1-exp(-t/τ))):
- `R0(SOC)` : ohmic resistance profile
- `R1(SOC)` : charge-transfer resistance profile
- `C1(SOC)` : double-layer capacitance profile (= τ/R1)
- `τ(SOC)`  : time constant profile

Use `scipy.optimize.curve_fit` for each SOC point.
Store as SOC-indexed arrays for PyBAMM interpolation.

### Task 4: SEI rate constraint  →  src/param_id/sei_selfdisc.py
Self-discharge data gives upper bound on SEI electron leakage current.
1. Fit OCV decay: dV/dt = -I_sd / C_cell
2. Extract self-discharge current I_sd [A]
3. Convert to SEI exchange current density: i_sd = I_sd / A_electrode
4. This bounds k_SEI: k_SEI_max = i_sd / (F * c_EC_0)
   where c_EC_0 is initial EC solvent concentration (~4500 mol/m³)

## Output format: configs/identified_params.yaml
```yaml
# Identified from characterisation data — cell batch: <fill in>
# Identification date: <fill in>

stoichiometry:
  x_100: 0.8                 # graphite fully lithiated
  x_0:   0.005               # graphite fully delithiated
  y_100: 0.06                # LFP fully lithiated
  y_0:   0.95                # LFP fully delithiated

capacity:
  Q_n_init_Ah:  # electrode capacity, negative
  Q_p_init_Ah:  # electrode capacity, positive

diffusion:
  # NOTE: electrode-specific Ds is usually not identifiable from full-cell GITT
  # alone. Use literature values or a model-based fit if you need this.
  Ds_n_m2s:
  Ds_p_m2s:

resistance:
  R0_Ohm:       # scalar or SOC-indexed
  R1_Ohm:       # SOC-indexed
  C1_F:         # SOC-indexed

sei:
  k_SEI_max_ms: # upper bound from self-discharge

fit_quality:
  ocv_rmse_mV:
  gitt_Ds_n_r2:
  gitt_Ds_p_r2:
  hppc_r2:
```

## Validation check before finishing
Run `src/simulation/validate_pybamm.py` to confirm your identified params
reproduce the measured HPPC and OCV curves within tolerance:
- OCV RMSE < 5 mV across full SOC range
- HPPC voltage RMSE < 10 mV at each SOC point
- DCIR within 5% of measured values

If validation fails, revisit the failing extraction step.

## Do not
- Invent parameters not identifiable from available data
- Use an NMC-based parameter set (e.g. Chen2020) as a “default” for an LFP cell
- Claim electrode-specific Dₛ from full-cell GITT unless you do model-based fitting (or have half-cell data)
- Skip validation
