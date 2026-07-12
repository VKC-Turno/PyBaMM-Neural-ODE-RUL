# Making this PyBaMM + Neural-ODE workflow scientifically defensible (25°C dataset)

This project can be made defensible **if you narrow the claims to what your data
can actually support** and you validate the mechanistic parts against hold-out
tests before using them as a synthetic-data generator.

## Dataset constraints (facts you must bake into the research claims)
- All experiments are at **25°C** (treat as isothermal; do not claim temperature generalisation).
- **8 cells** are present (`0001`–`0008`), but not all cells have the same depth of
  characterisation (e.g., some tests such as SelfDischarge/PeakPower exist only
  for a subset). The current *defensible* default cohort is recorded in
  `configs/dataset.yaml` and justified in `data/processed/cell_selection_report.md`.
- Long-term cycling spans roughly **46–201 cycles** depending on cell and protocol,
  and does **not** reach EOL (80% SOH) in the available data. This is typically
  insufficient to validate “RUL-to-80%” without additional aging history.

## What is defensible vs. what is not

### Defensible
- Using PyBaMM as a **physics prior / simulator** to generate plausible degradation
  shapes *within a validated operating envelope* (25°C, your tested C-rates, your voltage limits).
- Training a Neural ODE on simulator trajectories and then fine-tuning on sparse real data
  (this is a “physics-guided surrogate” workflow).
- Enforcing monotonicity of SOH (capacity does not increase in normal aging) as a constraint.

### Not defensible (unless you add data / change method)
- Claiming **material parameters** (e.g., separate graphite vs LFP solid diffusivities) are
  “identified” from full-cell GITT voltage alone.
- Sweeping **temperature** in synthetic generation when all real validation is at 25°C.
- Using an **NMC** chemistry parameter set as the base for an **LFP** cell and calling it “physics”.

## Minimum defensible modelling scope

### Stage A: Electrochemical performance model (voltage response)
Goal: show that a PyBaMM model with an LFP-capable parameterisation reproduces your
measured voltage profiles at 25°C across protocols.

Recommended:
1. Start with an LFP-capable parameter set (`Prada2013`) and treat it as a *prior*.
2. Calibrate only parameters that are actually constrained by your tests (effective/lumped):
   - OCV mapping / stoichiometry window (from OCVSOC / low-rate curves)
   - Ohmic resistance (DCIR)
   - Dynamic resistance/RC behaviour (HPPC)
3. Validate on hold-out protocols not used in fitting:
   - RateCapability, ConstantPower, PeakPower

If Stage A fails, do **not** proceed to Stage B. Synthetic aging data from an
unvalidated model is not a defensible prior.

### Stage B: Degradation model (capacity fade trajectory generator)
Goal: produce a synthetic dataset that spans plausible SOH trajectories for your
tested envelope at 25°C.

Defensible constraints:
- Fix temperature at 25°C.
- Limit C-rate range to what you operate/validate.
- Treat degradation parameters as **scenario parameters** unless you can fit them
  and validate on held-out cells/protocols.

## GITT: what you can claim with this repo
- Use GITT to compute **step-level metrics** (ΔEs, ΔEτ, dV/d√t, τ) and to sanity-check
  that early-time V vs √t is linear (high R²).
- Only compute an **apparent** diffusion coefficient if you explicitly state and justify
  an assumed diffusion length `L`. Do not report “Ds_graphite” and “Ds_LFP” from full-cell GITT
  unless you switch to model-based pulse fitting or have half-cell data.

## Neural ODE (“PINN”) training: make the physics term honest
- If you pretrain on PyBaMM trajectories, the training is already physics-consistent by
  construction.
- If you want an additional “physics” residual term, avoid a single closed-form SEI-only
  expression unless you validate it for your protocol. Instead:
  - compute `dSOH/dn` targets from PyBaMM outputs,
  - fit a small differentiable teacher `g_φ`,
  - regularise the Neural ODE derivative toward `g_φ`.

## Validation protocol (avoid leakage)
With multiple cells, prefer **leave-one-cell-out** (or at minimum a held-out cell):
- Train (PyBaMM calibration + any PINN fine-tune) on N−1 cells, test on the held-out cell.
- Report uncertainty bounds and failure cases (do not only report the best cell).

For electrochemical validation:
- Fit on OCVSOC + HPPC + DCIR for the training cells.
- Test on RateCapability / ConstantPower / PeakPower for the held-out cell.

For degradation/RUL:
- Be explicit that long-term data in this repo likely does not reach EOL (80% SOH),
  so “RUL-to-EOL” cannot be validated without additional aging history.

## Repo changes already made to support defensibility
- `src/data_loader.py` now understands your EVE CSV schema, reconstructs a monotonic time base
  from `absolute_time`, standardises `cell_id`, and fixes capacity-unit heuristics for ~105 Ah cells.
- `src/param_id/gitt_ds.py` now outputs step metrics and only computes D_app if an explicit
  diffusion length is provided.
- Configs are now under `configs/` and fixed to 25°C (`configs/sweep_config.yaml`).
- `src/experiment_tracking.py` adds local run folders under `outputs/experiments/` that snapshot
  configs and environment (`pip_freeze.txt`) and stream metrics (`metrics.jsonl`).
