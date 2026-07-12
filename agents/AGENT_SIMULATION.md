# AGENT: PyBAMM Simulation Sweep

## Your role
Generate synthetic degradation datasets by sweeping degradation parameters
around the identified real-cell values. This is the PINN's primary training
data. You must produce diverse, physically realistic aging trajectories.

## Inputs
- `configs/identified_params.yaml`  — from AGENT_PARAM_ID (must exist)
- `configs/sweep_config.yaml`       — sweep ranges and sampling strategy
- `configs/pybamm_base_params.yaml` — base PyBAMM configuration

## Outputs
- `data/synthetic/trajectories.parquet`  — one row per (cell_id, cycle_n)
- `data/synthetic/ic_curves/`           — IC curves per simulation
- `data/synthetic/sweep_manifest.yaml`  — what was run and with what params
- `data/synthetic/validation_plots/`    — sim vs real overlays

## PyBAMM model configuration

```python
import pybamm

model = pybamm.lithium_ion.DFN(options={
    "SEI": "solvent-diffusion limited",
    "SEI porosity change": "true",
    "lithium plating": "irreversible",
    "loss of active material": "stress-driven",
})
```

Use `pybamm.ParameterValues("Prada2013")` as the chemistry base (LFP-capable),
then override with identified_params.yaml values before sweeping degradation parameters.

Chemistry correctness note:
- In PyBaMM docs/tutorials, `Chen2020` is an LG M50 *NMC* parameterisation (positive OCP is NMC),
  so it is not a defensible base choice for LFP cells.

## Sweep design  →  src/simulation/run_sweep.py

### Degradation parameters to sweep (6 primary)
Load ranges from `configs/sweep_config.yaml`. Default ranges:

```yaml
degradation_parameters:
  k_SEI_ms:       # SEI kinetic rate [m/s]
    min: 1.0e-14
    max: 5.0e-13
    scale: log    # sample in log space

  SEI_partial_molar_volume_m3mol:  # [m³/mol]
    min: 5.0e-5
    max: 1.5e-4
    scale: linear

  lithium_plating_exchange_current_A_m2:    # exchange current density [A/m²]
    min: 1.0e-7
    max: 1.0e-5
    scale: log

  LAM_positive_rate_s:     # stress-driven LAM rate [s⁻¹]
    min: 1.0e-4
    max: 1.0e-2
    scale: log

  temperature_K:           # operating temperature
    min: 298.15            # 25°C (fixed; matches your lab tests)
    max: 298.15
    scale: linear

  c_rate:                  # cycling C-rate
    min: 0.10
    max: 1.00
    scale: linear

sweep:
  n_samples: 800             # total simulations
  sampling_strategy: sobol   # sobol quasi-random, better coverage than random
```

### Simulation loop
```python
from scipy.stats.qmc import Sobol
import pybamm, pandas as pd, numpy as np

def run_single_simulation(params_dict, n_cycles=500):
    """Run one degradation simulation, return per-cycle features."""
    model = pybamm.lithium_ion.DFN(options={...})
    param = pybamm.ParameterValues("Prada2013")
    param.update(params_dict)

    experiment = pybamm.Experiment([
        (
            f"Discharge at {params_dict['c_rate']}C until 2.5 V",
            "Rest for 10 minutes",
            f"Charge at {params_dict['c_rate']}C until 3.65 V",
            "Rest for 10 minutes",
        )
    ] * n_cycles)

    sim = pybamm.Simulation(model, parameter_values=param,
                             experiment=experiment)
    sim.solve()
    return extract_features(sim)
```

### Feature extraction  →  src/simulation/extract_features.py
For each simulated cycle n, extract:

```python
features = {
    "cycle_n":          n,
    "Q_Ah":             # discharge capacity this cycle
    "SOH":              # Q_Ah / Q_0
    "dcir_mOhm":        # simulated DCIR at 50% SOC, C/5 pulse
    "V_mean_discharge": # mean voltage during discharge
    "ic_peak1_V":       # first dQ/dV peak position (LFP lower plateau)
    "ic_peak2_V":       # second dQ/dV peak position (LFP upper plateau)
    "ic_peak1_area":    # area under peak 1 (tracks LLI)
    "ic_peak2_area":    # area under peak 2 (tracks LAM)
    "SEI_thickness_m":  # internal state variable
    "LAM_positive":     # loss of active material, positive electrode
    "T_K":              # temperature
    "c_rate":           # C-rate used
    "k_SEI":            # parameter used (for sweep tracking)
}
```

### IC curve extraction
```python
def extract_ic_curve(voltage, capacity, n_points=500):
    """Compute dQ/dV (incremental capacity) curve."""
    from scipy.signal import savgol_filter
    # Sort by voltage
    idx = np.argsort(voltage)
    V = voltage[idx]
    Q = capacity[idx]
    dQdV = np.gradient(Q, V)
    dQdV_smooth = savgol_filter(dQdV, window_length=21, polyorder=3)
    V_interp = np.linspace(V.min(), V.max(), n_points)
    dQdV_interp = np.interp(V_interp, V, dQdV_smooth)
    return V_interp, dQdV_interp
```

## Parallelisation
Run sweep in parallel using `multiprocessing.Pool` or `joblib.Parallel`.
PyBAMM simulations are independent — embarrassingly parallel.

```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1, verbose=10)(
    delayed(run_single_simulation)(p) for p in param_list
)
```

Expected runtime: ~2–5 min per simulation on CPU, ~30 hours serial.
With 8 cores: ~4 hours. With 32 cores: ~1 hour.

## Validation  →  src/simulation/validate_pybamm.py
Before running full sweep, validate the base parameters reproduce
your real data:

1. Run one simulation with `identified_params.yaml` (no degradation sweep)
2. Compare simulated vs measured:
   - OCV curve: RMSE < 5 mV
   - HPPC discharge pulses at SOC = 0.2, 0.5, 0.8: RMSE < 10 mV
   - Initial DCIR at 50% SOC: within 10%
3. Save overlay plots to `data/synthetic/validation_plots/`
4. If validation fails: stop, report to user, do not run sweep

## Dataset quality checks
After sweep completes:
- Confirm SOH spans 0.6–1.0 across the dataset (good degradation coverage)
- Confirm temperature coverage spans full range
- Flag any simulations that failed or produced non-physical results
  (e.g. SOH > 1.0, negative capacity, solver errors)
- Remove failed simulations from training set
- Report: N_successful / N_total, SOH distribution histogram

## Do not
- Run the full sweep without validating base params first
- Use fewer than 500 successful simulations for training
- Skip IC curve extraction — it's essential for LAM/LLI decomposition
