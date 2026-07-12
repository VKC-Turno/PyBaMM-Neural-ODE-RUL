# AGENT: RUL Inference + Online Updating

## Your role
Given new characterisation snapshots from a cell in the field, extract
health features, run RUL prediction with uncertainty bounds, and optionally
update the model with new observations.

## Inputs
- `outputs/models/pinn_finetuned.pt`     — trained model
- New characterisation data (any subset of available tests)
- `configs/identified_params.yaml`       — for feature normalisation

## Outputs
- RUL point estimate (cycles remaining to EOL)
- 90% confidence interval on RUL
- SOH trajectory prediction (current → EOL)
- Feature importance report
- Updated model (if online update requested)

## Health feature extraction  →  src/inference/health_features.py

### Feature vector definition
The PINN expects this exact 5-feature vector (in this order):

```python
HEALTH_FEATURES = [
    "temperature_C",        # mean operating temperature
    "c_rate",               # mean cycling C-rate
    "dcir_mOhm",            # DCIR at 50% SOC (from DCIR or HPPC test)
    "ic_peak1_shift_V",     # shift of lower IC peak vs fresh cell baseline
    "ic_peak2_area_norm",   # area of upper IC peak, normalised to fresh
]
```

### Extraction from each available test
```python
def extract_health_features(data_dict, baseline_params):
    """
    data_dict: dict with keys matching available test names
    baseline_params: fresh-cell reference values from identified_params.yaml
    Returns: np.array of shape (5,)
    """
    features = {}

    # Temperature and C-rate: from any cycling data
    features["temperature_C"] = data_dict.get("temperature_C", 25.0)
    features["c_rate"] = data_dict.get("c_rate", 0.5)

    # DCIR: from DCIR test or HPPC R0 at 50% SOC
    if "DCIR" in data_dict:
        features["dcir_mOhm"] = extract_dcir_at_50soc(data_dict["DCIR"])
    elif "HPPC" in data_dict:
        features["dcir_mOhm"] = extract_r0_from_hppc(data_dict["HPPC"], soc=0.5)
    else:
        features["dcir_mOhm"] = baseline_params["resistance"]["R0_Ohm"] * 1000

    # IC peaks: from RPT slow-discharge or OCV/SOC data
    if "RPT" in data_dict:
        ic_V, ic_dQdV = compute_ic_curve(data_dict["RPT"])
        peaks = find_ic_peaks(ic_V, ic_dQdV)
        features["ic_peak1_shift_V"] = peaks[0]["position"] - baseline_params["ic"]["peak1_V_fresh"]
        features["ic_peak2_area_norm"] = peaks[1]["area"] / baseline_params["ic"]["peak2_area_fresh"]
    else:
        features["ic_peak1_shift_V"] = 0.0
        features["ic_peak2_area_norm"] = 1.0  # assume healthy if no IC data

    return np.array([features[k] for k in HEALTH_FEATURES])
```

### IC peak detection
```python
from scipy.signal import find_peaks, savgol_filter

def find_ic_peaks(V, dQdV):
    """Find the two characteristic LFP IC peaks."""
    dQdV_smooth = savgol_filter(dQdV, 21, 3)
    peak_idx, props = find_peaks(dQdV_smooth, prominence=0.05, distance=30)
    # LFP has two peaks: ~3.3 V (lower plateau) and ~3.4 V (upper plateau)
    # Sort by voltage; return top 2 by prominence
    peaks_sorted = sorted(zip(peak_idx, props["prominences"]),
                          key=lambda x: -x[1])[:2]
    peaks_sorted = sorted(peaks_sorted, key=lambda x: V[x[0]])
    return [{"position": V[idx], "area": props["prominences"][i]}
            for i, (idx, _) in enumerate(peaks_sorted)]
```

## RUL prediction  →  src/inference/predict_rul.py

### Point prediction
```python
def predict_rul(model, soh_now, cycle_now, x_health,
                eol_threshold=0.8, max_cycles=2000):
    model.eval()
    with torch.no_grad():
        soh_0 = torch.tensor([[soh_now]], dtype=torch.float32)
        n_future = torch.linspace(cycle_now, cycle_now + max_cycles, 500)
        x_h = torch.tensor(x_health, dtype=torch.float32).unsqueeze(0)
        trajectory = model(soh_0, n_future, x_h).squeeze()

    # Find first cycle where SOH < EOL threshold
    below_eol = (trajectory < eol_threshold).nonzero(as_tuple=True)[0]
    if len(below_eol) == 0:
        return max_cycles, trajectory, n_future  # RUL > max_cycles
    n_eol = n_future[below_eol[0]].item()
    rul = n_eol - cycle_now
    return rul, trajectory.numpy(), n_future.numpy()
```

### Uncertainty via Monte Carlo dropout
```python
def predict_rul_with_uncertainty(model, soh_now, cycle_now, x_health,
                                  n_samples=200, eol_threshold=0.8):
    """Enable dropout at inference for uncertainty estimation."""
    model.train()  # enables dropout
    rul_samples = []

    for _ in range(n_samples):
        # Add small noise to health features (aleatoric uncertainty)
        x_noisy = x_health + np.random.randn(*x_health.shape) * 0.01
        rul, _, _ = predict_rul(model, soh_now, cycle_now, x_noisy, eol_threshold)
        rul_samples.append(rul)

    rul_samples = np.array(rul_samples)
    return {
        "rul_mean": np.mean(rul_samples),
        "rul_median": np.median(rul_samples),
        "rul_p5":  np.percentile(rul_samples, 5),
        "rul_p95": np.percentile(rul_samples, 95),
        "rul_std": np.std(rul_samples),
    }
```

Note: Model must have `Dropout(p=0.1)` layers in the ODE network for this
to work. Add dropout to `DegradationODE` hidden layers if not already present.

## Online model update  →  src/inference/update.py

When a new RPT measurement is available:

```python
def online_update(model, optimizer, new_soh, new_cycle, x_health,
                  n_steps=20, lr=1e-5):
    """
    Lightweight online update — do not catastrophically forget pre-training.
    Uses elastic weight consolidation (EWC) style regularisation.
    """
    model.train()
    # Freeze all but last layer
    for name, p in model.named_parameters():
        p.requires_grad = ("net.4" in name)  # only last linear layer

    for _ in range(n_steps):
        optimizer.zero_grad()
        pred_soh = model.predict_soh_at_cycle(new_cycle, x_health)
        loss = F.mse_loss(pred_soh, torch.tensor([[new_soh]]))
        loss.backward()
        optimizer.step()

    return model
```

Trigger online update when:
- New RPT shows SOH deviating > 2% from model prediction
- A full new characterisation snapshot is available

## Output report format
```json
{
  "cell_id": "...",
  "assessment_date": "...",
  "cycle_now": 245,
  "soh_now": 0.934,
  "rul_mean_cycles": 387,
  "rul_p5_cycles": 201,
  "rul_p95_cycles": 612,
  "eol_threshold": 0.8,
  "health_features": {
    "temperature_C": 28.5,
    "c_rate": 0.5,
    "dcir_mOhm": 42.1,
    "ic_peak1_shift_V": 0.008,
    "ic_peak2_area_norm": 0.91
  },
  "dominant_mechanism": "SEI_growth",
  "model_version": "pinn_finetuned_v1"
}
```

### Dominant mechanism diagnosis
Infer from health features:
- `ic_peak1_shift_V > 0.01` and `ic_peak2_area_norm > 0.95` → LLI-dominated (SEI)
- `ic_peak2_area_norm < 0.90` → LAM contribution significant
- `dcir_mOhm > 1.5 * baseline` → resistance-dominated (plating risk)
