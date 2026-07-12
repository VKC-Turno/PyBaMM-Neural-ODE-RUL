# AGENT: PINN Training

## Your role
Train the physics-guided Neural ODE for LFP RUL prediction. Two-stage:
(1) pre-train on PyBaMM synthetic data, (2) fine-tune on real RPT + Longterm
data. Output a production-ready model checkpoint.

Scientific scope note (this dataset):
- Ambient temperature is **25°C for all tests** (treat as isothermal).
- Temperature is therefore not a learnable driver from real data; keep it as a
  constant feature only (or drop it entirely).

## Inputs
- `data/synthetic/trajectories.parquet`  — from AGENT_SIMULATION
- `data/processed/rpt_features.parquet`  — real RPT capacity + IC features
- `data/processed/longterm_cycles.parquet` — real long-term Q(n) (partial)
- `configs/pinn_config.yaml`             — architecture and training config

## Outputs
- `outputs/models/pinn_pretrained.pt`   — after Phase 1
- `outputs/models/pinn_finetuned.pt`    — after Phase 2 (deploy this)
- `outputs/models/pinn_config.yaml`     — saved config for reproducibility
- `outputs/results/training_curves.png`
- `outputs/results/validation_metrics.json`

## Architecture  →  src/pinn/model.py

### Neural ODE formulation
The network learns the degradation dynamics as a continuous ODE:

  dSOH/dn = f_θ(SOH, n, x_health)

where `x_health` is the vector of health indicators:
  [temperature_C, c_rate, dcir_mOhm, ic_peak1_shift, ic_peak2_area_norm]

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint

class DegradationODE(nn.Module):
    """Right-hand side of dSOH/dn = f(SOH, n, x_health)"""
    def __init__(self, health_dim=5, hidden=64, n_layers=3):
        super().__init__()
        layers = [nn.Linear(1 + 1 + health_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        # Initialise output layer near zero — start with slow degradation
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, n, state):
        # state = [SOH], shape (batch, 1)
        # n = scalar cycle number (broadcast)
        n_feat = n.expand(state.shape[0], 1)
        inp = torch.cat([state, n_feat, self.health_context], dim=-1)
        dSOH = -torch.abs(self.net(inp))  # enforce monotonic decrease
        return dSOH

class RULPredictor(nn.Module):
    def __init__(self, health_dim=5, hidden=64, eol_threshold=0.8):
        super().__init__()
        self.ode = DegradationODE(health_dim, hidden)
        self.eol = eol_threshold

    def forward(self, soh_0, n_eval, x_health):
        self.ode.health_context = x_health
        trajectory = odeint(
            self.ode, soh_0, n_eval,
            method='dopri5', rtol=1e-4, atol=1e-6
        )
        return trajectory  # (T, batch, 1)
```

### Why `torchdiffeq` + dopri5
- Adaptive step size handles flat early-life + accelerating late-life regions
- Gradients flow through the ODE solver (use `odeint_adjoint` if you need adjoint)
- More stable than fixed-step Euler for smooth degradation ODEs

## Loss function  →  src/pinn/loss.py

```python
def total_loss(pred_traj, true_Q, n_cycles, pybamm_ode_fn, lambda_phys=0.1, lambda_mono=0.05):
    """
    pred_traj : (T, batch, 1) — predicted SOH trajectory
    true_Q    : (batch, T)    — measured SOH from RPT/synthetic
    pybamm_ode_fn : callable  — differentiable "physics teacher" for dSOH/dn
    """
    # L_data: supervised loss on known Q(n) points
    L_data = F.mse_loss(pred_traj.squeeze(-1).T, true_Q)

    # L_physics: ODE residual at collocation points
    dSOH_pred = torch.gradient(pred_traj.squeeze(-1), spacing=(n_cycles,), dim=0)[0]
    dSOH_phys = pybamm_ode_fn(pred_traj, n_cycles)
    L_physics = F.mse_loss(dSOH_pred, dSOH_phys)

    # L_monotonicity: penalise any SOH increase
    diffs = pred_traj[1:] - pred_traj[:-1]
    L_mono = torch.relu(diffs).pow(2).mean()

    return L_data + lambda_phys * L_physics + lambda_mono * L_mono
```

### PyBaMM as the physics teacher (defensible)
Do **not** hard-code a single analytical SEI-only form as the “physics residual”
unless your synthetic generator is restricted to that mechanism and you have
validated that the same functional form matches your protocol/data.

Recommended approach in this repo:
1. Generate synthetic trajectories from PyBaMM (your physics generator).
2. Compute a derivative target `dSOH/dn` from PyBaMM outputs (finite differences).
3. Fit a small differentiable surrogate `g_φ(SOH, n, x_health)` to predict that
   derivative on synthetic data.
4. Use `g_φ` as `pybamm_ode_fn` inside `L_physics` during PINN training.

This keeps the “physics term” consistent with the simulator you are using,
without claiming a closed-form mechanism model.

## Training procedure  →  src/pinn/train.py

### Phase 1: Pre-training on synthetic data
```python
config = {
    "lr": 1e-3,
    "batch_size": 64,
    "epochs": 200,
    "scheduler": "cosine",
    "lambda_phys": 0.1,
    "lambda_mono": 0.05,
    "patience": 20,           # early stopping
    "val_split": 0.15,
    "seed": 456,
}
```

Training split: 70% train / 15% val / 15% test (from synthetic data).
Validation: held-out simulations at unseen parameter combinations.

### Phase 2: Fine-tuning on real data
```python
finetune_config = {
    "lr": 1e-4,              # 10× smaller than pre-training
    "batch_size": 8,         # small — few real cells
    "epochs": 100,
    "freeze_layers": 2,      # freeze first 2 ODE layers
    "lambda_phys": 0.3,      # increase physics weight — trust physics more
    "lambda_mono": 0.1,
}
```

Fine-tuning strategy:
1. Load `pinn_pretrained.pt`
2. Freeze the first N layers of the ODE network (preserves learned dynamics)
3. Train only the last layer + output on real RPT + Longterm data
4. Use all available RPT capacity points as supervised targets
5. If Longterm data exists, add it — even 50 real cycles matters

### Dataset construction  →  src/pinn/dataset.py
```python
class DegradationDataset(Dataset):
    """
    Each item: one cell's partial trajectory
    - soh_0: initial SOH (float)
    - n_observed: cycle numbers where SOH is known (tensor)
    - soh_observed: known SOH values (tensor)
    - x_health: health feature vector at each observed point
    """
```

For synthetic data: use full trajectory (all 500 cycles).
For real data: use only observed RPT/Longterm points (sparse).

## Validation metrics
Report after each phase:
- MAE on SOH trajectory (held-out simulations)
- RUL MAE in cycles (at SOH = 0.8 threshold)
- Physics residual norm (L_physics at convergence)
- Monotonicity violations (should be 0 after training)

Acceptable thresholds (guidelines, not guarantees):
- SOH MAE < 0.01 (1% SOH error) on synthetic test set
- RUL MAE < 50 cycles on synthetic test set

## Do not
- Skip Phase 1 pre-training and go straight to fine-tuning on real data
- Use λ_phys = 0 (this makes it a pure data-driven model)
- Fine-tune with a high learning rate (will destroy pre-trained dynamics)
- Report RUL without uncertainty bounds (see AGENT_INFERENCE for this)
