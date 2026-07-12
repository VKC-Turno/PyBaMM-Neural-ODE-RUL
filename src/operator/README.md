# Phase 3 — θ-conditioned DeepONet for SoH Trajectory Prediction

## Problem statement

**Input:** electrochemical fingerprint of a single cell at BOL (or on receipt)
+ a partial cycling record + the PyBaMM parameter vector θ (either ground-truth
from synthetic data, or fitted from characterisation via Phase 1).

**Output:** the full SoH-vs-cycle curve out to end-of-life, for that cell under
the observed cycling protocol.

## Input specification (branch encoder — 5 streams)

The operator's branch net sees five concatenated feature streams:

### Stream 1 — DCIR fingerprint  `(9,)`
Internal resistance sampled across SOC:
```
r_dcir = [R_ohm(SOC=0.9), R_ohm(SOC=0.8), ..., R_ohm(SOC=0.1)]   # mΩ
```
Extracted from the DCIR test, or derived from HPPC pulse decomposition.

### Stream 2 — RPT fingerprint  `(6,)`
```
q_bol                 : initial discharge capacity [Ah]
q_rpt_dischg          : latest measured capacity from RPT [Ah]
delta_q_over_delta_v  : slope of the mid-plateau (LFP-specific fingerprint)
ic_peak1_area         : incremental-capacity peak 1 area
ic_peak2_area         : IC peak 2 area
ocv_span              : OCV difference (V_at_100%) - (V_at_0%)
```

### Stream 3 — early cycling window  `(K,)` — K = 50 by default
```
soh_early = [SoH_1, SoH_2, ..., SoH_K]
```
Encoded through a small MLP into a 32-dim embedding.

### Stream 4 — θ (PyBaMM param vector)  `(10,)`
The 5 swept degradation params + 5 identified BOL params:
```
[k_SEI, V_SEI, plating_i0, LAM_pos_rate, LAM_neg_rate,   # swept
 x_100, y_100, Q_n_init, R0_Ohm, C1_F]                    # identified
```
For synthetic data: use Sobol-sample values verbatim. For real cells: use
`configs/identified_params.yaml` from Phase 1.

### Stream 5 — Protocol  `(4,)`
```
c_rate, DoD, temperature_K, rest_time_min
```

## Query (trunk encoder)

Cycle number `n` normalised by `n_norm_scale = 5000`.

## Output

Hard-monotonic construction:
```
SoH_hat(n) = SoH_init - softplus( dot(branch, trunk(n)) + bias )
```
Always ≤ SoH_init and (weakly) monotonic decreasing when the softplus argument
is monotonic increasing. A ReLU-based monotonicity penalty enforces the rest.

## Loss

```
L = L_data + λ_mono · L_mono + λ_bc · L_bc

L_data = MSE(SoH_hat(n), SoH_measured(n))   over K+1 ≤ n ≤ N_max
L_mono = ReLU(SoH_hat(n+1) - SoH_hat(n)).mean()
L_bc   = MSE(SoH_hat(n=0), SoH_init)
```

## Architecture size (default `OperatorConfig`)

- Branch input: 61 dims (9 + 6 + 32 + 10 + 4)
- Branch: 4 tanh layers × 256 hidden → 128-dim embedding
- Trunk: 4 tanh layers × 128 hidden → 128-dim embedding
- Cycling encoder: 3 tanh layers × 64 hidden → 32 dim
- Output: dot product + softplus + boundary anchor
- **Total: ~322 k parameters**

## Training data flow

**Pretraining stage (synthetic):**
1. Load `data/synthetic/trajectories.parquet` (each row: one cycle of one sim).
2. Per sim: extract Stream 1–5 features from Sobol-sampled θ + first-cycle
   PyBaMM outputs. Trajectory = target `SoH(n)` for all cycles.
3. Train `L_data` on synthetic pairs. Corpus size ≥ 300 sims × 3000 cy each.

**Fine-tuning stage (real cells):**
4. For each real cell with full characterisation (EVE 0005–0008, REPT 0001):
   - Compute Stream 1 (DCIR) from measured pulse test
   - Compute Stream 2 (RPT) from measured RPT csv
   - Take first 50 measured cycles as Stream 3
   - Take θ from `configs/identified_params.yaml`
   - Target: measured SoH(n) trajectory
5. Train `L_data` on these pairs (much smaller batch, high LR-schedule scale)
   to correct any sim-to-real residual.

## Files in this directory

- `model.py`     — `ThetaDeepONet` class + `OperatorConfig` + `loss_fn`
- `dataset.py`   — (TODO) build training tensors from `trajectories.parquet`
- `train.py`     — (TODO) two-stage pretraining + fine-tuning driver
- `real_data.py` — (TODO) load Stream 1–5 features from measured CSVs

## Status

- [x] Model architecture built + smoke-tested
- [ ] `dataset.py` — synthetic pipeline
- [ ] `dataset.py` — real-cell pipeline
- [ ] `train.py` — pretraining loop
- [ ] `train.py` — fine-tuning loop
- [ ] Evaluation harness — hold-one-out cross-validation on real cells
