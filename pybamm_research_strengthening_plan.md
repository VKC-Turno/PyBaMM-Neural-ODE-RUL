# Recommended Research Strengthening Plan

Assuming there is no deadline, the current approach should be strengthened before finalising the abstract. The goal is to remove the main methodological weaknesses, improve validation, and rewrite the abstract around stronger evidence.

## Recommended Position

Do not submit the current approach after only adding a limitations paragraph. First strengthen the validation and address the key methodological gaps.

For the abstract, include:

- Linear and exponential baselines
- A no-θ ablation
- Parameter-identifiability results
- Leakage-safe grouped splitting
- Leave-one-anchor-out or leave-one-supplier-out validation
- Uncertainty intervals
- Clear separation between experimentally validated and extrapolated forecast regions

These additions establish whether the model complexity is justified and whether the fitted physical parameters carry meaningful information.

## Priority Improvements

### 1. Leave-one-anchor-out validation

Exclude each anchor cell and every synthetic trajectory generated from it. This checks whether the model generalises beyond local perturbations of known anchors.

### 2. Leave-one-supplier-out validation

Train on two suppliers and test on the third. This is required before making a strong cross-supplier claim.

### 3. Grouped split verification

Ensure that all windows derived from the same simulated trajectory remain in the same split. Otherwise, overlapping context-target windows may inflate model performance.

### 4. Diagnostic-similarity prior

Replace the nearest-supplier prior with a weighted prior based on measurable cell features. Supplier identity can remain an auxiliary feature, but it should not determine the prior.

### 5. Uncertainty estimation

Report prediction intervals rather than only point forecasts. An ensemble of independently trained operators is an acceptable first implementation.

### 6. Synthetic-to-real calibration

Compare simulated and experimental distributions for degradation rate, curve curvature, knee-point behaviour, DCIR growth, and measurement noise. Add realistic noise or a learned residual correction when required.

### 7. Context-length study

Evaluate at least K = 10, 20, 50, and 100. This establishes the trade-off between observation time and forecast accuracy.

### 8. Operating-condition coverage

Generate synthetic trajectories across multiple charge rates, discharge rates, depths of discharge, temperatures, and duty cycles. This reduces the risk that the model simply learns anchor-specific protocols.

## Answers to the Three Questions

### Question 1

Include the three proposed additional results:

- Linear baseline
- No-θ ablation
- θ identifiability analysis

However, do not stop there. The final abstract should also include a validation result based on either leave-one-anchor-out or leave-one-supplier-out testing. The current held-out-cell result alone is not enough to support a strong cross-supplier claim.

### Question 2

Trim the workflow paragraph first.

Remove implementation-level details such as:

- Specific ODE solver
- Full decoder implementation
- Complete parameter list
- Low-level training details

Keep:

- A compact pipeline figure
- A compact results figure
- Mean model performance
- Variation across cells
- One baseline comparison

Move the detailed per-supplier table to supplementary material if needed.

### Question 3

Create a structured research plan and treat it as the main development plan rather than only a post-submission follow-up.

The plan should include:

- Tasks
- Dependencies
- Acceptance criteria
- Expected outputs
- Estimated effort

## Recommended Work Order

### Stage 1 — Verify the Current Result

- Audit grouped train/validation/test splitting
- Run linear and exponential baselines
- Run no-θ ablation
- Perform parameter-identifiability analysis
- Repeat training with multiple random seeds

**Acceptance criteria**

- No trajectory-derived leakage across splits
- Proposed model clearly outperforms simple baselines
- θ-conditioning provides measurable value
- Reported metrics are stable across seeds
- Parameter-fit confidence or limitations are documented

### Stage 2 — Test Generalisation

- Leave-one-anchor-out validation
- Leave-one-supplier-out validation
- Leave-one-protocol-out validation
- Context-length comparison

**Acceptance criteria**

- Performance remains acceptable on unseen anchors
- Cross-supplier claims are supported by supplier-held-out testing
- The model does not fail completely under unseen protocols
- The minimum useful context length is identified

### Stage 3 — Improve Inference

- Replace nearest-supplier prior with diagnostic-similarity weighting
- Interpolate among multiple anchors
- Support previously unseen suppliers
- Add prediction intervals

**Acceptance criteria**

- Supplier label is no longer the only prior-selection mechanism
- Multi-anchor priors outperform or match single-anchor selection
- Forecast uncertainty is calibrated
- New-supplier inference is technically possible

### Stage 4 — Improve Physical Credibility

- Fit multiple observables
- Build a hierarchical population/supplier/cell parameter model
- Expand PyBaMM simulations across operating conditions
- Quantify and correct synthetic-to-real mismatch

**Candidate fitting targets**

- SoH
- DCIR
- Voltage response
- Charge or discharge duration
- Temperature
- Selected ICA or differential-voltage features

**Acceptance criteria**

- Parameter identifiability improves
- Synthetic trajectories cover experimental behaviour
- Domain-gap metrics are reported
- The physical prior remains interpretable

### Stage 5 — Validate the Application

- Experimentally validate the full forecast horizon
- Define the end-of-second-life threshold
- Report threshold-crossing-cycle error
- Convert forecasts into a practical decision metric

**Possible decision outputs**

- Expected cycles to threshold
- Probability of surviving a warranty period
- Second-life suitability class
- Conservative lower-bound RUL
- Recommended reuse application

## Minimum Evidence Package Before a Strong Abstract

1. Linear or exponential baseline
2. No-θ ablation
3. Parameter-identifiability analysis
4. Leakage-safe grouped splitting
5. Leave-one-anchor-out validation
6. Leave-one-supplier-out validation
7. Uncertainty intervals
8. Clear separation of validated and extrapolated regions

## Recommended Claim Language

### Strong claim, only after full validation

> A PyBaMM-trained, physics-conditioned neural operator improves SoH forecasting over empirical baselines and transfers across held-out LFP suppliers, with quantified uncertainty.

### Safer current claim

> Preliminary transfer was observed across cells from three evaluated LFP suppliers.

## Final Recommendation

With no deadline, do not choose between submitting the current abstract and doing all improvements later.

Strengthen the methodology first, then rewrite the abstract around the improved validation.

The most important next steps are:

1. Leakage audit
2. Baseline comparison
3. No-θ ablation
4. Leave-one-anchor-out validation
5. Leave-one-supplier-out validation
6. Uncertainty estimation
