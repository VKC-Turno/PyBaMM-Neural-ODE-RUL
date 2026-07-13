# LFP RUL PINN — V8 Execution Companion

**Source of truth for scope + intent**:
[`pybamm_research_strengthening_plan.md`](../pybamm_research_strengthening_plan.md).
That document defines *what* to build, *why*, and *in what order* for the
**V8** iteration of this project. This companion adds only the tactical
execution layer: effort estimates, dependencies, artifact paths, and
quantitative acceptance tests for each compute job.

**Status**: V7 (current) is superseded. **V8** is the emerging deliverable —
same core idea (PyBaMM-trained physics-conditioned neural operator) but
with the eight-item minimum evidence package fully attached before any
abstract goes out. The PyBaMM Conference 2026 abstract will NOT be
submitted in its v7.1 form; V8 is drafted only after Stages 1-2 finish.

**V7 → V8 naming convention**:
- V7 artifacts (`configs/phase3_corpus/_v7_dataset.parquet`,
  `outputs/models/pinn_phase3_v7_1_operator.pt`,
  `paper/pybamm_conf_abstract_v7.tex`) are frozen; do not overwrite.
- V8 artifacts get a `_v8` suffix (`_v8_dataset.parquet`,
  `pinn_phase3_v8_operator.pt`, `pybamm_conf_abstract_v8.tex`, etc.) so V7
  results remain diff-able against V8 for the eventual comparison.

**Evidence gate at every step** (user directive 2026-07-13):

- Every task listed below emits a *result artifact* (a table, plot, JSON, or
  short markdown report) BEFORE the next task starts.
- I show you the result and wait for your review. Only after you confirm do
  I move on. No batching, no chained "I ran three tasks, here's a summary".
- If a task takes > 45 min compute wall, I detach it, wait for completion,
  show you the result, and stop.
- If any acceptance criterion fails, we discuss the failure and decide
  whether to fix or replan — never quietly proceed to the next task.

**Cross-cutting rules** (in addition to the strategy plan):

- Never write "cross-supplier transfer" until Stage 2's LOSO passes. Interim
  honest phrasing: *"preliminary transfer across the three evaluated LFP
  suppliers"*.
- Every forecast plot separates the experimentally-validated region (green
  shade) from extrapolated region (pink shade); anchor observed window drawn
  explicitly.
- Every model comparison table lists: (a) linear baseline, (b) exponential
  baseline, (c) no-θ ablation, (d) the current model, on the *same*
  train/val/test split.
- Grouped splitting: all context/target windows from one simulated
  trajectory belong to ONE split. Anchor-grouped splits preferred over
  sim-grouped when the DE-fit chain is being tested.
- Compute budget: `n_jobs ≤ 5` at all times ([[system-unstable-under-full-load]]);
  detach long jobs; never run at full core count.

---

## Stage 1 — Verify the current result

| # | Task | Depends on | Effort | Acceptance criterion (quantitative) | Output path |
|---|---|---|---|---|---|
| 1.1 | Grouped-split audit | none | 1 h | Zero cross-split leakage on `sim_id`; anchor-grouped v6/v7 splits report identical anchor→split map | `outputs/results/grouped_split_audit.json` + splitter fix in `phase3_v7_train.py` if leak found |
| 1.2 | Linear + exponential baselines on K=50 | 1.1 | 1-2 h | Per-cell RMSE for both baselines, same 3 held-out cells, same forecast horizon as v7.1 | `outputs/results/baselines_linear_exp.parquet` |
| 1.3 | No-θ ablation retrain | 1.1 | 45 min compute + 30 min analysis | Retrain OperatorV7 with `theta_norm` zeroed; per-cell RMSE reported on same 3 held-out cells | `outputs/models/pinn_phase3_v7_1_no_theta.pt` + eval report |
| 1.4 | θ identifiability analysis | none | 1-2 h | For each anchor, top-10% population span per θ parameter (see `_identifiability_from_population` in `phase2_de_fit.py:463`); flag `well_identified` when span < 30% of search-bound width | `outputs/results/theta_identifiability.md` |
| 1.5 | Multi-seed training (5 seeds) | 1.1 | 5 × 45 min = ~4 h | Mean ± std RMSE across seeds; RMSE variance across seeds < 0.5× cross-cell RMSE variance | `outputs/results/multi_seed_variance.json` |
| 1.6 | Corpus BOL normalisation audit | none | 30 min | Document explicitly: corpus SoH → sim BOL (=1.0); observed cells enter at their own BOL; SoH-offset augmentation bridges the gap | `docs/normalisation_conventions.md` |

**Stage 1 gate**: proceed to Stage 2 only if
(a) at least one physics-conditioned model beats the best baseline by ≥ 0.2 pp mean RMSE,
(b) no split leakage detected,
(c) at least 3 of 5 θ parameters per anchor are well-identified.

### Stage 1 status snapshot — 2026-07-13

| Sub-gate | Verdict |
|---|---|
| Split integrity (1.1 → 1.1b) | **PASS after repair** |
| Baseline superiority (1.2) | **FAIL for v7.1** — operator loses to exp on 2 of 3 cells |
| Parameter identifiability (1.4) | **FAIL overall** — 5/7 anchors PASS individually, but only 2 of 5 θ params reliable across anchors |
| BOL normalisation (1.6) | **PASS WITH LIMITATIONS** — SoH-invariance is engineering assumption, not physics |
| Clean v8 retrain (1.1c) | **PENDING** |

**Overall Stage 1**: **FAIL** until v8 clean retrain evidence is in. Core superiority and physics-conditioning claims are not yet established.

### Branch conditions after v8 clean + no-θ + reliable-θ ablations

**Branch A — v8 beats baselines on majority of cells** → proceed to Stage 2 (LOAO / LOSO / context-length / uncertainty).

**Branch B — v8 still loses on ≥ 2 of 3 cells** → do NOT proceed to expensive architecture extensions. First run:
- trajectory-complexity analysis (`01_2b` — completed 2026-07-13; correlations +0.86 to +0.95 with n=3, directional not statistically significant)
- hybrid model-selection framework (`01_2c` + `01_2d`) — exponential for smooth trajectories, operator for complex
- shorter and longer context experiments (K ∈ {10, 20, 100}) restricted to complex-trajectory cells

### Revised scientific framing (2026-07-13, on Branch B trajectory)

**Old question**: *"Does a PyBaMM-conditioned neural operator outperform empirical models on cross-supplier SoH forecasting?"*

**New question**: *"Can early-cycle trajectory complexity be used to select between a simple empirical forecast and a PyBaMM-trained neural operator?"*

**Safe provisional contribution statement**:
> Initial results indicate complementary behaviour: exponential extrapolation performs best on smooth trajectories, whereas the physics-informed neural operator may offer an advantage when early-cycle behaviour departs from a simple exponential form. We therefore investigate an inference-time hybrid selector based on context-only model-fit residuals.

The decisive next evidence is not just whether v8 clean improves, but whether **a threshold chosen without using the three external cells can reliably select the better model**. See `01_2d_hybrid_nested_validation.ipynb`.

### Scope FROZEN for 2026-07-15 submission (locked 2026-07-13 evening)

See memory `v8-scope-frozen-july15`. Ship a 1-page abstract by 2026-07-15; skip all Stages 2-5 items and Stage 1 items not in the frozen list.

**Ship list (in order)**:
1. `01_1c` v8 clean external RMSE
2. `01_1d` leaked-vs-clean comparison
3. `01_3` + `01_3b` in parallel (no-θ, reliable-θ)
4. `01_2d` frozen hybrid selector
5. Pause and choose abstract framing
6. Abstract rewrite (morning of 2026-07-14)
7. 1-page fit + proofread (afternoon of 2026-07-14)
8. Submit 2026-07-15

**Final decision table** (5 outcomes → 5 framings): see memory `v8-scope-frozen-july15`.

**Frozen-numbers sheet**: one final results table with per-cell RMSE for exp/v7.1/v8/no-θ/reliable-θ/frozen-hybrid + mean per method + hybrid picks. Single source of truth for the abstract.

**Submission-safe limitation sentence** (verbatim, do not paraphrase):
> Evaluation is currently limited to three external cells, and longer-horizon projections remain experimentally unverified.

### Value-of-selection gate — PASS (2026-07-13)

Oracle hybrid mean 0.45 pp vs always-exponential 0.70 pp and always-operator 0.74 pp. The **0.25 pp oracle gap over the best fixed strategy justifies selector development**. Context-holdout residual (fit exp on first 80% of context, RMSE on last 20% — see `validation-standards`) cleanly separates all 3 real cells' oracle picks with margin 0.273 → 0.707 pp. Deployable-selector verdict remains PENDING until 01_2d Level-1/2/3 nested validation yields a stable τ chosen from data disjoint from the 3 external cells.

**Branch C — no-θ equals or beats full model** → simplify the scientific claim; PyBaMM remains the corpus generator but is not an inference-time conditioning source. Rewrite architecture around context-only encoder.

### Success criteria for v8 clean retrain

Minimum defensible outcome:
- lower mean OR median RMSE than the best per-cell simple baseline
- improvement on ≥ 2 of 3 cells
- no catastrophic regression on the third
- stable across ≥ 3 seeds
- full model better than no-θ, if the physics-conditioning claim is retained

Stronger requirement (original gate): beat best baseline by ≥ 0.2 pp on the majority of held-out cells. With only 3 cells, any conclusion remains preliminary.

### Deprecated language for the V8 abstract (do not use)

- "outperforms empirical models"
- "accurate cross-supplier forecasting"
- "identified degradation parameters"
- "physics conditioning improves transfer"
- "supplier-agnostic"
- "forecast to end-of-second-life"

Prefer instead: *"a five-dimensional physics-informed conditioning vector obtained from degradation-model fitting"* or *"physics-informed latent descriptors derived from PyBaMM calibration"*. If parameter names are used, note "although not all components are independently identifiable from the available SoH data."

Safe provisional description: *"We investigate whether PyBaMM-generated degradation trajectories can support early-cycle SoH forecasting across commercial LFP cells. Initial results show heterogeneous performance relative to simple extrapolation baselines, motivating leakage-safe retraining, parameter-conditioning ablations and broader validation."*

---

## Stage 2 — Test generalisation

| # | Task | Depends on | Effort | Acceptance criterion | Output |
|---|---|---|---|---|---|
| 2.1 | Leave-one-anchor-out (LOAO) | Stage 1 gate | 7 × (~2 h corpus regen + 45 min retrain + 30 min eval) ≈ 24 h | For each of 7 anchors: retrain on the other 6 anchors' 420 sims + augmentation; evaluate on held-out anchor's real observed Longterm cycling. Report per-anchor RMSE. | `outputs/models/pinn_v7_loao_{anchor}.pt` × 7 + summary |
| 2.2 | Leave-one-supplier-out (LOSO) | Stage 1 gate | 3 × (~2 h corpus regen + 45 min retrain + 30 min eval) ≈ 11 h | For each supplier: retrain on the other two; evaluate on ALL held-out cells of the excluded supplier. Report mean RMSE per supplier + std. | `outputs/models/pinn_v7_loso_{CALB,EVE,REPT}.pt` + supplier-transfer table |
| 2.3 | Leave-one-protocol-out (LOPO) | 4.3 (needs protocol variation in corpus) | ~6 h once 4.3 exists | Train excluding one C-rate or DoD; test on that protocol | LOPO table |
| 2.4 | Context-length study | Stage 1 gate | 4 × 45 min ≈ 3 h | Retrain at K ∈ {10, 20, 50, 100}; plot RMSE vs K; report the K that reaches 95% of K=50 accuracy | `outputs/results/context_length_study.pdf` |

**Stage 2 gate**: LOSO mean RMSE within 1.5× the in-supplier RMSE.

---

## Stage 3 — Improve inference

| # | Task | Depends on | Effort | Acceptance criterion | Output |
|---|---|---|---|---|---|
| 3.1 | Diagnostic-similarity weighted prior | 1.4 + 2.2 | 6-8 h | $\theta_{\text{prior}} = \sum_i w_i \theta_i$ with $w_i \propto \exp(-d(f_{\text{new}}, f_i)/\tau)$ over anchors using capacity, DCIR, partial voltage curve, GITT-derived diffusivity. Supplier ID retained as auxiliary feature only. | new `theta_prior_from_features()` in `phase3_v7_validate.py` + re-eval on LOSO |
| 3.2 | Unseen-supplier stress test | 3.1 | 3-4 h | Rerun LOSO with weighted prior; expected to track whichever anchor is closest in diagnostic space. | delta table vs 2.2 |
| 3.3 | Uncertainty via ensemble | 1.5 | 8-12 h | Ensemble of 5+ independently-trained operators; per-cell p10/p50/p90 forecast bands. Calibration: p80 band contains ~80% of observed points within the observed window. | `outputs/results/ensemble_calibration.pdf` + validate.py path |
| 3.4 | Probabilistic Neural ODE (optional) | 3.3 | 2-3 days | Explore torchsde or MC-dropout on the ODE parameters | prototype |

---

## Stage 4 — Improve physical credibility

| # | Task | Depends on | Effort | Acceptance criterion | Output |
|---|---|---|---|---|---|
| 4.1 | Multi-observable DE loss | 1.4 | 3-5 days rewrite + 1-2 h × 7 anchors refit | Extend `de_loss` with weighted terms for SoH + DCIR-growth + voltage-profile residuals; refit all 7 anchors; compare identifiability before/after | `phase2_de_fit_multi.py` + updated anchor θ |
| 4.2 | Hierarchical θ model | 4.1 stable | 3-5 days | $\theta_i = \theta_{\text{pop}} + \Delta\theta_{\text{supplier}} + \Delta\theta_i$ via PyMC/NumPyro; posterior spans at each level | posterior parquet + priors for unseen suppliers |
| 4.3 | Operating-condition sweep in corpus | 4.1 | 1-2 days rework + retrain | Sample (C-rate ∈ {0.25, 0.5, 1.0}) × (DoD ∈ {80%, 100%}) × (T ∈ {25, 35, 45 °C}) per θ draw; reject unphysical trajectories; corpus grows to ~10k+ trajectories | `configs/phase3_corpus/*_expanded.parquet` |
| 4.4 | Synthetic-to-real calibration | 4.1-4.3 | 1-2 weeks | Distribution comparison (degradation-rate, curvature, knee, DCIR growth); residual correction: `SoH_real = SoH_pybamm + f_residual(state, cycle)` | `outputs/results/domain_gap_report.pdf` + `residual_correction.pt` |

---

## Stage 5 — Application validation

| # | Task | Depends on | Effort | Acceptance criterion | Output |
|---|---|---|---|---|---|
| 5.1 | Long-horizon experimental verification | Stage 4 | months (cycling-bound) | Continue cycling held-out cells past K=50; compare pilot 15k-cy SPMe vs observed for cells near BOL at ingest. Rolling parquet of observation extensions. | `outputs/results/horizon_verification.md` (monthly rev) |
| 5.2 | Define EoSL threshold | 5.1 partial | 1-2 h business alignment | Coordinate with product on 0.80 vs application-specific | `configs/eosl_threshold.yaml` |
| 5.3 | Threshold-crossing RUL metric | 5.2 | 2-3 h | Rework abstract's headline metric from SoH-RMSE to *cycles-to-EoSL RMSE* + P(SoH > threshold at N cycles) | `outputs/results/rul_metrics.parquet` |
| 5.4 | Suitability classifier / warranty predictor | 5.3 | 4-6 h | Forecast + uncertainty → discrete recommendation ("suitable for BESS reuse", "recycle") with confidence | `src/inference/suitability.py` |

---

## Rough wall-time summary (n_jobs ≤ 5)

| Stage | Compute + analysis wall time |
|---|---|
| Stage 1 | ~1 week (some overnight) |
| Stage 2 | ~2 weeks |
| Stage 3 | ~1-2 weeks |
| Stage 4 | ~3-4 weeks |
| Stage 5 | months (bounded by real-cell cycling) |

**Realistic gate for a defensible new abstract**: end of Stage 3 (~4-5 weeks
from 2026-07-13). Stage 4 outputs strengthen it further but are not blockers.
Stage 5's long-horizon verification is naturally deferred to a full
manuscript.

---

## Immediate next steps (this week)

Ordering by dependency:

1. **1.4 θ identifiability** — no compute needed; the DE final populations
   are already stored in the fit outputs. Fastest way to catch fatal issues
   before we spend compute.
2. **1.1 grouped-split audit** — read-only, ~1 h; must pass before any
   retrain is trustworthy.
3. **1.6 corpus BOL normalisation audit** — 30 min; documents the invariance
   assumption explicitly so the next abstract can reference it.
4. **1.2 linear + exponential baselines** — cheap, immediately diagnostic.
5. **1.3 no-θ ablation retrain** — first compute-bound task; ~45 min
   detached.
6. **1.5 multi-seed training** — 4 h detached; overnight-friendly.
7. Stage 1 gate check.

Every retrain detached and monitored via the standing rule; never blocking
the shell.
