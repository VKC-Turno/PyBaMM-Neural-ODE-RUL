# V8 execution notebooks

Each notebook = one task in the [research roadmap](../../paper/research_roadmap.md).
One result artifact per notebook; user reviews before the next task starts.

## Stages


### Stage 1

- [`01_1_grouped_split_audit.ipynb`](01_1_grouped_split_audit.ipynb) — Grouped-split audit — no cross-split leakage on sim_id
- [`01_1b_rebuild_grouped_dataset.ipynb`](01_1b_rebuild_grouped_dataset.ipynb) — Rebuild dataset with trajectory-grouped splits **(added on 2026-07-13, blocks all downstream Stage 1 tasks)**
- [`01_1c_retrain_clean_split.ipynb`](01_1c_retrain_clean_split.ipynb) — Retrain OperatorV7 from scratch on the clean split
- [`01_1d_compare_leaked_vs_clean.ipynb`](01_1d_compare_leaked_vs_clean.ipynb) — Side-by-side comparison: v7 leaked-split vs v8 clean
- [`01_2_baselines_linear_exp.ipynb`](01_2_baselines_linear_exp.ipynb) — Linear + exponential baselines on K=50 context
- [`01_3_no_theta_ablation.ipynb`](01_3_no_theta_ablation.ipynb) — No-θ ablation — retrain OperatorV7 with theta_norm removed
- [`01_4_theta_identifiability.ipynb`](01_4_theta_identifiability.ipynb) — θ identifiability — population span per anchor per θ from DE fits
- [`01_5_multi_seed_variance.ipynb`](01_5_multi_seed_variance.ipynb) — Multi-seed training (5 seeds) — variance vs cross-cell RMSE
- [`01_6_corpus_bol_normalisation_audit.ipynb`](01_6_corpus_bol_normalisation_audit.ipynb) — Corpus BOL normalisation audit — sim-BOL=1.0 vs observed-BOL

### Stage 2

- [`02_1_leave_one_anchor_out.ipynb`](02_1_leave_one_anchor_out.ipynb) — Leave-one-anchor-out — 7× corpus regen + retrain + eval
- [`02_2_leave_one_supplier_out.ipynb`](02_2_leave_one_supplier_out.ipynb) — Leave-one-supplier-out — the cross-supplier test
- [`02_3_leave_one_protocol_out.ipynb`](02_3_leave_one_protocol_out.ipynb) — Leave-one-protocol-out — protocol variation (depends on 4.3)
- [`02_4_context_length_study.ipynb`](02_4_context_length_study.ipynb) — Context-length study — K ∈ {10, 20, 50, 100}

### Stage 3

- [`03_1_diagnostic_similarity_prior.ipynb`](03_1_diagnostic_similarity_prior.ipynb) — Diagnostic-similarity weighted prior over anchors
- [`03_2_unseen_supplier_stress_test.ipynb`](03_2_unseen_supplier_stress_test.ipynb) — Unseen-supplier stress test — LOSO with weighted prior
- [`03_3_uncertainty_ensemble.ipynb`](03_3_uncertainty_ensemble.ipynb) — Uncertainty via 5-model ensemble — p10/p50/p90 bands
- [`03_4_probabilistic_neural_ode.ipynb`](03_4_probabilistic_neural_ode.ipynb) — Probabilistic Neural ODE (optional) — torchsde or MC-dropout

### Stage 4

- [`04_1_multi_observable_de_loss.ipynb`](04_1_multi_observable_de_loss.ipynb) — Multi-observable DE loss — SoH + DCIR + voltage
- [`04_2_hierarchical_theta_model.ipynb`](04_2_hierarchical_theta_model.ipynb) — Hierarchical θ model — θ_i = θ_pop + Δθ_supplier + Δθ_i
- [`04_3_operating_condition_sweep.ipynb`](04_3_operating_condition_sweep.ipynb) — Operating-condition sweep in corpus — C-rate × DoD × T
- [`04_4_synthetic_to_real_calibration.ipynb`](04_4_synthetic_to_real_calibration.ipynb) — Synthetic-to-real calibration — domain gap + residual correction

### Stage 5

- [`05_1_long_horizon_verification.ipynb`](05_1_long_horizon_verification.ipynb) — Long-horizon experimental verification — continued cycling
- [`05_2_eosl_threshold.ipynb`](05_2_eosl_threshold.ipynb) — Define EoSL threshold — business alignment
- [`05_3_threshold_crossing_rul.ipynb`](05_3_threshold_crossing_rul.ipynb) — Threshold-crossing RUL metric — cycles-to-EoSL RMSE
- [`05_4_suitability_classifier.ipynb`](05_4_suitability_classifier.ipynb) — Suitability classifier / warranty predictor
