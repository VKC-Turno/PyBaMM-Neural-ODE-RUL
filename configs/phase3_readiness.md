# Phase 3 Readiness Report

**Date:** 2026-07-09  **Scope:** pre-flight for the 15h full sweep (7 anchors × 70 sims).

---

## 1. Config files written
| Path | Status |
|---|---|
| `/home/hj/Desktop/PINNs/configs/phase3_sweep.yaml` | WRITTEN — parses via `yaml.safe_load`; 7 anchors enumerated; 5 fitted θ per anchor copied verbatim from `configs/deg_params/<CELL>.yaml` |
| `/home/hj/Desktop/PINNs/configs/phase3_operator.yaml` | WRITTEN — parses cleanly |

**Deviation:** `LAM_pos_rate_s` is not in per-cell deg-params (only 5 θ fitted). Kept in `perturbation_sigma` (σ=0.3 dec) but sweep script must draw around `pybamm_base_params.yaml` default — no anchor value embedded.

## 2. Module implementations
| Path | Status |
|---|---|
| `Voltaris/Data_Exploration/phase3_corpus.py` | DELIVERED — public API (`load_anchor_theta`, `draw_sobol_perturbations`, `run_one_sim`, `run_anchor_block`, `run_full_sweep`) matches spec; subprocess-thread pool pattern reused from `src/simulation/run_sweep.py` |
| `Voltaris/Data_Exploration/phase3_features.py` | DELIVERED — `extract_x_health` mirrors `src/pinn/dataset.HEALTH_FEATURES` (5 features); σ-unit math verified |
| `Voltaris/Data_Exploration/phase3_operator.py` | DELIVERED — 13-dim input (soh + n_norm + 5 x_health + 6 θ), 3 tanh hidden layers; θ flows from input |
| `Voltaris/Data_Exploration/phase3_train_val.py` | DELIVERED — validation returns all 6 summary metrics |

## 3. Smoke test (end-to-end)
**Wall: 8.5s / cap 1200s. All 7 steps PASS, 0 warnings.**

| Step | Result | Elapsed |
|---|---|---|
| load_config | PASS | 0.0s |
| import_modules | PASS | 0.0s |
| run_corpus (2 anchors × 3 sims) | PASS 6/6 ok | 7.2s |
| write trajectories.parquet | PASS (205 rows × 25 cols) | 0.0s |
| build _dataset.parquet | PASS (schema OK) | 0.0s |
| train_operator (5 epochs) | PASS — loss 0.01333 → 0.01158, 0 NaN | 1.2s |
| run_validation | PASS — all keys present | 0.0s |

## 4. Adversarial audit findings

### phase3_corpus.py
- **HIGH — Fast-fade booster NOT IMPLEMENTED.** `phase3_sweep.yaml` declares `fast_fade_booster.min_samples_in_quadrant: 10` for REPT_0007/REPT_0057, but `run_anchor_block` draws once, filters, writes — no post-hoc quadrant top-up loop exists. Design requirement R2 is unenforced.
- **MED — No mid-anchor resume.** Parquet written once after full pool completes; crash at sample 69/70 re-runs all 70 on restart.
- **LOW — Rejected sims still written.** `run_one_sim` returns `quality_flag="rejected"` and `run_anchor_block` writes them; downstream must filter on `outcome=="ok"`.
- **LOW — `LAM_pos_rate_s`** in `perturbation_sigma` has no `SIGMA_ALIAS` → silently ignored (no explicit skip-log).

### phase3_features.py + phase3_train_val.py
- **CRITICAL — θ silently zeroed in dataset.** `phase3_corpus` writes θ columns as `theta_<k>`; `phase3_features._THETA_COLUMN_MAP` looks for `k_SEI`/`k_SEI_ms`; `phase3_train_val._load_corpus_parquet` looks for `theta_norm_<k>`. **Consequence:** `_dataset.parquet` `theta_norm` is `[0,0,0,0,0,0]` for every row (verified). Operator would train with zero θ signal — the entire θ→SOH mapping is lost. Smoke did not catch it because loss still descends on x_health alone.

### phase3_operator.py
- **MED — `theta_norm` stats not saved as buffers.** `feat_mean`/`feat_std` are saved for `x_health` but no `theta_mean`/`theta_std` buffer exists. Inference on a different corpus recomputes z-score externally and silently shifts the operator response.

## 5. Blockers before the 15h sweep
1. **Fix θ-column plumbing** in `phase3_features._THETA_COLUMN_MAP` and `phase3_train_val._load_corpus_parquet` to match the `theta_<k>` names emitted by `phase3_corpus`. Add an assertion that `_dataset.parquet` `theta_norm` variance > 0 across rows.
2. **Implement fast-fade booster** for REPT_0007/REPT_0057 — post-hoc rejection-sample until ≥10 samples in the `D_SEI:floor, LAM_neg:ceiling` quadrant.
3. **Save `theta_mean`/`theta_std` as buffers** in `phase3_operator.py` (mirror the `feat_mean`/`feat_std` pattern) so checkpoints carry corpus stats.

## 6. Non-blocking improvements
- Periodic partial-write with `.partial` suffix in `run_anchor_block` for mid-anchor resume.
- Explicit `skip-log` when `SIGMA_ALIAS` has no `THETA_AXES` entry (e.g. `LAM_pos_rate_s`).
- Downstream filter `outcome=="ok"` documented at every parquet consumer.

---

## Verdict: **NOT-READY**

Smallest fix set: (a) rename θ column lookup in `phase3_features` + `phase3_train_val` to match `theta_<k>`; (b) add fast-fade quadrant top-up in `run_anchor_block`; (c) persist `theta_mean/std` buffers in the operator. Re-run smoke, confirm `_dataset.parquet` `theta_norm` is non-zero and varied, then launch the full sweep.
