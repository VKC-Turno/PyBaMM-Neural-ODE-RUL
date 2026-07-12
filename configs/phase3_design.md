# Phase 3 — Perturbation Corpus + Operator Retrain + Held-Out Validation

## 1. Goal
Build a synthetic corpus that **breaks the SEI ↔ LAM degeneracy** and **anchors the fast-fade axis**, then retrain the θ-aware Neural ODE operator so it recovers mechanism-attributed RUL — not just net fade — on unseen real cells across CALB / EVE / REPT.

## 2. Success metric
- **Operator SoH RMSE ≤ 3.0 pp** on ≥ 3 held-out real cells spanning all 3 makes (1 CALB + 1 EVE + 1 REPT).
- **Fast-fade slice**: knee-cycle MAE ≤ 1 RPT interval on a leave-one-out with REPT_0007 and REPT_0057 removed from the anchor set.
- **Mechanism attribution (R1 gate)**: on 50 held-out synthetic pairs with matched terminal SoH but different (V_SEI, k_LAM) mix, operator recovers each θ component within **±20 %**; Fisher-column cosine similarity |cos(∂SoH/∂log k_SEI, ∂SoH/∂log k_LAM_neg)| **< 0.3** across the trajectory.

## 3. Corpus generation

### 3.1 Anchor set (7 cells)
CALB_0003, CALB_0009, CALB_0010, CALB_0015, EVE_0004, REPT_0007, REPT_0057. Retained because they span the k_LAM axis (2.42 dec), the bimodal D_SEI gap, and the only two fast-fade points. Suspicious trio (CALB_0008, EVE_0002, EVE_0008) is **excluded from anchors** (R3) but held for validation only if knee-cycle error ≤ 1 RPT interval; otherwise dropped and logged in `configs/deg_params/_phase3_keep.yaml`.

### 3.2 Perturbation (independent log-normal draws around each anchor)

| θ | space | σ | rationale |
|---|---|---|---|
| `k_SEI_ms` | log10 | **0.6 dec** | wide enough to cross the SEI-only-plausible region, includes DE-fit cluster [−12, −11] |
| `SEI_partial_molar_volume` | linear | **15 %** | weakly identified; modest widening |
| `D_SEI` | log10 | **0.7 dec** | bridge the bimodal gap (−21.6 → −20.0) |
| `k_plating` | log10 | **0.5 dec** | axis is nearly degenerate; keep informative but not wasteful |
| `LAM_neg_rate_s` | log10 | **0.8 dec** | dominant fade term, must decorrelate from k_SEI |
| `LAM_pos_rate_s` | log10 | **0.3 dec** | minor contributor |
| `c_rate` | linear | pinned to anchor's identified protocol | |

**Decorrelation gate**: reject the draw if the per-anchor sample Spearman |ρ(log k_SEI, log k_LAM_neg)| > 0.10. Bounds widened 2× on REPT_0007 / REPT_0057 (R4) with interior-minimum re-check emitted to the fit log.

### 3.3 Initial SoH
Every sim starts at BoL (SoH = 1.0). The gap [0.80, 0.95] is filled by dense per-cycle snapshots — no pre-aging offsets (they re-introduce degeneracy).

### 3.4 Protocol assignment
Per-anchor protocol only (matches how θ was identified). Cross-protocol generalisation deferred to Phase 4.

### 3.5 Size + horizon
**70 sims × 7 anchors = 490 sims** (Sobol per anchor, seed = 789 + anchor_idx). Horizon 2500 cycles or SoH < 0.65, whichever first. Explicit **fast-fade booster**: ≥ 10 of the 70 REPT_0007 draws and ≥ 10 of the 70 REPT_0057 draws must fall in the (D_SEI floor, k_LAM ceiling) quadrant → satisfies R2's ≥ 5 additional fast-fade trajectories per anchor.

### 3.6 Model
**SPMe**, isothermal 25 °C. DFN not justified (≈ 10× compute for ≈ 2× fidelity in this regime).

### 3.7 Quality filters (reject-any)
final SoH > 0.98 or < 0.55; monotonicity violation > 0.005 SoH; NaN / negative Q; IDAKLU failure < cycle 100. Keep partial trajectories cycle ≥ 100. Expect ~25 % rejection → floor ≈ 365 usable sims.

## 4. Operator training
- **Architecture**: reuse `RULPredictor` Neural ODE (softplus-monotonic decoder preserved). Expand branch input from `x_health(5)` to `x_health(5) + θ_norm(6)` so the network sees θ explicitly (fixes flat-SoH failure). Refresh `feat_mean/feat_std`.
- **Loss**: `L = L_data + 0.2·L_physics + 0.5·L_mono + 0.3·L_shape`. New **shape term** = weighted MSE on curvature + knee-cycle location (satisfies R1).
- **Optim**: Adam, lr 1e-3 → cosine to 1e-5, batch 32, 250 epochs, patience 25, grad-clip 5.0, seed 456.
- **Splits**: 70 / 15 / 15, **stratified by anchor** so each anchor appears in val + test.

## 5. Held-out validation

### 5.1 Cells (never seen during training)
- **1 clean CALB** (not in anchor set) — generalisation baseline
- **1 EVE** (drawn from the surviving suspicious trio if knee-error passes; else next-cleanest EVE)
- **1 REPT** — leave-one-out on REPT_0007 or REPT_0057

### 5.2 Metrics (reported per cell + per make)
SoH RMSE (pp), SoH-shape MAE, knee-cycle MAE (cycles → RPT intervals), RUL-at-EOL absolute error, fast-fade slice reported separately.

### 5.3 LAM-decorrelation ablation (R1 gate)
1. Fisher-column cosine test (target < 0.3).
2. **Regime-swap replay on CALB_0003**: feed operator (a) joint DE θ and (b) SEI-only θ; forward SoH must diverge > 1 pp somewhere in [0.95, 0.80].
3. **Attribution slope test**: +1σ k_SEI vs +1σ k_LAM_neg must differ in slope, not just intercept.

### 5.4 Pass/fail
Ship only if §2 metrics met **and** all three §5.3 checks pass. Any failure → widen σ on failing axis, re-sweep failing anchor cluster (partial re-run).

## 6. Compute budget + timeline
- Sweep: 490 sims × ~110 s ≈ **15 h** at n_jobs=5 (detached; system unstable above 5). SPMe not DFN.
- Feature extraction + parquet build: ~1 h.
- Operator training: ~2 h on GPU, 250 epochs.
- Validation + ablations: ~1 h.
- **End-to-end ≈ 19 h.** Checkpoint after every 70-sim anchor block; `run_sweep.py` resume-from-manifest; operator saves best-val each epoch.

## 7. Risk log
1. **Residual degeneracy** despite ρ-gate — mitigation: shape loss + Fisher-column test as hard gate, not report-only.
2. **Fast-fade OOD** if held-out REPT lands outside anchor cloud — mitigation: booster quadrant + 2× widened bounds on REPT anchors; report per-cell distance to nearest anchor in θ-space.
3. **Compute overrun / OOM** — mitigation: n_jobs=5 cap, detached run, anchor-block checkpointing; fall back to 50-per-anchor (350 total) if wall time > 24 h.
