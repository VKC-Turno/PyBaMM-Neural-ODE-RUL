# Stage B Verdict — Phase 2 DE Fits

Scope: adversarial review of the 13-cell fit cohort in `configs/deg_params/`. Combines independent RMSE reproduction, θ-space stress tests on the loss surface, and a devil's-advocate audit of the Stage A aggregate.

---

## 1. What Stage B CONFIRMED

- **Persistence pipeline is clean.** All 13 cells reproduce `best_rmse_pp` from `fitted_params[*].value` to full float precision (Δ ≡ 0.000000 pp). The θ vectors on disk *are* what DE returned — no encoding/normalisation drift between optimiser and YAML.
- **Loss function discriminates.** On CALB_0003 the best fit (0.83 pp) sits ~4× below the zero-degradation floor (3.54 pp) and >10× below solver-fail (10 pp). DE had a real gradient to descend; results are not noise.
- **Ordering is physically sensible.** SEI-only < zero-deg < LAM-only < max-deg matches expected sensitivity ranking for a 25 °C, 402-cycle CALB fit.

## 2. What Stage B REFUTED or WEAKENED

- **"10 cells trustworthy" is over-stated.** RMSE alone does not validate SoH-*shape* agreement (curvature, knee timing). Three cells in the alleged keep list — CALB_0008 (1.57 pp), EVE_0008 (1.07 pp), EVE_0002 (0.98 pp) — sit at or above the SEI-only baseline (1.20 pp) observed on CALB_0003. They are not clearly better than a monolithic-SEI stand-in.
- **"Physically sensible bound clips" is a hand-wave.** The aggregate never argued *why* D_SEI / k_plating flooring is expected at 25 °C, 0.25C. The argument is available in the literature but was not made. Downgrade to "not obviously unphysical".
- **REPT vs CALB/EVE separation on D_SEI is unsupported.** KS test with n=2 REPT vs n=5/3 CALB/EVE is powerless; the cohort-level claim of a cathode/chemistry-family split cannot be defended from this data.
- **Fast-fade axis coverage is thinner than reported.** After dropping the 3 REPT failures, only REPT_0007 and REPT_0057 anchor fast fade — both with clipped parameters *and* RMSE < 0.07 pp on ~2.6 pp measured fade, i.e. inside the collapsed-discrimination window. Effective independent anchors on this axis: ~1.

## 3. Real bugs found

None at the code level. The reproduce agent found zero delta across 13 cells; the stress agent found zero solver misbehaviour inside bounds. All Stage B damage is *interpretive* — the aggregate's confidence exceeded what the RMSE evidence supports.

- **Severity: Medium — reporting/interpretation.** `_aggregate.md` frames V_SEI and k_LAM as adequately identified. Stress test shows a SEI↔LAM degeneracy on CALB_0003 (SEI-only within 1.5× best-RMSE). Fix: in `_aggregate.md`, downgrade V_SEI and k_LAM "well_identified" flags to "under-constrained; correlated"; add per-cell SoH overlay plot (measured vs simulated) before any cell is promoted to Phase 3 keep-list.

## 4. Verdict on the 13-cell cohort

### Confirmed keep (7 cells)
Cells with RMSE clearly below the SEI-only degeneracy floor (~1.20 pp) *and* measured fade wide enough to make RMSE meaningful:
- **CALB_0003** (0.83) — anchor for CALB, well below floor.
- **CALB_0009** (0.46), **CALB_0010** (0.31), **CALB_0015** (0.21) — tight fits on non-trivial fade.
- **EVE_0004** (0.69) — clean.
- **REPT_0007** (0.055), **REPT_0057** (0.063) — retained *only* because they are the sole fast-fade anchors; flag as clip-limited.

### Suspicious — reinspect before Phase 3 (3 cells)
- **CALB_0008** (1.57), **EVE_0002** (0.98), **EVE_0008** (1.07) — at or above SEI-only baseline. Require SoH-shape overlay + knee-cycle error before promotion.

### Drop (3 cells — unchanged from Stage A)
- **REPT_0004, REPT_0012, REPT_0046** — RMSE ~2.5 pp, at horizon-coverage / LAM-ceiling failure mode.

### Hard STOP?
**No.** Phase 3 can proceed with the 7-cell confirmed set. However, before training on the full 10-cell set, generate per-cell measured-vs-simulated SoH overlays and reject any cell whose knee-cycle error exceeds one RPT interval. Also add a second fast-fade cell (or accept that the fast-fade regime is effectively single-anchor) — this is the largest residual risk to Phase 3 generalisation.
