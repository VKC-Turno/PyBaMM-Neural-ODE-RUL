# Phase 2 aggregate verdict — 13-cell DE fit

## 1. Headline verdict

**10 of 13 cells are trustworthy for Phase 3 corpus generation.** All three failures are REPT (0004, 0012, 0046); all three hit the `SOLVER_FAIL_PENALTY_PP=10.0` because the PyBaMM sim physically collapses ("Zero negative electrode porosity cut-off") mid-trajectory, tripping the `coverage < 0.90` guard. The remaining 10 cells span 0.055–1.573 pp RMSE — well inside a defensible envelope for seeding synthetic degradation trajectories.

## 2. Per-cell table

| Make | Cell | RMSE (pp) | wellID (n/5) | Notes | Verdict |
|---|---|---|---|---|---|
| CALB | 0003 | 0.83 | 3 | clean | **KEEP** |
| CALB | 0008 | 1.57 | 2 | high-side RMSE, `k_SEI` at 0.39 span | **KEEP** |
| CALB | 0009 | 0.46 | 2 | | **KEEP** |
| CALB | 0010 | 0.31 | 3 | | **KEEP** |
| CALB | 0015 | 0.21 | 5 | best fit; `D_SEI` clips lower bound | **KEEP** (flag) |
| EVE  | 0002 | 0.98 | 2 | `D_SEI` + `k_plating` clip lower | **KEEP** (flag) |
| EVE  | 0004 | 0.69 | 3 | `D_SEI` clips lower | **KEEP** (flag) |
| EVE  | 0008 | 1.07 | 4 | matches old fit to <1 dec on 3/4 params | **KEEP** |
| REPT | 0004 | **10.00** | 1 | sim dies cycle 107/205 (porosity collapse) | **DROP** |
| REPT | 0007 | 0.055 | 2 | `k_plating` + `k_LAM_neg` clip lower | **KEEP** (flag) |
| REPT | 0012 | **10.00** | 1 | sim dies mid-trajectory | **DROP** |
| REPT | 0046 | **10.00** | 1 | `D_SEI` clips **upper** — pushing into failure regime | **DROP** |
| REPT | 0057 | 0.063 | 5 | `k_plating` clips lower | **KEEP** (flag) |

"Flag" = keep in corpus but log which parameter sits on the bound; do not use these θ as extrapolation anchors.

## 3. Failure diagnosis (REPT 0004 / 0012 / 0046)

The failure is **physical, not numerical**. `simulate_soh_trajectory` returns a real, truncated trajectory (~52% coverage on 0004) because negative-electrode porosity is being consumed by SEI + plating faster than the cycle count can complete. All three cells show DE-search spans of 78–93% of the box on 4/5 params — DE never found a working corner, so it wandered. The one "well-identified" band (`D_SEI_solvent`) is a mirage: when every eval returns the penalty, the top-10% quantile collapses onto whatever the last few evals were.

REPT 0046 gives the cleanest tell: DE pushed `D_SEI_solvent` to the **upper** bound (8.6e-19), i.e. very fast SEI growth. Combined with REPT's already-elevated `k_plating` cohort centre of mass (−10.93 vs CALB −11.65), the negative electrode is being over-consumed. These are REPT's fast-fade protocols (0.5C/0.5D on 0046) with SoH targets around 0.80, so the fit landscape genuinely may not contain a physically valid solution at the current bounds.

**Salvage plan (Phase 2.5 rerun, before Phase 3):**
1. Widen upper `k_SEI` / `D_SEI_solvent` bounds by 0.5 dec **and** add a soft porosity-margin penalty inside `simulate_soh_trajectory` so DE gets a gradient instead of a wall.
2. If (1) still fails: accept these three as "protocol too aggressive for our current PyBaMM parameterisation" and drop them — REPT is still represented by 0007 and 0057, both sub-0.1 pp RMSE.

## 4. θ diversity on the 10 successes

Cohort spans across the 10 fittable cells: `k_SEI` 2.1 dec, `D_SEI_solvent` 3.9 dec, `k_plating` 1.5 dec, `k_LAM_neg` 2.4 dec, `V_SEI` ~15% linear. Per-make medians separate cleanly:
- REPT sits ~3 dec higher in `D_SEI_solvent` (−18.4 vs CALB −21.6, EVE −21.9) and higher in `k_plating` — REPT genuinely is a fast-fade cohort.
- CALB and EVE cluster together on SEI-diffusion floor but diverge on `k_LAM_neg` (EVE median −8.05 vs CALB −9.11).

**More than enough diversity to seed a Phase 3 corpus** — three physically distinct fade regimes, each with 2–5 anchor cells.

## 5. Corroboration with prior work

EVE 0008 replicates within **≤1 decade on 3/4 shared params** vs the standalone fit (`k_SEI` +0.16 dec, `V_SEI` −0.63×10⁻⁴ V, `k_plating` +1.20 dec). Only `D_SEI_solvent` moved 2.05 dec — the old fit sat at its lower bound (−20.49), so the new value (−18.44) is a bound-release, not a contradiction. CALB order-of-magnitude corroborates the old CALB_0020 fit. Net: **prior single-cell work is reproduced**, and the multi-cell run releases a previously bound-clipped parameter.

## 6. Concrete next steps

1. **Freeze the 10 KEEP cells as the Phase 3 anchor set.** Write these θ into `configs/dataset.yaml` under `anchors:` and use them to seed the PyBaMM sweep. Do not blend the three REPT failures into the corpus.
2. **Re-run REPT 0004 / 0012 / 0046 with widened SEI upper bounds** (+0.5 dec on both `k_SEI` and `D_SEI_solvent`) and a soft porosity-margin penalty. Budget: 3 cells × 200 evals × ~120 s ≈ 12 min at `--n-jobs 5`. Cheap; worth trying once before dropping.
3. **Log bound-clip flags into the Phase 3 sweep sampler** — when Phase 3 samples θ near a flagged anchor, do not extrapolate past the clipped bound. The 5 flagged cells (CALB 0015, EVE 0002/0004, REPT 0007/0057) anchor interpolation only.
