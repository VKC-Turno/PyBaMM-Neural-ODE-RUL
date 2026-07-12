# Phase-3 Censored-Loss Recommendation (final, N=490)

Corpus: 7 anchors × 70 = **N=490** (CALB_0003/0009/0010/0015, EVE_0004, REPT_0007/0057).
Kept for training (`outcome == "ok"`) = **466**; rejected = 24.
Final decile refit, 2026-07-11.

## Empirical constants (pooled `ok` reaching-EoL, N=228)

- Censored fraction at horizon 2500 cy (SoH ≥ 0.80): **238 / 466 = 51.07 %** — inside [40 %, 60 %].
- Reaching-EoL decile edges p10, p18, …, p90:
  **[302, 478, 618, 793, 1001, 1134, 1297, 1448, 1643, 1898, 2188]**
- Natural per-decile counts (below-p10, 10 middle gaps, above-p90):
  `23 | 18 19 18 18 18 17 19 18 19 18 | 23`  (Σ = 228).
- Loss-code 10-bin counts (bucketize + clamp(1,10)−1, boundaries p18…p82):
  `[41, 19, 18, 18, 18, 17, 19, 18, 19, 41]`.
- Censored SoH-at-horizon **p10 / p50 / p90 = 0.823 / 0.889 / 0.956**.

## Inverse-frequency weights

```
Raw w = 1/(count+1):   [0.02381, 0.05000, 0.05263, 0.05263, 0.05263,
                        0.05556, 0.05000, 0.05263, 0.05000, 0.02381]
Normalised (mean=1):   [0.513, 1.078, 1.135, 1.135, 1.135,
                        1.198, 1.078, 1.135, 1.078, 0.513]
max/min = 2.333        ≤ 3.0 target — no clipping needed.
Paper vector (2dp):    [0.51, 1.08, 1.14, 1.14, 1.14,
                        1.20, 1.08, 1.14, 1.08, 0.51]
```

## Recommended constants (`src/pinn/loss.py`)

```
BIN_EDGES_CY   = [302, 478, 618, 793, 1001, 1134, 1297, 1448, 1643, 1898, 2188]
BIN_WEIGHTS    = [0.51, 1.08, 1.14, 1.14, 1.14, 1.20, 1.08, 1.14, 1.08, 0.51]
CENSORED_W     = 1.00     # match mean(w_reach); do NOT inverse-freq censored
LAMBDA_TOBIT   = 1.0      # reverse-hinge on censored SoH-at-horizon
LAMBDA_MONO    = 0.3      # unchanged (AGENT_PINN)
SOH_EOL        = 0.80
HORIZON_CY     = 2500
```

- Middle mass (p18–p82) at 1.08–1.20 — within 12 % of unity; mid-range signal essentially unweighted.
- Tails (b1, b10) drop to 0.51 because they absorb the below-p10 + above-p90 mass (41 vs ~18 per middle bin).
- CENSORED_W = 1.00 matches mean(w_reach). Inverse-freq on 238 censored would out-vote every observed bin ~12× and collapse predictions to horizon.
- `LAMBDA_TOBIT = 1.0`: every censored sim ran full 2500 cy → informative SoH lower bound, not unknown crossing time.

## Fast-fade tail — no specialisation

EoL < 500 cy = **44 / 228 = 19.3 %** (per anchor CALB 4/8/5/1, EVE 4, REPT 11/11) — already over-represented vs field prior. Rejected samples with SoH-min < 0.30 = **1 / 24**, negligible. **No** extra weight on EoL < 500 cy.

## Drift vs prior CALB-only fit (N=280 → N=490)

| Decile | CALB-only | N=490 pooled | Drift |
|---|---|---|---|
| p10 | 588  | 302  | **−48.6 %** |
| p50 | 1178 | 1134 | −3.7 %      |
| p90 | 1934 | 2188 | +13.1 %     |

Lower-tail collapse confirmed by EVE + REPT-2× fade-heavier draws. Middle mass p42–p66 stable within ±5 %. **Final refit — no further recal inside Phase 3.**

## Rolling recal — status at N=420 (post-REPT_0007)

Actual pooled censored fraction = **51.67%** (inside [40%, 60%] band). Deciles moved as follows:

| Decile | CALB-only edge | Pooled edge | Drift |
|---|---|---|---|
| p10 | 588 | 314 | **-46.5%** (breached) |
| p20 | 772 | 526 | -31.8% |
| p50 | 1178 | 1159 | -1.6% |
| p80 | 1767 | 1834 | +3.8% |
| p90 | 1934 | 2161 | +11.7% (borderline) |

The lower tail shifted hard (EVE_0004 + REPT_0007 both contributed fast-failing samples the CALB-only bin b1 wasn't calibrated for). Middle mass p40–p80 is stable within ±5%.

**Recal decision**: HOLD the current constants until N=490 (post-REPT_0057). Refitting b1 to ~0.45 at N=420 would be undone by REPT_0057's projected contribution (~87% censored, slower reaching-EoL tail). One full decile refit after REPT_0057 close is cheaper than two partial refits.

## REPT booster bug (2026-07-11)

The fast-fade booster in `phase3_corpus.py:_apply_fast_fade_booster` hardcoded `widen=1.0` while REPT baseline draws use `widen=2.0`. Booster samples came from a **narrower** kernel than baseline — core-piling, not tail-inflation. Fixed in-tree (lines 287/341/808) but REPT_0007 and REPT_0057 in the corpus are stuck with broken booster. Downstream training should include `outcome=rejected` samples with trajectory truncation at SOH < 0.30 to recover the fast-fade signal the natural sweep already produced (5 samples per REPT anchor with EoL < 200 cy are natural fast-fade extremes marked rejected, not intentional boosters).
