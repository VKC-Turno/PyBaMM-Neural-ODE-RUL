# SoH normalisation conventions — V8 pipeline

## Summary

| Layer | Convention | Fit source |
|---|---|---|
| Corpus SPMe (`_extract_soh_from_solution`) | SoH_sim ÷ SoH_sim[cycle 1] → starts at 1.0 | per-sim (its own cycle 1) |
| DE fit (`de_loss`) | Compares SoH_sim and observed_meas both normalised to their own cycle-1 | shape-only |
| v7 dataset builder (`phase3_v7_dataset.py`) | `context_delta = ctx - ctx[0]`; absolute `context_soh_start` passed separately | encoder sees delta only |
| SoH-offset augmentation | `[0, -0.1, -0.2, -0.3]` applied to both context and target values | hardcoded — expands training SoH range downward |
| Inference (`forecast_v7`) | Observed cell's actual first-cycle SoH used as ODE integration initial condition | per-cell (real) |

## The bridge

Corpus training data covers SoH range approximately `[0.65, 1.05]` (after
augmentation shifts `[0, -0.1, -0.2, -0.3]` are applied to sim curves
that start at 1.0 and end near 0.65 at 2500 cycles).

Real held-out second-life cells enter at SoH values `0.44 – 1.00` per this
dataset. CALB_0029 in particular enters at ~0.44, BELOW the operator's
training range.

## The engineering assumption

The pipeline assumes **fade dynamics are translation-invariant in absolute
SoH**: a cell at SoH=0.44 fades the same shape as a cell at SoH=0.74.

This is APPROXIMATELY true for LFP because of the broad flat-voltage
plateau, but electrochemistry actually varies with absolute SoH
(intercalation stoichiometry, plating propensity, LFP two-phase behaviour
near SoC extremes). The pipeline works well in practice within the
`[0.44, 1.00]` window we tested, but this is engineering agreement with
observation, not physics-derived proof.

## Implications for the V8 abstract

- Never claim "the operator learned the underlying degradation physics" —
  it learned the fade *shape* under a translation-invariance assumption.
- Prefer language such as "the operator predicts SoH forecasts within an
  observed second-life SoH window bounded by the SoH-offset augmentation
  range".
- Any deployment to cells outside `[0.35, 1.05]` requires either
  re-augmentation with an appropriate offset or explicit fine-tuning.

## Corpus vs observed SoH — quantified

See `outputs/results/corpus_bol_normalisation_domain.pdf`.
