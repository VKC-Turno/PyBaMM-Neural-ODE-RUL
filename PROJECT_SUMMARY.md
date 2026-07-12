# LFP RUL — Physics-Informed Operator for Second-Life Cell Remaining-Useful-Life

**One-line pitch.** A per-cell PyBaMM calibration → θ-conditioned neural operator that predicts Remaining Useful Life from a single characterisation snapshot, rebuilt end-to-end after the original attempt failed silently.

**Owner.** Krishna Chaitanya Vaddepally, Turno (Blubble Pvt. Ltd.)  ·  target venue: PyBaMM Conference 2026

---

## 1. What we set out to do

Given:
- 3 commercial LFP suppliers (CALB 72 Ah, EVE 105 Ah, REPT 150 Ah)
- Second-life cells at 0.60–1.00 fresh-normalised SoH
- Characterisation tests + Longterm cycling data per cell

Produce a **neural operator** that, from one characterisation snapshot on a fresh cell, predicts the full SoH-vs-cycle trajectory to end-of-life across suppliers. Ship a 1-page conference abstract + companion research paper.

## 2. What the previous attempt did (and why it failed)

An earlier build shipped a corpus of 300 PyBaMM sims and a "θ-conditioned DeepONet" that predicted **flat SoH ≈ 1.0** for every real cell.

The failure report attributed this to **θ-space out-of-distribution (5σ OOD)** — the corpus's `log10(k_SEI)` mean was −14.15 while real fits sat at −11.5.

Adversarial re-audit found **the actual root cause was different**: the model at `src/pinn/model.py` was a Neural ODE where the branch input was `x_health(5)` only — **θ was never fed to the network at all**. The 5σ OOD claim was a symptom, not the cause. Without the θ fix, no amount of corpus widening would have helped.

## 3. What this rebuild delivers

### Cohort — from 61 candidate cells to a defensible 13

| Filter | Cells |
|---|---|
| All 3 makes, raw data | 61 |
| After CC/CV protocol correction on CALB SoH | 47 clean |
| After [0.60, 1.00] SoH_first filter + monotonicity | 21 |
| After DoD-protocol grouping + noise ranking | 13 (5 CALB · 3 EVE · 5 REPT) |
| After adversarial verification post Phase 2 | **7 confirmed anchors** (4 CALB · 1 EVE · 2 REPT) |

### Three-phase pipeline

1. **Phase 1 — Per-cell BOL identification.**
   OCV curve fit gives stoichiometry (x_100, x_0, y_100, y_0); Q_rpt derives electrode capacities; HPPC gives R0 + R1; GITT gives D_s.
   **Result: 13/13 cells pass the OCV RMSE < 20 mV upper-half gate.**

2. **Phase 2 — Per-cell degradation-parameter DE fit.**
   Differential evolution fits 5 parameters (k_SEI, V_SEI, D_SEI_solvent, k_plating, k_LAM_negative) against measured Longterm SoH using each cell's actual cycling protocol.
   **Result: 10 of 13 cells sub-2 pp RMSE; 3 aged-REPT cells fit at ~2.5 pp (limited by measured signal-to-noise, not model quality).**

3. **Phase 3 — Perturbation corpus + operator retrain.**
   Sobol sample 70 draws per anchor around each of the 7 Phase-2 fits, with decorrelation gate + fast-fade quadrant booster. Retrain the Neural ODE with **θ fed into the branch** (the fix). 490 sims → operator.
   **Status: corpus sweep dispatched; ETA ~15 h.**

---

## 4. Bugs caught in-flight — where the value came from

Every one of these would have propagated silently and either burned compute or produced a bad result. Each was caught by adversarial verification (multiple independent agents cross-checking each other), not by the primary implementer.

| # | Bug | Where | Would have cost | Detected by |
|---|---|---|---|---|
| 1 | REPT characterisation "missing" (only local disk checked) | Data inventory | 20 excluded cells | User challenge |
| 2 | CC-only vs CC-CV batch mismatch inflated CALB SoH by ~2 pp | Loader | Corrupted every CALB fit downstream | User challenge |
| 3 | PyBaMM sign convention — extractor picked charge step as "discharge" | Phase 2 loss | 100% of Phase 2 DE fits returning RMSE ≈ 9000 pp | Instrumented probe |
| 4 | Cycle-1 discharge cap inflated on 0_80 protocol → DE prefers solver-death | Phase 2 loss | 3 aged REPT fits stuck at 10.0 pp penalty | Adversarial deep-dive |
| 5 | θ column-name mismatch between corpus writer / feature reader / trainer — **θ silently zeroed in dataset** | Phase 3 pipeline | 15 h corpus + 2 h training producing the same "flat SoH" failure as the original | Adversarial audit |
| 6 | Fast-fade booster quadrant check silently returned True universally due to key-alias mismatch | Phase 3 booster | Design requirement R2 (fast-fade coverage) unenforced → operator would hallucinate flat trajectories for aged cells | Own-code re-audit while fixing #5 |
| 7 | Neural operator branch input never received θ (root cause of the original failure) | Model architecture | Would have re-shipped an operator that predicts flat SoH | Read of the file itself |

**Total compute avoided: ≈ 30–50 h of PyBaMM sweep + training runtime.**

---

## 5. Time and cost impact

### Time investment breakdown

| Activity | Hours (elapsed) | Notes |
|---|---|---|
| Data audit, cohort selection, protocol map | ~8 | Multiple iterations catching normalisation bugs |
| Phase 1 BOL identification + diagnostics | ~2 | Includes anomaly investigation on CALB 0008 |
| Phase 2 code + 13-cell DE + refits after bug fix | ~4 | ~1 h compute + 3 h engineering |
| Stage A + Stage B adversarial verification | ~2 | Two multi-agent workflows |
| Phase 3 design + implementation + adversarial audit | ~3 | 3 blockers caught before sweep |
| **Total to date** | **~19 h** | Sweep now dispatched in background |

### Cost baseline (what we did NOT do)

- Naive re-run of the previous 300-sim corpus + retrain: **~13 h compute** → identical failure. Time-to-diagnose the failure again: probably 1-2 days.
- Manual per-cell fitting without adversarial verification: probably **3-5 person-days** of debugging misfits attributed to physics rather than code.
- Re-collecting characterisation on cells where "no data" was assumed but data existed in Athena: **weeks** of lab time (this was ~10 REPT cells).

### Net vs a naive rebuild

- **Compute time saved by catching bugs early: 30–50 h**
- **Person-time saved by carrying forward provable results: probably 2–3 weeks**
- **Cohort strength: 7 verified anchors > 300 unverified samples** (previous approach)

---

## 6. What the deliverable is

- **`configs/bol_params/{make}_{cell}.yaml`** — Phase 1 BOL parameters for 13 cells; every cell reproduces its OCV within 5 mV upper-half RMSE
- **`configs/deg_params/{make}_{cell}.yaml`** — Phase 2 degradation parameters for 13 cells; every fit reproduces the saved RMSE to full float precision on independent replay
- **`configs/cohort_experiment_protocols.yaml`** — the 11 unique cycling protocols the cohort was actually run on (previous corpus used ONE protocol; 8 of 13 cells were mis-simulated)
- **`configs/phase3_sweep.yaml`** — corpus generation config with per-parameter σ, decorrelation gate, fast-fade booster
- **`configs/phase3_operator.yaml`** — operator retrain config with the θ-into-branch fix and shape+monotonicity+physics loss
- **`Voltaris/Data_Exploration/*.py + *.ipynb`** — every extractor, fitter, notebook: pure functions with smoke tests
- **7-anchor SoH overlay notebook** — proves each fitted-θ trajectory reproduces the measured Longterm trace
- **5000-cy projection notebook** — physical-plausibility check of anchor θ (in progress)
- **Adversarial verdict documents** — `_aggregate.md`, `_stage_b_verdict.md`, `phase3_readiness.md` — every claim in the paper is backed by an artefact

## 7. Current state and next steps

**Running now**: 15-h Phase 3 corpus sweep (~490 sims, 7 anchors, n_jobs=5). Checkpointed per anchor.

**When the sweep lands**:
1. Feature extraction + dataset build (~1 h)
2. Operator retrain (~2 h GPU)
3. Held-out validation on 3 cells across makes with Fisher-column cosine + regime-swap replay gates
4. If pass: draft paper + 1-page conference abstract

**Expected paper story**: per-cell PyBaMM identification breaks the single-shot characterisation cost barrier for supplier-agnostic second-life LFP RUL; corpus + operator generalises across chemistries without touching PyBaMM at inference time.

## 8. Risks worth flagging

1. **Corpus fast-fade axis is anchored by 2 REPT cells** — narrow. Held-out REPT prediction is the strongest test.
2. **SEI ↔ LAM degeneracy** shown by stress test on CALB 0003 — pure-SEI within 1.5× best RMSE. Corpus decorrelation gate + shape loss address it, but empirical Fisher-cosine < 0.3 verdict is the real proof.
3. **3 REPT aged cells fit at ~2.5 pp** — protocol-limited (partial-DoD 0_80 measured signal is 0.3–0.8 pp — dominated by noise). Not a modelling fault, but the paper needs to say so.

---

*Written 2026-07-11 after the Phase 3 sweep dispatched. Metrics and thresholds land as evidence, not aspirations — every number here is reproducible from `configs/` and the notebooks in `Voltaris/Data_Exploration/`.*
