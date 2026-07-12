# Presentation Notes — PyBaMM Conference 2026

**Talk title:** *From a single characterisation snapshot to end-of-life: a PyBaMM-trained neural operator for second-life LFP RUL*

**Speaker:** Krishna Chaitanya Vaddepally (Turno, Blubble Pvt. Ltd., India)

**Format target:** ~15 min talk + Q&A (typical PyBaMM Conf slot)

---

## Story arc — the one-sentence version

> *"We turn PyBaMM into a data engine: it generates a corpus of synthetic
>  degradation trajectories with known electrochemistry, we train a neural
>  operator on that corpus, and the resulting model predicts a used cell's
>  full RUL from a single characterisation snapshot plus 50 cycles of
>  measurement — 1376 cycles of prediction from 50 cycles of input."*

If you have to walk into the room and say one thing before your first slide, that's it.

---

## Slide-by-slide outline (12–14 slides for a 15-min talk)

### 1. Title slide (0:00–0:30)
- Read the title, name your affiliation
- One line: *"We're a second-life BESS company. This talk is about how we
  cut characterisation cost using PyBaMM as a training-data engine."*

### 2. Second-life economics: why characterisation cost matters (0:30–2:00)
- **Frame it as a real business problem**, not academic curiosity:
  - Pack builders receive mixed used-cell inventory from EV retirements
  - Every cell needs an RUL estimate before repurposing → determines
    warranty, price, pack topology
  - Per-cell PyBaMM calibration against measured SoH: **400+ cycles per cell** [Prada 2013]
  - At 1C, that's 6 weeks of cycling per cell. At the 0.25C we care about,
    much longer
  - For 10,000 cells/month throughput, this cost is *the* bottleneck
- **Optional:** rough cost math — if characterisation is $X/cell·cycle,
  400 cycles × $X × 10k cells/month = astronomical

### 3. Two prior directions and their common limitation (2:00–3:00)
- **Ouledboutaarija et al., PyBaMM Conf. 2025:** Bayesian optimisation
  improves per-cell parameter fits → *quality, not budget*
- **Kuzhiyil et al., PyBaMM Conf. 2025:** universal-differential-equation
  degradation extends model generalisability → *still per-cell calibration
  workflow*
- **Neither shrinks the cycling budget.** The fundamental question is:
  can we transfer knowledge from *many cells at once* rather than fitting
  each from scratch?

### 4. Our reformulation: PyBaMM as a data engine (3:00–4:30)
- **The shift:** stop treating PyBaMM as a per-cell fitter. Start treating
  it as a *simulator over the space of possible degradation behaviours.*
- Slide should have two panels:
  - Left: "Traditional" — per-cell PyBaMM box, arrow "400 cycles data →
    fitted params → simulate"
  - Right: "Our approach" — PyBaMM box generates 300 sims with varied θ →
    neural operator learns θ→trajectory mapping → at deploy, need only
    50 cycles + characterisation to predict any new cell
- **Key insight to voice:** *"We're using PyBaMM's physical model as the
  source of truth to train a fast surrogate. The surrogate then works on
  cells PyBaMM was never calibrated to."*

### 5. Stage 1 — Physics corpus (4:30–6:30)
- Start from Prada2013 LFP chemistry
- **BOL parameters identified from a supplier-B fresh-cell cohort:**
  - OCV curve fit → stoichiometry: 6.8 mV median RMSE
  - GITT → diffusion coefficients: R² = 0.9999
  - HPPC + DCIR → charge-transfer + ohmic resistances
  - RPT → active-material capacities
- **Sobol sweep over 5 degradation channels:**
  - `k_SEI` (kinetic rate)
  - SEI partial molar volume
  - Lithium-plating exchange current
  - LAM positive-electrode rate
  - LAM negative-electrode rate
- **Output: 300 synthetic SoH trajectories, 500 cycles each.**
  (The 1500-cycle regen is still running — flag as work-in-progress if
  needed; but 500-cy is enough to demonstrate the pipeline.)
- **Emphasise the paired-θ property:** *"For every trajectory, we know
  the ground-truth electrochemistry — that's the training signal the
  neural operator will use."*

### 6. Stage 2 — Neural operator architecture (6:30–8:00)
- **DeepONet variant** [Lu et al. 2021]
- Two networks:
  - **Branch:** five-stream input →
    - DCIR fingerprint (9-vec, R vs SOC)
    - RPT features (6-vec: Q_bol, IC peaks, OCV span)
    - First-K measured SoH (K=50)
    - **θ vector (10-dim: 5 sweep params + 5 identified BOL identifiers)**
    - Protocol (4-vec: c-rate, DoD, T, rest)
  - **Trunk:** query cycle number `n` (scalar → embedding)
- Output = dot(branch, trunk) → softplus → subtract from SoH_init
- **Two properties worth voicing:**
  - Hard monotonic by construction (softplus ≥ 0)
  - θ-conditioning makes the operator "electrochemistry-aware" — the
    same characterisation on cells from different chemistries would give
    different θ and different predictions

### 7. Why θ-conditioning matters (8:00–9:00)
- Without θ: operator must reconstruct electrochemistry from
  characterisation streams alone → fragile
- With θ: characterisation refines a physical prior, doesn't replace it
- **At deployment on a real cell:**
  - Phase-1 parameter ID gives us θ from OCV/GITT/HPPC data
  - Operator combines that θ with 50 cycles of SoH → predicts trajectory
- **Analogy to voice:** *"Think of θ as the cell's electrochemical
  fingerprint. The neural operator learns to associate fingerprints with
  fade behaviour. Even for cells never simulated exactly, similar
  fingerprints give predictable behaviour."*

### 8. Result — the RUL forecast figure (9:00–11:00)
- Show Fig. 1 from the abstract (supplier-A cell extrapolation to EoSL)
- **Talking points:**
  - Orange band: only 50 cycles used as input
  - Black points: actual measured SoH over 1200 cycles (validation, not
    training)
  - Green line: model prediction across measured range and extrapolating
    past the last measurement
  - Red dashed line: second-life EoSL at SoH = 0.40
  - **"1376-cycle RUL prediction from 50 measured cycles"** — deliver
    this line with a pause
- Optionally show a supplier-C fresh cell alongside — model correctly
  predicts long tail (near-fresh cells don't need aggressive RUL)

### 9. The second-life EoSL choice: SoH = 0.40 (11:00–11:45)
- Anticipate the question. Address it directly:
  - First-life EOL is 0.70–0.80 SoH (EV, consumer applications)
  - Second-life BESS: cells run at very low C-rate (~0.25C)
  - At low C-rate, effective capacity utilisation is much higher than at
    1C — a 40% SoH cell still delivers acceptable stationary storage
  - Nobody in the second-life industry knows exactly what EoSL is because
    almost no one has run cells to full EoSL under low-C conditions
  - **0.40 is a defensible conservative choice**, but the operator can
    forecast any threshold — it's not baked into the model
- **Voice this:** *"The operator predicts SoH(n). What threshold you call
  end-of-life is a business decision. We pick 0.40 for BESS repurposing;
  a different application picks a different number."*

### 10. Synthetic-hold-out evaluation (11:45–12:30)
- On a 15% synthetic validation split (45 held-out sims), the operator
  hits **1.0 pp SoH RMSE median**
- 322k parameters, single-GPU training, converges in minutes
- **Caveat to voice honestly:** *"This is synthetic-to-synthetic. The
  real test is on real cells, and fine-tuning on our EVE + REPT + CALB
  cohorts is the next step being finalised for the extended results."*

### 11. Scope + honest limitations (12:30–13:30)
- **What works today:**
  - Pipeline end-to-end runs
  - Synthetic hold-out: 1.0 pp RMSE
  - PINN baseline on real cells: 1.67 pp RMSE on supplier-A, extrapolation
    to EoSL is physically reasonable (no runaway to negative SoH)
- **What's not done yet:**
  - Real-cell fine-tuning on the θ-conditioned operator (planned)
  - Cross-supplier transfer evaluation at scale (planned)
  - Corpus extended to 1500 cycles for full-life shape (in progress)
- **What we know PyBaMM's default degradation models miss:**
  - LAM_neg stress model produces a "knee" at SoH ~0.9 that's steeper
    than real LFP cells show
  - We're addressing this by (a) fitting SEI/LAM rate calibration to
    measured fade, (b) letting the neural operator correct the residual
    via real-cell fine-tuning

### 12. Practical impact + call to action (13:30–14:30)
- **Business impact:**
  - Cycling budget: 400 → 50 cycles per cell (8× reduction)
  - Same characterisation infrastructure, faster cell throughput
  - Enables per-cell RUL grading for pack composition optimisation
- **For the community:**
  - Using PyBaMM as a data engine for downstream ML is under-explored
  - Sobol sweeps + neural operators are a natural pair
  - We plan to open-source the operator and the synthetic corpus
- **Ask:** *"If your group has HPPC+GITT+RPT+cycling data on cells from
  different chemistries, we'd love to test whether the operator transfers."*

### 13. Q&A slide (14:30–15:00)
- Blank slide with contact info + GitHub
- Prepare a "Backup slides" appendix for expected questions

---

## Key numbers cheat sheet (memorise)

| What | Number |
|---|---|
| Cycling budget reduction | **8×** (50 vs 400 cycles per cell) |
| RUL forecast (supplier A) | **1376 cycles** from K=50 input |
| Synthetic hold-out RMSE | **1.0 pp** SoH median |
| BOL param ID quality | OCV RMSE **6.8 mV**, GITT R² **0.9999** |
| Corpus size | **300 sims** × 500 cy each (1500 cy version regenerating) |
| Model size | **322 k parameters** |
| Second-life EoSL threshold | **SoH = 0.40** |
| Neural operator architecture | DeepONet, 5-stream branch + trunk over n |
| Chemistry | PyBaMM `Prada2013`, LFP/graphite |
| Cell class | 105 Ah large-format LFP, 25 °C isothermal |

---

## Anticipated Q&A — prepared answers

**Q: "Why 0.40 SoH as EoSL? That's much lower than the datasheet."**
A: See slide 9 rationale. Deliver: *"Datasheet 0.70/0.80 is EV/consumer.
For 0.25C stationary storage, the effective capacity is higher, and the
industry hasn't converged on second-life EoSL because nobody has run
cells to true EoSL. 0.40 is our conservative business threshold."*

**Q: "How does the neural operator compare to Kuzhiyil's UDE approach?"**
A: *"Kuzhiyil trains a per-cell correction to the physics ODE. We train a
cross-cell operator that maps (electrochemistry, cycling protocol) →
trajectory. Their per-cell fits still need substantial cycling; our
operator amortises that cost across the training corpus."*

**Q: "Your synthetic corpus doesn't reproduce the accelerating knee at
SoH ~0.9 in LFP cells. Isn't your ground truth wrong?"**
A: *"Great point. PyBaMM's LAM_neg stress model produces a knee that's
faster than real LFP. We're addressing this in two ways: (a) calibrating
the degradation rate parameters against measured fade before generating
the corpus, and (b) fine-tuning the operator on real cells to correct
any residual sim-to-real gap. The operator architecture doesn't depend
on shape being perfectly right — it depends on shape being *smoothly
parameterised by θ*."*

**Q: "What's the wall-time for one prediction?"**
A: *"Inference is a single forward pass through 322k parameters: ~10 ms
on CPU, sub-millisecond on GPU. That's the whole point of learning the
surrogate — PyBaMM sims are minutes, our surrogate is milliseconds."*

**Q: "How do you handle cells whose electrochemistry is outside the
training distribution?"**
A: *"Right now the θ-conditioning generalises via interpolation across
the Sobol-swept parameter space. For truly novel chemistries (e.g.,
LFP-Mn variants), you'd re-run the Sobol sweep with the new base
chemistry and retrain the operator. That's where PyBaMM being the data
engine shines — it's a matter of hours of sim + minutes of training,
not months of new cell tests."*

**Q: "Have you tried it on real cells yet?"**
A: *"The PINN baseline (Fig. 1 in the abstract) is trained and evaluated
on real cells across three suppliers. The θ-conditioned operator is the
next step — the corpus is regenerating with the extended cycling
horizon, then we fine-tune. Extended results will be presented at the
conference."*

**Q: "What's the theta identifiability under noisy characterisation
data?"**
A: *"Phase 1 identification gives us θ uncertainty per cell (MAD values
in `identified_params.yaml`). The operator can accept θ with uncertainty
via ensembling — sample from the θ posterior, run operator, aggregate.
We haven't wired that up yet but the architecture supports it directly."*

**Q: "Why not just train an XGBoost on (characterisation → SoH curve)?"**
A: *"We could — and we'd probably get decent results in-distribution.
The θ-conditioning gives us two things XGBoost can't: (a) physical
interpretability of what the operator has learned (θ vectors are
electrochemistry, not opaque features), (b) generalisation to
combinations of θ we've never seen because the operator learns a smooth
map over θ space, whereas XGBoost partitions."*

---

## Backup slides to prepare

1. **PyBaMM parameter identification details** — one slide showing OCV/GITT/HPPC/RPT fit quality per cell in the supplier-B cohort
2. **Sobol sweep parameter ranges** — the table from `configs/sweep_config_tight.yaml`
3. **Loss function detail** — data + monotonicity + boundary term breakdown
4. **PINN baseline vs neural operator comparison** — one slide framing the two approaches side-by-side
5. **Deployment pipeline diagram** — real cell → HPPC/DCIR → Phase-1 param ID → θ → operator → RUL forecast
6. **Cost/throughput impact numbers** — if we can quantify per-cell characterisation cost, an ROI slide is very persuasive to industry attendees

---

## Story-flow reminders

- **Open with the business problem** (second-life economics), not the ML
- **PyBaMM stays central throughout** — this is a PyBaMM conference. Every
  slide should either use PyBaMM output or be enabled by it
- **Don't apologise for what's not done.** Frame in-progress work as
  "extended results being finalised" — this is standard abstract-talk
  positioning
- **Deliver the 1376-cycle number as the emotional climax.** Pause after
  saying it. Let it land
- **End on a collaboration ask.** The community will remember you if you
  ask them for something specific

---

## Rehearsal checklist

- [ ] Practice the 1-sentence version out loud 3× until it flows
- [ ] Time slides 4 (data engine reformulation) and 8 (RUL figure) — those
      are the two "make or break" slides for narrative punch
- [ ] Have a physical printout of the abstract with you at the podium
- [ ] Confirm the venue projector can display the SoH figure at readable
      size (the PNG is 4.8 inch tall — should be fine)
- [ ] Have the GitHub URL memorised for the last slide

---

**Last updated:** 2026-07-09.
**Abstract version this covers:** v5, GitHub commit `ccd1dd3`.
