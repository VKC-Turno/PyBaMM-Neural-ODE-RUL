# LFP RUL PINN — Project Instructions for Claude Code

## Project root
```
/home/hj/Desktop/PINNs/
```

## What this project does
Physics-Informed Neural Network for Remaining Useful Life (RUL) prediction
of used LFP (Lithium Iron Phosphate) cells. PyBAMM is used as the physics
proxy — generating synthetic degradation trajectories that train the PINN.
Real characterisation data is used for parameter identification and fine-tuning.

## Experimental condition (this dataset)
- Ambient temperature: **25°C** for all tests (treat as isothermal)
- Cell class: large-format LFP (~105 Ah)

## Actual data location
Your raw test data lives here (do not move it):
```
/home/hj/Desktop/PINNs/Data/
  ├── ConstantPower/
  ├── DCIR/
  ├── GITT/
  ├── HPPC/
  ├── Longterm/
  ├── OCVSOC/          ← note: loader handles OCVSOC ↔ OCV_SOC name difference
  ├── PeakPower/
  ├── RateCapability/
  ├── RPT/
  └── SelfDischarge/
```

`setup_project.py` creates symlinks at `data/raw/<StandardName>` → `Data/<ActualName>`
so all agents can use canonical names without touching your originals.

## Full project structure (after setup_project.py is run)
```
/home/hj/Desktop/PINNs/
├── CLAUDE.md                      ← you are here (Claude Code reads this first)
├── setup_project.py               ← run once to scaffold folders + symlinks
├── requirements.txt
├── agents/
│   ├── AGENT_PARAM_ID.md          ← Phase 1: parameter identification
│   ├── AGENT_SIMULATION.md        ← Phase 2: PyBAMM sweep
│   ├── AGENT_PINN.md              ← Phase 3: PINN training
│   └── AGENT_INFERENCE.md         ← Phase 4: RUL inference
├── configs/
│   ├── pybamm_base_params.yaml
│   ├── sweep_config.yaml
│   └── pinn_config.yaml
├── src/
│   ├── data_loader.py             ← shared; reads from Data/ directly
│   ├── param_id/
│   │   ├── ocv_fit.py
│   │   ├── gitt_ds.py
│   │   ├── dcir_hppc.py
│   │   └── sei_selfdisc.py
│   ├── simulation/
│   │   ├── run_sweep.py
│   │   ├── extract_features.py
│   │   └── validate_pybamm.py
│   ├── pinn/
│   │   ├── model.py
│   │   ├── loss.py
│   │   ├── train.py
│   │   └── dataset.py
│   └── inference/
│       ├── predict_rul.py
│       ├── health_features.py
│       └── update.py
├── Data/                          ← YOUR ORIGINAL DATA — never modified
├── data/
│   ├── raw/                       ← symlinks → Data/ subfolders
│   ├── processed/                 ← cleaned outputs from param_id
│   └── synthetic/                 ← PyBAMM simulation outputs
├── outputs/
│   ├── models/                    ← PINN checkpoints (.pt files)
│   ├── results/                   ← plots, metrics, RUL reports
│   └── logs/
└── tests/
```

## Pipeline — run phases in this order

### Phase 1 — Parameter identification  [AGENT_PARAM_ID]
Reads: `Data/OCVSOC/`, `Data/GITT/`, `Data/DCIR/`, `Data/HPPC/`, `Data/SelfDischarge/`
Writes: `configs/identified_params.yaml`, `data/processed/param_id_report.md`
Can run in parallel with Phase 2.

### Phase 2 — PyBAMM simulation sweep  [AGENT_SIMULATION]
Reads: `configs/identified_params.yaml` (uses pybamm_base_params.yaml as fallback)
Writes: `data/synthetic/trajectories.parquet`, `data/synthetic/ic_curves/`
Can run in parallel with Phase 1.

### Phase 3 — PINN training            [AGENT_PINN]
Reads: `data/synthetic/`, `Data/RPT/`, `Data/Longterm/`
Writes: `outputs/models/pinn_pretrained.pt`, `outputs/models/pinn_finetuned.pt`
Must wait for Phase 2.

### Phase 4 — RUL inference            [AGENT_INFERENCE]
Reads: `outputs/models/pinn_finetuned.pt` + any new characterisation snapshot
Writes: RUL report JSON + SOH trajectory plot

## How to spawn sub-agents in Claude Code
Open a new task panel (Cmd/Ctrl+Shift+P → "Claude: New Task") and paste
the contents of the relevant AGENT_*.md file. Agents 1 and 2 can run
simultaneously in separate task panels.

## Key design decisions

### Neural ODE not vanilla PINN
LFP degradation is governed by coupled ODEs (not PDEs). Neural ODE learns
`dSOH/dn = f(SOH, n, T, x_health)` — simpler, more stable, better suited
to sparse real data.

### Loss function
```
L = L_data + λ_phys * L_physics + λ_mono * L_monotonicity
```
λ_phys = 0.1 during pre-training, 0.3 during fine-tuning (trust physics more
when real data is scarce).

### EOL threshold
SOH < 0.80 → end of life → RUL = n_EOL − n_now

### Data format
All processed/synthetic data: `.parquet`
Columns: `cell_id, cycle_n, temperature_C, c_rate, Q_Ah, SOH, dcir_mOhm,
          ic_peak1_V, ic_peak1_area, ic_peak2_V, ic_peak2_area`

## Environment setup
```bash
cd /home/hj/Desktop/PINNs
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python setup_project.py
.venv/bin/python src/data_loader.py    # sanity check — confirms all Data/ folders found
```

## Experiment tracking (reproducibility)
This repo uses lightweight local run folders (no external service required):
- Tracker: `src/experiment_tracking.py`
- Output: `outputs/experiments/<run_id>/` (meta + config snapshots + metrics + artifacts)

Smoke test:
```bash
.venv/bin/python src/experiment_tracking.py
```

## Manuscript (LaTeX)
Living publication draft lives in `paper/`:
```bash
cd paper
latexmk -pdf main.tex
```

## Bootstrap (first defensible artifacts)
Generate initial processed datasets + plots (OCV, SOH, GITT metrics):
```bash
.venv/bin/python src/bootstrap.py
```

## Cell selection (recommended cohort)
After importing additional cells, generate an auditable selection report + config:
```bash
.venv/bin/python src/cell_selection.py
```
This writes:
- `data/processed/cell_selection_report.md`
- `configs/dataset.yaml` (used as the default cell list by `src/bootstrap.py`)

## Notebook (see what’s happening)
Open `notebooks/01_project_overview.ipynb` for an interactive view of data, artifacts, and quick PyBaMM sanity checks.

## Code conventions
- All paths via `pathlib.Path`, never string concatenation
- Physical units: SI inside PyBAMM (A, V, m, s, K); Celsius and **Ah** in this dataset’s CSV exports
- Temperature: Kelvin inside PyBAMM, Celsius everywhere else
- Random seeds: param_id=42, sweep=123, pinn=456
- Never hardcode cell IDs — always discover from `list_cells(test_name)`
