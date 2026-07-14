# V9 · Careful re-computation of characterisation test results

Motivation (from 01_2e/01_2g audit and `data/processed/param_id_report.md`):

- OCV fit RMSE 6.79 mV (target < 5 mV) — **not met**
- DCIR cross-cell variance ~50%, cell 0006 flagged as outlier — **not met, not cleaned**
- HPPC only probed SOC 0.97–1.00 — DCIR at other SOC states not identified
- No systematic outlier detection, replicate consistency, or physical-bounds
  validation in the existing pipeline
- x_health feature and θ prior at inference inherit these issues

**Scope**: rigorously recompute per-cell parameters for the 6 core tests
across the 40 fully-covered cells (19 CALB + 4 EVE + 17 REPT), with:

1. Time-series outlier detection per raw sample
2. Multi-SOC breakpoints where the protocol allows
3. Cross-pulse consistency checks per cell
4. Cohort-level validation per supplier + batch
5. Cross-check against the existing `data/processed/*.parquet` outputs
6. Uncertainty estimates per cell

**Notebooks** (executed in order):

- `01_dcir_recompute.ipynb` — DCIR / R0 (priority: feeds the operator's
  x_health feature; existing 50% cross-cell variance under investigation)
- `02_hppc_recompute.ipynb` — HPPC R1, C1, tau (RC parameters)
- `03_gitt_recompute.ipynb` — GITT diffusivity Ds
- `04_ocv_recompute.ipynb` — OCV / stoichiometry
- `05_rpt_recompute.ipynb` — RPT capacity
- `06_selfdischarge_recompute.ipynb` — SEI ceiling

Each notebook writes `data/processed/v9_<test>_summary.parquet` alongside
the existing (uncleaned) v0 outputs so downstream code can switch by path.
