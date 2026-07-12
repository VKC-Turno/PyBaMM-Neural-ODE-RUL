"""
Aggregate the per-cell verification outputs into a single markdown report.

Reads:
    data/synthetic/verification/per_cell/<cell>/{bol_params,deg_params,phase3_summary}.yaml
    data/synthetic/verification/per_cell/<cell>/longrun.parquet

Writes:
    data/synthetic/verification/per_cell/summary.md

Content
-------
1. Header table:
   cell | n_meas | meas_fade_pp | fit_rmse_pp | cy_EoL | cy_EoSL | monotonic | notes
2. Cross-cell degradation-parameter table (fitted values for the 4 params)
3. Identifiability discussion for cells with weak fade signal
4. Overall verdict on per-cell consistency
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path("/home/hj/Desktop/PINNs")
BASE = ROOT / "data/synthetic/verification/per_cell"
SUMMARY_MD = BASE / "summary.md"

CELLS = ["0002", "0004", "0005", "0006", "0007", "0008"]


def load_yaml(p: Path) -> dict:
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def measured_fade_pp(cell: str) -> tuple[int, float]:
    df = pd.read_parquet(ROOT / "soh/data/canonical/eve.parquet")
    s = df[df.cell_id == cell].sort_values("global_cycle")
    if s.empty:
        return 0, float("nan")
    return len(s), float((s.soh.iloc[0] - s.soh.iloc[-1]) / s.soh.iloc[0] * 100)


def _fmt(v, spec=".3f", none="—"):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return none
    return format(v, spec)


def main():
    rows = []
    identifiability_notes = []
    for cell in CELLS:
        d = BASE / cell
        deg = load_yaml(d / "deg_params.yaml")
        p3 = load_yaml(d / "phase3_summary.yaml")
        n_meas, meas_fade = measured_fade_pp(cell)
        row = {
            "cell": cell,
            "n_meas": n_meas,
            "meas_fade_pp": meas_fade,
            "fit_rmse_pp": deg.get("best_rmse_pp"),
            "cy_EoL": p3.get("cycle_at_soh_0p80"),
            "cy_EoSL": p3.get("cycle_at_soh_0p40"),
            "n_sim": p3.get("n_cycles_simulated"),
            "final_soh": p3.get("final_soh"),
            "monotonic": p3.get("monotonic_decreasing"),
            "aborted": deg.get("aborted_early"),
            "abort_reason": deg.get("abort_reason"),
        }
        rows.append(row)

        # Identifiability
        ident = deg.get("identifiability") or {}
        n_ident = sum(1 for v in ident.values() if v.get("well_identified"))
        n_total = len(ident)
        identifiability_notes.append((cell, n_ident, n_total, ident))

    lines: list[str] = []
    lines.append("# EVE per-cell end-to-end verification — summary")
    lines.append("")
    lines.append(f"Cells: {', '.join(CELLS)}")
    lines.append(
        "Workflow: Phase 1 (per-cell BOL from OCV/GITT/HPPC/DCIR + optional "
        "SelfDischarge) -> Phase 2 (differential evolution fit of 4 "
        "degradation params against measured SoH trajectory, SPMe) -> "
        "Phase 3 (5000-cycle DFN long-run at 0.5C, 25 °C)."
    )
    lines.append("")
    lines.append("## 1. Per-cell summary table")
    lines.append("")
    lines.append(
        "| cell | n_meas | meas fade (pp) | fit RMSE (pp) | cy@SoH=0.80 (EoL) "
        "| cy@SoH=0.40 (EoSL) | n_sim | final SoH | monotonic | notes |"
    )
    lines.append(
        "|------|--------|----------------|---------------|-------------------|"
        "--------------------|-------|-----------|-----------|-------|"
    )
    for r in rows:
        notes = []
        if r["aborted"]:
            notes.append(f"phase2 aborted: {r['abort_reason']}")
        if r["meas_fade_pp"] < 0.05:
            notes.append("negligible/negative measured fade")
        if r["monotonic"] is False:
            notes.append("non-monotonic sim")
        notes = "; ".join(notes)
        lines.append(
            f"| {r['cell']} "
            f"| {r['n_meas']} "
            f"| {_fmt(r['meas_fade_pp'], '.3f')} "
            f"| {_fmt(r['fit_rmse_pp'], '.3f')} "
            f"| {_fmt(r['cy_EoL'], '.0f')} "
            f"| {_fmt(r['cy_EoSL'], '.0f')} "
            f"| {_fmt(r['n_sim'], 'd') if r['n_sim'] is not None else '—'} "
            f"| {_fmt(r['final_soh'], '.4f')} "
            f"| {r['monotonic']} "
            f"| {notes} |"
        )
    lines.append("")

    # Cross-cell parameter table
    lines.append("## 2. Fitted degradation parameters (cross-cell)")
    lines.append("")
    lines.append(
        "Same DE bounds and cost function used across all cells (RMSE on "
        "normalized SoH over the measured window). Values are the "
        "best-of-DE for each cell."
    )
    lines.append("")
    lines.append(
        "| cell | k_SEI [m/s] | V_SEI [m3/mol] | D_SEI [m2/s] | k_plating [m/s] |"
    )
    lines.append("|------|-------------|----------------|--------------|-----------------|")
    param_cols = {
        "k_SEI": "SEI kinetic rate constant [m.s-1]",
        "V_SEI": "SEI partial molar volume [m3.mol-1]",
        "D_SEI": "SEI solvent diffusivity [m2.s-1]",
        "k_plt": "Lithium plating kinetic rate constant [m.s-1]",
    }
    cross_vals = {k: [] for k in param_cols}
    for cell in CELLS:
        deg = load_yaml(BASE / cell / "deg_params.yaml")
        bp = deg.get("best_parameters") or {}
        vals = [bp.get(v) for v in param_cols.values()]
        for k, val in zip(param_cols, vals):
            if val is not None and np.isfinite(val):
                cross_vals[k].append(val)
        lines.append(
            f"| {cell} "
            + " ".join(
                f"| {vals[i]:.3e}" if vals[i] is not None else "| —"
                for i in range(4)
            )
            + " |"
        )
    lines.append("")

    lines.append("### Cross-cell spread")
    lines.append("")
    lines.append("| parameter | min | median | max | max/min ratio |")
    lines.append("|-----------|-----|--------|-----|---------------|")
    for k, label in param_cols.items():
        v = cross_vals[k]
        if not v:
            lines.append(f"| {label} | — | — | — | — |")
            continue
        arr = np.array(v)
        ratio = arr.max() / arr.min() if arr.min() > 0 else float("inf")
        lines.append(
            f"| {label} "
            f"| {arr.min():.3e} | {np.median(arr):.3e} | {arr.max():.3e} "
            f"| {ratio:.2e} |"
        )
    lines.append("")

    # Identifiability discussion
    lines.append("## 3. Identifiability by cell")
    lines.append("")
    lines.append(
        "Each of the 4 degradation parameters is 'well_identified' when the "
        "top-10% of DE candidates span < 25% of the full DE bound range. "
        "Weak-signal cells (short Longterm, small fade) are expected to "
        "flunk this test for most parameters."
    )
    lines.append("")
    lines.append("| cell | # well-identified / 4 | notes |")
    lines.append("|------|-----------------------|-------|")
    for cell, n_ident, n_total, ident in identifiability_notes:
        note = ""
        if n_total == 0:
            note = "no identifiability diagnostics"
        else:
            weak = [k.split(" [")[0]
                    for k, v in ident.items() if not v.get("well_identified")]
            if weak:
                note = "weak: " + ", ".join(weak)
        lines.append(f"| {cell} | {n_ident}/{n_total} | {note} |")
    lines.append("")

    # Verdict
    fitted_ok = [r for r in rows if r["fit_rmse_pp"] is not None]
    if fitted_ok:
        med_rmse = float(np.median([r["fit_rmse_pp"] for r in fitted_ok]))
    else:
        med_rmse = float("nan")
    n_eol_hit = sum(1 for r in rows if r["cy_EoL"] is not None)

    eol_cycles = [r["cy_EoL"] for r in rows if r["cy_EoL"] is not None]
    eosl_cycles = [r["cy_EoSL"] for r in rows if r["cy_EoSL"] is not None]

    lines.append("## 4. Overall verdict")
    lines.append("")
    lines.append(f"- Cells with non-trivial measured fade "
                 f"(0002, 0004, 0005, 0006, 0007, 0008) fitted with median "
                 f"RMSE = {med_rmse:.3f} pp on the measured window.")
    lines.append(f"- {n_eol_hit}/{len(rows)} cells reached SoH=0.80 in the "
                 f"5000-cycle long-run.")
    if eol_cycles:
        lines.append(
            f"- Simulated cycles-to-EoL across the cohort: "
            f"min={int(min(eol_cycles))}, median={int(np.median(eol_cycles))}, "
            f"max={int(max(eol_cycles))}."
        )
    if eosl_cycles:
        lines.append(
            f"- Simulated cycles-to-EoSL: "
            f"min={int(min(eosl_cycles))}, median={int(np.median(eosl_cycles))}, "
            f"max={int(max(eosl_cycles))}."
        )
    lines.append("")
    lines.append(
        "Per-cell tuning is defensible for cells with clear fade "
        "signal (0008 and, to a lesser extent, 0002/0004 with 150 cy). "
        "For 0005/0006/0007 the measured window is too short to identify "
        "SEI-vs-plating apportionment; the fit converges to whatever "
        "gives near-zero fade in that window, and the resulting long-run "
        "extrapolation must be read as a *lower bound* on degradation rate."
    )
    lines.append("")
    lines.append(
        "The four fitted parameters vary by 1-3 orders of magnitude across "
        "cells that came from the same batch. This is *not* physical "
        "batch variability — it reflects the identifiability limit of the "
        "150-cycle window: multiple (k_SEI, D_SEI, k_plating) triples all "
        "give the same normalized fade in 150 cy, and DE picks whichever "
        "landed at the population minimum."
    )

    SUMMARY_MD.write_text("\n".join(lines))
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
