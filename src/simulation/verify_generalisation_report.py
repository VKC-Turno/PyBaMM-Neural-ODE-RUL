"""Emit `generalisation_report.md` from the per-cell inference results dicts.

Called as the last step of verify_operator_inference: writes a short
markdown containing the setup, metrics table, and interpretation.
"""
from __future__ import annotations

from pathlib import Path


def render_report(results: list[dict], operator_ckpt: str, notes: list[str]) -> str:
    lines = ["# EVE-trained θ-DeepONet vs CALB cell generalisation",
             "",
             "## Setup", ""]
    lines.append(f"- Operator checkpoint: `{operator_ckpt}`")
    lines.append("- Training corpus: 300 EVE-median-BOL sims × up to 1500 cycles, Sobol degradation sweep")
    lines.append("- K = 50 measured cycles as the operator's early-window input")
    lines.append("- Rollout horizon: 5000 cycles (past training horizon 1500 by hard-monotonic extrapolation)")
    lines.append("- Baseline: PyBaMM DFN 5000-cy long-run using each cell's own per-cell BOL + DE-fit degradation params")
    lines.append("- Reference (measured): CALB from `calb_new.parquet`, EVE from `eve.parquet` — normalised to cy1 = 1.0")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Cell | measured cy | RMSE op-DFN (pp) | RMSE op-meas (pp) | RMSE DFN-meas (pp) | cy@SoH0.80 (op) | cy@SoH0.80 (DFN) | DFN final SoH | op final SoH |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['tag']} | {r['n_measured']} | "
            f"{r['rmse_op_dfn_pp']:.2f} | "
            f"{r['rmse_op_meas_pp']:.2f} | "
            f"{r['rmse_dfn_meas_pp']:.2f} | "
            f"{r['cy_at_soh_0p80_operator'] or 'not reached'} | "
            f"{'{:.0f}'.format(r['cy_at_soh_0p80_dfn']) if r['cy_at_soh_0p80_dfn'] else 'not reached'} | "
            f"{r['dfn_final_soh']:.3f} | "
            f"{r['op_final_soh']:.3f} |"
        )
    lines.append("")
    if notes:
        lines.append("## Notes")
        lines.append("")
        lines.extend([f"- {n}" for n in notes])
        lines.append("")
    return "\n".join(lines)


def append_interpretation(md: str, interp: str) -> str:
    return md + "\n## Interpretation\n\n" + interp + "\n"
