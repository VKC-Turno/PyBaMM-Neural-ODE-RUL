"""Top-level driver: run operator inference for eve_0008 and calb_0020,
then write the combined overlay + markdown report.

Requires that both cells' bol/deg/longrun yamls+parquet already exist
(built by phases 1-3 for CALB, pre-existing for EVE).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/home/hj/Desktop/PINNs")

from src.simulation.verify_operator_inference import (
    main as run_operator_inference,
    OUT_DIR,
    CKPT,
)
from src.simulation.verify_generalisation_report import (
    render_report,
    append_interpretation,
)


def build_interpretation(results: list[dict]) -> str:
    """Write a data-driven interpretation from the metrics."""
    by_tag = {r["tag"]: r for r in results}
    lines = []

    def compare_widen(a, b):
        return "matches" if abs(a - b) < 1.0 else "widens" if a > b else "narrows"

    if "eve_0008" in by_tag and "calb_0020" in by_tag:
        e = by_tag["eve_0008"]
        c = by_tag["calb_0020"]
        lines.append(
            f"Operator-vs-DFN RMSE is **{e['rmse_op_dfn_pp']:.1f} pp** for the "
            f"in-cohort EVE 0008 and **{c['rmse_op_dfn_pp']:.1f} pp** for CALB "
            f"0020 (a cell manufacturer entirely absent from the training set). "
            f"The gap {compare_widen(c['rmse_op_dfn_pp'], e['rmse_op_dfn_pp'])} "
            f"on the CALB branch."
        )
        lines.append("")
        lines.append(
            f"Operator-vs-measured is **{e['rmse_op_meas_pp']:.1f} pp** on EVE "
            f"and **{c['rmse_op_meas_pp']:.1f} pp** on CALB; the DFN baselines "
            f"achieve **{e['rmse_dfn_meas_pp']:.1f} pp / {c['rmse_dfn_meas_pp']:.1f} pp** "
            f"on the same window because they were explicitly fit to the "
            f"first-150-cycle SoH shape."
        )
        lines.append("")
        lines.append(
            "**Failure mode when the gap widens.** The training corpus's "
            "sweep parameters live in a narrow range in log10-space (see "
            "`SyntheticTrajectoryDataset.stats['theta_vec']` — `k_SEI` "
            "std ≈ 0.49, mean ≈ -14.15). Both real cells' DE-fit "
            "parameters sit well *outside* this training range (the "
            "measured cells fade fast enough that the identified `k_SEI` "
            "is 10^-11 - 10^-12 rather than 10^-14). When θ lands "
            "> 3σ from the training mean, the branch network extrapolates "
            "into a regime where its dot-product-with-trunk output is not "
            "well-defined, and the hard-monotonic softplus decrement "
            "typically collapses to ≈ 0 → the operator predicts a flat "
            "SoH curve for the full 5000-cycle horizon. This is the "
            "regime we would need to expand by refreshing the sweep "
            "bounds to include realistic k_SEI values."
        )
    elif "eve_0008" in by_tag:
        e = by_tag["eve_0008"]
        lines.append(
            f"CALB branch failed to produce a DFN long-run (see logs). "
            f"EVE 0008 alone: op-DFN RMSE = **{e['rmse_op_dfn_pp']:.1f} pp**, "
            f"op-meas = **{e['rmse_op_meas_pp']:.1f} pp**, DFN-meas = "
            f"**{e['rmse_dfn_meas_pp']:.1f} pp**. Cannot conclude on "
            f"cross-manufacturer generalisation without a matched CALB DFN baseline."
        )
    else:
        lines.append("No cells produced results — see logs.")

    return "\n".join(lines)


def main() -> None:
    tags = ["eve_0008", "calb_0020"]
    results = run_operator_inference(tags)

    notes = [
        "CALB 0020 BOL fit needed a manual patch: OCV L-BFGS-B lower-bound-clipped "
        "y_100 to 0.0 (LFP fully delithiated); we raised it to 0.02 to avoid a "
        "degenerate PyBaMM initial state. Original fit value preserved in the yaml.",
        "θ_vec's LAM_positive_rate_s / LAM_negative_rate_s dims are not identified "
        "by the DE fit (which optimises k_SEI, V_SEI, k_plating, D_SEI_solvent); "
        "we substitute the training-corpus mean so these dims contribute the "
        "standardiser's neutral 0. Only the 3 dims that overlap between the sweep "
        "and the fit are truly per-cell.",
        "Operator was trained on trajectories up to 1500 cycles; predictions from "
        "1500-5000 rely on the network's inductive bias (hard-monotonic softplus "
        "decrement) rather than direct supervision.",
    ]

    md = render_report(results,
                        operator_ckpt=str(CKPT.relative_to(
                            Path("/home/hj/Desktop/PINNs"))),
                        notes=notes)
    interp = build_interpretation(results)
    md = append_interpretation(md, interp)

    out_md = OUT_DIR / "generalisation_report.md"
    out_md.write_text(md)
    print(f"\nWrote report: {out_md}")


if __name__ == "__main__":
    main()
