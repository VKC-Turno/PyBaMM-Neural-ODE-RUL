#!/usr/bin/env python
"""Summarise a Voltaris tuning cohort sweep.

After AGENT_VOLTARIS_TUNING has run on every cell in a cohort, this script
walks the matching `_aging_calibrated.json` files and produces:

  - A flat per-cell table (`cohort_summary_table.csv`)
  - A markdown report with outlier flags (`cohort_summary.md`)
  - A two-panel PNG plot (D_SEI distribution + residual vs SoH) saved next
    to the JSON files.

Usage (from repo root):

    .venv/bin/python Voltaris/scripts/cohort_summary.py \\
        --cohort EVE \\
        --cells-glob 'Voltaris/outputs/tuned_params/EVE_*_aging_calibrated.json'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_cell_summaries(cells_glob: str) -> pd.DataFrame:
    """Walk JSON files matching the glob and build a per-cell dataframe."""
    rows = []
    for path in sorted(Path(".").glob(cells_glob)):
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            print(f"  ! skipping {path.name}: {e}", file=sys.stderr)
            continue
        rows.append({
            "cell": data.get("cell"),
            "cohort": data.get("cohort"),
            "batch": data.get("batch"),
            # SoH key differs between cohorts:
            #   EVE (single batch): soh_pct          (workbook char SoH)
            #   REPT (two batches): soh_b1_pct       (workbook b1 SoH)
            "soh_pct_init": (data.get("soh_pct")
                              or data.get("soh_b1_pct")),
            "D_SEI_m2_s": data.get("calibrated_value"),
            # Target-slope key differs similarly
            "target_slope": (data.get("measured_target_pp_per_100cy")
                              or data.get("target_slope_pp_per_100cy")),
            "achieved_slope": (data.get("achieved_pp_per_100cy")
                               or data.get("achieved_slope_pp_per_100cy")),
            "residual_pp_100cy": data.get("residual_pp_per_100cy"),
            "rel_err_pct": data.get("relative_error_pct"),
            "n_evals": data.get("n_evaluations"),
            "n_fresh_sims": (data.get("n_fresh_sims")
                             # Older runs that pre-date the new field
                             or data.get("calibration_result", {}).get("n_fresh_sims")),
            "classification": data.get("classification"),
            "gates_tripped": ",".join(
                sorted(k for k, v in (data.get("gate_audit") or {}).items()
                       if isinstance(v, dict) and v.get("tripped"))
            ),
            "fallbacks": ",".join(data.get("fallback_strategies_invoked") or []),
            "json_path": str(path),
        })
    return pd.DataFrame(rows)


def flag_outliers(df: pd.DataFrame, col: str, k: float = 2.0) -> pd.Series:
    """Flag values more than k median-absolute-deviations from the median."""
    if df[col].dropna().empty:
        return pd.Series([False] * len(df), index=df.index)
    med = df[col].median()
    mad = (df[col] - med).abs().median() or 1e-12
    return (df[col] - med).abs() > k * 1.4826 * mad


def plot_summary(df: pd.DataFrame, out_path: Path) -> None:
    """Two panels: D_SEI distribution (log-scaled) + residual vs SoH."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.2))

    # ── (a) D_SEI distribution ──
    if df["D_SEI_m2_s"].dropna().empty:
        ax_a.text(0.5, 0.5, "no D_SEI values", ha="center", va="center",
                  transform=ax_a.transAxes)
    else:
        log_d = np.log10(df["D_SEI_m2_s"].astype(float))
        ax_a.scatter(df["cell"], log_d, color="#1f77b4", zorder=3)
        for _, row in df.iterrows():
            ax_a.annotate(row["classification"] or "?",
                          (row["cell"], np.log10(row["D_SEI_m2_s"])),
                          xytext=(4, 4), textcoords="offset points", fontsize=8)
        med = log_d.median()
        ax_a.axhline(med, ls="--", color="grey", alpha=0.6, label=f"median 1e{med:.2f}")
        ax_a.set_ylabel("log10(D_SEI / m² s⁻¹)")
        ax_a.set_xlabel("Cell")
        ax_a.set_title("Calibrated D_SEI per cell")
        ax_a.legend(loc="best", fontsize=8)
        ax_a.grid(alpha=0.3)

    # ── (b) residual vs SoH_init ──
    if df["soh_pct_init"].dropna().empty:
        ax_b.text(0.5, 0.5, "no SoH values", ha="center", va="center",
                  transform=ax_b.transAxes)
    else:
        ax_b.scatter(df["soh_pct_init"], df["rel_err_pct"], color="#d62728", zorder=3)
        for _, row in df.iterrows():
            ax_b.annotate(row["cell"],
                          (row["soh_pct_init"], row["rel_err_pct"]),
                          xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax_b.axhline(25, ls="--", color="green", alpha=0.4, label="25 % (GOOD)")
        ax_b.axhline(50, ls="--", color="orange", alpha=0.4, label="50 % (FAIR)")
        ax_b.set_xlabel("SoH at char (%)")
        ax_b.set_ylabel("relative error (%)")
        ax_b.set_title("Calibration residual vs initial SoH")
        ax_b.legend(loc="best", fontsize=8)
        ax_b.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, help="Cohort label for the report header.")
    ap.add_argument("--cells-glob", required=True,
                     help="Glob pattern matching <cohort>_*_aging_calibrated.json files.")
    ap.add_argument("--out-dir", type=Path,
                     default=Path("Voltaris/outputs/tuned_params"))
    args = ap.parse_args()

    df = load_cell_summaries(args.cells_glob)
    if df.empty:
        print(f"No JSON files matched {args.cells_glob!r}", file=sys.stderr)
        return 1

    df["D_SEI_outlier_2sigma"] = flag_outliers(df, "D_SEI_m2_s")
    df["residual_outlier_2sigma"] = flag_outliers(df, "rel_err_pct")

    csv_path = args.out_dir / f"{args.cohort}_cohort_summary.csv"
    md_path  = args.out_dir / f"{args.cohort}_cohort_summary.md"
    png_path = args.out_dir / f"{args.cohort}_cohort_summary.png"

    df.to_csv(csv_path, index=False)
    plot_summary(df, png_path)

    # ── markdown ──
    n_good = int((df["classification"] == "GOOD").sum())
    n_fair = int((df["classification"] == "FAIR").sum())
    n_poor = int((df["classification"] == "POOR").sum())
    outliers = df[df["D_SEI_outlier_2sigma"] | df["residual_outlier_2sigma"]]

    # `to_markdown` needs the `tabulate` package (optional dep). Skip it if
    # missing and fall back to a CSV-style table — keeps the report readable
    # without a new dependency.
    def _md_table(d: pd.DataFrame) -> str:
        try:
            return d.to_markdown(index=False, floatfmt=".4g")
        except ImportError:
            return d.to_csv(index=False, float_format="%.4g")

    md = dedent(f"""\
        # {args.cohort} cohort — Voltaris tuning summary

        **Cells calibrated**: {len(df)}
        **Classification mix**: {n_good} GOOD · {n_fair} FAIR · {n_poor} POOR
        **D_SEI median**: `{df["D_SEI_m2_s"].median():.3e} m²/s`
        **Relative error median**: `{df["rel_err_pct"].median():.2f} %`
        **Total fresh PyBaMM sims**: {int(df["n_fresh_sims"].fillna(0).sum())}

        ## Per-cell table

        ```
        {_md_table(df[['cell','classification','D_SEI_m2_s','rel_err_pct',
             'soh_pct_init','gates_tripped','fallbacks','n_fresh_sims']])}
        ```

        ## Outliers (>2σ MAD from cohort median)

        ```
        {_md_table(outliers[['cell','classification','D_SEI_m2_s','rel_err_pct',
                   'gates_tripped']]) if not outliers.empty else "(none)"}
        ```

        ## Files
        - Per-cell calibrated JSON / report / PNG: `Voltaris/outputs/tuned_params/{args.cohort}_*`
        - Cohort summary CSV: `{csv_path.name}`
        - Cohort summary plot: `{png_path.name}`
        """)
    md_path.write_text(md)

    print(f"Wrote:\n  {csv_path}\n  {md_path}\n  {png_path}")
    print(f"\n  {n_good}/{len(df)} GOOD, {n_fair} FAIR, {n_poor} POOR")
    print(f"  D_SEI median: {df['D_SEI_m2_s'].median():.3e} m²/s")
    if not outliers.empty:
        print(f"  Outliers: {', '.join(outliers['cell'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
