#!/usr/bin/env python3
"""
src/cell_selection.py
---------------------
Analyze available raw test files and recommend a defensible set of cells for the
PyBaMM + Neural ODE workflow.

This script creates:
- data/processed/cell_selection_report.md
- configs/dataset.yaml  (selected cells + rationale + suggested splits)

Design goal: choose a cohort that supports *validation* of the electrochemical
model and feature extraction (i.e., cells with the most complete
characterisation suite). With this dataset (25°C only), that is typically more
defensible than maximizing the number of cycling-only cells.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path("/home/hj/Desktop/PINNs")
DATA_DIR = PROJECT_ROOT / "Data"


ESSENTIAL_TESTS = [
    "OCVSOC",
    "RPT",
    "Longterm",
    "HPPC",
    "DCIR",
    "RateCapability",
    "ConstantPower",
]

OPTIONAL_TESTS = [
    "GITT",
    "PeakPower",
    "SelfDischarge",
]


def _cell_from_filename(name: str) -> str | None:
    m = re.search(r"cell_(\d+)", name)
    return m.group(1).zfill(4) if m else None


def _test_from_filename(name: str) -> str | None:
    m = re.match(r"EVE_(.*?)_cell_\d+\.csv", name)
    return m.group(1) if m else None


def _read_longterm_protocol(cell_id: str) -> dict[str, Any]:
    """
    Extract protocol-identifying fields from the longterm file without reading
    full columns unnecessarily.
    """
    p = DATA_DIR / "Longterm" / f"EVE_Longterm_cell_{cell_id}.csv"
    if not p.exists():
        return {"dod": None, "crate": None, "drate": None, "vcut_V": None, "n_cycles": None}

    df = pd.read_csv(p, usecols=["dod", "crate", "drate", "cycle_no", "current_a", "volt_v"], low_memory=False)
    dod = str(df["dod"].dropna().iloc[0]) if df["dod"].notna().any() else None
    crate = str(df["crate"].dropna().iloc[0]) if df["crate"].notna().any() else None
    drate = str(df["drate"].dropna().iloc[0]) if df["drate"].notna().any() else None
    n_cycles = int(df["cycle_no"].nunique()) if "cycle_no" in df.columns else None
    dis = df[df["current_a"] < 0]
    vcut = float(dis["volt_v"].min()) if not dis.empty else None
    return {"dod": dod, "crate": crate, "drate": drate, "vcut_V": vcut, "n_cycles": n_cycles}


def analyze_cells() -> pd.DataFrame:
    files = list(DATA_DIR.rglob("EVE_*_cell_*.csv"))
    if not files:
        raise FileNotFoundError(f"No EVE cell CSV files found under {DATA_DIR}")

    # Build availability matrix from file names
    avail: dict[str, set[str]] = {}
    for f in files:
        cell = _cell_from_filename(f.name)
        test = _test_from_filename(f.name)
        if cell is None or test is None:
            continue
        avail.setdefault(cell, set()).add(test)

    rows = []
    for cell_id in sorted(avail):
        present = avail[cell_id]
        row: dict[str, Any] = {"cell_id": cell_id}
        for t in ESSENTIAL_TESTS + OPTIONAL_TESTS:
            row[f"has_{t}"] = t in present
        row["essential_count"] = sum(row[f"has_{t}"] for t in ESSENTIAL_TESTS)
        row["optional_count"] = sum(row[f"has_{t}"] for t in OPTIONAL_TESTS)
        row["essential_ok"] = row["essential_count"] == len(ESSENTIAL_TESTS)
        row["optional_ok"] = row["optional_count"] == len(OPTIONAL_TESTS)

        proto = _read_longterm_protocol(cell_id)
        row.update(
            {
                "longterm_dod": proto["dod"],
                "longterm_crate": proto["crate"],
                "longterm_drate": proto["drate"],
                "longterm_vcut_V": proto["vcut_V"],
                "longterm_n_cycles": proto["n_cycles"],
            }
        )

        # Simple score: prioritize complete characterisation, then more longterm cycles.
        row["score"] = (
            100.0 * float(row["essential_ok"])
            + 10.0 * float(row["optional_ok"])
            + 1.0 * row["optional_count"]
            + 0.01 * float(row["longterm_n_cycles"] or 0)
        )
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["score", "cell_id"], ascending=[False, True]).reset_index(drop=True)
    return df


def recommended_cohort(df: pd.DataFrame) -> list[str]:
    """
    Default recommendation: cells with full optional suite (GITT + PeakPower +
    SelfDischarge). If none exist, fall back to essential-only cells.
    """
    full = df[df["optional_ok"] & df["essential_ok"]]["cell_id"].tolist()
    if full:
        return sorted(full)
    return sorted(df[df["essential_ok"]]["cell_id"].tolist())


def write_report(df: pd.DataFrame, selected: list[str]) -> Path:
    out = PROJECT_ROOT / "data" / "processed" / "cell_selection_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    def fmt_bool(x: bool) -> str:
        return "yes" if bool(x) else "no"

    def to_markdown_table(frame: pd.DataFrame) -> str:
        headers = list(frame.columns)
        rows = frame.astype(str).values.tolist()
        # Compute column widths
        widths = [len(h) for h in headers]
        for r in rows:
            for j, cell in enumerate(r):
                widths[j] = max(widths[j], len(cell))

        def fmt_row(r):
            return "| " + " | ".join(str(r[j]).ljust(widths[j]) for j in range(len(headers))) + " |"

        out_lines = []
        out_lines.append(fmt_row(headers))
        out_lines.append("| " + " | ".join("-" * widths[j] for j in range(len(headers))) + " |")
        for r in rows:
            out_lines.append(fmt_row(r))
        return "\n".join(out_lines)

    # A compact markdown table
    cols = [
        "cell_id",
        "essential_ok",
        "optional_ok",
        "essential_count",
        "optional_count",
        "longterm_dod",
        "longterm_crate",
        "longterm_vcut_V",
        "longterm_n_cycles",
    ]
    table = df[cols].copy()
    table["essential_ok"] = table["essential_ok"].map(fmt_bool)
    table["optional_ok"] = table["optional_ok"].map(fmt_bool)
    table["longterm_vcut_V"] = table["longterm_vcut_V"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "")

    md = []
    md.append("# Cell selection report (25°C dataset)\n")
    md.append("## Goal\n")
    md.append(
        "Choose a defensible set of cells for the PyBaMM + Neural ODE workflow. "
        "Primary criterion is **characterisation completeness** (supports parameter ID and validation). "
        "Secondary criteria include long-term cycling availability.\n"
    )

    md.append("## Selection criteria (rule-based)\n")
    md.append("- Must have essential tests: " + ", ".join(ESSENTIAL_TESTS) + ".\n")
    md.append("- Prefer cells with optional tests: " + ", ".join(OPTIONAL_TESTS) + ".\n")
    md.append(
        "- Note: Long-term cycling protocols differ across cells (DoD windows and discharge cutoffs). "
        "We therefore avoid mixing protocol-incompatible cells unless protocol features are added to the model.\n"
    )

    md.append("## Recommended cohort\n")
    md.append("- Selected cells: " + ", ".join(selected) + "\n")
    md.append(
        "- Rationale: these cells have the most complete characterisation suite, enabling "
        "more defensible electrochemical validation before using PyBaMM as a synthetic data generator.\n"
    )
    excluded = sorted([c for c in df["cell_id"].tolist() if c not in set(selected)])
    if excluded:
        md.append(
            "- Not selected by default: "
            + ", ".join(excluded)
            + " (typically missing one or more preferred optional tests such as PeakPower/SelfDischarge and/or GITT; "
              "they can still be used later once the pipeline supports protocol features and broader validation).\n"
        )

    md.append("## Audit table\n")
    md.append(to_markdown_table(table))
    md.append("\n")

    out.write_text("\n".join(md), encoding="utf-8")
    return out


def write_dataset_config(selected: list[str]) -> Path:
    out = PROJECT_ROOT / "configs" / "dataset.yaml"
    selected = sorted(selected)
    # Prefer holding out 0007 as a first evaluation cell if present (commonly the "newly imported" one).
    holdout = "0007" if "0007" in selected and len(selected) > 1 else (selected[-1] if len(selected) > 1 else None)
    train_cells = [c for c in selected if c != holdout] if holdout else selected
    val_cells = [holdout] if holdout else []
    cfg = {
        "dataset": {
            "selected_cells": selected,
            "ambient_temperature_C": 25.0,
            "selection_policy": {
                "essential_tests": ESSENTIAL_TESTS,
                "preferred_optional_tests": OPTIONAL_TESTS,
                "primary_objective": "characterisation_completeness",
            },
        },
        # Suggested starting split: hold out the newest/extra cell for evaluation if possible.
        "splits": {
            "train_cells": train_cells,
            "val_cells": val_cells,
            "test_cells": [],
        },
    }
    class _QuotedStrDumper(yaml.SafeDumper):
        pass

    def _repr_str(dumper: yaml.SafeDumper, data: str):
        # Ensure IDs like "0008" stay strings when re-parsed by YAML loaders.
        style = '"' if data.isdigit() else None
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)

    _QuotedStrDumper.add_representer(str, _repr_str)

    out.write_text(yaml.dump(cfg, Dumper=_QuotedStrDumper, sort_keys=False), encoding="utf-8")
    return out


def main() -> int:
    df = analyze_cells()
    selected = recommended_cohort(df)
    report_path = write_report(df, selected)
    config_path = write_dataset_config(selected)
    print(f"Wrote report: {report_path}")
    print(f"Wrote config: {config_path}")
    print(f"Selected cells: {selected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
