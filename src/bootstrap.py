#!/usr/bin/env python3
"""
src/bootstrap.py
----------------
Project bootstrap script:
- verifies data access
- generates first processed artifacts (OCV curves, capacity fade, GITT metrics)
- writes a dataset manifest for auditability
- records everything in a local experiment run folder

This is intentionally "safe": it does not run any parameter fitting, PyBaMM
simulation sweeps, or training. It only prepares defensible, data-derived
products from the raw test exports.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

# Ensure repo root is on sys.path when running as a script (python src/bootstrap.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Matplotlib cache/config directory must be writable in many sandboxed envs.
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.data_loader import (
    PROJECT_ROOT,
    TEST_FOLDER_VARIANTS,
    _find_test_dir,
    list_cells,
    load_longterm_capacity_fade,
    load_ocv_curve,
    load_rpt_capacity_fade,
    save_processed,
)
from src.experiment_tracking import ExperimentRun
from src.param_id.gitt_ds import export_gitt_metrics, extract_gitt_step_metrics


def _write_yaml(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")

def _load_selected_cells_from_dataset_config() -> list[str] | None:
    """
    If present, read `configs/dataset.yaml` and return dataset.selected_cells.
    Returns None if missing or malformed.
    """
    cfg_path = Path("configs/dataset.yaml")
    if not cfg_path.exists():
        return None
    try:
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        selected = cfg.get("dataset", {}).get("selected_cells")
        if not selected:
            return None
        return [str(c).zfill(4) for c in list(selected)]
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Bootstrap the LFP RUL PINN project.")
    ap.add_argument(
        "--run-name",
        default="bootstrap",
        help="Experiment run name (stored under outputs/experiments/).",
    )
    ap.add_argument(
        "--cells",
        nargs="*",
        default=None,
        help="Optional list of cell IDs (e.g. 0005 0006). Default: use configs/dataset.yaml if present, else all discovered cells.",
    )
    ap.add_argument(
        "--skip-longterm",
        action="store_true",
        help="Skip long-term cycling capacity-fade extraction (largest files).",
    )
    args = ap.parse_args()

    config_paths = [
        "configs/dataset.yaml",
        "configs/pybamm_base_params.yaml",
        "configs/sweep_config.yaml",
        "configs/pinn_config.yaml",
    ]
    run = ExperimentRun.start(
        name=args.run_name,
        config_paths=[p for p in config_paths if Path(p).exists()],
        tags={"stage": "bootstrap"},
    )

    # --- Dataset manifest (what exists on disk) ---
    manifest = {"project_root": str(PROJECT_ROOT), "tests": {}}
    for test in TEST_FOLDER_VARIANTS:
        try:
            d = _find_test_dir(test)
            cells = list_cells(test)
            manifest["tests"][test] = {
                "path": str(d),
                "n_cell_files": len(cells),
                "cells": cells,
            }
        except FileNotFoundError:
            manifest["tests"][test] = {"path": None, "n_cell_files": 0, "cells": []}

    out_manifest = Path("data/processed/dataset_manifest.yaml")
    _write_yaml(out_manifest, manifest)
    run.log_artifact(out_manifest)

    # --- Decide which cells to process ---
    # Use union over all discovered tests (not all cells have all tests, e.g. GITT).
    all_cells: list[str] = []
    for v in manifest["tests"].values():
        all_cells.extend(v.get("cells", []))
    all_cells = sorted(set(all_cells))

    cfg_cells = _load_selected_cells_from_dataset_config() if args.cells is None else None
    if args.cells is not None:
        cells = [str(c).zfill(4) for c in args.cells]
    elif cfg_cells:
        cells = cfg_cells
    else:
        cells = all_cells

    # Filter to only those discovered (avoid typos in config/args)
    discovered = set(all_cells)
    cells = [c for c in cells if c in discovered] if discovered else cells
    run.log_params({"cells": cells})

    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- OCV curves ---
    ocv_frames: list[pd.DataFrame] = []
    for cid in cells:
        ocv = load_ocv_curve(cid)
        if not ocv.empty:
            ocv_frames.append(ocv)
    if ocv_frames:
        ocv_all = pd.concat(ocv_frames, ignore_index=True)
        ocv_path = save_processed(ocv_all, "ocv_curves")
        run.log_artifact(ocv_path)

        fig, ax = plt.subplots(figsize=(6, 4))
        for cid, g in ocv_all.groupby("cell_id"):
            ax.plot(g["soc"], g["voltage"], label=str(cid), linewidth=1.0)
        ax.set(xlabel="SOC (estimated)", ylabel="Voltage [V]", title="OCV curves (25°C)")
        ax.legend(title="cell_id", fontsize=8)
        fig.tight_layout()
        ocv_fig = results_dir / "ocv_curves.png"
        fig.savefig(ocv_fig, dpi=150)
        plt.close(fig)
        run.log_artifact(ocv_fig)

    # --- RPT capacity fade (short, but useful baseline) ---
    rpt_frames: list[pd.DataFrame] = []
    for cid in cells:
        rpt = load_rpt_capacity_fade(cid)
        if not rpt.empty:
            rpt_frames.append(rpt)
    if rpt_frames:
        rpt_all = pd.concat(rpt_frames, ignore_index=True)
        rpt_path = save_processed(rpt_all, "rpt_capacity_fade")
        run.log_artifact(rpt_path)

        fig, ax = plt.subplots(figsize=(6, 4))
        for cid, g in rpt_all.groupby("cell_id"):
            ax.plot(g["cycle_n"], g["SOH"], marker="o", label=str(cid))
        ax.set(xlabel="RPT cycle index", ylabel="SOH (normalised to first RPT)", title="RPT SOH (25°C)")
        ax.legend(title="cell_id", fontsize=8)
        fig.tight_layout()
        rpt_fig = results_dir / "rpt_soh.png"
        fig.savefig(rpt_fig, dpi=150)
        plt.close(fig)
        run.log_artifact(rpt_fig)

    # --- Long-term cycling capacity fade (largest files) ---
    if not args.skip_longterm:
        lt_frames: list[pd.DataFrame] = []
        for cid in cells:
            lt = load_longterm_capacity_fade(cid)
            if not lt.empty:
                lt_frames.append(lt)
        if lt_frames:
            lt_all = pd.concat(lt_frames, ignore_index=True)
            lt_path = save_processed(lt_all, "longterm_capacity_fade")
            run.log_artifact(lt_path)

            fig, ax = plt.subplots(figsize=(6, 4))
            for cid, g in lt_all.groupby("cell_id"):
                ax.plot(g["cycle_n"], g["SOH"], label=str(cid), linewidth=1.5)
            ax.set(xlabel="Cycle number", ylabel="SOH (normalised to first longterm cycle)", title="Long-term SOH (25°C)")
            ax.legend(title="cell_id", fontsize=8)
            fig.tight_layout()
            lt_fig = results_dir / "longterm_soh.png"
            fig.savefig(lt_fig, dpi=150)
            plt.close(fig)
            run.log_artifact(lt_fig)

    # --- GITT step metrics (per cell; defensible baseline products) ---
    for cid in cells:
        try:
            metrics = extract_gitt_step_metrics(cell_id=cid, Q_total_Ah=None, diffusion_length_m=None)
        except Exception:
            # Not all cells necessarily have GITT; skip quietly in bootstrap.
            continue
        if metrics.empty:
            continue
        out_path = Path(f"data/processed/gitt_metrics_cell_{cid}.parquet")
        export_gitt_metrics(metrics, out_path)
        run.log_artifact(out_path)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(metrics["cycle_step"], metrics["delta_Es_V"], label=r"$\Delta E_s$ (rest) [V]")
        ax.plot(metrics["cycle_step"], metrics["delta_Etau_V"], label=r"$\Delta E_{\tau}$ (pulse) [V]", alpha=0.8)
        ax.set(xlabel="GITT step (cycle_no)", ylabel="Voltage change [V]", title=f"GITT step metrics — cell {cid}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig_path = results_dir / f"gitt_metrics_{cid}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        run.log_artifact(fig_path)

    # --- Minimal run metrics ---
    run.log_metrics(
        {
            "n_cells": len(cells),
            "n_tests_found": sum(1 for t in manifest["tests"].values() if t.get("path")),
        },
        step=0,
    )

    print(f"\nBootstrap complete. Run folder:\n  {run.run_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
