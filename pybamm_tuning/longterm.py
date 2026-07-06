"""
Load longterm cycling data for simulation validation.

Two data sources:
  1. characterisation workbook MFR_C sheet — batch 1 + batch 2 measurements
     give an actual cell-cohort fade rate between RPT checkpoints.
  2. data/raw/Longterm/  — per-cell continuous cycling files, when available.

Returns a LongtermData object with cycle-indexed SoH points the
validation module can compare against a PyBaMM trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


DEFAULT_CHAR_PATH = Path("data/char_consolidated.xlsx")
LONGTERM_DIR = Path("data/raw/Longterm")


@dataclass(frozen=True)
class LongtermData:
    cell_id: str
    cohort: str
    source: str         # "RPT_batch_pairs" | "longterm_csv"
    cycles_per_batch: int  # nominal cycles between batch 1 and batch 2
    soh_pct_series: np.ndarray   # measured SoH at each cycle index
    cycle_index:    np.ndarray   # cycles at which each SoH was measured
    notes: str = ""

    @property
    def has_data(self) -> bool:
        return self.soh_pct_series.size > 0

    def linear_fade_rate_pp_per_100cy(self) -> Optional[float]:
        """Slope of SoH vs cycle, in pp per 100 cycles."""
        if self.soh_pct_series.size < 2:
            return None
        # Linear regression
        x = self.cycle_index.astype(float)
        y = self.soh_pct_series.astype(float)
        slope, _ = np.polyfit(x, y, 1)
        return float(slope * 100.0)


def _extract_cell_pair_from_mfr_c(df: pd.DataFrame,
                                  cell_id: str) -> Optional[LongtermData]:
    """If a cell has both batch 1 and batch 2 in MFR_C, build a 2-point series."""
    sub = df[df["cell_id"].astype(str) == cell_id]
    if "batch" not in sub.columns or sub["batch"].nunique() < 2:
        return None
    b1 = sub[sub["batch"] == 1]["Soh"].astype(float).iloc[0]
    b2 = sub[sub["batch"] == 2]["Soh"].astype(float).iloc[0]
    cohort = str(sub["cohort"].iloc[0])
    # Default assumption: batches separated by a known cycle count.
    # In practice this comes from the cycler logs — fallback constant here.
    cycles_per_batch = int(sub.attrs.get("cycles_per_batch", 600))
    return LongtermData(
        cell_id=cell_id,
        cohort=cohort,
        source="RPT_batch_pairs",
        cycles_per_batch=cycles_per_batch,
        soh_pct_series=np.array([b1, b2], dtype=float),
        cycle_index=np.array([0, cycles_per_batch], dtype=float),
        notes="2-point linear approximation from RPT batch 1 -> 2",
    )


def load_longterm(
    cell_id: str,
    *,
    cohort: Optional[str] = None,
    path: Union[str, Path] = DEFAULT_CHAR_PATH,
    cycles_per_batch: int = 600,
    longterm_csv_dir: Optional[Path] = None,
) -> LongtermData:
    """
    Load longterm validation data for a cell or module.

    Tries in order:
      1. RPT batch-1 → batch-2 pair from characterisation workbook.MFR_C
      2. Continuous cycling CSV from data/raw/Longterm/<cell_id>/

    Returns a LongtermData with at least 2 points if any source matches.
    """
    path = Path(path)
    try:
        df = pd.read_excel(path, sheet_name="MFR_C")
        df.attrs["cycles_per_batch"] = cycles_per_batch
        result = _extract_cell_pair_from_mfr_c(df, cell_id)
        if result is not None:
            return result
    except Exception:
        pass

    # Fall back to CSV directory
    csv_dir = Path(longterm_csv_dir) if longterm_csv_dir else LONGTERM_DIR
    candidates = list(csv_dir.glob(f"**/*{cell_id}*.csv")) if csv_dir.exists() else []
    if candidates:
        df = pd.read_csv(candidates[0])
        if {"cycle", "soh"}.issubset(df.columns):
            return LongtermData(
                cell_id=cell_id, cohort=cohort or "?",
                source="longterm_csv", cycles_per_batch=int(df["cycle"].max()),
                soh_pct_series=df["soh"].values * 100.0,
                cycle_index=df["cycle"].values.astype(float),
                notes=f"Loaded from {candidates[0]}",
            )

    return LongtermData(
        cell_id=cell_id, cohort=cohort or "?",
        source="none", cycles_per_batch=0,
        soh_pct_series=np.array([]), cycle_index=np.array([]),
        notes="No longterm data found",
    )


def compute_actual_fade_rate(
    cohort: str = "MFR_C",
    manufacturer: Optional[str] = None,
    *,
    cycles_per_batch: int = 600,
    path: Union[str, Path] = DEFAULT_CHAR_PATH,
) -> dict:
    """
    Aggregate actual MFR_C batch 1 -> batch 2 fade rate across all paired cells.
    Returns the mean/median fade in pp/100cy with cohort statistics.
    """
    df = pd.read_excel(path, sheet_name="MFR_C")
    if manufacturer:
        df = df[df["manufacturer"] == manufacturer]
    fade_rates = []
    for cid in df["cell_id"].unique():
        sub = df[df["cell_id"] == cid]
        if sub["batch"].nunique() < 2:
            continue
        b1 = sub[sub["batch"] == 1]["Soh"].astype(float).iloc[0]
        b2 = sub[sub["batch"] == 2]["Soh"].astype(float).iloc[0]
        fade_rates.append((b2 - b1) / cycles_per_batch * 100.0)  # pp per 100cy
    if not fade_rates:
        return {"n_cells_paired": 0}
    arr = np.array(fade_rates)
    return {
        "n_cells_paired":    int(arr.size),
        "mean_pp_per_100cy": float(arr.mean()),
        "median_pp_per_100cy": float(np.median(arr)),
        "std_pp_per_100cy":  float(arr.std()),
        "min_pp_per_100cy":  float(arr.min()),
        "max_pp_per_100cy":  float(arr.max()),
        "cycles_per_batch":  cycles_per_batch,
    }
