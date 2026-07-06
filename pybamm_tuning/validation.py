"""
Compare a PyBaMM simulation trajectory against measured longterm data.

Validation metrics:
  - simulated vs measured fade rate (pp / 100 cycles)
  - relative bias
  - RMSE between the two SoH curves at the measurement cycle indices
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

from .longterm import LongtermData


@dataclass
class ValidationReport:
    cell_id: str
    n_cycles_sim: int
    sim_fade_rate_pp_per_100cy: float
    measured_fade_rate_pp_per_100cy: Optional[float]
    relative_bias_pct: Optional[float]
    rmse_pp: Optional[float]
    measured_source: str
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        meas = (f"{self.measured_fade_rate_pp_per_100cy:+.4f}"
                if self.measured_fade_rate_pp_per_100cy is not None else "n/a")
        bias = (f"{self.relative_bias_pct:+.1f}%"
                if self.relative_bias_pct is not None else "n/a")
        return (f"Cell {self.cell_id}: "
                f"sim={self.sim_fade_rate_pp_per_100cy:+.4f} pp/100cy  "
                f"meas={meas} pp/100cy  bias={bias}  "
                f"(source={self.measured_source})")


def _slope_pp_per_100cy(cycle: np.ndarray, soh_pct: np.ndarray) -> float:
    if cycle.size < 2:
        return float("nan")
    slope, _ = np.polyfit(cycle.astype(float), soh_pct.astype(float), 1)
    return float(slope * 100.0)


def validate(simulation_df: pd.DataFrame,
              longterm: LongtermData,
              *,
              cycle_col: str = "cycle_n",
              soh_col: str = "SOH") -> ValidationReport:
    """
    Compare simulation SoH curve to measured longterm data.

    Parameters
    ----------
    simulation_df : per-cycle features DataFrame from Simulation.run()
    longterm : LongtermData from load_longterm()
    cycle_col, soh_col : column names in simulation_df

    The simulation_df SoH is on a 0-1 scale; we convert to % for comparison.
    """
    sim_cycle = simulation_df[cycle_col].to_numpy(dtype=float)
    sim_soh_pct = simulation_df[soh_col].to_numpy(dtype=float) * 100.0
    sim_slope = _slope_pp_per_100cy(sim_cycle, sim_soh_pct)

    meas_slope = longterm.linear_fade_rate_pp_per_100cy() if longterm.has_data else None

    rel_bias = None
    if meas_slope is not None and meas_slope != 0.0:
        rel_bias = (sim_slope - meas_slope) / abs(meas_slope) * 100.0

    rmse = None
    if longterm.has_data and longterm.soh_pct_series.size >= 2:
        # Interpolate sim curve onto measurement cycles (extrapolating linearly
        # if measurement extends beyond simulated range).
        if sim_cycle.size >= 2:
            sim_at_meas = np.interp(longterm.cycle_index, sim_cycle, sim_soh_pct,
                                     left=sim_soh_pct[0], right=sim_soh_pct[-1])
            rmse = float(np.sqrt(np.mean(
                (sim_at_meas - longterm.soh_pct_series) ** 2)))

    return ValidationReport(
        cell_id=longterm.cell_id,
        n_cycles_sim=int(sim_cycle.max()) if sim_cycle.size else 0,
        sim_fade_rate_pp_per_100cy=sim_slope,
        measured_fade_rate_pp_per_100cy=meas_slope,
        relative_bias_pct=rel_bias,
        rmse_pp=rmse,
        measured_source=longterm.source,
        notes=longterm.notes,
    )
