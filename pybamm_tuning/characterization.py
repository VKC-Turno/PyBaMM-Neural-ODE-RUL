"""
Characterization data loader for the PyBaMM tuning package.

The characterisation workbook workbook stores array-valued columns as bracketed
strings; this module parses them back to numpy arrays and exposes a clean
Characterization dataclass that downstream modules can consume.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


DEFAULT_CHAR_PATH = Path("data/char_consolidated.xlsx")
KNOWN_SHEETS = ("MFR_B_A", "MFR_C")


def _parse_array(x) -> np.ndarray:
    """Convert a characterisation workbook cell value into a numpy array of floats."""
    if isinstance(x, np.ndarray):
        return x.astype(float)
    if isinstance(x, (list, tuple)):
        return np.array(x, dtype=float)
    if pd.isna(x):
        return np.array([], dtype=float)
    s = re.sub(r"\s+", " ", str(x)).replace("[", "").replace("]", "").strip()
    parts = [v for v in s.replace(",", " ").split() if v]
    if not parts:
        return np.array([], dtype=float)
    try:
        return np.array([float(v) for v in parts], dtype=float)
    except ValueError:
        return np.array([], dtype=float)


@dataclass(frozen=True)
class Characterization:
    """
    Immutable container for one cell's (or aggregated cohort's) characterization.

    All arrays are 1-D numpy arrays paired with a SoC grid. Resistances are
    in mΩ (matching the workbook); voltages in V; capacities in Ah.
    """
    cell_id:          str
    cohort:           str
    manufacturer:     str
    batch:            int
    nominal_capacity_ah: float
    q_rpt_ah:         float            # measured discharge capacity at batch
    soh_pct:          float            # q_rpt / nominal × 100

    # OCV(SoC) — measured by slow C/25 (or similar) charge–discharge
    ocv_soc_grid:     np.ndarray
    ocv_v_curve:      np.ndarray
    ocv_soc_grid_chg: np.ndarray = field(default_factory=lambda: np.array([]))
    ocv_v_chg_curve:  np.ndarray = field(default_factory=lambda: np.array([]))

    # Charge-side RPT + rate capability counterparts
    q_rpt_chg_ah:        float = float("nan")
    rate_cap_c_rates_chg: np.ndarray = field(default_factory=lambda: np.array([]))
    rate_cap_q_chg_curve: np.ndarray = field(default_factory=lambda: np.array([]))

    # DCIR (3 SoC anchors): R0 from short pulse, full SoC range coverage
    dcir_soc_grid:    np.ndarray = field(default_factory=lambda: np.array([]))
    dcir_r0_mohm:     np.ndarray = field(default_factory=lambda: np.array([]))

    # HPPC pulses (typically narrower SoC range, more pulses)
    hppc_soc_grid:    np.ndarray = field(default_factory=lambda: np.array([]))
    hppc_r0_mohm:     np.ndarray = field(default_factory=lambda: np.array([]))
    hppc_r1_mohm:     np.ndarray = field(default_factory=lambda: np.array([]))
    hppc_c1_F:        np.ndarray = field(default_factory=lambda: np.array([]))
    hppc_r2_mohm:     np.ndarray = field(default_factory=lambda: np.array([]))
    hppc_c2_F:        np.ndarray = field(default_factory=lambda: np.array([]))

    # GITT (long-pulse + rest)
    gitt_soc_grid:    np.ndarray = field(default_factory=lambda: np.array([]))
    gitt_r_pulse_mohm: np.ndarray = field(default_factory=lambda: np.array([]))
    gitt_tau_diff_s:  np.ndarray = field(default_factory=lambda: np.array([]))
    gitt_v_inf_V:     np.ndarray = field(default_factory=lambda: np.array([]))

    # Module-specific: topology (1 for single cells)
    n_series_in_module:   int = 1
    n_parallel_in_module: int = 1

    @property
    def is_module(self) -> bool:
        return (self.n_series_in_module > 1) or (self.n_parallel_in_module > 1)

    def per_cell_q_rpt_ah(self) -> float:
        """Module Q_RPT divided by parallel count (= per-cell capacity)."""
        return self.q_rpt_ah / self.n_parallel_in_module

    def per_cell_ocv(self) -> tuple[np.ndarray, np.ndarray]:
        """Module OCV divided by series count (= per-cell OCV curve)."""
        return self.ocv_soc_grid, self.ocv_v_curve / self.n_series_in_module

    def per_cell_dcir_r0_mohm(self) -> tuple[np.ndarray, np.ndarray]:
        """Module pack R0 collapsed to per-cell: divide by (N_series/N_parallel)."""
        if self.dcir_r0_mohm.size == 0:
            return self.dcir_soc_grid, self.dcir_r0_mohm
        factor = self.n_series_in_module / self.n_parallel_in_module
        return self.dcir_soc_grid, self.dcir_r0_mohm / factor

    def per_cell_hppc_r0_mohm(self) -> tuple[np.ndarray, np.ndarray]:
        if self.hppc_r0_mohm.size == 0:
            return self.hppc_soc_grid, self.hppc_r0_mohm
        factor = self.n_series_in_module / self.n_parallel_in_module
        return self.hppc_soc_grid, self.hppc_r0_mohm / factor

    # Physically plausible R0 envelope for the cell chemistries we work with
    # (~50–200 Ah LFP). Anything outside [R0_SANITY_MIN_mΩ, R0_SANITY_MAX_mΩ]
    # is a measurement artefact (e.g. open-circuit pulse, sensor noise) and
    # is silently dropped from the interpolation grid.
    R0_SANITY_MIN_mOhm = 0.1
    R0_SANITY_MAX_mOhm = 5.0

    def r0_at_soc(self, soc: float, prefer: str = "dcir") -> Optional[float]:
        """Interpolate R0 (mΩ) at a given SoC. Uses DCIR by default; HPPC fallback.

        Anchors outside the sanity envelope are dropped before interpolation —
        previously the agent had to filter these case-by-case (e.g. MFR_C_1
        had a 0.0003 mΩ HPPC anchor that biased the interpolation).
        """
        sources = []
        if prefer == "dcir":
            sources = [(self.dcir_soc_grid, self.dcir_r0_mohm),
                       (self.hppc_soc_grid, self.hppc_r0_mohm)]
        else:
            sources = [(self.hppc_soc_grid, self.hppc_r0_mohm),
                       (self.dcir_soc_grid, self.dcir_r0_mohm)]
        for grid, vals in sources:
            if not (grid.size and vals.size):
                continue
            good = ((vals >= self.R0_SANITY_MIN_mOhm) &
                    (vals <= self.R0_SANITY_MAX_mOhm) & np.isfinite(vals))
            if not good.any():
                continue
            g, v = grid[good], vals[good]
            order = np.argsort(g)
            return float(np.interp(soc, g[order], v[order]))
        return None


def _row_to_characterization(
    row: pd.Series,
    n_series: int = 1,
    n_parallel: int = 1,
) -> Characterization:
    nominal = float(row.get("nominal_cap_ah", 0.0) or 0.0)
    q_rpt = float(row.get("q_rpt_ah", 0.0) or 0.0)
    # MFR_B_A sheet has no Soh column; compute from q_rpt / nominal
    soh = float(row.get("Soh", 0.0) or 0.0)
    if soh == 0.0 and nominal > 0.0:
        soh = (q_rpt / nominal) * 100.0
    return Characterization(
        cell_id=str(row.get("cell_id", "?")),
        cohort=str(row.get("cohort", "?")),
        manufacturer=str(row.get("manufacturer", "?")),
        batch=int(row.get("batch", 1) or 1),
        nominal_capacity_ah=nominal,
        q_rpt_ah=q_rpt,
        soh_pct=soh,
        ocv_soc_grid=_parse_array(row.get("ocv_soc_grid")),
        ocv_v_curve=_parse_array(row.get("v_oc_curve")),
        ocv_soc_grid_chg=_parse_array(row.get("ocv_soc_grid_chg")),
        ocv_v_chg_curve=_parse_array(row.get("v_oc_chg_curve")),
        q_rpt_chg_ah=float(row.get("q_rpt_chg_ah") or float("nan")),
        rate_cap_c_rates_chg=_parse_array(row.get("rate_cap_c_rates_chg")),
        rate_cap_q_chg_curve=_parse_array(row.get("rate_cap_q_chg_curve")),
        dcir_soc_grid=_parse_array(row.get("dcir_soc_nominal")),
        dcir_r0_mohm=_parse_array(row.get("r_dc_curve")),
        hppc_soc_grid=_parse_array(row.get("hppc_soc_at_pulse_dchg")),
        hppc_r0_mohm=_parse_array(row.get("r0_dchg_curve")),
        hppc_r1_mohm=_parse_array(row.get("r1_dchg_curve")),
        hppc_c1_F=_parse_array(row.get("c1_dchg_curve")),
        hppc_r2_mohm=_parse_array(row.get("r2_dchg_curve")),
        hppc_c2_F=_parse_array(row.get("c2_dchg_curve")),
        gitt_soc_grid=_parse_array(row.get("gitt_soc_grid")),
        gitt_r_pulse_mohm=_parse_array(row.get("r_pulse_curve")),
        gitt_tau_diff_s=_parse_array(row.get("tau_diff_curve")),
        gitt_v_inf_V=_parse_array(row.get("v_inf_curve")),
        n_series_in_module=n_series,
        n_parallel_in_module=n_parallel,
    )


def _aggregate_rows(rows: list[Characterization], cohort_label: str) -> Characterization:
    """Median-aggregate a list of single-cell Characterizations."""
    def med(getter):
        arrs = [getter(r) for r in rows if getter(r).size]
        if not arrs:
            return np.array([])
        n = min(len(a) for a in arrs)
        return np.median(np.array([a[:n] for a in arrs]), axis=0)

    first = rows[0]
    return Characterization(
        cell_id=f"{cohort_label}_median(n={len(rows)})",
        cohort=cohort_label,
        manufacturer=first.manufacturer,
        batch=first.batch,
        nominal_capacity_ah=float(np.median([r.nominal_capacity_ah for r in rows])),
        q_rpt_ah=float(np.median([r.q_rpt_ah for r in rows])),
        soh_pct=float(np.median([r.soh_pct for r in rows])),
        ocv_soc_grid=first.ocv_soc_grid,
        ocv_v_curve=med(lambda r: r.ocv_v_curve),
        ocv_soc_grid_chg=first.ocv_soc_grid_chg,
        ocv_v_chg_curve=med(lambda r: r.ocv_v_chg_curve),
        q_rpt_chg_ah=float(np.nanmedian([r.q_rpt_chg_ah for r in rows])),
        rate_cap_c_rates_chg=first.rate_cap_c_rates_chg,
        rate_cap_q_chg_curve=med(lambda r: r.rate_cap_q_chg_curve),
        dcir_soc_grid=first.dcir_soc_grid,
        dcir_r0_mohm=med(lambda r: r.dcir_r0_mohm),
        hppc_soc_grid=first.hppc_soc_grid,
        hppc_r0_mohm=med(lambda r: r.hppc_r0_mohm),
        hppc_r1_mohm=med(lambda r: r.hppc_r1_mohm),
        hppc_c1_F=med(lambda r: r.hppc_c1_F),
        hppc_r2_mohm=med(lambda r: r.hppc_r2_mohm),
        hppc_c2_F=med(lambda r: r.hppc_c2_F),
        gitt_soc_grid=first.gitt_soc_grid,
        gitt_r_pulse_mohm=med(lambda r: r.gitt_r_pulse_mohm),
        gitt_tau_diff_s=med(lambda r: r.gitt_tau_diff_s),
        gitt_v_inf_V=med(lambda r: r.gitt_v_inf_V),
    )


def list_available_cells(path: Union[str, Path] = DEFAULT_CHAR_PATH) -> pd.DataFrame:
    """Return a summary of all cells/modules available in the characterization file."""
    path = Path(path)
    rows = []
    for sheet in KNOWN_SHEETS:
        try:
            df = pd.read_excel(path, sheet_name=sheet)
        except Exception:
            continue
        df["_sheet"] = sheet
        if "Soh" not in df.columns:
            df["Soh"] = (df["q_rpt_ah"].astype(float) / df["nominal_cap_ah"].astype(float)) * 100.0
        rows.append(df[["_sheet", "cohort", "cell_id", "manufacturer", "batch",
                         "nominal_cap_ah", "q_rpt_ah", "Soh"]].rename(
                         columns={"nominal_cap_ah": "nominal_cap"}))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_characterization(
    path: Union[str, Path] = DEFAULT_CHAR_PATH,
    *,
    cohort: Optional[str] = None,
    manufacturer: Optional[str] = None,
    cell_id: Optional[str] = None,
    batch: Optional[int] = None,
    sheet: Optional[str] = None,
    aggregate: bool = False,
    n_series: int = 1,
    n_parallel: int = 1,
) -> Characterization:
    """
    Load characterization for one cell, one module, or a median-aggregated cohort.

    Parameters
    ----------
    cohort, manufacturer, cell_id : optional filters
    sheet : override which sheet to read (else: search both sheets)
    aggregate : if True, return cohort median; else require a unique match
    n_series, n_parallel : module topology (1, 1 for single cells)

    Examples
    --------
    # Single MFR_B cell, batch 1
    mfr_b = load_characterization(cell_id="MFR_A_0001")

    # Median of all 8 MFR_B cells in batch 1
    mfr_b_med = load_characterization(manufacturer="MFR_B", cohort="MFR_B",
                                    aggregate=True)

    # Module P012_M04 (12S2P)
    mod = load_characterization(cell_id="P012_M04",
                                n_series=12, n_parallel=2)
    """
    path = Path(path)
    sheets_to_search = (sheet,) if sheet else KNOWN_SHEETS
    matches: list[tuple[str, pd.Series]] = []
    for sh in sheets_to_search:
        try:
            df = pd.read_excel(path, sheet_name=sh)
        except Exception:
            continue
        mask = pd.Series([True] * len(df))
        if cohort is not None:
            mask &= df["cohort"].astype(str) == cohort
        if manufacturer is not None:
            mask &= df["manufacturer"].astype(str) == manufacturer
        if cell_id is not None:
            mask &= df["cell_id"].astype(str) == cell_id
        if batch is not None:
            mask &= df["batch"].astype(int) == int(batch)
        for _, row in df[mask].iterrows():
            matches.append((sh, row))

    if not matches:
        raise ValueError(
            f"No characterization found matching cohort={cohort}, "
            f"manufacturer={manufacturer}, cell_id={cell_id}"
        )

    chars = [_row_to_characterization(r, n_series=n_series, n_parallel=n_parallel)
             for _, r in matches]
    if aggregate or len(chars) > 1:
        if not aggregate and len(chars) > 1:
            raise ValueError(
                f"{len(chars)} cells matched the filters. "
                f"Set aggregate=True to take the cohort median or narrow the filters."
            )
        label = cohort or manufacturer or "aggregate"
        return _aggregate_rows(chars, cohort_label=str(label))
    return chars[0]
