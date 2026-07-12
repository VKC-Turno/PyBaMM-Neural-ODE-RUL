"""
Phase 1 — per-cell BOL (Beginning-of-Life) parameter identification.

Pure functions used by phase1_bol_identification.ipynb. Each function is
scoped to one physical concept:

    - load_prada_ocps()          → callable half-cell OCPs (LFP + graphite)
    - fit_stoichiometry()        → (x_100, x_0, y_100, y_0, RMSE)
    - derive_capacities()        → (Q_n_Ah, Q_p_Ah)
    - compute_D_s()              → solid-phase diffusivity from GITT tau
    - identify_cell()            → wraps the above for one (make, cell)

Design decisions:
  - Stoichiometry fit uses the Prada2013 half-cell OCPs as fixed thermodynamic
    templates and fits only the four stoichiometry endpoints. The alternative
    (freeing the OCP itself) is under-determined by a full-cell OCV curve.
  - Capacities are derived, not fitted. Q_n = Q_rpt / (x_100 - x_0) and
    Q_p = Q_rpt / (y_0 - y_100). This makes them consistent with both the
    fitted stoichiometry AND the measured RPT capacity.
  - D_s is computed via the standard GITT time-constant relationship
    D_s = R_p^2 / tau_diff, using the Prada2013 particle radius as the
    length scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# --------------------------------------------------------------------------- #
# PyBaMM Prada2013 OCP loaders
# --------------------------------------------------------------------------- #

def load_prada_ocps() -> tuple[Callable[[float | np.ndarray], np.ndarray],
                                 Callable[[float | np.ndarray], np.ndarray]]:
    """Return (U_p, U_n) — callable half-cell OCPs from Prada2013.

    Signature: U(stoichiometry: float | array) -> volts (array).
    """
    import pybamm
    param = pybamm.ParameterValues("Prada2013")

    U_p_fn = param["Positive electrode OCP [V]"]
    U_n_fn = param["Negative electrode OCP [V]"]

    def _eval(fn, s):
        s_arr = np.atleast_1d(np.asarray(s, dtype=float))
        try:
            out = fn(s_arr)
            if hasattr(out, "evaluate"):
                out = out.evaluate()
            return np.asarray(out, dtype=float).ravel()
        except Exception:
            # element-wise fallback
            vals = np.array([float(fn(float(v))) for v in s_arr])
            return vals

    def U_p(s): return _eval(U_p_fn, s)
    def U_n(s): return _eval(U_n_fn, s)
    return U_p, U_n


# --------------------------------------------------------------------------- #
# Stoichiometry fit
# --------------------------------------------------------------------------- #

@dataclass
class StoichFit:
    x_100: float
    x_0:   float
    y_100: float
    y_0:   float
    ocv_rmse_upper_half_mV: float   # SoC in [0.50, 0.90]
    ocv_rmse_full_mV:       float   # SoC in [0.02, 0.98]
    fit_success: bool
    n_points_used: int
    _bound_clipped: bool = False


def _v_predicted(soc: np.ndarray,
                  x_100: float, x_0: float,
                  y_100: float, y_0: float,
                  U_p: Callable, U_n: Callable) -> np.ndarray:
    x = x_0 + soc * (x_100 - x_0)
    y = y_0  + soc * (y_100 - y_0)
    return U_p(y) - U_n(x)


def fit_stoichiometry(ocv_curve: pd.DataFrame,
                       U_p: Callable, U_n: Callable,
                       *, soc_col: str = "soc_pct", v_col: str = "v",
                       soc_fit_range: tuple[float, float] = (0.02, 0.98),
                       soc_eval_range_upper: tuple[float, float] = (0.50, 0.90),
                       ) -> StoichFit:
    """Fit (x_100, x_0, y_100, y_0) to match measured full-cell OCV.

    ocv_curve columns: soc_pct (0..100) and v (volts). Measured discharge
    branch is expected (SoC decreases across the record).
    """
    df = ocv_curve.copy().dropna(subset=[soc_col, v_col])
    if df.empty:
        return StoichFit(np.nan, np.nan, np.nan, np.nan,
                         np.nan, np.nan, False, 0)

    df["soc"] = df[soc_col] / 100.0
    df = df[(df["soc"] >= soc_fit_range[0]) & (df["soc"] <= soc_fit_range[1])]
    if len(df) < 20:
        return StoichFit(np.nan, np.nan, np.nan, np.nan,
                         np.nan, np.nan, False, len(df))
    df = df.sort_values("soc").reset_index(drop=True)

    soc = df["soc"].to_numpy()
    v_meas = df[v_col].to_numpy()

    # Initial guess: Prada2013 cohort values from earlier identified_params.yaml
    p0 = np.array([0.879, 0.122, 0.010, 0.951])

    # Bounds: keep physically sensible ordering
    bounds = [(0.60, 0.99),   # x_100  (charged graphite fraction lithiated)
              (0.01, 0.30),   # x_0    (discharged graphite fraction lithiated)
              (0.001, 0.10),  # y_100  (charged LFP fraction lithiated - low)
              (0.60, 0.99)]   # y_0    (discharged LFP fraction lithiated - high)

    def loss(p):
        x100, x0, y100, y0 = p
        v_pred = _v_predicted(soc, x100, x0, y100, y0, U_p, U_n)
        residual = v_pred - v_meas
        # Weight upper half (plateau) 3x — that's where fit accuracy matters
        w = np.where(soc >= 0.5, 3.0, 1.0)
        return float(np.mean(w * residual**2))

    result = minimize(loss, p0, method="L-BFGS-B", bounds=bounds,
                       options=dict(maxiter=200, ftol=1e-10))

    x_100, x_0, y_100, y_0 = result.x
    bound_clipped = bool(any(
        (abs(v - b[0]) < 1e-5 or abs(v - b[1]) < 1e-5)
        for v, b in zip(result.x, bounds)
    ))

    # RMSE budgets on common grids
    v_pred_all = _v_predicted(soc, x_100, x_0, y_100, y_0, U_p, U_n)
    resid = v_pred_all - v_meas
    mask_upper = (soc >= soc_eval_range_upper[0]) & (soc <= soc_eval_range_upper[1])
    rmse_upper = float(np.sqrt(np.mean(resid[mask_upper]**2)) * 1000.0) if mask_upper.any() else np.nan
    rmse_full  = float(np.sqrt(np.mean(resid**2)) * 1000.0)

    return StoichFit(
        x_100=float(x_100), x_0=float(x_0),
        y_100=float(y_100), y_0=float(y_0),
        ocv_rmse_upper_half_mV=rmse_upper,
        ocv_rmse_full_mV=rmse_full,
        fit_success=bool(result.success),
        n_points_used=int(len(df)),
        _bound_clipped=bound_clipped,
    )


# --------------------------------------------------------------------------- #
# Derived quantities
# --------------------------------------------------------------------------- #

def derive_capacities(stoich: StoichFit, Q_rpt_Ah: float) -> dict:
    """Compute Q_n_Ah and Q_p_Ah from stoichiometry + measured RPT capacity.

    Q_n = Q_rpt / (x_100 - x_0)   — negative electrode capacity
    Q_p = Q_rpt / (y_0  - y_100)  — positive electrode capacity
    """
    dx = stoich.x_100 - stoich.x_0
    dy = stoich.y_0   - stoich.y_100
    Q_n = float(Q_rpt_Ah / dx) if dx > 1e-6 else np.nan
    Q_p = float(Q_rpt_Ah / dy) if dy > 1e-6 else np.nan
    return dict(Q_n_Ah=Q_n, Q_p_Ah=Q_p, Q_rpt_used=float(Q_rpt_Ah))


# --------------------------------------------------------------------------- #
# Solid-phase diffusivity from GITT tau
# --------------------------------------------------------------------------- #

def compute_D_s(tau_diff_s: float,
                 particle_radius_m: float = 5.22e-6,
                 ) -> dict:
    """Solid-phase diffusivity from GITT relaxation time.

    D_s = R_p^2 / tau_diff  (standard GITT relationship for a spherical
    particle; Prada2013 R_p = 5.22 μm for the positive electrode by default).
    """
    if not (isinstance(tau_diff_s, (int, float)) and np.isfinite(tau_diff_s)
            and tau_diff_s > 0):
        return dict(D_s_m2_s=np.nan, tau_diff_used=np.nan,
                     particle_radius_m=particle_radius_m)
    D_s = (particle_radius_m ** 2) / tau_diff_s
    return dict(D_s_m2_s=float(D_s), tau_diff_used=float(tau_diff_s),
                 particle_radius_m=float(particle_radius_m))


# --------------------------------------------------------------------------- #
# Full per-cell identification wrapper
# --------------------------------------------------------------------------- #

@dataclass
class CellBOL:
    make: str
    cell: str
    nominal_capacity_Ah: float
    stoichiometry: dict = field(default_factory=dict)
    capacity:      dict = field(default_factory=dict)
    resistance:    dict = field(default_factory=dict)
    diffusion:     dict = field(default_factory=dict)
    validation:    dict = field(default_factory=dict)
    _provenance:   dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _cell_ocv_curve_path(make: str, cell: str) -> Optional[Path]:
    p = Path(f"/home/hj/Desktop/PINNs/Data/OCVSOC/{make}_OCVSOC_cell_{cell}.csv")
    return p if p.exists() else None


def identify_cell(make: str, cell: str, *,
                  Q_rpt_Ah: float,
                  HPPC_R0_mOhm: float, HPPC_R1_mOhm: float,
                  GITT_tau_diff_s: float,
                  nominal_capacity_Ah: float,
                  measured_soh_first: float = np.nan,
                  U_p: Callable, U_n: Callable,
                  particle_radius_m: float = 5.22e-6,
                  extract_module=None,
                  ) -> CellBOL:
    """End-to-end BOL identification for one cell.

    Loads the OCV curve (via extract.py), fits stoichiometry, derives
    capacities + D_s, packages resistances from HPPC scalars.
    """
    from datetime import datetime, timezone

    # ---- Load OCV curve ----
    csv = _cell_ocv_curve_path(make, cell)
    if csv is None:
        return CellBOL(make=make, cell=cell,
                        nominal_capacity_Ah=float(nominal_capacity_Ah),
                        validation=dict(error="OCV CSV not found"))
    if extract_module is None:
        import extract as _ex
    else:
        _ex = extract_module
    curve = _ex.extract_ocv_curve(csv)

    # ---- Stoichiometry fit ----
    stoich = fit_stoichiometry(curve, U_p, U_n)

    # ---- Derived capacities ----
    caps = derive_capacities(stoich, Q_rpt_Ah)

    # ---- D_s from GITT ----
    diff = compute_D_s(GITT_tau_diff_s, particle_radius_m=particle_radius_m)

    # ---- Resistances (from HPPC scalars, already extracted) ----
    resistance = dict(
        R0_Ohm=float(HPPC_R0_mOhm) / 1000.0 if np.isfinite(HPPC_R0_mOhm) else np.nan,
        R1_Ohm=float(HPPC_R1_mOhm) / 1000.0 if np.isfinite(HPPC_R1_mOhm) else np.nan,
    )

    # ---- Validation ----
    passed_upper = (stoich.ocv_rmse_upper_half_mV < 20.0
                     if np.isfinite(stoich.ocv_rmse_upper_half_mV) else False)
    validation = dict(
        ocv_rmse_upper_half_mV=stoich.ocv_rmse_upper_half_mV,
        ocv_rmse_full_mV=stoich.ocv_rmse_full_mV,
        rmse_upper_bound_mV=20.0,
        passed_upper_rmse_budget=bool(passed_upper),
        stoich_bound_clipped=bool(stoich._bound_clipped),
        n_ocv_points_used=int(stoich.n_points_used),
    )

    provenance = dict(
        identified_at_utc=datetime.now(timezone.utc).isoformat(),
        source_notebook="phase1_bol_identification.ipynb",
        pipeline_version=1,
        prada_ocp_source="pybamm.ParameterValues('Prada2013')",
    )

    stoich_dict = dict(
        x_100=stoich.x_100, x_0=stoich.x_0,
        y_100=stoich.y_100, y_0=stoich.y_0,
        fit_success=stoich.fit_success,
    )

    return CellBOL(
        make=make, cell=cell,
        nominal_capacity_Ah=float(nominal_capacity_Ah),
        stoichiometry=stoich_dict,
        capacity=caps,
        resistance=resistance,
        diffusion=diff,
        validation=validation,
        _provenance=provenance,
    )
