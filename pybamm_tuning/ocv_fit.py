"""
Stoichiometry fit from a measured full-cell OCV(SoC) curve.

The idea: PyBaMM stores half-cell OCPs (U_n(x_n), U_p(y_p)) per electrode.
The full-cell OCV at SoC = s is

    OCV(s) = U_p(y_0 + (y_100 - y_0) * s) - U_n(x_0 + (x_100 - x_0) * s)

so the four stoichiometric limits (x_0, x_100, y_0, y_100) shift each
electrode's operating window. Fitting them to a measured OCV curve gives
us the EVE-specific cell balance without needing half-cell data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybamm
from scipy.optimize import minimize


# Sanity band for the OCV anchor at SoC=1.0.
#
#   Lower 3.40 V — fully relaxed LFP equilibrium voltage at SoC=1.0 (after
#                   hours of rest post-CCCV); values below this mean the
#                   workbook is missing the CV tail or the SoC mapping is offset.
#   Upper 3.65 V — CCCV charge cut-off voltage (the physical maximum the cell
#                   can reach during the standard char protocol). Values above
#                   this would imply a higher cut, which our protocol doesn't use.
#
# Real lab data typically lands in [3.50, 3.60] V because the OCV sample is
# taken after a short rest (not the multi-hour rest needed to fully relax).
# That's normal — not a defect — so the band excludes only the clearly
# truncated / corrupted cases.
LFP_FULL_CHARGE_V_BAND = (3.40, 3.65)


@dataclass(frozen=True)
class StoichiometryResult:
    x_100: float
    x_0:   float
    y_100: float
    y_0:   float
    rmse_mV: float
    n_anchors: int
    ocv_top_v: float = float("nan")           # V_OC measured at SoC=1.0
    ocv_top_outside_lfp_band: bool = False    # True if ocv_top_v outside LFP_FULL_CHARGE_V_BAND


def _evaluate_half_cell_ocp(param: pybamm.ParameterValues, electrode: str,
                            stoich: np.ndarray) -> np.ndarray:
    """Evaluate U_n(x) or U_p(y) at given stoichiometries (PyBaMM OCP functions
    take stoichiometry, not concentration)."""
    key = "Negative electrode OCP [V]" if electrode == "negative" else "Positive electrode OCP [V]"
    f = param[key]
    if callable(f):
        return np.array([float(f(float(s))) for s in stoich])
    arr = np.asarray(f)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return np.interp(stoich, arr[:, 0], arr[:, 1])
    raise ValueError(f"Unsupported OCP representation for {electrode}")


def fit_stoichiometry_from_ocv(
    soc_grid: np.ndarray,
    ocv_v_measured: np.ndarray,
    base: str = "Prada2013",
    initial_guess: tuple[float, float, float, float] = (0.85, 0.05, 0.05, 0.95),
    bounds: tuple[tuple[float, float], ...] = ((0.5, 0.99), (0.001, 0.4),
                                                (0.001, 0.4), (0.6, 0.99)),
    method: str = "L-BFGS-B",
) -> StoichiometryResult:
    """
    Fit (x_100, x_0, y_100, y_0) so PyBaMM's full-cell OCV matches measurement.

    Convention: SoC=1 corresponds to (x_100, y_100); SoC=0 to (x_0, y_0).
    For LFP/graphite: x_100 ≈ 0.85, x_0 ≈ 0.04, y_100 ≈ 0.04, y_0 ≈ 0.95.

    Parameters
    ----------
    soc_grid : SoC anchors (typically [0, 0.1, ..., 1.0])
    ocv_v_measured : measured OCV [V] at those anchors
    base : PyBaMM base parameter set name (provides the half-cell OCP functions)

    Returns
    -------
    StoichiometryResult with the four fitted limits and the residual RMSE in mV.
    """
    soc_grid = np.asarray(soc_grid, dtype=float)
    ocv_v_measured = np.asarray(ocv_v_measured, dtype=float)
    if soc_grid.size != ocv_v_measured.size or soc_grid.size < 4:
        raise ValueError(
            f"Need matching SoC and OCV arrays with ≥4 anchors; "
            f"got {soc_grid.size} and {ocv_v_measured.size}"
        )

    param = pybamm.ParameterValues(base)

    def model_ocv(stoich_params: np.ndarray) -> np.ndarray:
        x_100, x_0, y_100, y_0 = stoich_params
        x = x_0 + (x_100 - x_0) * soc_grid
        y = y_0 + (y_100 - y_0) * soc_grid
        U_n = _evaluate_half_cell_ocp(param, "negative", x)
        U_p = _evaluate_half_cell_ocp(param, "positive", y)
        return U_p - U_n

    def objective(params_arr: np.ndarray) -> float:
        try:
            v_model = model_ocv(params_arr)
        except Exception:
            return 1e6
        if not np.all(np.isfinite(v_model)):
            return 1e6
        return float(np.mean((v_model - ocv_v_measured) ** 2))

    res = minimize(objective, x0=np.array(initial_guess),
                   method=method, bounds=bounds,
                   options={"maxiter": 500, "ftol": 1e-9})
    x_100, x_0, y_100, y_0 = (float(v) for v in res.x)
    v_fit = model_ocv(res.x)
    rmse = float(np.sqrt(np.mean((v_fit - ocv_v_measured) ** 2))) * 1000.0  # mV

    # Sanity-check the full-charge anchor. The agent's gate audit needs this
    # because a truncated OCV (e.g. top sample at 3.48 V instead of ~3.45 V
    # with a CV tail) fits the visible curve well but yields stoichiometric
    # limits that misrepresent the cell at true 100 % SoC.
    top_idx = int(np.argmax(soc_grid))
    ocv_top = float(ocv_v_measured[top_idx])
    outside = (ocv_top < LFP_FULL_CHARGE_V_BAND[0] or
                ocv_top > LFP_FULL_CHARGE_V_BAND[1])

    return StoichiometryResult(
        x_100=x_100, x_0=x_0, y_100=y_100, y_0=y_0,
        rmse_mV=rmse, n_anchors=int(soc_grid.size),
        ocv_top_v=ocv_top, ocv_top_outside_lfp_band=outside,
    )
