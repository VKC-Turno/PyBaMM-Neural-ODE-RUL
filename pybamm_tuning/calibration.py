"""
1-D back-calibration of k_SEI to match a measured fade rate.

For LFP cells the SEI kinetic rate constant is the strongest lever on the
cycle-fade slope. If we have a measured (-1.00 pp/100cy from REPT, say)
we can solve for the k_SEI that makes a short PyBaMM run produce that
exact slope, holding all other identified parameters fixed.

Uses scipy.optimize.brentq over log10(k_SEI) for stability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .characterization import Characterization
from .parameters import build_pybamm_parameters
from .simulation import CyclingProtocol, Simulation
from .validation import _slope_pp_per_100cy


# DFN options preset that isolates SEI growth from plating and stress-driven LAM —
# the right choice when calibrating a single SEI parameter.
SEI_ONLY_DFN_OPTIONS: dict = {
    "SEI": "solvent-diffusion limited",
    "SEI porosity change": "true",
    "lithium plating": "none",
    "loss of active material": "none",
}


@dataclass
class CalibrationResult:
    parameter_name:                str
    fitted_value:                  float
    achieved_slope_pp_per_100cy:   float
    target_slope_pp_per_100cy:     float
    residual_pp_per_100cy:         float
    n_evaluations:                 int
    log10_bracket_used:            tuple[float, float]
    # `n_evaluations` includes cached lookups (≈ instant). `n_fresh_sims`
    # counts only actual PyBaMM solves — the wall-time-relevant number for
    # honest accounting in reports.
    n_fresh_sims:                  int = 0

    @property
    def k_SEI_fitted_m_per_s(self) -> float:
        """Back-compat alias for old SEI-kinetic-rate calibrations."""
        return self.fitted_value if "SEI kinetic" in self.parameter_name else float("nan")


def calibrate_sei_diffusivity(
    char: Characterization,
    target_slope_pp_per_100cy: float,
    *,
    base: str = "Prada2013",
    protocol: Optional[CyclingProtocol] = None,
    temperature_K: float = 298.15,
    n_cycles: int = 8,
    log10_bracket: tuple[float, float] = (-26.0, -19.0),
    rtol: float = 0.05,
    max_iter: int = 12,
    cache_dir=None,
    pre_age_to_soh: Optional[float] = None,
) -> CalibrationResult:
    """
    Calibrate `SEI solvent diffusivity [m2.s-1]` to match a measured fade rate.

    This is the appropriate calibration for PyBaMM's default SEI:"solvent-diffusion
    limited" model. Uses SEI_ONLY_DFN_OPTIONS so the slope is driven purely by
    the SEI parameter — plating and LAM are turned off during the fit.
    """
    protocol = protocol or CyclingProtocol()
    key = "SEI solvent diffusivity [m2.s-1]"

    n_fresh = [0]   # mutable so the closure can bump it without `nonlocal`

    def slope_for(log10_D: float) -> float:
        params = build_pybamm_parameters(
            char, base=base, temperature_K=temperature_K,
            extra_overrides={key: 10 ** log10_D},
            pre_age_to_soh=pre_age_to_soh,
        )
        sim = Simulation(params, protocol=protocol, cache_dir=cache_dir,
                          dfn_options=SEI_ONLY_DFN_OPTIONS)
        df = sim.run(n_cycles=n_cycles)
        if not getattr(sim, "last_was_cached", True):
            n_fresh[0] += 1
        cyc = df["cycle_n"].to_numpy(dtype=float)[1:]   # skip warm-up
        soh = df["SOH"].to_numpy(dtype=float)[1:] * 100.0
        return _slope_pp_per_100cy(cyc, soh)

    lo, hi = log10_bracket
    slope_lo, slope_hi = slope_for(lo), slope_for(hi)
    n_evals = 2

    if (slope_lo - target_slope_pp_per_100cy) * (slope_hi - target_slope_pp_per_100cy) > 0:
        if abs(slope_lo - target_slope_pp_per_100cy) < abs(slope_hi - target_slope_pp_per_100cy):
            best_log10 = lo;  best_slope = slope_lo
        else:
            best_log10 = hi;  best_slope = slope_hi
        return CalibrationResult(
            parameter_name=key, fitted_value=10 ** best_log10,
            achieved_slope_pp_per_100cy=best_slope,
            target_slope_pp_per_100cy=target_slope_pp_per_100cy,
            residual_pp_per_100cy=best_slope - target_slope_pp_per_100cy,
            n_evaluations=n_evals, log10_bracket_used=log10_bracket,
            n_fresh_sims=n_fresh[0],
        )

    mid = 0.5 * (lo + hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        slope_mid = slope_for(mid); n_evals += 1
        if abs(slope_mid - target_slope_pp_per_100cy) <= rtol * max(abs(target_slope_pp_per_100cy), 0.05):
            break
        if (slope_lo - target_slope_pp_per_100cy) * (slope_mid - target_slope_pp_per_100cy) < 0:
            hi, slope_hi = mid, slope_mid
        else:
            lo, slope_lo = mid, slope_mid

    return CalibrationResult(
        parameter_name=key, fitted_value=10 ** mid,
        achieved_slope_pp_per_100cy=slope_mid,
        target_slope_pp_per_100cy=target_slope_pp_per_100cy,
        residual_pp_per_100cy=slope_mid - target_slope_pp_per_100cy,
        n_evaluations=n_evals, log10_bracket_used=log10_bracket,
        n_fresh_sims=n_fresh[0],
    )


def calibrate_k_sei(
    char: Characterization,
    target_slope_pp_per_100cy: float,
    *,
    base: str = "Prada2013",
    protocol: Optional[CyclingProtocol] = None,
    temperature_K: float = 298.15,
    n_cycles: int = 10,
    log10_bracket: tuple[float, float] = (-16.0, -12.0),
    rtol: float = 0.05,           # 5 % relative tolerance on slope
    max_iter: int = 12,
    cache_dir=None,
    aging_overrides_base: Optional[dict] = None,
) -> CalibrationResult:
    """
    Find k_SEI such that PyBaMM's predicted fade rate matches the target.

    Strategy: bisection over log10(k_SEI). Each iteration is a fresh PyBaMM
    cycle run (cached by the simulation layer if cache_dir is set, so a
    repeated call with the same intermediate k_SEI is free).

    Parameters
    ----------
    target_slope_pp_per_100cy : measured fade rate to match (negative for fade)
    n_cycles : short runs are fine; we only need a slope, not full cycle life
    log10_bracket : (lo, hi) initial bracket for log10(k_SEI)

    Returns CalibrationResult.
    """
    protocol = protocol or CyclingProtocol()
    aging_overrides_base = dict(aging_overrides_base) if aging_overrides_base else {}
    key = "SEI kinetic rate constant [m.s-1]"  # parameter_name on result
    n_fresh = [0]

    def slope_for(log10_k: float) -> tuple[float, dict]:
        aging = dict(aging_overrides_base)
        aging["k_SEI_ms"] = 10 ** log10_k
        params = build_pybamm_parameters(char, base=base,
                                          aging_overrides=aging,
                                          temperature_K=temperature_K)
        sim = Simulation(params, protocol=protocol, cache_dir=cache_dir)
        df = sim.run(n_cycles=n_cycles)
        if not getattr(sim, "last_was_cached", True):
            n_fresh[0] += 1
        cyc = df["cycle_n"].to_numpy(dtype=float)
        soh_pct = df["SOH"].to_numpy(dtype=float) * 100.0
        return _slope_pp_per_100cy(cyc, soh_pct), aging

    lo, hi = log10_bracket
    n_evals = 0
    slope_lo, _ = slope_for(lo); n_evals += 1
    slope_hi, _ = slope_for(hi); n_evals += 1

    # If target lies outside the bracket, return the nearest endpoint.
    if (slope_lo - target_slope_pp_per_100cy) * (slope_hi - target_slope_pp_per_100cy) > 0:
        if abs(slope_lo - target_slope_pp_per_100cy) < abs(slope_hi - target_slope_pp_per_100cy):
            best_log10 = lo;  best_slope = slope_lo
        else:
            best_log10 = hi;  best_slope = slope_hi
        return CalibrationResult(
            parameter_name=key, fitted_value=10 ** best_log10,
            achieved_slope_pp_per_100cy=best_slope,
            target_slope_pp_per_100cy=target_slope_pp_per_100cy,
            residual_pp_per_100cy=best_slope - target_slope_pp_per_100cy,
            n_evaluations=n_evals, log10_bracket_used=(lo, hi),
            n_fresh_sims=n_fresh[0],
        )

    # Bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        slope_mid, _ = slope_for(mid); n_evals += 1
        if abs(slope_mid - target_slope_pp_per_100cy) <= rtol * abs(target_slope_pp_per_100cy):
            return CalibrationResult(
                parameter_name=key, fitted_value=10 ** mid,
                achieved_slope_pp_per_100cy=slope_mid,
                target_slope_pp_per_100cy=target_slope_pp_per_100cy,
                residual_pp_per_100cy=slope_mid - target_slope_pp_per_100cy,
                n_evaluations=n_evals,
                log10_bracket_used=(log10_bracket[0], log10_bracket[1]),
                n_fresh_sims=n_fresh[0],
            )
        if (slope_lo - target_slope_pp_per_100cy) * (slope_mid - target_slope_pp_per_100cy) < 0:
            hi, slope_hi = mid, slope_mid
        else:
            lo, slope_lo = mid, slope_mid

    return CalibrationResult(
        parameter_name=key, fitted_value=10 ** mid,
        achieved_slope_pp_per_100cy=slope_mid,
        target_slope_pp_per_100cy=target_slope_pp_per_100cy,
        residual_pp_per_100cy=slope_mid - target_slope_pp_per_100cy,
        n_evaluations=n_evals,
        log10_bracket_used=(log10_bracket[0], log10_bracket[1]),
        n_fresh_sims=n_fresh[0],
    )
