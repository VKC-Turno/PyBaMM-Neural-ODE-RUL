"""Real ODE governing rxn-limited SEI + optional LAM degradation.

Three physics ODE forms of increasing fidelity:

Level 1 (SoH-dependent rxn-lim SEI):
    dSoH/dn = -k_SEI · SoH^p                                    (2 params)

Level 2 (SEI + delayed LAM activation):
    dSoH/dn = -k_SEI · SoH^p - k_LAM · exp((n-n_c)/tau) · [n > n_c]   (5 params)

Level 3 (PyBaMM precomputed prior):
    dSoH/dn = interp( dSoH_PyBaMM/dn, at n )                    (tabulated)

Physical basis: rxn-limited SEI grows monotonically per cycle; growth
rate depends on remaining stoichiometry window (SoH^p). LAM activates
after a critical cycle count once electrode stress accumulates past a
threshold — see O'Kane et al. Phys. Chem. Chem. Phys. 24, 7909 (2022).
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.optimize import curve_fit, differential_evolution
from .data import CellData


# ────────────────────── Level 1: SoH-dependent rxn-lim SEI ──────────────────────

def _integrate_L1(k_SEI: float, p: float, soh_0: float,
                   n_start: float, n_eval: np.ndarray, dn: float = 1.0) -> np.ndarray:
    """Numerically integrate dSoH/dn = -k_SEI · SoH^p from soh_0 at n_start."""
    n_grid = np.arange(n_start, n_eval.max() + dn, dn)
    soh = np.zeros_like(n_grid, dtype=np.float64)
    soh[0] = soh_0
    for i in range(1, len(n_grid)):
        rate = -k_SEI * (max(soh[i-1], 1e-6) ** p)
        soh[i] = soh[i-1] + rate * dn
    return np.interp(n_eval, n_grid, soh)


def fit_L1(cell: CellData, K: int) -> dict:
    """Fit (k_SEI, p) on the first K cycles of the measured trajectory."""
    n = cell.n_traj.numpy(); s = cell.soh_traj.numpy()
    first_cy = float(n[0])
    k_end = first_cy + K
    mask = n <= k_end
    n_tr, s_tr = n[mask], s[mask]
    soh_0 = float(s_tr[0])

    def residual(params):
        k, p = params
        s_pred = _integrate_L1(k, p, soh_0, first_cy, n_tr)
        return float(np.mean((s_pred - s_tr) ** 2))

    # Fit with differential evolution (robust to bad initial conditions)
    result = differential_evolution(
        residual, bounds=[(1e-8, 1e-2), (0.0, 2.0)],
        seed=42, maxiter=100, tol=1e-8, popsize=20,
    )
    k_SEI, p = result.x
    return dict(k_SEI=float(k_SEI), p=float(p), soh_0=soh_0,
                first_cy=first_cy, residual_mse=float(result.fun))


def physics_trajectory_L1(soh_0: float, k_SEI: float, p: float,
                            n_eval: np.ndarray, n_start: float) -> np.ndarray:
    """Closed-form-ish trajectory using the Level-1 ODE."""
    return _integrate_L1(k_SEI, p, soh_0, n_start, n_eval)


def physics_rate_L1(soh: torch.Tensor, k_SEI: float, p: float) -> torch.Tensor:
    """Pointwise dSoH/dn = -k_SEI · SoH^p — used inside the PINN physics loss."""
    return -k_SEI * torch.clamp(soh, min=1e-6) ** p


# ────────────────── Level 2: SEI + delayed LAM activation ──────────────────

def _integrate_L2(k_SEI: float, p: float, k_LAM: float, n_c: float,
                   tau: float, soh_0: float, n_start: float,
                   n_eval: np.ndarray, dn: float = 1.0) -> np.ndarray:
    """dSoH/dn = -k_SEI · SoH^p - k_LAM · exp((n-n_c)/tau) · [n > n_c]."""
    n_grid = np.arange(n_start, n_eval.max() + dn, dn)
    soh = np.zeros_like(n_grid, dtype=np.float64)
    soh[0] = soh_0
    for i in range(1, len(n_grid)):
        n = n_grid[i-1]
        sei_rate = k_SEI * max(soh[i-1], 1e-6) ** p
        # Cap the LAM exponential to avoid overflow in bad fits
        arg = min((n - n_c) / max(tau, 1.0), 20.0)
        lam_rate = k_LAM * np.exp(arg) if n > n_c else 0.0
        rate = -(sei_rate + lam_rate)
        soh[i] = max(soh[i-1] + rate * dn, 0.0)
    return np.interp(n_eval, n_grid, soh)


def fit_L2(cell: CellData, K: int) -> dict:
    """Fit 5-parameter SEI+LAM ODE on first K cycles."""
    n = cell.n_traj.numpy(); s = cell.soh_traj.numpy()
    first_cy = float(n[0])
    k_end = first_cy + K
    mask = n <= k_end
    n_tr, s_tr = n[mask], s[mask]
    soh_0 = float(s_tr[0])

    def residual(params):
        k_SEI, p, k_LAM, n_c, tau = params
        s_pred = _integrate_L2(k_SEI, p, k_LAM, n_c, tau,
                                soh_0, first_cy, n_tr)
        return float(np.mean((s_pred - s_tr) ** 2))

    result = differential_evolution(
        residual,
        bounds=[(1e-8, 1e-2),   # k_SEI
                (0.0, 2.0),      # p
                (0.0, 1e-4),     # k_LAM  (small — activates gradually)
                (K, 2000.0),     # n_c    — LAM must activate after training window
                (50.0, 800.0)],  # tau
        seed=42, maxiter=200, tol=1e-9, popsize=30,
    )
    k_SEI, p, k_LAM, n_c, tau = result.x
    return dict(k_SEI=float(k_SEI), p=float(p),
                k_LAM=float(k_LAM), n_c=float(n_c), tau=float(tau),
                soh_0=soh_0, first_cy=first_cy,
                residual_mse=float(result.fun))


def physics_trajectory_L2(soh_0: float, params: dict,
                            n_eval: np.ndarray, n_start: float) -> np.ndarray:
    return _integrate_L2(params["k_SEI"], params["p"], params["k_LAM"],
                          params["n_c"], params["tau"],
                          soh_0, n_start, n_eval)


def physics_rate_L2(soh: torch.Tensor, n_cycle: torch.Tensor,
                     params: dict) -> torch.Tensor:
    """Pointwise dSoH/dn for the Level-2 ODE — used in PINN physics loss."""
    k_SEI = params["k_SEI"]; p = params["p"]
    k_LAM = params["k_LAM"]; n_c = params["n_c"]; tau = params["tau"]

    sei_rate = k_SEI * torch.clamp(soh, min=1e-6) ** p
    arg = torch.clamp((n_cycle - n_c) / max(tau, 1.0), max=20.0)
    lam_rate = k_LAM * torch.exp(arg) * (n_cycle > n_c).float()
    return -(sei_rate + lam_rate)


# ────────── Backwards-compat aliases used by Day-1 scripts ──────────

def estimate_k_sei_from_window(cell: CellData, K: int) -> float:
    """Legacy linear-rate estimator — kept for Day-1 scripts."""
    n = cell.n_traj.numpy(); s = cell.soh_traj.numpy()
    first_cy = float(n[0])
    mask = n <= first_cy + K
    if mask.sum() < 5:
        return 1e-4
    slope, _ = np.polyfit(n[mask] - first_cy, s[mask], 1)
    return max(-slope, 0.0)


def physics_trajectory(soh_init: float, k_SEI: float,
                        n_eval: torch.Tensor, n_start: float) -> torch.Tensor:
    """Legacy linear-fade trajectory — kept for Day-1 scripts."""
    return soh_init - k_SEI * (n_eval - n_start)
