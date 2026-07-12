"""
src/simulation/extract_features.py
----------------------------------
Per-cycle feature extraction from a PyBaMM degradation simulation.

For each completed cycle we record:
    cycle_n                discharge cycle index, 1-based
    Q_Ah                   discharge capacity this cycle
    SOH                    Q_Ah / Q_Ah_cycle_1
    V_mean_discharge       mean voltage during the discharge step
    dcir_mOhm              R0 estimate from the discharge onset
    ic_peak1_V             1st dQ/dV peak voltage (lower plateau side)
    ic_peak2_V             2nd dQ/dV peak voltage (upper plateau side)
    ic_peak1_area          integrated dQ/dV around peak 1
    ic_peak2_area          integrated dQ/dV around peak 2
    SEI_thickness_m        x-averaged negative SEI thickness, end of cycle
    LAM_negative_pct       % loss of negative active material, end of cycle
    LAM_positive_pct       % loss of positive active material, end of cycle
    dead_lithium_mol       cumulative loss of Li to plating + SEI, end of cycle
    T_K                    mean temperature this cycle (used only for sanity)
    c_rate                 simulation C-rate (carried through)
    k_SEI                  simulation k_SEI (carried through)

IC curves are saved one file per cycle into a directory if requested.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe(sol, key, default=np.nan):
    try:
        return sol[key].entries
    except Exception:
        return default


def extract_ic_curve(voltage: np.ndarray, capacity: np.ndarray,
                     n_points: int = 500,
                     v_lo: float = 2.6, v_hi: float = 3.55
                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a smoothed dQ/dV curve over a uniform V grid in [v_lo, v_hi].

    Implementation:
        - sort by V
        - drop duplicate voltage samples (LFP plateau yields many)
        - low-pass via Savitzky–Golay on Q(V), then gradient
    """
    from scipy.signal import savgol_filter

    idx = np.argsort(voltage)
    V = voltage[idx]
    Q = capacity[idx]

    # Keep only the monotonic-V portion
    keep = np.concatenate(([True], np.diff(V) > 1e-6))
    V, Q = V[keep], Q[keep]
    if len(V) < 25:
        v_grid = np.linspace(v_lo, v_hi, n_points)
        return v_grid, np.full_like(v_grid, np.nan)

    v_grid = np.linspace(v_lo, v_hi, n_points)
    q_grid = np.interp(v_grid, V, Q)

    window = max(21, (n_points // 10) | 1)  # odd
    polyorder = 3
    q_smooth = savgol_filter(q_grid, window_length=window, polyorder=polyorder)
    dqdv = np.gradient(q_smooth, v_grid)
    return v_grid, dqdv


def find_ic_peaks(v_grid: np.ndarray, dqdv: np.ndarray, n_peaks: int = 2
                  ) -> list[dict]:
    """Return up to `n_peaks` of |dQ/dV|, sorted by peak height descending."""
    from scipy.signal import find_peaks

    y = np.abs(dqdv)
    if not np.isfinite(y).any():
        return []
    peaks, props = find_peaks(y, prominence=np.nanmax(y) * 0.05, distance=10)
    if len(peaks) == 0:
        return []
    order = np.argsort(props["prominences"])[::-1][:n_peaks]
    out = []
    for i in order:
        idx = int(peaks[i])
        lo = max(0, idx - 15)
        hi = min(len(v_grid), idx + 15)
        area = float(np.trapezoid(np.abs(dqdv[lo:hi]), v_grid[lo:hi]))
        out.append({
            "V": float(v_grid[idx]),
            "height": float(y[idx]),
            "area": area,
        })
    # Sort the returned peaks by voltage ascending so peak1=low-V, peak2=high-V
    return sorted(out, key=lambda d: d["V"])


def per_cycle_features(sol, params_used: dict[str, Any],
                       save_ic_dir: Path | None = None) -> pd.DataFrame:
    """
    Walk the cycles of a PyBaMM Solution and return a DataFrame of
    per-cycle features. PyBaMM's `sol.cycles` here corresponds to whatever
    blocks the experiment list defined.

    Convention: each *cycle* of our experiment is one (Discharge, Rest,
    Charge, Rest) tuple; we identify the discharge step as the one whose
    current is consistently negative.
    """
    rows: list[dict] = []
    q0: float | None = None

    for n, cycle in enumerate(sol.cycles, start=1):
        # Find the discharge step (mean current < 0)
        disc = None
        for step in cycle.steps:
            try:
                I_mean = float(np.nanmean(step["Current [A]"].entries))
            except Exception:
                continue
            if I_mean < -1e-3:
                disc = step
                break
        if disc is None:
            continue

        V = disc["Voltage [V]"].entries
        I = disc["Current [A]"].entries
        Q = disc["Discharge capacity [A.h]"].entries
        try:
            T = disc["Cell temperature [K]"].entries
            T_mean = float(np.nanmean(T))
        except Exception:
            T_mean = float("nan")

        # PyBaMM's "Discharge capacity [A.h]" is signed: positive when current
        # leaves the cell (discharge). Take the magnitude so Q_Ah is always
        # non-negative regardless of step direction or sign convention.
        Q_Ah = abs(float(Q[-1] - Q[0]))
        if q0 is None:
            q0 = Q_Ah if Q_Ah > 1e-6 else 1.0
        SOH = Q_Ah / q0 if q0 > 0 else float("nan")
        V_mean = float(np.nanmean(V))

        # DCIR estimate from the discharge onset (ΔV / ΔI in the first ~10 s)
        dcir_mOhm = float("nan")
        if len(I) > 5 and abs(I[3] - I[0]) > 0.01:
            dV = V[3] - V[0]
            dI = I[3] - I[0]
            if abs(dI) > 1e-6:
                dcir_mOhm = abs(dV / dI) * 1000.0

        # IC curve from this cycle's discharge
        v_grid, dqdv = extract_ic_curve(V, Q)
        peaks = find_ic_peaks(v_grid, dqdv, n_peaks=2)
        peak1 = peaks[0] if len(peaks) >= 1 else {"V": np.nan, "area": np.nan}
        peak2 = peaks[1] if len(peaks) >= 2 else {"V": np.nan, "area": np.nan}

        if save_ic_dir is not None:
            save_ic_dir.mkdir(parents=True, exist_ok=True)
            np.savez(save_ic_dir / f"ic_cycle_{n:04d}.npz",
                     V=v_grid, dQdV=dqdv,
                     peak1_V=peak1["V"], peak1_area=peak1["area"],
                     peak2_V=peak2["V"], peak2_area=peak2["area"])

        # End-of-cycle degradation state — take last value of the last step
        last_step = cycle.steps[-1]
        sei_thickness = _last_or_nan(last_step, "X-averaged negative SEI thickness [m]")
        lam_n = _last_or_nan(last_step, "Loss of active material in negative electrode [%]")
        lam_p = _last_or_nan(last_step, "Loss of active material in positive electrode [%]")
        lpl_n = _last_or_nan(last_step, "Loss of capacity to negative lithium plating [A.h]")

        rows.append({
            "cycle_n": n,
            "Q_Ah": Q_Ah,
            "SOH": SOH,
            "V_mean_discharge": V_mean,
            "dcir_mOhm": dcir_mOhm,
            "ic_peak1_V": peak1.get("V", np.nan),
            "ic_peak2_V": peak2.get("V", np.nan),
            "ic_peak1_area": peak1.get("area", np.nan),
            "ic_peak2_area": peak2.get("area", np.nan),
            "SEI_thickness_m": sei_thickness,
            "LAM_negative_pct": lam_n,
            "LAM_positive_pct": lam_p,
            "dead_lithium_Ah": lpl_n,
            "T_K": T_mean,
            **{k: params_used.get(k) for k in (
                "c_rate", "k_SEI_ms",
                "SEI_partial_molar_volume_m3mol",
                "lithium_plating_exchange_current_A_m2",
                "LAM_positive_rate_s",
                "LAM_negative_rate_s",
                "temperature_K",
                "sample_id",
            ) if k in params_used},
        })
    return pd.DataFrame(rows)


def _last_or_nan(step, key: str) -> float:
    try:
        arr = step[key].entries
        return float(arr.flat[-1])
    except Exception:
        return float("nan")
