"""
src/param_id/gitt_ds.py
Process GITT steps into *defensible* step-level metrics and (optionally)
an **apparent** diffusion coefficient using the classical GITT relation.

Important scientific note (why this file was rewritten):
- Full-cell GITT voltage contains contributions from *both* electrodes plus
  ohmic and kinetic overpotentials. Without half-cell measurements or a
  model-based fit, you should not claim separate Ds for graphite and LFP.
- The common "GITT diffusivity" formula requires a diffusion length (L)
  and careful definitions of ΔEs and ΔEτ; results can vary by orders of
  magnitude if these are chosen poorly.

This module therefore:
  1) Computes step-level metrics (ΔEs, ΔEτ, dV/d√t, τ) that are data-driven.
  2) Computes an *apparent* D only if you explicitly provide a diffusion
     length L (meters). Otherwise it reports NaN for D_app_m2s.

References for the classical relation and pitfalls:
- D^(GITT) = (4 L^2 / (π τ)) * (ΔEs / ΔEτ)^2  (common simplified form)
- See e.g. discussions of GITT assumptions/pitfalls in the literature.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from src.data_loader import load_gitt_pulses

def _window_mean_end(df: pd.DataFrame, seconds: float) -> float:
    """Mean voltage over the last `seconds` of df (requires monotonic 'time')."""
    if df.empty:
        return float("nan")
    t_end = float(df["time"].iloc[-1])
    w = df[df["time"] >= (t_end - seconds)]
    if w.empty:
        w = df.tail(max(1, int(seconds)))
    return float(w["voltage"].mean())

def _window_mean_start(df: pd.DataFrame, seconds: float) -> float:
    """Mean voltage over the first `seconds` of df (requires monotonic 'time')."""
    if df.empty:
        return float("nan")
    t0 = float(df["time"].iloc[0])
    w = df[df["time"] <= (t0 + seconds)]
    if w.empty:
        w = df.head(max(1, int(seconds)))
    return float(w["voltage"].mean())


def compute_soc_from_gitt(df: pd.DataFrame, Q_total_Ah: float) -> np.ndarray:
    """Estimate SOC at each pulse from cumulative charge passed."""
    t = df["time"].values.astype(float)
    i = df["current"].values.astype(float)
    dt = np.gradient(t)
    charge = np.cumsum(np.abs(i) * dt) / 3600.0  # Ah
    soc = 1.0 - charge / float(Q_total_Ah)
    return np.clip(soc, 0.0, 1.0)


def extract_gitt_step_metrics(
    cell_id: str,
    Q_total_Ah: float | None = None,
    diffusion_length_m: float | None = None,
    I_threshold_A: float = 0.01,
    fit_start_s: float = 5.0,
    fit_end_s: float = 60.0,
    rest_window_s: float = 30.0,
) -> pd.DataFrame:
    """
    Extract step-level metrics from full-cell GITT.

    Args:
        cell_id: standardised cell identifier (e.g. "0005")
        Q_total_Ah: nominal capacity used for SOC estimation. If None, tries
            to infer from the raw 'max_cap' column (EVE exports).
        diffusion_length_m: diffusion length L (meters). If provided, computes
            an *apparent* D_app_m2s from the simplified GITT relation:
                D_app = (4 L^2 / (pi tau)) * (ΔEs/ΔEτ)^2
            If None, D_app_m2s is reported as NaN.

    Returns:
        DataFrame with columns:
          cycle_step, soc, I_A, tau_s, dV_dsqrt_t_V_sqrt_s, fit_r2,
          delta_Es_V, delta_Etau_V, ratio_Es_over_Etau, D_app_m2s
    """
    df = load_gitt_pulses(cell_id)
    if Q_total_Ah is None:
        if "max_cap" in df.columns and df["max_cap"].notna().any():
            Q_total_Ah = float(df["max_cap"].dropna().iloc[0])
        else:
            # As a fallback, user must specify; do not guess from GITT step capacity
            raise ValueError("Q_total_Ah not provided and could not infer from 'max_cap'")

    df = df.sort_values("time").reset_index(drop=True)
    soc_series = compute_soc_from_gitt(df, Q_total_Ah)
    df["soc_est"] = soc_series

    if "cycle" not in df.columns:
        raise ValueError("GITT data must include a 'cycle' column (e.g. cycle_no)")

    results: list[dict] = []
    prev_rest_end_voltage = float("nan")
    for step in sorted(df["cycle"].dropna().unique()):
        step_df = df[df["cycle"] == step].sort_values("time").reset_index(drop=True)
        is_pulse = step_df["current"].abs() > I_threshold_A
        if not bool(is_pulse.any()):
            continue

        pulse_idx = np.where(is_pulse.to_numpy())[0]
        i0 = int(pulse_idx[0])
        i1 = int(pulse_idx[-1])
        pulse = step_df.iloc[i0:i1 + 1]
        rest_before = step_df.iloc[:i0]
        rest_after = step_df.iloc[i1 + 1:]

        # Pulse duration
        tau_s = float(pulse["time"].iloc[-1] - pulse["time"].iloc[0])
        if tau_s <= 0:
            continue

        # SOC estimate at pulse start (from global coulomb counting)
        soc_at_step = float(pulse["soc_est"].iloc[0])

        # Current magnitude
        I_A = float(pulse["current"].abs().median())

        # ΔEs: steady-state voltage change (rest-before -> rest-after)
        if not rest_before.empty:
            E_before = _window_mean_end(rest_before, rest_window_s)
        else:
            E_before = prev_rest_end_voltage
        E_after = _window_mean_end(rest_after, rest_window_s)
        prev_rest_end_voltage = E_after
        delta_Es_V = float(E_after - E_before) if np.isfinite(E_before) and np.isfinite(E_after) else float("nan")

        # ΔEτ: voltage change during pulse (smoothed by short windows)
        # Note: this still contains ohmic/kinetic contributions; treat as an approximation.
        pulse_window_s = min(5.0, max(1.0, tau_s / 10))
        E_pulse_start = _window_mean_start(pulse, seconds=pulse_window_s)
        E_pulse_end = _window_mean_end(pulse, seconds=pulse_window_s)
        delta_Etau_V = float(E_pulse_end - E_pulse_start)

        ratio = float("nan")
        if np.isfinite(delta_Es_V) and np.isfinite(delta_Etau_V) and abs(delta_Etau_V) > 1e-9:
            ratio = float(delta_Es_V / delta_Etau_V)

        # dV/dsqrt(t) fit on early-time portion of pulse
        t_rel = (pulse["time"].values.astype(float) - float(pulse["time"].iloc[0]))
        V = pulse["voltage"].values.astype(float)
        fit_mask = (t_rel >= fit_start_s) & (t_rel <= min(fit_end_s, t_rel.max()))
        dV_dsqrt_t = float("nan")
        fit_r2 = float("nan")
        if fit_mask.sum() >= 8:
            x = np.sqrt(t_rel[fit_mask])
            y = V[fit_mask]
            a, b = np.polyfit(x, y, 1)  # y ≈ a*x + b
            yhat = a * x + b
            ss_res = float(((y - yhat) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            fit_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            dV_dsqrt_t = float(a)

        # Apparent diffusion coefficient if L provided
        D_app = float("nan")
        if diffusion_length_m is not None and np.isfinite(ratio):
            L = float(diffusion_length_m)
            if L > 0:
                D_app = (4.0 * L * L / (np.pi * tau_s)) * (ratio ** 2)

        results.append(
            {
                "cell_id": cell_id,
                "cycle_step": int(step),
                "soc": soc_at_step,
                "I_A": I_A,
                "tau_s": tau_s,
                "dV_dsqrt_t_V_sqrt_s": dV_dsqrt_t,
                "fit_r2": fit_r2,
                "delta_Es_V": delta_Es_V,
                "delta_Etau_V": delta_Etau_V,
                "ratio_Es_over_Etau": ratio,
                "D_app_m2s": D_app,
            }
        )

    out = pd.DataFrame(results)
    if out.empty:
        return out
    return out.sort_values(["cell_id", "cycle_step"]).reset_index(drop=True)


def export_gitt_metrics(metrics_df: pd.DataFrame, out_path: Path) -> None:
    """Save GITT step metrics as parquet for auditability."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_parquet(out_path, index=False)
    print(f"Saved GITT metrics ({len(metrics_df):,} steps) → {out_path}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Replace with your cell id. To see available IDs, run:
    #   from src.data_loader import list_cells; print(list_cells("GITT"))
    cell = "0005"
    # If you want an apparent diffusivity in m^2/s, you MUST provide an assumed diffusion length.
    # For example, use a representative particle radius (order ~1e-6 to 1e-5 m) and be explicit
    # about this assumption in any write-up.
    metrics = extract_gitt_step_metrics(cell_id=cell, Q_total_Ah=105.0, diffusion_length_m=None)
    print(metrics.head())
    export_gitt_metrics(metrics, Path(f"data/processed/gitt_metrics_cell_{cell}.parquet"))

    if not metrics.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(metrics["cycle_step"], metrics["delta_Es_V"], label="ΔEs (rest) [V]")
        ax.plot(metrics["cycle_step"], metrics["delta_Etau_V"], label="ΔEτ (pulse) [V]", alpha=0.8)
        ax.set(xlabel="GITT step (cycle_no)", ylabel="Voltage change [V]",
               title=f"GITT step metrics — cell {cell}")
        ax.legend()
        plt.tight_layout()
        Path("outputs/results").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"outputs/results/gitt_metrics_{cell}.png", dpi=150)
        print(f"Saved plot to outputs/results/gitt_metrics_{cell}.png")
