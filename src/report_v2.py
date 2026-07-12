"""
src/report_v2.py
----------------
Regenerate the corrected per-test plots for the EVE lab test report:

  1. OCV-SOC line plot (no area fill)                  → outputs/results/ocv_curves.png
  2. DCIR + HPPC pulse table (now includes charge R₀)  → data/processed/dcir_hppc_pulses.parquet
  3. HPPC 2RC fits                                     → data/processed/hppc_2rc_pulses.parquet
  4. GITT SOC-binned table + D at every 10 % SOC       → data/processed/gitt_per_10pct_soc.parquet
                                                          outputs/results/gitt_metrics_per10soc.png
                                                          outputs/results/gitt_D_per10soc.png
  5. Rate capability per C-rate (0.1–0.5 C × 3 rounds) → data/processed/rate_capability_per_crate.parquet
                                                          outputs/results/rate_capability.png
  6. Constant-power per-pulse (all 3 pulses)           → data/processed/constant_power_per_pulse.parquet
                                                          outputs/results/constantpower_curves.png
  7. Peak-power per ~10 % SOC point                    → data/processed/peakpower_per_soc.parquet
                                                          outputs/results/peakpower_pulse.png
  8. Self-discharge capacity retention                 → data/processed/selfdischarge_fit.parquet
                                                          outputs/results/selfdischarge_decay.png
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import load_test


OUT_PLOTS = Path("outputs/results")
OUT_DATA = Path("data/processed")
OUT_PLOTS.mkdir(parents=True, exist_ok=True)
OUT_DATA.mkdir(parents=True, exist_ok=True)

CELLS_COHORT = ["0005", "0006", "0007", "0008"]
Q_NOMINAL = 105.0   # Ah


# =========================================================================
# 1. OCV — measured line plot only (no area fill)
# =========================================================================
def plot_ocv_lines() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for cid in CELLS_COHORT:
        df = load_test("OCV_SOC", cell_id=cid).sort_values("time").reset_index(drop=True)
        disc = df[df["step_name"] == "CC_DChg"]
        q_abs = disc["capacity"].abs().to_numpy(dtype=float)
        soc = 1.0 - q_abs / q_abs.max()
        V = disc["voltage"].to_numpy(dtype=float)
        order = np.argsort(soc)
        ax.plot(soc[order], V[order], lw=1.2, label=cid, alpha=0.9)
    ax.set(xlabel="SOC", ylabel="Voltage [V]",
           title="OCV vs SOC (C/20 discharge, 25 °C)", xlim=(0, 1), ylim=(2.45, 3.7))
    ax.grid(True, alpha=0.3)
    ax.legend(title="cell", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PLOTS / "ocv_curves.png", dpi=150)
    plt.close(fig)
    print("wrote ocv_curves.png")


# =========================================================================
# 2-3. DCIR + HPPC re-extraction with charge R0 and 2RC option
# =========================================================================
def _detect_current_segments(df: pd.DataFrame, threshold_A: float = 20.0
                             ) -> pd.DataFrame:
    """Annotate `df` with a seg_id, return per-segment summary."""
    dI = df["current"].diff().abs().fillna(0.0)
    df["seg_id"] = (dI > threshold_A).cumsum()
    g = df.groupby("seg_id").agg(
        t_start=("time", "first"),
        t_end=("time", "last"),
        I_mean=("current", "mean"),
        V_start=("voltage", "first"),
        V_end=("voltage", "last"),
        n=("voltage", "size"),
    ).reset_index()
    g["duration_s"] = g["t_end"] - g["t_start"]
    return g


def _coulomb_counted_soc(df: pd.DataFrame, Q_Ah: float = Q_NOMINAL,
                         start_soc: float = 1.0) -> pd.Series:
    t = df["time"].to_numpy(dtype=float)
    I = df["current"].to_numpy(dtype=float)
    dt = np.diff(t, prepend=t[0])
    soc = start_soc + np.cumsum(I * dt) / 3600.0 / Q_Ah
    return pd.Series(np.clip(soc, 0.0, 1.0), index=df.index, name="SOC_est")


# ── OCV-anchored SOC (uses LFP_OCV_SOC_Table.xlsx) ───────────────────────
_OCV_TABLE_PATH = Path("LFP_OCV_SOC_Table.xlsx")
_OCV_V_GRID: np.ndarray | None = None
_SOC_FRAC_GRID: np.ndarray | None = None


def _load_ocv_table() -> tuple[np.ndarray, np.ndarray]:
    """Lazy-load and cache the canonical LFP OCV→SOC table."""
    global _OCV_V_GRID, _SOC_FRAC_GRID
    if _OCV_V_GRID is None or _SOC_FRAC_GRID is None:
        t = pd.read_excel(_OCV_TABLE_PATH, sheet_name="OCV_SOC_LFP").sort_values("OCV (V)")
        _OCV_V_GRID = t["OCV (V)"].to_numpy(dtype=float)
        _SOC_FRAC_GRID = t["SOC (%)"].to_numpy(dtype=float) / 100.0
    return _OCV_V_GRID, _SOC_FRAC_GRID


def _soc_from_ocv(v: float) -> float:
    Vg, Sg = _load_ocv_table()
    return float(np.interp(v, Vg, Sg))


def _ocv_anchored_soc(df: pd.DataFrame, Q_Ah: float = Q_NOMINAL,
                       min_rest_dur_s: float = 5.0,
                       max_rest_I_A: float = 0.5) -> pd.Series:
    """Coulomb-count SOC, anchored at the first sustained rest's end voltage
    looked up through the LFP OCV-SOC table. Falls back to the first-row
    voltage if no rest of sufficient duration exists.
    """
    d = df.sort_values("time").reset_index(drop=True)
    is_rest = d["current"].abs() < max_rest_I_A
    run_id = (is_rest != is_rest.shift()).cumsum()
    v_anchor, t_anchor = float(d["voltage"].iloc[0]), float(d["time"].iloc[0])
    for rid, g in d.groupby(run_id):
        if not is_rest.loc[g.index].iloc[0]:
            continue
        dur = float(g["time"].iloc[-1] - g["time"].iloc[0])
        if dur < min_rest_dur_s:
            continue
        v_anchor = float(g["voltage"].iloc[-1])
        t_anchor = float(g["time"].iloc[-1])
        break
    start_soc = _soc_from_ocv(v_anchor)

    t = d["time"].to_numpy(dtype=float)
    I = d["current"].to_numpy(dtype=float)
    dt = np.diff(t, prepend=t[0])
    cumQ = np.cumsum(I * dt) / 3600.0
    cumQ_anchor = float(np.interp(t_anchor, t, cumQ))
    soc = start_soc + (cumQ - cumQ_anchor) / Q_Ah
    return pd.Series(np.clip(soc, 0.0, 1.0), index=d.index, name="SOC_est")


def _fit_1rc(t_rel: np.ndarray, v: np.ndarray, V_pre: float, dI: float
             ) -> tuple[float, float, float, float]:
    """V(t) = V_pre + dI*(R0 + R1*(1-exp(-t/τ)))  →  (R0, R1, τ, RMSE_V)."""
    from scipy.optimize import curve_fit
    if len(t_rel) < 5:
        return float("nan"), float("nan"), float("nan"), float("nan")
    t = t_rel[1:]; v_obs = v[1:]

    def model(t, R0, R1, tau):
        return V_pre + dI * (R0 + R1 * (1 - np.exp(-t / tau)))

    R0_0 = max(1e-4, abs((v_obs[0] - V_pre) / dI))
    R1_0 = max(1e-4, abs((v_obs[-1] - V_pre) / dI) - R0_0)
    tau_0 = max(1.0, float(t[-1]) / 4.0)
    try:
        popt, _ = curve_fit(model, t, v_obs, p0=[R0_0, R1_0, tau_0],
                            bounds=([1e-5, 1e-5, 0.1], [0.5, 0.5, 600.0]),
                            maxfev=10000)
        rmse = float(np.sqrt(np.mean((v_obs - model(t, *popt)) ** 2)))
        return float(popt[0]), float(popt[1]), float(popt[2]), rmse
    except Exception:
        return float("nan"), float("nan"), float("nan"), float("nan")


def _fit_2rc(t_rel: np.ndarray, v: np.ndarray, V_pre: float, dI: float
             ) -> tuple[float, float, float, float, float, float]:
    """V(t) = V_pre + dI*(R0 + R1*(1-e^-t/τ1) + R2*(1-e^-t/τ2)).

    Returns (R0, R1, τ1, R2, τ2, RMSE_V).
    """
    from scipy.optimize import curve_fit
    if len(t_rel) < 10:
        nan = float("nan")
        return nan, nan, nan, nan, nan, nan
    t = t_rel[1:]; v_obs = v[1:]

    def model(t, R0, R1, tau1, R2, tau2):
        return V_pre + dI * (R0 + R1 * (1 - np.exp(-t / tau1))
                                + R2 * (1 - np.exp(-t / tau2)))

    # Seed: split R into fast (τ1≈1 s) and slow (τ2≈30 s) branches
    R0_0 = max(1e-4, abs((v_obs[0] - V_pre) / dI))
    R_rem = max(1e-4, abs((v_obs[-1] - V_pre) / dI) - R0_0)
    try:
        popt, _ = curve_fit(
            model, t, v_obs,
            p0=[R0_0, R_rem / 2, 1.0, R_rem / 2, 30.0],
            bounds=([1e-5, 1e-5, 0.05, 1e-5, 1.0],
                    [0.5,  0.5,  10.0, 0.5,  600.0]),
            maxfev=20000,
        )
        R0, R1, tau1, R2, tau2 = popt
        # Order so τ1 < τ2 (fast / slow convention)
        if tau1 > tau2:
            R1, tau1, R2, tau2 = R2, tau2, R1, tau1
        rmse = float(np.sqrt(np.mean((v_obs - model(t, *popt)) ** 2)))
        return float(R0), float(R1), float(tau1), float(R2), float(tau2), rmse
    except Exception:
        nan = float("nan")
        return nan, nan, nan, nan, nan, nan


def extract_dcir_hppc(cells=CELLS_COHORT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-extract DCIR + HPPC pulses including charge pulses (from CCCV onset)
    and produce both 1RC and 2RC fits per discharge pulse.

    Returns (one_rc_df, two_rc_df).
    """
    rows_1rc: list[dict] = []
    rows_2rc: list[dict] = []
    PULSE_PRESAMPLE_S = 5.0

    for cell_id in cells:
        for test in ("DCIR", "HPPC"):
            df = load_test(test, cell_id=cell_id).sort_values("time").reset_index(drop=True)
            df["SOC_est"] = _coulomb_counted_soc(df)
            segs = _detect_current_segments(df)
            if len(segs) < 2:
                continue

            for i, row in segs.iterrows():
                I_mag = abs(float(row["I_mean"]))
                duration = float(row["duration_s"])
                # Keep proper short pulses AND the long CCCV charge onsets
                # (long charge captures the charge-direction R0).
                is_short_pulse = (I_mag >= 30.0
                                  and 5.0 <= duration <= 120.0)
                is_charge_step = (i > 0
                                  and row["I_mean"] > 25.0   # charge direction
                                  and duration > 200.0)      # CCCV-like
                if not (is_short_pulse or is_charge_step):
                    continue
                if i == 0:
                    continue
                prev_seg_id = segs.loc[i - 1, "seg_id"]
                prev = df[df["seg_id"] == prev_seg_id]
                if prev.empty:
                    continue

                # V_pre / I_pre from end of preceding segment
                t_pp_end = float(prev["time"].iloc[-1])
                tail = prev[prev["time"] >= t_pp_end - PULSE_PRESAMPLE_S]
                if tail.empty:
                    tail = prev.tail(5)
                V_pre = float(tail["voltage"].mean())
                I_pre = float(tail["current"].mean())

                pulse_rows = df[df["seg_id"] == row["seg_id"]].copy()
                t0 = float(pulse_rows["time"].iloc[0])
                t_rel = pulse_rows["time"].to_numpy(dtype=float) - t0
                v = pulse_rows["voltage"].to_numpy(dtype=float)
                I_pulse = float(pulse_rows["current"].mean())
                dI = I_pulse - I_pre

                # For long charge steps, only fit the first 30 s so R0 is well-defined
                if is_charge_step:
                    mask = t_rel <= 30.0
                    t_rel = t_rel[mask]; v = v[mask]

                direction = "discharge" if dI < 0 else "charge"
                soc_at_pulse = float(pulse_rows["SOC_est"].iloc[0])

                # 1RC fit (skip for long charge steps — R1/τ meaningless there)
                if is_charge_step:
                    R0_1, R1_1, tau_1, rmse_1 = _fit_1rc(t_rel, v, V_pre, dI)
                else:
                    R0_1, R1_1, tau_1, rmse_1 = _fit_1rc(t_rel, v, V_pre, dI)

                rows_1rc.append({
                    "cell_id": cell_id, "test": test,
                    "t_start_s": t0, "duration_s": duration,
                    "direction": direction, "I_A": I_mag, "SOC_est": soc_at_pulse,
                    "V_pre_V": V_pre, "R0_Ohm": R0_1, "R1_Ohm": R1_1,
                    "tau_s": tau_1, "C1_F": (tau_1 / R1_1) if R1_1 and R1_1 > 1e-9 else float("nan"),
                    "rmse_mV": rmse_1 * 1000.0 if rmse_1 == rmse_1 else float("nan"),
                    "pulse_kind": "short" if is_short_pulse else "charge_onset",
                })

                # 2RC fit for short pulses in BOTH directions (per HPPC template).
                # Long CCCV-onset charge steps are excluded — R1/τ meaningless when
                # the cell is driven hard for thousands of seconds.
                if is_short_pulse:
                    R0, R1, tau1, R2, tau2, rmse_2 = _fit_2rc(t_rel, v, V_pre, dI)
                    rows_2rc.append({
                        "cell_id": cell_id, "test": test,
                        "t_start_s": t0, "duration_s": duration,
                        "direction": direction, "I_A": I_mag, "SOC_est": soc_at_pulse,
                        "V_pre_V": V_pre,
                        "R0_Ohm": R0, "R1_Ohm": R1, "tau1_s": tau1,
                        "R2_Ohm": R2, "tau2_s": tau2,
                        "rmse_mV": rmse_2 * 1000.0 if rmse_2 == rmse_2 else float("nan"),
                    })

    df_1rc = pd.DataFrame(rows_1rc)
    df_2rc = pd.DataFrame(rows_2rc)
    df_1rc.to_parquet(OUT_DATA / "dcir_hppc_pulses.parquet", index=False)
    df_2rc.to_parquet(OUT_DATA / "hppc_2rc_pulses.parquet", index=False)
    print(f"wrote dcir_hppc_pulses.parquet ({len(df_1rc)} rows) and "
          f"hppc_2rc_pulses.parquet ({len(df_2rc)} rows)")
    return df_1rc, df_2rc


def plot_hppc_r0_box(df_1rc: pd.DataFrame) -> None:
    d = df_1rc[(df_1rc["direction"] == "discharge")
                & (df_1rc["pulse_kind"] == "short")].copy()
    d["R0_mOhm"] = d["R0_Ohm"] * 1000
    cells = sorted(d["cell_id"].unique())
    data = [d.loc[d["cell_id"] == c, "R0_mOhm"].values for c in cells]
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(data, tick_labels=cells, patch_artist=True)
    for p in bp["boxes"]:
        p.set_facecolor("lightsteelblue")
    ax.axhline(1.8, ls="--", color="red", alpha=0.6, label="EVE LF105 spec ≤ 1.8 mΩ")
    ax.set(xlabel="cell", ylabel=r"R$_0$ [mΩ]",
           title="HPPC + DCIR discharge-pulse R₀ per cell")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PLOTS / "hppc_R0_box.png", dpi=150)
    plt.close(fig)
    print("wrote hppc_R0_box.png")


# =========================================================================
# 4-5. GITT — per-10%-SOC table + diffusion coefficient
# =========================================================================
def extract_gitt_per_10pct(cells=CELLS_COHORT,
                           particle_radius_m: float = 5e-6) -> pd.DataFrame:
    """
    For each cell, bucket GITT pulses into 10 %-SOC bins (10, 20, …, 100 %)
    and compute apparent solid-state diffusion coefficient via the classical
    simplified GITT relation:

        D_app = (4 / (π τ)) · (L · ΔE_s / ΔE_τ)²

    where τ is the pulse duration, ΔE_s the steady-state OCV change across the
    pulse + relaxation, ΔE_τ the voltage change during the pulse, and L the
    characteristic diffusion length (default = LFP particle radius 5 µm).
    """
    rows: list[dict] = []
    for cell_id in cells:
        df = load_test("GITT", cell_id=cell_id).sort_values("time").reset_index(drop=True)
        # OCV-anchored SOC (start anchor from first sustained rest's settled
        # voltage, looked up via LFP_OCV_SOC_Table.xlsx).
        df["SOC_est"] = _ocv_anchored_soc(df, Q_NOMINAL)

        prev_E_after = float("nan")  # carry the tail of last cycle's rest
        for cyc, g in df.groupby("cycle"):
            g = g.sort_values("time").reset_index(drop=True)
            is_pulse = g["current"].abs() > 0.01
            if not bool(is_pulse.any()):
                continue
            pulse_idx = np.where(is_pulse.to_numpy())[0]
            i0, i1 = int(pulse_idx[0]), int(pulse_idx[-1])
            pulse = g.iloc[i0:i1 + 1]
            rest_before = g.iloc[:i0]
            rest_after = g.iloc[i1 + 1:]

            tau = float(pulse["time"].iloc[-1] - pulse["time"].iloc[0])
            if tau <= 0:
                continue
            I_A = float(pulse["current"].abs().median())
            soc_at = float(pulse["SOC_est"].iloc[0])

            # ΔE_s: rest-to-rest. If this cycle has no rest_before, fall back
            # to the previous cycle's rest_after tail (which is the same
            # physical rest from the perspective of the protocol).
            if not rest_before.empty:
                E_before = float(rest_before["voltage"].tail(20).mean())
            else:
                E_before = prev_E_after
            E_after = float(rest_after["voltage"].tail(20).mean()) if not rest_after.empty else float("nan")
            dE_s = (E_after - E_before) if (np.isfinite(E_before) and np.isfinite(E_after)) else float("nan")
            prev_E_after = E_after if np.isfinite(E_after) else prev_E_after

            E_pulse_start = float(pulse["voltage"].head(5).mean())
            E_pulse_end = float(pulse["voltage"].tail(5).mean())
            dE_t = E_pulse_end - E_pulse_start

            # GITT relation requires ΔE_s to be meaningfully > 0. On the LFP
            # plateau ΔE_s is essentially zero (OCV is flat with SOC), making
            # the ratio noise-dominated and any "D_app" meaningless. We gate
            # by |ΔE_s| ≥ 2 mV — pulses below this threshold are reported as
            # NaN ("LFP plateau, GITT not applicable").
            DE_S_MIN_V = 2e-3
            ratio = (dE_s / dE_t) if (abs(dE_t) > 1e-9 and np.isfinite(dE_s)
                                        and abs(dE_s) >= DE_S_MIN_V) else float("nan")
            D_app = float("nan")
            if np.isfinite(ratio):
                L = particle_radius_m
                D_app = (4.0 * L * L / (np.pi * tau)) * (ratio ** 2)

            # Direction from the signed pulse current (positive = charge, negative = discharge)
            I_signed = float(pulse["current"].median())
            direction = "charge" if I_signed > 0 else "discharge"
            rows.append({
                "cell_id": cell_id, "cycle_step": int(cyc),
                "direction": direction,
                "SOC": soc_at, "tau_s": tau, "I_A": I_A,
                "dE_s_V": dE_s, "dE_tau_V": dE_t,
                "ratio_Es_Etau": ratio, "D_app_m2_s": D_app,
                "L_assumed_m": particle_radius_m,
            })
    df = pd.DataFrame(rows)

    # Now bucket per 10 % SOC
    if df.empty:
        return df
    df["soc_bin"] = (df["SOC"] * 10).round().astype("Int64") * 10  # 0,10,20,...,100
    df["soc_bin"] = df["soc_bin"].clip(0, 100)
    df.to_parquet(OUT_DATA / "gitt_per_10pct_soc.parquet", index=False)
    print(f"wrote gitt_per_10pct_soc.parquet ({len(df)} rows)")
    return df


def plot_gitt_per_10pct(gitt_df: pd.DataFrame) -> None:
    if gitt_df.empty:
        return
    # Compact summary: median ΔE_s, ΔE_τ, D per (cell, 10 % SOC bin)
    # Now grouped by direction as well so charge / discharge can be plotted
    # separately downstream.
    agg = (gitt_df.groupby(["cell_id", "soc_bin"])
                 .agg(dE_s_V=("dE_s_V", "median"),
                      dE_tau_V=("dE_tau_V", "median"),
                      D_app_m2_s=("D_app_m2_s", "median"),
                      n=("cycle_step", "size"))
                 .reset_index())

    # ── Plot 1: ΔE_s and ΔE_τ vs SOC, line + scatter, per cell ──
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for cid, g in agg.groupby("cell_id"):
        g = g.sort_values("soc_bin")
        axes[0].plot(g["soc_bin"], g["dE_s_V"] * 1000, marker="o", lw=1, alpha=0.85, label=cid)
        axes[1].plot(g["soc_bin"], g["dE_tau_V"] * 1000, marker="o", lw=1, alpha=0.85, label=cid)
    for ax, ylbl, title in [(axes[0], r"$\Delta E_s$ [mV]", "Rest-to-rest change"),
                            (axes[1], r"$\Delta E_\tau$ [mV]", "Pulse change")]:
        ax.set(xlabel="SOC [%]", ylabel=ylbl, title=title,
               xlim=(0, 100))
        ax.grid(True, alpha=0.3)
        ax.legend(title="cell", fontsize=7)
    fig.suptitle("GITT step metrics binned to every 10 % SOC")
    fig.tight_layout()
    fig.savefig(OUT_PLOTS / "gitt_metrics_per10soc.png", dpi=150)
    plt.close(fig)
    print("wrote gitt_metrics_per10soc.png")

    # ── Plot 2: D_app vs SOC, log-y, line + scatter ──
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cid, g in agg.groupby("cell_id"):
        g = g.sort_values("soc_bin")
        ax.plot(g["soc_bin"], g["D_app_m2_s"], marker="o", lw=1, alpha=0.85, label=cid)
    ax.set(xlabel="SOC [%]", ylabel=r"$D_{\rm app}$ [m$^2$/s]",
           title="Apparent solid-state diffusion coefficient (L = 5 µm assumed)",
           xlim=(0, 100), yscale="log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="cell", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PLOTS / "gitt_D_per10soc.png", dpi=150)
    plt.close(fig)
    print("wrote gitt_D_per10soc.png")


# =========================================================================
# 6. Rate capability — all 5 C-rates × 3 repeats
# =========================================================================
def extract_rate_capability(cells=CELLS_COHORT) -> pd.DataFrame:
    rows: list[dict] = []
    for cid in cells:
        df = load_test("RateCapability", cell_id=cid).sort_values("time").reset_index(drop=True)
        df["_seg"] = (df["step_name"] != df["step_name"].shift()).cumsum()
        seg = df.groupby("_seg").agg(
            step=("step_name", "first"),
            dur_s=("time", lambda s: float(s.max() - s.min())),
            I_mean=("current", "mean"),
            cap0=("capacity", "first"), capN=("capacity", "last"),
            V0=("voltage", "first"), VN=("voltage", "last"),
        ).reset_index(drop=True)
        # Keep proper full discharges (the 5 C-rate stages, ignore quick precon)
        disc = seg[(seg["step"] == "CC_DChg") & (seg["dur_s"] > 1000)].copy()
        disc["C_rate"] = (disc["I_mean"].abs() / Q_NOMINAL).round(2)
        disc["Q_Ah"] = (disc["capN"] - disc["cap0"]).abs()
        # Number the repeat (1, 2, 3) per C-rate
        disc["round"] = disc.groupby("C_rate").cumcount() + 1
        for _, r in disc.iterrows():
            rows.append({
                "cell_id": cid, "C_rate": float(r["C_rate"]),
                "round": int(r["round"]), "Q_Ah": float(r["Q_Ah"]),
                "I_A": float(abs(r["I_mean"])), "dur_s": float(r["dur_s"]),
                "V_start": float(r["V0"]), "V_end": float(r["VN"]),
            })
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_DATA / "rate_capability_per_crate.parquet", index=False)
    print(f"wrote rate_capability_per_crate.parquet ({len(df)} rows)")
    return df


def plot_rate_capability(rc_df: pd.DataFrame) -> None:
    if rc_df.empty: return
    # Median capacity per cell at each actual C-rate, with scatter for repeats
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for cid, g in rc_df.groupby("cell_id"):
        med = g.groupby("C_rate")["Q_Ah"].median().reset_index()
        ax.plot(med["C_rate"], med["Q_Ah"], marker="o", lw=1.1, alpha=0.85, label=cid)
        # show individual repeats as small dots
        ax.scatter(g["C_rate"], g["Q_Ah"], s=10, alpha=0.4)
    ax.set(xlabel="Discharge C-rate", ylabel="Capacity [Ah]",
           title="Rate capability — capacity vs C-rate (25 °C, 5 rates × 3 repeats)",
           xticks=[0.1, 0.2, 0.3, 0.4, 0.5])
    ax.grid(True, alpha=0.3)
    ax.legend(title="cell", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_PLOTS / "rate_capability.png", dpi=150)
    plt.close(fig)
    print("wrote rate_capability.png")


# =========================================================================
# 7. Constant power — three pulses, each a separate curve (no end→start link)
# =========================================================================
def extract_constant_power(cells=CELLS_COHORT) -> pd.DataFrame:
    rows: list[dict] = []
    for cid in cells:
        df = load_test("ConstantPower", cell_id=cid).sort_values("time").reset_index(drop=True)
        df["_seg"] = (df["step_name"] != df["step_name"].shift()).cumsum()
        for sid, g in df.groupby("_seg"):
            step = g["step_name"].iloc[0]
            if step != "CP_DChg":
                continue
            P_inst = -g["voltage"].to_numpy(dtype=float) * g["current"].to_numpy(dtype=float)
            energy_Wh = float(np.trapezoid(P_inst, g["time"].to_numpy()) / 3600.0)
            q_ah = float(g["capacity"].abs().max())
            t_rel = g["time"].to_numpy() - g["time"].iloc[0]
            rows.append({
                "cell_id": cid,
                "pulse_index": (df.loc[df["_seg"] <= sid, "step_name"] == "CP_DChg").sum(),
                "P_mean_W": float(np.mean(P_inst)),
                "P_peak_W": float(np.max(P_inst)),
                "energy_Wh": energy_Wh,
                "Q_Ah": q_ah,
                "dur_s": float(t_rel[-1]),
                # Keep per-pulse arrays for plotting (as JSON-like lists)
                # — instead we re-load in the plot fn to avoid huge parquet
            })
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_DATA / "constant_power_per_pulse.parquet", index=False)
    print(f"wrote constant_power_per_pulse.parquet ({len(df)} rows)")
    return df


def plot_constant_power(cp_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cid in CELLS_COHORT:
        df = load_test("ConstantPower", cell_id=cid).sort_values("time").reset_index(drop=True)
        df["_seg"] = (df["step_name"] != df["step_name"].shift()).cumsum()
        pulse_count = 0
        for sid, g in df.groupby("_seg"):
            if g["step_name"].iloc[0] != "CP_DChg":
                continue
            pulse_count += 1
            Q = g["capacity"].abs().to_numpy(dtype=float)
            V = g["voltage"].to_numpy(dtype=float)
            # Plot each pulse as its own line — DO NOT connect to next
            ax.plot(Q - Q[0], V, lw=1.0, alpha=0.85,
                    label=f"{cid} pulse {pulse_count}" if pulse_count <= 3 else None)
    ax.set(xlabel="Capacity within pulse [Ah]", ylabel="Voltage [V]",
           title="Constant-power discharge — V(Q) per pulse (3 × 90 W per cell)")
    ax.axhline(2.5, ls="--", color="red", alpha=0.5, label="cut-off")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_PLOTS / "constantpower_curves.png", dpi=150)
    plt.close(fig)
    print("wrote constantpower_curves.png")


# =========================================================================
# 8. Peak power — per 10 % SOC stage
# =========================================================================
def extract_peak_power(cells=CELLS_COHORT) -> pd.DataFrame:
    rows: list[dict] = []
    for cid in cells:
        df = load_test("PeakPower", cell_id=cid).sort_values("time").reset_index(drop=True)
        df["SOC_est"] = _coulomb_counted_soc(df, Q_NOMINAL, start_soc=0.0)  # starts depleted-ish
        df["_seg"] = (df["step_name"] != df["step_name"].shift()).cumsum()
        for sid, g in df.groupby("_seg"):
            step = g["step_name"].iloc[0]
            if step != "CC_DChg":
                continue
            I_mag = float(g["current"].abs().max())
            # Only count the "peak-power" pulses (high current, short)
            dur = float(g["time"].iloc[-1] - g["time"].iloc[0])
            if I_mag < 60.0 or dur > 60.0:
                continue
            P = (-g["voltage"] * g["current"]).max()
            soc = float(g["SOC_est"].iloc[0])
            rows.append({
                "cell_id": cid,
                "SOC": soc,
                "I_peak_A": I_mag,
                "P_peak_W": float(P),
                "dur_s": dur,
            })
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_DATA / "peakpower_per_soc.parquet", index=False)
    print(f"wrote peakpower_per_soc.parquet ({len(df)} rows)")
    return df


def plot_peak_power(pp_df: pd.DataFrame) -> None:
    if pp_df.empty:
        print("(peak-power dataframe empty — nothing to plot)")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cid, g in pp_df.groupby("cell_id"):
        g = g.sort_values("SOC")
        ax.plot(g["SOC"] * 100, g["P_peak_W"], marker="o", lw=1.1, alpha=0.85, label=cid)
    ax.set(xlabel="SOC [%]", ylabel="Peak power [W]",
           title="Peak-power output per SOC stage", xlim=(0, 100))
    ax.grid(True, alpha=0.3)
    ax.legend(title="cell", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PLOTS / "peakpower_pulse.png", dpi=150)
    plt.close(fig)
    print("wrote peakpower_pulse.png")


# =========================================================================
# 9. Self-discharge — add capacity retention rate
# =========================================================================
def extract_selfdischarge_with_retention(cells=CELLS_COHORT) -> pd.DataFrame:
    """Re-write the self-discharge fit, adding capacity-retention rate.

    Retention is computed as Q_after_rest / Q_before_rest using the full
    discharge that follows the long top-of-charge rest, against the
    full discharge that precedes the long rest.
    """
    rows: list[dict] = []
    for cid in cells:
        df = load_test("SelfDischarge", cell_id=cid).sort_values("time").reset_index(drop=True)
        df["_seg"] = (df["step_name"] != df["step_name"].shift()).cumsum()
        # Identify the long Rest segment (>= 24h)
        seg = df.groupby("_seg").agg(
            step=("step_name", "first"),
            dur_s=("time", lambda s: float(s.max() - s.min())),
        ).reset_index()
        long = seg[(seg["step"] == "Rest") & (seg["dur_s"] >= 24*3600)]
        if long.empty:
            continue
        long_seg_id = int(long.sort_values("dur_s").iloc[-1]["_seg"])
        rest = df[df["_seg"] == long_seg_id]
        rest_t = rest["time"].values - rest["time"].values[0]
        rest_V = rest["voltage"].values
        # late-time linear fit (skip first 24 h)
        m = rest_t >= 24 * 3600
        if m.sum() < 50:
            continue
        a, b = np.polyfit(rest_t[m], rest_V[m], 1)  # V/s
        dV_dt_uV_per_s = a * 1e6

        # Find discharge segments BEFORE and AFTER the long rest
        discharge_segs = seg[seg["step"] == "CC_DChg"].copy()
        discharge_segs["mid_seg"] = discharge_segs["_seg"]
        before = discharge_segs[discharge_segs["mid_seg"] < long_seg_id]
        after = discharge_segs[discharge_segs["mid_seg"] > long_seg_id]
        if before.empty or after.empty:
            retention = float("nan")
            Q_before = Q_after = float("nan")
        else:
            sid_b = int(before["mid_seg"].iloc[-1])
            sid_a = int(after["mid_seg"].iloc[0])
            d_before = df[df["_seg"] == sid_b]
            d_after = df[df["_seg"] == sid_a]
            Q_before = float(d_before["capacity"].abs().max())
            Q_after = float(d_after["capacity"].abs().max())
            retention = Q_after / Q_before if Q_before > 0 else float("nan")

        # I_sd_late from the late-time slope dV/dt mapped through dV/dQ at the
        # top-of-charge kink. The top-of-OCV slope from this dataset is ≈ 72
        # V per unit-SOC (steep kink between 3.31 V plateau and 3.65 V CV).
        # I_sd_late = Q · |dV/dt| / |dV/dSOC|, conservative because it
        # excludes the early polarization-decay portion of the rest.
        DV_DSOC_TOP_BRANCH = 72.0   # V / unit-SOC near top of LFP OCV
        dSOC_dt_per_s = abs(a) / DV_DSOC_TOP_BRANCH
        I_sd_late_uA = float(Q_NOMINAL * dSOC_dt_per_s * 3600.0 * 1e6)

        # I_sd_total via ΔQ over the full rest (includes initial polarization
        # decay so this is an UPPER bound, not the true self-discharge rate).
        dQ_Ah = (Q_before - Q_after) if (np.isfinite(Q_before) and np.isfinite(Q_after)) else float("nan")
        I_sd_total_uA = (dQ_Ah * 1e6 / (float(rest_t[-1]) / 3600.0)
                        if (np.isfinite(dQ_Ah) and rest_t[-1] > 0) else float("nan"))

        rows.append({
            "cell_id": cid,
            "rest_duration_s": float(rest_t[-1]),
            "rest_duration_h": float(rest_t[-1]) / 3600.0,
            "V_start_V": float(rest_V[0]),
            "V_end_V": float(rest_V[-1]),
            "dV_dt_uV_per_s": float(dV_dt_uV_per_s),
            "Q_before_rest_Ah": Q_before,
            "Q_after_rest_Ah": Q_after,
            "capacity_retention": retention,
            "capacity_retention_pct": retention * 100 if np.isfinite(retention) else float("nan"),
            "delta_Q_Ah": dQ_Ah,
            "I_sd_late_uA": I_sd_late_uA,
            "I_sd_total_upper_uA": I_sd_total_uA,
        })

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_DATA / "selfdischarge_fit.parquet", index=False)
    print(f"wrote selfdischarge_fit.parquet ({len(df)} rows)")
    return df


def plot_selfdischarge(cells=CELLS_COHORT) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for cid in cells:
        try:
            df = load_test("SelfDischarge", cell_id=cid).sort_values("time").reset_index(drop=True)
            df["_seg"] = (df["step_name"] != df["step_name"].shift()).cumsum()
            seg = df.groupby("_seg").agg(
                step=("step_name", "first"),
                dur_s=("time", lambda s: float(s.max() - s.min())),
            ).reset_index()
            long = seg[(seg["step"] == "Rest") & (seg["dur_s"] >= 24*3600)]
            if long.empty:
                continue
            long_seg_id = int(long.sort_values("dur_s").iloc[-1]["_seg"])
            rest = df[df["_seg"] == long_seg_id]
            t_h = (rest["time"].values - rest["time"].values[0]) / 3600.0
            ax.plot(t_h, rest["voltage"].values, lw=1.0, label=cid)
        except Exception:
            continue
    ax.set(xlabel="Hours since top-of-charge", ylabel="OCV [V]",
           title="Self-discharge OCV decay — top-of-charge rest")
    ax.grid(True, alpha=0.3)
    ax.legend(title="cell", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PLOTS / "selfdischarge_decay.png", dpi=150)
    plt.close(fig)
    print("wrote selfdischarge_decay.png")


# =========================================================================
# Driver
# =========================================================================
def main():
    print("=" * 60)
    print("Regenerating corrected lab-report artifacts")
    print("=" * 60)
    print()

    print("[1/8] OCV line plot")
    plot_ocv_lines()

    print()
    print("[2-3/8] DCIR + HPPC re-extraction with charge R0 and 2RC")
    df_1rc, df_2rc = extract_dcir_hppc()
    plot_hppc_r0_box(df_1rc)

    print()
    print("[4-5/8] GITT per-10%-SOC + D_app")
    gitt_df = extract_gitt_per_10pct()
    plot_gitt_per_10pct(gitt_df)

    print()
    print("[6/8] Rate capability per C-rate")
    rc_df = extract_rate_capability()
    plot_rate_capability(rc_df)

    print()
    print("[7/8] Constant power — 3 pulses each, no end→start linking")
    cp_df = extract_constant_power()
    plot_constant_power(cp_df)

    print()
    print("[8/8] Peak power per ~10 % SOC stage")
    pp_df = extract_peak_power()
    plot_peak_power(pp_df)

    print()
    print("[+] Self-discharge with retention")
    extract_selfdischarge_with_retention()
    plot_selfdischarge()

    print()
    print("All artifacts written.")


if __name__ == "__main__":
    main()
