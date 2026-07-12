"""
src/param_id/dcir_hppc.py
-------------------------
Extract resistance parameters from DCIR and HPPC tests.

For each detected current pulse we compute:

    R0    : instantaneous (ohmic + sampling-limited) resistance  [Ω]
            = - ΔV / ΔI at the pulse transition
    R1    : charge-transfer resistance of a first-order RC element [Ω]
    τ     : RC time constant   [s]
    C1    : double-layer capacitance = τ / R1   [F]

via a non-linear fit to

    V(t) = V0 - I * (R0 + R1 * (1 - exp(-t/τ)))

over the duration of each discharge pulse, where V0 is the rest voltage
just before the pulse and I is the pulse current magnitude (positive).

SOC at each pulse is estimated by coulomb counting from the end of the
preceding CCCV charge (assumed SOC ≈ 1) using the dataset's nominal
capacity. This is approximate but sufficient for indexing resistance vs SOC.

Important caveats
~~~~~~~~~~~~~~~~~
- Sampling is ~1 Hz, so the extracted "R0" is not the truly instantaneous
  resistance — it is the resistance at the first sample after the step,
  which is dominated by ohmic + electrolyte but already includes a small
  charge-transfer contribution.
- HPPC for this dataset only probes a narrow SOC window (≈ 0.97–1.00 SOC)
  because no SOC-step discharge is interleaved. R(SOC) outside that
  window is therefore not identifiable here.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data_loader import load_test  # noqa: E402


PULSE_DI_THRESHOLD_A = 20.0       # minimum |ΔI| to register a transition
PULSE_MIN_DURATION_S = 5.0        # ignore pulses shorter than this
PULSE_MAX_DURATION_S = 120.0      # ignore "pulses" that are really long discharges
REST_PRESAMPLE_S = 5.0            # rest-voltage window used as V0 estimate


@dataclass
class PulseFit:
    cell_id: str
    test: str            # "DCIR" or "HPPC"
    t_start_s: float
    duration_s: float
    direction: str       # "discharge" or "charge"
    I_A: float           # magnitude (positive)
    SOC_est: float
    V0_V: float          # rest voltage just before pulse
    R0_Ohm: float
    R1_Ohm: float
    tau_s: float
    C1_F: float
    rmse_mV: float


def _segments_by_current(df: pd.DataFrame, threshold_A: float = PULSE_DI_THRESHOLD_A
                        ) -> pd.DataFrame:
    """
    Annotate `df` (in place) with a `seg_id` column marking segments of
    approximately constant current, and return a per-segment summary frame.

    A new segment starts whenever |I[t] - I[t-1]| > threshold_A.
    """
    dI = df["current"].diff().abs().fillna(0.0)
    df["seg_id"] = (dI > threshold_A).cumsum()

    grouped = df.groupby("seg_id").agg(
        t_start=("time", "first"),
        t_end=("time", "last"),
        I_mean=("current", "mean"),
        V_start=("voltage", "first"),
        V_end=("voltage", "last"),
        n=("voltage", "size"),
    ).reset_index()
    grouped["duration_s"] = grouped["t_end"] - grouped["t_start"]
    return grouped


def _coulomb_counted_soc(df: pd.DataFrame, Q_nominal_Ah: float) -> pd.Series:
    """
    Estimate SOC at every row by integrating current since the most recent
    end-of-CCCV-charge (assumed to be SOC = 1.0).

    Falls back to integrating from the start of the file (SOC = 1.0 assumed)
    if no CCCV step is found.
    """
    d = df.sort_values("time").reset_index(drop=True)
    t = d["time"].to_numpy(dtype=float)
    I = d["current"].to_numpy(dtype=float)

    cccv = d.index[d["step_name"] == "CCCV_Chg"]
    if len(cccv) > 0:
        start = int(cccv[-1]) + 1
    else:
        start = 0

    dt = np.diff(t, prepend=t[0])
    Q = np.zeros_like(t)
    Q[start:] = np.cumsum((I[start:]) * dt[start:]) / 3600.0  # Ah, signed
    soc = 1.0 + Q / float(Q_nominal_Ah)
    return pd.Series(np.clip(soc, 0.0, 1.0), index=d.index, name="SOC_est")


def _fit_rc(t_rel: np.ndarray, v: np.ndarray, V_pre: float, dI: float
            ) -> tuple[float, float, float, float]:
    """
    Fit V(t) = V_pre + dI * (R0 + R1*(1 - exp(-t/τ))) to the pulse window.

    Here `dI = I_pulse - I_baseline` is signed (negative for discharge step
    onset, positive for charge step). The first time sample is skipped
    because it sits on the current transition itself.

    Returns (R0, R1, τ, rmse_V). NaNs if the fit fails.
    """
    from scipy.optimize import curve_fit

    if len(t_rel) < 5:
        return float("nan"), float("nan"), float("nan"), float("nan")

    # Skip the transition sample (t=0) which is on the step itself
    t = t_rel[1:]
    v_obs = v[1:]

    def model(t, R0, R1, tau):
        return V_pre + dI * (R0 + R1 * (1.0 - np.exp(-t / tau)))

    R0_0 = max(1e-4, abs((v_obs[0] - V_pre) / dI))
    R1_0 = max(1e-4, abs((v_obs[-1] - V_pre) / dI) - R0_0)
    tau_0 = max(1.0, float(t[-1]) / 4.0)

    p0 = [R0_0, R1_0, tau_0]
    bounds = ([1e-5, 1e-5, 0.1], [0.5, 0.5, 600.0])
    try:
        popt, _ = curve_fit(model, t, v_obs, p0=p0, bounds=bounds, maxfev=10000)
    except Exception:
        return float("nan"), float("nan"), float("nan"), float("nan")

    R0, R1, tau = (float(x) for x in popt)
    v_hat = model(t, R0, R1, tau)
    rmse_V = float(np.sqrt(np.mean((v_obs - v_hat) ** 2)))
    return R0, R1, tau, rmse_V


def extract_pulses(cell_id: str, test: str, Q_nominal_Ah: float = 105.0
                  ) -> list[PulseFit]:
    """
    Find all current pulses in DCIR or HPPC data for one cell and fit an RC
    model to each.

    Returns a list of PulseFit records (one per pulse).
    """
    df_full = load_test(test, cell_id=cell_id).sort_values("time").reset_index(drop=True)
    df_full["SOC_est"] = _coulomb_counted_soc(df_full, Q_nominal_Ah)

    segs = _segments_by_current(df_full)
    if len(segs) < 2:
        return []

    out: list[PulseFit] = []
    for i, row in segs.iterrows():
        I_mag = float(abs(row["I_mean"]))
        # Only consider proper test pulses, not the slow baseline discharge
        # or rest segments
        if I_mag < 30.0:
            continue
        if not (PULSE_MIN_DURATION_S <= row["duration_s"] <= PULSE_MAX_DURATION_S):
            continue

        pulse_rows = df_full[df_full["seg_id"] == row["seg_id"]].copy()
        if i == 0:
            continue
        prev_seg_id = segs.loc[i - 1, "seg_id"]
        prev_rows = df_full[df_full["seg_id"] == prev_seg_id]
        if prev_rows.empty:
            continue

        # V_pre and I_pre: mean over the tail of the previous segment.
        # The tail is the steady state immediately preceding the pulse,
        # so it works whether the prior segment was a rest (HPPC) or a
        # slow baseline discharge (DCIR).
        t_pp_end = float(prev_rows["time"].iloc[-1])
        tail = prev_rows[prev_rows["time"] >= t_pp_end - REST_PRESAMPLE_S]
        if tail.empty:
            tail = prev_rows.tail(5)
        V_pre = float(tail["voltage"].mean())
        I_pre = float(tail["current"].mean())

        I_pulse = float(pulse_rows["current"].mean())
        dI = I_pulse - I_pre  # signed step in applied current

        # Build (t, V) relative to pulse start
        t0 = float(pulse_rows["time"].iloc[0])
        t_rel = pulse_rows["time"].to_numpy(dtype=float) - t0
        v = pulse_rows["voltage"].to_numpy(dtype=float)
        direction = "discharge" if dI < 0 else "charge"

        R0, R1, tau, rmse_V = _fit_rc(t_rel, v, V_pre=V_pre, dI=dI)
        if not np.isfinite(R0):
            continue
        C1 = tau / R1 if R1 > 1e-9 else float("nan")

        soc_at_pulse = float(pulse_rows["SOC_est"].iloc[0])
        out.append(PulseFit(
            cell_id=cell_id, test=test,
            t_start_s=t0, duration_s=float(row["duration_s"]),
            direction=direction, I_A=I_mag,
            SOC_est=soc_at_pulse, V0_V=V_pre,
            R0_Ohm=R0, R1_Ohm=R1, tau_s=tau, C1_F=C1,
            rmse_mV=float(rmse_V * 1000.0),
        ))
    return out


def extract_all(cells: list[str], Q_nominal_Ah: float = 105.0) -> pd.DataFrame:
    rows: list[dict] = []
    for cid in cells:
        for test in ("DCIR", "HPPC"):
            try:
                pulses = extract_pulses(cid, test, Q_nominal_Ah=Q_nominal_Ah)
            except Exception as e:
                print(f"  ✗ cell {cid} {test}: {type(e).__name__}: {e}")
                continue
            for pf in pulses:
                rows.append(vars(pf))
            n_disc = sum(1 for p in pulses if p.direction == "discharge")
            n_chg = sum(1 for p in pulses if p.direction == "charge")
            if pulses:
                med_R0 = float(np.median([p.R0_Ohm for p in pulses if p.direction == "discharge"] or [np.nan])) * 1000
                med_R1 = float(np.median([p.R1_Ohm for p in pulses if p.direction == "discharge"] or [np.nan])) * 1000
                print(f"  ✓ cell {cid} {test}: {len(pulses)} pulses "
                      f"(disc={n_disc}, chg={n_chg})  "
                      f"median R0={med_R0:.2f} mΩ  median R1={med_R1:.2f} mΩ")
            else:
                print(f"  · cell {cid} {test}: no clean pulses found")
    return pd.DataFrame(rows)


def aggregate_per_cell(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-cell summary using DISCHARGE pulses only (more reliable for ohmic
    and charge-transfer extraction at typical HPPC currents).
    """
    if df.empty:
        return pd.DataFrame()
    d = df[df["direction"] == "discharge"]
    agg = d.groupby("cell_id").agg(
        n_pulses=("R0_Ohm", "size"),
        R0_median_Ohm=("R0_Ohm", "median"),
        R0_mad_Ohm=("R0_Ohm", lambda s: float(np.median(np.abs(s - np.median(s))))),
        R1_median_Ohm=("R1_Ohm", "median"),
        tau_median_s=("tau_s", "median"),
        C1_median_F=("C1_F", "median"),
        rmse_mV_median=("rmse_mV", "median"),
        SOC_min=("SOC_est", "min"),
        SOC_max=("SOC_est", "max"),
    ).reset_index()
    return agg


if __name__ == "__main__":
    import yaml
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg_path = Path("configs/dataset.yaml")
    cells: list[str] = []
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        cells = [str(c).zfill(4) for c in cfg.get("dataset", {}).get("selected_cells", [])]
    if not cells:
        cells = ["0005", "0006", "0007", "0008"]

    df = extract_all(cells, Q_nominal_Ah=105.0)
    if df.empty:
        print("No pulses extracted — check data and thresholds.")
        sys.exit(1)

    out_path = Path("data/processed/dcir_hppc_pulses.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved pulse table ({len(df)} rows) → {out_path}")

    summary = aggregate_per_cell(df)
    print("\nPer-cell summary (discharge pulses only):")
    print(summary.to_string(index=False))
    summary_path = Path("data/processed/dcir_hppc_summary.parquet")
    summary.to_parquet(summary_path, index=False)

    # Diagnostic plot: R0 vs SOC for each cell (discharge pulses)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cid, g in df[df["direction"] == "discharge"].groupby("cell_id"):
        ax.scatter(g["SOC_est"], g["R0_Ohm"] * 1000.0, label=cid, s=14, alpha=0.7)
    ax.set(xlabel="SOC (estimated)", ylabel=r"$R_0$ [mΩ]",
           title="DCIR/HPPC ohmic resistance (discharge pulses)")
    ax.legend(title="cell_id", fontsize=8)
    fig.tight_layout()
    fig_path = Path("outputs/results/dcir_hppc_R0.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {fig_path}")
