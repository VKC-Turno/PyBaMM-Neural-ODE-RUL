"""
Extraction utilities for the CALB/REPT/EVE data-quality review.

Reads raw characterization CSVs from ``PINNs/Data/`` and the canonical SoH
parquets from ``PINNs/soh/data/canonical/``. Returns simple per-cell scalar
tables and per-cell SoH-vs-cycle traces.

The extractors intentionally use robust heuristics (percentiles, medians,
sign-flip pulse detection) rather than reusing ``src/param_id/*.py`` — the
existing param_id code has EVE-specific assumptions.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path("/home/hj/Desktop/PINNs")
DATA_ROOT = PROJECT_ROOT / "Data"
CANONICAL_SOH = PROJECT_ROOT / "soh" / "data" / "canonical"
OUT_DIR = PROJECT_ROOT / "Voltaris" / "Data_Exploration"

MAKES = ("CALB", "REPT", "EVE")

# Colours (fixed across figures)
MAKE_COLOR = {
    "CALB": "#c94a3c",
    "REPT": "#3c7cc9",
    "EVE": "#4dab5c",
}

# Physical bounds used to flag a cell as suspect.
BOUNDS: dict[str, tuple[float, float]] = {
    "V_top": (3.35, 3.75),
    "V_bottom": (2.30, 2.80),
    "V_plateau": (3.15, 3.35),
    "R0_mOhm": (0.5, 3.0),
    "coulombic_efficiency_pct": (95.0, 100.0),
    # capacity: BOL >= 90, second-life may be lower; only flag < 40
    "capacity_Ah": (40.0, 115.0),
    "dV_per_sqrt_t": (-1e-2, -1e-4),  # negative during discharge
    # monotone_frac lower-bound only
    "monotone_frac": (0.60, np.inf),
}


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #
_CELL_RE = re.compile(r"cell_(\d+)")


def _cell_id_from_name(name: str) -> str | None:
    m = _CELL_RE.search(name)
    return m.group(1) if m else None


def list_cells(test_folder: str, make: str) -> list[tuple[str, Path]]:
    """Return sorted [(cell_id, csv_path), ...] for a given test/make."""
    root = DATA_ROOT / test_folder
    if not root.exists():
        return []
    out: list[tuple[str, Path]] = []
    for p in sorted(root.glob(f"{make}_*.csv")):
        cid = _cell_id_from_name(p.name)
        if cid is None:
            continue
        # keep only the canonical filename per cell (skip _b2 variants for the
        # scalar table — they are duplicates of the first HPPC/DCIR pass).
        if "_b2" in p.stem:
            continue
        out.append((cid, p))
    return out


# --------------------------------------------------------------------------- #
# Small numerical helpers
# --------------------------------------------------------------------------- #
def _time_seconds(df: pd.DataFrame) -> np.ndarray:
    """Robust absolute_time -> monotone seconds array."""
    if "absolute_time" not in df.columns:
        return np.arange(len(df), dtype=float)
    t = pd.to_datetime(df["absolute_time"], errors="coerce")
    if t.isna().all():
        return np.arange(len(df), dtype=float)
    sec = (t - t.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    # some rows may parse to NaT; forward-fill
    if np.isnan(sec).any():
        for i in range(1, len(sec)):
            if np.isnan(sec[i]):
                sec[i] = sec[i - 1]
        sec = np.nan_to_num(sec, nan=0.0)
    return sec


def _label_pulse_events(current: np.ndarray, time_s: np.ndarray,
                        thresh_A: float = 0.5,
                        min_gap_s: float = 5.0) -> list[tuple[int, int]]:
    """Return list of (start_idx, end_idx) for contiguous non-zero current runs.

    A pulse is a run of samples where |I| > thresh_A. Runs closer than
    ``min_gap_s`` are merged.
    """
    if len(current) == 0:
        return []
    active = np.abs(current) > thresh_A
    idx = np.arange(len(current))
    if not active.any():
        return []
    # boundaries
    diff = np.diff(active.astype(int))
    starts = list(idx[1:][diff == 1])
    ends = list(idx[1:][diff == -1])
    if active[0]:
        starts = [0] + starts
    if active[-1]:
        ends = ends + [len(current) - 1]
    pairs = list(zip(starts, ends))
    # merge close pairs
    merged: list[tuple[int, int]] = []
    for s, e in pairs:
        if merged and time_s[s] - time_s[merged[-1][1]] < min_gap_s:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


# --------------------------------------------------------------------------- #
# Per-test scalar extraction
# --------------------------------------------------------------------------- #
def extract_ocv_scalars(csv: Path) -> dict:
    df = pd.read_csv(csv, usecols=["step_name", "volt_v", "current_a"])
    dis = df[df["step_name"].str.contains("dchg|discharge", case=False, na=False)]
    if dis.empty:
        return {}
    v = dis["volt_v"].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 20:
        return {}
    v_top = float(np.percentile(v, 99))
    v_bottom = float(np.percentile(v, 1))
    mid_lo, mid_hi = np.percentile(v, [30, 70])
    plateau = v[(v >= mid_lo) & (v <= mid_hi)]
    v_plateau = float(np.median(plateau)) if plateau.size else float(np.median(v))
    return {
        "V_top": v_top,
        "V_bottom": v_bottom,
        "V_plateau": v_plateau,
        "dV_total": v_top - v_bottom,
        "n_points_ocv": int(v.size),
    }


def extract_ocv_curve(csv: Path,
                      max_samples: int = 4000) -> pd.DataFrame | None:
    """Return a decimated discharge branch (SoC%, V) for overlay plotting."""
    df = pd.read_csv(csv, usecols=["step_name", "volt_v", "current_a",
                                    "capacity_ah"])
    dis = df[df["step_name"].str.contains("dchg|discharge", case=False, na=False)]
    if dis.empty:
        return None
    cap = dis["capacity_ah"].to_numpy(dtype=float)
    v = dis["volt_v"].to_numpy(dtype=float)
    ok = np.isfinite(cap) & np.isfinite(v)
    cap = cap[ok]
    v = v[ok]
    if cap.size < 50:
        return None
    # capacity_ah is a step-relative counter; can be positive (magnitude) or
    # negative (signed discharge). Work with absolute magnitude.
    abs_cap = np.abs(cap)
    q_max = float(np.nanmax(abs_cap))
    if q_max <= 0:
        return None
    soc_pct = 100.0 * (1.0 - abs_cap / q_max)  # 100 -> 0 during discharge
    # decimate
    if len(soc_pct) > max_samples:
        step = int(len(soc_pct) / max_samples)
        soc_pct = soc_pct[::step]
        v = v[::step]
    return pd.DataFrame({"soc_pct": soc_pct, "v": v})


def _fit_exp_tau(t_s: np.ndarray, v: np.ndarray) -> float:
    """Fit V(t) = V_inf + dV * exp(-t/tau) to a rest segment.

    Uses log-linear fit on |V - V_inf_est| where V_inf_est is the tail-mean.
    Robust to noise, no scipy dependency. Returns tau in seconds, or NaN.
    """
    if len(t_s) < 15:
        return np.nan
    v_inf = float(np.percentile(v[-max(3, len(v) // 5):], 50))
    dv = v - v_inf
    # Sign flip if the tail is above the start (relaxation direction depends
    # on whether preceding step was charge or discharge)
    if abs(dv[0]) < 1e-4:
        return np.nan
    sign = 1.0 if dv[0] > 0 else -1.0
    signed_dv = sign * dv
    pos = signed_dv > 1e-4
    if pos.sum() < 8:
        return np.nan
    try:
        log_dv = np.log(signed_dv[pos])
        slope, _ = np.polyfit(t_s[pos], log_dv, 1)
        if slope >= 0:  # not decaying — bad fit
            return np.nan
        tau = -1.0 / slope
        return float(tau) if 5.0 < tau < 50000.0 else np.nan
    except Exception:
        return np.nan


def extract_gitt_scalars(csv: Path) -> dict:
    df = pd.read_csv(csv, usecols=["step_name", "absolute_time",
                                    "volt_v", "current_a"])
    if df.empty:
        return {}
    t = _time_seconds(df)
    i = df["current_a"].to_numpy(dtype=float)
    v = df["volt_v"].to_numpy(dtype=float)
    step = df["step_name"].astype(str)

    # focus on discharge pulses (negative current beyond threshold)
    pulses = _label_pulse_events(i, t, thresh_A=1.0, min_gap_s=30.0)
    dis_pulses = [(s, e) for (s, e) in pulses if np.nanmean(i[s:e + 1]) < 0]
    durations: list[float] = []
    slopes: list[float] = []
    for s, e in dis_pulses:
        dur = float(t[e] - t[s])
        if dur < 5.0 or dur > 6000.0:
            continue
        durations.append(dur)
        seg_t = t[s:e + 1] - t[s]
        seg_v = v[s:e + 1]
        seg_ok = np.isfinite(seg_t) & np.isfinite(seg_v) & (seg_t > 1e-3)
        if seg_ok.sum() < 8:
            continue
        sqrt_t = np.sqrt(seg_t[seg_ok])
        # slope over the middle 20-80 % of the pulse (avoid ohmic overshoot)
        lo, hi = int(0.2 * seg_ok.sum()), int(0.8 * seg_ok.sum())
        if hi - lo < 5:
            continue
        try:
            slope, _ = np.polyfit(sqrt_t[lo:hi], seg_v[seg_ok][lo:hi], 1)
            slopes.append(float(slope))
        except Exception:
            continue

    # ---- tau_diff from post-pulse rest segments ----
    # Iterate contiguous segments; process each Rest that lasts long enough
    # to capture the diffusion relaxation (>200 s).
    taus: list[float] = []
    segs = _segments_by_step(step)
    for seg_s, seg_e, name in segs:
        if "rest" not in name.lower():
            continue
        seg_dur = float(t[seg_e] - t[seg_s])
        if seg_dur < 200.0 or seg_e - seg_s < 40:
            continue
        seg_t = t[seg_s:seg_e + 1] - t[seg_s]
        seg_v = v[seg_s:seg_e + 1]
        ok = np.isfinite(seg_t) & np.isfinite(seg_v)
        if ok.sum() < 30:
            continue
        tau = _fit_exp_tau(seg_t[ok], seg_v[ok])
        if np.isfinite(tau):
            taus.append(tau)

    return {
        "gitt_n_pulses": len(dis_pulses),
        "mean_pulse_duration_s": float(np.median(durations)) if durations else np.nan,
        "dV_per_sqrt_t": float(np.median(slopes)) if slopes else np.nan,
        "GITT_tau_diff_s": float(np.median(taus)) if len(taus) >= 3 else np.nan,
        "GITT_tau_n": len(taus),
    }


def extract_selfdisch_scalars(csv: Path) -> dict:
    """Self-discharge dSoC/day from the longest high-voltage rest segment.

    Approach:
      1. Segment by step_name; identify Rest segments starting at V > 3.3 V
         (LFP high-SoC / charged state) and lasting > 24 h.
      2. Take the LONGEST qualifying segment (the true self-discharge hold).
      3. Linear fit V vs t → dV/day.
      4. Convert to dSoC/day using LFP dV/dSoC ≈ 0.5 V per unit SoC at the
         high-SoC knee (order-of-magnitude estimate; comparable-in-sign
         to the pre-processed value).

    Returns dV_per_day (V/day, negative if V drops) and dsoc_per_day
    (fraction/day, positive if cell self-discharges).
    """
    try:
        df = pd.read_csv(csv, usecols=["step_name", "absolute_time",
                                         "volt_v", "current_a"])
    except Exception:
        return {"self_disch_dV_per_day": np.nan,
                "self_disch_dsoc_per_day": np.nan,
                "self_disch_rest_duration_h": np.nan}
    if df.empty:
        return {"self_disch_dV_per_day": np.nan,
                "self_disch_dsoc_per_day": np.nan,
                "self_disch_rest_duration_h": np.nan}
    t = _time_seconds(df)
    v = df["volt_v"].to_numpy(dtype=float)
    step = df["step_name"].astype(str)

    best = None  # (duration_h, v_start, v_end, seg_t, seg_v)
    for seg_s, seg_e, name in _segments_by_step(step):
        if "rest" not in name.lower():
            continue
        if seg_e - seg_s < 30:
            continue
        v_start = float(v[seg_s])
        if v_start < 3.3:
            continue
        dur_s = float(t[seg_e] - t[seg_s])
        if dur_s < 24 * 3600:
            continue
        # discard segments with time discontinuities (data gaps > 1 h)
        seg_t = t[seg_s:seg_e + 1]
        seg_v = v[seg_s:seg_e + 1]
        dts = np.diff(seg_t)
        # require > 90 % of samples with dt < 3600 s
        if len(dts) == 0 or (np.mean(dts < 3600.0) < 0.9):
            continue
        dur_h = dur_s / 3600.0
        v_end = float(v[seg_e])
        if best is None or dur_h > best[0]:
            best = (dur_h, v_start, v_end, seg_t - seg_t[0], seg_v)

    if best is None:
        return {"self_disch_dV_per_day": np.nan,
                "self_disch_dsoc_per_day": np.nan,
                "self_disch_rest_duration_h": np.nan}

    dur_h, v_start, v_end, seg_t, seg_v = best
    # Linear fit V vs t_days over the entire rest
    t_days = seg_t / 86400.0
    try:
        dv_per_day, _ = np.polyfit(t_days, seg_v, 1)
    except Exception:
        dv_per_day = (v_end - v_start) / (dur_h / 24.0)

    LFP_DVDSOC_HIGH_SOC = 0.5  # V per unit SoC (order-of-magnitude LFP knee)
    dsoc_per_day = -float(dv_per_day) / LFP_DVDSOC_HIGH_SOC

    return {
        "self_disch_dV_per_day": float(dv_per_day),
        "self_disch_dsoc_per_day": dsoc_per_day,
        "self_disch_rest_duration_h": float(dur_h),
    }


def _r0_from_pulses(csv: Path,
                    only_discharge: bool = True) -> tuple[float, list[float]]:
    """Estimate median R0 (mΩ) across current-step edges.

    Returns (median, all_values).
    """
    df = pd.read_csv(csv, usecols=["absolute_time", "volt_v", "current_a"])
    if df.empty:
        return np.nan, []
    t = _time_seconds(df)
    v = df["volt_v"].to_numpy(dtype=float)
    i = df["current_a"].to_numpy(dtype=float)
    r_vals: list[float] = []
    # find current transitions (rest -> load)
    active = np.abs(i) > 1.0
    starts = np.where(np.diff(active.astype(int)) == 1)[0] + 1
    for s in starts:
        # baseline just before edge, dV over first ~10-30 ms after edge
        if s < 3 or s + 3 >= len(v):
            continue
        v_pre = float(np.mean(v[max(0, s - 3):s]))
        i_pre = float(np.mean(i[max(0, s - 3):s]))
        # first sample >= 10 ms after edge
        t_target = t[s] + 0.03
        j = s
        while j < len(t) and t[j] < t_target:
            j += 1
        if j >= len(v):
            continue
        v_post = float(v[j])
        i_post = float(i[j])
        if only_discharge and i_post > -0.5:
            continue
        di = i_post - i_pre
        if abs(di) < 5.0:  # need a real load step
            continue
        r_ohm = abs((v_post - v_pre) / di)
        if 1e-5 < r_ohm < 0.1:
            r_vals.append(r_ohm * 1000.0)  # mΩ
    if not r_vals:
        return np.nan, []
    return float(np.median(r_vals)), r_vals


def extract_dcir_scalars(csv: Path) -> dict:
    r0_med, r0_all = _r0_from_pulses(csv)
    return {
        "DCIR_R0_mOhm": r0_med,
        "DCIR_R0_n": len(r0_all),
    }


def extract_hppc_scalars(csv: Path) -> dict:
    """Compute R0 median and R1 (tau) median for HPPC discharge pulses."""
    df = pd.read_csv(csv, usecols=["absolute_time", "volt_v", "current_a"])
    if df.empty:
        return {}
    t = _time_seconds(df)
    v = df["volt_v"].to_numpy(dtype=float)
    i = df["current_a"].to_numpy(dtype=float)
    active = np.abs(i) > 1.0
    starts = np.where(np.diff(active.astype(int)) == 1)[0] + 1
    ends = np.where(np.diff(active.astype(int)) == -1)[0] + 1
    r0_vals: list[float] = []
    r1_vals: list[float] = []
    for s in starts:
        # matching end
        e_idx = ends[ends > s]
        if len(e_idx) == 0:
            continue
        e = int(e_idx[0])
        if e - s < 5:
            continue
        # ---- R0 ----
        v_pre = float(np.mean(v[max(0, s - 3):s]))
        i_pre = float(np.mean(i[max(0, s - 3):s]))
        t_target = t[s] + 0.03
        j = s
        while j < len(t) and t[j] < t_target:
            j += 1
        if j >= len(v):
            continue
        i_post = float(i[j])
        if i_post > -0.5:  # discharge only
            continue
        v_post = float(v[j])
        di = i_post - i_pre
        if abs(di) < 5.0:
            continue
        r0 = abs((v_post - v_pre) / di) * 1000.0
        if 0.05 < r0 < 20.0:
            r0_vals.append(r0)
        # ---- R1 from recovery segment ----
        # relaxation window: e+1 .. e+ some samples covering ~30 s
        rec_end = min(len(t) - 1, e + 200)
        # advance rec_end until 30 s after edge
        while rec_end < len(t) - 1 and t[rec_end] - t[e] < 30.0:
            rec_end += 1
        seg_t = t[e:rec_end + 1] - t[e]
        seg_v = v[e:rec_end + 1]
        ok = np.isfinite(seg_t) & np.isfinite(seg_v) & (seg_t > 0.01)
        if ok.sum() < 15:
            continue
        seg_t = seg_t[ok]
        seg_v = seg_v[ok]
        v_inf = float(np.percentile(seg_v[-max(3, len(seg_v) // 5):], 90))
        dv = v_inf - seg_v
        pos = dv > 1e-4
        if pos.sum() < 8:
            continue
        try:
            log_dv = np.log(dv[pos])
            slope, intercept = np.polyfit(seg_t[pos], log_dv, 1)
            if slope >= 0:
                continue
            tau = -1.0 / slope
            deltaV = float(np.exp(intercept))  # magnitude at t=0
            # R1 = deltaV / |I_pulse|
            i_pulse = float(np.mean(i[s:e + 1]))
            r1 = abs(deltaV / i_pulse) * 1000.0
            if 0.05 < r1 < 20.0 and 0.5 < tau < 300.0:
                r1_vals.append(r1)
        except Exception:
            continue
    return {
        "HPPC_R0_mOhm": float(np.median(r0_vals)) if r0_vals else np.nan,
        "HPPC_R0_n": len(r0_vals),
        "HPPC_R1_mOhm": float(np.median(r1_vals)) if r1_vals else np.nan,
        "HPPC_R1_n": len(r1_vals),
    }


def _segments_by_step(step: pd.Series) -> list[tuple[int, int, str]]:
    """Return contiguous [(start, end, step_name)] runs of a step column."""
    if len(step) == 0:
        return []
    labels = step.fillna("").to_numpy()
    change = np.where(labels[1:] != labels[:-1])[0] + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change - 1, [len(labels) - 1]))
    return [(int(s), int(e), str(labels[s])) for s, e in zip(starts, ends)]


def extract_rpt_scalars(csv: Path) -> dict:
    df = pd.read_csv(csv, usecols=["step_name", "absolute_time",
                                    "volt_v", "current_a", "capacity_ah"])
    if df.empty:
        return {}
    t = _time_seconds(df)
    i = df["current_a"].to_numpy(dtype=float)
    step = df["step_name"].astype(str)
    dt = np.diff(t, prepend=t[0])
    dt[dt < 0] = 0.0
    dt[dt > 60.0] = 0.0
    segs = _segments_by_step(step)
    dis_caps: list[float] = []
    chg_caps: list[float] = []
    for s, e, name in segs:
        low = name.lower()
        q = float(np.nansum(np.abs(i[s:e + 1]) * dt[s:e + 1]) / 3600.0)
        if q < 1.0:
            continue
        if "dchg" in low or "discharge" in low:
            dis_caps.append(q)
        elif "chg" in low:
            chg_caps.append(q)
    # "capacity" = largest single discharge event (real cell capacity)
    q_dis_max = max(dis_caps) if dis_caps else np.nan
    q_chg_max = max(chg_caps) if chg_caps else np.nan
    # Coulombic efficiency: pair discharges with the immediately preceding
    # charge (same segment iteration) and use the *matched-max* pair. This
    # avoids the "5 discharges vs 4 charges" imbalance in RPT protocols.
    pairs: list[tuple[float, float]] = []
    last_chg: float | None = None
    for s, e, name in segs:
        low = name.lower()
        q = float(np.nansum(np.abs(i[s:e + 1]) * dt[s:e + 1]) / 3600.0)
        if q < 1.0:
            continue
        if "chg" in low and "dchg" not in low and "discharge" not in low:
            last_chg = q
        elif ("dchg" in low or "discharge" in low) and last_chg is not None:
            pairs.append((last_chg, q))
            last_chg = None
    if pairs:
        best_chg, best_dis = max(pairs, key=lambda p: p[1])
        ce_pct = 100.0 * best_dis / best_chg if best_chg > 0 else np.nan
    else:
        ce_pct = np.nan
    return {
        "capacity_Ah": q_dis_max,
        "capacity_chg_Ah": q_chg_max,
        "coulombic_efficiency_pct": ce_pct,
        "duration_h": float((t[-1] - t[0]) / 3600.0),
        "rpt_n_discharge_segments": len(dis_caps),
    }


# --------------------------------------------------------------------------- #
# Longterm SoH loaders
# --------------------------------------------------------------------------- #
# Nameplate capacity of the CALB batch-2 cells in this dataset (from
# ``max_cap`` column of every Longterm CSV). Used as the SoH denominator —
# NOT the first-cycle discharge capacity, because these cells arrived
# already >1000 cycles into life from a prior campaign not present in the
# Athena export.
CALB_NAMEPLATE_AH: float = 72.0


# CC-only capacity extraction for CALB batch=2 (batch=2 protocol switched
# from CC to CC-CV, so raw discharge_cap in the Longterm cycle table is
# inflated by the CV tail). df_calb_cc_cap.csv gives the CC portion only,
# yielding a protocol-invariant SoH continuous across the batch-1/2 seam.
CALB_CC_CAP_CSV = PROJECT_ROOT / "df_calb_cc_cap.csv"


def _load_calb_batch2_cc_cap() -> pd.DataFrame:
    """Return {cell_no (str, zero-padded), cycle_no (int), cc_capacity_ah}
    for CALB batch=2 only."""
    if not CALB_CC_CAP_CSV.exists():
        warnings.warn(f"CALB CC cap file not found at {CALB_CC_CAP_CSV}; "
                      "falling back to Longterm discharge_cap_ah for batch=2 "
                      "(WILL over-report SoH by the CV tail).")
        return pd.DataFrame(columns=["cell_no", "cycle_no", "cc_capacity_ah"])
    cc = pd.read_csv(CALB_CC_CAP_CSV,
                      usecols=["cell_no", "batch", "cycle_no", "cc_capacity_ah"])
    cc["cell_no"] = cc["cell_no"].astype(str).str.zfill(4)
    cc["batch"] = cc["batch"].astype(int)
    cc = cc[cc["batch"] == 2][["cell_no", "cycle_no", "cc_capacity_ah"]]
    cc = (cc.sort_values(["cell_no", "cycle_no"])
            .drop_duplicates(["cell_no", "cycle_no"], keep="last"))
    return cc


def calb_longterm_soh() -> dict[str, pd.DataFrame]:
    """Return {cell_id: DataFrame(cycle, soh, batch)} for CALB Longterm.

    Protocol note: CALB batch=1 used a CC-only charging protocol, so
    ``discharge_cap_ah`` in the Longterm cycle table already IS the CC-only
    capacity. Batch=2 switched to CC-CV, so its raw ``discharge_cap_ah``
    over-represents true capacity by the CV-tail contribution. The
    ``df_calb_cc_cap.csv`` file provides the extracted CC-only capacity
    per cycle for batch=2 — using that for batch=2 gives a protocol-
    invariant SoH continuous across the batch-1/2 seam.

    Also: CALB batch=2 cells have been cycled >1000 cycles in a prior
    campaign not present in this Athena export, so batch=1 cy 1 SoH is
    already below 1.0. Normalise by nameplate (72 Ah), not by first-cycle
    discharge capacity.
    """
    b2_cc = _load_calb_batch2_cc_cap()
    b2_cc_by_cell: dict[str, pd.DataFrame] = {
        cid: g[["cycle_no", "cc_capacity_ah"]].copy()
        for cid, g in b2_cc.groupby("cell_no")
    }

    out: dict[str, pd.DataFrame] = {}
    for cid, csv in list_cells("Longterm", "CALB"):
        try:
            df = pd.read_csv(csv, usecols=["cycle_no", "discharge_cap_ah",
                                             "batch", "max_cap"])
        except Exception as ex:
            warnings.warn(f"CALB Longterm {csv.name}: {ex}")
            continue
        df = df.dropna(subset=["discharge_cap_ah"]).copy()
        df["discharge_cap_ah"] = df["discharge_cap_ah"].astype(float)
        df["batch"] = df["batch"].astype(int)
        df = (df.sort_values(["batch", "cycle_no"])
                .drop_duplicates(subset=["batch", "cycle_no"], keep="last")
                .reset_index(drop=True))

        # Choose per-batch capacity source:
        #   batch=1 -> Longterm discharge_cap_ah (native CC)
        #   batch=2 -> CC file cc_capacity_ah (CC-only extracted from CC-CV)
        cap = df["discharge_cap_ah"].astype(float).copy()
        cc_lookup = b2_cc_by_cell.get(cid)
        if cc_lookup is not None and not cc_lookup.empty:
            merged = df.merge(cc_lookup, on="cycle_no", how="left")
            mask_b2 = df["batch"] == 2
            cap.loc[mask_b2] = merged.loc[mask_b2, "cc_capacity_ah"].astype(float)
            # If CC file is missing some batch=2 cycles, fall back to dchg
            missing = cap.isna()
            if missing.any():
                cap.loc[missing] = df.loc[missing, "discharge_cap_ah"].astype(float)
        df["cap_ah"] = cap
        df["global_cycle"] = np.arange(1, len(df) + 1)
        try:
            nameplate = float(pd.Series(df["max_cap"]).dropna().mode().iloc[0])
        except Exception:
            nameplate = CALB_NAMEPLATE_AH
        if not np.isfinite(nameplate) or nameplate <= 0:
            nameplate = CALB_NAMEPLATE_AH
        df["soh"] = df["cap_ah"] / nameplate
        out[cid] = df[["global_cycle", "soh", "batch"]].rename(
            columns={"global_cycle": "cycle"})
    return out


def parquet_soh(make: str) -> dict[str, pd.DataFrame]:
    fname = {"REPT": "rept.parquet", "EVE": "eve.parquet"}[make]
    p = CANONICAL_SOH / fname
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    out: dict[str, pd.DataFrame] = {}
    for cid, sub in df.groupby("cell_id"):
        sub2 = sub.sort_values("global_cycle").reset_index(drop=True)
        cols = ["global_cycle", "soh"] + (["batch"] if "batch" in sub2.columns
                                            else [])
        renamed = sub2[cols].rename(columns={"global_cycle": "cycle"})
        out[str(cid)] = renamed
    return out


def soh_scalars(traces: dict[str, pd.DataFrame]) -> dict[str, dict]:
    out = {}
    for cid, df in traces.items():
        if df.empty:
            continue
        soh = df["soh"].to_numpy(dtype=float)
        cyc = df["cycle"].to_numpy(dtype=float)
        soh = soh[np.isfinite(soh)]
        if soh.size < 3:
            continue
        s_first = float(soh[0])
        s_last = float(soh[-1])
        mono = float(np.mean(np.diff(soh) <= 1e-3))
        out[cid] = {
            "n_cycles": int(np.nanmax(cyc)),
            "soh_first": s_first,
            "soh_last": s_last,
            "fade_pct": 100.0 * (s_first - s_last),
            "monotone_frac": mono,
        }
    return out


# --------------------------------------------------------------------------- #
# Top-level driver
# --------------------------------------------------------------------------- #
@dataclass
class ExtractionResult:
    scalars: pd.DataFrame
    soh_traces: dict[str, dict[str, pd.DataFrame]]  # {make: {cell: df}}
    ocv_curves: dict[str, dict[str, pd.DataFrame]]  # {make: {cell: df}}


def _safe(fn, csv: Path, label: str) -> dict:
    try:
        return fn(csv)
    except Exception as ex:  # noqa: BLE001
        warnings.warn(f"{label} extract failed for {csv.name}: {ex}")
        return {}


def run_extraction(makes: Iterable[str] = MAKES) -> ExtractionResult:
    rows: list[dict] = []
    ocv_curves: dict[str, dict[str, pd.DataFrame]] = {m: {} for m in makes}
    for make in makes:
        # per-cell scalar aggregation across all tests
        cells: dict[str, dict] = {}
        # --- OCV_SOC ---
        for cid, csv in list_cells("OCVSOC", make):
            cells.setdefault(cid, {"cell_id": cid, "make": make})
            cells[cid].update(_safe(extract_ocv_scalars, csv, "OCV"))
            curve = _safe(lambda c=csv: extract_ocv_curve(c), csv, "OCV curve")
            if isinstance(curve, pd.DataFrame):
                ocv_curves[make][cid] = curve
        # --- GITT ---
        for cid, csv in list_cells("GITT", make):
            cells.setdefault(cid, {"cell_id": cid, "make": make})
            cells[cid].update(_safe(extract_gitt_scalars, csv, "GITT"))
        # --- DCIR ---
        for cid, csv in list_cells("DCIR", make):
            cells.setdefault(cid, {"cell_id": cid, "make": make})
            cells[cid].update(_safe(extract_dcir_scalars, csv, "DCIR"))
        # --- HPPC ---
        for cid, csv in list_cells("HPPC", make):
            cells.setdefault(cid, {"cell_id": cid, "make": make})
            cells[cid].update(_safe(extract_hppc_scalars, csv, "HPPC"))
        # --- RPT ---
        for cid, csv in list_cells("RPT", make):
            cells.setdefault(cid, {"cell_id": cid, "make": make})
            cells[cid].update(_safe(extract_rpt_scalars, csv, "RPT"))
        # --- SelfDischarge ---
        for cid, csv in list_cells("SelfDischarge", make):
            cells.setdefault(cid, {"cell_id": cid, "make": make})
            cells[cid].update(_safe(extract_selfdisch_scalars, csv, "SelfDisch"))
        rows.extend(cells.values())

    scalars = pd.DataFrame(rows).sort_values(["make", "cell_id"]).reset_index(drop=True)

    # --- Longterm SoH per make ---
    soh_traces: dict[str, dict[str, pd.DataFrame]] = {}
    soh_scalar_rows: list[dict] = []
    for make in makes:
        if make == "CALB":
            traces = calb_longterm_soh()
        else:
            traces = parquet_soh(make)
        soh_traces[make] = traces
        for cid, sc in soh_scalars(traces).items():
            soh_scalar_rows.append({"cell_id": cid, "make": make, **sc})

    if soh_scalar_rows:
        soh_df = pd.DataFrame(soh_scalar_rows)
        scalars = scalars.merge(soh_df, on=["cell_id", "make"], how="outer")

    # ensure a stable column order
    lead = ["cell_id", "make"]
    tail = [c for c in scalars.columns if c not in lead]
    scalars = scalars[lead + tail]
    return ExtractionResult(scalars=scalars,
                             soh_traces=soh_traces,
                             ocv_curves=ocv_curves)


# --------------------------------------------------------------------------- #
# Flagging
# --------------------------------------------------------------------------- #
def flag_cells(scalars: pd.DataFrame) -> pd.DataFrame:
    """Return a long DataFrame of (cell_id, make, metric, value, reason)."""
    rows: list[dict] = []
    for metric, (lo, hi) in BOUNDS.items():
        if metric not in scalars.columns:
            continue
        for _, r in scalars.iterrows():
            v = r[metric]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            if v < lo:
                rows.append({"cell_id": r["cell_id"], "make": r["make"],
                             "metric": metric, "value": v,
                             "reason": f"below {lo}"})
            elif v > hi:
                rows.append({"cell_id": r["cell_id"], "make": r["make"],
                             "metric": metric, "value": v,
                             "reason": f"above {hi}"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    res = run_extraction()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "characterization_scalars.csv"
    res.scalars.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}: {len(res.scalars)} rows")
    flags = flag_cells(res.scalars)
    flags_csv = OUT_DIR / "flagged_cells.csv"
    flags.to_csv(flags_csv, index=False)
    print(f"Wrote {flags_csv}: {len(flags)} flag rows")
