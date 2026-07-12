"""
src/inference/health_features.py
--------------------------------
Extract the 5-feature health vector that the PINN consumes from whatever
characterisation tests happen to be available for a given cell.

    HEALTH_FEATURES = [
        "temperature_C",
        "c_rate",
        "dcir_mOhm",
        "ic_peak1_shift_V",
        "ic_peak2_area_norm",
    ]

`extract_for_cell` uses the project's data loader and the Phase-1
identification report (`configs/identified_params.yaml`) as the fresh-cell
baseline. Missing tests fall back to defensible defaults documented inline.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data_loader import load_test, load_rpt_capacity_fade  # noqa: E402

HEALTH_FEATURES = [
    "temperature_C", "c_rate", "dcir_mOhm",
    "ic_peak1_shift_V", "ic_peak2_area_norm",
]

PROJECT_ROOT = Path("/home/hj/Desktop/PINNs")
IDENTIFIED_PARAMS = PROJECT_ROOT / "configs" / "identified_params.yaml"


@dataclass
class HealthFeatures:
    cell_id: str
    temperature_C: float
    c_rate: float
    dcir_mOhm: float
    ic_peak1_shift_V: float
    ic_peak2_area_norm: float
    sources: dict[str, str]   # which test each feature came from

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in HEALTH_FEATURES],
                        dtype=np.float32)


# ── helpers ───────────────────────────────────────────────────────────────

def _baseline_from_identified_params(path: Path = IDENTIFIED_PARAMS) -> dict:
    """
    Load fresh-cell reference values from the Phase-1 identification report.
    Returns a flat dict with the items we actually use here.
    """
    if not path.exists():
        return {}
    cfg = yaml.safe_load(path.read_text()) or {}
    out = {}
    if "resistance" in cfg and "R0_Ohm" in cfg["resistance"]:
        out["R0_baseline_mOhm"] = float(cfg["resistance"]["R0_Ohm"]) * 1000.0
    return out


def _r0_from_hppc_discharge(cell_id: str) -> Optional[float]:
    """
    Cell-level R₀ in mΩ from the first detectable HPPC discharge pulse,
    computed exactly as `dcir_hppc.py` does (ΔV / ΔI at the pulse onset).
    Returns None if no pulse is detectable.
    """
    from src.param_id.dcir_hppc import extract_pulses

    pulses = extract_pulses(cell_id, test="HPPC", Q_nominal_Ah=105.0)
    discharges = [p for p in pulses if p.direction == "discharge"]
    if not discharges:
        return None
    # Pick the pulse closest to SOC=0.5; this dataset only probes ≥0.97 so
    # in practice it picks the median of what's available.
    target_soc = 0.5
    nearest = min(discharges, key=lambda p: abs(p.SOC_est - target_soc))
    return float(nearest.R0_Ohm * 1000.0)


def _ic_curve_from_rpt(cell_id: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Compute a (V, dQ/dV) curve from the most recent RPT discharge for this
    cell. Returns None if the RPT data is missing or unusable.
    """
    try:
        df = load_test("RPT", cell_id=cell_id).sort_values("time").reset_index(drop=True)
    except Exception:
        return None
    # Use the last full CC_DChg in the file (most recent measurement)
    disc = df[df["step_name"] == "CC_DChg"]
    if disc.empty:
        return None
    last_cycle = disc["cycle"].max()
    disc_last = disc[disc["cycle"] == last_cycle].sort_values("time")
    if len(disc_last) < 50:
        return None
    V = disc_last["voltage"].to_numpy(dtype=float)
    Q = disc_last["capacity"].abs().to_numpy(dtype=float)
    # Sort by V and drop ties — LFP plateau yields many same-V samples
    order = np.argsort(V)
    V = V[order]; Q = Q[order]
    keep = np.concatenate(([True], np.diff(V) > 1e-6))
    V, Q = V[keep], Q[keep]
    if len(V) < 25:
        return None
    return V, Q


def _ic_peaks(V: np.ndarray, Q: np.ndarray) -> list[dict]:
    """
    Compute dQ/dV from (V, Q) and return the two largest peaks sorted by V.
    """
    from scipy.signal import find_peaks, savgol_filter

    v_grid = np.linspace(2.8, 3.5, 400)
    q_grid = np.interp(v_grid, V, Q)
    q_smooth = savgol_filter(q_grid, window_length=21, polyorder=3)
    dqdv = np.abs(np.gradient(q_smooth, v_grid))
    if not np.isfinite(dqdv).any():
        return []
    peaks, props = find_peaks(dqdv, prominence=np.nanmax(dqdv) * 0.05, distance=10)
    if len(peaks) == 0:
        return []
    order = np.argsort(props["prominences"])[::-1][:2]
    out = []
    for i in order:
        idx = int(peaks[i])
        lo = max(0, idx - 15)
        hi = min(len(v_grid), idx + 15)
        area = float(np.trapezoid(dqdv[lo:hi], v_grid[lo:hi]))
        out.append({"V": float(v_grid[idx]), "area": area})
    return sorted(out, key=lambda d: d["V"])


# ── main API ──────────────────────────────────────────────────────────────

def extract_for_cell(
    cell_id: str,
    temperature_C: float = 25.0,
    c_rate: float = 0.5,
    baseline_ic: Optional[dict] = None,
    available_tests: Optional[list[str]] = None,
) -> HealthFeatures:
    """
    Build the health-feature vector for one cell from whatever tests it has.

    Args:
        cell_id: standardised cell ID, e.g. "0005".
        temperature_C, c_rate: operating-condition tags (recorded in the
            telemetry / contract; not extracted from the data here).
        baseline_ic: optional dict {peak1_V_fresh, peak2_area_fresh}. If
            absent, the cell's own first IC measurement is treated as the
            baseline (so shift=0, area_norm=1 by construction). When you
            have an authoritative fresh-cell reference, pass it explicitly.
        available_tests: optional subset of canonical test names to consider.
            If None, all of {HPPC, DCIR, RPT} are tried.
    """
    available = set(available_tests) if available_tests is not None else {"HPPC", "DCIR", "RPT"}
    sources: dict[str, str] = {
        "temperature_C": "supplied", "c_rate": "supplied",
    }
    baseline = _baseline_from_identified_params()

    # DCIR — prefer HPPC R0 at the closest SOC available, else fall back
    dcir_mOhm = float("nan")
    if "HPPC" in available:
        r0 = _r0_from_hppc_discharge(cell_id)
        if r0 is not None:
            dcir_mOhm = r0
            sources["dcir_mOhm"] = "HPPC R0"
    if not np.isfinite(dcir_mOhm) and "DCIR" in available:
        # The DCIR test in this dataset doesn't expose clean pulses preceded
        # by rests; fall back to the cohort baseline.
        dcir_mOhm = baseline.get("R0_baseline_mOhm", 2.0)
        sources["dcir_mOhm"] = "baseline (cohort R0)"
    elif not np.isfinite(dcir_mOhm):
        dcir_mOhm = baseline.get("R0_baseline_mOhm", 2.0)
        sources["dcir_mOhm"] = "baseline (cohort R0)"

    # IC peaks — derive shift & area-norm against `baseline_ic` if supplied,
    # otherwise treat the current measurement as the baseline.
    ic_peak1_shift_V = 0.0
    ic_peak2_area_norm = 1.0
    sources["ic_peak1_shift_V"] = "no RPT"
    sources["ic_peak2_area_norm"] = "no RPT"
    if "RPT" in available:
        ic = _ic_curve_from_rpt(cell_id)
        if ic is not None:
            V, Q = ic
            peaks = _ic_peaks(V, Q)
            if len(peaks) >= 2:
                peak1_V = peaks[0]["V"]
                peak2_area = peaks[1]["area"]
                if baseline_ic:
                    ic_peak1_shift_V = peak1_V - float(baseline_ic["peak1_V_fresh"])
                    if baseline_ic.get("peak2_area_fresh", 0.0) > 1e-12:
                        ic_peak2_area_norm = peak2_area / float(baseline_ic["peak2_area_fresh"])
                    sources["ic_peak1_shift_V"] = "RPT vs supplied baseline"
                    sources["ic_peak2_area_norm"] = "RPT vs supplied baseline"
                else:
                    sources["ic_peak1_shift_V"] = "RPT (self-referenced → 0)"
                    sources["ic_peak2_area_norm"] = "RPT (self-referenced → 1)"

    return HealthFeatures(
        cell_id=cell_id,
        temperature_C=float(temperature_C),
        c_rate=float(c_rate),
        dcir_mOhm=float(dcir_mOhm),
        ic_peak1_shift_V=float(ic_peak1_shift_V),
        ic_peak2_area_norm=float(ic_peak2_area_norm),
        sources=sources,
    )


if __name__ == "__main__":
    for cid in ["0005", "0006", "0007", "0008"]:
        try:
            h = extract_for_cell(cid)
            print(f"cell {cid}: {h.as_array().tolist()}")
            for k, v in h.sources.items():
                print(f"    {k:<22} ← {v}")
        except Exception as e:
            print(f"cell {cid}: {type(e).__name__}: {e}")
