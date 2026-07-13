"""
Voltaris/Data_Exploration/phase3_v7_dataset.py
==============================================

v7 dataset builder: turns each 2500-cycle PyBaMM simulation into multiple
(context, target) training pairs so the encoder-decoder OperatorV7 sees
partial-trajectory forecasting examples starting from a variety of SoH
values (not just SoH=1.0).

Design decisions (all documented so v7 is reproducible):

1. x_health is REDUCED from 5 dims to 3 dims.
   IC-peak fields (positions 3, 4) were hardcoded to (0, 1) in v6 and
   therefore carried zero signal. Rather than fake them again, v7 drops
   them entirely:
       v7 x_health = [temperature_C, c_rate, dcir_mOhm]
   temperature_C is still 25.0 across the cohort (isothermal test bed),
   so effectively only 2 dims carry variation — but they carry REAL
   variation (per-anchor c_rate from protocol_id, per-anchor DCIR from
   Phase-1 BOL yaml).

2. The encoder receives DELTA-FROM-CONTEXT-START, not raw SoH:
       context_delta[i] = SoH[s+i] - SoH[s]     for i in 0..K-1
   Rationale: raw SoH at a second-life cell (~0.4 nameplate) is OOD for
   an operator trained on sims starting at 1.0. Passing the delta form
   makes the context encode fade SHAPE, independent of absolute starting
   SoH. The absolute SoH[s] is passed separately as the ODE integration
   initial condition.

3. K = 50 context cycles per training pair (user pick). Forecast horizon
   is a fixed 400 cycles per pair (kept uniform so tensor stacking is
   straightforward). Sims with < 50+400 = 450 usable cycles are skipped
   for that context_start.

4. Multiple context starts per sim s ∈ {0, 100, 300, 500, 800, 1200,
   1500, 1800} give the encoder ~8 training pairs per sim from varied
   starting-SoH points. Total pairs ≈ 490 sims × 8 starts ≈ 3920,
   modulo skipped short sims — a ~8× data multiplier over v6.

Output parquet schema:
    anchor_id: str
    sample_id: str
    context_start: int          # s
    K:            int           # 50
    forecast_len: int           # 400
    x_health:     list[float]   # 3-dim
    theta_norm:   list[float]   # 6-dim (identical to v6 encoding)
    context_delta: list[float]  # K entries, SoH[s+i] - SoH[s]
    context_soh_start: float    # SoH[s] — passed to ODE as initial condition
    target_cycles: list[int]    # s+K, s+K+1, ..., s+K+forecast_len-1
    target_soh:   list[float]   # SoH values at those cycles
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CORPUS_DIR = _PROJECT_ROOT / "configs" / "phase3_corpus"
_BOL_PARAMS_DIR = _PROJECT_ROOT / "configs" / "bol_params"
_SWEEP_CONFIG = _PROJECT_ROOT / "configs" / "phase3_sweep.yaml"
_OUT_DEFAULT = _CORPUS_DIR / "_v7_dataset.parquet"

# Reuse the same v6 theta normalisation so v7 uses identical θ encoding.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase3_features import (      # noqa: E402
    extract_theta_norm,
    _load_anchors_and_sigma,
)

# ------------------------------------------------------------------------
# v7 constants
# ------------------------------------------------------------------------
V7_HEALTH_FEATURES: list[str] = ["temperature_C", "c_rate", "dcir_mOhm"]
K_CONTEXT = 50
FORECAST_LEN = 400
CONTEXT_STARTS = [0, 100, 300, 500, 800, 1200, 1500, 1800]

# SoH translation-augmentation offsets. For each training pair, we produce
# copies shifted by these offsets (added to both context and target SoH
# values, and to context_soh_start). This teaches the operator to be
# translation-invariant in absolute SoH — needed because the raw sims
# bottom out at SoH~0.65 while real second-life cells enter at SoH~0.45.
# Offsets are chosen to keep resulting SoH values in the (~0.35, ~1.05)
# window; augmented pairs whose shifted SoH escapes that window are dropped.
SOH_OFFSETS = [0.0, -0.1, -0.2, -0.3]
SOH_MIN_AFTER_OFFSET = 0.30
SOH_MAX_AFTER_OFFSET = 1.05

_PROTOCOL_RE = re.compile(r"^[A-Z]+_([\d.]+)C_([\d.]+)D_(\d+)_(\d+)$")

_BOL_DCIR_CACHE: dict[str, float] = {}


def _parse_c_rate(protocol_id: str) -> float:
    if not protocol_id or not isinstance(protocol_id, str):
        return float("nan")
    m = _PROTOCOL_RE.match(protocol_id)
    return float(m.group(1)) if m else float("nan")


def _bol_dcir_mohm(anchor_id: str) -> float:
    if anchor_id in _BOL_DCIR_CACHE:
        return _BOL_DCIR_CACHE[anchor_id]
    p = _BOL_PARAMS_DIR / f"{anchor_id}.yaml"
    if not p.exists():
        _BOL_DCIR_CACHE[anchor_id] = float("nan")
        return float("nan")
    try:
        doc = yaml.safe_load(p.read_text()) or {}
        r0 = float(doc.get("resistance", {}).get("R0_Ohm", "nan"))
        val = r0 * 1000.0 if np.isfinite(r0) else float("nan")
    except Exception:  # noqa: BLE001
        val = float("nan")
    _BOL_DCIR_CACHE[anchor_id] = val
    return val


def _extract_v7_x_health(trajectory_df: pd.DataFrame,
                          ambient_C: float = 25.0) -> np.ndarray:
    """3-dim x_health = [T, c_rate, DCIR]."""
    if trajectory_df.empty:
        return np.full(3, np.nan, dtype=np.float32)
    first = trajectory_df.sort_values("cycle_n").iloc[0]
    c_rate = _parse_c_rate(str(first.get("protocol_id", "")))
    anchor_id = str(first.get("anchor_id", ""))
    dcir = _bol_dcir_mohm(anchor_id) if anchor_id else float("nan")
    x = np.array([ambient_C, c_rate, dcir], dtype=np.float32)
    return np.where(np.isfinite(x), x, 0.0).astype(np.float32)


def build_v7_dataset(corpus_dir: Path = _CORPUS_DIR,
                      out_path: Path = _OUT_DEFAULT,
                      sweep_config: Path = _SWEEP_CONFIG,
                      K: int = K_CONTEXT,
                      forecast_len: int = FORECAST_LEN,
                      context_starts: list[int] = CONTEXT_STARTS,
                      ) -> Path:
    corpus_dir = Path(corpus_dir)
    traj_path = corpus_dir / "trajectories.parquet"
    if not traj_path.exists():
        raise FileNotFoundError(
            f"Corpus trajectories.parquet not found at {traj_path}. "
            f"Run phase3_corpus.py first (or concat per-anchor parquets)."
        )
    df = pd.read_parquet(traj_path)
    anchors, sigma = _load_anchors_and_sigma(sweep_config)

    rows: list[dict[str, Any]] = []
    skipped_short = 0
    for (aid, sid), g in df.groupby(["anchor_id", "sample_id"], sort=False):
        g = g.sort_values("cycle_n").reset_index(drop=True)
        soh = g["SOH"].to_numpy(dtype=np.float32)
        # Filter NaN / non-finite from the tail.
        finite = np.isfinite(soh)
        if not finite.all():
            soh = soh[: int(np.argmax(~finite))] if (~finite).any() else soh
        N = len(soh)
        if N < K + 1:
            skipped_short += 1
            continue

        anchor_theta = anchors.get(str(aid), {})
        x_health = _extract_v7_x_health(g)
        theta_norm = extract_theta_norm(g, anchor_theta=anchor_theta,
                                          perturbation_sigma=sigma)

        for s in context_starts:
            # Need s + K + 1 cycles minimum (context + at least one target
            # sample). We keep as many target cycles as available up to
            # forecast_len; short-target pairs are still useful signal.
            if s + K + 1 > N:
                continue
            f_actual = min(forecast_len, N - (s + K))
            if f_actual < 1:
                continue

            ctx = soh[s : s + K]
            ctx_start = float(soh[s])
            ctx_delta = (ctx - ctx_start).astype(np.float32)
            tgt_cycles = np.arange(s + K, s + K + f_actual, dtype=np.int32)
            tgt_soh_base = soh[s + K : s + K + f_actual].astype(np.float32)

            for offset in SOH_OFFSETS:
                # ctx_delta is offset-invariant. tgt_soh and ctx_start shift.
                ctx_start_off = ctx_start + offset
                tgt_soh_off = tgt_soh_base + offset
                # Skip augmented copies whose SoH values escape the training
                # window — those would confuse the operator with unphysical
                # extrapolations.
                soh_min = min(float(ctx_start_off + ctx_delta.min()),
                              float(tgt_soh_off.min()))
                soh_max = max(float(ctx_start_off + ctx_delta.max()),
                              float(tgt_soh_off.max()))
                if soh_min < SOH_MIN_AFTER_OFFSET:
                    continue
                if soh_max > SOH_MAX_AFTER_OFFSET:
                    continue

                rows.append({
                    "anchor_id":         str(aid),
                    "sample_id":         str(sid),
                    "context_start":     int(s),
                    "K":                 int(K),
                    "forecast_len":      int(f_actual),
                    "soh_offset":        float(offset),
                    "x_health":          x_health.tolist(),
                    "theta_norm":        theta_norm.tolist(),
                    "context_delta":     ctx_delta.tolist(),
                    "context_soh_start": ctx_start_off,
                    "target_cycles":     tgt_cycles.tolist(),
                    "target_soh":        tgt_soh_off.tolist(),
                })

    if not rows:
        raise ValueError("v7 dataset build produced zero pairs.")

    out = pd.DataFrame(rows)
    out.attrs["v7_health_features"] = V7_HEALTH_FEATURES
    out.attrs["K"] = K
    out.attrs["forecast_len_target"] = forecast_len
    out.attrs["context_starts"] = context_starts

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[phase3_v7_dataset] wrote {out_path}  "
          f"pairs={len(out):,}  from {out['sample_id'].nunique()} sims "
          f"({skipped_short} sims skipped for length < {K + 1})",
          flush=True)

    per_anchor = out.groupby("anchor_id").size()
    print("[phase3_v7_dataset] pairs per anchor:", flush=True)
    for aid, n in per_anchor.items():
        print(f"  {aid}: {n}", flush=True)

    # x_health variation across anchors
    print("[phase3_v7_dataset] x_health per anchor (first row):", flush=True)
    for aid, grp in out.groupby("anchor_id"):
        x = grp["x_health"].iloc[0]
        print(f"  {aid}: T={x[0]}, c_rate={x[1]:.3f}, "
              f"dcir={x[2]:.3f} mOhm", flush=True)

    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-dir", type=Path, default=_CORPUS_DIR)
    p.add_argument("--out-path", type=Path, default=_OUT_DEFAULT)
    p.add_argument("--sweep-config", type=Path, default=_SWEEP_CONFIG)
    p.add_argument("--K", type=int, default=K_CONTEXT)
    p.add_argument("--forecast-len", type=int, default=FORECAST_LEN)
    args = p.parse_args()
    build_v7_dataset(args.corpus_dir, args.out_path, args.sweep_config,
                      K=args.K, forecast_len=args.forecast_len)
