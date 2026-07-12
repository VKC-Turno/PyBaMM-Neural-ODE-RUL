"""
Voltaris/Data_Exploration/phase3_validate.py
============================================

Held-out validation script for the Phase-3 theta-aware operator.

Given a trained checkpoint (``outputs/models/phase3_operator.pt`` or whatever
name ``phase3_train_val.train_operator`` produced), this script runs SoH
prediction + a battery of R1 gates on the three held-out cells listed in
``configs/phase3_heldout.yaml``:

    CALB_0029  (fast fader, crosses SoH 0.80 within coverage)
    EVE_0003   (mild fader, cycle-summary must be regenerated from raw V-t)
    REPT_0031  (pre-knee coverage only; excluded from EoL-bucket metric)

Held-out cells have NEITHER a ``deg_params/{make}_{cell}.yaml`` nor a
``bol_params/{make}_{cell}.yaml`` on disk (per 2026-07-12 corpus close).
Per the task spec we degrade gracefully: theta_norm falls back to the
anchor centre (zeros) whenever the fitted YAML is missing, and x_health
falls back to the cohort default (25 degC, C/2, defaults).

Public API
----------
- ``load_operator_from_checkpoint(path) -> RULPredictor``
- ``predict_cell_soh(model, cell_id, make) -> (n_cycles_arr, pred_soh)``
- ``heldout_metrics(cell_id, make, pred_soh, obs_soh, obs_cycles) -> dict``
- ``fisher_cosine_gate(model, cell_id, make) -> dict``
- ``regime_swap_replay(model, cell_a, cell_b, make_a, make_b) -> dict``

CLI
---
    .venv/bin/python -m Voltaris.Data_Exploration.phase3_validate \\
        --checkpoint outputs/models/phase3_operator.pt \\
        [--out outputs/results/phase3_heldout_validation.md]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml

# ---------------------------------------------------------------------------
# Project imports — reuse the Phase-3 machinery for consistency.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path("/home/hj/Desktop/PINNs")
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pinn.model import RULPredictor  # noqa: E402
from Voltaris.Data_Exploration.phase3_train_val import (  # noqa: E402
    BRANCH_DIM,
    K_SEI_IDX,
    LAM_NEG_IDX,
    N_HEALTH_FEATURES,
    N_THETA,
    THETA_KEYS,
    Phase3Sample,
    default_theta_stats,
    fisher_column_cosine as _fisher_column_cosine,
    load_operator as _load_operator,
    normalise_theta,
)


HELDOUT_YAML = _PROJECT_ROOT / "configs" / "phase3_heldout.yaml"
DEG_PARAMS_DIR = _PROJECT_ROOT / "configs" / "deg_params"
BOL_PARAMS_DIR = _PROJECT_ROOT / "configs" / "bol_params"
DCIR_SUMMARY_PARQUET = _PROJECT_ROOT / "data" / "processed" / "dcir_hppc_summary.parquet"
LONGTERM_DIR = _PROJECT_ROOT / "Data" / "Longterm"

# Nameplate capacities (Ah) per LFP supplier. Used to compute actual SoH
# (discharge_cap_Ah / nominal_Ah) rather than test-window-relative SoH.
NOMINAL_CAPACITY_AH: dict[str, float] = {
    "CALB": 72.0,
    "EVE":  105.0,
    "REPT": 150.0,
}

# Gate thresholds (mirror configs/phase3_operator.yaml + phase3_heldout.yaml).
SOH_RMSE_PP_MAX = 3.0
FISHER_COSINE_MAX = 0.3

# YAML → phase3 theta key mapping. The fitted params YAML uses
# `k_LAM_negative`; phase3 code uses `LAM_neg_rate_s`; LAM_pos is absent
# from the Phase-2 DE fits and defaults to anchor-centre.
_YAML_TO_PHASE3 = {
    "k_SEI": "k_SEI",
    "V_SEI": "V_SEI",
    "D_SEI_solvent": "D_SEI_solvent",
    "k_plating": "k_plating",
    "k_LAM_negative": "LAM_neg_rate_s",
}


# ---------------------------------------------------------------------------
# 1. Load-checkpoint helper
# ---------------------------------------------------------------------------
def load_operator_from_checkpoint(path: str | Path) -> RULPredictor:
    """Restore a Phase-3 ``RULPredictor`` (branch_dim=11) from ``path``.

    Delegates to ``phase3_train_val.load_operator`` which already validates
    the branch dim, replays the state dict, and sets the model to eval mode.
    """
    return _load_operator(path)


# ---------------------------------------------------------------------------
# Cell-specific input builders
# ---------------------------------------------------------------------------
def _load_theta_norm(cell_id: str, make: str) -> tuple[np.ndarray, bool]:
    """Return the 6-dim ``theta_norm`` conditioning vector for a held-out cell.

    Reads ``configs/deg_params/{make}_{cell}.yaml`` when present; otherwise
    returns zeros (i.e. the anchor centre in standardised log-space) and
    signals the fallback via the second return value.
    """
    yaml_path = DEG_PARAMS_DIR / f"{make}_{cell_id}.yaml"
    if not yaml_path.exists():
        return np.zeros(N_THETA, dtype=np.float32), False

    doc = yaml.safe_load(yaml_path.read_text()) or {}
    fitted = doc.get("fitted_params", {}) or {}

    theta_phys: dict[str, float] = {}
    for yaml_key, phase3_key in _YAML_TO_PHASE3.items():
        rec = fitted.get(yaml_key)
        if isinstance(rec, dict) and "value" in rec:
            theta_phys[phase3_key] = float(rec["value"])
    # Fill any missing phase3 key (e.g. LAM_pos_rate_s) with the anchor
    # centre so ``normalise_theta`` produces a defined vector.
    stats = default_theta_stats()
    for k in THETA_KEYS:
        if k in theta_phys:
            continue
        s = stats[k]
        theta_phys[k] = (10.0 ** s["mean"]) if s["space"] == "log10" else float(s["mean"])
    return normalise_theta(theta_phys, stats).astype(np.float32), True


def _load_x_health(cell_id: str, make: str,
                   ambient_C: float = 25.0,
                   default_c_rate: float = 0.5) -> np.ndarray:
    """Return the 5-dim ``x_health`` snapshot for a held-out cell.

    Fields mirror ``src/pinn/dataset.HEALTH_FEATURES`` exactly:
        [temperature_C, c_rate, dcir_mOhm, ic_peak1_shift_V, ic_peak2_area_norm]

    We pull ``dcir_mOhm`` from the DCIR-HPPC summary parquet if the cell is
    in it; otherwise the cohort default (0.0 after NaN-replacement) is used.
    Peak-shift/area default to (0.0, 1.0) by definition at cycle 1.
    """
    dcir_mOhm = np.nan
    if DCIR_SUMMARY_PARQUET.exists():
        try:
            d = pd.read_parquet(DCIR_SUMMARY_PARQUET)
            ids = d["cell_id"].astype(str)
            hit = d[(ids == cell_id) | (ids == f"{make}_{cell_id}")]
            if not hit.empty and "R0_median_Ohm" in hit.columns:
                dcir_mOhm = float(hit.iloc[0]["R0_median_Ohm"]) * 1000.0
        except Exception:  # noqa: BLE001
            pass
    x = np.array([ambient_C, default_c_rate, dcir_mOhm, 0.0, 1.0], dtype=np.float32)
    return np.where(np.isfinite(x), x, 0.0).astype(np.float32)


def _load_longterm_soh(cell_id: str, make: str) -> tuple[np.ndarray, np.ndarray]:
    """Load per-cycle (cycles, SoH) for a held-out cell from raw CSVs.

    Handles two on-disk layouts (per heldout availability note):
      * ``{MAKE}_Longterm_cell_{ID}_cycle.csv`` — cycle-summary form with
        ``discharge_cap_ah`` (CALB, REPT).
      * ``{MAKE}_Longterm_cell_{ID}.csv`` — raw V-t form (EVE); SoH is
        estimated from max-|discharge capacity| per ``cycle_no``.

    Returns empty arrays if no file is on disk (script degrades gracefully).
    """
    cycle_csv = LONGTERM_DIR / f"{make}_Longterm_cell_{cell_id}_cycle.csv"
    raw_csv = LONGTERM_DIR / f"{make}_Longterm_cell_{cell_id}.csv"

    # Actual SoH = discharge capacity divided by nameplate nominal capacity
    # (72 Ah CALB / 105 Ah EVE / 150 Ah REPT). NOT normalised to the first
    # Longterm cycle — held-out cells are second-life so their first-cycle
    # discharge already reflects prior life.
    nominal = float(NOMINAL_CAPACITY_AH.get(make.upper(), 0.0))
    if nominal <= 0:
        return np.array([]), np.array([])

    if cycle_csv.exists():
        df = pd.read_csv(cycle_csv)
        # Some CALB/REPT files carry multiple batches at the same cycle_no —
        # aggregate by median discharge_cap_ah per cycle_no to be safe.
        if "discharge_cap_ah" not in df.columns or "cycle_no" not in df.columns:
            return np.array([]), np.array([])
        df = df[df["discharge_cap_ah"] > 0].copy()
        per = (df.groupby("cycle_no", as_index=False)["discharge_cap_ah"]
                 .median()
                 .sort_values("cycle_no"))
        q = per["discharge_cap_ah"].to_numpy(dtype=np.float32)
        n = per["cycle_no"].to_numpy(dtype=np.float32)
        if q.size == 0:
            return np.array([]), np.array([])
        soh = q / nominal
        return n, soh.astype(np.float32)

    if raw_csv.exists():
        # EVE raw V-t: 6.6 M rows for EVE_0003. Do a chunked reduction.
        it = pd.read_csv(raw_csv, usecols=["cycle_no", "capacity_ah"],
                         chunksize=500_000)
        agg: dict[int, float] = {}
        for chunk in it:
            chunk = chunk[chunk["capacity_ah"] < 0]
            if chunk.empty:
                continue
            grp = chunk.groupby("cycle_no")["capacity_ah"].min()
            for k, v in grp.items():
                if k in agg:
                    agg[k] = min(agg[k], float(v))
                else:
                    agg[k] = float(v)
        if not agg:
            return np.array([]), np.array([])
        n = np.array(sorted(agg.keys()), dtype=np.float32)
        q = np.array([-agg[int(c)] for c in n], dtype=np.float32)
        soh = q / nominal
        return n, soh.astype(np.float32)

    return np.array([]), np.array([])


# ---------------------------------------------------------------------------
# 2. Per-cell SoH prediction
# ---------------------------------------------------------------------------
def predict_cell_soh(model: RULPredictor,
                     cell_id: str,
                     make: str,
                     n_cycles: Optional[int] = None,
                     soh_0: float = 1.0,
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Predict SoH trajectory for a held-out cell.

    Steps
    -----
    1. Load fitted theta (falls back to anchor centre when the YAML is
       missing — expected for all 3 held-out cells at 2026-07-12).
    2. Compute the 5-dim ``x_health`` snapshot from BoL/characterisation.
    3. Build the 11-dim branch input by concatenation.
    4. Integrate the Neural-ODE over ``[0, ..., N_cycles]`` where
       ``N_cycles`` matches the on-disk Longterm coverage (or the caller's
       override).

    Returns
    -------
    (n_grid, pred_soh) — both length-``N_cycles+1`` float32 numpy arrays.
    """
    theta_norm, _has_theta = _load_theta_norm(cell_id, make)
    x_health = _load_x_health(cell_id, make)

    if n_cycles is None:
        obs_n, _obs_soh = _load_longterm_soh(cell_id, make)
        n_cycles = int(obs_n[-1]) if obs_n.size else 200

    n_grid = np.arange(0.0, float(n_cycles) + 1.0, 1.0, dtype=np.float32)
    branch = np.concatenate([x_health, theta_norm]).astype(np.float32)

    with torch.no_grad():
        soh_0_t = torch.tensor([[float(soh_0)]], dtype=torch.float32)
        n_t = torch.from_numpy(n_grid)
        b_t = torch.from_numpy(branch).unsqueeze(0)
        traj = model(soh_0_t, n_t, b_t).squeeze(-1).squeeze(-1).cpu().numpy()
    return n_grid, traj.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Held-out metrics
# ---------------------------------------------------------------------------
def _crossing_cycle(n: np.ndarray, soh: np.ndarray, threshold: float
                    ) -> Optional[float]:
    """First cycle at which SoH crosses ``threshold`` (linearly interpolated).

    Returns ``None`` if the trajectory never reaches the threshold.
    """
    below = np.where(soh < threshold)[0]
    if below.size == 0:
        return None
    i = int(below[0])
    if i == 0:
        return float(n[0])
    n1, n2 = float(n[i - 1]), float(n[i])
    s1, s2 = float(soh[i - 1]), float(soh[i])
    if s1 == s2:
        return n2
    frac = (s1 - threshold) / (s1 - s2)
    return n1 + frac * (n2 - n1)


def heldout_metrics(cell_id: str,
                    make: str,
                    pred_soh: np.ndarray,
                    obs_soh: np.ndarray,
                    obs_cycles: np.ndarray,
                    pred_cycles: Optional[np.ndarray] = None,
                    ) -> dict:
    """Compute the four Phase-3 held-out metrics.

    * ``soh_rmse_pp``            RMSE(pred - obs) x 100 (percentage points).
    * ``knee_abs_err_cycles``    |cycle(pred==0.90) - cycle(obs==0.90)|;
                                 NaN if either trajectory never reaches 0.90.
    * ``eol_abs_err_cycles``     Same for the 0.80 crossing; the string
                                 ``"beyond horizon"`` is returned if the
                                 observed trajectory stops above 0.80.
    * ``pearson_r``              Pearson r between aligned (pred, obs).

    ``pred_cycles`` may be omitted iff ``pred_soh`` was evaluated on the
    same cycle grid as ``obs_cycles``; otherwise we interpolate.
    """
    if pred_cycles is None:
        pred_cycles = obs_cycles.astype(np.float32)
    if pred_cycles.shape != obs_cycles.shape or not np.allclose(pred_cycles, obs_cycles):
        pred_at_obs = np.interp(obs_cycles, pred_cycles, pred_soh)
    else:
        pred_at_obs = pred_soh

    diff = pred_at_obs - obs_soh
    rmse_pp = float(np.sqrt(np.mean(diff * diff)) * 100.0)

    kn_pred = _crossing_cycle(pred_cycles, pred_soh, 0.90)
    kn_obs = _crossing_cycle(obs_cycles, obs_soh, 0.90)
    knee_err = (abs(kn_pred - kn_obs)
                if (kn_pred is not None and kn_obs is not None)
                else float("nan"))

    eol_pred = _crossing_cycle(pred_cycles, pred_soh, 0.80)
    eol_obs = _crossing_cycle(obs_cycles, obs_soh, 0.80)
    if eol_obs is None:
        eol_err: float | str = "beyond horizon"
    elif eol_pred is None:
        eol_err = float("nan")
    else:
        eol_err = abs(eol_pred - eol_obs)

    if obs_soh.size < 2 or float(np.std(obs_soh)) < 1e-8 or float(np.std(pred_at_obs)) < 1e-8:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(pred_at_obs, obs_soh)[0, 1])

    return {
        "cell_id": cell_id,
        "make": make,
        "n_obs": int(obs_soh.size),
        "soh_rmse_pp": rmse_pp,
        "knee_abs_err_cycles": knee_err,
        "eol_abs_err_cycles": eol_err,
        "pearson_r": pearson,
    }


# ---------------------------------------------------------------------------
# 4. Fisher-column cosine gate
# ---------------------------------------------------------------------------
def _build_sample(cell_id: str, make: str,
                  n_grid: np.ndarray, obs_soh: np.ndarray) -> Phase3Sample:
    theta_norm, _ = _load_theta_norm(cell_id, make)
    x_health = _load_x_health(cell_id, make)
    return Phase3Sample(
        sample_id=f"{make}_{cell_id}",
        anchor_id=f"{make}_{cell_id}",
        cell_id=cell_id,
        n_traj=torch.from_numpy(n_grid.astype(np.float32)),
        soh_traj=torch.from_numpy(obs_soh.astype(np.float32)),
        x_health=torch.from_numpy(x_health),
        theta_norm=torch.from_numpy(theta_norm),
    )


def fisher_cosine_gate(model: RULPredictor,
                       cell_id: str,
                       make: str,
                       eps: float = 1e-3,
                       threshold: float = FISHER_COSINE_MAX,
                       ) -> dict:
    """R1 gate 1: |cos(dSoH/dlog k_SEI, dSoH/dlog LAM_neg)|.

    Delegates the numerical Fisher-column build to
    ``phase3_train_val.fisher_column_cosine`` (finite-difference in the
    standardised-log branch coordinate). Adds a boolean ``passes`` field
    for the ``< threshold`` gate: SEI and LAM_neg are considered cleanly
    discriminated when the operator's Jacobian columns are near-orthogonal.
    """
    obs_n, obs_soh = _load_longterm_soh(cell_id, make)
    if obs_n.size < 3:
        # No observed trajectory: use a synthetic cycle grid so the finite
        # differences remain well-defined; SoH values are placeholders.
        obs_n = np.arange(0.0, 201.0, 1.0, dtype=np.float32)
        obs_soh = np.linspace(1.0, 0.85, obs_n.size, dtype=np.float32)

    sample = _build_sample(cell_id, make, obs_n, obs_soh)
    result = _fisher_column_cosine(model, sample, eps=eps)
    abs_cos = result["abs_cosine"]
    passes = (isinstance(abs_cos, float)
              and not math.isnan(abs_cos)
              and abs_cos < threshold)
    result["passes"] = bool(passes)
    result["threshold"] = float(threshold)
    result["cell_id"] = cell_id
    result["make"] = make
    return result


# ---------------------------------------------------------------------------
# 5. Regime-swap replay
# ---------------------------------------------------------------------------
def _forward_with_branch(model: RULPredictor,
                         x_health: np.ndarray,
                         theta_norm: np.ndarray,
                         n_grid: np.ndarray,
                         soh_0: float = 1.0) -> np.ndarray:
    branch = np.concatenate([x_health, theta_norm]).astype(np.float32)
    with torch.no_grad():
        soh_0_t = torch.tensor([[float(soh_0)]], dtype=torch.float32)
        n_t = torch.from_numpy(n_grid.astype(np.float32))
        b_t = torch.from_numpy(branch).unsqueeze(0)
        return model(soh_0_t, n_t, b_t).squeeze(-1).squeeze(-1).cpu().numpy()


def _rmse_pp(a: np.ndarray, b: np.ndarray, n_a: np.ndarray, n_b: np.ndarray) -> float:
    """RMSE between two SoH trajectories evaluated on possibly-different grids."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    common_n = n_a if n_a.size <= n_b.size else n_b
    a_i = np.interp(common_n, n_a, a)
    b_i = np.interp(common_n, n_b, b)
    return float(np.sqrt(np.mean((a_i - b_i) ** 2)) * 100.0)


def regime_swap_replay(model: RULPredictor,
                       cell_a: str, cell_b: str,
                       make_a: str, make_b: str,
                       ) -> dict:
    """R1 gate 2: swap fitted theta between two held-out cells.

    Rationale
    ---------
    If the operator has learned to use theta at all, forwarding cell A's
    x_health with cell B's theta should produce a trajectory that looks
    LESS like cell A's own observed curve and MORE like cell B's — i.e.
    the swap moves the prediction toward cell B, not toward cell A.

    Metric
    ------
    For each cell we compute
        ``rmse(swapped, obs_self) - rmse(native, obs_self)``  (delta_self)
    plus
        ``rmse(swapped, obs_other) - rmse(native, obs_other)``  (delta_other).

    A "theta-sensitive" swap has ``delta_self > 0`` (worse fit to self)
    AND ``delta_other < 0`` (moves toward the other cell). We pass the
    gate if at least ONE of the two swaps shows ``delta_self > 0`` — i.e.
    the swapped prediction does not resemble the original. Zero of two
    passing means theta has no effect on the forward pass.
    """
    theta_a, _ = _load_theta_norm(cell_a, make_a)
    theta_b, _ = _load_theta_norm(cell_b, make_b)
    x_a = _load_x_health(cell_a, make_a)
    x_b = _load_x_health(cell_b, make_b)

    obs_na, obs_sa = _load_longterm_soh(cell_a, make_a)
    obs_nb, obs_sb = _load_longterm_soh(cell_b, make_b)

    # If either observed track is empty, fall back to a common horizon so
    # the forward pass still exercises the swap.
    horizon = int(max(200, obs_na[-1] if obs_na.size else 200,
                      obs_nb[-1] if obs_nb.size else 200))
    n_grid = np.arange(0.0, horizon + 1.0, 1.0, dtype=np.float32)

    native_a = _forward_with_branch(model, x_a, theta_a, n_grid)
    native_b = _forward_with_branch(model, x_b, theta_b, n_grid)
    swap_a_uses_b = _forward_with_branch(model, x_a, theta_b, n_grid)
    swap_b_uses_a = _forward_with_branch(model, x_b, theta_a, n_grid)

    def _pack(cell, obs_n, obs_s, native, swapped, other_obs_n, other_obs_s) -> dict:
        rmse_native_self = _rmse_pp(native, obs_s, n_grid, obs_n) if obs_n.size else float("nan")
        rmse_swap_self = _rmse_pp(swapped, obs_s, n_grid, obs_n) if obs_n.size else float("nan")
        rmse_native_other = (_rmse_pp(native, other_obs_s, n_grid, other_obs_n)
                             if other_obs_n.size else float("nan"))
        rmse_swap_other = (_rmse_pp(swapped, other_obs_s, n_grid, other_obs_n)
                           if other_obs_n.size else float("nan"))
        delta_self = rmse_swap_self - rmse_native_self
        delta_other = rmse_swap_other - rmse_native_other
        return {
            "cell_id": cell,
            "rmse_native_vs_self_pp": rmse_native_self,
            "rmse_swap_vs_self_pp": rmse_swap_self,
            "rmse_native_vs_other_pp": rmse_native_other,
            "rmse_swap_vs_other_pp": rmse_swap_other,
            "delta_self_pp": delta_self,
            "delta_other_pp": delta_other,
            "theta_matters": (isinstance(delta_self, float)
                              and not math.isnan(delta_self)
                              and delta_self > 0.0),
        }

    A = _pack(cell_a, obs_na, obs_sa, native_a, swap_a_uses_b, obs_nb, obs_sb)
    B = _pack(cell_b, obs_nb, obs_sb, native_b, swap_b_uses_a, obs_na, obs_sa)

    passes = A["theta_matters"] or B["theta_matters"]
    return {
        "cell_a": cell_a, "make_a": make_a,
        "cell_b": cell_b, "make_b": make_b,
        "A_native_swaps_B_theta": A,
        "B_native_swaps_A_theta": B,
        "passes": bool(passes),
        "criterion": "at_least_one_swap_shifts_off_native (delta_self > 0)",
    }


# ---------------------------------------------------------------------------
# 6. CLI + top-level runner
# ---------------------------------------------------------------------------
def _load_heldout_config() -> list[dict]:
    doc = yaml.safe_load(HELDOUT_YAML.read_text())
    return doc.get("heldout_cells", []) or []


def _fmt(v, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "PASS" if v else "FAIL"
    if isinstance(v, float):
        return "NaN" if math.isnan(v) else f"{v:.{digits}g}"
    return str(v)


def run_validation(checkpoint_path: str | Path,
                   out_md_path: Optional[str | Path] = None,
                   ) -> dict:
    """Run the held-out validation suite end-to-end and return a report dict.

    Optionally writes ``<out_md_path>`` (.md) and its ``.json`` sibling.
    """
    model = load_operator_from_checkpoint(checkpoint_path)
    cells = _load_heldout_config()

    per_cell_rows: list[dict] = []
    metrics_by_cell: dict[str, dict] = {}
    fisher_by_cell: dict[str, dict] = {}

    for c in cells:
        cell_id = str(c["id"]).split("_")[-1]  # "CALB_0029" -> "0029"
        make = str(c["make"])
        obs_n, obs_soh = _load_longterm_soh(cell_id, make)
        if obs_n.size == 0:
            per_cell_rows.append({
                "cell_id": cell_id, "make": make,
                "note": "no on-disk Longterm observation available",
            })
            continue
        # Start prediction at the observed first-cycle SoH so the operator's
        # fade curve is anchored to the same point as the ground truth. The
        # operator was trained on trajectories starting at SoH=1.0, so this
        # is a slight extrapolation, but keeps pred/obs on the same scale.
        soh_0 = float(obs_soh[0])
        pred_n, pred_soh = predict_cell_soh(model, cell_id, make, soh_0=soh_0)
        metrics = heldout_metrics(cell_id, make, pred_soh, obs_soh, obs_n,
                                  pred_cycles=pred_n)
        fisher = fisher_cosine_gate(model, cell_id, make)
        metrics["fisher_abs_cos"] = fisher["abs_cosine"]
        metrics["fisher_passes"] = fisher["passes"]
        metrics["rmse_passes"] = (isinstance(metrics["soh_rmse_pp"], float)
                                   and metrics["soh_rmse_pp"] <= SOH_RMSE_PP_MAX)
        per_cell_rows.append(metrics)
        metrics_by_cell[f"{make}_{cell_id}"] = metrics
        fisher_by_cell[f"{make}_{cell_id}"] = fisher

    # Regime-swap: pair the fast-fader (CALB_0029) with the mild EVE_0003.
    swap = regime_swap_replay(model,
                              cell_a="0029", cell_b="0003",
                              make_a="CALB", make_b="EVE")

    # Aggregate gate verdicts (overall pass = every present gate passes).
    rmse_pass = all(r.get("rmse_passes", True) for r in per_cell_rows if "soh_rmse_pp" in r)
    fisher_pass = all(r.get("fisher_passes", True) for r in per_cell_rows if "fisher_abs_cos" in r)
    swap_pass = swap.get("passes", False)
    overall = bool(rmse_pass and fisher_pass and swap_pass)

    report = {
        "checkpoint": str(checkpoint_path),
        "branch_dim": BRANCH_DIM,
        "per_cell": per_cell_rows,
        "fisher_gate": fisher_by_cell,
        "regime_swap": swap,
        "gates": {
            "soh_rmse_pp<=3.0": rmse_pass,
            "fisher_cosine<=0.3": fisher_pass,
            "regime_swap_theta_matters": swap_pass,
            "overall_pass": overall,
        },
    }

    _print_summary_table(report)

    if out_md_path is not None:
        out_md_path = Path(out_md_path)
        out_md_path.parent.mkdir(parents=True, exist_ok=True)
        out_md_path.write_text(_render_markdown(report))
        js = out_md_path.with_suffix(".json")
        js.write_text(json.dumps(report, indent=2, default=str))
        print(f"[phase3_validate] wrote {out_md_path} + {js}")

    return report


def _print_summary_table(report: dict) -> None:
    print("\n[phase3_validate] Per-cell held-out results")
    print("| cell | n_obs | RMSE (pp) | knee err | EOL err | Pearson r | Fisher |cos| |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in report["per_cell"]:
        if "note" in r:
            print(f"| {r['make']}_{r['cell_id']} | - | - | - | - | - | note: {r['note']} |")
            continue
        print(f"| {r['make']}_{r['cell_id']} | {r['n_obs']} | "
              f"{_fmt(r['soh_rmse_pp'])} | {_fmt(r['knee_abs_err_cycles'])} | "
              f"{_fmt(r['eol_abs_err_cycles'])} | {_fmt(r['pearson_r'])} | "
              f"{_fmt(r['fisher_abs_cos'])} |")
    print("\n[phase3_validate] Gates (design §5.4 + phase3_operator.yaml)")
    for k, v in report["gates"].items():
        print(f"  {k}: {_fmt(v)}")


def _render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# Phase 3 held-out validation")
    lines.append("")
    lines.append(f"- Checkpoint: `{report['checkpoint']}`")
    lines.append(f"- Branch dim: {report['branch_dim']} (5 x_health + 6 theta_norm)")
    lines.append("")
    lines.append("## Per-cell metrics")
    lines.append("| cell | n_obs | SoH RMSE (pp) | knee err (cy) | EOL err (cy) | Pearson r | Fisher |cos| |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in report["per_cell"]:
        if "note" in r:
            lines.append(f"| {r['make']}_{r['cell_id']} | - | - | - | - | - | note: {r['note']} |")
            continue
        lines.append(f"| {r['make']}_{r['cell_id']} | {r['n_obs']} | "
                     f"{_fmt(r['soh_rmse_pp'])} | {_fmt(r['knee_abs_err_cycles'])} | "
                     f"{_fmt(r['eol_abs_err_cycles'])} | {_fmt(r['pearson_r'])} | "
                     f"{_fmt(r['fisher_abs_cos'])} |")
    lines.append("")
    lines.append("## Regime-swap replay")
    swap = report["regime_swap"]
    lines.append(f"- Swap pair: {swap['make_a']}_{swap['cell_a']}  <->  "
                 f"{swap['make_b']}_{swap['cell_b']}")
    for side in ("A_native_swaps_B_theta", "B_native_swaps_A_theta"):
        s = swap[side]
        lines.append(f"  * {side}: delta_self = {_fmt(s['delta_self_pp'])} pp, "
                     f"delta_other = {_fmt(s['delta_other_pp'])} pp, "
                     f"theta_matters = {_fmt(s['theta_matters'])}")
    lines.append("")
    lines.append("## Gates")
    for k, v in report["gates"].items():
        lines.append(f"- {k}: {_fmt(v)}")
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--checkpoint",
                   default="outputs/models/phase3_operator.pt",
                   help="Path to the Phase-3 operator checkpoint (default: "
                        "outputs/models/phase3_operator.pt)")
    p.add_argument("--out",
                   default="outputs/results/phase3_heldout_validation.md",
                   help="Output Markdown path (JSON companion is auto-derived).")
    p.add_argument("--self-check", action="store_true",
                   help="Skip model I/O and only verify imports + config paths.")
    return p.parse_args(argv)


def _self_check() -> int:
    """Import-only + config-path check; does NOT require a checkpoint."""
    print("[phase3_validate] self-check")
    print(f"  BRANCH_DIM              = {BRANCH_DIM}")
    print(f"  N_HEALTH_FEATURES       = {N_HEALTH_FEATURES}")
    print(f"  N_THETA                 = {N_THETA}")
    print(f"  THETA_KEYS              = {THETA_KEYS}")
    print(f"  K_SEI_IDX / LAM_NEG_IDX = {K_SEI_IDX} / {LAM_NEG_IDX}")
    print(f"  heldout yaml exists     = {HELDOUT_YAML.exists()}")
    cells = _load_heldout_config()
    print(f"  n heldout cells         = {len(cells)}")
    for c in cells:
        cell_id = str(c["id"]).split("_")[-1]
        make = str(c["make"])
        theta, has_theta = _load_theta_norm(cell_id, make)
        x = _load_x_health(cell_id, make)
        print(f"    {make}_{cell_id}: theta_yaml={has_theta}, "
              f"theta_norm.shape={theta.shape}, x_health={x.tolist()}")
    print("  OK")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    if args.self_check:
        return _self_check()
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"[phase3_validate] checkpoint not found: {ckpt}\n"
              f"  train it first via phase3_train_val.train_operator, or\n"
              f"  re-run with --self-check to smoke-test imports.",
              file=sys.stderr)
        return 2
    run_validation(ckpt, out_md_path=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
