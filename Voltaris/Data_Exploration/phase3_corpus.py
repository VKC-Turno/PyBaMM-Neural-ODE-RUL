"""
Voltaris/Data_Exploration/phase3_corpus.py
------------------------------------------
Phase 3 — 7-anchor Sobol θ-perturbation corpus generator + PyBaMM
simulation runner (SPMe + SEI + plating + stress-LAM, isothermal 25 °C).

Public API
~~~~~~~~~~
    load_anchor_theta(anchor_id) -> dict
        {theta:{5}, bol_params_path, deg_params_path, protocol_id,
         protocol_steps, nominal_capacity_Ah}

    draw_sobol_perturbations(anchor_theta, n_samples, sigma_config,
                             decorrelation_gate, *, seed, widen,
                             max_attempts) -> ndarray[n_samples, 5]
        Sobol draws in the canonical θ order (k_SEI, V_SEI, D_SEI_solvent,
        k_plating, k_LAM_negative). Re-draws until |ρ_spearman| between the
        gate's target_pair falls below abs_threshold; falls back to the
        least-correlated draw on max_attempts exhaustion.

    run_one_sim(anchor_id, theta_perturbed, config) -> dict | None
        Single-shot PyBaMM SPMe cycle-life sim. Returns None when the sim
        passes solver but fails a quality filter (per §3.7 of phase3_design).
        Returns {trajectory, meta, quality_flag='ok'|'error'} otherwise.

    run_anchor_block(anchor_id, config, *, resume=True) -> list[dict]
        Sweep n_sims_per_anchor draws for one anchor via the subprocess
        pool pattern from src/simulation/run_sweep.py. Writes a checkpoint
        parquet at configs/phase3_corpus/{anchor_id}.parquet.

    run_full_sweep(config) -> dict
        Loops all anchors in the config; writes a global manifest at
        configs/phase3_corpus/_manifest.yaml.

Reuse
~~~~~
    - PyBaMM parameter construction / θ application from
      Voltaris/Data_Exploration/phase2_de_fit.py — keeps the corpus
      consistent with the anchors that produced fitted_theta.
    - Subprocess-thread pool from src/simulation/run_sweep.py — a
      SIGKILLed or crashed single sim never takes down the pool.

Worker entry point
~~~~~~~~~~~~~~~~~~
    python <this file> --worker <input_json> <output_pkl>

Smoke
~~~~~
    python <this file>
        Runs 2 sims on CALB_0003 with a 5-minute hard wall-clock cap.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import os
import pickle
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import yaml

# --------------------------------------------------------------------------- #
# Module-level paths
# --------------------------------------------------------------------------- #

_PROJECT_ROOT = Path("/home/hj/Desktop/PINNs")
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

THIS_FILE = Path(__file__).resolve()

CONFIGS_DIR = _PROJECT_ROOT / "configs"
PHASE3_SWEEP_YAML = CONFIGS_DIR / "phase3_sweep.yaml"
PHASE3_CORPUS_DIR = CONFIGS_DIR / "phase3_corpus"
GLOBAL_MANIFEST_PATH = PHASE3_CORPUS_DIR / "_manifest.yaml"
COHORT_PROTOCOLS_YAML = CONFIGS_DIR / "cohort_experiment_protocols.yaml"

# The 5 anchor θ axes, in canonical order (matches phase2_de_fit.THETA_SPEC
# and the fitted_theta blocks in configs/deg_params/*.yaml).
THETA_AXES: tuple[str, ...] = (
    "k_SEI",
    "V_SEI",
    "D_SEI_solvent",
    "k_plating",
    "k_LAM_negative",
)

# perturbation_sigma keys in the config use two spellings for the LAM axis.
# Resolve everything through this alias table onto the canonical name.
SIGMA_ALIAS: dict[str, str] = {
    "k_SEI": "k_SEI",
    "V_SEI": "V_SEI",
    "D_SEI_solvent": "D_SEI_solvent",
    "k_plating": "k_plating",
    "k_LAM_negative": "k_LAM_negative",
    "LAM_neg_rate_s": "k_LAM_negative",
}

# PyBaMM keys per axis (informational — the actual application goes through
# phase2_de_fit.apply_deg_params so the two stay in sync).
PYBAMM_KEYS: dict[str, str] = {
    "k_SEI":          "SEI kinetic rate constant [m.s-1]",
    "V_SEI":          "SEI partial molar volume [m3.mol-1]",
    "D_SEI_solvent":  "SEI solvent diffusivity [m2.s-1]",
    "k_plating":      "Lithium plating kinetic rate constant [m.s-1]",
    "k_LAM_negative": "Negative electrode LAM constant proportional term [s-1]",
}


# --------------------------------------------------------------------------- #
# Config loaders
# --------------------------------------------------------------------------- #

def _load_sweep_config(path: Path | str = PHASE3_SWEEP_YAML) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _load_cohort_protocols() -> dict:
    return yaml.safe_load(COHORT_PROTOCOLS_YAML.read_text())


def _resolve_project_path(maybe_rel: str) -> str:
    """Config paths like `configs/deg_params/X.yaml` are stored as
    project-root-relative. Turn them into absolutes."""
    p = Path(maybe_rel)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return str(p)


def load_anchor_theta(anchor_id: str,
                      config_path: Path | str = PHASE3_SWEEP_YAML) -> dict:
    """Read an anchor's 5-θ, BOL param path, protocol id and steps.

    Returns
    -------
    dict with keys:
        anchor_id, theta (dict of 5 physical-units floats),
        bol_params_path (abs path), deg_params_path (abs path),
        protocol_id, protocol_steps (list[str]), nominal_capacity_Ah
    """
    cfg = _load_sweep_config(config_path)
    anchor = next((a for a in cfg["anchors"] if a["id"] == anchor_id), None)
    if anchor is None:
        raise KeyError(f"anchor_id {anchor_id!r} not present in {config_path}")

    theta = {name: float(anchor["fitted_theta"][name]) for name in THETA_AXES}

    proto_id = anchor["protocol_id"]
    protos = _load_cohort_protocols()
    if proto_id not in protos["protocols"]:
        raise KeyError(
            f"protocol_id {proto_id!r} missing from {COHORT_PROTOCOLS_YAML}"
        )
    proto = protos["protocols"][proto_id]

    return dict(
        anchor_id=anchor_id,
        theta=theta,
        bol_params_path=_resolve_project_path(anchor["bol_params_source"]),
        deg_params_path=_resolve_project_path(anchor["deg_params_source"]),
        protocol_id=proto_id,
        protocol_steps=list(proto["experiment_steps"]),
        nominal_capacity_Ah=float(proto["nominal_capacity_Ah"]),
    )


# --------------------------------------------------------------------------- #
# Sobol perturbation draw
# --------------------------------------------------------------------------- #

def _sample_axis(u: np.ndarray, center: float, sigma_spec: dict,
                 widen: float = 1.0) -> np.ndarray:
    """Map Sobol uniforms u∈[0,1] → perturbed physical values around `center`.

    log10 space: v = 10 ** (log10(center) + sigma_dec * Φ⁻¹(u) * widen)
    linear space: v = center * (1 + sigma_rel * Φ⁻¹(u) * widen), clipped > 0.
    """
    from scipy.stats import norm

    z = norm.ppf(np.clip(u, 1e-9, 1.0 - 1e-9))
    space = sigma_spec.get("space", "log10")
    if space == "log10":
        sigma = float(sigma_spec["sigma_dec"]) * float(widen)
        return 10.0 ** (np.log10(max(center, 1e-30)) + sigma * z)
    if space == "linear":
        sigma_rel = float(sigma_spec["sigma_rel"]) * float(widen)
        vals = center * (1.0 + sigma_rel * z)
        return np.maximum(vals, 1e-30)
    raise ValueError(f"Unknown perturbation space: {space}")


def _abs_spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    if len(x) < 3:
        return 0.0
    rho, _ = spearmanr(x, y)
    return float(abs(rho)) if np.isfinite(rho) else 0.0


_QUADRANT_ALIAS: dict[str, str] = {
    # Config uses the design-doc shorthand; map to canonical axis names.
    "D_SEI":          "D_SEI_solvent",
    "D_SEI_solvent":  "D_SEI_solvent",
    "LAM_neg":        "k_LAM_negative",
    "LAM_neg_rate_s": "k_LAM_negative",
    "k_LAM_negative": "k_LAM_negative",
    "k_SEI":          "k_SEI",
    "V_SEI":          "V_SEI",
    "k_plating":      "k_plating",
}


def _in_quadrant(theta_row: np.ndarray, anchor_theta: dict,
                 sigma_config: dict, spec: dict) -> bool:
    """Check whether one θ draw lies in the design's fast-fade quadrant.

    ``spec['quadrant']`` describes which side of each axis counts (e.g.
    ``{D_SEI: 'floor', LAM_neg: 'ceiling'}``). Interpretation: 'floor' means
    the draw sits below the anchor centre by ≥ 0.5 σ; 'ceiling' means it
    sits above the anchor centre by ≥ 0.5 σ. Evaluated in σ-normalised
    space so scale-invariant.

    Bug fix (2026-07-10): the spec uses shorthand names like `D_SEI` and
    `LAM_neg` that DON'T match either `SIGMA_ALIAS` keys or `THETA_AXES`.
    Introduced ``_QUADRANT_ALIAS`` to map them properly — previously every
    condition silently skipped and this function returned True universally.
    """
    quadrant = spec.get("quadrant", {}) or {}
    n_checked = 0
    for spec_key, side in quadrant.items():
        canon = _QUADRANT_ALIAS.get(spec_key)
        if canon is None or canon not in THETA_AXES:
            # Loud failure — silent-skip was the previous bug's foothold
            raise ValueError(
                f"Unknown quadrant axis {spec_key!r}. "
                f"Add to _QUADRANT_ALIAS."
            )
        n_checked += 1
        axis_i = THETA_AXES.index(canon)
        val = float(theta_row[axis_i])
        center = float(anchor_theta["theta"][canon])
        # Convert to σ units
        sigma_spec = None
        for k, s in sigma_config.items():
            if SIGMA_ALIAS.get(k, k) == canon:
                sigma_spec = s
                break
        if sigma_spec is None:
            continue
        if sigma_spec.get("space", "log10") == "log10":
            sd = float(sigma_spec.get("sigma_dec", 1.0))
            if val <= 0 or center <= 0 or sd <= 0:
                return False
            z = (np.log10(val) - np.log10(center)) / sd
        else:
            sr = float(sigma_spec.get("sigma_rel", 1.0))
            denom = sr * center
            if denom == 0:
                return False
            z = (val - center) / denom
        if side == "floor" and z > -0.5:
            return False
        if side == "ceiling" and z < 0.5:
            return False
    return True


def _apply_fast_fade_booster(thetas: np.ndarray,
                              anchor_theta: dict,
                              sigma_config: dict,
                              booster_spec: dict,
                              seed_base: int,
                              widen: float = 1.0) -> np.ndarray:
    """Rejection-sample additional draws until ≥ N of them sit in the design's
    fast-fade quadrant (design R2).

    Preserves all original samples; appends supplementary samples if the
    baseline set is short. This does NOT expand ``n_samples`` beyond
    ``base + booster``; the number of surviving draws in the quadrant
    replaces the same number of least-quadrant-y baseline draws.

    Design intent: the anchor's baseline Sobol draw may land only a few
    samples in the aggressive-fade region. Post-hoc top-up guarantees the
    corpus has ≥ N such trajectories so the operator has fast-fade signal.
    """
    from scipy.stats import qmc

    min_in = int(booster_spec.get("min_samples_in_quadrant", 10))
    quadrant = booster_spec.get("quadrant", {}) or {}
    if not quadrant or min_in <= 0:
        return thetas

    # Count baseline samples already in quadrant
    n_axes = len(THETA_AXES)
    in_mask = np.array([_in_quadrant(t, anchor_theta, sigma_config,
                                        booster_spec)
                         for t in thetas])
    n_in = int(in_mask.sum())
    if n_in >= min_in:
        return thetas

    n_needed = min_in - n_in
    print(f"[phase3] fast-fade booster: baseline has {n_in}/{min_in} "
          f"in-quadrant; rejection-sampling {n_needed} more "
          f"({quadrant})", flush=True)

    # Rejection-sample via a fresh Sobol stream
    booster_thetas: list[np.ndarray] = []
    engine = qmc.Sobol(d=n_axes, seed=int(seed_base), scramble=True)
    attempts = 0
    hard_cap = 4096  # avoid infinite loops on unreachable quadrants
    while len(booster_thetas) < n_needed and attempts < hard_cap:
        u = engine.random(1)[0]
        row = np.zeros(n_axes, dtype=float)
        for i, ax in enumerate(THETA_AXES):
            center = float(anchor_theta["theta"][ax])
            # Find sigma spec for this axis
            sigma_spec = None
            for k, s in sigma_config.items():
                if SIGMA_ALIAS.get(k, k) == ax:
                    sigma_spec = s
                    break
            if sigma_spec is None:
                row[i] = center
                continue
            row[i] = _sample_axis(np.array([u[i]]), center, sigma_spec,
                                    widen=widen)[0]
        if _in_quadrant(row, anchor_theta, sigma_config, booster_spec):
            booster_thetas.append(row)
        attempts += 1

    if not booster_thetas:
        print(f"[phase3] fast-fade booster: no draws reached the quadrant "
              f"after {hard_cap} attempts; keeping baseline", flush=True)
        return thetas

    # Replace the LEAST-quadrant-like baseline draws with booster draws.
    # Simplest rule: replace draws whose in_mask is False. If more booster
    # draws than out-of-quadrant baselines, cap to n_needed.
    out_idx = np.where(~in_mask)[0]
    n_replace = min(len(booster_thetas), len(out_idx))
    result = thetas.copy()
    for r in range(n_replace):
        result[out_idx[r]] = booster_thetas[r]
    print(f"[phase3] fast-fade booster: replaced {n_replace} baseline draws "
          f"with in-quadrant boosters (attempts={attempts})", flush=True)
    return result


def draw_sobol_perturbations(anchor_theta: dict,
                              n_samples: int,
                              sigma_config: dict,
                              decorrelation_gate: dict,
                              *,
                              seed: int = 789,
                              widen: float = 1.0,
                              max_attempts: int = 32) -> np.ndarray:
    """Draw n Sobol-perturbed θ samples around one anchor.

    Returns [n_samples, 5] ndarray with the canonical THETA_AXES order.
    Re-draws (with an incremented sub-seed) until the Spearman gate on
    the target_pair passes, or max_attempts is exhausted (in which case
    the least-correlated draw seen is returned).
    """
    from scipy.stats.qmc import Sobol

    theta_center = anchor_theta["theta"]

    # Resolve sigma spec keyed by canonical axis name.
    axis_sigma: dict[str, dict] = {}
    for cfg_name, spec in sigma_config.items():
        canon = SIGMA_ALIAS.get(cfg_name, cfg_name)
        if canon in THETA_AXES:
            axis_sigma[canon] = spec

    target_pair = decorrelation_gate.get("target_pair",
                                         ["k_SEI", "k_LAM_negative"])
    a_name = SIGMA_ALIAS.get(target_pair[0], target_pair[0])
    b_name = SIGMA_ALIAS.get(target_pair[1], target_pair[1])
    idx_a = THETA_AXES.index(a_name)
    idx_b = THETA_AXES.index(b_name)
    thresh = float(decorrelation_gate.get("abs_threshold", 0.10))

    # Sobol needs to draw a power-of-two batch for its balance guarantee;
    # take the first n_samples afterwards.
    n_pow2 = 1 << max(0, int(np.ceil(np.log2(max(2, n_samples)))))

    best_theta: Optional[np.ndarray] = None
    best_rho = float("inf")

    for attempt in range(max_attempts):
        rng = Sobol(d=len(THETA_AXES), seed=int(seed) + attempt, scramble=True)
        u = rng.random(n=n_pow2)[:n_samples]  # (n, 5)

        thetas = np.zeros((n_samples, len(THETA_AXES)), dtype=float)
        for j, ax in enumerate(THETA_AXES):
            spec = axis_sigma.get(ax)
            if spec is None:
                # No sigma configured for this axis: hold at anchor value
                thetas[:, j] = float(theta_center[ax])
                continue
            thetas[:, j] = _sample_axis(u[:, j], float(theta_center[ax]),
                                         spec, widen=widen)

        rho = _abs_spearman(thetas[:, idx_a], thetas[:, idx_b])
        if rho < best_rho:
            best_rho = rho
            best_theta = thetas.copy()
        if rho < thresh:
            return thetas

    # No attempt passed the gate — return the least-correlated draw.
    assert best_theta is not None
    return best_theta


# --------------------------------------------------------------------------- #
# PyBaMM simulation
# --------------------------------------------------------------------------- #

def _build_pv_with_theta(bol_params_path: str, theta: dict,
                          temperature_C: float = 25.0):
    """BOL + θ → ParameterValues. Reuses phase2_de_fit helpers so the θ
    application layer is the single source of truth."""
    from Voltaris.Data_Exploration.phase2_de_fit import (
        build_pybamm_parameters, apply_deg_params,
    )
    bol = yaml.safe_load(Path(bol_params_path).read_text())
    pv = build_pybamm_parameters(bol)
    pv = apply_deg_params(
        pv,
        k_SEI=theta["k_SEI"],
        V_SEI=theta["V_SEI"],
        D_SEI_solvent=theta["D_SEI_solvent"],
        k_plating=theta["k_plating"],
        k_LAM_negative=theta["k_LAM_negative"],
    )
    T_K = 273.15 + float(temperature_C)
    pv.update({
        "Ambient temperature [K]": T_K,
        "Initial temperature [K]": T_K,
    })
    return pv


def _extract_trajectory(sol) -> pd.DataFrame:
    """Per-cycle SoH + LAM% + SEI thickness (end-of-cycle values)."""
    rows: list[dict] = []
    q_ref: Optional[float] = None
    for n, cy in enumerate(sol.cycles, start=1):
        best_dQ = 0.0
        for step in cy.steps:
            try:
                Imean = float(np.nanmean(step["Current [A]"].entries))
            except Exception:
                continue
            if Imean <= 1e-3:            # PyBaMM: I>0 = discharge
                continue
            try:
                Q = step["Discharge capacity [A.h]"].entries
                dQ = abs(float(Q[-1] - Q[0]))
            except Exception:
                continue
            if dQ > best_dQ:
                best_dQ = dQ
        if best_dQ <= 0:
            continue
        if q_ref is None:
            q_ref = best_dQ
        soh = best_dQ / q_ref if q_ref > 0 else float("nan")

        last_step = cy.steps[-1] if cy.steps else None

        def _last(key: str) -> float:
            if last_step is None:
                return float("nan")
            try:
                arr = last_step[key].entries
                return float(arr.flat[-1])
            except Exception:
                return float("nan")

        rows.append(dict(
            cycle_n=n,
            Q_Ah=best_dQ,
            SOH=soh,
            LAM_negative_pct=_last(
                "Loss of active material in negative electrode [%]"),
            LAM_positive_pct=_last(
                "Loss of active material in positive electrode [%]"),
            SEI_thickness_m=_last(
                "X-averaged negative SEI thickness [m]"),
            dead_lithium_Ah=_last(
                "Loss of capacity to negative lithium plating [A.h]"),
        ))
    return pd.DataFrame(rows)


def _simulate(pv, protocol_steps: Iterable[str], n_cycles: int,
              rtol: float, atol: float) -> pd.DataFrame:
    """Solve SPMe + degradation for n_cycles; return per-cycle features.
    Returns an empty DataFrame on solver failure — no raise."""
    import pybamm
    from Voltaris.Data_Exploration.phase2_de_fit import (
        MODEL_OPTIONS, _submesh_pts,
    )
    try:
        model = pybamm.lithium_ion.SPMe(options=MODEL_OPTIONS)
        solver = pybamm.IDAKLUSolver(rtol=rtol, atol=atol)
        experiment = pybamm.Experiment(
            [tuple(protocol_steps)] * int(n_cycles)
        )
        sim = pybamm.Simulation(
            model, parameter_values=pv, experiment=experiment,
            solver=solver, var_pts=_submesh_pts(),
        )
        sol = sim.solve()
        traj = _extract_trajectory(sol)
        del sol, sim, experiment
        gc.collect()
        return traj
    except Exception:
        gc.collect()
        return pd.DataFrame()


def _apply_quality_filters(traj: pd.DataFrame, filters: dict
                           ) -> tuple[bool, str]:
    """§3.7 quality filters. Returns (passed, reason)."""
    if traj.empty:
        return False, "empty_trajectory"
    if len(traj) < int(filters.get("min_cycles_kept", 100)):
        return False, f"too_few_cycles({len(traj)})"
    if traj["SOH"].isna().any() or (traj["Q_Ah"] <= 0).any():
        return False, "nan_or_negative_capacity"
    soh_final = float(traj["SOH"].iloc[-1])
    if soh_final > float(filters.get("final_soh_max", 0.98)):
        return False, f"soh_final_too_high({soh_final:.3f})"
    if soh_final < float(filters.get("final_soh_min", 0.55)):
        return False, f"soh_final_too_low({soh_final:.3f})"
    tol = float(filters.get("monotonicity_tol", 0.005))
    soh_arr = traj["SOH"].to_numpy(dtype=float)
    # Positive diff = SoH rising cycle-over-cycle (unphysical); small
    # tol accommodates numerical noise.
    diffs = soh_arr[1:] - soh_arr[:-1]
    if diffs.size and float(np.max(diffs)) > tol:
        return False, "monotonicity_violation"
    return True, "ok"


# --------------------------------------------------------------------------- #
# Public: run_one_sim
# --------------------------------------------------------------------------- #

def run_one_sim(anchor_id: str,
                theta_perturbed: dict | list | tuple | np.ndarray,
                config: dict) -> Optional[dict]:
    """Run one PyBaMM sim for a perturbed θ around `anchor_id`.

    Returns
    -------
    dict {trajectory, meta, quality_flag}   when solver produced a
                                            trajectory (quality_flag is
                                            'ok' if filters passed, else
                                            'rejected' with meta['reason']).
    None                                    when the solver failed to
                                            produce any usable cycles.

    (The design brief said "returns None on filter fail"; we tighten that
    to "None on solver-death" and preserve rejected-but-run sims with a
    quality_flag so run_anchor_block can log the θ that caused each
    rejection — the manifest is more useful with the reason attached.)
    """
    if isinstance(theta_perturbed, dict):
        theta = {k: float(theta_perturbed[k]) for k in THETA_AXES}
    else:
        arr = np.asarray(theta_perturbed, dtype=float).ravel()
        if arr.size != len(THETA_AXES):
            raise ValueError(
                f"theta_perturbed must be length {len(THETA_AXES)}, "
                f"got {arr.size}"
            )
        theta = dict(zip(THETA_AXES, [float(v) for v in arr]))

    anchor = load_anchor_theta(anchor_id)

    model_cfg = config.get("model", {}) or {}
    horizon = config.get("horizon", {}) or {}
    filters = config.get("quality_filters", {}) or {}

    n_cycles = int(horizon.get("max_cycles", 2500))
    soh_stop = float(horizon.get("soh_stop", 0.65))
    rtol = float(model_cfg.get("rtol", 1e-6))
    atol = float(model_cfg.get("atol", 1e-6))
    T_C = float(model_cfg.get("temperature_C", 25.0))

    t0 = time.time()

    try:
        pv = _build_pv_with_theta(anchor["bol_params_path"], theta,
                                    temperature_C=T_C)
    except Exception as e:
        return dict(
            trajectory=pd.DataFrame(),
            meta=dict(anchor_id=anchor_id, theta=theta,
                       protocol_id=anchor["protocol_id"],
                       elapsed_s=time.time() - t0,
                       reason=f"pv_build_error: {type(e).__name__}: {e}"),
            quality_flag="error",
        )

    traj = _simulate(pv, anchor["protocol_steps"], n_cycles=n_cycles,
                      rtol=rtol, atol=atol)

    if traj.empty:
        # Solver died before any cycle — per spec, treat as filter fail.
        return None

    # Horizon cut: truncate at first cycle where SoH < soh_stop.
    soh_arr = traj["SOH"].to_numpy(dtype=float)
    below = np.where(soh_arr < soh_stop)[0]
    if below.size:
        traj = traj.iloc[: int(below[0]) + 1].reset_index(drop=True)

    passed, reason = _apply_quality_filters(traj, filters)

    meta = dict(
        anchor_id=anchor_id,
        theta=theta,
        protocol_id=anchor["protocol_id"],
        n_cycles_run=int(len(traj)),
        soh_final=float(traj["SOH"].iloc[-1]) if not traj.empty else float("nan"),
        elapsed_s=time.time() - t0,
        reason=reason,
    )

    if passed:
        return dict(trajectory=traj, meta=meta, quality_flag="ok")
    return dict(trajectory=traj, meta=meta, quality_flag="rejected")


# --------------------------------------------------------------------------- #
# Subprocess pool (reuse the run_sweep.py pattern)
# --------------------------------------------------------------------------- #

def _run_subprocess_pool(payloads: list[dict], *, n_jobs: int, timeout_s: int
                          ) -> list[dict]:
    """Dispatch each sim as `python <this file> --worker in.json out.pkl`.

    A ThreadPoolExecutor with `n_jobs` workers serialises `subprocess.run`
    calls; each subprocess uses one CPU core. On timeout we fabricate an
    'error' result and continue — the pool never blocks or dies with the
    child.
    """
    tmp = Path(tempfile.mkdtemp(prefix="phase3_pool_"))
    results: list[Optional[dict]] = [None] * len(payloads)
    completed = 0
    t_start = time.time()

    def _one(idx: int) -> dict:
        pay = payloads[idx]
        sid = pay["sample_id"]
        in_p = tmp / f"{sid}.json"
        out_p = tmp / f"{sid}.pkl"
        in_p.write_text(json.dumps(pay, default=str))
        argv = [
            sys.executable, str(THIS_FILE),
            "--worker", str(in_p), str(out_p),
        ]
        t0 = time.time()
        try:
            subprocess.run(argv, timeout=timeout_s, check=False,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           cwd=str(_PROJECT_ROOT))
        except subprocess.TimeoutExpired:
            return {
                "outcome": "error",
                "trajectory": pd.DataFrame(),
                "meta": {"sample_id": sid,
                          "error": f"TimeoutExpired ({timeout_s}s)",
                          "elapsed_s": time.time() - t0},
            }
        if out_p.exists():
            try:
                with open(out_p, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                return {
                    "outcome": "error",
                    "trajectory": pd.DataFrame(),
                    "meta": {"sample_id": sid,
                              "error": f"unpickle: {type(e).__name__}: {e}",
                              "elapsed_s": time.time() - t0},
                }
        return {
            "outcome": "error",
            "trajectory": pd.DataFrame(),
            "meta": {"sample_id": sid,
                      "error": "subprocess produced no output file",
                      "elapsed_s": time.time() - t0},
        }

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as pool:
            futures = {pool.submit(_one, i): i for i in range(len(payloads))}
            for fut in concurrent.futures.as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
                completed += 1
                r = results[idx]
                elapsed = time.time() - t_start
                print(f"  [{completed:4d}/{len(payloads)}] "
                      f"{payloads[idx]['sample_id']}: "
                      f"{r['outcome']}  "
                      f"(wall={elapsed:.1f}s)",
                      flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return [r for r in results if r is not None]


# --------------------------------------------------------------------------- #
# Public: run_anchor_block, run_full_sweep
# --------------------------------------------------------------------------- #

def _anchor_index(anchor_id: str, cfg: dict) -> int:
    for i, a in enumerate(cfg["anchors"]):
        if a["id"] == anchor_id:
            return i
    raise KeyError(f"anchor_id {anchor_id!r} not in config")


def _worker_config_slice(config: dict) -> dict:
    """Only ship the pieces of the config the worker actually reads —
    keeps the payload lean and JSON-safe."""
    return {
        "model": dict(config.get("model", {})),
        "horizon": dict(config.get("horizon", {})),
        "quality_filters": dict(config.get("quality_filters", {})),
    }


def run_anchor_block(anchor_id: str, config: dict, *,
                      resume: bool = True) -> list[dict]:
    """Sobol-sweep one anchor. Writes a checkpoint parquet at
    configs/phase3_corpus/{anchor_id}.parquet.

    Returns a list of per-sim summaries (one dict per attempted sim,
    regardless of outcome).
    """
    PHASE3_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = PHASE3_CORPUS_DIR / f"{anchor_id}.parquet"
    if resume and ckpt_path.exists():
        print(f"[phase3] {anchor_id}: checkpoint present, skipping "
               f"({ckpt_path})", flush=True)
        return [{"sample_id": None, "anchor_id": anchor_id,
                  "outcome": "skipped_from_checkpoint", "n_cycles": 0}]

    anchor = load_anchor_theta(anchor_id)
    anchor_idx = _anchor_index(anchor_id, config)

    # 2× wider bounds on REPT anchors per R4.
    widen = (float(config.get("rept_anchor_bound_widen", 1.0))
              if anchor_id.startswith("REPT_") else 1.0)

    n_samples = int(config.get("n_sims_per_anchor", 70))
    seed = int(config.get("seed_base", 789)) + anchor_idx

    thetas = draw_sobol_perturbations(
        anchor_theta=anchor,
        n_samples=n_samples,
        sigma_config=config["perturbation_sigma"],
        decorrelation_gate=config["decorrelation_gate"],
        seed=seed,
        widen=widen,
    )

    # Bug fix (2026-07-10 adversarial audit): fast-fade booster (design R2).
    # For REPT fast-fade anchors, require ≥N samples in the
    # (D_SEI = floor, LAM_neg = ceiling) quadrant to guarantee coverage of
    # the aggressive-fade regime; top up by rejection-sampling from a
    # secondary Sobol stream if the primary draw is short.
    booster = config.get("fast_fade_booster") or {}
    booster_anchors = booster.get("anchors", []) or []
    if booster and anchor_id in booster_anchors:
        thetas = _apply_fast_fade_booster(
            thetas,
            anchor_theta=anchor,
            sigma_config=config["perturbation_sigma"],
            booster_spec=booster,
            seed_base=seed + 10_000,
            widen=widen,
        )
    if len(thetas) != n_samples:
        n_samples = len(thetas)

    worker_cfg = _worker_config_slice(config)

    payloads: list[dict] = []
    for i in range(n_samples):
        payloads.append({
            "sample_id": f"{anchor_id}_s{i:04d}",
            "anchor_id": anchor_id,
            "theta_list": [float(v) for v in thetas[i]],
            "config": worker_cfg,
        })

    compute_cfg = config.get("compute", {}) or {}
    n_jobs = int(compute_cfg.get("n_jobs", 5))
    timeout_s = int(compute_cfg.get("timeout_per_sim_s", 900))

    print(f"[phase3] {anchor_id}: {n_samples} sims, "
           f"n_jobs={n_jobs}, timeout={timeout_s}s, widen={widen}",
           flush=True)

    pool_out = _run_subprocess_pool(payloads, n_jobs=n_jobs,
                                     timeout_s=timeout_s)

    # Assemble the checkpoint parquet — all per-cycle rows for sims that
    # produced a trajectory (ok or rejected).
    all_frames: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for pay, res in zip(payloads, pool_out):
        outcome = res.get("outcome", "error")
        traj = res.get("trajectory", pd.DataFrame())
        meta = res.get("meta", {}) or {}
        if not traj.empty:
            enriched = traj.copy()
            enriched.insert(0, "sample_id", pay["sample_id"])
            enriched.insert(1, "anchor_id", anchor_id)
            enriched["outcome"] = outcome
            for j, ax in enumerate(THETA_AXES):
                enriched[f"theta_{ax}"] = pay["theta_list"][j]
            enriched["protocol_id"] = anchor["protocol_id"]
            all_frames.append(enriched)
        summaries.append(dict(
            sample_id=pay["sample_id"],
            anchor_id=anchor_id,
            outcome=outcome,
            n_cycles=int(len(traj)),
            soh_final=(float(traj["SOH"].iloc[-1])
                        if not traj.empty else float("nan")),
            elapsed_s=float(meta.get("elapsed_s", float("nan"))),
            reason=str(meta.get("reason", meta.get("error", ""))),
            theta_list=pay["theta_list"],
        ))

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_parquet(ckpt_path, index=False)
        print(f"[phase3] {anchor_id}: wrote {len(combined):,} rows → {ckpt_path}",
               flush=True)
    else:
        print(f"[phase3] {anchor_id}: no usable trajectories to write",
               flush=True)

    return summaries


def run_full_sweep(config: dict | None = None) -> dict:
    """Loop all anchors; write configs/phase3_corpus/_manifest.yaml."""
    if config is None:
        config = _load_sweep_config()
    PHASE3_CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    per_anchor: dict[str, list[dict]] = {}
    for anchor in config["anchors"]:
        per_anchor[anchor["id"]] = run_anchor_block(anchor["id"], config)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config_snapshot": config,
        "elapsed_seconds": time.time() - t0,
        "per_anchor": {
            a: {
                "n_sims": len(rows),
                "n_ok":       sum(1 for r in rows if r["outcome"] == "ok"),
                "n_rejected": sum(1 for r in rows if r["outcome"] == "rejected"),
                "n_error":    sum(1 for r in rows if r["outcome"] == "error"),
                "n_skipped":  sum(1 for r in rows
                                    if r["outcome"] == "skipped_from_checkpoint"),
                "summaries": rows,
            }
            for a, rows in per_anchor.items()
        },
    }
    GLOBAL_MANIFEST_PATH.write_text(yaml.safe_dump(manifest, sort_keys=False))
    print(f"[phase3] manifest written → {GLOBAL_MANIFEST_PATH}", flush=True)
    return manifest


# --------------------------------------------------------------------------- #
# Worker mode — invoked by the subprocess pool
# --------------------------------------------------------------------------- #

def _worker_main(input_json: str, output_pkl: str) -> int:
    payload = json.loads(Path(input_json).read_text())
    try:
        res = run_one_sim(
            anchor_id=payload["anchor_id"],
            theta_perturbed=payload["theta_list"],
            config=payload["config"],
        )
        if res is None:
            out = {"outcome": "rejected",
                    "trajectory": pd.DataFrame(),
                    "meta": {"sample_id": payload["sample_id"],
                              "reason": "solver_death_or_empty_trajectory"}}
        else:
            out = {"outcome": res.get("quality_flag", "ok"),
                    "trajectory": res["trajectory"],
                    "meta": {**res.get("meta", {}),
                              "sample_id": payload["sample_id"]}}
    except BaseException as e:
        out = {"outcome": "error",
                "trajectory": pd.DataFrame(),
                "meta": {"sample_id": payload.get("sample_id", "?"),
                          "error": f"{type(e).__name__}: {e}",
                          "traceback": traceback.format_exc()}}
    Path(output_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(out, f)
    return 0


# --------------------------------------------------------------------------- #
# Smoke — 2 sims on CALB_0003 with a 5-minute hard wall-clock cap
# --------------------------------------------------------------------------- #

class _SmokeTimeout(Exception):
    pass


def _smoke() -> int:
    print("=== phase3_corpus smoke: CALB_0003, n_samples=2, "
          "5-min hard cap ===", flush=True)

    cfg = _load_sweep_config()
    cfg["n_sims_per_anchor"] = 2
    # Tighten the horizon and filters so a smoke-scale sim doesn't get
    # auto-rejected purely for lack of cycles.
    cfg["horizon"] = {"max_cycles": 40, "soh_stop": 0.20}
    cfg["quality_filters"] = {
        "final_soh_max": 1.05,
        "final_soh_min": 0.01,
        "monotonicity_tol": 0.05,
        "min_cycles_kept": 5,
    }
    cfg["compute"] = {
        **(cfg.get("compute", {}) or {}),
        "n_jobs": 2,
        "timeout_per_sim_s": 240,
    }

    # Wipe any existing CALB_0003 checkpoint so the smoke actually runs
    # sims (rather than skipping into a resume path).
    ckpt = PHASE3_CORPUS_DIR / "CALB_0003.parquet"
    if ckpt.exists():
        ckpt.unlink()

    # Hard 5-minute wall-clock timeout on the driver process.
    def _sigalrm(_signum, _frame):
        raise _SmokeTimeout("smoke exceeded 5-minute wall cap")

    signal.signal(signal.SIGALRM, _sigalrm)
    signal.alarm(5 * 60)

    t0 = time.time()
    try:
        summaries = run_anchor_block("CALB_0003", cfg, resume=False)
    except _SmokeTimeout as e:
        print(f"smoke: HARD TIMEOUT: {e}", flush=True)
        return 2
    finally:
        signal.alarm(0)
    wall = time.time() - t0

    counts = {"ok": 0, "rejected": 0, "error": 0, "other": 0}
    for s in summaries:
        counts[s["outcome"]] = counts.get(s["outcome"], 0) + 1

    n_completed = counts.get("ok", 0) + counts.get("rejected", 0)
    print(f"smoke: {n_completed}/2 sims completed in {wall:.1f}s "
          f"(ok={counts.get('ok', 0)}, "
          f"rejected={counts.get('rejected', 0)}, "
          f"errored={counts.get('error', 0)})", flush=True)

    for s in summaries:
        print(f"    {s['sample_id']}: outcome={s['outcome']}  "
              f"n_cycles={s['n_cycles']}  "
              f"soh_final={s['soh_final']:.4f}  "
              f"elapsed={s['elapsed_s']:.1f}s", flush=True)

    return 0 if n_completed >= 1 else 1


# --------------------------------------------------------------------------- #
# CLI dispatch
# --------------------------------------------------------------------------- #

def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode")

    p_worker = sub.add_parser("--worker",
                               help="subprocess worker: run one sim")
    p_worker.add_argument("input_json")
    p_worker.add_argument("output_pkl")

    # Also allow bare --worker positional form for parity with the argv
    # the pool actually builds.
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        if len(sys.argv) < 4:
            print("usage: phase3_corpus.py --worker <input_json> <output_pkl>",
                   file=sys.stderr)
            return 2
        return _worker_main(sys.argv[2], sys.argv[3])

    # No args → smoke
    return _smoke()


if __name__ == "__main__":
    raise SystemExit(_main())
