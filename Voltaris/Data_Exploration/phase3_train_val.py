"""
Voltaris/Data_Exploration/phase3_train_val.py
=============================================

Phase 3 — theta-aware operator training loop + held-out validation harness,
implementing the R1-gate ablations specified in
`configs/phase3_design.md` sections 4 and 5.

The operator is `RULPredictor` (src/pinn/model.py) with the branch input
widened to `x_health(5) + theta_norm(6) = 11`. Loss composition reuses the
Phase-1 pieces from `src/pinn/loss.py` and ADDS `L_shape` = weighted MSE on
curvature + knee-cycle location.

Public API
----------
- ``train_operator(dataset_parquet_path, config_path) -> (checkpoint_path, curves_png)``
- ``run_validation_suite(checkpoint_path, held_out_cells, corpus_dir) -> dict``
- ``save_validation_report(report, path) -> None``

R1 gate ablations (all inside ``run_validation_suite``):

1. **Fisher-column cosine**: |cos(dSoH/dlog k_SEI, dSoH/dlog LAM_neg)|
   computed along each held-out trajectory. Design target: < 0.3.
2. **Regime-swap replay on CALB_0003**: forward SoH with (a) joint DE theta
   vs (b) SEI-only theta (LAM_neg driven to floor). Reports max |Delta SoH|
   over the [0.95, 0.80] window; must exceed 0.01 (1 pp).
3. **Attribution-slope test**: forward SoH sensitivities to +1 sigma k_SEI
   vs +1 sigma LAM_neg. Reports both intercept-difference and slope
   difference of dSoH/dn along the trajectory; slopes must differ.

Smoke test at the bottom of this file:

    .venv/bin/python /home/hj/Desktop/PINNs/Voltaris/Data_Exploration/phase3_train_val.py

instantiates the operator with random weights, generates 4 dummy trajectory
samples in memory, runs every validation function and prints the resulting
report dict. It does NOT touch real corpus data.
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# ---------------------------------------------------------------------------
# Project imports (Neural-ODE + base loss composition)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path("/home/hj/Desktop/PINNs")
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pinn.model import RULPredictor  # noqa: E402
from src.pinn.loss import (  # noqa: E402
    LossWeights,
    _ode_derivative_along_trajectory,
    _physics_target,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#
# Theta parameter order used everywhere in Phase 3. This mirrors
# `configs/phase3_sweep.yaml` (6 perturbed parameters).
#
THETA_KEYS: tuple[str, ...] = (
    "k_SEI",
    "V_SEI",
    "D_SEI_solvent",
    "k_plating",
    "LAM_neg_rate_s",
    "LAM_pos_rate_s",
)
K_SEI_IDX = THETA_KEYS.index("k_SEI")
LAM_NEG_IDX = THETA_KEYS.index("LAM_neg_rate_s")

# Match src/pinn/dataset.py::HEALTH_FEATURES; only the length matters here.
N_HEALTH_FEATURES = 5
N_THETA = len(THETA_KEYS)
BRANCH_DIM = N_HEALTH_FEATURES + N_THETA  # 11


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Phase3Sample:
    """A single trajectory used by Phase-3 training / validation.

    Attributes
    ----------
    sample_id : str
    anchor_id : str
        Which of the seven anchors this sample was drawn around (used for
        stratified splitting and per-anchor validation aggregation).
    cell_id : str | None
        For real held-out cells; None for synthetic corpus rows.
    n_traj : torch.Tensor  (T,)
    soh_traj : torch.Tensor  (T,)
    x_health : torch.Tensor  (5,)
    theta_norm : torch.Tensor  (6,)
        Standardised theta in log-space where applicable (see
        ``normalise_theta``). This is the branch conditioning.
    """

    sample_id: str
    anchor_id: str
    n_traj: torch.Tensor
    soh_traj: torch.Tensor
    x_health: torch.Tensor
    theta_norm: torch.Tensor
    cell_id: Optional[str] = None

    @property
    def branch(self) -> torch.Tensor:
        """Concatenated 11-dim conditioning vector."""
        return torch.cat([self.x_health, self.theta_norm], dim=-1)


# ---------------------------------------------------------------------------
# Theta normalisation
# ---------------------------------------------------------------------------
def default_theta_stats() -> dict[str, dict[str, float]]:
    """Fallback theta normalisation stats (means / stds).

    In production these are refreshed from the corpus at ``train_operator``
    time. Values below are order-of-magnitude anchors so unit tests and the
    smoke path do not accidentally divide by zero.
    """
    return {
        "k_SEI": {"space": "log10", "mean": -11.5, "std": 0.6},
        "V_SEI": {"space": "linear", "mean": 1.3e-4, "std": 2e-5},
        "D_SEI_solvent": {"space": "log10", "mean": -20.8, "std": 0.7},
        "k_plating": {"space": "log10", "mean": -11.7, "std": 0.5},
        "LAM_neg_rate_s": {"space": "log10", "mean": -9.5, "std": 0.8},
        "LAM_pos_rate_s": {"space": "log10", "mean": -10.5, "std": 0.3},
    }


def _to_encoding(value: float, space: str) -> float:
    if space == "log10":
        v = max(float(value), 1e-30)
        return math.log10(v)
    return float(value)


def normalise_theta(theta_dict: dict[str, float],
                    stats: Optional[dict[str, dict[str, float]]] = None
                    ) -> np.ndarray:
    """Standardise a physical-unit theta dict to a 6-vector.

    log10-space parameters are transformed to log10 first, linear-space
    parameters stay linear; both are then z-scored by ``stats``.
    """
    stats = stats or default_theta_stats()
    out = np.zeros(N_THETA, dtype=np.float32)
    for i, k in enumerate(THETA_KEYS):
        s = stats[k]
        v = _to_encoding(theta_dict[k], s["space"])
        out[i] = (v - s["mean"]) / (s["std"] + 1e-12)
    return out


# ---------------------------------------------------------------------------
# Shape loss (curvature + knee-cycle location)
# ---------------------------------------------------------------------------
def _second_derivative(n: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """d2y/dn2 via nested torch.gradient (central + one-sided ends)."""
    d1 = torch.gradient(y, spacing=(n,))[0]
    d2 = torch.gradient(d1, spacing=(n,))[0]
    return d2


def _knee_cycle(n: torch.Tensor, soh: torch.Tensor) -> torch.Tensor:
    """Soft knee-cycle estimate: expected cycle under the softmax over
    -d^2 SoH/dn^2 (i.e. the inflection point where fade curvature is most
    negative). Differentiable wrt ``soh`` if it was produced under autograd.
    """
    d2 = _second_derivative(n, soh)
    logits = -d2 * 50.0                     # sharpen the peak
    weights = torch.softmax(logits, dim=0)
    return (weights * n).sum()


def shape_loss(n_traj: torch.Tensor,
               soh_pred: torch.Tensor,
               soh_obs: torch.Tensor,
               curvature_weight: float = 1.0,
               knee_weight: float = 1.0,
               ) -> torch.Tensor:
    """L_shape = curvature_weight * MSE(curvature) + knee_weight * MSE(knee_cycle).

    Curvatures are computed on both trajectories against the same cycle grid;
    the knee-cycle term uses the soft argmin defined in ``_knee_cycle`` so
    the loss remains differentiable.
    """
    if n_traj.numel() < 3:
        # Not enough points to define curvature — return zero (no signal).
        return torch.zeros((), dtype=soh_pred.dtype, device=soh_pred.device)

    d2_pred = _second_derivative(n_traj, soh_pred)
    d2_obs = _second_derivative(n_traj, soh_obs)
    L_curv = F.mse_loss(d2_pred, d2_obs)

    n_span = float(n_traj[-1] - n_traj[0] + 1e-6)
    kn_pred = _knee_cycle(n_traj, soh_pred)
    kn_obs = _knee_cycle(n_traj, soh_obs)
    L_knee = ((kn_pred - kn_obs) / n_span).pow(2)

    return curvature_weight * L_curv + knee_weight * L_knee


# ---------------------------------------------------------------------------
# Loss composition (reuses src/pinn/loss.py pieces)
# ---------------------------------------------------------------------------
@dataclass
class Phase3LossWeights:
    data: float = 1.0
    physics: float = 0.2
    monotonicity: float = 0.5
    shape: float = 0.3
    shape_curvature: float = 1.0
    shape_knee: float = 1.0


def phase3_trajectory_loss(model: RULPredictor,
                           sample: Phase3Sample,
                           weights: Phase3LossWeights,
                           ) -> dict[str, torch.Tensor]:
    """Compute the Phase-3 composite loss for a single sample.

    Uses ``x_health`` widened to include ``theta_norm`` so the branch sees
    theta explicitly. Returns a dict of scalar tensors — keep them attached
    to the graph for backward.
    """
    n = sample.n_traj
    soh_obs = sample.soh_traj
    branch = sample.branch                 # (11,)
    soh_0 = soh_obs[0].view(1, 1)

    traj = model(soh_0, n, branch.unsqueeze(0)).squeeze(-1).squeeze(-1)  # (T,)

    L_data = F.mse_loss(traj, soh_obs)

    # Physics — dSOH/dn from ODE vs finite-difference of observation
    target_d = _physics_target(n, soh_obs)
    pred_d = _ode_derivative_along_trajectory(model, traj.detach(), n, branch)
    L_phys = F.mse_loss(pred_d, target_d)

    # Monotonicity
    diffs = traj[1:] - traj[:-1]
    L_mono = torch.relu(diffs).pow(2).mean() if diffs.numel() > 0 else torch.zeros_like(L_data)

    # Shape (curvature + knee)
    L_shape = shape_loss(n, traj, soh_obs,
                         curvature_weight=weights.shape_curvature,
                         knee_weight=weights.shape_knee)

    total = (weights.data * L_data
             + weights.physics * L_phys
             + weights.monotonicity * L_mono
             + weights.shape * L_shape)
    return {
        "total": total,
        "data": L_data,
        "physics": L_phys,
        "monotonicity": L_mono,
        "shape": L_shape,
    }


# ---------------------------------------------------------------------------
# Batched Phase-3 trajectory loss
# ---------------------------------------------------------------------------
def _erode_mask_1d(m: torch.Tensor, k: int = 2) -> torch.Tensor:
    """Boolean erosion along dim 0 by ``k`` positions on each side.

    A cell survives only if every neighbour within +-k along dim 0 is also
    True (i.e. every one of the 2k+1 consecutive rows centred on it is real).
    """
    out = m
    zero_row = torch.zeros_like(out[:1])
    for _ in range(k):
        shift_l = torch.cat([out[1:], zero_row], dim=0)   # neighbour t+1
        shift_r = torch.cat([zero_row, out[:-1]], dim=0)  # neighbour t-1
        out = out & shift_l & shift_r
    return out


def phase3_trajectory_loss_batched(model: RULPredictor,
                                   samples: Sequence[Phase3Sample],
                                   weights: Phase3LossWeights,
                                   ) -> dict[str, torch.Tensor]:
    """Batched Phase-3 composite loss over an anchor-homogeneous batch.

    All samples are padded to a shared ``n_grid`` (right-pad with the last
    real value, so finite-difference derivatives on the padded tail are ~0
    and self-cancel in the physics term). A boolean ``mask: (T_max, B)``
    marks real timesteps; every loss component averages only over real
    (t, sample) entries.

    Optimisations vs the per-sample loop:
      * one ``odeint`` call instead of B
      * physics term calls ``model.ode.net`` in bulk on ``(T_max*B, in_dim)``,
        skipping the T-length Python loop through ``model.ode.forward``
      * curvature / knee terms use vectorised ``torch.gradient`` along dim 0

    Contract: ``samples`` MUST share ``anchor_id`` (PerAnchorBatchSampler
    already guarantees this). The function does not check; violating the
    contract still returns a valid loss but shape/knee terms lose meaning.
    """
    B = len(samples)
    if B == 0:
        raise ValueError("phase3_trajectory_loss_batched: empty batch")

    ref_dtype = samples[0].soh_traj.dtype
    ref_device = samples[0].soh_traj.device

    lengths = [int(s.n_traj.numel()) for s in samples]
    T_max = max(lengths)

    # Detect the features-dataset common case where every n_traj == arange(T_i)
    def _is_arange(n: torch.Tensor) -> bool:
        return bool(torch.equal(
            n.to(dtype=torch.float32),
            torch.arange(n.numel(), dtype=torch.float32),
        ))

    features_case = all(_is_arange(s.n_traj) for s in samples)

    mask = torch.zeros(T_max, B, dtype=torch.bool, device=ref_device)
    soh_obs_pad = torch.zeros(T_max, B, dtype=ref_dtype, device=ref_device)

    if features_case:
        n_grid = torch.arange(T_max, dtype=ref_dtype, device=ref_device)
        for j, s in enumerate(samples):
            L = lengths[j]
            soh_obs_pad[:L, j] = s.soh_traj
            if L < T_max:
                soh_obs_pad[L:, j] = s.soh_traj[-1]
            mask[:L, j] = True
    else:
        # Raw-corpus fallback: shared grid = sorted union of unique cycle
        # values across the batch; each sample marks the rows it actually
        # populated. Pads are held at the sample's last real value.
        union = torch.unique(torch.cat([s.n_traj for s in samples]))
        n_grid, _ = torch.sort(union.to(dtype=ref_dtype, device=ref_device))
        T_max = int(n_grid.numel())
        mask = torch.zeros(T_max, B, dtype=torch.bool, device=ref_device)
        soh_obs_pad = torch.zeros(T_max, B, dtype=ref_dtype, device=ref_device)
        for j, s in enumerate(samples):
            idx = torch.searchsorted(n_grid, s.n_traj.to(dtype=ref_dtype))
            soh_obs_pad[idx, j] = s.soh_traj
            mask[idx, j] = True
            # constant extrap: fill any non-real row for this column with the
            # sample's terminal SOH so gradients on the pad cancel out.
            last = float(s.soh_traj[-1])
            col_mask = mask[:, j]
            if (~col_mask).any():
                soh_obs_pad[~col_mask, j] = last

    soh_0 = torch.stack([s.soh_traj[0].view(1) for s in samples], dim=0)  # (B, 1)
    branch = torch.stack([s.branch for s in samples], dim=0)              # (B, D)

    # --- one odeint call over the padded grid ---
    traj = model(soh_0, n_grid, branch).squeeze(-1)  # (T_max, B)

    mask_f = mask.to(dtype=traj.dtype)
    n_real = mask_f.sum().clamp_min(1.0)

    # --- L_data (masked MSE over real cells) ---
    L_data = ((traj - soh_obs_pad) ** 2 * mask_f).sum() / n_real

    # --- L_physics (vectorised bulk ode.net call, mask-averaged) ---
    target_d = torch.gradient(soh_obs_pad, spacing=(n_grid,), dim=0)[0]  # (T_max, B)

    traj_det = traj.detach()
    n_norm = (n_grid.view(T_max, 1) / model.ode.n_max).expand(T_max, B)
    branch_norm = model.normalise_features(branch)                         # (B, D)
    branch_bcast = branch_norm.unsqueeze(0).expand(T_max, B, branch_norm.shape[-1])
    ode_inp = torch.cat([
        traj_det.unsqueeze(-1),   # (T_max, B, 1)
        n_norm.unsqueeze(-1),     # (T_max, B, 1)
        branch_bcast,             # (T_max, B, D)
    ], dim=-1)
    raw = model.ode.net(ode_inp.reshape(T_max * B, -1)).view(T_max, B)
    pred_d = -F.softplus(raw)
    L_phys = ((pred_d - target_d) ** 2 * mask_f).sum() / n_real

    # --- L_monotonicity (positive-step penalty on real->real pairs) ---
    if T_max >= 2:
        diffs = traj[1:] - traj[:-1]                       # (T_max-1, B)
        mask_pair = (mask[1:] & mask[:-1]).to(dtype=traj.dtype)
        denom_pair = mask_pair.sum().clamp_min(1.0)
        L_mono = ((F.relu(diffs) ** 2) * mask_pair).sum() / denom_pair
    else:
        L_mono = torch.zeros((), dtype=traj.dtype, device=traj.device)

    # --- L_shape (curvature MSE + knee-cycle location) ---
    if T_max >= 3:
        d1_pred = torch.gradient(traj, spacing=(n_grid,), dim=0)[0]
        d2_pred = torch.gradient(d1_pred, spacing=(n_grid,), dim=0)[0]
        d1_obs = torch.gradient(soh_obs_pad, spacing=(n_grid,), dim=0)[0]
        d2_obs = torch.gradient(d1_obs, spacing=(n_grid,), dim=0)[0]

        # Drop the two boundary rows on each side of every pad transition.
        mask_e = _erode_mask_1d(mask, k=2).to(dtype=traj.dtype)
        curv_denom = mask_e.sum().clamp_min(1.0)
        L_curv = ((d2_pred - d2_obs) ** 2 * mask_e).sum() / curv_denom

        # Knee cycle via masked softmax over -d2 (per column).
        neg_inf = torch.tensor(float("-inf"), dtype=traj.dtype, device=traj.device)
        logits_pred = torch.where(mask, -d2_pred * 50.0, neg_inf)
        logits_obs = torch.where(mask, -d2_obs * 50.0, neg_inf)
        w_pred = torch.softmax(logits_pred, dim=0)
        w_obs = torch.softmax(logits_obs, dim=0)
        n_col = n_grid.view(T_max, 1)
        knee_pred = (w_pred * n_col).sum(dim=0)  # (B,)
        knee_obs = (w_obs * n_col).sum(dim=0)    # (B,)
        n_span = float(n_grid[-1] - n_grid[0] + 1e-6)
        L_knee = (((knee_pred - knee_obs) / n_span) ** 2).mean()

        L_shape = weights.shape_curvature * L_curv + weights.shape_knee * L_knee
    else:
        L_shape = torch.zeros((), dtype=traj.dtype, device=traj.device)

    total = (weights.data * L_data
             + weights.physics * L_phys
             + weights.monotonicity * L_mono
             + weights.shape * L_shape)
    return {
        "total": total,
        "data": L_data,
        "physics": L_phys,
        "monotonicity": L_mono,
        "shape": L_shape,
    }


# ---------------------------------------------------------------------------
# Operator factory
# ---------------------------------------------------------------------------
def _new_operator(hidden: int = 64,
                  n_layers: int = 3,
                  dropout_p: float = 0.1,
                  eol_threshold: float = 0.80,
                  ) -> RULPredictor:
    """Instantiate a theta-aware RULPredictor with branch dim = 11."""
    m = RULPredictor(
        health_dim=BRANCH_DIM,
        hidden=hidden,
        n_layers=n_layers,
        dropout_p=dropout_p,
        eol_threshold=eol_threshold,
    )
    # Identity normalisation until refreshed by the training loop.
    m.set_normalisation(torch.zeros(BRANCH_DIM), torch.ones(BRANCH_DIM))
    return m


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------
_CORPUS_COLUMNS = {"sample_id", "anchor_id", "cycle_n", "SOH"}
# Bug fix (2026-07-10 adversarial audit): the features-produced
# _dataset.parquet stores one row per SAMPLE with soh_traj as a padded list,
# while a raw corpus (from phase3_corpus.py's checkpoint parquets) stores
# one row per (sample, cycle) with a scalar SOH column. Accept either layout
# so downstream code doesn't care which artefact it points at.
_FEATURES_DATASET_COLUMNS = {"sample_id", "anchor_id", "soh_traj", "n_cycles"}


def _load_corpus_parquet(parquet_path: Path) -> list[Phase3Sample]:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    cols = set(df.columns)

    if _FEATURES_DATASET_COLUMNS.issubset(cols):
        # Features-produced _dataset.parquet layout: one row per sample.
        return _load_features_dataset(df)
    if _CORPUS_COLUMNS.issubset(cols):
        # Raw corpus layout: one row per (sample, cycle).
        return _load_raw_corpus(df)
    missing_feat = sorted(_FEATURES_DATASET_COLUMNS - cols)
    missing_raw  = sorted(_CORPUS_COLUMNS - cols)
    raise ValueError(
        f"parquet at {parquet_path} matches neither expected layout — "
        f"features-dataset missing {missing_feat}; raw-corpus missing {missing_raw}"
    )


def _load_features_dataset(df) -> list[Phase3Sample]:
    """Features-produced _dataset.parquet: one row per (anchor, sample)."""
    theta_cols = [f"theta_norm_{k}" for k in THETA_KEYS]
    have_flat_theta = all(c in df.columns for c in theta_cols)
    have_list_theta = "theta_norm" in df.columns

    samples: list[Phase3Sample] = []
    for _, r in df.iterrows():
        soh_traj = np.asarray(r["soh_traj"], dtype=np.float32)
        n_cy = int(r["n_cycles"])
        soh_traj = soh_traj[:n_cy]
        soh_traj = soh_traj[np.isfinite(soh_traj)]
        if soh_traj.size < 5:
            continue
        n_grid = np.arange(soh_traj.size, dtype=np.float32)

        x_raw = r.get("x_health", None)
        if x_raw is not None:
            x_arr = np.asarray(list(x_raw), dtype=np.float32)
            x_arr = np.where(np.isfinite(x_arr), x_arr, 0.0)
        else:
            x_arr = np.zeros(N_XHEALTH, dtype=np.float32)

        if have_flat_theta:
            th = np.array([float(r[c]) for c in theta_cols], dtype=np.float32)
        elif have_list_theta:
            th = np.asarray(list(r["theta_norm"]), dtype=np.float32)
        else:
            th = np.zeros(N_THETA, dtype=np.float32)

        samples.append(Phase3Sample(
            sample_id=str(r["sample_id"]),
            anchor_id=str(r["anchor_id"]),
            cell_id=None,
            n_traj=torch.from_numpy(n_grid),
            soh_traj=torch.from_numpy(soh_traj),
            x_health=torch.from_numpy(x_arr),
            theta_norm=torch.from_numpy(th),
        ))
    return samples


def _load_raw_corpus(df) -> list[Phase3Sample]:
    """Raw corpus layout: one row per (sample, cycle)."""
    theta_cols = [f"theta_norm_{k}" for k in THETA_KEYS]
    have_flat_theta = all(c in df.columns for c in theta_cols)
    # Fallback: raw θ columns as written by phase3_corpus.py
    raw_theta_cols = [
        "theta_k_SEI", "theta_V_SEI", "theta_D_SEI_solvent",
        "theta_k_plating", "theta_k_LAM_negative",
    ]
    have_raw_theta = all(c in df.columns for c in raw_theta_cols[:5])

    samples: list[Phase3Sample] = []
    for sid, g in df.groupby("sample_id"):
        g = g.sort_values("cycle_n").reset_index(drop=True)
        if len(g) < 5:
            continue
        first = g.iloc[0]
        x_health = np.array([
            float(first.get("temperature_C", 25.0)),
            float(first.get("c_rate", 0.5)),
            float(first.get("dcir_mOhm", np.nan)),
            0.0, 1.0,
        ], dtype=np.float32)
        x_health = np.where(np.isfinite(x_health), x_health, 0.0).astype(np.float32)

        if have_flat_theta:
            th = np.array([float(first[c]) for c in theta_cols], dtype=np.float32)
        elif have_raw_theta:
            # Raw physical θ present but not normalised — pass as-is; the
            # trainer's dataset-level normaliser handles it. Pad to 6-D
            # with zero for LAM_pos (not written by phase3_corpus).
            raw = np.array([float(first[c]) for c in raw_theta_cols[:5]],
                            dtype=np.float32)
            th = np.concatenate([raw, np.zeros(1, dtype=np.float32)])
        else:
            th = np.zeros(N_THETA, dtype=np.float32)

        samples.append(Phase3Sample(
            sample_id=str(sid),
            anchor_id=str(first.get("anchor_id", sid)),
            cell_id=str(first["cell_id"]) if "cell_id" in g.columns else None,
            n_traj=torch.tensor(g["cycle_n"].to_numpy(dtype=np.float32)),
            soh_traj=torch.tensor(g["SOH"].to_numpy(dtype=np.float32)),
            x_health=torch.from_numpy(x_health),
            theta_norm=torch.from_numpy(th),
        ))
    return samples


# ---------------------------------------------------------------------------
# Public: training loop
# ---------------------------------------------------------------------------
def _refresh_normalisation(model: RULPredictor,
                           samples: Sequence[Phase3Sample]) -> None:
    if not samples:
        return
    feats = torch.stack([s.branch for s in samples], dim=0)
    mean = feats.mean(dim=0)
    std = feats.std(dim=0)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    model.set_normalisation(mean, std)


class PerAnchorBatchSampler:
    """Batch sampler that yields mini-batches drawn from ONE anchor at a time.

    Rationale (R3 mitigation)
    -------------------------
    Across the pooled Phase-3 corpus the theta parameters exhibit
    ``|rho(k_SEI, k_LAM_neg)| = 0.185`` — enough for the operator to latch onto
    a spurious between-anchor sampling artefact. Within a single anchor's
    70-sim pool the same correlation drops to <= 0.07. Constraining every
    optimiser step to samples that share an ``anchor_id`` collapses the
    between-anchor confound while still letting the model see all seven
    anchors over the course of an epoch.

    Iteration protocol
    ------------------
    ``__iter__`` yields ``list[int]`` batches of indices into ``samples``.
    Every list is homogeneous in ``anchor_id`` and is at most ``batch_size``
    long. Anchor visitation order and within-anchor sample order are both
    re-shuffled on each ``__iter__`` call when ``shuffle=True``. An anchor
    whose pool is smaller than ``batch_size`` still emits a single (undersized)
    batch — no anchor is silently dropped unless ``drop_last=True``.

    Compatibility
    -------------
    Follows the ``torch.utils.data.BatchSampler`` protocol (yields lists of
    indices, exposes ``__len__``) so it can drive a ``DataLoader`` via
    ``batch_sampler=...`` — or be iterated directly by the Phase-3 trainer,
    which already consumes ``Phase3Sample`` objects one-by-one.
    """

    def __init__(self,
                 samples: Sequence[Phase3Sample],
                 batch_size: int,
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 drop_last: bool = False):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self._base_seed = seed
        self._epoch = 0

        by_anchor: dict[str, list[int]] = {}
        for i, s in enumerate(samples):
            anchor = str(getattr(s, "anchor_id"))
            by_anchor.setdefault(anchor, []).append(i)
        # Preserve insertion order so a fixed dataset yields a fixed anchor list
        # (shuffle only touches order, not membership).
        self._by_anchor: dict[str, list[int]] = by_anchor
        self._anchors: list[str] = list(by_anchor.keys())

    def set_epoch(self, epoch: int) -> None:
        """Advance the epoch counter; when a ``seed`` was supplied at
        construction time this makes the per-epoch shuffle reproducible."""
        self._epoch = int(epoch)

    def _rng(self) -> np.random.Generator:
        if self._base_seed is None:
            return np.random.default_rng()
        # Deterministic per-epoch stream when seeded.
        return np.random.default_rng(self._base_seed + self._epoch)

    def __iter__(self):
        rng = self._rng()
        anchor_order = list(self._anchors)
        if self.shuffle:
            rng.shuffle(anchor_order)

        for anchor in anchor_order:
            indices = list(self._by_anchor[anchor])
            if self.shuffle:
                rng.shuffle(indices)
            n = len(indices)
            for start in range(0, n, self.batch_size):
                batch = indices[start:start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch

        # Auto-advance for the next unseeded iteration; safe under a set_epoch
        # override because __iter__ recomputes the RNG from _epoch each call.
        self._epoch += 1

    def __len__(self) -> int:
        total = 0
        for idxs in self._by_anchor.values():
            n = len(idxs)
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total

    @property
    def anchors(self) -> list[str]:
        """Anchor ids seen in the wrapped sample list, in insertion order."""
        return list(self._anchors)


def _stratified_split(samples: list[Phase3Sample],
                      val_pct: float,
                      test_pct: float,
                      seed: int,
                      ) -> tuple[list[Phase3Sample], list[Phase3Sample], list[Phase3Sample]]:
    rng = np.random.default_rng(seed)
    by_anchor: dict[str, list[Phase3Sample]] = {}
    for s in samples:
        by_anchor.setdefault(s.anchor_id, []).append(s)
    train: list[Phase3Sample] = []
    val: list[Phase3Sample] = []
    test: list[Phase3Sample] = []
    for anchor, group in by_anchor.items():
        idx = np.arange(len(group))
        rng.shuffle(idx)
        n = len(group)
        n_val = max(1, int(round(n * val_pct))) if n >= 3 else 0
        n_test = max(1, int(round(n * test_pct))) if n >= 3 else 0
        n_train = max(1, n - n_val - n_test)
        take = lambda a, b: [group[i] for i in idx[a:b]]
        train += take(0, n_train)
        val += take(n_train, n_train + n_val)
        test += take(n_train + n_val, n_train + n_val + n_test)
    return train, val, test


def _save_curve_plot(history: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, lbl in [("train_total", "train"),
                     ("val_total", "val"),
                     ("val_shape", "val_shape")]:
        ys = [h.get(key, np.nan) for h in history]
        ax.plot(epochs, ys, label=lbl)
    ax.set(xlabel="epoch", ylabel="loss", yscale="log",
           title="Phase-3 operator training curves")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train_operator(dataset_parquet_path: str | Path,
                   config_path: str | Path,
                   checkpoint_dir: str | Path = "outputs/models",
                   results_dir: str | Path = "outputs/results",
                   checkpoint_name: str = "pinn_phase3_operator.pt",
                   ) -> tuple[Path, Path]:
    """Train the theta-aware operator on the Phase-3 perturbation corpus.

    Returns
    -------
    (checkpoint_path, training_curves_png_path)
    """
    parquet_path = Path(dataset_parquet_path)
    cfg = yaml.safe_load(Path(config_path).read_text())

    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
    split_cfg = cfg["splits"]

    weights = Phase3LossWeights(
        data=float(loss_cfg["data_weight"]),
        physics=float(loss_cfg["physics_weight"]),
        monotonicity=float(loss_cfg["monotonicity_weight"]),
        shape=float(loss_cfg["shape_weight"]),
    )
    seed = int(train_cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    samples = _load_corpus_parquet(parquet_path)
    if not samples:
        raise RuntimeError(f"No usable samples in {parquet_path}")
    train, val, _test = _stratified_split(
        samples,
        val_pct=float(split_cfg["val_pct"]) / 100.0,
        test_pct=float(split_cfg["test_pct"]) / 100.0,
        seed=seed,
    )
    model = _new_operator()
    _refresh_normalisation(model, samples)

    optim = torch.optim.Adam(model.parameters(), lr=float(train_cfg["lr_max"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=int(train_cfg["epochs"]),
        eta_min=float(train_cfg["lr_min"]),
    )

    batch_size = int(train_cfg["batch_size"])
    train_sampler = PerAnchorBatchSampler(
        samples=train, batch_size=batch_size,
        shuffle=True, seed=seed, drop_last=False,
    )
    val_sampler = PerAnchorBatchSampler(
        samples=val, batch_size=batch_size,
        shuffle=False, seed=seed, drop_last=False,
    )
    print(f"[phase3] using PerAnchorBatchSampler, batch_size={batch_size}",
          flush=True)

    history: list[dict] = []
    best_val = float("inf")
    best_state: Optional[dict] = None
    bad_epochs = 0

    for ep in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        train_sampler.set_epoch(ep)
        train_sums = {"total": 0.0, "shape": 0.0}
        train_count = 0
        for batch_indices in train_sampler:
            batch = [train[i] for i in batch_indices]
            optim.zero_grad()
            parts = phase3_trajectory_loss_batched(model, batch, weights)
            parts["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(train_cfg["grad_clip_norm"])
            )
            optim.step()
            bs = len(batch)
            train_sums["total"] += float(parts["total"].detach()) * bs
            train_sums["shape"] += float(parts["shape"].detach()) * bs
            train_count += bs
        sched.step()

        model.eval()
        val_sampler.set_epoch(ep)
        val_sums = {"total": 0.0, "shape": 0.0}
        val_count = 0
        with torch.no_grad():
            for batch_indices in val_sampler:
                batch = [val[i] for i in batch_indices]
                parts = phase3_trajectory_loss_batched(model, batch, weights)
                bs = len(batch)
                val_sums["total"] += float(parts["total"]) * bs
                val_sums["shape"] += float(parts["shape"]) * bs
                val_count += bs

        row = {
            "epoch": ep,
            "train_total": train_sums["total"] / max(1, train_count),
            "train_shape": train_sums["shape"] / max(1, train_count),
            "val_total": val_sums["total"] / max(1, val_count),
            "val_shape": val_sums["shape"] / max(1, val_count),
        }
        history.append(row)

        if row["val_total"] < best_val:
            best_val = row["val_total"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            marker = "*"
        else:
            bad_epochs += 1
            marker = " "
        print(f"[phase3] epoch {ep:3d}/{int(train_cfg['epochs'])}  "
              f"train_total={row['train_total']:.6f}  "
              f"val_total={row['val_total']:.6f}  "
              f"best_val={best_val:.6f} {marker}  "
              f"bad_epochs={bad_epochs}", flush=True)
        if bad_epochs >= int(train_cfg["patience"]):
            print(f"[phase3] early stop at epoch {ep} "
                  f"(patience {int(train_cfg['patience'])} exceeded)", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_dir = Path(checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / checkpoint_name
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "history": history,
        "best_val_total": best_val,
        "phase": "phase3_operator",
        "branch_dim": BRANCH_DIM,
        "theta_keys": list(THETA_KEYS),
    }, ckpt_path)

    curves_path = Path(results_dir) / "phase3_training_curves.png"
    _save_curve_plot(history, curves_path)
    return ckpt_path, curves_path


# ---------------------------------------------------------------------------
# Operator loading + forward helper
# ---------------------------------------------------------------------------
def load_operator(checkpoint_path: str | Path) -> RULPredictor:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    branch_dim = int(ckpt.get("branch_dim", BRANCH_DIM))
    if branch_dim != BRANCH_DIM:
        raise ValueError(f"checkpoint branch_dim={branch_dim} != {BRANCH_DIM}")
    model = _new_operator()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _forward_soh(model: RULPredictor,
                 sample: Phase3Sample,
                 branch_override: Optional[torch.Tensor] = None,
                 ) -> torch.Tensor:
    """Forward SoH trajectory on a sample; optional branch override for
    ablations that inject synthetic theta."""
    branch = branch_override if branch_override is not None else sample.branch
    soh_0 = sample.soh_traj[0].view(1, 1)
    return model(soh_0, sample.n_traj, branch.unsqueeze(0)).squeeze(-1).squeeze(-1)


# ---------------------------------------------------------------------------
# Per-cell metrics
# ---------------------------------------------------------------------------
def _knee_cycle_hard(n: np.ndarray, soh: np.ndarray, soh_knee: float = 0.90
                      ) -> float:
    """Cycle at which SoH first crosses ``soh_knee``. Falls back to NaN if
    the trajectory never crosses. Used for reporting (not backprop)."""
    below = np.where(soh < soh_knee)[0]
    if below.size == 0:
        return float("nan")
    i = int(below[0])
    if i == 0:
        return float(n[0])
    # Linear interpolation between i-1 and i
    n1, n2 = float(n[i - 1]), float(n[i])
    s1, s2 = float(soh[i - 1]), float(soh[i])
    if s1 == s2:
        return n2
    frac = (s1 - soh_knee) / (s1 - s2)
    return n1 + frac * (n2 - n1)


def per_cell_metrics(model: RULPredictor,
                     sample: Phase3Sample,
                     rpt_interval_cycles: float = 100.0,
                     ) -> dict[str, float]:
    """SoH RMSE (percentage points) + knee-cycle MAE (RPT intervals)."""
    with torch.no_grad():
        pred = _forward_soh(model, sample).cpu().numpy()
    obs = sample.soh_traj.cpu().numpy()
    n = sample.n_traj.cpu().numpy()

    rmse_pp = float(np.sqrt(np.mean((pred - obs) ** 2)) * 100.0)

    kn_pred = _knee_cycle_hard(n, pred)
    kn_obs = _knee_cycle_hard(n, obs)
    if math.isnan(kn_pred) or math.isnan(kn_obs):
        knee_mae_intervals = float("nan")
    else:
        knee_mae_intervals = abs(kn_pred - kn_obs) / rpt_interval_cycles

    return {
        "sample_id": sample.sample_id,
        "cell_id": sample.cell_id or sample.sample_id,
        "anchor_id": sample.anchor_id,
        "soh_rmse_pp": rmse_pp,
        "knee_cycle_pred": float(kn_pred) if not math.isnan(kn_pred) else None,
        "knee_cycle_obs": float(kn_obs) if not math.isnan(kn_obs) else None,
        "knee_cycle_mae_rpt_intervals": knee_mae_intervals,
    }


# ---------------------------------------------------------------------------
# R1 gate 1: Fisher-column cosine similarity
# ---------------------------------------------------------------------------
def fisher_column_cosine(model: RULPredictor,
                         sample: Phase3Sample,
                         eps: float = 1e-3,
                         ) -> dict[str, float]:
    """|cos(dSoH/dlog k_SEI, dSoH/dlog LAM_neg)| along the trajectory.

    Both derivatives are per-cycle sensitivity vectors of length T. Because
    theta_norm is already in log-standardised space, a bump in
    ``theta_norm[k_SEI_IDX]`` corresponds to a fraction of a decade in
    log k_SEI. That is the operational definition we use here.
    """
    def _forward_bumped(idx: int, delta: float) -> np.ndarray:
        branch = sample.branch.clone()
        branch[N_HEALTH_FEATURES + idx] += delta
        with torch.no_grad():
            return _forward_soh(model, sample, branch_override=branch).cpu().numpy()

    base = _forward_bumped(K_SEI_IDX, 0.0)
    d_ksei = (_forward_bumped(K_SEI_IDX, +eps) - base) / eps
    d_lam = (_forward_bumped(LAM_NEG_IDX, +eps) - base) / eps

    dot = float(np.dot(d_ksei, d_lam))
    n1 = float(np.linalg.norm(d_ksei))
    n2 = float(np.linalg.norm(d_lam))
    if n1 < 1e-12 or n2 < 1e-12:
        cos = float("nan")
    else:
        cos = dot / (n1 * n2)
    return {
        "sample_id": sample.sample_id,
        "cosine": float(cos),
        "abs_cosine": float(abs(cos)) if not math.isnan(cos) else float("nan"),
        "norm_dSoH_dlog_k_SEI": n1,
        "norm_dSoH_dlog_LAM_neg": n2,
    }


# ---------------------------------------------------------------------------
# R1 gate 2: Regime-swap replay on CALB_0003
# ---------------------------------------------------------------------------
def regime_swap_replay(model: RULPredictor,
                       sample: Phase3Sample,
                       lam_neg_shift_dec: float = -2.0,
                       soh_window: tuple[float, float] = (0.80, 0.95),
                       ) -> dict[str, float]:
    """Replay ``sample`` (expected: CALB_0003) with (a) joint theta vs
    (b) SEI-only theta (LAM_neg driven ``lam_neg_shift_dec`` decades below
    its identified value). Reports max |Delta SoH| over the SoH window.
    """
    with torch.no_grad():
        joint = _forward_soh(model, sample).cpu().numpy()

    branch_sei_only = sample.branch.clone()
    # theta_norm is standardised in log10 space for LAM_neg; a decade shift
    # translates to (dec / std) standardised units.
    std_lam = float(default_theta_stats()["LAM_neg_rate_s"]["std"])
    branch_sei_only[N_HEALTH_FEATURES + LAM_NEG_IDX] += lam_neg_shift_dec / max(std_lam, 1e-6)
    with torch.no_grad():
        sei_only = _forward_soh(model, sample,
                                branch_override=branch_sei_only).cpu().numpy()

    lo, hi = soh_window
    mask = (joint >= lo) & (joint <= hi)
    if not mask.any():
        max_abs_delta = float("nan")
        cycle_of_max = None
    else:
        delta = np.abs(joint - sei_only)[mask]
        i_local = int(np.argmax(delta))
        max_abs_delta = float(delta[i_local])
        cycles = sample.n_traj.cpu().numpy()[mask]
        cycle_of_max = float(cycles[i_local])
    return {
        "sample_id": sample.sample_id,
        "max_abs_delta_soh": max_abs_delta,
        "cycle_of_max": cycle_of_max,
        "window": [lo, hi],
        "lam_neg_shift_dec": lam_neg_shift_dec,
        "passes_1pp_gate": (isinstance(max_abs_delta, float)
                            and not math.isnan(max_abs_delta)
                            and max_abs_delta > 0.01),
    }


# ---------------------------------------------------------------------------
# R1 gate 3: Attribution-slope test
# ---------------------------------------------------------------------------
def attribution_slope_test(model: RULPredictor,
                           sample: Phase3Sample,
                           sigma: float = 1.0,
                           ) -> dict[str, float]:
    """+1 sigma k_SEI vs +1 sigma LAM_neg forward SoH slope comparison.

    Because theta_norm is already z-scored, adding +1.0 in the normalised
    coordinate IS a +1 sigma shift in the physical (log) parameter. We
    compare the ``dSoH/dn`` slope of each perturbed trajectory over the
    portion where both remain in [0.80, 0.98].
    """
    def _perturbed(idx: int) -> np.ndarray:
        branch = sample.branch.clone()
        branch[N_HEALTH_FEATURES + idx] += sigma
        with torch.no_grad():
            return _forward_soh(model, sample, branch_override=branch).cpu().numpy()

    n = sample.n_traj.cpu().numpy()
    base = None
    with torch.no_grad():
        base = _forward_soh(model, sample).cpu().numpy()
    ksei = _perturbed(K_SEI_IDX)
    lam = _perturbed(LAM_NEG_IDX)

    def _slope_and_intercept(y: np.ndarray) -> tuple[float, float]:
        # Least-squares linear fit — slope has physical units 1/cycle.
        A = np.vstack([n, np.ones_like(n)]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(m), float(c)

    slope_ksei, intercept_ksei = _slope_and_intercept(ksei)
    slope_lam, intercept_lam = _slope_and_intercept(lam)
    slope_base, intercept_base = _slope_and_intercept(base)

    return {
        "sample_id": sample.sample_id,
        "sigma": sigma,
        "slope_base": slope_base,
        "slope_+1s_k_SEI": slope_ksei,
        "slope_+1s_LAM_neg": slope_lam,
        "slope_delta_k_SEI": slope_ksei - slope_base,
        "slope_delta_LAM_neg": slope_lam - slope_base,
        "intercept_diff": intercept_ksei - intercept_lam,
        "slope_diff": slope_ksei - slope_lam,
        "distinct_slopes": abs(slope_ksei - slope_lam) > 1e-6,
    }


# ---------------------------------------------------------------------------
# Public: validation suite
# ---------------------------------------------------------------------------
def _load_held_out_samples(corpus_dir: Path,
                           held_out_cells: Sequence[str],
                           ) -> list[Phase3Sample]:
    """Load held-out cells from ``corpus_dir``.

    Accepts one of two layouts:
    * ``<corpus_dir>/held_out/<cell_id>.parquet`` — canonical.
    * ``<corpus_dir>/<cell_id>.parquet``          — flat.

    Silently drops requested cells whose parquet does not exist so the
    caller can catch that in the returned per-cell block.
    """
    samples: list[Phase3Sample] = []
    for cid in held_out_cells:
        for candidate in (
            corpus_dir / "held_out" / f"{cid}.parquet",
            corpus_dir / f"{cid}.parquet",
        ):
            if candidate.exists():
                got = _load_corpus_parquet(candidate)
                for s in got:
                    s.cell_id = cid
                samples.extend(got)
                break
    return samples


def run_validation_suite(checkpoint_path: str | Path,
                         held_out_cells: Sequence[str],
                         corpus_dir: str | Path,
                         *,
                         calb_0003_sample: Optional[Phase3Sample] = None,
                         held_out_samples: Optional[Sequence[Phase3Sample]] = None,
                         model: Optional[RULPredictor] = None,
                         ) -> dict[str, Any]:
    """Held-out validation harness.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to a Phase-3 operator checkpoint. Ignored if ``model`` is set.
    held_out_cells : Sequence[str]
        Canonical cell IDs to load from ``corpus_dir``.
    corpus_dir : str | Path
        Directory containing per-cell parquet trajectories.
    calb_0003_sample : Phase3Sample | None
        Override for the regime-swap replay input (useful in the smoke
        test where the corpus directory does not exist).
    held_out_samples : Sequence[Phase3Sample] | None
        Direct sample override — bypasses ``corpus_dir`` loading.
    model : RULPredictor | None
        Bypass ``checkpoint_path`` loading (used by the smoke test).
    """
    if model is None:
        model = load_operator(checkpoint_path)
    else:
        model = model
        model.eval()

    if held_out_samples is None:
        samples = _load_held_out_samples(Path(corpus_dir), held_out_cells)
    else:
        samples = list(held_out_samples)

    per_cell = [per_cell_metrics(model, s) for s in samples]
    fisher = [fisher_column_cosine(model, s) for s in samples]

    if calb_0003_sample is None:
        calb_0003_sample = next(
            (s for s in samples if (s.cell_id or "").upper() == "CALB_0003"),
            None,
        )
    regime = (regime_swap_replay(model, calb_0003_sample)
              if calb_0003_sample is not None else None)

    attribution = [attribution_slope_test(model, s) for s in samples]

    def _mean(vs: list[float]) -> Optional[float]:
        vs = [v for v in vs if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return float(np.mean(vs)) if vs else None

    summary = {
        "n_held_out_samples": len(samples),
        "mean_soh_rmse_pp": _mean([r["soh_rmse_pp"] for r in per_cell]),
        "mean_knee_cycle_mae_rpt_intervals": _mean([r["knee_cycle_mae_rpt_intervals"] for r in per_cell]),
        "mean_abs_fisher_cosine": _mean([r["abs_cosine"] for r in fisher]),
        "regime_swap_max_abs_delta": (regime["max_abs_delta_soh"] if regime else None),
        "attribution_mean_slope_diff": _mean([a["slope_diff"] for a in attribution]),
    }

    # Pass/fail gates (design doc section 5.4)
    gates = {
        "soh_rmse_pp<=3.0": (summary["mean_soh_rmse_pp"] is not None
                             and summary["mean_soh_rmse_pp"] <= 3.0),
        "knee_mae<=1_rpt_interval": (summary["mean_knee_cycle_mae_rpt_intervals"] is not None
                                     and summary["mean_knee_cycle_mae_rpt_intervals"] <= 1.0),
        "fisher_cosine<0.3": (summary["mean_abs_fisher_cosine"] is not None
                              and summary["mean_abs_fisher_cosine"] < 0.3),
        "regime_swap>1pp": (regime is not None and regime.get("passes_1pp_gate", False)),
        "attribution_distinct_slopes": all(a["distinct_slopes"] for a in attribution) if attribution else False,
    }
    gates["overall_pass"] = all(gates.values())

    return {
        "meta": {
            "checkpoint": str(checkpoint_path),
            "held_out_cells": list(held_out_cells),
            "corpus_dir": str(corpus_dir),
            "n_theta": N_THETA,
            "branch_dim": BRANCH_DIM,
        },
        "per_cell": per_cell,
        "fisher_cosine": fisher,
        "regime_swap_calb_0003": regime,
        "attribution_slope": attribution,
        "summary": summary,
        "gates": gates,
    }


# ---------------------------------------------------------------------------
# Public: report writer
# ---------------------------------------------------------------------------
def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        return f"{v:.{digits}g}"
    if isinstance(v, bool):
        return "YES" if v else "NO"
    return str(v)


def save_validation_report(report: dict[str, Any], path: str | Path) -> None:
    """Write both a Markdown report and a JSON dump next to it.

    ``path`` is treated as the Markdown target. The JSON companion has the
    same stem with a `.json` extension.
    """
    md_path = Path(path)
    js_path = md_path.with_suffix(".json")
    md_path.parent.mkdir(parents=True, exist_ok=True)

    def _json_default(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"unserialisable: {type(o)}")

    js_path.write_text(json.dumps(report, indent=2, default=_json_default))

    lines: list[str] = []
    meta = report.get("meta", {})
    summary = report.get("summary", {})
    gates = report.get("gates", {})
    lines.append("# Phase 3 held-out validation report")
    lines.append("")
    lines.append(f"- Checkpoint: `{meta.get('checkpoint')}`")
    lines.append(f"- Corpus dir: `{meta.get('corpus_dir')}`")
    lines.append(f"- Held-out cells: {', '.join(meta.get('held_out_cells', []))}")
    lines.append(f"- Branch dim: {meta.get('branch_dim')} "
                 f"(5 x_health + {meta.get('n_theta')} theta_norm)")
    lines.append("")
    lines.append("## Summary")
    for k, v in summary.items():
        lines.append(f"- **{k}**: {_fmt(v)}")
    lines.append("")
    lines.append("## Pass / fail gates (design §5.4)")
    for k, v in gates.items():
        lines.append(f"- {k}: {_fmt(v)}")
    lines.append("")

    if report.get("per_cell"):
        lines.append("## Per-cell metrics")
        lines.append("| cell_id | anchor_id | SoH RMSE (pp) | knee MAE (RPT int.) |")
        lines.append("|---|---|---|---|")
        for r in report["per_cell"]:
            lines.append(f"| {r['cell_id']} | {r['anchor_id']} | "
                         f"{_fmt(r['soh_rmse_pp'])} | "
                         f"{_fmt(r['knee_cycle_mae_rpt_intervals'])} |")
        lines.append("")
    if report.get("fisher_cosine"):
        lines.append("## Fisher-column cosine (target |cos| < 0.3)")
        lines.append("| sample_id | |cos| | ||dSoH/dlog k_SEI|| | ||dSoH/dlog LAM_neg|| |")
        lines.append("|---|---|---|---|")
        for r in report["fisher_cosine"]:
            lines.append(f"| {r['sample_id']} | {_fmt(r['abs_cosine'])} | "
                         f"{_fmt(r['norm_dSoH_dlog_k_SEI'])} | "
                         f"{_fmt(r['norm_dSoH_dlog_LAM_neg'])} |")
        lines.append("")
    if report.get("regime_swap_calb_0003"):
        r = report["regime_swap_calb_0003"]
        lines.append("## Regime-swap replay on CALB_0003")
        lines.append(f"- max |Delta SoH| in window {r['window']}: {_fmt(r['max_abs_delta_soh'])}")
        lines.append(f"- cycle of max: {_fmt(r['cycle_of_max'])}")
        lines.append(f"- passes 1 pp gate: {_fmt(r['passes_1pp_gate'])}")
        lines.append("")
    if report.get("attribution_slope"):
        lines.append("## Attribution-slope test (+1 sigma k_SEI vs +1 sigma LAM_neg)")
        lines.append("| sample_id | slope_base | slope_k_SEI | slope_LAM_neg | slope_diff |")
        lines.append("|---|---|---|---|---|")
        for r in report["attribution_slope"]:
            lines.append(f"| {r['sample_id']} | "
                         f"{_fmt(r['slope_base'])} | "
                         f"{_fmt(r['slope_+1s_k_SEI'])} | "
                         f"{_fmt(r['slope_+1s_LAM_neg'])} | "
                         f"{_fmt(r['slope_diff'])} |")
        lines.append("")

    md_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _make_dummy_sample(sample_id: str,
                       anchor_id: str,
                       cell_id: Optional[str] = None,
                       n_cycles: int = 120,
                       fade_rate: float = 3e-4,
                       curvature: float = 8e-8,
                       seed: int = 0,
                       ) -> Phase3Sample:
    rng = np.random.default_rng(seed)
    n = np.arange(1, n_cycles + 1, dtype=np.float32)
    soh = np.clip(1.0 - fade_rate * n - curvature * n ** 2
                  + rng.normal(0.0, 5e-4, size=n_cycles), 0.55, 1.0).astype(np.float32)
    x_health = np.array([25.0, 0.5, 1.74, 0.0, 1.0], dtype=np.float32)
    theta_norm = rng.normal(0.0, 0.5, size=N_THETA).astype(np.float32)
    return Phase3Sample(
        sample_id=sample_id,
        anchor_id=anchor_id,
        cell_id=cell_id,
        n_traj=torch.tensor(n),
        soh_traj=torch.tensor(soh),
        x_health=torch.tensor(x_health),
        theta_norm=torch.tensor(theta_norm),
    )


def _smoke() -> int:
    torch.manual_seed(0)
    print("[phase3_train_val] smoke test")

    # 1. Operator with random weights.
    # RULPredictor zero-initialises the final layer (so an untrained model
    # produces constant-rate decay regardless of branch input); nudge those
    # weights so theta perturbations propagate and R1 diagnostics have
    # numerically meaningful outputs.
    model = _new_operator()
    with torch.no_grad():
        model.ode.net[-1].weight.normal_(0.0, 0.1)
        model.ode.net[-1].bias.fill_(-3.0)
    print(f"  operator branch_dim={BRANCH_DIM}, params={model.n_parameters():,}")

    # 2. Four dummy samples — one is CALB_0003 for the regime-swap replay
    samples = [
        _make_dummy_sample("smoke_calb_0003_0", "CALB_0003",
                           cell_id="CALB_0003", seed=1),
        _make_dummy_sample("smoke_eve_0002_0", "EVE_0004",
                           cell_id="EVE_0002", seed=2,
                           fade_rate=4e-4, curvature=1.2e-7),
        _make_dummy_sample("smoke_rept_holdback_0", "REPT_0007",
                           cell_id="REPT_0007", seed=3,
                           fade_rate=6e-4, curvature=1.8e-7),
        _make_dummy_sample("smoke_calb_0008_0", "CALB_0015",
                           cell_id="CALB_0008", seed=4,
                           fade_rate=2e-4, curvature=5e-8),
    ]

    # Sanity-check the loss composition (single forward+backward pass)
    weights = Phase3LossWeights()
    parts = phase3_trajectory_loss(model, samples[0], weights)
    parts["total"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).item()
    parts_f = {k: float(v.detach()) for k, v in parts.items()}
    model.zero_grad()
    print(f"  shape-aware loss OK: total={parts_f['total']:.4e}, "
          f"data={parts_f['data']:.4e}, physics={parts_f['physics']:.4e}, "
          f"mono={parts_f['monotonicity']:.4e}, shape={parts_f['shape']:.4e}, "
          f"grad_norm={grad_norm:.3g}")

    # 3. Run each validation function
    per_cell = [per_cell_metrics(model, s) for s in samples]
    print(f"  per_cell_metrics OK on {len(per_cell)} samples "
          f"(mean RMSE pp = {np.mean([r['soh_rmse_pp'] for r in per_cell]):.2f})")

    fisher = [fisher_column_cosine(model, s) for s in samples]
    print(f"  fisher_column_cosine OK "
          f"(mean |cos| = {np.mean([r['abs_cosine'] for r in fisher]):.3f})")

    regime = regime_swap_replay(model, samples[0])
    print(f"  regime_swap_replay OK (max |Delta SoH| = {regime['max_abs_delta_soh']:.4f})")

    attribution = [attribution_slope_test(model, s) for s in samples]
    print(f"  attribution_slope_test OK "
          f"(mean slope_diff = {np.mean([a['slope_diff'] for a in attribution]):.3e})")

    # Full validation suite (bypass checkpoint / corpus by passing model + samples)
    report = run_validation_suite(
        checkpoint_path="<smoke:none>",
        held_out_cells=["CALB_0003", "EVE_0002", "REPT_0007", "CALB_0008"],
        corpus_dir="<smoke:none>",
        model=model,
        held_out_samples=samples,
        calb_0003_sample=samples[0],
    )
    print("\n[phase3_train_val] REPORT STRUCTURE")
    print(f"  top-level keys: {sorted(report.keys())}")
    print(f"  meta keys:      {sorted(report['meta'].keys())}")
    print(f"  summary:        {json.dumps(report['summary'], indent=2, default=float)}")
    print(f"  gates:          {json.dumps(report['gates'], indent=2, default=str)}")
    print(f"  per_cell rows:  {len(report['per_cell'])}")
    print(f"  fisher rows:    {len(report['fisher_cosine'])}")
    print(f"  attribution rows: {len(report['attribution_slope'])}")
    print(f"  regime_swap keys: {sorted(report['regime_swap_calb_0003'].keys())}")

    # Round-trip through save_validation_report
    out = Path("/tmp/phase3_smoke_report.md")
    save_validation_report(report, out)
    js = out.with_suffix(".json")
    print(f"\n  wrote report: {out} ({out.stat().st_size} B)")
    print(f"  wrote json:   {js} ({js.stat().st_size} B)")
    return 0


def _per_anchor_sampler_smoke() -> int:
    """Load the 7 anchor parquets, build PerAnchorBatchSampler(batch_size=16),
    iterate 5 batches, and assert each batch is homogeneous in anchor_id."""
    corpus_dir = Path("/home/hj/Desktop/PINNs/configs/phase3_corpus")
    parquets = sorted(p for p in corpus_dir.glob("*.parquet")
                      if not p.name.startswith("_"))
    print(f"[per_anchor_sampler_smoke] found {len(parquets)} anchor parquets "
          f"under {corpus_dir}")
    if not parquets:
        print("  FAIL: no anchor parquets found")
        return 1

    samples: list[Phase3Sample] = []
    for p in parquets:
        got = _load_corpus_parquet(p)
        print(f"  {p.name}: {len(got)} samples")
        samples.extend(got)
    print(f"  total samples: {len(samples)}")

    sampler = PerAnchorBatchSampler(samples, batch_size=16, shuffle=True, seed=456)
    print(f"  sampler: len={len(sampler)} batches, anchors={sampler.anchors}")

    n_check = 5
    it = iter(sampler)
    for i in range(n_check):
        try:
            batch = next(it)
        except StopIteration:
            print(f"  FAIL: sampler exhausted at batch {i} (< {n_check})")
            return 1
        anchor_ids = {samples[j].anchor_id for j in batch}
        assert len(anchor_ids) == 1, (
            f"batch {i} straddles anchors: {anchor_ids} "
            f"(indices={batch})"
        )
        print(f"  batch {i}: size={len(batch)} anchor={next(iter(anchor_ids))}")

    print("[per_anchor_sampler_smoke] OK — 5 batches, all homogeneous")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sampler-smoke":
        raise SystemExit(_per_anchor_sampler_smoke())
    raise SystemExit(_smoke())
