"""
src/pinn/loss.py
----------------
Composite loss for the Neural-ODE PINN.

    L = L_data + λ_physics · L_physics + λ_monotonicity · L_monotonicity

where
    L_data         = MSE between predicted SOH(n) and observed SOH(n)
    L_physics      = MSE between dSOH/dn predicted by the ODE network and the
                     finite-difference derivative of the *observed* trajectory.
                     The synthetic data IS the physics here (it came from
                     PyBaMM), so this term anchors the network to the PyBaMM-
                     consistent derivative without requiring a separate
                     analytical SEI / LAM model.
    L_monotonicity = soft penalty on any positive increment of the predicted
                     trajectory; the model already uses -softplus so this
                     should converge to ~0 quickly.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    data: float = 1.0
    physics: float = 0.1
    monotonicity: float = 0.05


@dataclass
class LossBreakdown:
    total: torch.Tensor
    data: torch.Tensor
    physics: torch.Tensor
    monotonicity: torch.Tensor

    def detach_floats(self) -> dict[str, float]:
        return {
            "loss/total": float(self.total.detach().item()),
            "loss/data": float(self.data.detach().item()),
            "loss/physics": float(self.physics.detach().item()),
            "loss/monotonicity": float(self.monotonicity.detach().item()),
        }


def _physics_target(n: torch.Tensor, soh: torch.Tensor) -> torch.Tensor:
    """
    Finite-difference derivative dSOH/dn from the observed trajectory.

    Uses torch.gradient (central differences with one-sided ends). The
    target shape matches the input shape (T,) → (T,).
    """
    return torch.gradient(soh, spacing=(n,))[0]


def _ode_derivative_along_trajectory(model, soh_traj_pred: torch.Tensor,
                                     n_traj: torch.Tensor,
                                     x_health: torch.Tensor) -> torch.Tensor:
    """
    Re-query the ODE network at every observed (n_t, SOH_pred_t) point to
    get the model's instantaneous dSOH/dn there. This is what `L_physics`
    compares against the finite-difference target.
    """
    model.ode.health_context = model.normalise_features(x_health.unsqueeze(0))
    dsoh = []
    for t in range(len(n_traj)):
        s = soh_traj_pred[t].view(1, 1)
        d = model.ode(n_traj[t], s).view(-1)
        dsoh.append(d)
    return torch.cat(dsoh, dim=0)


def trajectory_loss(model, sample, weights: LossWeights) -> LossBreakdown:
    """
    Compute the composite loss for a single sample.

    The model integrates from n_traj[0] to n_traj[-1] using the observed
    initial SOH as the IC. Predictions are then compared point-wise to the
    observed SOH at each n_t.
    """
    n_traj = sample["n_traj"]
    soh_obs = sample["soh_traj"]
    x_health = sample["x_health"]

    soh_0 = soh_obs[0].view(1, 1)
    traj_pred = model(soh_0, n_traj, x_health.unsqueeze(0)).squeeze(-1).squeeze(-1)  # (T,)

    # L_data
    L_data = F.mse_loss(traj_pred, soh_obs)

    # L_physics — compare model dSOH/dn against finite-diff target
    target_dsoh = _physics_target(n_traj, soh_obs)
    pred_dsoh = _ode_derivative_along_trajectory(model, traj_pred.detach(), n_traj, x_health)
    L_phys = F.mse_loss(pred_dsoh, target_dsoh)

    # L_monotonicity — penalise any positive step in predicted trajectory
    diffs = traj_pred[1:] - traj_pred[:-1]
    L_mono = torch.relu(diffs).pow(2).mean() if diffs.numel() > 0 else torch.zeros((), device=diffs.device)

    total = (weights.data * L_data
             + weights.physics * L_phys
             + weights.monotonicity * L_mono)
    return LossBreakdown(total=total, data=L_data, physics=L_phys, monotonicity=L_mono)


def batch_loss(model, batch: dict, weights: LossWeights) -> LossBreakdown:
    """
    Sum trajectory losses over a variable-length batch (no padding — each
    sample's trajectory is integrated independently). Returns mean over
    the batch.
    """
    parts: list[LossBreakdown] = []
    for i in range(len(batch["sample_id"])):
        s = {
            "n_traj": batch["n_traj"][i],
            "soh_traj": batch["soh_traj"][i],
            "x_health": batch["x_health"][i],
        }
        parts.append(trajectory_loss(model, s, weights))
    total = torch.stack([p.total for p in parts]).mean()
    data = torch.stack([p.data for p in parts]).mean()
    phys = torch.stack([p.physics for p in parts]).mean()
    mono = torch.stack([p.monotonicity for p in parts]).mean()
    return LossBreakdown(total=total, data=data, physics=phys, monotonicity=mono)


# ---------------------------------------------------------------------------
# Phase 3 censored / Tobit loss
# ---------------------------------------------------------------------------

BIN_EDGES_CY = torch.tensor(
    [302, 478, 618, 793, 1001, 1134, 1297, 1448, 1643, 1898, 2188],
    dtype=torch.float32,
)
BIN_WEIGHTS = torch.tensor(
    [0.51, 1.08, 1.14, 1.14, 1.14, 1.20, 1.08, 1.14, 1.08, 0.51],
    dtype=torch.float32,
)
CENSORED_W = 1.00
LAMBDA_TOBIT = 1.0
LAMBDA_MONO = 0.3
SOH_EOL = 0.80
HORIZON_CY = 2500


def cycle_bin_weight(tgt_eol_cy: torch.Tensor) -> torch.Tensor:
    edges = BIN_EDGES_CY.to(tgt_eol_cy.device)
    idx = torch.bucketize(tgt_eol_cy, edges).clamp(1, 10) - 1
    return BIN_WEIGHTS.to(tgt_eol_cy.device)[idx]


def phase3_loss(pred_soh_traj: torch.Tensor,
                pred_eol_cy: torch.Tensor,
                tgt_eol_cy: torch.Tensor,
                soh_at_horizon: torch.Tensor,
                is_censored: torch.Tensor) -> torch.Tensor:
    """
    Censored Phase-3 loss.

    Shapes:
      pred_soh_traj  : (B, T)   SoH trajectory from odeint over cycle grid
      pred_eol_cy    : (B,)     predicted cycle where SoH crosses SOH_EOL
      tgt_eol_cy     : (B,)     target EoL cycle (NaN if censored)
      soh_at_horizon : (B,)     predicted SoH at HORIZON_CY
      is_censored    : (B,) bool  True if sim ran full HORIZON_CY without EoL
    """
    reach = ~is_censored
    if reach.any():
        w_reach = cycle_bin_weight(tgt_eol_cy[reach])
        l_reach = (w_reach * F.smooth_l1_loss(
            pred_eol_cy[reach], tgt_eol_cy[reach], reduction="none")).mean()
    else:
        l_reach = torch.zeros((), device=pred_soh_traj.device,
                              dtype=pred_soh_traj.dtype)

    if is_censored.any():
        l_tobit = CENSORED_W * F.relu(SOH_EOL - soh_at_horizon[is_censored]).mean()
    else:
        l_tobit = torch.zeros((), device=pred_soh_traj.device,
                              dtype=pred_soh_traj.dtype)

    dsoh = pred_soh_traj[:, 1:] - pred_soh_traj[:, :-1]
    l_mono = F.relu(dsoh).mean()

    return l_reach + LAMBDA_TOBIT * l_tobit + LAMBDA_MONO * l_mono


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T = 4, 100

    # Dummy differentiable trajectory: decreasing SoH with a learnable scale
    scale = torch.nn.Parameter(torch.tensor(1.0))
    base = torch.linspace(1.0, 0.7, T).unsqueeze(0).expand(B, T)
    pred_soh_traj = 1.0 - scale * (1.0 - base)

    # Half reach EoL, half censored
    is_censored = torch.tensor([False, False, True, True])
    tgt_eol_cy = torch.tensor([800.0, 1500.0, float("nan"), float("nan")])
    pred_eol_cy = scale * torch.tensor([850.0, 1450.0, 0.0, 0.0])
    soh_at_horizon = pred_soh_traj[:, -1]

    loss = phase3_loss(pred_soh_traj, pred_eol_cy, tgt_eol_cy,
                       soh_at_horizon, is_censored)
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    loss.backward()
    assert scale.grad is not None and torch.isfinite(scale.grad), \
        f"grad not finite: {scale.grad}"
    print(f"phase3_loss smoke OK: loss={float(loss):.6f} grad={float(scale.grad):.6f}")
