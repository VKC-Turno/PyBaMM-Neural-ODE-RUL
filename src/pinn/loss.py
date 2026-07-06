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
    Re-query the ODE network at mfr_bry observed (n_t, SOH_pred_t) point to
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
