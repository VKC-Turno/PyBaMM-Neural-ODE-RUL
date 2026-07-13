"""
Voltaris/Data_Exploration/phase3_v7_operator.py
==============================================

v7 encoder-decoder operator: adds a context-cycles input on top of the
v6 theta-conditioned Neural ODE architecture.

Key changes vs v6:

1. Encoder MLP compresses K observed context-cycle deltas into an
   8-dim context vector. The context is DELTA-FROM-CONTEXT-START, not
   raw SoH (see phase3_v7_dataset for why).

2. The Neural ODE branch input concatenates:
       (SoH_now, cycle_norm, x_health[3], theta_norm[6], context[8])
   giving in_dim = 1 + 1 + 3 + 6 + 8 = 19.

3. Integration starts from context_soh_start (the SoH at s = context_start
   in the original trajectory) — NOT from a hardcoded 1.0. This makes the
   operator natively able to forecast from any second-life starting SoH.

4. The softplus-monotonic decoder is preserved from v6:
       dSoH/dn = -softplus(net(...))
   guaranteeing non-increasing forecasts.

Architecture summary:
   Encoder:  MLP(K=50 -> 32 -> 16 -> 8, Tanh, zero-init final layer)
   ODE:      MLP(19 -> 64 -> 64 -> 64 -> 1, Tanh, softplus decoder)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint


X_HEALTH_DIM = 3      # v7: [T, c_rate, DCIR] — no IC peaks
THETA_DIM = 6
CONTEXT_LATENT_DIM = 8


class ContextEncoder(nn.Module):
    """Compresses K observed context-cycle deltas to CONTEXT_LATENT_DIM."""

    def __init__(self, K: int, latent_dim: int = CONTEXT_LATENT_DIM,
                 hidden: tuple[int, int] = (32, 16)):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(K, h1), nn.Tanh(),
            nn.Linear(h1, h2), nn.Tanh(),
            nn.Linear(h2, latent_dim),
        )
        # Zero-init final layer so early training doesn't over-condition.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context_delta: torch.Tensor) -> torch.Tensor:
        # context_delta: (B, K)
        return self.net(context_delta)


class _ODERHS(nn.Module):
    """Right-hand side of dSoH/dn = -softplus(net(state)).

    torchdiffeq only accepts (t, y) signatures, so the branch context is
    stashed on the module before each odeint call (v1/v6 pattern).
    """

    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 3,
                 dropout: float = 0.1, bias_init: float = -5.0):
        super().__init__()
        blocks: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            blocks += [nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.Tanh()]
        blocks += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*blocks)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, bias_init)
        self.branch_context: Optional[torch.Tensor] = None
        self.n_max = 2500.0

    def forward(self, n: torch.Tensor, soh: torch.Tensor) -> torch.Tensor:
        batch = soh.shape[0]
        n_norm = (n / self.n_max).expand(batch, 1)
        ctx = self.branch_context   # (B, in_dim - 2)
        inp = torch.cat([soh, n_norm, ctx], dim=-1)
        raw = self.net(inp)
        return -F.softplus(raw)


class OperatorV7(nn.Module):
    """v7 encoder-decoder operator.

    Public API:
        model = OperatorV7(K=50)
        pred = model(x_health, theta_norm, context_delta, context_soh_start,
                      target_cycles)      # (B, T)
    """

    def __init__(self,
                 K: int = 50,
                 x_health_dim: int = X_HEALTH_DIM,
                 theta_dim: int = THETA_DIM,
                 context_latent: int = CONTEXT_LATENT_DIM,
                 hidden: int = 64,
                 ode_layers: int = 3,
                 dropout: float = 0.1,
                 rtol: float = 1e-4,
                 atol: float = 1e-6):
        super().__init__()
        self.K = int(K)
        self.encoder = ContextEncoder(K=self.K, latent_dim=context_latent)
        in_dim = 1 + 1 + x_health_dim + theta_dim + context_latent
        self.rhs = _ODERHS(in_dim=in_dim, hidden=hidden, layers=ode_layers,
                            dropout=dropout)
        self.rtol = rtol
        self.atol = atol

        # Normalisation buffers (populated by set_*_normalisation before
        # training) — used at inference to standardise raw x_health / θ.
        self.register_buffer("xh_mean", torch.zeros(x_health_dim))
        self.register_buffer("xh_std",  torch.ones(x_health_dim))
        self.register_buffer("th_mean", torch.zeros(theta_dim))
        self.register_buffer("th_std",  torch.ones(theta_dim))

    def set_x_health_stats(self, mean: torch.Tensor, std: torch.Tensor):
        self.xh_mean.copy_(mean)
        self.xh_std.copy_(std)

    def set_theta_stats(self, mean: torch.Tensor, std: torch.Tensor):
        self.th_mean.copy_(mean)
        self.th_std.copy_(std)

    def _normalise_xh(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.xh_mean) / (self.xh_std + 1e-8)

    def _normalise_theta(self, t: torch.Tensor) -> torch.Tensor:
        # θ arrives pre-normalised from the v6 extractor (log-space z-scores),
        # so this is an identity for the training path. Kept for symmetry
        # when a caller feeds raw physical θ at inference.
        return (t - self.th_mean) / (self.th_std + 1e-8)

    def forward(self,
                x_health: torch.Tensor,       # (B, x_health_dim)
                theta_norm: torch.Tensor,      # (B, theta_dim), pre-normalised
                context_delta: torch.Tensor,   # (B, K)
                context_soh_start: torch.Tensor,   # (B,) scalar per sample
                target_cycles: torch.Tensor,   # (T,) — monotone int/float
                ) -> torch.Tensor:
        """
        Returns forecast SoH trajectory over target_cycles.
        Shape: (B, T).
        """
        # Build the branch context: normalise x_health, keep theta as-is,
        # append encoder output.
        xh_n = self._normalise_xh(x_health)
        ctx_latent = self.encoder(context_delta)          # (B, latent)
        branch = torch.cat([xh_n, theta_norm, ctx_latent], dim=-1)  # (B, 17)
        self.rhs.branch_context = branch

        # Integrate FROM the context_soh_start (per-sample scalar).
        soh0 = context_soh_start.view(-1, 1).to(x_health.dtype)
        traj = odeint(self.rhs, soh0, target_cycles.to(x_health.dtype),
                       method="dopri5",
                       rtol=self.rtol, atol=self.atol,
                       options={"max_num_steps": 1000})
        # traj: (T, B, 1) -> (B, T)
        return traj.squeeze(-1).transpose(0, 1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@dataclass
class OperatorV7Config:
    K: int = 50
    x_health_dim: int = X_HEALTH_DIM
    theta_dim: int = THETA_DIM
    context_latent: int = CONTEXT_LATENT_DIM
    hidden: int = 64
    ode_layers: int = 3
    dropout: float = 0.1


if __name__ == "__main__":
    # Smoke: dummy forward + backward. Confirms shapes and grad flow.
    torch.manual_seed(0)
    model = OperatorV7(K=50)
    B, K, T = 4, 50, 100
    x_health = torch.zeros(B, X_HEALTH_DIM)
    theta = torch.zeros(B, THETA_DIM)
    ctx_delta = torch.zeros(B, K)
    soh_start = torch.full((B,), 0.9)
    target_cy = torch.linspace(50, 149, T)
    pred = model(x_health, theta, ctx_delta, soh_start, target_cy)
    assert pred.shape == (B, T), pred.shape
    loss = ((pred - 0.8) ** 2).mean()
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()), "no finite grads"
    print(f"[phase3_v7_operator] smoke OK  "
          f"params={model.n_parameters():,}  "
          f"pred shape={tuple(pred.shape)}  "
          f"pred[0,:3]={pred[0, :3].detach().cpu().numpy()}")
