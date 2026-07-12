"""PINN model variants for the July-15 experiment.

All models take the same interface:
    forward(n_norm: (B,1), x_health: (B,F)) -> soh_pred: (B,1)

The physics constraint is applied externally in the loss module by
computing dSoH/dn via autograd and comparing to physics_rate(k_SEI).

Three variants:
- StandardPINN     — plain MLP with softplus monotonicity guard
- CausalPINN       — same architecture, causal temporal weighting at training time
- OpAugPINN        — small operator-style encoder for x_health, then MLP
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardPINN(nn.Module):
    """Direct solution representation with monotonic decrement.

        SoH(n) = SoH_init - softplus(NN_theta(n_norm, x_health))

    The softplus guarantees SoH is monotonically non-increasing.
    NN_theta output is unbounded; softplus maps it to [0, inf).
    """
    def __init__(self, n_features: int, hidden: int = 64,
                  n_layers: int = 3, feat_mean: torch.Tensor = None,
                  feat_std: torch.Tensor = None):
        super().__init__()
        self.n_features = n_features
        in_dim = 1 + n_features   # n_norm + features

        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

        # Init final layer to output ~0 initially → no fade at start
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -3.0)

        # Feature normalisation (registered so state saves with model)
        if feat_mean is None: feat_mean = torch.zeros(n_features)
        if feat_std  is None: feat_std  = torch.ones(n_features)
        self.register_buffer("feat_mean", feat_mean.clone())
        self.register_buffer("feat_std",  feat_std.clone())

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.feat_mean) / (self.feat_std + 1e-8)

    def forward(self, n_norm: torch.Tensor, x_health: torch.Tensor,
                 soh_init: torch.Tensor) -> torch.Tensor:
        """
        Args:
            n_norm:   (B,1) normalised cycle number ∈ [0, ~1]
            x_health: (B,F) raw features
            soh_init: (B,1) initial SoH per row
        Returns:
            soh_pred: (B,1)
        """
        h = self._norm(x_health)
        z = torch.cat([n_norm, h], dim=-1)
        decrement = F.softplus(self.net(z))
        return soh_init - decrement

    def dsoh_dnnorm(self, n_norm: torch.Tensor, x_health: torch.Tensor,
                     soh_init: torch.Tensor) -> torch.Tensor:
        """Autograd d(SoH)/d(n_norm). Convert to per-cycle via chain rule:
            dSoH/dn = dSoH/d(n_norm) * (1/n_norm_scale)
        where n_norm_scale is the divisor used to construct n_norm."""
        n_norm = n_norm.detach().requires_grad_(True)
        soh    = self.forward(n_norm, x_health, soh_init)
        grad,  = torch.autograd.grad(soh.sum(), n_norm, create_graph=True)
        return grad   # per unit n_norm; caller divides by n_norm_scale


class CausalPINN(StandardPINN):
    """Same architecture as StandardPINN. Causal training done externally
    via a weight schedule that emphasises early n_norm at start of training,
    gradually expanding coverage.

    The 'causal' character lives in the loss/weight schedule, not the model.
    We subclass so the runner code can dispatch by isinstance().
    """
    pass


class OpAugPINN(nn.Module):
    """Operator-augmented PINN: small encoder projects x_health into a
    latent embedding, then the trajectory network consumes (n_norm, z).

    Roughly DeepONet-inspired but degenerate (no separate trunk-branch
    inner-product) — just a feature encoder + trajectory decoder.
    """
    def __init__(self, n_features: int, embed_dim: int = 8, hidden: int = 64,
                  n_layers: int = 3, feat_mean: torch.Tensor = None,
                  feat_std: torch.Tensor = None):
        super().__init__()
        self.n_features = n_features
        # Encoder for characterisation features
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden), nn.Tanh(),
            nn.Linear(hidden, embed_dim),
        )
        # Trajectory net: (n_norm, z) -> decrement
        in_dim = 1 + embed_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.decoder = nn.Sequential(*layers)
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.constant_(self.decoder[-1].bias, -3.0)

        if feat_mean is None: feat_mean = torch.zeros(n_features)
        if feat_std  is None: feat_std  = torch.ones(n_features)
        self.register_buffer("feat_mean", feat_mean.clone())
        self.register_buffer("feat_std",  feat_std.clone())

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.feat_mean) / (self.feat_std + 1e-8)

    def forward(self, n_norm: torch.Tensor, x_health: torch.Tensor,
                 soh_init: torch.Tensor) -> torch.Tensor:
        h = self._norm(x_health)
        z = self.encoder(h)
        inp = torch.cat([n_norm, z], dim=-1)
        decrement = F.softplus(self.decoder(inp))
        return soh_init - decrement

    def dsoh_dnnorm(self, n_norm: torch.Tensor, x_health: torch.Tensor,
                     soh_init: torch.Tensor) -> torch.Tensor:
        n_norm = n_norm.detach().requires_grad_(True)
        soh    = self.forward(n_norm, x_health, soh_init)
        grad,  = torch.autograd.grad(soh.sum(), n_norm, create_graph=True)
        return grad


def build(variant: str, n_features: int, feat_mean=None, feat_std=None) -> nn.Module:
    """Factory. Variant ∈ {'standard', 'causal', 'op_aug'}."""
    v = variant.lower()
    if v == "standard": return StandardPINN(n_features, feat_mean=feat_mean, feat_std=feat_std)
    if v == "causal":   return CausalPINN  (n_features, feat_mean=feat_mean, feat_std=feat_std)
    if v == "op_aug":   return OpAugPINN   (n_features, feat_mean=feat_mean, feat_std=feat_std)
    raise ValueError(f"unknown variant: {variant}")
