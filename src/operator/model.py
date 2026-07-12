"""theta-conditioned DeepONet for SoH-trajectory prediction.

Architecture:
    branch: MLP over concat(dcir_fp, rpt_fp, cycling_encoder(soh_early),
                             theta_vec, protocol_vec)   ->  D-dim embedding
    trunk : MLP over normalised cycle number             ->  D-dim embedding
    out   : dot(branch, trunk)  +  bias
    output nonlinearity: SoH_hat = SoH_init - softplus(raw)   (hard monotonic)

Inputs (one sample):
    dcir_fp    : (9,)  internal resistance at 9 SOC points [mΩ]
    rpt_fp     : (6,)  RPT-derived features (Q_bol, Q_rpt, delta_Q_delta_V,
                        ic_peak1_area, ic_peak2_area, OCV_span)
    soh_early  : (K,)  measured SoH for first K cycles (K=50 default)
    theta_vec  : (T,)  PyBaMM param embedding — either raw sweep params
                       (T~=10) or a PCA/AE-compressed vector (T~=4).
                       In practice we use the 5 swept params + a few
                       BOL identifiers = 10 dims by default.
    protocol   : (4,)  c_rate, DoD, temperature, rest_time
    n          : (1,)  query cycle number, normalised by N_norm

We support batched inference: all input arrays get a leading batch dim.

Loss (during training):
    L_data     = MSE(soh_hat(n_query), soh_target(n_query))    on synth or real
    L_mono     = ReLU(soh_hat(n+1) - soh_hat(n)).mean()         monotonic penalty
    L_bc       = MSE(soh_hat(n=0), soh_init)                    boundary

At inference:
    Query n = linspace(0, N_max, ...); operator produces SoH(n) curve.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OperatorConfig:
    dcir_dim: int         = 9
    rpt_dim: int          = 6
    theta_dim: int        = 10
    protocol_dim: int     = 4
    early_K: int          = 50
    early_embed_dim: int  = 32
    embed_dim: int        = 128           # dot-product embedding size
    branch_hidden: int    = 256
    branch_layers: int    = 4
    trunk_hidden: int     = 128
    trunk_layers: int     = 4
    n_norm_scale: float   = 5000.0        # cycle-number normalisation

    # Regularisation weights
    lam_mono: float       = 1.0
    lam_bc: float         = 1.0


class _CyclingEncoder(nn.Module):
    """Small MLP that ingests the K-cycle soh_early sequence."""
    def __init__(self, K: int, out_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(K, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, soh_early: torch.Tensor) -> torch.Tensor:
        return self.net(soh_early)


class ThetaDeepONet(nn.Module):
    def __init__(self, cfg: OperatorConfig):
        super().__init__()
        self.cfg = cfg
        self.cycling_encoder = _CyclingEncoder(cfg.early_K, cfg.early_embed_dim)

        branch_in = (cfg.dcir_dim + cfg.rpt_dim + cfg.early_embed_dim +
                     cfg.theta_dim + cfg.protocol_dim)
        b_layers = [nn.Linear(branch_in, cfg.branch_hidden), nn.Tanh()]
        for _ in range(cfg.branch_layers - 1):
            b_layers += [nn.Linear(cfg.branch_hidden, cfg.branch_hidden), nn.Tanh()]
        b_layers += [nn.Linear(cfg.branch_hidden, cfg.embed_dim)]
        self.branch = nn.Sequential(*b_layers)

        t_layers = [nn.Linear(1, cfg.trunk_hidden), nn.Tanh()]
        for _ in range(cfg.trunk_layers - 1):
            t_layers += [nn.Linear(cfg.trunk_hidden, cfg.trunk_hidden), nn.Tanh()]
        t_layers += [nn.Linear(cfg.trunk_hidden, cfg.embed_dim)]
        self.trunk = nn.Sequential(*t_layers)

        self.output_bias = nn.Parameter(torch.zeros(1))

    def encode_branch(self, dcir_fp, rpt_fp, soh_early, theta_vec, protocol):
        h_early = self.cycling_encoder(soh_early)
        x = torch.cat([dcir_fp, rpt_fp, h_early, theta_vec, protocol], dim=-1)
        return self.branch(x)                # (B, embed_dim)

    def encode_trunk(self, n_query):
        n_norm = n_query.unsqueeze(-1) / self.cfg.n_norm_scale
        return self.trunk(n_norm)             # (B, N, embed_dim)

    def raw_output(self, dcir_fp, rpt_fp, soh_early, theta_vec, protocol,
                    n_query, soh_init):
        """Return raw pre-nonlinearity output for shape-preserving loss."""
        b = self.encode_branch(dcir_fp, rpt_fp, soh_early, theta_vec, protocol)
        if n_query.dim() == 1:
            n_query = n_query.unsqueeze(0)     # (1, N)
        # Broadcast batch dim
        if b.dim() == 2 and n_query.dim() == 2:
            # trunk shape (B, N, embed_dim)
            t = self.encode_trunk(n_query)
        else:
            raise ValueError("Shape mismatch")
        # Dot product across embed dim
        raw = (b.unsqueeze(1) * t).sum(-1) + self.output_bias    # (B, N)
        return raw

    def forward(self, dcir_fp, rpt_fp, soh_early, theta_vec, protocol,
                 n_query, soh_init):
        """Hard-monotonic decrement: SoH_hat = SoH_init - softplus(raw)."""
        raw = self.raw_output(dcir_fp, rpt_fp, soh_early, theta_vec, protocol,
                                n_query, soh_init)
        return soh_init.unsqueeze(-1) - F.softplus(raw)


def loss_fn(model: ThetaDeepONet, batch: dict, cfg: OperatorConfig) -> dict:
    """
    batch expects: dcir_fp, rpt_fp, soh_early, theta_vec, protocol,
                    n_query (B,N), soh_target (B,N), soh_init (B,)
    """
    soh_hat = model(batch["dcir_fp"], batch["rpt_fp"], batch["soh_early"],
                     batch["theta_vec"], batch["protocol"],
                     batch["n_query"], batch["soh_init"])

    L_data = F.mse_loss(soh_hat, batch["soh_target"])

    # Monotonicity along cycle axis
    diff = soh_hat[:, 1:] - soh_hat[:, :-1]
    L_mono = F.relu(diff).mean()

    # Boundary at first query point
    if batch["n_query"].min() < 1.0:
        L_bc = F.mse_loss(soh_hat[:, 0], batch["soh_init"])
    else:
        L_bc = soh_hat.new_zeros(())

    loss = L_data + cfg.lam_mono * L_mono + cfg.lam_bc * L_bc
    return {"loss": loss, "L_data": L_data, "L_mono": L_mono, "L_bc": L_bc}
