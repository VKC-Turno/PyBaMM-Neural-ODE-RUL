"""Hybrid linear + NN decrement PINN.

Change from JointPINN:
    old:  SoH(n) = SoH_init - softplus(NN_θ(n_norm, x_health, z_cell))
    new:  SoH(n) = SoH_init - softplus(log_a[cell]) * n_norm - softplus(NN_θ(...))

The linear part `softplus(log_a[cell]) * n_norm` provides a per-cell learnable
linear-fade term. The nonlinear part `softplus(NN)` corrects the shape.

Motivation: the softplus(NN) alone starts at ~0 (because we init NN's last
layer near zero), giving a flat trajectory that the network has to "learn to
bend". At small K (K=50 means 4% of a 1200-cy trajectory), there's not
enough signal to shape the extrapolation smoothly, producing the "flat then
drop" artefact visible in Fig. 2 of v2.

Warm-start `log_a` from the linear fit slope so the model starts predicting
roughly correct linear fade from step 0. The NN then adjusts the shape.

Both terms are non-negative in derivative wrt n_norm:
- softplus(log_a) > 0     — always positive linear rate
- d(softplus(NN))/dn      — sign-indeterminate, but existing mono loss keeps
                             it small if it would violate monotonicity
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import CellData
from .physics import estimate_k_sei_from_window


@dataclass
class HybridConfig:
    K: int              = 50
    epochs: int         = 10000
    lr: float           = 1e-3
    lam_phys: float     = 2.0
    lam_mono: float     = 0.05
    lam_bc: float       = 1.0
    n_norm_scale: float = 2000.0
    n_col_per_cell: int = 400
    p_init: float       = 0.5
    verbose_every: int  = 99999


class JointPINN_Hybrid(nn.Module):
    """Hybrid: per-cell linear fade + shared NN correction."""

    def __init__(self, n_cells: int, n_shared_features: int,
                  embed_dim: int = 8, hidden: int = 128, n_layers: int = 5,
                  feat_mean: torch.Tensor = None,
                  feat_std:  torch.Tensor = None,
                  p_init: float = 0.5):
        super().__init__()
        self.n_cells = n_cells
        self.embed = nn.Embedding(n_cells, embed_dim)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.1)

        in_dim = 1 + n_shared_features + embed_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        # Init NN correction to ~0 initially (softplus(-6) ≈ 0.0025)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -6.0)

        # ── Learnable physics parameters ──
        # log_a is inverse-softplus of the desired linear rate in normalised
        # coordinates. Warm-start externally after construction (via train_joint_hybrid).
        self.log_a = nn.Parameter(torch.full((n_cells,), -3.0))    # softplus(-3) ~ 0.05
        # SoH-dependence exponent (per-cell)
        self.p_raw = nn.Parameter(torch.zeros(n_cells))
        self.p_min = max(p_init - 0.5, 0.0)
        self.p_max = p_init + 0.5
        # SEI kinetic rate (per-cell, learnable)
        self.log_k_sei = nn.Parameter(torch.full((n_cells,), -9.0))

        if feat_mean is None: feat_mean = torch.zeros(n_shared_features)
        if feat_std  is None: feat_std  = torch.ones(n_shared_features)
        self.register_buffer("feat_mean", feat_mean.clone())
        self.register_buffer("feat_std",  feat_std.clone())

    def _norm(self, x): return (x - self.feat_mean) / (self.feat_std + 1e-8)

    def linear_rate(self, cell_idx):
        """Per-cell linear-fade rate in normalised coords (softplus for positivity)."""
        return F.softplus(self.log_a[cell_idx])

    def p_value(self, idx=None):
        p = self.p_min + (self.p_max - self.p_min) * torch.sigmoid(self.p_raw)
        return p if idx is None else p[idx]

    def k_sei(self, idx): return torch.exp(self.log_k_sei[idx])

    def forward(self, n_norm, x_shared, cell_idx, soh_init):
        """SoH(n) = SoH_init - softplus(log_a[cell]) * n_norm - softplus(NN)."""
        h = self._norm(x_shared)
        z = self.embed(cell_idx)
        inp = torch.cat([n_norm, h, z], dim=-1)
        nn_decrement = F.softplus(self.net(inp))
        # Per-row linear-rate lookup based on cell_idx
        a_row = self.linear_rate(cell_idx).unsqueeze(-1)   # (B, 1)
        linear_decrement = a_row * n_norm                    # (B, 1)
        return soh_init - linear_decrement - nn_decrement

    def dsoh_dnnorm(self, n_norm, x_shared, cell_idx, soh_init):
        n_norm = n_norm.detach().requires_grad_(True)
        soh = self.forward(n_norm, x_shared, cell_idx, soh_init)
        grad, = torch.autograd.grad(soh.sum(), n_norm, create_graph=True)
        return grad, soh


def _cell_tensors(cell: CellData, K: int, scale: float, cell_idx: int,
                   device: torch.device) -> dict:
    first_cy = float(cell.n_traj[0])
    k_end = first_cy + K
    mask = cell.n_traj <= k_end
    n_train = cell.n_traj[mask].to(device)
    s_train = cell.soh_traj[mask].to(device)
    x_shared = cell.x_health[:-1].to(device).unsqueeze(0).expand(len(n_train), -1)
    n_norm = (n_train - first_cy).unsqueeze(-1) / scale
    s_meas = s_train.unsqueeze(-1)
    soh_init = torch.full_like(s_meas, cell.soh_init)
    idx_t = torch.full((len(n_train),), cell_idx, dtype=torch.long, device=device)
    return dict(n_norm=n_norm, s_meas=s_meas, x_shared=x_shared,
                cell_idx=idx_t, soh_init=soh_init,
                first_cy=first_cy, n_total_norm=(cell.n_total-first_cy)/scale)


def _inverse_softplus(x):
    """log(exp(x) - 1) — inverse of softplus."""
    import math
    x = max(x, 1e-6)
    if x > 20: return x       # softplus(x) ≈ x for large x
    return math.log(math.exp(x) - 1)


def train_hybrid(model: JointPINN_Hybrid, cells: list[CellData],
                  cfg: HybridConfig, device: torch.device) -> dict:
    model.to(device).train()

    # Warm-start log_a from linear fit on training window
    k_init = [estimate_k_sei_from_window(c, cfg.K) for c in cells]
    with torch.no_grad():
        for i, k in enumerate(k_init):
            # Desired linear rate per unit n_norm: k * n_norm_scale
            target = k * cfg.n_norm_scale
            # Store inverse-softplus so softplus(log_a) = target
            model.log_a[i] = _inverse_softplus(target)
            # Also seed log_k_SEI for physics constraint
            model.log_k_sei[i] = float(torch.log(torch.tensor(max(k, 1e-6))))
    a_init_display = F.softplus(model.log_a).detach().cpu().tolist()
    print(f"  Linear-rate init: {[f'{v:.3f}' for v in a_init_display]}")

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    train_data = [_cell_tensors(c, cfg.K, cfg.n_norm_scale, i, device)
                   for i, c in enumerate(cells)]
    x_static = [c.x_health[:-1].to(device) for c in cells]

    history = []
    for ep in range(cfg.epochs):
        optim.zero_grad()

        L_data_sum = 0.0; L_phys_sum = 0.0
        L_bc_sum   = 0.0; L_mono_sum = 0.0

        for i, (cell, T) in enumerate(zip(cells, train_data)):
            # Data
            soh_pred = model(T["n_norm"], T["x_shared"], T["cell_idx"], T["soh_init"])
            L_data_sum = L_data_sum + F.mse_loss(soh_pred, T["s_meas"])

            # Physics on collocation across full domain
            n_col = torch.rand(cfg.n_col_per_cell, 1, device=device) * T["n_total_norm"]
            x_col = x_static[i].unsqueeze(0).expand(cfg.n_col_per_cell, -1)
            idx_col = torch.full((cfg.n_col_per_cell,), i, dtype=torch.long, device=device)
            soh0_col = torch.full((cfg.n_col_per_cell, 1), cell.soh_init, device=device)
            grad_norm, soh_at_col = model.dsoh_dnnorm(n_col, x_col, idx_col, soh0_col)
            grad_per_cycle = grad_norm / cfg.n_norm_scale
            k_sei_i = model.k_sei(idx_col).unsqueeze(-1)
            p_i = model.p_value(idx_col).unsqueeze(-1)
            target = -k_sei_i * torch.clamp(soh_at_col, min=1e-6) ** p_i
            L_phys_sum = L_phys_sum + F.mse_loss(grad_per_cycle, target)
            L_mono_sum = L_mono_sum + F.relu(grad_per_cycle).mean()

            # Boundary
            n_zero = torch.zeros(1, 1, device=device)
            x_zero = x_static[i].unsqueeze(0)
            idx_bc = torch.tensor([i], dtype=torch.long, device=device)
            soh0_bc = torch.full((1,1), cell.soh_init, device=device)
            soh_bc = model(n_zero, x_zero, idx_bc, soh0_bc)
            L_bc_sum = L_bc_sum + F.mse_loss(soh_bc, soh0_bc)

        N = len(cells)
        L_data = L_data_sum / N
        L_phys = L_phys_sum / N
        L_bc   = L_bc_sum   / N
        L_mono = L_mono_sum / N
        loss = (L_data + cfg.lam_phys * L_phys
                + cfg.lam_bc * L_bc + cfg.lam_mono * L_mono)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        sched.step()

        if (ep+1) % cfg.verbose_every == 0 or ep == 0:
            history.append(dict(epoch=ep+1,
                                 L_data=L_data.item(), L_phys=L_phys.item(),
                                 L_bc=L_bc.item(), L_mono=L_mono.item(),
                                 loss=loss.item()))

    return dict(
        history=history,
        a_final=F.softplus(model.log_a).detach().cpu().tolist(),
        p_final=model.p_value().detach().cpu().tolist(),
        k_sei_final=torch.exp(model.log_k_sei).detach().cpu().tolist(),
    )


@torch.no_grad()
def predict_hybrid(model: JointPINN_Hybrid, cell: CellData, cell_idx: int,
                    cfg: HybridConfig, device: torch.device) -> torch.Tensor:
    model.eval()
    first_cy = float(cell.n_traj[0])
    n = cell.n_traj.to(device)
    n_norm = (n - first_cy).unsqueeze(-1) / cfg.n_norm_scale
    x_shared = cell.x_health[:-1].to(device).unsqueeze(0).expand(len(n), -1)
    idx_t = torch.full((len(n),), cell_idx, dtype=torch.long, device=device)
    soh_init = torch.full((len(n), 1), cell.soh_init, device=device)
    return model(n_norm, x_shared, idx_t, soh_init).squeeze(-1).cpu()
