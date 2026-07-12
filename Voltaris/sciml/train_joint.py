"""Joint multi-cell PINN with LEARNABLE ODE parameters.

Rewrite: instead of pre-fitting (k_SEI, p) analytically and feeding
them as targets, we make them LEARNABLE parameters trained end-to-end
alongside the NN. This is the proper joint PINN formulation.

Architecture:
- One NN per cohort. Per-cell latent embedding.
- Per-cell learnable log_k_SEI (7 or 9 scalars).
- Shared exponent `p` (physics-motivated: rxn-lim SEI has p ~ 0.3-0.7).

Loss:
    L = L_data + λ_phys · MSE( dNN/dn, -exp(log_k_SEI) · SoH^p )
        + λ_bc · MSE( NN(0), soh_init )
        + λ_mono · ReLU(dNN/dn).mean()
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from .data import CellData
from .physics import estimate_k_sei_from_window


@dataclass
class JointConfig:
    K: int              = 100
    epochs: int         = 4000
    lr: float           = 1e-3
    lam_phys: float     = 1.0
    lam_mono: float     = 0.05
    lam_bc: float       = 1.0
    n_norm_scale: float = 2000.0
    n_col_per_cell: int = 150
    p_init: float       = 0.5      # SoH^0.5 (Ramadass-canonical rxn-lim SEI)
    p_min: float        = 0.1
    p_max: float        = 1.5
    verbose_every: int  = 500


class JointPINN(nn.Module):
    """Joint PINN with learnable per-cell k_SEI and shared exponent p."""

    def __init__(self, n_cells: int, n_shared_features: int,
                  embed_dim: int = 4, hidden: int = 64, n_layers: int = 4,
                  feat_mean: torch.Tensor = None,
                  feat_std:  torch.Tensor = None,
                  p_init: float = 0.5):
        super().__init__()
        self.n_cells   = n_cells
        self.embed = nn.Embedding(n_cells, embed_dim)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.1)

        in_dim = 1 + n_shared_features + embed_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        # Init: near-zero decrement everywhere. softplus(-6) ~ 2.5e-3.
        # This keeps the initial predicted trajectory FLAT (no fade) so the
        # physics loss doesn't blow up trying to match a spurious steep NN.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -6.0)

        # ── Learnable physics parameters (all per-cell) ──
        # log_k_SEI: per-cell scalar in log-space, initialised by caller
        # p_raw:     per-cell scalar, mapped via sigmoid to [p_min, p_max]
        # Initial log_k_SEI is set externally after construction — see
        # train_joint(), which passes the per-cell linear-fit k_SEI to
        # anchor the physics constraint from step 1.
        self.log_k_sei = nn.Parameter(torch.full((n_cells,), -9.0))
        self.p_raw     = nn.Parameter(torch.zeros(n_cells))
        self.p_min = max(p_init - 0.5, 0.0)
        self.p_max = p_init + 0.5

        if feat_mean is None: feat_mean = torch.zeros(n_shared_features)
        if feat_std  is None: feat_std  = torch.ones(n_shared_features)
        self.register_buffer("feat_mean", feat_mean.clone())
        self.register_buffer("feat_std",  feat_std.clone())

    def _norm(self, x): return (x - self.feat_mean) / (self.feat_std + 1e-8)

    def p_value(self, cell_idx: torch.Tensor = None) -> torch.Tensor:
        p_all = self.p_min + (self.p_max - self.p_min) * torch.sigmoid(self.p_raw)
        return p_all if cell_idx is None else p_all[cell_idx]

    def k_sei(self, cell_idx: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_k_sei[cell_idx])

    def forward(self, n_norm, x_shared, cell_idx, soh_init):
        h = self._norm(x_shared)
        z = self.embed(cell_idx)
        inp = torch.cat([n_norm, h, z], dim=-1)
        decrement = F.softplus(self.net(inp))
        return soh_init - decrement

    def dsoh_dnnorm(self, n_norm, x_shared, cell_idx, soh_init):
        n_norm = n_norm.detach().requires_grad_(True)
        soh    = self.forward(n_norm, x_shared, cell_idx, soh_init)
        grad,  = torch.autograd.grad(soh.sum(), n_norm, create_graph=True)
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
    cell_idx_t = torch.full((len(n_train),), cell_idx, dtype=torch.long, device=device)
    return dict(n_norm=n_norm, s_meas=s_meas, x_shared=x_shared,
                cell_idx=cell_idx_t, soh_init=soh_init,
                first_cy=first_cy, n_total_norm=(cell.n_total-first_cy)/scale)


def train_joint(model: JointPINN, cells: list[CellData], cfg: JointConfig,
                 device: torch.device) -> dict:
    model.to(device).train()

    # ── Initialise per-cell log_k_SEI from linear fit on training window ──
    # This gives the physics constraint a sensible target from step 1 rather
    # than the random -9 default. Without this, the NN's softplus(bias=-3)
    # initial output drives the physics loss to fit k_SEI to a huge value.
    k_init = [estimate_k_sei_from_window(c, cfg.K) for c in cells]
    with torch.no_grad():
        for i, k in enumerate(k_init):
            model.log_k_sei[i] = float(torch.log(torch.tensor(max(k, 1e-6))))
    print(f"  Physics prior init k_SEI: {[f'{k:.2e}' for k in k_init]}")

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    train_data = [_cell_tensors(c, cfg.K, cfg.n_norm_scale, i, device)
                   for i, c in enumerate(cells)]
    x_shared_static = [c.x_health[:-1].to(device) for c in cells]

    history = []
    for ep in range(cfg.epochs):
        optim.zero_grad()

        L_data_sum = 0.0; L_phys_sum = 0.0
        L_bc_sum   = 0.0; L_mono_sum = 0.0

        for i, (cell, T) in enumerate(zip(cells, train_data)):
            # Data
            soh_pred = model(T["n_norm"], T["x_shared"], T["cell_idx"], T["soh_init"])
            L_data_sum = L_data_sum + F.mse_loss(soh_pred, T["s_meas"])

            # Physics on collocation across full domain — dNN/dn ≈ -k(i) · NN^p(i)
            n_col = torch.rand(cfg.n_col_per_cell, 1, device=device) * T["n_total_norm"]
            x_col = x_shared_static[i].unsqueeze(0).expand(cfg.n_col_per_cell, -1)
            idx_col = torch.full((cfg.n_col_per_cell,), i, dtype=torch.long, device=device)
            soh0_col = torch.full((cfg.n_col_per_cell, 1), cell.soh_init, device=device)
            grad_norm, soh_at_col = model.dsoh_dnnorm(n_col, x_col, idx_col, soh0_col)
            grad_per_cycle = grad_norm / cfg.n_norm_scale
            k_sei_i = model.k_sei(idx_col).unsqueeze(-1)
            p_i = model.p_value(idx_col).unsqueeze(-1)
            target = -k_sei_i * torch.clamp(soh_at_col, min=1e-6) ** p_i
            L_phys_sum = L_phys_sum + F.mse_loss(grad_per_cycle, target)

            # Monotonicity
            L_mono_sum = L_mono_sum + F.relu(grad_per_cycle).mean()

            # Boundary
            n_zero = torch.zeros(1, 1, device=device)
            x_zero = x_shared_static[i].unsqueeze(0)
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
                                 p_per_cell=model.p_value().detach().cpu().tolist(),
                                 k_sei=torch.exp(model.log_k_sei).detach().cpu().tolist(),
                                 loss=loss.item()))

    p_final = model.p_value().detach().cpu().tolist()
    k_final = torch.exp(model.log_k_sei).detach().cpu().tolist()
    return dict(history=history, p_final=p_final, k_sei_final=k_final)


@torch.no_grad()
def predict_full_trajectory_joint(model: JointPINN, cell: CellData,
                                    cell_idx: int, cfg: JointConfig,
                                    device: torch.device) -> torch.Tensor:
    model.eval()
    first_cy = float(cell.n_traj[0])
    n = cell.n_traj.to(device)
    n_norm = (n - first_cy).unsqueeze(-1) / cfg.n_norm_scale
    x_shared = cell.x_health[:-1].to(device).unsqueeze(0).expand(len(n), -1)
    cell_idx_t = torch.full((len(n),), cell_idx, dtype=torch.long, device=device)
    soh_init = torch.full((len(n), 1), cell.soh_init, device=device)
    return model(n_norm, x_shared, cell_idx_t, soh_init).squeeze(-1).cpu()
