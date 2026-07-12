"""Joint PINN with L2 SEI+LAM physics prior.

Each cell gets per-cell learnable parameters:
- log_k_SEI  (SEI kinetic rate)
- p          (SoH exponent, sigmoid → [p_min, p_max])
- log_k_LAM  (LAM strength, sigmoid can force it near zero for cells
              that don't need LAM contribution)
- n_c        (LAM activation cycle)
- log_tau    (LAM time constant)

The extra parameters give cells like 19 (delayed acceleration) the
ability to fit an ODE that captures the mid-life LAM kick-in that
Level 1 (SoH^p) cannot describe.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from .data import CellData
from .physics import estimate_k_sei_from_window
from .physics_torch import rate_L2


@dataclass
class L2Config:
    K: int              = 100
    epochs: int         = 6000
    lr: float           = 1e-3
    lam_phys: float     = 1.0
    lam_mono: float     = 0.05
    lam_bc: float       = 1.0
    n_norm_scale: float = 2000.0
    n_col_per_cell: int = 200
    p_init: float       = 0.5
    verbose_every: int  = 1000
    # LAM parameter bounds
    n_c_min: float      = 100.0
    n_c_max: float      = 2000.0
    tau_min: float      = 50.0
    tau_max: float      = 800.0


class JointPINN_L2(nn.Module):
    def __init__(self, n_cells: int, n_shared_features: int,
                  embed_dim: int = 4, hidden: int = 64, n_layers: int = 4,
                  feat_mean: torch.Tensor = None,
                  feat_std:  torch.Tensor = None,
                  p_init: float = 0.5,
                  n_c_min: float = 100.0, n_c_max: float = 2000.0,
                  tau_min: float = 50.0, tau_max: float = 800.0):
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
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -6.0)

        # SEI params
        self.log_k_sei = nn.Parameter(torch.full((n_cells,), -9.0))
        self.p_raw = nn.Parameter(torch.zeros(n_cells))
        self.p_min = max(p_init - 0.5, 0.0)
        self.p_max = p_init + 0.5

        # LAM params — start with log_k_LAM small (LAM off by default);
        # cells that need it will push it up during training
        self.log_k_lam = nn.Parameter(torch.full((n_cells,), -12.0))   # 6e-6 initial
        self.n_c_raw = nn.Parameter(torch.zeros(n_cells))
        self.log_tau = nn.Parameter(torch.full((n_cells,), 5.0))       # tau ~ 150
        self.n_c_min = n_c_min; self.n_c_max = n_c_max
        self.tau_min = tau_min; self.tau_max = tau_max

        if feat_mean is None: feat_mean = torch.zeros(n_shared_features)
        if feat_std  is None: feat_std  = torch.ones(n_shared_features)
        self.register_buffer("feat_mean", feat_mean.clone())
        self.register_buffer("feat_std",  feat_std.clone())

    def _norm(self, x): return (x - self.feat_mean) / (self.feat_std + 1e-8)

    def p_value(self, idx=None):
        p = self.p_min + (self.p_max - self.p_min) * torch.sigmoid(self.p_raw)
        return p if idx is None else p[idx]

    def k_sei(self, idx): return torch.exp(self.log_k_sei[idx])
    def k_lam(self, idx): return torch.exp(self.log_k_lam[idx])
    def n_c(self, idx):
        return self.n_c_min + (self.n_c_max - self.n_c_min) * torch.sigmoid(self.n_c_raw[idx])
    def tau(self, idx):
        raw = self.log_tau[idx]
        return torch.clamp(torch.exp(raw), min=self.tau_min, max=self.tau_max)

    def forward(self, n_norm, x_shared, cell_idx, soh_init):
        h = self._norm(x_shared)
        z = self.embed(cell_idx)
        inp = torch.cat([n_norm, h, z], dim=-1)
        decrement = F.softplus(self.net(inp))
        return soh_init - decrement

    def dsoh_dnnorm(self, n_norm, x_shared, cell_idx, soh_init):
        n_norm = n_norm.detach().requires_grad_(True)
        soh = self.forward(n_norm, x_shared, cell_idx, soh_init)
        grad, = torch.autograd.grad(soh.sum(), n_norm, create_graph=True)
        return grad, soh


def _cell_tensors(cell: CellData, K: int, scale: float, idx: int,
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
    idx_t = torch.full((len(n_train),), idx, dtype=torch.long, device=device)
    return dict(n_norm=n_norm, s_meas=s_meas, x_shared=x_shared,
                cell_idx=idx_t, soh_init=soh_init,
                first_cy=first_cy, n_total_norm=(cell.n_total-first_cy)/scale)


def train_joint_L2(model: JointPINN_L2, cells: list[CellData],
                    cfg: L2Config, device: torch.device) -> dict:
    model.to(device).train()

    # Warm-start log_k_sei from linear fit
    k_init = [estimate_k_sei_from_window(c, cfg.K) for c in cells]
    with torch.no_grad():
        for i, k in enumerate(k_init):
            model.log_k_sei[i] = float(torch.log(torch.tensor(max(k, 1e-6))))
    print(f"  Physics prior init k_SEI: {[f'{k:.2e}' for k in k_init]}")

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

            # Physics collocation on full domain
            n_col_norm = torch.rand(cfg.n_col_per_cell, 1, device=device) * T["n_total_norm"]
            n_col_cyc  = n_col_norm * cfg.n_norm_scale + T["first_cy"]
            x_col = x_static[i].unsqueeze(0).expand(cfg.n_col_per_cell, -1)
            idx_col = torch.full((cfg.n_col_per_cell,), i, dtype=torch.long, device=device)
            soh0_col = torch.full((cfg.n_col_per_cell, 1), cell.soh_init, device=device)
            grad_norm, soh_at_col = model.dsoh_dnnorm(n_col_norm, x_col, idx_col, soh0_col)
            grad_per_cyc = grad_norm / cfg.n_norm_scale
            # L2 target
            target = rate_L2(
                soh_at_col, n_col_cyc,
                model.k_sei(idx_col).unsqueeze(-1),
                model.p_value(idx_col).unsqueeze(-1),
                model.k_lam(idx_col).unsqueeze(-1),
                model.n_c(idx_col).unsqueeze(-1),
                model.tau(idx_col).unsqueeze(-1),
            )
            L_phys_sum = L_phys_sum + F.mse_loss(grad_per_cyc, target)
            L_mono_sum = L_mono_sum + F.relu(grad_per_cyc).mean()

            # Boundary
            n_zero = torch.zeros(1, 1, device=device)
            x_zero = x_static[i].unsqueeze(0)
            idx_bc = torch.tensor([i], dtype=torch.long, device=device)
            soh0_bc = torch.full((1,1), cell.soh_init, device=device)
            soh_bc = model(n_zero, x_zero, idx_bc, soh0_bc)
            L_bc_sum = L_bc_sum + F.mse_loss(soh_bc, soh0_bc)

        N = len(cells)
        L_data = L_data_sum / N; L_phys = L_phys_sum / N
        L_bc   = L_bc_sum / N;   L_mono = L_mono_sum / N
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

    # Extract final params
    with torch.no_grad():
        params = dict(
            k_sei=torch.exp(model.log_k_sei).cpu().tolist(),
            p=model.p_value().cpu().tolist(),
            k_lam=torch.exp(model.log_k_lam).cpu().tolist(),
            n_c=(model.n_c_min + (model.n_c_max - model.n_c_min)
                  * torch.sigmoid(model.n_c_raw)).cpu().tolist(),
            tau=torch.clamp(torch.exp(model.log_tau), min=cfg.tau_min, max=cfg.tau_max).cpu().tolist(),
        )
    return dict(history=history, params=params)


@torch.no_grad()
def predict_L2(model: JointPINN_L2, cell: CellData, idx: int,
                cfg: L2Config, device: torch.device) -> torch.Tensor:
    model.eval()
    first_cy = float(cell.n_traj[0])
    n = cell.n_traj.to(device)
    n_norm = (n - first_cy).unsqueeze(-1) / cfg.n_norm_scale
    x_shared = cell.x_health[:-1].to(device).unsqueeze(0).expand(len(n), -1)
    idx_t = torch.full((len(n),), idx, dtype=torch.long, device=device)
    soh_init = torch.full((len(n), 1), cell.soh_init, device=device)
    return model(n_norm, x_shared, idx_t, soh_init).squeeze(-1).cpu()
