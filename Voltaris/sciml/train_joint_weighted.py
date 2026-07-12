"""Weighted variant of the joint PINN training loop.

Key change vs train_joint.py: each cell has a per-cell weight applied
to its DATA LOSS contribution (physics loss stays full weight for all).

Rationale: synthetic cells provide useful *physics-constraint diversity*
(varied k_SEI, varied fade shapes) which helps the network learn the
ODE structure. But they shouldn't dominate the data-loss objective —
their SoH values are PyBaMM-simulated, not measured. Down-weighting
synthetic data-loss to (say) 0.3 lets synthetic cells contribute to
the physics prior calibration without biasing real-cell predictions.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from .data import CellData
from .physics import estimate_k_sei_from_window
from .train_joint import JointPINN


@dataclass
class WeightedConfig:
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
    synth_weight: float = 0.3         # weight of synthetic cells in data loss (0.0-1.0)


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


def train_joint_weighted(model: JointPINN, cells: list[CellData],
                           weights: list[float], cfg: WeightedConfig,
                           device: torch.device) -> dict:
    """Weighted joint training. `weights[i]` scales cell i's data-loss."""
    assert len(weights) == len(cells)
    model.to(device).train()

    # Warm-start per-cell log_k_SEI
    k_init = [estimate_k_sei_from_window(c, cfg.K) for c in cells]
    with torch.no_grad():
        for i, k in enumerate(k_init):
            model.log_k_sei[i] = float(torch.log(torch.tensor(max(k, 1e-6))))

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
        total_weight = 0.0

        for i, (cell, T, w) in enumerate(zip(cells, train_data, weights)):
            # Data — weighted by cell's importance
            soh_pred = model(T["n_norm"], T["x_shared"], T["cell_idx"], T["soh_init"])
            L_data_sum = L_data_sum + w * F.mse_loss(soh_pred, T["s_meas"])
            total_weight += w

            # Physics — full weight for all cells (synthetic still teaches physics)
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
        # Data loss uses total_weight normalisation so a heavier-weighted set
        # gets appropriately averaged; other losses use plain N.
        L_data = L_data_sum / max(total_weight, 1e-6)
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

    p_final = model.p_value().detach().cpu().tolist()
    k_final = torch.exp(model.log_k_sei).detach().cpu().tolist()
    return dict(history=history, p_final=p_final, k_sei_final=k_final)
