"""Training loop for PINN variants.

Loss:
    L = L_data + λ_phys · L_physics + λ_mono · L_monotonicity

- L_data     = MSE(soh_pred, soh_measured) on training cycles
- L_physics  = MSE(dSoH/dn, -k_SEI) where k_SEI is fit on the same window
- L_monotonicity = ReLU(dSoH/dn).mean() — extra safety on softplus guard

For CausalPINN, we additionally weight samples by a temporal schedule
that starts sharp (early cycles) and broadens over epochs.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .data import CellData
from .physics import estimate_k_sei_from_window
from .models import CausalPINN


@dataclass
class TrainConfig:
    K: int              = 100      # training window (cycles)
    epochs: int         = 800
    lr: float           = 3e-4
    lam_phys: float     = 1.0      # in normalised coords L_phys is O(1e-2)
    lam_mono: float     = 0.1
    n_norm_scale: float = 2000.0   # cycles → n_norm ∈ [0, ~1]
    n_collocation: int  = 200      # physics-loss collocation points sampled across full [0, N_total]
    n_boundary_weight: float = 1.0 # weight of SoH(0) = soh_init anchor
    # Causal-PINN schedule params
    causal_alpha_start: float = 8.0
    causal_alpha_end:   float = 1.0
    verbose_every: int = 100


def _prepare_tensors(cell: CellData, K: int, scale: float,
                      device: torch.device) -> dict:
    """Slice cell to the training window [0, K], return per-cycle tensors."""
    first_cy = float(cell.n_traj[0])
    k_end = first_cy + K
    mask = cell.n_traj <= k_end
    n_train = cell.n_traj[mask].to(device)
    s_train = cell.soh_traj[mask].to(device)
    n_norm  = (n_train - first_cy).unsqueeze(-1) / scale        # (T,1)
    s_meas  = s_train.unsqueeze(-1)                              # (T,1)
    x_h = cell.x_health.to(device).unsqueeze(0).expand(len(n_train), -1)   # (T,F)
    soh_init = torch.full_like(s_meas, cell.soh_init)             # (T,1)
    return dict(n_norm=n_norm, s_meas=s_meas, x_health=x_h,
                soh_init=soh_init, first_cy=first_cy)


def _causal_weights(n_norm: torch.Tensor, alpha: float) -> torch.Tensor:
    """Causal weighting: exp(-alpha · L_j) where L_j is a running residual
    proxy. We use n_norm directly as a proxy (early cycles have higher weight
    when alpha is large; converges to uniform as alpha → 0)."""
    return torch.exp(-alpha * n_norm.squeeze(-1))


def train_one_cell(model: nn.Module, cell: CellData,
                    cfg: TrainConfig, device: torch.device) -> dict:
    """Train the model on a single cell's [0, K] window. Returns final losses."""
    model.to(device).train()
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)
    k_sei = estimate_k_sei_from_window(cell, cfg.K)
    k_sei_t = torch.tensor(k_sei, dtype=torch.float32, device=device)

    T = _prepare_tensors(cell, cfg.K, cfg.n_norm_scale, device)
    is_causal = isinstance(model, CausalPINN)

    # Collocation sampling range: from 0 to full trajectory length (in norm units)
    n_full_norm_max = float(cell.n_total - float(cell.n_traj[0])) / cfg.n_norm_scale

    history = []
    for ep in range(cfg.epochs):
        optim.zero_grad()

        # --- Data loss: prediction vs measured on training cycles ---
        soh_pred = model(T["n_norm"], T["x_health"], T["soh_init"])
        L_data = F.mse_loss(soh_pred, T["s_meas"])

        # --- Physics loss: enforce dSoH/d(n_norm) = -k_SEI · n_norm_scale on
        # collocation points sampled from the FULL domain. Working in
        # normalised units keeps the target O(0.05-0.5) rather than O(1e-5),
        # so the MSE is on a numerically sensible scale. ---
        n_norm_col = torch.rand(cfg.n_collocation, 1, device=device) * n_full_norm_max
        x_h_col    = cell.x_health.to(device).unsqueeze(0).expand(cfg.n_collocation, -1)
        soh_i_col  = torch.full((cfg.n_collocation, 1), cell.soh_init, device=device)
        grad = model.dsoh_dnnorm(n_norm_col, x_h_col, soh_i_col)   # d(SoH)/d(n_norm)
        target_grad = -k_sei_t * cfg.n_norm_scale                   # scalar target in normalised units
        L_phys = F.mse_loss(grad, target_grad.expand_as(grad))
        dsoh_dn = grad / cfg.n_norm_scale                            # per-cycle for monotonicity check

        # --- Boundary anchor: SoH(0) = soh_init ---
        n_zero = torch.zeros(1, 1, device=device)
        x_h_bc = cell.x_health.to(device).unsqueeze(0)
        soh_bc = model(n_zero, x_h_bc, torch.full((1,1), cell.soh_init, device=device))
        L_bc = F.mse_loss(soh_bc, torch.full((1,1), cell.soh_init, device=device))

        # --- Monotonicity (belt-and-braces; softplus already enforces) ---
        L_mono = F.relu(dsoh_dn).mean()

        # --- Causal weighting on the data term ---
        if is_causal:
            alpha = cfg.causal_alpha_start * (1 - ep/cfg.epochs) + \
                     cfg.causal_alpha_end   * (ep/cfg.epochs)
            w = _causal_weights(T["n_norm"], alpha).unsqueeze(-1)
            L_data = ((soh_pred - T["s_meas"])**2 * w).mean() / (w.mean() + 1e-8)

        loss = (L_data
                + cfg.lam_phys * L_phys
                + cfg.n_boundary_weight * L_bc
                + cfg.lam_mono * L_mono)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        sched.step()

        if (ep+1) % cfg.verbose_every == 0 or ep == 0:
            history.append(dict(epoch=ep+1,
                                 L_data=L_data.item(),
                                 L_phys=L_phys.item(),
                                 L_mono=L_mono.item(),
                                 loss=loss.item()))

    return dict(k_sei=k_sei, history=history)


@torch.no_grad()
def predict_full_trajectory(model: nn.Module, cell: CellData,
                             cfg: TrainConfig, device: torch.device) -> torch.Tensor:
    """Predict SoH at every cycle in cell.n_traj (train + test)."""
    model.eval()
    first_cy = float(cell.n_traj[0])
    n = cell.n_traj.to(device)
    n_norm = (n - first_cy).unsqueeze(-1) / cfg.n_norm_scale
    x_h = cell.x_health.to(device).unsqueeze(0).expand(len(n), -1)
    soh_init = torch.full((len(n), 1), cell.soh_init, device=device)
    return model(n_norm, x_h, soh_init).squeeze(-1).cpu()
