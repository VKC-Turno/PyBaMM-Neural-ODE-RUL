"""Warm-start variant of the joint PINN.

Same architecture as JointPINN (softplus(NN) decrement only, no extra
linear term). The change is purely in TRAINING: before the main physics-
loss loop kicks in, we pre-train the NN alone to reproduce a linear-fade
curve derived from the training-window slope estimate.

Motivation: the baseline JointPINN inits the last-layer bias to -6 so
softplus(NN) ≈ 0 everywhere at t=0. This gives a flat initial trajectory;
data + physics losses have to bend it into the correct fade shape from
scratch. At small K (K=50, only ~4% of a 1200-cy trajectory), the data
signal is too weak to shape the extrapolation smoothly — the network
converges to a "flat start then plunge at end" artefact even though
overall RMSE stays low.

Warm-start fix: run a brief pre-training phase where the NN targets
`soh_init - k_L0 · (n - first_cy)` on collocation points spanning the
FULL cycle domain. This gives the NN a sensible initial slope from step 1.
The main training loop then refines this — data loss on the K-cycle window
corrects the shape, physics loss enforces the ODE structure.

Unlike Option 3 (hybrid linear+NN), the linear term is only used as
INITIALISATION, not as a fixed component. The NN retains full flexibility
to bend away from linear during main training — a critical requirement
because early LFP fade is steeper than long-run fade and any fixed linear
term would overshoot at the tail.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data      import CellData
from .physics   import estimate_k_sei_from_window
from .train_joint import JointPINN, _cell_tensors, JointConfig, train_joint


@dataclass
class WarmStartConfig:
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
    # Warm-start-specific:
    warmup_epochs: int      = 800
    warmup_lr: float        = 3e-3
    warmup_n_col: int       = 200
    # Fraction of full domain used as warm-start target range. 1.0 = full domain.
    # Lower values (e.g. 0.6) mean we only warm-start against the near-domain,
    # letting the NN discover the correct long-run shape from physics loss.
    warmup_domain_frac: float = 1.0


def _linear_target(soh_init: float, k_L0: float,
                    n_norm_grid: torch.Tensor, n_norm_scale: float) -> torch.Tensor:
    """soh_init - k_L0 · (n - first_cy) evaluated at n_norm points."""
    n_cycles = n_norm_grid * n_norm_scale
    return torch.clamp(soh_init - k_L0 * n_cycles, min=0.05)


def _warmup_phase(model: JointPINN, cells: list[CellData],
                    k_init: list[float], cfg: WarmStartConfig,
                    device: torch.device) -> None:
    """Pre-train NN alone to match linear-fade targets.

    Freezes log_k_sei, p_raw, embedding for the warmup phase — only the NN
    parameters move. This prevents the physics parameters from drifting
    while the NN is still learning basic decrement structure.

    IMPORTANT: JointPINN's default init sets last-layer weight=0, bias=-6, which
    gives softplus(NN) ≈ 0.0025 everywhere at init. The gradient signal through
    softplus is sigmoid(-6) ≈ 0.0025, a near-vanishing "gradient sink" that
    prevents warmup from moving the NN meaningfully. Here we reinit the last
    layer to a smaller-bias, small-random-weight configuration where the
    gradient signal is ~20× stronger (sigmoid(-3) ≈ 0.047).
    """
    # Reinit last layer so warmup can actually take
    with torch.no_grad():
        nn.init.xavier_normal_(model.net[-1].weight, gain=0.1)
        model.net[-1].bias.fill_(-3.0)

    # Freeze non-NN params
    for p in [model.log_k_sei, model.p_raw]:
        p.requires_grad_(False)
    model.embed.weight.requires_grad_(False)

    nn_params = [p for p in model.net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(nn_params, lr=cfg.warmup_lr)

    x_static = [c.x_health[:-1].to(device) for c in cells]
    n_total_norm = [(c.n_total - float(c.n_traj[0])) / cfg.n_norm_scale
                     for c in cells]

    for ep in range(cfg.warmup_epochs):
        opt.zero_grad()
        L_sum = 0.0
        for i, cell in enumerate(cells):
            n_max = n_total_norm[i] * cfg.warmup_domain_frac
            n_col = torch.rand(cfg.warmup_n_col, 1, device=device) * n_max
            x_col = x_static[i].unsqueeze(0).expand(cfg.warmup_n_col, -1)
            idx_col = torch.full((cfg.warmup_n_col,), i, dtype=torch.long,
                                   device=device)
            soh0 = torch.full((cfg.warmup_n_col, 1), cell.soh_init, device=device)
            soh_pred = model(n_col, x_col, idx_col, soh0)
            soh_target = _linear_target(cell.soh_init, k_init[i], n_col,
                                          cfg.n_norm_scale)
            L_sum = L_sum + F.mse_loss(soh_pred, soh_target)
        loss = L_sum / len(cells)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nn_params, max_norm=1.0)
        opt.step()

        if (ep + 1) % 200 == 0:
            print(f"    warmup ep {ep+1}: MSE = {loss.item():.6f}")

    # Unfreeze everything for the main training loop
    for p in [model.log_k_sei, model.p_raw]:
        p.requires_grad_(True)
    model.embed.weight.requires_grad_(True)


def train_joint_warmstart(model: JointPINN, cells: list[CellData],
                            cfg: WarmStartConfig,
                            device: torch.device) -> dict:
    """Warm-started joint PINN training.

    Phase 1: pre-train NN to match linear-fade target (all physics params frozen).
    Phase 2: standard train_joint loop (all params unfrozen).
    """
    model.to(device).train()

    k_init = [estimate_k_sei_from_window(c, cfg.K) for c in cells]
    with torch.no_grad():
        for i, k in enumerate(k_init):
            model.log_k_sei[i] = float(torch.log(torch.tensor(max(k, 1e-6))))
    print(f"  Physics prior init k_SEI: {[f'{k:.2e}' for k in k_init]}")

    print(f"  Warm-start: {cfg.warmup_epochs} epochs @ lr={cfg.warmup_lr}, "
          f"domain frac={cfg.warmup_domain_frac}")
    _warmup_phase(model, cells, k_init, cfg, device)

    # Measure initial slope after warm-start (sanity check)
    with torch.no_grad():
        first_cell = cells[0]
        n_probe = torch.tensor([[0.0], [0.05]], device=device)
        x_probe = first_cell.x_health[:-1].to(device).unsqueeze(0).expand(2, -1)
        idx_probe = torch.zeros(2, dtype=torch.long, device=device)
        soh0_probe = torch.full((2, 1), first_cell.soh_init, device=device)
        soh_probe = model(n_probe, x_probe, idx_probe, soh0_probe).squeeze()
        init_slope = (soh_probe[1] - soh_probe[0]) / 0.05
        print(f"  After warm-start: cell 0 initial slope in n_norm = {init_slope.item():+.4f}")

    # Now run the main training loop with the standard JointConfig-shaped cfg.
    main_cfg = JointConfig(
        K=cfg.K, epochs=cfg.epochs, lr=cfg.lr,
        lam_phys=cfg.lam_phys, lam_mono=cfg.lam_mono, lam_bc=cfg.lam_bc,
        n_norm_scale=cfg.n_norm_scale, n_col_per_cell=cfg.n_col_per_cell,
        p_init=cfg.p_init, verbose_every=cfg.verbose_every,
    )

    # We already ran log_k_SEI init inside train_joint too — that's fine, it
    # just reinits to the same values. The critical thing is the NN warm-start
    # persists (we don't re-init the NN inside train_joint).
    return train_joint(model, cells, main_cfg, device)
