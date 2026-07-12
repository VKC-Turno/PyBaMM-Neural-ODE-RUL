"""Torch-native versions of the L1/L2 physics rate functions for
use inside the PINN training loop.

Kept separate from physics.py (which is numpy/scipy for offline fitting)
because torch autograd needs pure-torch operations."""
from __future__ import annotations
import torch


def rate_L1(soh: torch.Tensor, k_SEI: torch.Tensor,
             p: torch.Tensor) -> torch.Tensor:
    """dSoH/dn = -k_SEI · SoH^p  (all batched tensors)."""
    return -k_SEI * torch.clamp(soh, min=1e-6) ** p


def rate_L2(soh: torch.Tensor, n_cycle: torch.Tensor,
             k_SEI: torch.Tensor, p: torch.Tensor,
             k_LAM: torch.Tensor, n_c: torch.Tensor,
             tau: torch.Tensor) -> torch.Tensor:
    """dSoH/dn = -k_SEI · SoH^p - k_LAM · exp((n-n_c)/tau) · [n > n_c].

    Uses smooth sigmoid gate instead of hard Heaviside for gradient flow.
    """
    sei_rate = k_SEI * torch.clamp(soh, min=1e-6) ** p
    arg = torch.clamp((n_cycle - n_c) / torch.clamp(tau, min=1.0), max=20.0)
    gate = torch.sigmoid((n_cycle - n_c) / torch.clamp(tau * 0.1, min=1.0))
    lam_rate = k_LAM * torch.exp(arg) * gate
    return -(sei_rate + lam_rate)
