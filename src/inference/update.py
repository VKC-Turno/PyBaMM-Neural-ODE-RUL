"""
src/inference/update.py
-----------------------
Lightweight online model update.

The use-case: a new RPT measurement arrives for a cell already being
tracked by an `outputs/models/pinn_finetuned.pt` checkpoint. We do a few
gradient steps on the *last layer only* to nudge the prediction toward
the new measurement, without overwriting the pre-training that captured
the underlying physics.

The update is **opt-in**: the caller decides whether the new measurement
warrants it (typical trigger: |observed SOH − predicted SOH| > 0.02). On
update we always save the updated weights to a *new* checkpoint file so
the original finetuned weights stay intact.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.pinn.model import RULPredictor  # noqa: E402

DEFAULT_CKPT = Path("outputs/models/pinn_finetuned.pt")


def _last_linear_idx(model: RULPredictor) -> int:
    """Return the index of the last nn.Linear in the ODE net."""
    last = -1
    for i, layer in enumerate(model.ode.net):
        if isinstance(layer, torch.nn.Linear):
            last = i
    return last


def predict_soh_at(model: RULPredictor, soh_0: float, n_at: float,
                   x_health: torch.Tensor, start_cycle: float = 0.0
                   ) -> torch.Tensor:
    """
    Forward-integrate SOH from `start_cycle` to `n_at` and return the
    final SOH as a (1,1) tensor. Differentiable end-to-end (relies on
    torchdiffeq's adjoint via the default path).
    """
    soh = torch.tensor([[soh_0]], dtype=torch.float32, requires_grad=False)
    n_eval = torch.tensor([float(start_cycle), float(n_at)], dtype=torch.float32)
    traj = model(soh, n_eval, x_health.unsqueeze(0))    # (2, 1, 1)
    return traj[-1].view(1, 1)


def online_update(model: RULPredictor,
                  new_soh: float,
                  new_cycle: float,
                  x_health: np.ndarray,
                  initial_soh: float = 1.0,
                  initial_cycle: float = 0.0,
                  n_steps: int = 20,
                  lr: float = 1e-4,
                  ewc_lambda: float = 1.0,
                  ) -> dict:
    """
    Apply `n_steps` updates to the last linear layer only, MSE-fitting the
    model's predicted SOH at `new_cycle` to `new_soh`. An EWC-style
    quadratic penalty pulls the last-layer weights back toward their
    pre-update values to discourage catastrophic forgetting.
    """
    # Snapshot pre-update last-layer weights for EWC
    last_idx = _last_linear_idx(model)
    last = model.ode.net[last_idx]
    theta_star = {
        n: p.detach().clone()
        for n, p in last.named_parameters()
    }

    # Freeze everything except the last linear layer
    for prm in model.parameters():
        prm.requires_grad = False
    for prm in last.parameters():
        prm.requires_grad = True

    optim = torch.optim.Adam(last.parameters(), lr=lr)
    x_t = torch.from_numpy(x_health.astype(np.float32))
    target = torch.tensor([[new_soh]], dtype=torch.float32)

    history: list[dict] = []
    for step in range(n_steps):
        optim.zero_grad()
        pred = predict_soh_at(model, soh_0=initial_soh, n_at=new_cycle,
                              x_health=x_t, start_cycle=initial_cycle)
        loss_data = F.mse_loss(pred, target)
        ewc = sum(((p - theta_star[n]) ** 2).sum()
                  for n, p in last.named_parameters())
        loss = loss_data + ewc_lambda * ewc
        loss.backward()
        optim.step()
        history.append({"step": step, "data_loss": float(loss_data.detach().item()),
                        "ewc_penalty": float(ewc.detach().item()),
                        "predicted_soh": float(pred.detach().item())})

    # Re-enable autograd on the full model (callers may want to keep training)
    for prm in model.parameters():
        prm.requires_grad = True

    return {
        "n_steps": n_steps,
        "lr": lr,
        "ewc_lambda": ewc_lambda,
        "history": history,
        "final_predicted_soh": history[-1]["predicted_soh"] if history else None,
        "target_soh": new_soh,
    }


def save_updated_checkpoint(model: RULPredictor, src_ckpt: Path,
                            dst_ckpt: Optional[Path] = None,
                            update_info: Optional[dict] = None) -> Path:
    """Save the updated model to a NEW file (preserves the original)."""
    src = Path(src_ckpt)
    if dst_ckpt is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dst_ckpt = src.with_name(f"{src.stem}_updated_{stamp}.pt")
    src_ckpt_data = torch.load(src, map_location="cpu", weights_only=False)
    src_ckpt_data["model_state_dict"] = model.state_dict()
    src_ckpt_data["online_update"] = update_info or {}
    torch.save(src_ckpt_data, dst_ckpt)
    return dst_ckpt


if __name__ == "__main__":
    # Smoke test against the finetuned checkpoint.
    from src.inference.predict_rul import load_model
    from src.inference.health_features import extract_for_cell

    model = load_model(DEFAULT_CKPT)
    h = extract_for_cell("0005").as_array()

    # Pretend cell 0005 was measured at cycle 200 with SOH = 0.90
    info = online_update(model, new_soh=0.90, new_cycle=200,
                         x_health=h, initial_soh=1.0,
                         n_steps=20, lr=1e-4)
    print(f"Final predicted SOH: {info['final_predicted_soh']:.4f} "
          f"(target {info['target_soh']:.4f})")
    out = save_updated_checkpoint(model, DEFAULT_CKPT, update_info=info)
    print(f"Saved updated checkpoint → {out}")
