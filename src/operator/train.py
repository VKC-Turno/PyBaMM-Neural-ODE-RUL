"""Train the theta-conditioned DeepONet.

Stage A (pretrain): synthetic corpus (data/synthetic/trajectories.parquet).
Stage B (fine-tune): real cells with known theta (EVE 0005-0008 + REPT 0001).

For the initial pass we only do Stage A.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, "/home/hj/Desktop/PINNs")

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.operator.model   import ThetaDeepONet, OperatorConfig, loss_fn
from src.operator.dataset import build_dataset, DatasetConfig


def train(
    n_epochs: int = 200,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_frac: float = 0.15,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: Path = Path("/home/hj/Desktop/PINNs/outputs/models/theta_deeponet.pt"),
):
    torch.manual_seed(seed); np.random.seed(seed)

    ds = build_dataset(DatasetConfig(K=50, n_query=30))
    n_total = len(ds)
    n_val = int(round(n_total * val_frac))
    n_tr = n_total - n_val
    tr_ds, va_ds = random_split(ds, [n_tr, n_val],
                                  generator=torch.Generator().manual_seed(seed))
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                             num_workers=0, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0)

    cfg = OperatorConfig()
    model = ThetaDeepONet(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params on {device}")
    print(f"Data: {n_tr} train + {n_val} val = {n_total} sims")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    best_val = float("inf")
    history = []
    for ep in range(n_epochs):
        model.train()
        t0 = time.time()
        train_losses = []
        for batch in tr_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = loss_fn(model, batch, cfg)
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(out["loss"].item())
        sched.step()

        # Val
        model.eval()
        val_losses = []; val_data_losses = []
        with torch.no_grad():
            for batch in va_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = loss_fn(model, batch, cfg)
                val_losses.append(out["loss"].item())
                val_data_losses.append(out["L_data"].item())
        v_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        v_data = float(np.mean(val_data_losses)) if val_data_losses else float("nan")
        t_loss = float(np.mean(train_losses))
        # SoH RMSE in pp (data-loss is MSE on [0,1] SoH => sqrt * 100 = pp)
        v_rmse_pp = np.sqrt(v_data) * 100

        history.append(dict(ep=ep, train_loss=t_loss, val_loss=v_loss, val_rmse_pp=v_rmse_pp))
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"ep {ep+1:>4d}: train {t_loss:.5f}  val {v_loss:.5f}  "
                  f"val RMSE {v_rmse_pp:.2f} pp  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  {time.time()-t0:.1f}s")

        if v_loss < best_val:
            best_val = v_loss
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(state_dict=model.state_dict(),
                              cfg=cfg.__dict__,
                              val_loss=v_loss, ep=ep), save_path)

    print(f"\nBest val_loss: {best_val:.5f}")
    print(f"Checkpoint: {save_path}")
    return history


if __name__ == "__main__":
    train()
