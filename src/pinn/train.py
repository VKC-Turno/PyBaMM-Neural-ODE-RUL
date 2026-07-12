"""
src/pinn/train.py
-----------------
Phase-1 (pre-train on synthetic) and Phase-2 (fine-tune on real) training
loops for the Neural ODE.

Usage
~~~~~
    # Quick pre-train dry-run (a few epochs, current synthetic dataset)
    .venv/bin/python -m src.pinn.train pretrain --epochs 5 --batch-size 4

    # Full pre-train using configs/pinn_config.yaml
    .venv/bin/python -m src.pinn.train pretrain

    # Fine-tune (loads pinn_pretrained.pt, trains on real RPT/Longterm)
    .venv/bin/python -m src.pinn.train finetune

Outputs
~~~~~~~
    outputs/models/pinn_pretrained.pt    after pretrain
    outputs/models/pinn_finetuned.pt     after finetune
    outputs/results/training_curves.png  per-epoch loss curves
    outputs/experiments/<run>/           full experiment-tracker bundle
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.pinn.dataset import (  # noqa: E402
    HEALTH_FEATURES, RealCellDataset, SyntheticTrajectoryDataset,
    collate_variable_length,
)
from src.pinn.loss import LossWeights, batch_loss  # noqa: E402
from src.pinn.model import RULPredictor  # noqa: E402
from src.experiment_tracking import ExperimentRun  # noqa: E402


CONFIG_PATH = Path("configs/pinn_config.yaml")
MODELS_DIR = Path("outputs/models")
RESULTS_DIR = Path("outputs/results")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensors in a variable-length batch onto `device`."""
    return {
        "sample_id": batch["sample_id"],
        "n_traj":   [t.to(device) for t in batch["n_traj"]],
        "soh_traj": [t.to(device) for t in batch["soh_traj"]],
        "x_health":  batch["x_health"].to(device),
        "soh_0":     batch["soh_0"].to(device),
    }


def _split_dataset(dataset, val_split: float, test_split: float, seed: int
                   ) -> tuple[Subset, Subset, Subset]:
    n_total = len(dataset)
    n_val = max(1, int(round(n_total * val_split)))
    n_test = max(1, int(round(n_total * test_split)))
    n_train = max(1, n_total - n_val - n_test)
    if n_train + n_val + n_test > n_total:
        # Pathological tiny datasets: take everything as train and reuse for val/test
        return (
            Subset(dataset, list(range(n_total))),
            Subset(dataset, list(range(n_total))),
            Subset(dataset, list(range(n_total))),
        )
    g = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=g)


def _set_normalisation_from_data(model: RULPredictor, dataset
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """Compute feature mean/std on the supplied dataset and apply to model."""
    if hasattr(dataset, "feature_matrix"):
        feats = dataset.feature_matrix()
    else:
        feats = np.stack([s.x_health.numpy() for s in dataset], axis=0)
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    # Avoid zero stds for constant features (e.g. temperature, peak shift baseline)
    std = np.where(std < 1e-6, 1.0, std)
    model.set_normalisation(torch.from_numpy(mean.astype(np.float32)),
                            torch.from_numpy(std.astype(np.float32)))
    return mean, std


def _make_loader(subset, batch_size: int) -> DataLoader:
    return DataLoader(
        subset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_variable_length, drop_last=False,
    )


def _train_one_epoch(model, loader, optim, weights, run, epoch: int,
                     device: torch.device | None = None
                     ) -> dict[str, float]:
    model.train()
    device = device or next(model.parameters()).device
    sums = {"total": 0.0, "data": 0.0, "physics": 0.0, "monotonicity": 0.0}
    n = 0
    for batch in loader:
        batch = _batch_to_device(batch, device)
        optim.zero_grad()
        out = batch_loss(model, batch, weights)
        out.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optim.step()

        bs = len(batch["sample_id"])
        for k in sums:
            sums[k] += float(getattr(out, k).detach().item()) * bs
        n += bs
    means = {k: v / max(1, n) for k, v in sums.items()}
    run.log_metrics({f"train_{k}": v for k, v in means.items()}, step=epoch)
    return means


@torch.no_grad()
def _evaluate(model, loader, weights,
              device: torch.device | None = None) -> dict[str, float]:
    model.eval()
    device = device or next(model.parameters()).device
    sums = {"total": 0.0, "data": 0.0, "physics": 0.0, "monotonicity": 0.0}
    n = 0
    for batch in loader:
        batch = _batch_to_device(batch, device)
        out = batch_loss(model, batch, weights)
        bs = len(batch["sample_id"])
        for k in sums:
            sums[k] += float(getattr(out, k).detach().item()) * bs
        n += bs
    return {k: v / max(1, n) for k, v in sums.items()}


def _save_curve_plot(history: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, lbl in [("train_total", "train"), ("val_total", "val")]:
        ys = [h.get(k, np.nan) for h in history]
        ax.plot(epochs, ys, label=lbl)
    ax.set(xlabel="epoch", ylabel="loss", yscale="log",
           title="Pre-training loss curves")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def pretrain(epochs: int | None = None, batch_size: int | None = None,
             lr: float | None = None, seed: int | None = None,
             checkpoint_name: str = "pinn_pretrained.pt",
             max_rate_per_cycle: float | None = None,
             min_n_cycles: int | None = None) -> dict:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    p = cfg["phase1_pretrain"]
    epochs = epochs or int(p["epochs"])
    batch_size = batch_size or int(p["batch_size"])
    lr = lr or float(p["lr"])
    seed = seed or int(p["seed"])
    val_split = float(p["val_split"])
    test_split = float(p["test_split"])
    patience = int(p["early_stopping_patience"])
    log_every = int(p.get("log_every_n_epochs", 1))

    _seed_everything(seed)

    weights = LossWeights(
        data=1.0,
        physics=float(p["lambda_physics"]),
        monotonicity=float(p["lambda_monotonicity"]),
    )

    # Data
    syn = SyntheticTrajectoryDataset(
        max_rate_per_cycle=max_rate_per_cycle,
        min_n_cycles=min_n_cycles,
    )
    print(f"  Synthetic dataset: {len(syn)} samples"
          + (f" (filter max_rate={max_rate_per_cycle:.0e}/cycle, "
             f"min_n_cycles={min_n_cycles})" if max_rate_per_cycle or min_n_cycles else ""))
    train_ds, val_ds, test_ds = _split_dataset(syn, val_split, test_split, seed)
    print(f"  Split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    train_loader = _make_loader(train_ds, batch_size)
    val_loader = _make_loader(val_ds, batch_size)
    test_loader = _make_loader(test_ds, batch_size)

    # Model
    model = RULPredictor.from_config(str(CONFIG_PATH))
    _set_normalisation_from_data(model, syn)
    device = _pick_device()
    model.to(device)
    print(f"  Model parameters: {model.n_parameters():,}")
    print(f"  Device: {device}")

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    if (p.get("scheduler") or "").lower() == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    else:
        sched = None

    # Run folder
    run = ExperimentRun.start(
        name="pinn_pretrain",
        config_paths=[str(CONFIG_PATH)],
        tags={"stage": "phase1_pretrain", "epochs": epochs,
              "batch_size": batch_size},
    )
    run.log_params({"epochs": epochs, "batch_size": batch_size, "lr": lr,
                    "seed": seed, "lambda_physics": weights.physics,
                    "lambda_monotonicity": weights.monotonicity,
                    "n_train": len(train_ds), "n_val": len(val_ds),
                    "n_test": len(test_ds)})

    history: list[dict] = []
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = _train_one_epoch(model, train_loader, optim,
                                         weights, run, ep, device=device)
        val_metrics = _evaluate(model, val_loader, weights, device=device)
        if sched is not None:
            sched.step()

        run.log_metrics({f"val_{k}": v for k, v in val_metrics.items()}, step=ep)
        history.append({
            "epoch": ep,
            "train_total": train_metrics["total"],
            "val_total": val_metrics["total"],
            "train_data": train_metrics["data"],
            "val_data": val_metrics["data"],
            "elapsed_s": time.time() - t0,
        })
        if ep == 1 or ep % log_every == 0 or ep == epochs:
            print(f"  epoch {ep:3d}/{epochs}  "
                  f"train_total={train_metrics['total']:.4e}  "
                  f"val_total={val_metrics['total']:.4e}  "
                  f"val_data={val_metrics['data']:.4e}  "
                  f"({history[-1]['elapsed_s']:.1f}s)")

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stop at epoch {ep} (no val improvement for {patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = _evaluate(model, test_loader, weights, device=device)
    run.log_metrics({f"test_{k}": v for k, v in test_metrics.items()}, step=ep)
    print(f"  TEST: {test_metrics}")

    # Save model + config
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS_DIR / checkpoint_name
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "phase": "pretrain",
        "best_val_loss": best_val,
        "test_metrics": test_metrics,
        "history": history,
    }, ckpt_path)
    run.log_artifact(ckpt_path)
    print(f"  saved checkpoint → {ckpt_path}")

    plot_path = RESULTS_DIR / "training_curves.png"
    _save_curve_plot(history, plot_path)
    run.log_artifact(plot_path)
    print(f"  saved training curves → {plot_path}")

    return {
        "n_epochs_run": history[-1]["epoch"],
        "best_val_total": best_val,
        "test_metrics": test_metrics,
        "checkpoint": str(ckpt_path),
        "run_dir": str(run.run_dir),
    }


def finetune(pretrain_ckpt: str = "outputs/models/pinn_pretrained.pt",
             epochs: int | None = None) -> dict:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    p = cfg["phase2_finetune"]
    epochs = epochs or int(p["epochs"])

    real = RealCellDataset()
    if len(real) == 0:
        raise RuntimeError("RealCellDataset is empty — check data/processed/")
    print(f"  Real dataset: {len(real)} cells")

    weights = LossWeights(
        data=1.0,
        physics=float(p["lambda_physics"]),
        monotonicity=float(p["lambda_monotonicity"]),
    )

    model = RULPredictor.from_config(str(CONFIG_PATH))
    ckpt = torch.load(pretrain_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    device = _pick_device()
    model.to(device)
    print(f"  Loaded pretrain checkpoint from {pretrain_ckpt}")
    print(f"  Device: {device}")

    # Freeze early layers
    n_freeze = int(p.get("freeze_first_n_layers", 0))
    for i, layer in enumerate(model.ode.net):
        if isinstance(layer, torch.nn.Linear) and i < n_freeze:
            for prm in layer.parameters():
                prm.requires_grad = False
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters after freeze: {n_trainable:,}")

    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                             lr=float(p["lr"]))

    run = ExperimentRun.start(name="pinn_finetune",
                              config_paths=[str(CONFIG_PATH)],
                              tags={"stage": "phase2_finetune"})
    run.log_params({"epochs": epochs, "n_real_cells": len(real)})

    loader = _make_loader(real, batch_size=int(p["batch_size"]))
    best = float("inf"); best_state = None; bad = 0
    history = []
    for ep in range(1, epochs + 1):
        m = _train_one_epoch(model, loader, optim, weights, run, ep, device=device)
        history.append({"epoch": ep, "train_total": m["total"]})
        run.log_metrics({f"train_{k}": v for k, v in m.items()}, step=ep)
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(f"  epoch {ep:3d}: train_total={m['total']:.4e}")
        if m["total"] < best:
            best, bad = m["total"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(p["early_stopping_patience"]):
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = MODELS_DIR / "pinn_finetuned.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": cfg,
                "phase": "finetune", "history": history,
                "best_train_loss": best}, ckpt_path)
    run.log_artifact(ckpt_path)
    print(f"  saved checkpoint → {ckpt_path}")

    return {"best_train_loss": best, "checkpoint": str(ckpt_path),
            "run_dir": str(run.run_dir)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["pretrain", "finetune"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-rate-per-cycle", type=float, default=None,
                    help="Filter synthetic samples whose mean fade rate "
                         "(SOH_start - SOH_end)/n_cycles exceeds this. "
                         "E.g. 5e-3 keeps ~29%% of the current sweep.")
    ap.add_argument("--min-n-cycles", type=int, default=None,
                    help="Filter synthetic samples with fewer completed "
                         "cycles than this.")
    args = ap.parse_args()

    if args.phase == "pretrain":
        result = pretrain(epochs=args.epochs, batch_size=args.batch_size,
                          lr=args.lr, seed=args.seed,
                          max_rate_per_cycle=args.max_rate_per_cycle,
                          min_n_cycles=args.min_n_cycles)
    else:
        result = finetune(epochs=args.epochs)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
