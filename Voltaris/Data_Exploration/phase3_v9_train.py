"""V9 noise-robust training.

Same clean-split trainer as phase3_v8_train, but adds on-the-fly noise
augmentation to the K=50 context during training so the operator learns
to handle real-cell measurement noise + occasional data artefacts.

Noise model:
  1. Gaussian: N(0, sigma_gauss) added to every context_delta value.
     sigma_gauss = 0.010 (1% SoH std) — matches the typical cell-to-cell
     noise magnitude observed in Longterm CSVs (0.005-0.02 range).
  2. Outlier injection: with probability p_outlier per sample, replace
     1-5 randomly-placed cycles of the context with an artefact of
     magnitude ±artefact_mag (uniform in [-0.3, +0.3]).

Both augmentations are applied ONLY during training. Validation and test
run on the raw (noise-free) synthetic contexts to allow direct
comparability with v8 clean.

Usage:
    nohup .venv/bin/python -u Voltaris/Data_Exploration/phase3_v9_train.py \\
      --dataset configs/phase3_corpus/_v8_dataset.parquet \\
      --out    outputs/models/pinn_phase3_v9_noise_robust.pt \\
      > outputs/logs/phase3_v9_train_<ts>.log 2>&1 &
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from phase3_v7_train import (      # noqa: E402
    OperatorV7, _batch_to_tensors, _compose_loss, _compute_normalisation,
    _iterate_batches, _load_dataset,
    BATCH_SIZE, EPOCHS, PATIENCE, LR_MAX, LR_MIN, LAMBDA_MONO, LAMBDA_SHAPE,
    GRAD_CLIP, X_HEALTH_DIM, THETA_DIM, SEED as V7_SEED,
)

_PROJECT_ROOT = _HERE.parents[1]
DEFAULT_DATASET = _PROJECT_ROOT / "configs" / "phase3_corpus" / "_v8_dataset.parquet"
DEFAULT_OUT = _PROJECT_ROOT / "outputs" / "models" / "pinn_phase3_v9_noise_robust.pt"

# Noise-augmentation hyperparameters
SIGMA_GAUSS = 0.010   # ~1% SoH stdev, matches observed real-cell noise
P_OUTLIER   = 0.25    # 25% of training samples get outlier injection
OUTLIER_MAG = 0.30    # ±0.3 SoH swing (matches REPT_0034/0011 severity)
OUTLIER_MAX_STREAK = 5  # up to 5 consecutive bad cycles (matches REPT_0037)


def _split_from_column(df: pd.DataFrame):
    return (
        df.index[df["split"] == "train"].to_numpy(),
        df.index[df["split"] == "val"].to_numpy(),
        df.index[df["split"] == "test"].to_numpy(),
    )


def _augment_context(context_delta: torch.Tensor,
                     context_soh_start: torch.Tensor,
                     rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Add Gaussian noise + occasional outlier streaks to the training context.

    Args:
        context_delta: (batch, K) SoH-delta values
        context_soh_start: (batch,) absolute starting SoH
        rng: torch generator for reproducibility
    Returns:
        (context_delta_aug, context_soh_start_aug)
    """
    B, K_ctx = context_delta.shape

    # 1. Gaussian noise on every cycle
    gauss = torch.randn(B, K_ctx, generator=rng) * SIGMA_GAUSS
    aug_delta = context_delta + gauss

    # 2. Also perturb the starting SoH slightly so first-cycle noise is realistic
    start_noise = torch.randn(B, generator=rng) * SIGMA_GAUSS
    aug_start = context_soh_start + start_noise

    # 3. Outlier streak injection (per-sample)
    for b in range(B):
        if torch.rand(1, generator=rng).item() >= P_OUTLIER:
            continue
        streak_len = int(torch.randint(1, OUTLIER_MAX_STREAK + 1,
                                        (1,), generator=rng).item())
        # Position the streak in cycles [0, K - streak_len]
        pos = int(torch.randint(0, max(1, K_ctx - streak_len),
                                 (1,), generator=rng).item())
        mag = (torch.rand(1, generator=rng).item() * 2 - 1) * OUTLIER_MAG
        aug_delta[b, pos:pos + streak_len] += mag

    return aug_delta, aug_start


def train_v9_noise_robust(dataset_path: Path = DEFAULT_DATASET,
                            out_path: Path = DEFAULT_OUT,
                            batch_size: int = BATCH_SIZE,
                            epochs: int = EPOCHS,
                            patience: int = PATIENCE,
                            lr_max: float = LR_MAX,
                            lr_min: float = LR_MIN,
                            seed: int = V7_SEED) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)
    aug_rng = torch.Generator()
    aug_rng.manual_seed(seed + 1000)

    df = _load_dataset(Path(dataset_path))
    train_idx, val_idx, test_idx = _split_from_column(df)
    print(f"[v9_train] loaded {len(df):,} rows from {dataset_path}", flush=True)
    print(f"[v9_train] split (from column): train={len(train_idx)}, "
          f"val={len(val_idx)}, test={len(test_idx)}", flush=True)
    print(f"[v9_train] noise augmentation: sigma_gauss={SIGMA_GAUSS}, "
          f"p_outlier={P_OUTLIER}, outlier_mag=+/-{OUTLIER_MAG}, "
          f"max_streak={OUTLIER_MAX_STREAK}", flush=True)

    stats = _compute_normalisation(df.loc[train_idx])
    print(f"[v9_train] x_health mean={stats['xh_mean'].tolist()}, "
          f"std={stats['xh_std'].tolist()}", flush=True)

    K = int(df["K"].iloc[0])
    model = OperatorV7(K=K)
    model.set_x_health_stats(stats["xh_mean"], stats["xh_std"])
    model.set_theta_stats(stats["th_mean"], stats["th_std"])
    print(f"[v9_train] model params={model.n_parameters():,}, K={K}",
          flush=True)

    optim = Adam(model.parameters(), lr=lr_max)
    sched = CosineAnnealingLR(optim, T_max=epochs, eta_min=lr_min)

    best_val = float("inf")
    best_state: Optional[dict] = None
    bad_epochs = 0
    history: list[dict] = []

    for ep in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        train_sums = {"data": 0.0, "mono": 0.0, "shape": 0.0, "total": 0.0}
        n_batches = 0
        for batch_df, tgt_cy in _iterate_batches(df, train_idx, batch_size,
                                                    seed, ep):
            t = _batch_to_tensors(batch_df)
            # === V9 noise augmentation ===
            aug_delta, aug_start = _augment_context(t["context_delta"],
                                                     t["context_soh_start"],
                                                     aug_rng)
            optim.zero_grad()
            pred = model(t["x_health"], t["theta_norm"],
                          aug_delta, aug_start, tgt_cy)
            loss, parts = _compose_loss(pred, t["target_soh"],
                                          LAMBDA_MONO, LAMBDA_SHAPE)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()
            for k in train_sums:
                train_sums[k] += parts[k]
            n_batches += 1
        for k in train_sums:
            train_sums[k] /= max(1, n_batches)
        sched.step()

        # Validation runs on RAW context (no augmentation) so val loss is
        # directly comparable to v8_clean.
        model.eval()
        val_sums = {"data": 0.0, "mono": 0.0, "shape": 0.0, "total": 0.0}
        v_batches = 0
        with torch.no_grad():
            for batch_df, tgt_cy in _iterate_batches(df, val_idx, batch_size,
                                                        seed, ep):
                t = _batch_to_tensors(batch_df)
                pred = model(t["x_health"], t["theta_norm"],
                              t["context_delta"], t["context_soh_start"],
                              tgt_cy)
                _, parts = _compose_loss(pred, t["target_soh"],
                                            LAMBDA_MONO, LAMBDA_SHAPE)
                for k in val_sums:
                    val_sums[k] += parts[k]
                v_batches += 1
        for k in val_sums:
            val_sums[k] /= max(1, v_batches)

        marker = " "
        if val_sums["total"] < best_val:
            best_val = val_sums["total"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            marker = "*"
        else:
            bad_epochs += 1

        history.append({"epoch": ep, "train": train_sums, "val": val_sums,
                         "bad_epochs": bad_epochs})
        wall = time.time() - t_ep
        print(f"[v9_train] epoch {ep:3d}/{epochs}  "
              f"train={train_sums['total']:.6f}  "
              f"val={val_sums['total']:.6f}  "
              f"best={best_val:.6f} {marker}  "
              f"bad={bad_epochs}  wall={wall:.1f}s", flush=True)
        if bad_epochs >= patience:
            print(f"[v9_train] early stop at epoch {ep} "
                  f"(patience {patience} exceeded)", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test on raw context
    model.eval()
    test_sums = {"data": 0.0, "mono": 0.0, "shape": 0.0, "total": 0.0}
    t_batches = 0
    with torch.no_grad():
        for batch_df, tgt_cy in _iterate_batches(df, test_idx, batch_size,
                                                    seed, 0):
            t = _batch_to_tensors(batch_df)
            pred = model(t["x_health"], t["theta_norm"],
                          t["context_delta"], t["context_soh_start"], tgt_cy)
            _, parts = _compose_loss(pred, t["target_soh"],
                                        LAMBDA_MONO, LAMBDA_SHAPE)
            for k in test_sums:
                test_sums[k] += parts[k]
            t_batches += 1
    for k in test_sums:
        test_sums[k] /= max(1, t_batches)
    print(f"[v9_train] final test (raw ctx): {test_sums}", flush=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict":     model.state_dict(),
        "history":        history,
        "config": {
            "K": K,
            "x_health_dim": X_HEALTH_DIM,
            "theta_dim": THETA_DIM,
            "batch_size": batch_size,
            "epochs":     epochs,
            "patience":   patience,
            "lr_max":     lr_max,
            "lr_min":     lr_min,
            "lambda_mono":  LAMBDA_MONO,
            "lambda_shape": LAMBDA_SHAPE,
            "split_source": "v8_clean_column",
            "dataset":     str(dataset_path),
            "noise_augmentation": {
                "sigma_gauss": SIGMA_GAUSS,
                "p_outlier": P_OUTLIER,
                "outlier_mag": OUTLIER_MAG,
                "outlier_max_streak": OUTLIER_MAX_STREAK,
            },
        },
        "best_val": float(best_val),
        "test": test_sums,
        "xh_mean": stats["xh_mean"].cpu().numpy().tolist(),
        "xh_std":  stats["xh_std"].cpu().numpy().tolist(),
        "th_mean": stats["th_mean"].cpu().numpy().tolist(),
        "th_std":  stats["th_std"].cpu().numpy().tolist(),
    }, out_path)
    print(f"[v9_train] wrote checkpoint {out_path}  "
          f"best_val={best_val:.6f}  test={test_sums['total']:.6f}", flush=True)
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--out",     type=Path, default=DEFAULT_OUT)
    p.add_argument("--epochs",  type=int, default=EPOCHS)
    p.add_argument("--seed",    type=int, default=V7_SEED)
    args = p.parse_args()
    train_v9_noise_robust(args.dataset, args.out, epochs=args.epochs,
                           seed=args.seed)
