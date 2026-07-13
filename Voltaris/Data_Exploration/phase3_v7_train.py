"""
Voltaris/Data_Exploration/phase3_v7_train.py
============================================

v7 training loop for OperatorV7. Reads the v7 dataset (context/target
pairs), stratifies by anchor_id, buckets by context_start so torchdiffeq
gets a shared time grid per batch, then trains with:

    L = L_data + λ_mono · L_monotonicity + λ_shape · L_shape

    L_data      = MSE(pred_soh, target_soh)
    L_mono      = mean( ReLU(pred[t+1] - pred[t])^2 )       # non-increasing
    L_shape     = MSE(d² pred / dn², d² target / dn²)       # curvature match

Cosine LR schedule, Adam optim, 60-epoch cap with patience=15. Logs one
line per epoch with flush=True so the training log stays live.

Usage:
    .venv/bin/python -u Voltaris/Data_Exploration/phase3_v7_train.py \
        --dataset configs/phase3_corpus/_v7_dataset.parquet \
        --out outputs/models/pinn_phase3_v7_operator.pt
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
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase3_v7_operator import (   # noqa: E402
    OperatorV7,
    X_HEALTH_DIM,
    THETA_DIM,
)


# ------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------
DEFAULT_DATASET   = _PROJECT_ROOT / "configs" / "phase3_corpus" / "_v7_dataset.parquet"
DEFAULT_OUT       = _PROJECT_ROOT / "outputs" / "models" / "pinn_phase3_v7_operator.pt"
DEFAULT_LOG_DIR   = _PROJECT_ROOT / "outputs" / "logs"

BATCH_SIZE   = 16
EPOCHS       = 60
PATIENCE     = 15
LR_MAX       = 1e-3
LR_MIN       = 1e-5
GRAD_CLIP    = 5.0
SEED         = 456
LAMBDA_MONO  = 0.5
LAMBDA_SHAPE = 0.3


# ------------------------------------------------------------------------
# Data plumbing
# ------------------------------------------------------------------------
def _to_tensor(x, dtype=torch.float32):
    return torch.tensor(np.asarray(x, dtype=np.float32), dtype=dtype)


def _load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Uniform-shape batching: only keep pairs with the modal forecast_len
    # (95 % of pairs already have forecast_len=400).
    modal = int(df["forecast_len"].mode().iloc[0])
    df = df[df["forecast_len"] == modal].reset_index(drop=True).copy()
    print(f"[v7_train] loaded {len(df):,} pairs "
          f"(forecast_len={modal}, K={int(df['K'].iloc[0])})", flush=True)
    return df


def _stratified_split(df: pd.DataFrame, val_frac: float = 0.15,
                       test_frac: float = 0.15, seed: int = SEED
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx) with per-anchor stratification."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for aid, grp in df.groupby("anchor_id"):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_val = int(round(n * val_frac))
        n_test = int(round(n * test_frac))
        n_train = n - n_val - n_test
        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train:n_train + n_val])
        test_idx.append(idx[n_train + n_val:])
    return (np.concatenate(train_idx),
            np.concatenate(val_idx),
            np.concatenate(test_idx))


def _compute_normalisation(df_train: pd.DataFrame) -> dict:
    """Fit z-score stats for x_health from the training rows.

    theta_norm is already pre-standardised in the extractor (log-space
    z-scores), so we leave those stats at (mean=0, std=1)."""
    xh = np.stack(df_train["x_health"].tolist()).astype(np.float32)
    xh_mean = xh.mean(axis=0)
    xh_std = xh.std(axis=0)
    # Guard against zero-variance columns (constant ambient temperature).
    xh_std = np.where(xh_std < 1e-8, 1.0, xh_std)
    return {
        "xh_mean": torch.tensor(xh_mean, dtype=torch.float32),
        "xh_std":  torch.tensor(xh_std, dtype=torch.float32),
        "th_mean": torch.zeros(THETA_DIM, dtype=torch.float32),
        "th_std":  torch.ones(THETA_DIM, dtype=torch.float32),
    }


# ------------------------------------------------------------------------
# Bucketed batch iterator (shares target_cycles across the batch)
# ------------------------------------------------------------------------
def _iterate_batches(df: pd.DataFrame, indices: np.ndarray,
                     batch_size: int, seed: int, epoch: int):
    """Yield (batch_df, target_cycles_tensor) tuples.

    Buckets by context_start so torchdiffeq gets a shared time grid.
    Bucket order + within-bucket order re-shuffled each epoch.
    """
    rng = np.random.default_rng(seed + epoch)
    df_sub = df.loc[indices].reset_index(drop=True)
    buckets: dict[int, np.ndarray] = {}
    for s, grp in df_sub.groupby("context_start"):
        buckets[int(s)] = grp.index.to_numpy()

    bucket_starts = np.array(list(buckets.keys()))
    rng.shuffle(bucket_starts)
    for s in bucket_starts:
        idx = buckets[int(s)].copy()
        rng.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            slice_idx = idx[i : i + batch_size]
            batch_df = df_sub.iloc[slice_idx].reset_index(drop=True)
            # All rows in this batch share target_cycles by construction.
            tgt_cy = _to_tensor(batch_df["target_cycles"].iloc[0])
            yield batch_df, tgt_cy


def _batch_to_tensors(batch_df: pd.DataFrame) -> dict:
    return {
        "x_health":         _to_tensor(np.stack(batch_df["x_health"].tolist())),
        "theta_norm":       _to_tensor(np.stack(batch_df["theta_norm"].tolist())),
        "context_delta":    _to_tensor(np.stack(batch_df["context_delta"].tolist())),
        "context_soh_start": _to_tensor(batch_df["context_soh_start"].to_numpy()),
        "target_soh":       _to_tensor(np.stack(batch_df["target_soh"].tolist())),
    }


# ------------------------------------------------------------------------
# Loss composition
# ------------------------------------------------------------------------
def _compose_loss(pred: torch.Tensor, tgt: torch.Tensor,
                   lam_mono: float, lam_shape: float
                   ) -> tuple[torch.Tensor, dict]:
    """pred, tgt: (B, T). Returns (total_loss, parts_dict)."""
    l_data = F.mse_loss(pred, tgt)

    # Monotone-non-increase penalty
    d1 = pred[:, 1:] - pred[:, :-1]
    l_mono = F.relu(d1).pow(2).mean()

    # Shape (curvature) — second finite difference
    d2_pred = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
    d2_tgt = tgt[:, 2:] - 2 * tgt[:, 1:-1] + tgt[:, :-2]
    l_shape = F.mse_loss(d2_pred, d2_tgt)

    total = l_data + lam_mono * l_mono + lam_shape * l_shape
    parts = {
        "data":  float(l_data.detach()),
        "mono":  float(l_mono.detach()),
        "shape": float(l_shape.detach()),
        "total": float(total.detach()),
    }
    return total, parts


# ------------------------------------------------------------------------
# Main training entry
# ------------------------------------------------------------------------
def train_v7(dataset_path: Path = DEFAULT_DATASET,
             out_path: Path = DEFAULT_OUT,
             batch_size: int = BATCH_SIZE,
             epochs: int = EPOCHS,
             patience: int = PATIENCE,
             lr_max: float = LR_MAX,
             lr_min: float = LR_MIN,
             seed: int = SEED) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)

    df = _load_dataset(Path(dataset_path))
    train_idx, val_idx, test_idx = _stratified_split(df, seed=seed)
    print(f"[v7_train] split: train={len(train_idx)}, "
          f"val={len(val_idx)}, test={len(test_idx)}", flush=True)

    stats = _compute_normalisation(df.loc[train_idx])
    print(f"[v7_train] x_health mean={stats['xh_mean'].tolist()}, "
          f"std={stats['xh_std'].tolist()}", flush=True)

    K = int(df["K"].iloc[0])
    model = OperatorV7(K=K)
    model.set_x_health_stats(stats["xh_mean"], stats["xh_std"])
    model.set_theta_stats(stats["th_mean"], stats["th_std"])
    print(f"[v7_train] model params={model.n_parameters():,}, K={K}",
          flush=True)

    optim = Adam(model.parameters(), lr=lr_max)
    sched = CosineAnnealingLR(optim, T_max=epochs, eta_min=lr_min)

    best_val = float("inf")
    best_state: Optional[dict] = None
    bad_epochs = 0
    history: list[dict] = []

    for ep in range(1, epochs + 1):
        t_ep = time.time()
        # ---- train ----
        model.train()
        train_sums = {"data": 0.0, "mono": 0.0, "shape": 0.0, "total": 0.0}
        n_batches = 0
        for batch_df, tgt_cy in _iterate_batches(df, train_idx, batch_size,
                                                    seed, ep):
            t = _batch_to_tensors(batch_df)
            optim.zero_grad()
            pred = model(t["x_health"], t["theta_norm"],
                          t["context_delta"], t["context_soh_start"], tgt_cy)
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

        # ---- val ----
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
        print(f"[v7_train] epoch {ep:3d}/{epochs}  "
              f"train={train_sums['total']:.6f}  "
              f"val={val_sums['total']:.6f}  "
              f"best={best_val:.6f} {marker}  "
              f"bad={bad_epochs}  wall={wall:.1f}s", flush=True)
        if bad_epochs >= patience:
            print(f"[v7_train] early stop at epoch {ep} "
                  f"(patience {patience} exceeded)", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

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
        },
        "normalisation": {
            "xh_mean": stats["xh_mean"].tolist(),
            "xh_std":  stats["xh_std"].tolist(),
        },
        "best_val": best_val,
    }, out_path)
    print(f"[v7_train] wrote checkpoint {out_path}  best_val={best_val:.6f}",
          flush=True)
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--out",     type=Path, default=DEFAULT_OUT)
    p.add_argument("--epochs",  type=int, default=EPOCHS)
    args = p.parse_args()
    train_v7(dataset_path=args.dataset, out_path=args.out, epochs=args.epochs)
