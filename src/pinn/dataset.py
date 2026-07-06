"""
src/pinn/dataset.py
-------------------
Dataset adapters that turn the Phase-2 synthetic trajectories
(`data/synthetic/trajectories.parquet`) and the Phase-1 measured RPT /
Longterm fade tables into the (soh_traj, n_traj, x_health) tuples that
`RULPredictor` consumes.

Two dataset flavours
~~~~~~~~~~~~~~~~~~~~
`SyntheticTrajectoryDataset` — one sample = one full PyBaMM run. Returns
the full SOH(n) trajectory for each sample and a per-sample health
feature vector. Used by Phase-1 pre-training.

`RealCellDataset` — one sample = one real cell's sparse RPT/Longterm
fade curve. Returns whatever cycles were measured plus a (synthetic
proxy for) health feature vector. Used by Phase-2 fine-tuning.

Health features expected by `RULPredictor` (per `configs/pinn_config.yaml`):
    temperature_C, c_rate, dcir_mOhm, ic_peak1_shift_V, ic_peak2_area_norm

`ic_peak1_shift_V` and `ic_peak2_area_norm` are derived from cycle 1
values inside `_compute_health_features` so the convention is uniform
across the two dataset flavours.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


HEALTH_FEATURES = [
    "temperature_C",
    "c_rate",
    "dcir_mOhm",
    "ic_peak1_shift_V",
    "ic_peak2_area_norm",
]


@dataclass
class TrajectorySample:
    sample_id: str
    n_traj: torch.Tensor       # (T,)   cycle numbers, float
    soh_traj: torch.Tensor     # (T,)   measured/simulated SOH
    x_health: torch.Tensor     # (H,)   health feature vector
    soh_0: float               # SOH at n_traj[0] (for ODE initial condition)


def _compute_health_features(df: pd.DataFrame, ambient_C: float = 25.0) -> dict:
    """
    Derive the standard 5-feature health vector from a per-cycle DataFrame.

    Each value is taken to be the *cycle-1* characteristic of the sample —
    the model uses these as a static context, not a per-cycle input.
    """
    df = df.sort_values("cycle_n").reset_index(drop=True)
    first = df.iloc[0]
    out = {
        "temperature_C": ambient_C,
        "c_rate": float(first.get("c_rate", np.nan)),
        "dcir_mOhm": float(first.get("dcir_mOhm", np.nan)),
        "ic_peak1_shift_V": 0.0,           # by definition zero at cycle 1
        "ic_peak2_area_norm": 1.0,         # normalised to itself
    }
    return out


def _to_tensor_traj(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    df = df.sort_values("cycle_n").reset_index(drop=True)
    n = torch.tensor(df["cycle_n"].to_numpy(dtype=np.float32))
    s = torch.tensor(df["SOH"].to_numpy(dtype=np.float32))
    return n, s


class SyntheticTrajectoryDataset(Dataset):
    """
    Wraps `data/synthetic/trajectories.parquet`. Each item is one sample's
    full per-cycle trajectory.
    """

    def __init__(self, parquet_path: Path | str = Path("data/synthetic/trajectories.parquet"),
                 ambient_C: float = 25.0,
                 min_cycles: int = 5,
                 drop_nan_features: bool = True,
                 max_rate_per_cycle: float | None = None,
                 min_n_cycles: int | None = None):
        """
        Args:
            max_rate_per_cycle: drop any sample whose mean fade rate
                (SOH_start - SOH_end) / n_cycles exceeds this value. Used to
                exclude pathological PyBaMM sweep samples whose dynamics are
                orders of magnitude faster than physically plausible LFP fade.
            min_n_cycles: drop samples with fewer completed cycles than this
                (often the early-terminated pathological runs).
        """
        self.path = Path(parquet_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        df = pd.read_parquet(self.path)
        self.samples: list[TrajectorySample] = []
        n_skipped_rate = 0
        n_skipped_cycles = 0
        for sid, g in df.groupby("sample_id"):
            if len(g) < min_cycles:
                continue
            if min_n_cycles is not None and len(g) < min_n_cycles:
                n_skipped_cycles += 1
                continue
            if max_rate_per_cycle is not None:
                g_sorted = g.sort_values("cycle_n")
                soh_start = float(g_sorted["SOH"].iloc[0])
                soh_end = float(g_sorted["SOH"].iloc[-1])
                n_cy = int(g_sorted["cycle_n"].iloc[-1])
                rate = (soh_start - soh_end) / max(1, n_cy)
                if rate > max_rate_per_cycle:
                    n_skipped_rate += 1
                    continue
            feats = _compute_health_features(g, ambient_C=ambient_C)
            x = np.array([feats[k] for k in HEALTH_FEATURES], dtype=np.float32)
            if drop_nan_features and (not np.isfinite(x).all()):
                # In our current synthetic data, dcir_mOhm is NaN — replace
                # with a neutral placeholder (zero in standardised space).
                x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)
            n, soh = _to_tensor_traj(g)
            self.samples.append(TrajectorySample(
                sample_id=str(sid),
                n_traj=n, soh_traj=soh,
                x_health=torch.from_numpy(x),
                soh_0=float(soh[0]),
            ))

        if not self.samples:
            raise ValueError(f"No usable samples in {self.path} "
                             f"(min_cycles={min_cycles}, "
                             f"max_rate_per_cycle={max_rate_per_cycle}, "
                             f"min_n_cycles={min_n_cycles})")
        if n_skipped_rate or n_skipped_cycles:
            print(f"  SyntheticTrajectoryDataset: filtered out "
                  f"{n_skipped_rate} (rate > {max_rate_per_cycle}) and "
                  f"{n_skipped_cycles} (n_cycles < {min_n_cycles}); "
                  f"{len(self.samples)} samples remain")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TrajectorySample:
        return self.samples[idx]

    def feature_matrix(self) -> np.ndarray:
        """Stack health features for normalisation statistics."""
        return np.stack([s.x_health.numpy() for s in self.samples], axis=0)


class RealCellDataset(Dataset):
    """
    Wraps the per-cell measured fade curves under `data/processed/`.

    Inputs:
        rpt_path:       data/processed/rpt_capacity_fade.parquet
        longterm_path:  data/processed/longterm_capacity_fade.parquet (optional)

    For each cell we merge whatever points are available and treat the
    union as the "ground truth" sparse trajectory.
    """

    def __init__(self,
                 rpt_path: Path | str = Path("data/processed/rpt_capacity_fade.parquet"),
                 longterm_path: Optional[Path | str] = Path("data/processed/longterm_capacity_fade.parquet"),
                 ambient_C: float = 25.0,
                 c_rate_proxy: float = 0.5,
                 dcir_proxy_mOhm: float = 1.74,   # from param_id_report median
                 ):
        self.samples: list[TrajectorySample] = []
        frames: list[pd.DataFrame] = []
        if rpt_path and Path(rpt_path).exists():
            r = pd.read_parquet(rpt_path)[["cell_id", "cycle_n", "SOH"]]
            r["source"] = "rpt"
            frames.append(r)
        if longterm_path and Path(longterm_path).exists():
            l = pd.read_parquet(longterm_path)[["cell_id", "cycle_n", "SOH"]]
            l["source"] = "longterm"
            frames.append(l)
        if not frames:
            return
        all_df = pd.concat(frames, ignore_index=True)
        all_df = all_df.dropna(subset=["SOH"]).sort_values(["cell_id", "cycle_n"])

        for cid, g in all_df.groupby("cell_id"):
            g = g.sort_values("cycle_n").drop_duplicates("cycle_n", keep="first")
            if len(g) < 3:
                continue
            n = torch.tensor(g["cycle_n"].to_numpy(dtype=np.float32))
            soh = torch.tensor(g["SOH"].to_numpy(dtype=np.float32))
            x = np.array([
                ambient_C, c_rate_proxy, dcir_proxy_mOhm, 0.0, 1.0,
            ], dtype=np.float32)
            self.samples.append(TrajectorySample(
                sample_id=str(cid),
                n_traj=n, soh_traj=soh,
                x_health=torch.from_numpy(x),
                soh_0=float(soh[0]),
            ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TrajectorySample:
        return self.samples[idx]


def collate_variable_length(batch: list[TrajectorySample]) -> dict:
    """
    PyTorch DataLoader collator: trajectories have different lengths
    across samples, so we return a list rather than a stacked tensor.
    """
    return {
        "sample_id": [s.sample_id for s in batch],
        "n_traj": [s.n_traj for s in batch],
        "soh_traj": [s.soh_traj for s in batch],
        "x_health": torch.stack([s.x_health for s in batch], dim=0),
        "soh_0": torch.tensor([[s.soh_0] for s in batch], dtype=torch.float32),
    }


if __name__ == "__main__":
    syn = SyntheticTrajectoryDataset()
    print(f"Synthetic: {len(syn)} samples")
    s0 = syn[0]
    print(f"  sample {s0.sample_id}: T={len(s0.n_traj)}, x_health={s0.x_health.tolist()}, "
          f"SOH_0={s0.soh_0:.4f}, SOH_end={s0.soh_traj[-1]:.4f}")

    feats = syn.feature_matrix()
    print(f"  feature mean: {feats.mean(axis=0)}")
    print(f"  feature std : {feats.std(axis=0)}")

    try:
        real = RealCellDataset()
        print(f"Real:      {len(real)} cells")
        for r in real:
            print(f"  cell {r.sample_id}: T={len(r.n_traj)}, "
                  f"SOH range [{r.soh_traj.min():.3f}, {r.soh_traj.max():.3f}]")
    except FileNotFoundError as e:
        print(f"Real dataset not available: {e}")
