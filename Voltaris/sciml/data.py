"""Data pipeline for CALB PINN comparison.

Loads 9 CALB cells (7 clean + 2 batch-artefact), computes health features,
splits each cell's trajectory at cycle K into (train, held-out) for the
same 5-K hold-out sweep the current abstract uses.

Cell IDs (canonical CALB numbering):
    clean:      [6, 7, 10, 14, 19, 20, 25]
    excluded:   [24, 30]   ← batch-transition discontinuities
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch

CANON_PQ  = Path("/home/hj/Desktop/PINNs/soh/data/canonical/calb_old.parquet")
CLEAN_IDS = [6, 7, 10, 14, 19, 20, 25]
DIRTY_IDS = [24, 30]
ALL_IDS   = CLEAN_IDS + DIRTY_IDS

K_VALUES  = [50, 100, 200, 400, 800]

# Health-feature columns pulled from each cell's early cycles
HEALTH_KEYS = ["dcir_start_mohm", "q_init_ah", "c_rate_mean", "cell_indicator"]


@dataclass
class CellData:
    cell_id: int
    is_clean: bool
    n_traj: torch.Tensor          # (T,) — global_cycle numbers, float32
    soh_traj: torch.Tensor        # (T,) — measured SoH, in [0, 1]
    x_health: torch.Tensor        # (F,) — static per-cell features
    soh_init: float               # SoH at n_traj[0]
    n_total: int                  # int, max cycle

    def split_at_K(self, k: int) -> tuple["CellData", "CellData"]:
        """Return (train, test) slices at global_cycle = first_cycle + k."""
        first_cy = float(self.n_traj[0])
        k_end = first_cy + k
        train_mask = self.n_traj <= k_end
        test_mask  = self.n_traj >  k_end
        train = CellData(
            cell_id=self.cell_id, is_clean=self.is_clean,
            n_traj=self.n_traj[train_mask], soh_traj=self.soh_traj[train_mask],
            x_health=self.x_health, soh_init=self.soh_init,
            n_total=self.n_total,
        )
        test = CellData(
            cell_id=self.cell_id, is_clean=self.is_clean,
            n_traj=self.n_traj[test_mask], soh_traj=self.soh_traj[test_mask],
            x_health=self.x_health, soh_init=self.soh_init,
            n_total=self.n_total,
        )
        return train, test


def _hampel_filter(series: pd.Series, k: float = 3.0, window: int = 5) -> pd.Series:
    """Return boolean mask of INLIERS."""
    if len(series) < window:
        return pd.Series([True] * len(series), index=series.index)
    med = series.rolling(window, center=True, min_periods=1).median()
    mad = (series - med).abs().rolling(window, center=True, min_periods=1).median()
    return (series - med).abs() <= k * 1.4826 * mad.clip(lower=1e-9)


def _load_one_cell(canon: pd.DataFrame, cid: int, skip_first: int = 1) -> CellData:
    is_clean = cid in CLEAN_IDS
    c = canon[canon.cell_id.astype(str).str.zfill(4) == f"{cid:04d}"].sort_values("global_cycle")
    c = c[(c.global_cycle >= skip_first + 1) & (c.soh > 0.05)].copy()
    keep = _hampel_filter(c.soh * 100.0)
    c = c[keep].copy()

    n = torch.tensor(c.global_cycle.to_numpy(dtype=np.float32))
    s = torch.tensor(c.soh.to_numpy(dtype=np.float32))

    # Health features — cheap, cell-specific, computed once
    ir = c.ir_ohm.dropna().iloc[:5].mean() if c.ir_ohm.notna().any() else 0.001
    q  = c.dchg_cap_ah.dropna().iloc[:5].mean() if c.dchg_cap_ah.notna().any() else 100.0
    cr = c.c_rate.dropna().mean() if c.c_rate.notna().any() else 0.5
    feats = np.array([
        float(ir) * 1000.0,       # dcir_start_mohm
        float(q),                 # q_init_ah
        float(cr),                # c_rate_mean
        float(ALL_IDS.index(cid)),  # cell_indicator (numeric embedding key)
    ], dtype=np.float32)

    return CellData(
        cell_id=cid, is_clean=is_clean,
        n_traj=n, soh_traj=s,
        x_health=torch.from_numpy(feats),
        soh_init=float(s[0]),
        n_total=int(n.max()),
    )


def load_all(cell_ids: list[int] | None = None) -> list[CellData]:
    """Load all requested CALB cells. Default = 9 (clean + dirty)."""
    if not CANON_PQ.exists():
        raise FileNotFoundError(CANON_PQ)
    canon = pd.read_parquet(CANON_PQ)
    ids = cell_ids or ALL_IDS
    return [_load_one_cell(canon, cid) for cid in ids]


def feature_normaliser(cells: list[CellData]) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-feature mean/std across a cell list. Cell-indicator kept as-is
    (categorical, not standardised)."""
    X = torch.stack([c.x_health for c in cells], dim=0)   # (N, F)
    mean = X.mean(dim=0)
    std  = X.std(dim=0).clamp(min=1e-6)
    # Don't standardise the cell-indicator feature (last column)
    mean[-1] = 0.0
    std[-1]  = 1.0
    return mean, std


if __name__ == "__main__":
    cells = load_all()
    mean, std = feature_normaliser(cells)
    print(f"Loaded {len(cells)} cells")
    for c in cells:
        tag = "clean" if c.is_clean else "DIRTY"
        print(f"  cell {c.cell_id:>2}  [{tag}]  N={c.n_total}  "
              f"SoH {c.soh_init:.3f} -> {float(c.soh_traj[-1]):.3f}  "
              f"features {c.x_health.tolist()}")
    print(f"\nFeature mean: {mean.tolist()}")
    print(f"Feature std:  {std.tolist()}")
