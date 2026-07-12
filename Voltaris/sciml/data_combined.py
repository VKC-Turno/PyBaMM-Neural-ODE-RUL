"""Combined data loader: real cells + PyBaMM-synthetic trajectories.

Merges:
  - Real cells from CALB, EVE, REPT canonical parquets (via data.py's schema)
  - Synthetic trajectories from Voltaris/outputs/synthetic/*.parquet

Each cell (real or synthetic) becomes a CellData instance with:
  - n_traj, soh_traj (measured cycles + SoH)
  - x_health (4-feature vector: dcir, q, c_rate, cell_indicator)
  - soh_init, n_total, is_clean, is_synthetic

The cell_indicator feature encodes cell ID across the pooled cohort
so the PINN's per-cell embedding can distinguish real from synthetic
and cell-from-cell within each pool.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import CellData, _hampel_filter


CALB_PQ  = Path("/home/hj/Desktop/PINNs/soh/data/canonical/calb_old.parquet")
REPT_PQ  = Path("/home/hj/Desktop/PINNs/soh/data/canonical/rept.parquet")
EVE_PQ   = Path("/home/hj/Desktop/PINNs/soh/data/canonical/eve.parquet")
SYNTH_PQ = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/synthetic/synthetic_trajectories.parquet")

CALB_CLEAN = ["0006", "0007", "0010", "0014", "0019", "0020", "0025"]
REPT_CLEAN = ["0001", "0003", "0007", "0028", "0043", "0057",
                "0074", "0078", "0087"]
EVE_CLEAN  = ["0002", "0003", "0004", "0008"]


def _load_real_cell(canon: pd.DataFrame, cid: str, tag: str, idx: int,
                     total_indicator_range: int, skip_first: int = 1) -> CellData:
    c = canon[canon.cell_id.astype(str) == str(cid)].sort_values("global_cycle")
    if not c.soh.notna().any():
        return None
    c = c[(c.global_cycle >= skip_first + 1) & (c.soh > 0.05)].copy()
    keep = _hampel_filter(c.soh * 100.0)
    c = c[keep].copy()
    if len(c) < 10:
        return None

    n = torch.tensor(c.global_cycle.to_numpy(dtype=np.float32))
    s = torch.tensor(c.soh.to_numpy(dtype=np.float32))
    ir = c.ir_ohm.dropna().iloc[:5].mean() if c.ir_ohm.notna().any() else 0.001
    q  = c.dchg_cap_ah.dropna().iloc[:5].mean() if c.dchg_cap_ah.notna().any() else 100.0
    cr = c.c_rate.dropna().mean() if c.c_rate.notna().any() else 0.5
    feats = np.array([
        float(ir) * 1000.0,
        float(q),
        float(cr),
        float(idx),
    ], dtype=np.float32)
    return CellData(
        cell_id=f"{tag}_{cid}",   # unique across cohorts
        is_clean=True,
        n_traj=n, soh_traj=s,
        x_health=torch.from_numpy(feats),
        soh_init=float(s[0]),
        n_total=int(n.max()),
    )


def _load_synth_cell(canon: pd.DataFrame, cid: str, idx: int) -> CellData:
    c = canon[canon.cell_id == cid].sort_values("global_cycle")
    if not c.soh.notna().any() or len(c) < 10:
        return None
    n = torch.tensor(c.global_cycle.to_numpy(dtype=np.float32))
    s = torch.tensor(c.soh.to_numpy(dtype=np.float32))
    # For synthetic, ir_ohm is NaN; use q_init and c_rate from the sim config
    q  = c.dchg_cap_ah.dropna().iloc[:5].mean() if c.dchg_cap_ah.notna().any() else 100.0
    cr = c.c_rate.iloc[0] if c.c_rate.notna().any() else 0.5
    feats = np.array([
        1.0,           # placeholder DCIR (matches most real cells which are also 1.0)
        float(q),
        float(cr),
        float(idx),
    ], dtype=np.float32)
    return CellData(
        cell_id=str(cid),
        is_clean=True,
        n_traj=n, soh_traj=s,
        x_health=torch.from_numpy(feats),
        soh_init=float(s[0]),
        n_total=int(n.max()),
    )


def load_combined(
    include_calb: bool = True,
    include_rept: bool = True,
    include_eve:  bool = True,
    include_synth: bool = True,
    synth_parquet: Path | str = SYNTH_PQ,
) -> tuple[list[CellData], dict]:
    """Load all requested cell pools with unified schema.

    Returns:
        cells: list of CellData instances
        meta:  dict recording origin per cell (make, is_synthetic, indicator)
    """
    cells: list[CellData] = []
    meta: dict = {}
    idx = 0

    if include_calb:
        canon = pd.read_parquet(CALB_PQ)
        for cid in CALB_CLEAN:
            c = _load_real_cell(canon, cid, "CALB", idx, 0)
            if c is None: continue
            cells.append(c)
            meta[c.cell_id] = dict(make="CALB", is_synthetic=False, idx=idx)
            idx += 1

    if include_rept:
        canon = pd.read_parquet(REPT_PQ)
        for cid in REPT_CLEAN:
            c = _load_real_cell(canon, cid, "REPT", idx, 0)
            if c is None: continue
            cells.append(c)
            meta[c.cell_id] = dict(make="REPT", is_synthetic=False, idx=idx)
            idx += 1

    if include_eve:
        canon = pd.read_parquet(EVE_PQ)
        for cid in EVE_CLEAN:
            c = _load_real_cell(canon, cid, "EVE", idx, 0)
            if c is None: continue
            cells.append(c)
            meta[c.cell_id] = dict(make="EVE", is_synthetic=False, idx=idx)
            idx += 1

    if include_synth and Path(synth_parquet).exists():
        canon = pd.read_parquet(synth_parquet)
        for cid in canon.cell_id.unique():
            c = _load_synth_cell(canon, cid, idx)
            if c is None: continue
            # Extract make from synth cell_id prefix
            make_key = str(cid).split("_")[0]  # e.g. CALB, EVE, REPT
            cells.append(c)
            meta[c.cell_id] = dict(make=f"SYNTH_{make_key}", is_synthetic=True, idx=idx)
            idx += 1

    return cells, meta


def feature_normaliser(cells: list[CellData]) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.stack([c.x_health for c in cells], dim=0)
    mean = X.mean(dim=0)
    std  = X.std(dim=0).clamp(min=1e-6)
    mean[-1] = 0.0; std[-1] = 1.0    # cell_indicator kept raw
    return mean, std


if __name__ == "__main__":
    cells, meta = load_combined()
    n_real = sum(1 for c in cells if not meta[c.cell_id]["is_synthetic"])
    n_synth = sum(1 for c in cells if meta[c.cell_id]["is_synthetic"])
    print(f"Loaded {len(cells)} cells total ({n_real} real + {n_synth} synthetic)")
    for c in cells[:8]:
        m = meta[c.cell_id]
        tag = f"[{m['make']}{'*' if m['is_synthetic'] else ''}]"
        print(f"  {tag:>18}  cell {c.cell_id:>15s}  N={c.n_total:>5}  "
              f"SoH {c.soh_init:.3f}->{float(c.soh_traj[-1]):.3f}")
    if n_synth > 0:
        print(f"  ... {n_synth} synthetic + more real")
