"""REPT canonical loader — make-agnostic test.

REPT trajectories are ~200 cycles with 2-3 pp SoH fade — much shorter
and shallower than CALB (1000-1500 cycles, 15-25 pp fade). This isn't
a direct second-life comparison, but validates that the same PINN
architecture handles a different manufacturer's data regime.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from .data import CellData, _hampel_filter

CANON_PQ = Path("/home/hj/Desktop/PINNs/soh/data/canonical/rept.parquet")

# Cells with real fade signal (start ≥ 0.7, ≥ 1 pp fade, ≥ 200 cycles)
REPT_IDS = ["0001", "0003", "0007", "0028", "0043",
             "0057", "0074", "0078", "0087"]


def _load_one_rept(canon: pd.DataFrame, cid: str, skip_first: int = 1) -> CellData:
    c = canon[canon.cell_id.astype(str) == cid].sort_values("global_cycle")
    c = c[(c.global_cycle >= skip_first + 1) & (c.soh > 0.05)].copy()
    keep = _hampel_filter(c.soh * 100.0)
    c = c[keep].copy()

    n = torch.tensor(c.global_cycle.to_numpy(dtype=np.float32))
    s = torch.tensor(c.soh.to_numpy(dtype=np.float32))

    # Same 4-feature basic vector used by Path B on CALB
    ir = c.ir_ohm.dropna().iloc[:5].mean() if c.ir_ohm.notna().any() else 0.001
    q  = c.dchg_cap_ah.dropna().iloc[:5].mean() if c.dchg_cap_ah.notna().any() else 100.0
    cr = c.c_rate.dropna().mean() if c.c_rate.notna().any() else 0.5
    feats = np.array([
        float(ir) * 1000.0,
        float(q),
        float(cr),
        float(REPT_IDS.index(cid)),
    ], dtype=np.float32)

    return CellData(
        cell_id=int(cid),
        is_clean=True,
        n_traj=n, soh_traj=s,
        x_health=torch.from_numpy(feats),
        soh_init=float(s[0]),
        n_total=int(n.max()),
    )


def load_rept(cell_ids=None) -> list[CellData]:
    canon = pd.read_parquet(CANON_PQ)
    ids = cell_ids or REPT_IDS
    return [_load_one_rept(canon, cid) for cid in ids]


def normaliser_rept(cells):
    X = torch.stack([c.x_health for c in cells], dim=0)
    mean = X.mean(dim=0)
    std  = X.std(dim=0).clamp(min=1e-6)
    mean[-1] = 0.0; std[-1] = 1.0
    return mean, std


if __name__ == "__main__":
    cells = load_rept()
    print(f"Loaded {len(cells)} REPT cells")
    for c in cells:
        print(f"  cell {c.cell_id:>4}  N={c.n_total:>4}  SoH {c.soh_init:.3f} -> "
              f"{float(c.soh_traj[-1]):.3f}  features {c.x_health.tolist()}")
