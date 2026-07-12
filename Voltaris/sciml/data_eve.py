"""EVE canonical loader.

EVE cohort in this dataset is smallest (n=4 usable cells, 150 cycles each,
1-2 pp total fade). Includes for make-agnostic architecture test.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from .data import CellData, _hampel_filter

CANON_PQ = Path("/home/hj/Desktop/PINNs/soh/data/canonical/eve.parquet")

# Cells with real fade signal + at least 100 cycles
EVE_IDS = ["0002", "0003", "0004", "0008"]


def _load_one_eve(canon, cid, skip_first=1):
    c = canon[canon.cell_id.astype(str) == cid].sort_values("global_cycle")
    c = c[(c.global_cycle >= skip_first + 1) & (c.soh > 0.05)].copy()
    keep = _hampel_filter(c.soh * 100.0)
    c = c[keep].copy()

    n = torch.tensor(c.global_cycle.to_numpy(dtype=np.float32))
    s = torch.tensor(c.soh.to_numpy(dtype=np.float32))

    ir = c.ir_ohm.dropna().iloc[:5].mean() if c.ir_ohm.notna().any() else 0.001
    q  = c.dchg_cap_ah.dropna().iloc[:5].mean() if c.dchg_cap_ah.notna().any() else 100.0
    cr = c.c_rate.dropna().mean() if c.c_rate.notna().any() else 0.5
    feats = np.array([
        float(ir) * 1000.0,
        float(q),
        float(cr),
        float(EVE_IDS.index(cid)),
    ], dtype=np.float32)

    return CellData(
        cell_id=int(cid), is_clean=True,
        n_traj=n, soh_traj=s,
        x_health=torch.from_numpy(feats),
        soh_init=float(s[0]),
        n_total=int(n.max()),
    )


def load_eve(cell_ids=None):
    canon = pd.read_parquet(CANON_PQ)
    ids = cell_ids or EVE_IDS
    return [_load_one_eve(canon, cid) for cid in ids]


def normaliser_eve(cells):
    X = torch.stack([c.x_health for c in cells], dim=0)
    mean = X.mean(dim=0)
    std  = X.std(dim=0).clamp(min=1e-6)
    mean[-1] = 0.0; std[-1] = 1.0
    return mean, std


if __name__ == "__main__":
    cells = load_eve()
    print(f"Loaded {len(cells)} EVE cells")
    for c in cells:
        print(f"  cell {c.cell_id:>2}  N={c.n_total:>3}  "
              f"SoH {c.soh_init:.3f} -> {float(c.soh_traj[-1]):.3f}   "
              f"features {c.x_health.tolist()}")
