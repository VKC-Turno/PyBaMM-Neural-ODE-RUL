"""Extended feature pipeline — 12 per-cell characterisation features
mined from the early-life cycle-level canonical data.

Rationale: the Path B PINN misses cells 6, 19 at K=50 because they
have delayed dynamics (post-formation recovery, mid-life LAM) that
aren't visible in the first-50-cycle SoH slope. But characterisation
features like SoH curvature, IR drift, and coulombic-efficiency trend
CAN carry early signal about these dynamics. Feeding them to the
joint PINN gives the cell embedding more to work with.

Features (per cell, computed on the first K cycles or full early-life
window depending on availability):
1.  q_init_ah          — initial capacity
2.  dcir_start_mohm    — initial DCIR
3.  dcir_end_mohm      — DCIR at end of training window
4.  dcir_slope_pp      — normalised IR drift in K cycles
5.  ce_mean            — mean coulombic efficiency (chg/dchg) in K
6.  ce_slope           — CE trend across K
7.  c_rate_mean
8.  d_rate_mean
9.  dod_range          — dod_high - dod_low
10. soh_curvature      — quadratic coefficient of SoH vs cycle on K
11. soh_slope_first20  — slope of SoH over first 20 cycles (recovery
                          signal for cell 6)
12. cell_indicator     — categorical
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

RICH_KEYS = [
    "q_init_ah", "dcir_start_mohm", "dcir_end_mohm",
    "dcir_slope_pp", "ce_mean", "ce_slope",
    "c_rate_mean", "d_rate_mean", "dod_range",
    "soh_curvature", "soh_slope_first20", "cell_indicator",
]


@dataclass
class RichCellData:
    cell_id: int
    is_clean: bool
    n_traj: torch.Tensor
    soh_traj: torch.Tensor
    x_health: torch.Tensor       # (n_features,)
    soh_init: float
    n_total: int

    def split_at_K(self, k: int):
        first_cy = float(self.n_traj[0])
        m_tr = self.n_traj <= first_cy + k
        m_te = self.n_traj >  first_cy + k
        train = RichCellData(self.cell_id, self.is_clean, self.n_traj[m_tr],
                              self.soh_traj[m_tr], self.x_health,
                              self.soh_init, self.n_total)
        test  = RichCellData(self.cell_id, self.is_clean, self.n_traj[m_te],
                              self.soh_traj[m_te], self.x_health,
                              self.soh_init, self.n_total)
        return train, test


def _hampel(series: pd.Series, k: float = 3.0, window: int = 5) -> pd.Series:
    if len(series) < window:
        return pd.Series([True]*len(series), index=series.index)
    med = series.rolling(window, center=True, min_periods=1).median()
    mad = (series - med).abs().rolling(window, center=True, min_periods=1).median()
    return (series - med).abs() <= k * 1.4826 * mad.clip(lower=1e-9)


def _rich_features(c: pd.DataFrame, K: int = 100) -> dict:
    """Extract 12 features from the first K cycles of a cell's canonical data."""
    c = c.sort_values("global_cycle").reset_index(drop=True)
    n = c.global_cycle.to_numpy()
    first_cy = float(n[0])
    train = c[n <= first_cy + K]
    early = c[n <= first_cy + 20]

    # Basic capacity + IR
    q_init = float(train.dchg_cap_ah.dropna().iloc[:5].mean()) if train.dchg_cap_ah.notna().any() else 100.0
    if train.ir_ohm.notna().any():
        ir_early = float(train.ir_ohm.dropna().iloc[:5].mean()) * 1000  # to mOhm
        ir_end   = float(train.ir_ohm.dropna().iloc[-5:].mean()) * 1000
    else:
        ir_early = ir_end = 1.0
    ir_slope = (ir_end - ir_early) / max(len(train), 1)   # per cycle

    # Coulombic efficiency (chg/dchg per cycle) — needs both columns
    if train.chg_cap_ah.notna().any() and train.dchg_cap_ah.notna().any():
        ce = train.dchg_cap_ah.dropna() / train.chg_cap_ah.dropna()
        ce = ce[ce.between(0.7, 1.1)]
        ce_mean = float(ce.mean()) if len(ce) else 1.0
        if len(ce) > 5:
            ce_slope = float(np.polyfit(np.arange(len(ce)), ce.values, 1)[0])
        else:
            ce_slope = 0.0
    else:
        ce_mean, ce_slope = 1.0, 0.0

    # Rates + DoD
    c_rate  = float(train.c_rate.dropna().mean()) if train.c_rate.notna().any() else 0.5
    d_rate  = float(train.d_rate.dropna().mean()) if train.d_rate.notna().any() else 0.5
    dod_lo  = float(train.dod_low.dropna().mean()) if train.dod_low.notna().any() else 0.0
    dod_hi  = float(train.dod_high.dropna().mean()) if train.dod_high.notna().any() else 1.0
    dod_range = dod_hi - dod_lo

    # SoH shape features (the delayed-transient signal for cells 6, 19)
    if len(train) >= 5:
        cy_tr = train.global_cycle.to_numpy(dtype=float)
        soh_tr = train.soh.to_numpy(dtype=float)
        coeffs = np.polyfit(cy_tr - first_cy, soh_tr, 2)   # quadratic fit
        soh_curvature = float(coeffs[0])                    # 2nd-order coefficient
    else:
        soh_curvature = 0.0

    if len(early) >= 5:
        cy_e = early.global_cycle.to_numpy(dtype=float)
        soh_e = early.soh.to_numpy(dtype=float)
        soh_slope_20 = float(np.polyfit(cy_e - first_cy, soh_e, 1)[0])
    else:
        soh_slope_20 = 0.0

    return dict(
        q_init_ah=q_init,
        dcir_start_mohm=ir_early,
        dcir_end_mohm=ir_end,
        dcir_slope_pp=ir_slope * 1000,          # scale for stability
        ce_mean=ce_mean,
        ce_slope=ce_slope,
        c_rate_mean=c_rate,
        d_rate_mean=d_rate,
        dod_range=dod_range,
        soh_curvature=soh_curvature * 1e6,      # scale (typically 1e-6 range)
        soh_slope_first20=soh_slope_20 * 1000,  # scale
    )


def _load_one_rich(canon: pd.DataFrame, cid: int, K: int, skip_first: int = 1) -> RichCellData:
    is_clean = cid in CLEAN_IDS
    c = canon[canon.cell_id.astype(str).str.zfill(4) == f"{cid:04d}"].sort_values("global_cycle")
    c = c[(c.global_cycle >= skip_first + 1) & (c.soh > 0.05)].copy()
    keep = _hampel(c.soh * 100.0)
    c = c[keep].copy()

    feats = _rich_features(c, K=K)
    x = np.array([feats[k] for k in RICH_KEYS[:-1]] + [float(ALL_IDS.index(cid))],
                  dtype=np.float32)

    n = torch.tensor(c.global_cycle.to_numpy(dtype=np.float32))
    s = torch.tensor(c.soh.to_numpy(dtype=np.float32))
    return RichCellData(
        cell_id=cid, is_clean=is_clean,
        n_traj=n, soh_traj=s, x_health=torch.from_numpy(x),
        soh_init=float(s[0]), n_total=int(n.max()),
    )


def load_all_rich(cell_ids=None, K: int = 100) -> list[RichCellData]:
    if not CANON_PQ.exists():
        raise FileNotFoundError(CANON_PQ)
    canon = pd.read_parquet(CANON_PQ)
    ids = cell_ids or CLEAN_IDS
    return [_load_one_rich(canon, cid, K=K) for cid in ids]


def normaliser_rich(cells: list[RichCellData]) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.stack([c.x_health for c in cells], dim=0)
    mean = X.mean(dim=0)
    std  = X.std(dim=0).clamp(min=1e-6)
    mean[-1] = 0.0; std[-1] = 1.0    # cell_indicator kept raw
    return mean, std


if __name__ == "__main__":
    cells = load_all_rich(K=50)
    print(f"Loaded {len(cells)} cells with rich features (K=50 basis)")
    print(f"\nFeature keys: {RICH_KEYS}\n")
    for c in cells:
        vals = ", ".join(f"{v:+7.3f}" for v in c.x_health.tolist())
        print(f"  cell {c.cell_id:>2}  N={c.n_total:>4}  SoH {c.soh_init:.3f}->{float(c.soh_traj[-1]):.3f}   [{vals}]")
    mean, std = normaliser_rich(cells)
    print(f"\nMean: {[f'{v:+7.3f}' for v in mean.tolist()]}")
    print(f"Std : {[f'{v:+7.3f}' for v in std.tolist()]}")
