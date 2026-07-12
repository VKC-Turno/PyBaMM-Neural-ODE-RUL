"""Build training tensors for the theta-conditioned DeepONet from the
existing PyBaMM sweep output (data/synthetic/trajectories.parquet).

Two-stage plan:
  Stage A - synthetic pretraining:
      Each row of trajectories.parquet is one (sim, cycle_n) sample.
      Group by sample_id => one full trajectory per sim.
      Extract:
          Stream 1 (DCIR fingerprint)   - derived from first-cycle features
                                            (dcir_mOhm at t=0)
          Stream 2 (RPT fingerprint)    - Q_Ah at cycle 1, IC peak areas,
                                            V_mean, OCV span proxy
          Stream 3 (soh_early K=50)     - measured SoH over first K cycles
          Stream 4 (theta_vec)          - 5 swept params + 5 identified BOL
                                            (BOL comes from identified_params.yaml
                                            or defaults if missing)
          Stream 5 (protocol)           - c_rate, DoD_default, T, rest_min
      Target: SoH(n) for n = K+1 ... N_final

  Stage B - real cell fine-tune:
      A separate small dataset from measured CSVs. Not built yet.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


@dataclass
class DatasetConfig:
    trajectories_path: Path = Path("/home/hj/Desktop/PINNs/data/synthetic/trajectories.parquet")
    identified_params_path: Path = Path("/home/hj/Desktop/PINNs/configs/identified_params.yaml")
    K: int          = 50           # length of early-window
    n_query: int    = 30           # number of query cycles per trajectory
    min_traj_cycles: int = 60      # skip sims that didn't run long enough
    seed: int       = 42


def _load_identified_theta_bol() -> dict:
    """Return the 5 BOL identifiers we use as fixed θ tail: x_100, y_100,
    Q_n_init, R0_Ohm, C1_F.  If the yaml is missing keys, use safe defaults."""
    p = Path("/home/hj/Desktop/PINNs/configs/identified_params.yaml")
    if not p.exists():
        return dict(x_100=0.88, y_100=0.01, Q_n_init=138.0, R0_Ohm=1.7e-3, C1_F=2.4e4)
    d = yaml.safe_load(p.read_text()) or {}
    stoich = d.get("stoichiometry", {}) or {}
    cap = d.get("capacity", {}) or {}
    res = d.get("resistance", {}) or {}
    return dict(
        x_100=float(stoich.get("x_100", 0.88)),
        y_100=float(stoich.get("y_100", 0.01)),
        Q_n_init=float(cap.get("Q_n_init_Ah", 138.0)),
        R0_Ohm=float(res.get("R0_Ohm", 1.7e-3)),
        C1_F=float(res.get("C1_F", 2.4e4)),
    )


def _safe(x, default: float = 0.0) -> float:
    """NaN-safe float coercion."""
    try:
        v = float(x)
        return v if not np.isnan(v) else default
    except (TypeError, ValueError):
        return default


def _extract_rpt_fp_from_cycle1(row: pd.Series) -> np.ndarray:
    """RPT features derived from first-cycle synthetic outputs."""
    return np.array([
        _safe(row.get("Q_Ah")),
        _safe(row.get("Q_Ah")),
        0.0,  # delta_q_over_delta_v placeholder
        _safe(row.get("ic_peak1_area")),
        _safe(row.get("ic_peak2_area")),
        _safe(row.get("V_mean_discharge")),
    ], dtype=np.float32)


def _extract_dcir_fp_from_cycle1(row: pd.Series) -> np.ndarray:
    """9-vector DCIR fingerprint. In the current synthetic corpus, per-cycle
    dcir_mOhm is NaN (never populated by the feature extractor), so this
    reduces to a constant zero stream — carries no signal until real DCIR
    parsing is wired up in extract_features.py."""
    r = _safe(row.get("dcir_mOhm"), 0.0)
    return np.full(9, r, dtype=np.float32)


def _extract_theta_vec(sample_params: dict, bol: dict) -> np.ndarray:
    """5 swept degradation params + 5 identified BOL params.

    Log-transform the sweep params (they span many decades) so downstream
    standardisation lands each dimension in ~[-1, 1]. Add a small floor
    to avoid log(0) when a param is missing.
    """
    def logify(x, floor=1e-20):
        return float(np.log10(max(abs(x), floor)))
    return np.array([
        logify(sample_params.get("k_SEI_ms",                              0.0)),
        logify(sample_params.get("SEI_partial_molar_volume_m3mol",        0.0)),
        logify(sample_params.get("lithium_plating_exchange_current_A_m2", 0.0)),
        logify(sample_params.get("LAM_positive_rate_s",                   0.0)),
        logify(sample_params.get("LAM_negative_rate_s",                   0.0)),
        bol["x_100"],
        bol["y_100"],
        bol["Q_n_init"],
        logify(bol["R0_Ohm"]),
        logify(bol["C1_F"]),
    ], dtype=np.float32)


def _extract_protocol(sample_params: dict) -> np.ndarray:
    """c_rate, DoD, temperature, rest_time_min. DoD hardcoded to 1.0
    since the sweep protocol always discharges to 2.5 V (full DoD).
    Rest = 10 minutes (from sweep_config)."""
    return np.array([
        float(sample_params.get("c_rate", 0.5)),
        1.0,
        float(sample_params.get("temperature_K", 298.15)),
        10.0,
    ], dtype=np.float32)


def _fit_standardiser(samples: list[dict], keys: tuple[str, ...]) -> dict:
    """Return per-key (mean, std) computed across the sample list."""
    stats = {}
    for k in keys:
        stacked = np.stack([s[k] for s in samples], axis=0).astype(np.float64)
        mean = stacked.mean(axis=0)
        std  = stacked.std(axis=0)
        std  = np.where(std < 1e-8, 1.0, std)     # avoid divide-by-zero
        stats[k] = (mean.astype(np.float32), std.astype(np.float32))
    return stats


class SyntheticTrajectoryDataset(torch.utils.data.Dataset):
    """One sample = one full sim trajectory.

    Emits a dict with keys expected by ThetaDeepONet.forward + loss_fn:
      dcir_fp     (9,)
      rpt_fp      (6,)
      soh_early   (K,)
      theta_vec   (10,)
      protocol    (4,)
      n_query     (Nq,)  cycle indices to be queried
      soh_target  (Nq,)  ground-truth SoH at those cycles
      soh_init    scalar (SoH at cycle 1)
    """

    def __init__(self, cfg: DatasetConfig,
                 recovered_params_csv: str | Path | None = None):
        self.cfg = cfg
        self.bol = _load_identified_theta_bol()
        traj = pd.read_parquet(cfg.trajectories_path)
        # Ensure LAM_negative_rate_s column exists — merge from recovered_params
        # (older sweeps dropped this column; new sweeps persist it).
        if "LAM_negative_rate_s" not in traj.columns:
            rp = pd.read_csv(recovered_params_csv or
                             "/home/hj/Desktop/PINNs/data/synthetic/recovered_sample_params.csv")
            traj = traj.merge(rp[["sample_id", "LAM_negative_rate_s"]],
                              on="sample_id", how="left")

        # Group by sample_id and filter usable
        self.samples: list[dict] = []
        for sid, sub in traj.groupby("sample_id"):
            sub = sub.sort_values("cycle_n").reset_index(drop=True)
            if len(sub) < cfg.min_traj_cycles or len(sub) < cfg.K + 5:
                continue
            first = sub.iloc[0]
            params = dict(
                k_SEI_ms=first.get("k_SEI_ms", 0.0),
                SEI_partial_molar_volume_m3mol=first.get("SEI_partial_molar_volume_m3mol", 0.0),
                lithium_plating_exchange_current_A_m2=first.get(
                    "lithium_plating_exchange_current_A_m2", 0.0),
                LAM_positive_rate_s=first.get("LAM_positive_rate_s", 0.0),
                LAM_negative_rate_s=first.get("LAM_negative_rate_s", 0.0),
                temperature_K=first.get("temperature_K", 298.15),
                c_rate=first.get("c_rate", 0.5),
            )
            self.samples.append(dict(
                sample_id=sid,
                dcir_fp=_extract_dcir_fp_from_cycle1(first),
                rpt_fp=_extract_rpt_fp_from_cycle1(first),
                theta_vec=_extract_theta_vec(params, self.bol),
                protocol=_extract_protocol(params),
                cycles=sub["cycle_n"].to_numpy(np.float32),
                soh=sub["SOH"].to_numpy(np.float32),
            ))
        self.rng = np.random.default_rng(cfg.seed)

        # Fit standardisation stats across all samples (for the fixed streams)
        self.stats = _fit_standardiser(
            self.samples, ("dcir_fp", "rpt_fp", "theta_vec", "protocol"))
        for s in self.samples:
            for k, (m, sd) in self.stats.items():
                s[k] = ((s[k] - m) / sd).astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        cycles = s["cycles"]; soh = s["soh"]
        K = self.cfg.K
        soh_early = soh[:K]

        # Sample n_query points uniformly from (K, end]
        end = len(cycles)
        available = np.arange(K, end)
        if len(available) < self.cfg.n_query:
            # Pad by resampling with replacement
            idxs = self.rng.choice(available, size=self.cfg.n_query, replace=True)
        else:
            idxs = self.rng.choice(available, size=self.cfg.n_query, replace=False)
        idxs = np.sort(idxs)

        return dict(
            dcir_fp    = torch.from_numpy(s["dcir_fp"]),
            rpt_fp     = torch.from_numpy(s["rpt_fp"]),
            soh_early  = torch.from_numpy(soh_early),
            theta_vec  = torch.from_numpy(s["theta_vec"]),
            protocol   = torch.from_numpy(s["protocol"]),
            n_query    = torch.from_numpy(cycles[idxs]),
            soh_target = torch.from_numpy(soh[idxs]),
            soh_init   = torch.tensor(float(soh[0])),
        )


def build_dataset(cfg: DatasetConfig | None = None) -> SyntheticTrajectoryDataset:
    return SyntheticTrajectoryDataset(cfg or DatasetConfig())
