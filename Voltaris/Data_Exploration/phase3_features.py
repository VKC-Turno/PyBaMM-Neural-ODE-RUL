"""
Voltaris/Data_Exploration/phase3_features.py
--------------------------------------------
Assemble the Phase-3 operator-training dataset from a Phase-3 perturbation
corpus (see `configs/phase3_design.md`). One row per simulation:

    {anchor_id, sample_id, theta_norm[6], x_health[5], soh_traj[padded], n_cycles}

Reads
~~~~~
`corpus_dir/trajectories.parquet` — the concatenated per-cycle output from
`phase3_corpus.py` (mirrors `src/simulation/run_sweep.py`'s convention). Rows
carry at minimum the columns emitted by
`src/simulation/extract_features.per_cycle_features` plus a `sample_id` and
an `anchor_id`.

`configs/phase3_sweep.yaml` — anchor set + perturbation σ config. Used to
look up each anchor's fitted θ (unit space) so we can compute `theta_norm`
in σ-units around the anchor (log10 or linear per the sweep config).

Writes
~~~~~~
`configs/phase3_corpus/_dataset.parquet` (default) — one row per surviving
simulation, list-typed columns for the vector fields. Downstream
`SyntheticTrajectoryDataset` (or a Phase-3 sibling) can materialise
`(soh_traj, x_health, theta_norm)` batches directly from this.

Notes
~~~~~
- `x_health` mirrors `src/pinn/dataset.HEALTH_FEATURES` exactly so the
  existing `RULPredictor` normalisation stats stay compatible after the
  Phase-3 branch-input expansion (`x_health(5) + theta_norm(6)`).
- The θ names use the Phase-3 design canonical form (k_SEI, V_SEI,
  D_SEI_solvent, k_plating, LAM_neg_rate_s, LAM_pos_rate_s). Trajectory
  column names from `extract_features` are mapped via `_THETA_COLUMN_MAP`.
- `build_dataset_parquet` does NOT run PyBaMM — it only walks a corpus
  parquet that already exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Constants — the two vector schemas the operator consumes.
# ---------------------------------------------------------------------------
# Mirror src/pinn/dataset.HEALTH_FEATURES verbatim.
HEALTH_FEATURES: list[str] = [
    "temperature_C",
    "c_rate",
    "dcir_mOhm",
    "ic_peak1_shift_V",
    "ic_peak2_area_norm",
]

# Phase-3 canonical θ axis order (matches phase3_sweep.yaml perturbation_sigma).
THETA_FEATURES: list[str] = [
    "k_SEI",
    "V_SEI",
    "D_SEI_solvent",
    "k_plating",
    "LAM_neg_rate_s",
    "LAM_pos_rate_s",
]

# Map from canonical θ name → candidate column names that phase3_corpus.py /
# extract_features may emit for that axis. We accept the first hit.
#
# Bug fix (2026-07-10 adversarial audit): phase3_corpus writes columns with
# the `theta_` prefix (line ~680 of that file: `enriched[f"theta_{ax}"]`), so
# every entry MUST include a `theta_<name>` alias. Without it the extractor
# silently returned zero for every θ and the operator trained with a flat
# input — exactly the failure mode the previous Phase 3 hit.
_THETA_COLUMN_MAP: dict[str, tuple[str, ...]] = {
    "k_SEI":         ("theta_k_SEI", "k_SEI", "k_SEI_ms"),
    "V_SEI":         ("theta_V_SEI", "V_SEI",
                       "SEI_partial_molar_volume_m3mol"),
    "D_SEI_solvent": ("theta_D_SEI_solvent", "D_SEI_solvent",
                       "SEI_solvent_diffusivity_m2s", "D_SEI_solvent_m2s"),
    "k_plating":     ("theta_k_plating", "k_plating", "k_plating_ms",
                       "lithium_plating_exchange_current_A_m2"),
    # phase3_corpus uses `k_LAM_negative` as the LAM-neg axis name; features
    # canonicalises to `LAM_neg_rate_s`. Accept the corpus name too.
    "LAM_neg_rate_s": ("theta_k_LAM_negative", "theta_LAM_neg_rate_s",
                        "LAM_neg_rate_s", "LAM_negative_rate_s"),
    "LAM_pos_rate_s": ("theta_LAM_pos_rate_s", "theta_k_LAM_positive",
                        "LAM_pos_rate_s", "LAM_positive_rate_s"),
}

# Fallback perturbation σ, used only if the sweep config can't be read.
# Values match configs/phase3_sweep.yaml perturbation_sigma.
_DEFAULT_SIGMA: dict[str, dict[str, Any]] = {
    "k_SEI":         {"space": "log10",  "sigma_dec": 0.6},
    "V_SEI":         {"space": "linear", "sigma_rel": 0.15},
    "D_SEI_solvent": {"space": "log10",  "sigma_dec": 0.7},
    "k_plating":     {"space": "log10",  "sigma_dec": 0.5},
    "LAM_neg_rate_s": {"space": "log10", "sigma_dec": 0.8},
    "LAM_pos_rate_s": {"space": "log10", "sigma_dec": 0.3},
}

_SWEEP_CONFIG_DEFAULT = Path("configs/phase3_sweep.yaml")
_DATASET_OUT_DEFAULT = Path("configs/phase3_corpus/_dataset.parquet")


# ---------------------------------------------------------------------------
# Public API — x_health, soh_traj, theta_norm
# ---------------------------------------------------------------------------
import re as _re

_PROTOCOL_RE = _re.compile(r"^[A-Z]+_([\d.]+)C_([\d.]+)D_(\d+)_(\d+)$")
_BOL_PARAMS_DIR = Path("configs/bol_params")
_BOL_DCIR_CACHE: dict[str, float] = {}


def _parse_c_rate(protocol_id: str) -> float:
    """Parse the C-rate from a protocol_id like 'CALB_0.5C_0.5D_15_100'.
    Returns NaN if the id doesn't match the canonical format."""
    if not protocol_id or not isinstance(protocol_id, str):
        return float("nan")
    m = _PROTOCOL_RE.match(protocol_id)
    return float(m.group(1)) if m else float("nan")


def _bol_dcir_mohm(anchor_id: str) -> float:
    """Load the anchor cell's BOL DCIR (R0) in milliohms from its
    Phase-1 bol_params yaml. Cached per anchor_id. NaN if missing."""
    if anchor_id in _BOL_DCIR_CACHE:
        return _BOL_DCIR_CACHE[anchor_id]
    p = _BOL_PARAMS_DIR / f"{anchor_id}.yaml"
    if not p.exists():
        _BOL_DCIR_CACHE[anchor_id] = float("nan")
        return float("nan")
    try:
        doc = yaml.safe_load(p.read_text()) or {}
        r0_ohm = float(doc.get("resistance", {}).get("R0_Ohm", "nan"))
        val = r0_ohm * 1000.0 if np.isfinite(r0_ohm) else float("nan")
    except Exception:  # noqa: BLE001
        val = float("nan")
    _BOL_DCIR_CACHE[anchor_id] = val
    return val


def extract_x_health(trajectory_df: pd.DataFrame,
                     anchor_theta: dict | None = None,
                     ambient_C: float = 25.0) -> np.ndarray:
    """
    Per-simulation health fingerprint (5-D), mirroring
    `src/pinn/dataset._compute_health_features`.

    Fields:
      0 temperature_C         ambient (25 °C by cohort convention)
      1 c_rate                cycling C-rate from the anchor's protocol_id
      2 dcir_mOhm             BOL R0 from configs/bol_params/{anchor}.yaml
      3 ic_peak1_shift_V      dQ/dV peak-1 shift vs cycle 1 (= 0 by defn)
      4 ic_peak2_area_norm    peak-2 area normalised to cycle 1 (= 1 by defn)

    Bug fix (2026-07-13): previous version tried `first.get("c_rate")` and
    `first.get("dcir_mOhm")` but the raw sweep parquets carry neither column
    (only `protocol_id`); every sample got NaN → 0 for both fields, so the
    operator saw x_health = [25, 0, 0, 0, 1] for every one of 489 training
    samples. Fixed to parse c_rate from protocol_id and load DCIR from the
    anchor's Phase-1 BOL yaml.
    """
    if trajectory_df.empty:
        return np.full(len(HEALTH_FEATURES), np.nan, dtype=np.float32)

    df = trajectory_df.sort_values("cycle_n").reset_index(drop=True)
    first = df.iloc[0]

    # c_rate from protocol_id (cohort convention: MAKE_{c}C_{d}D_{lo}_{hi}).
    c_rate = _parse_c_rate(str(first.get("protocol_id", "")))
    # DCIR from anchor BOL yaml (per-anchor constant).
    anchor_id = str(first.get("anchor_id", "")) or ""
    dcir_mOhm = _bol_dcir_mohm(anchor_id) if anchor_id else float("nan")

    x = np.array([
        ambient_C,
        c_rate,
        dcir_mOhm,
        0.0,   # ic_peak1_shift_V — zero at cycle 1 by construction
        1.0,   # ic_peak2_area_norm — normalised to itself at cycle 1
    ], dtype=np.float32)
    return np.where(np.isfinite(x), x, 0.0).astype(np.float32)


def extract_soh_trajectory(trajectory_df: pd.DataFrame) -> np.ndarray:
    """SoH-per-cycle, sorted by cycle_n. Returns np.float32."""
    if trajectory_df.empty:
        return np.array([], dtype=np.float32)
    df = (trajectory_df[["cycle_n", "SOH"]]
          .dropna(subset=["SOH"])
          .sort_values("cycle_n")
          .drop_duplicates("cycle_n", keep="first"))
    return df["SOH"].to_numpy(dtype=np.float32)


def extract_theta_norm(trajectory_df: pd.DataFrame,
                       anchor_theta: dict,
                       perturbation_sigma: dict | None = None) -> np.ndarray:
    """
    6-element θ vector normalised into σ-units around the anchor's fitted θ.

    log10 axes:   theta_norm = (log10(θ) − log10(θ_anchor)) / σ_dec
    linear axes:  theta_norm = (θ − θ_anchor) / (σ_rel · θ_anchor)

    Missing / non-finite values → 0.0 (anchor centre, standardised-space).
    """
    if perturbation_sigma is None:
        perturbation_sigma = _DEFAULT_SIGMA
    if trajectory_df.empty:
        return np.zeros(len(THETA_FEATURES), dtype=np.float32)

    first = trajectory_df.sort_values("cycle_n").iloc[0]
    out = np.zeros(len(THETA_FEATURES), dtype=np.float32)

    for i, name in enumerate(THETA_FEATURES):
        val = _first_present(first, _THETA_COLUMN_MAP[name])
        if val is None or not np.isfinite(val):
            continue
        anchor_val = anchor_theta.get(name)
        if anchor_val is None or not np.isfinite(anchor_val):
            continue
        sigma = perturbation_sigma.get(name, _DEFAULT_SIGMA[name])
        space = sigma.get("space", "log10")
        if space == "log10":
            sd = float(sigma.get("sigma_dec", 1.0))
            if val <= 0 or anchor_val <= 0 or sd <= 0:
                continue
            out[i] = (np.log10(val) - np.log10(anchor_val)) / sd
        else:  # linear
            sr = float(sigma.get("sigma_rel", 1.0))
            denom = sr * float(anchor_val)
            if denom == 0.0:
                continue
            out[i] = (float(val) - float(anchor_val)) / denom
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Corpus assembly
# ---------------------------------------------------------------------------
@dataclass
class _SimRow:
    anchor_id: str
    sample_id: str
    theta_norm: np.ndarray
    x_health: np.ndarray
    soh_traj: np.ndarray


def build_dataset_parquet(corpus_dir: Path | str,
                          out_path: Path | str = _DATASET_OUT_DEFAULT,
                          sweep_config_path: Path | str = _SWEEP_CONFIG_DEFAULT,
                          min_cycles: int = 5,
                          pad_value: float = np.nan) -> Path:
    """
    Walk `corpus_dir/trajectories.parquet` and materialise a per-sim
    training row. Writes to `out_path` (default
    `configs/phase3_corpus/_dataset.parquet`).

    Returns the resolved output path.
    """
    corpus_dir = Path(corpus_dir)
    out_path = Path(out_path)
    sweep_config_path = Path(sweep_config_path)

    traj_path = corpus_dir / "trajectories.parquet"
    if not traj_path.exists():
        raise FileNotFoundError(
            f"Corpus trajectories.parquet not found at {traj_path}. "
            f"Run phase3_corpus.py first (or its smoke)."
        )

    df = pd.read_parquet(traj_path)
    if "sample_id" not in df.columns:
        raise ValueError(f"{traj_path} missing required column 'sample_id'")
    if "anchor_id" not in df.columns:
        # phase3_corpus is expected to inject anchor_id per sim; fall back to a
        # per-sample manifest.yaml lookup if present, else fail loudly.
        anchor_map = _load_sample_anchor_map(corpus_dir)
        if not anchor_map:
            raise ValueError(
                f"{traj_path} missing 'anchor_id' and no sample→anchor map "
                f"found in {corpus_dir}/manifest.yaml"
            )
        df = df.copy()
        df["anchor_id"] = df["sample_id"].map(anchor_map)

    anchors, sigma = _load_anchors_and_sigma(sweep_config_path)

    # Assemble one row per (anchor_id, sample_id).
    rows: list[_SimRow] = []
    for (aid, sid), g in df.groupby(["anchor_id", "sample_id"], sort=False):
        if len(g) < min_cycles:
            continue
        anchor_theta = anchors.get(str(aid), {})
        x = extract_x_health(g, anchor_theta=anchor_theta)
        th = extract_theta_norm(g, anchor_theta=anchor_theta,
                                perturbation_sigma=sigma)
        soh = extract_soh_trajectory(g)
        if soh.size == 0:
            continue
        rows.append(_SimRow(str(aid), str(sid), th, x, soh))

    if not rows:
        raise ValueError(
            f"No usable simulations after filtering (min_cycles={min_cycles})."
        )

    max_len = max(int(r.soh_traj.size) for r in rows)
    records: list[dict[str, Any]] = []
    for r in rows:
        pad = np.full(max_len, pad_value, dtype=np.float32)
        pad[: r.soh_traj.size] = r.soh_traj
        rec = {
            "anchor_id":  r.anchor_id,
            "sample_id":  r.sample_id,
            "theta_norm": r.theta_norm.tolist(),
            "x_health":   r.x_health.tolist(),
            "soh_traj":   pad.tolist(),
            "n_cycles":   int(r.soh_traj.size),
        }
        # Bug fix (2026-07-10 adversarial audit): phase3_train_val._load_corpus_parquet
        # expects six flat `theta_norm_<axis>` columns, not the list column. Emit
        # both so downstream code doesn't care which convention it uses.
        for j, name in enumerate(THETA_FEATURES):
            rec[f"theta_norm_{name}"] = float(r.theta_norm[j])
        records.append(rec)

    # Bug fix (2026-07-10): guardrail — the whole point of Phase 3 is that θ
    # varies across the corpus. If every sample has theta_norm ≈ [0,0,0,0,0,0]
    # (the exact failure mode we're trying to prevent), the operator would
    # train with zero θ signal. Refuse to emit a silently-broken dataset.
    theta_arr = np.stack([r.theta_norm for r in rows], axis=0)
    theta_std = float(theta_arr.std())
    if theta_std < 1e-6:
        raise ValueError(
            f"Assembled theta_norm has near-zero variance across the corpus "
            f"(std={theta_std:.2e}). Column-name mismatch between "
            f"phase3_corpus.py (writes theta_<axis>) and phase3_features "
            f"(_THETA_COLUMN_MAP). Reject to avoid the previous Phase 3 "
            f"'flat SoH' failure mode."
        )

    out_df = pd.DataFrame.from_records(records)
    out_df.attrs["theta_features"] = THETA_FEATURES
    out_df.attrs["health_features"] = HEALTH_FEATURES
    out_df.attrs["padded_length"] = int(max_len)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _first_present(row: pd.Series, candidates: Iterable[str]) -> Optional[float]:
    for c in candidates:
        if c in row.index:
            v = row[c]
            if pd.notna(v):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
    return None


def _load_anchors_and_sigma(sweep_config_path: Path) -> tuple[dict, dict]:
    """Return (anchors_by_id, perturbation_sigma) from phase3_sweep.yaml."""
    if not sweep_config_path.exists():
        return {}, _DEFAULT_SIGMA
    with open(sweep_config_path) as f:
        cfg = yaml.safe_load(f) or {}
    anchors: dict[str, dict[str, float]] = {}
    for a in cfg.get("anchors", []) or []:
        aid = a.get("id")
        if aid is None:
            continue
        theta = dict(a.get("fitted_theta") or {})
        # Map YAML key `k_LAM_negative` → canonical `LAM_neg_rate_s`, etc.
        alias = {
            "k_LAM_negative": "LAM_neg_rate_s",
            "k_LAM_positive": "LAM_pos_rate_s",
        }
        for src, dst in alias.items():
            if src in theta and dst not in theta:
                theta[dst] = theta.pop(src)
        anchors[str(aid)] = {k: float(v) for k, v in theta.items()
                             if v is not None}
    sigma = cfg.get("perturbation_sigma") or _DEFAULT_SIGMA
    return anchors, sigma


def _load_sample_anchor_map(corpus_dir: Path) -> dict[str, str]:
    mpath = corpus_dir / "manifest.yaml"
    if not mpath.exists():
        return {}
    with open(mpath) as f:
        m = yaml.safe_load(f) or {}
    out: dict[str, str] = {}
    for sim in m.get("sims", []) or []:
        sid = sim.get("sample_id")
        aid = sim.get("anchor_id")
        if sid is not None and aid is not None:
            out[str(sid)] = str(aid)
    return out


# ---------------------------------------------------------------------------
# Smoke — no PyBaMM. Fabricates a 2-sim corpus consistent with what
# phase3_corpus.py's own smoke is expected to emit, then runs the full
# extraction pipeline end-to-end.
# ---------------------------------------------------------------------------
def _smoke(tmp_root: Path | None = None) -> Path:
    """
    Build a mock 2-sim corpus in a temp dir, assemble the dataset, and
    sanity-check the shapes. Prints a one-line PASS + returns the output
    parquet path.
    """
    import shutil
    import tempfile

    tmp_root = Path(tmp_root or tempfile.mkdtemp(prefix="phase3_features_smoke_"))
    corpus_dir = tmp_root / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # Pull two anchors from configs/phase3_sweep.yaml so theta_norm is
    # exercised against a real fitted_theta record.
    anchors, sigma = _load_anchors_and_sigma(_SWEEP_CONFIG_DEFAULT)
    if len(anchors) < 2:
        raise RuntimeError(
            "phase3_sweep.yaml missing anchors; smoke needs at least 2."
        )
    aid_a, aid_b = list(anchors.keys())[:2]
    theta_a, theta_b = anchors[aid_a], anchors[aid_b]

    n_cycles = 40
    cycles = np.arange(1, n_cycles + 1)
    # Two mock trajectories: linear-ish SoH fade, different anchor and slight θ perturbation.
    def _mk(anchor_id: str, sample_id: str, theta_anchor: dict,
            fade_slope: float) -> pd.DataFrame:
        soh = 1.0 - fade_slope * (cycles - 1) / n_cycles
        # Perturb θ by +0.3 σ on k_SEI (log10 axis) to give theta_norm a signal.
        k_sei = theta_anchor.get("k_SEI", 1e-11) * (10 ** 0.3)
        v_sei = theta_anchor.get("V_SEI", 1e-4) * 1.05
        d_sei = theta_anchor.get("D_SEI_solvent", 1e-21) * (10 ** -0.2)
        k_pl  = theta_anchor.get("k_plating",     1e-12) * (10 ** 0.1)
        k_lam_n = theta_anchor.get("LAM_neg_rate_s", 1e-9) * (10 ** 0.4)
        k_lam_p = theta_anchor.get("LAM_pos_rate_s", 1e-9) * (10 ** 0.05) \
            if "LAM_pos_rate_s" in theta_anchor else 1e-9

        return pd.DataFrame({
            "cycle_n": cycles.astype(int),
            "Q_Ah":    72.0 * soh,
            "SOH":     soh.astype(np.float32),
            "V_mean_discharge": np.full(n_cycles, 3.25, dtype=np.float32),
            "dcir_mOhm":        np.full(n_cycles, 1.74, dtype=np.float32),
            "ic_peak1_V":       np.full(n_cycles, 3.30, dtype=np.float32),
            "ic_peak2_V":       np.full(n_cycles, 3.42, dtype=np.float32),
            "ic_peak1_area":    np.full(n_cycles, 12.0, dtype=np.float32),
            "ic_peak2_area":    np.full(n_cycles, 8.0,  dtype=np.float32),
            "T_K":              np.full(n_cycles, 298.15, dtype=np.float32),
            "c_rate":           np.full(n_cycles, 0.5, dtype=np.float32),
            "k_SEI_ms":                             k_sei,
            "SEI_partial_molar_volume_m3mol":       v_sei,
            "D_SEI_solvent":                        d_sei,
            "lithium_plating_exchange_current_A_m2": k_pl,
            "LAM_negative_rate_s":                  k_lam_n,
            "LAM_positive_rate_s":                  k_lam_p,
            "temperature_K":                        298.15,
            "sample_id":                            sample_id,
            "anchor_id":                            anchor_id,
        })

    df_a = _mk(aid_a, "s00000", theta_a, fade_slope=0.15)
    df_b = _mk(aid_b, "s00001", theta_b, fade_slope=0.25)
    pd.concat([df_a, df_b], ignore_index=True).to_parquet(
        corpus_dir / "trajectories.parquet"
    )

    out_path = corpus_dir / "_dataset.parquet"
    build_dataset_parquet(corpus_dir, out_path=out_path,
                          sweep_config_path=_SWEEP_CONFIG_DEFAULT,
                          min_cycles=5)

    out = pd.read_parquet(out_path)
    assert len(out) == 2, f"expected 2 rows, got {len(out)}"
    for _, r in out.iterrows():
        assert len(r["theta_norm"]) == 6, "theta_norm must be length-6"
        assert len(r["x_health"]) == 5,   "x_health must be length-5"
        assert len(r["soh_traj"]) == n_cycles, "soh_traj must be padded to max_len"
        assert r["n_cycles"] == n_cycles, "n_cycles mismatch"
    print(f"[phase3_features] smoke PASS "
          f"({len(out)} rows, padded_len={n_cycles}, out={out_path})")
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sm = sub.add_parser("smoke", help="Run the 2-sim smoke.")
    sm.add_argument("--tmp-root", type=Path, default=None,
                    help="Optional temp directory (defaults to tempfile.mkdtemp).")

    bd = sub.add_parser("build", help="Assemble _dataset.parquet from a corpus.")
    bd.add_argument("--corpus-dir", type=Path, required=True)
    bd.add_argument("--out-path", type=Path, default=_DATASET_OUT_DEFAULT)
    bd.add_argument("--sweep-config", type=Path, default=_SWEEP_CONFIG_DEFAULT)

    args = ap.parse_args()
    if args.cmd == "smoke":
        _smoke(args.tmp_root)
    elif args.cmd == "build":
        p = build_dataset_parquet(args.corpus_dir, args.out_path,
                                  sweep_config_path=args.sweep_config)
        print(f"[phase3_features] wrote {p}")
