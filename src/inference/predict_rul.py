"""
src/inference/predict_rul.py
----------------------------
End-to-end RUL inference pipeline.

For a given cell + current SOH + cycle count + health-feature vector, the
predictor returns:

    rul_mean            point estimate of cycles to EOL
    rul_p5 / rul_p95    90 % CI from MC-dropout sampling
    soh_trajectory      mean predicted SOH(n) curve
    dominant_mechanism  qualitative driver (SEI / LAM / resistance)
    health_features     the input feature vector for traceability

Usage:
    .venv/bin/python -m src.inference.predict_rul \
        --cell-id 0005 --soh-now 0.92 --cycle-now 200
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.inference.health_features import (  # noqa: E402
    HEALTH_FEATURES, HealthFeatures, extract_for_cell,
)
from src.pinn.model import RULPredictor  # noqa: E402


DEFAULT_CKPT = Path("outputs/models/pinn_finetuned.pt")
RESULTS_DIR = Path("outputs/results")


def load_model(ckpt_path: Path = DEFAULT_CKPT,
               config_path: Path = Path("configs/pinn_config.yaml")
               ) -> RULPredictor:
    model = RULPredictor.from_config(str(config_path))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _single_rul(model: RULPredictor, soh_now: float, cycle_now: float,
                x_health: torch.Tensor, eol: float, max_cycles: int,
                n_points: int = 500) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Integrate the ODE from `cycle_now` for `max_cycles` and find the first
    crossing of `eol`. The raw ODE has no physical lower bound on SoH, so
    the returned trajectory is **clipped to [0, 1.0]** for display and
    extended beyond the EOL crossing with a flat 0 line.
    """
    with torch.no_grad():
        soh_0 = torch.tensor([[soh_now]], dtype=torch.float32)
        n_future = torch.linspace(cycle_now, cycle_now + max_cycles, n_points)
        traj = model(soh_0, n_future, x_health.unsqueeze(0)).squeeze().numpy()
    below = np.where(traj < eol)[0]
    if len(below) == 0:
        rul = float(max_cycles)
    else:
        rul = float(n_future[below[0]].item() - cycle_now)
    # Clip the displayed trajectory: SoH is physically bounded in [0, 1].
    traj_clipped = np.clip(traj, 0.0, 1.0)
    return rul, traj_clipped, n_future.numpy()


def predict_rul_with_uncertainty(
    model: RULPredictor,
    soh_now: float,
    cycle_now: float,
    x_health: np.ndarray,
    n_samples: int = 200,
    feature_noise_std: float = 0.01,
    eol: float | None = None,
    max_cycles: int | None = None,
    n_points: int = 500,
) -> dict:
    # Default to the model's configured EOL/horizon (from pinn_config.yaml)
    if eol is None:
        eol = float(model.eol)
    if max_cycles is None:
        max_cycles = 8000
    """
    Uncertainty via MC-dropout on the ODE network + small input-feature
    perturbations. Returns rul_mean, rul_p5, rul_p95, rul_std plus the
    mean SOH trajectory.
    """
    # Enable dropout (and any other train-mode layers) for stochastic forwards
    model.train()
    rul_samples: list[float] = []
    traj_samples: list[np.ndarray] = []
    n_axis: np.ndarray | None = None
    for _ in range(n_samples):
        x_noisy = x_health + np.random.randn(*x_health.shape).astype(x_health.dtype) * feature_noise_std
        x_t = torch.from_numpy(x_noisy)
        rul, traj, n_future = _single_rul(model, soh_now, cycle_now, x_t,
                                          eol=eol, max_cycles=max_cycles,
                                          n_points=n_points)
        rul_samples.append(rul)
        traj_samples.append(traj)
        n_axis = n_future
    model.eval()

    rul_arr = np.array(rul_samples)
    traj_arr = np.stack(traj_samples, axis=0)   # (S, n_points)
    return {
        "rul_mean": float(np.mean(rul_arr)),
        "rul_median": float(np.median(rul_arr)),
        "rul_p5":   float(np.percentile(rul_arr, 5)),
        "rul_p95":  float(np.percentile(rul_arr, 95)),
        "rul_std":  float(np.std(rul_arr)),
        "soh_trajectory_mean": traj_arr.mean(axis=0).tolist(),
        "soh_trajectory_p5":   np.percentile(traj_arr, 5, axis=0).tolist(),
        "soh_trajectory_p95":  np.percentile(traj_arr, 95, axis=0).tolist(),
        "n_axis": (n_axis.tolist() if n_axis is not None else []),
        "n_mc_samples": n_samples,
        "feature_noise_std": feature_noise_std,
    }


def diagnose_mechanism(features: HealthFeatures, baseline_dcir_mOhm: float) -> str:
    """
    Rules-of-thumb diagnosis from the 5-feature vector.

    These are the same heuristics used in AGENT_INFERENCE.md and are
    intended as *hints* rather than definitive attributions — a quantitative
    decomposition requires fitting the PyBaMM degradation submodels to the
    cell's history.
    """
    shift_v = features.ic_peak1_shift_V
    p2_area = features.ic_peak2_area_norm
    dcir = features.dcir_mOhm

    if dcir > 1.5 * baseline_dcir_mOhm:
        return "resistance-dominated (plating / contact loss risk)"
    if p2_area < 0.90:
        return "LAM-dominated (positive electrode utilisation loss)"
    if shift_v > 0.010 and p2_area > 0.95:
        return "LLI-dominated (SEI growth)"
    return "low-degradation regime (no dominant mechanism yet)"


def report(cell_id: str, soh_now: float, cycle_now: float,
           features: HealthFeatures, ckpt_path: Path = DEFAULT_CKPT,
           **mc_kwargs) -> dict:
    model = load_model(ckpt_path)
    out = predict_rul_with_uncertainty(
        model, soh_now=soh_now, cycle_now=cycle_now,
        x_health=features.as_array(), **mc_kwargs,
    )
    out.update({
        "cell_id": cell_id,
        "assessment_utc": datetime.now(timezone.utc).isoformat(),
        "cycle_now": cycle_now,
        "soh_now": soh_now,
        "eol_threshold": float(model.eol),
        "health_features": {k: float(getattr(features, k)) for k in HEALTH_FEATURES},
        "health_feature_sources": features.sources,
        "dominant_mechanism": diagnose_mechanism(features, baseline_dcir_mOhm=1.74),
        "model_checkpoint": str(ckpt_path),
    })
    return out


# EVE LF105 spec-sheet targets (RD-LF105-S01-LF rev C, March 2022) — used
# as a sanity reference on the inference plot. These are the manufacturer
# guarantees against which the model's predicted fade should be benchmarked.
SPEC_CYCLE_LIFE_25C = 4000     # cycles @ 25 °C, 0.5C/0.5C, 300 kgf compression
SPEC_EOL_RETENTION = 0.80      # capacity retention at the cycle-life rating
SPEC_DCR_MAX_mOhm = 1.8        # 25 °C, 50% SOC, 1C, 10s, fresh
SPEC_NOMINAL_CAPACITY_AH = 105.0


def latest_lab_anchor(cell_id: str) -> tuple[float, float] | None:
    """
    Return (cycle_n, SOH) of the most recently measured SoH for this cell,
    taken as the *union* of RPT and Longterm capacity-fade tables. The
    caller can use this as the initial condition for the ODE so the model
    extrapolates from real data instead of a hypothetical (cycle, SOH).
    Returns None if no measurements exist for the cell.
    """
    import pandas as pd
    last: tuple[float, float] | None = None
    for path in (Path("data/processed/rpt_capacity_fade.parquet"),
                 Path("data/processed/longterm_capacity_fade.parquet")):
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        sub = df[df["cell_id"].astype(str) == str(cell_id)]
        if sub.empty:
            continue
        row = sub.sort_values("cycle_n").iloc[-1]
        candidate = (float(row["cycle_n"]), float(row["SOH"]))
        if last is None or candidate[0] > last[0]:
            last = candidate
    return last


def _load_measured_soh(cell_id: str) -> dict[str, "pd.DataFrame"]:
    """
    Pull SoH measurements for this cell from the Phase-1 processed
    capacity-fade parquets. Returns a dict keyed by source ('rpt',
    'longterm') with columns [cycle_n, SOH]; missing files yield an
    empty dict entry.
    """
    import pandas as pd
    out: dict[str, "pd.DataFrame"] = {}
    for label, path in [("rpt",      Path("data/processed/rpt_capacity_fade.parquet")),
                        ("longterm", Path("data/processed/longterm_capacity_fade.parquet"))]:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "cell_id" not in df.columns or not (df["cell_id"].astype(str) == str(cell_id)).any():
            continue
        sub = df[df["cell_id"].astype(str) == str(cell_id)][["cycle_n", "SOH"]].dropna()
        if not sub.empty:
            out[label] = sub.sort_values("cycle_n").reset_index(drop=True)
    return out


def _save_trajectory_plot(out: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = np.asarray(out["n_axis"])
    mean = np.asarray(out["soh_trajectory_mean"])
    p5 = np.asarray(out["soh_trajectory_p5"])
    p95 = np.asarray(out["soh_trajectory_p95"])
    measured = _load_measured_soh(out["cell_id"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(n, p5, p95, alpha=0.2, label="90% CI")
    ax.plot(n, mean, lw=1.5, label="mean SOH (model)")

    # Overlay measured RPT + Longterm
    if "rpt" in measured:
        ax.scatter(measured["rpt"]["cycle_n"], measured["rpt"]["SOH"],
                   s=24, color="tab:green", marker="o",
                   edgecolor="black", linewidths=0.4,
                   label=f"lab RPT (n={len(measured['rpt'])})", zorder=3)
    if "longterm" in measured:
        ax.scatter(measured["longterm"]["cycle_n"], measured["longterm"]["SOH"],
                   s=10, color="tab:orange", marker="^", alpha=0.7,
                   label=f"lab Longterm (n={len(measured['longterm'])})", zorder=3)

    ax.axhline(out["eol_threshold"], ls="--", color="red", alpha=0.6,
               label=f"user EOL = {out['eol_threshold']}")
    n_eol = out["cycle_now"] + out["rul_mean"]
    ax.axvline(n_eol, ls=":", color="black", alpha=0.6,
               label=f"n_EOL (model) = {n_eol:.0f}")

    # Spec-sheet reference point: cycle 4000 at retention 0.80 (25 °C)
    ax.scatter([SPEC_CYCLE_LIFE_25C], [SPEC_EOL_RETENTION],
               s=90, marker="*", color="gold", edgecolor="black",
               linewidths=0.6, zorder=4,
               label=f"spec: ≥{SPEC_CYCLE_LIFE_25C} cy @ {int(SPEC_EOL_RETENTION*100)}%")

    # X-axis: span from cycle 0 (so lab + spec are visible) to max(n_EOL, spec)
    x_max = max(1.15 * n_eol, 1.15 * SPEC_CYCLE_LIFE_25C)
    x_max = min(n[-1], x_max)
    ax.set(xlabel="cycle", ylabel="SOH",
           xlim=(0, x_max), ylim=(-0.02, 1.05),
           title=f"Cell {out['cell_id']}: RUL = {out['rul_mean']:.0f} cycles "
                 f"[{out['rul_p5']:.0f}, {out['rul_p95']:.0f}] @ EOL={out['eol_threshold']}  "
                 f"(start SOH={out['soh_now']:.3f} @ cycle {out['cycle_now']:.0f})")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-id", required=True)
    ap.add_argument("--soh-now", type=float, default=None,
                    help="If omitted (and --anchor-to-lab is set), uses the "
                         "latest lab-measured SOH for this cell.")
    ap.add_argument("--cycle-now", type=float, default=None,
                    help="If omitted (and --anchor-to-lab is set), uses the "
                         "latest lab-measured cycle for this cell.")
    ap.add_argument("--anchor-to-lab", action="store_true",
                    help="Override --soh-now/--cycle-now with the most recent "
                         "RPT or Longterm measurement for this cell.")
    ap.add_argument("--temperature-C", type=float, default=25.0)
    ap.add_argument("--c-rate", type=float, default=0.5)
    ap.add_argument("--n-mc-samples", type=int, default=200)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/results/rul_report.json"))
    args = ap.parse_args()

    if args.anchor_to_lab or args.soh_now is None or args.cycle_now is None:
        anchor = latest_lab_anchor(args.cell_id)
        if anchor is None:
            raise SystemExit(f"No lab data found for cell {args.cell_id}; "
                             f"supply --soh-now and --cycle-now explicitly.")
        cycle_now, soh_now = anchor
        if args.soh_now is not None:
            soh_now = args.soh_now
        if args.cycle_now is not None:
            cycle_now = args.cycle_now
        print(f"  anchoring to lab: cell {args.cell_id} cycle {cycle_now:.0f} "
              f"SOH {soh_now:.4f}")
    else:
        cycle_now, soh_now = args.cycle_now, args.soh_now

    h = extract_for_cell(args.cell_id, temperature_C=args.temperature_C,
                         c_rate=args.c_rate)
    out = report(args.cell_id, soh_now, cycle_now, h,
                 ckpt_path=args.ckpt, n_samples=args.n_mc_samples)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")

    plot_path = args.out.with_suffix(".png")
    _save_trajectory_plot(out, plot_path)
    print(f"Wrote {plot_path}")

    # Short stdout summary
    print(json.dumps({k: out[k] for k in (
        "cell_id", "cycle_now", "soh_now",
        "rul_mean", "rul_median", "rul_p5", "rul_p95", "rul_std",
        "dominant_mechanism", "eol_threshold"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
