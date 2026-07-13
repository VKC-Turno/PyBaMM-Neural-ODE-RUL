"""
Voltaris/Data_Exploration/phase3_plot_heldout.py
================================================

3-panel held-out overlay figure for the manuscript. Reads the JSON companion
written by ``phase3_validate.run_validation`` and renders a publication-quality
PDF (+ optional PNG) that overlays predicted SoH on the observed Longterm
trajectory for CALB_0029, EVE_0003, and REPT_0031.

When the JSON already carries per-cell ``pred_cycle`` / ``soh_pred`` (and
optional ``soh_p10`` / ``soh_p90``) arrays the plot uses them verbatim.
Otherwise it re-loads the operator checkpoint referenced by the report
(overridable via ``--checkpoint``) and re-runs ``predict_cell_soh`` — so the
figure never depends on state that lives only in the training run.

Design
------
- Serif (Times) 8 pt, thin axes, TrueType embedded fonts (Type 42).
- Deep-blue observed line, firebrick dashed prediction, 15% band for
  optional (p10, p90) uncertainty, grey dotted SoH=0.80 EoL rule.
- 3-panel strip, 6.5 in wide × 2.2 in tall — fits a single manuscript column.

CLI
---
    .venv/bin/python -m Voltaris.Data_Exploration.phase3_plot_heldout \\
        --validation-json outputs/results/phase3_heldout_validation.json \\
        --out-pdf paper/figures/phase3_heldout_overlay.pdf \\
        [--out-png paper/figures/phase3_heldout_overlay.png] \\
        [--checkpoint outputs/models/phase3_operator.pt]

Smoke test (no checkpoint, no on-disk Longterm CSVs required):
    .venv/bin/python -m Voltaris.Data_Exploration.phase3_plot_heldout --smoke
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project imports — reuse phase3_validate for observed-SoH + prediction I/O
# so we never diverge from the validation harness.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path("/home/hj/Desktop/PINNs")
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Voltaris.Data_Exploration import phase3_validate as _pv  # noqa: E402
from Voltaris.Data_Exploration.phase3_validate import (  # noqa: E402
    _load_longterm_soh,
    load_operator_from_checkpoint,
    predict_cell_soh,
)


# ---------------------------------------------------------------------------
# Publication style — kept in a dict so callers can rc_context() it.
# ---------------------------------------------------------------------------
_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1.1,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

OBS_COLOR = "#1F4E79"   # deep blue — observed
PRED_COLOR = "#B22222"  # firebrick — predicted
BAND_COLOR = "#B22222"  # same hue at alpha=0.15 — uncertainty band
EOL_COLOR = "#7F7F7F"   # neutral grey — SoH=0.80 rule

# Panel spec: order + subtitle + y-limits follow the design mock.
_PANELS = [
    dict(cell_id="0029", make="CALB",
         subtitle="fast (EoL 292 cy)", ylim=(0.60, 1.00)),
    dict(cell_id="0003", make="EVE",
         subtitle="slow (no EoL, 236 cy)", ylim=(0.70, 1.00)),
    dict(cell_id="0031", make="REPT",
         subtitle="mid (212 cy pre-knee)", ylim=(0.60, 1.00)),
]


# ---------------------------------------------------------------------------
# 1. I/O helpers
# ---------------------------------------------------------------------------
def load_validation_results(json_path: str | Path) -> dict:
    """Parse and return the report dict produced by ``phase3_validate``.

    The JSON is the ``.json`` sibling of the validation Markdown report —
    schema is exactly what ``phase3_validate.run_validation`` emits (keys
    include ``checkpoint``, ``per_cell``, ``fisher_gate``, ``regime_swap``,
    ``gates``). This helper only parses; it does not mutate or validate.
    """
    return json.loads(Path(json_path).read_text())


def load_observed_soh(cell_id: str, make: str
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(cycles, soh)`` for a held-out cell from the Longterm CSV.

    Thin wrapper around ``phase3_validate._load_longterm_soh`` so the two
    scripts always agree on how observed SoH is defined. Returns empty
    arrays when no on-disk CSV is present (caller decides how to warn).
    """
    return _load_longterm_soh(cell_id, make)


# ---------------------------------------------------------------------------
# 2. Prediction lookup (inline arrays preferred; fall back to checkpoint)
# ---------------------------------------------------------------------------
def _get_prediction(row: dict,
                    checkpoint: Optional[Path],
                    model_cache: dict,
                    ) -> Optional[tuple[np.ndarray, np.ndarray,
                                        Optional[np.ndarray],
                                        Optional[np.ndarray]]]:
    """Return ``(cycles, soh_pred, p10, p90)`` for a per-cell report row.

    Priority order:
      1. Inline arrays on the row (``pred_cycle``, ``soh_pred``, optional
         ``soh_p10``/``soh_p90``) — used when the report was augmented by
         a downstream step.
      2. Re-run ``predict_cell_soh`` using ``checkpoint``; models are
         cached in ``model_cache`` so multi-panel figures reload once.

    Returns ``None`` when neither path is available (caller must warn).
    """
    if "pred_cycle" in row and "soh_pred" in row:
        n = np.asarray(row["pred_cycle"], dtype=np.float32)
        y = np.asarray(row["soh_pred"], dtype=np.float32)
        p10 = (np.asarray(row["soh_p10"], dtype=np.float32)
               if "soh_p10" in row else None)
        p90 = (np.asarray(row["soh_p90"], dtype=np.float32)
               if "soh_p90" in row else None)
        return n, y, p10, p90

    if checkpoint is None or not Path(checkpoint).exists():
        return None

    key = str(checkpoint)
    model = model_cache.get(key)
    if model is None:
        model = load_operator_from_checkpoint(checkpoint)
        model_cache[key] = model
    # Anchor the prediction at the observed first-cycle SoH (the cell's
    # second-life starting state) so pred/obs share the same starting point.
    # Falls back to soh_0=1.0 if no observation is available.
    obs = load_observed_soh(row["cell_id"], row["make"])
    soh_0 = float(obs[1][0]) if obs[0].size else 1.0
    n, y = predict_cell_soh(model, row["cell_id"], row["make"], soh_0=soh_0)
    return n, y, None, None


# ---------------------------------------------------------------------------
# 3. Main figure builder
# ---------------------------------------------------------------------------
def plot_heldout_overlay(validation_results: dict,
                         out_path: str | Path,
                         out_png: Optional[str | Path] = None,
                         checkpoint: Optional[str | Path] = None,
                         ) -> Path:
    """Render the 3-panel held-out overlay to ``out_path`` (PDF).

    Behaviour
    ---------
    - Missing cells (no report row AND no checkpoint, or no observed CSV)
      are skipped with a stderr warning; remaining panels still render.
    - Predictions are truncated to ``[0, obs_cycles.max()]`` so we never
      plot beyond the horizon the observation can score against — this
      makes truncated pred arrays (shorter than obs) render cleanly too.
    - PDF is saved at dpi=300; the optional PNG at dpi=150.

    Returns the resolved PDF path (parent is mkdir-p'd).
    """
    # Resolve checkpoint: explicit arg > report's own field > None.
    ckpt_path: Optional[Path] = Path(checkpoint) if checkpoint else None
    if ckpt_path is None:
        rc = validation_results.get("checkpoint")
        if rc and Path(rc).exists():
            ckpt_path = Path(rc)

    rows_by_key: dict[str, dict] = {}
    for r in validation_results.get("per_cell", []) or []:
        if "cell_id" in r and "make" in r:
            rows_by_key[f"{r['make']}_{r['cell_id']}"] = r

    with mpl.rc_context(_RCPARAMS):
        fig, axes = plt.subplots(
            1, 3, figsize=(6.5, 2.2), sharey=False,
            gridspec_kw=dict(wspace=0.28, left=0.07, right=0.995,
                             bottom=0.20, top=0.86),
        )
        model_cache: dict = {}

        for ax, panel in zip(axes, _PANELS):
            key = f"{panel['make']}_{panel['cell_id']}"
            row = rows_by_key.get(key,
                                  {"cell_id": panel["cell_id"],
                                   "make": panel["make"]})

            obs_n, obs_soh = load_observed_soh(panel["cell_id"],
                                               panel["make"])
            if obs_n.size == 0:
                print(f"[phase3_plot_heldout] WARN: no observed SoH for "
                      f"{key}; panel disabled", file=sys.stderr)
                ax.set_axis_off()
                ax.set_title(f"{key}\n(no observation)")
                continue

            pred = _get_prediction(row, ckpt_path, model_cache)
            if pred is None:
                print(f"[phase3_plot_heldout] WARN: no prediction for "
                      f"{key}; showing observation only", file=sys.stderr)
                n_p = np.array([], dtype=np.float32)
                y_p = np.array([], dtype=np.float32)
                p10 = p90 = None
            else:
                n_p, y_p, p10, p90 = pred

            # Clip prediction to the observed horizon so we do not
            # extrapolate visually past what we can score.
            if n_p.size:
                horizon = float(obs_n[-1])
                mask = n_p <= horizon + 1e-6
                n_p = n_p[mask]
                y_p = y_p[mask]
                if p10 is not None:
                    p10 = np.asarray(p10, dtype=np.float32)[mask]
                if p90 is not None:
                    p90 = np.asarray(p90, dtype=np.float32)[mask]

            # Uncertainty band (behind the mean lines).
            if (p10 is not None and p90 is not None
                    and n_p.size == p10.size == p90.size and n_p.size):
                ax.fill_between(n_p, p10, p90, color=BAND_COLOR,
                                alpha=0.15, lw=0, zorder=1)

            # EoL reference.
            ax.axhline(0.80, color=EOL_COLOR, lw=0.6, ls=":", zorder=2)

            # Observed + predicted overlays.
            ax.plot(obs_n, obs_soh, color=OBS_COLOR, lw=1.2,
                    label="observed", zorder=3)
            if n_p.size:
                ax.plot(n_p, y_p, color=PRED_COLOR, lw=1.2,
                        ls=(0, (4, 2)), label="predicted", zorder=4)

            xmax = max(float(obs_n[-1]),
                       float(n_p[-1]) if n_p.size else 0.0)
            ax.set_xlim(0, xmax)
            ax.set_ylim(*panel["ylim"])
            ax.set_xlabel("cycle number")
            ax.set_title(f"{key}  ({panel['subtitle']})")
            ax.tick_params(direction="out", length=2.5)

        axes[0].set_ylabel("state of health")
        # One legend on the leftmost panel keeps the strip uncluttered.
        axes[0].legend(loc="lower left", frameon=False)

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, format="pdf", bbox_inches="tight")
        if out_png is not None:
            png = Path(out_png)
            png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(png, dpi=150, format="png", bbox_inches="tight")
        plt.close(fig)
        return out


# ---------------------------------------------------------------------------
# 4. Smoke fixture — dummy report + fake Longterm CSVs
# ---------------------------------------------------------------------------
def _build_smoke_report(tmp_dir: Path) -> dict:
    """Fabricate a validation report + on-disk Longterm CSVs for --smoke.

    Uses ``linspace + noise`` so every panel gets a plausible fade
    trajectory that also exercises the EoL rule and the p10/p90 band.
    """
    rng = np.random.default_rng(0)
    (tmp_dir / "Longterm").mkdir(parents=True, exist_ok=True)
    per_cell: list[dict] = []

    for panel in _PANELS:
        n = np.linspace(0, 300, 61)
        obs = np.clip(1.0 - 0.00085 * n + rng.normal(0, 0.005, n.size),
                      0.5, 1.05)
        pred = np.clip(1.0 - 0.00080 * n + rng.normal(0, 0.002, n.size),
                       0.5, 1.05)

        # Write a CALB/REPT-style cycle-summary CSV that _load_longterm_soh
        # can read — capacity in Ah, positive, indexed by cycle_no.
        cap = obs * 105.0
        csv_path = (tmp_dir / "Longterm"
                    / f"{panel['make']}_Longterm_cell_"
                      f"{panel['cell_id']}_cycle.csv")
        pd.DataFrame({"cycle_no": n.astype(int),
                      "discharge_cap_ah": cap}).to_csv(csv_path,
                                                       index=False)

        per_cell.append({
            "cell_id": panel["cell_id"],
            "make": panel["make"],
            "pred_cycle": n.tolist(),
            "soh_pred": pred.tolist(),
            "soh_p10": (pred - 0.02).tolist(),
            "soh_p90": (pred + 0.02).tolist(),
        })

    return {"checkpoint": "SMOKE-NO-CHECKPOINT",
            "branch_dim": 11,
            "per_cell": per_cell}


def _run_smoke(out_pdf: Optional[Path],
               out_png: Optional[Path]) -> Path:
    """End-to-end smoke: patch LONGTERM_DIR, build fixtures, render."""
    tmpdir = Path(tempfile.mkdtemp(prefix="phase3_plot_smoke_"))
    original = _pv.LONGTERM_DIR
    try:
        _pv.LONGTERM_DIR = tmpdir / "Longterm"
        report = _build_smoke_report(tmpdir)
        pdf = out_pdf if out_pdf is not None else tmpdir / "smoke.pdf"
        plot_heldout_overlay(report, pdf, out_png=out_png)
        print(f"[phase3_plot_heldout] SMOKE OK -> {pdf}")
        return pdf
    finally:
        _pv.LONGTERM_DIR = original
        # Keep the tmpdir only if smoke wrote its PDF inside it, so the
        # user can inspect the output. Otherwise clean up.
        if out_pdf is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Render the 3-panel held-out SoH overlay figure.")
    p.add_argument("--validation-json", type=Path,
                   default=Path("outputs/results/"
                                "phase3_heldout_validation.json"),
                   help="JSON companion produced by "
                        "phase3_validate.run_validation")
    p.add_argument("--out-pdf", type=Path,
                   default=Path("outputs/results/"
                                "phase3_heldout_overlay.pdf"),
                   help="Output PDF path (parent auto-mkdir'd).")
    p.add_argument("--out-png", type=Path, default=None,
                   help="Optional PNG companion (dpi=150).")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Model checkpoint to (re-)predict trajectories "
                        "not inlined in the JSON. Defaults to "
                        "report['checkpoint'] when present.")
    p.add_argument("--smoke", action="store_true",
                   help="Fabricate dummy data and render — no checkpoint "
                        "or Longterm CSVs required.")
    args = p.parse_args(argv or sys.argv[1:])

    if args.smoke:
        # Ignore the default out_pdf when running smoke unless the user
        # explicitly requested a non-default path.
        default_pdf = Path("outputs/results/phase3_heldout_overlay.pdf")
        out_pdf = args.out_pdf if args.out_pdf != default_pdf else None
        _run_smoke(out_pdf, args.out_png)
        return 0

    if not args.validation_json.exists():
        print(f"[phase3_plot_heldout] validation JSON not found: "
              f"{args.validation_json}\n"
              f"  run phase3_validate first, or re-run with --smoke",
              file=sys.stderr)
        return 2

    report = load_validation_results(args.validation_json)
    out = plot_heldout_overlay(report, args.out_pdf,
                               out_png=args.out_png,
                               checkpoint=args.checkpoint)
    print(f"[phase3_plot_heldout] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
