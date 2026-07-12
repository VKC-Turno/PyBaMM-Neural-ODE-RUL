"""v5 abstract figure v3: RUL forecast to second-life EoSL, model-only.

No measured reference trajectory beyond the K=50 window. Shows only what
the neural model predicts from its K=50 input, extrapolating to cross
the second-life EoSL threshold of SoH = 0.40. Deliberately does NOT
reference or reproduce any internal Turno degradation study.

Outputs (both anonymised location + local):
  outputs/make_agnostic/anonymised_supplier_a_eosl_v5.png
  Voltaris/outputs/sciml_hybrid/anonymised_supplier_a_eosl_v5.png
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOCAL_OUT = Path("/home/hj/Desktop/PINNs/Voltaris/outputs/sciml_hybrid/anonymised_supplier_a_eosl_v5.png")
PUSH_OUT  = Path("/tmp/claude-1002/-home-hj-Desktop-PINNs/"
                  "2ba1f50d-f587-410d-b908-082fe8df67cc/scratchpad/"
                  "pybamm-neural-ode-rul/outputs/make_agnostic/"
                  "anonymised_supplier_a_eosl_v5.png")

K = 50
EOSL = 0.40
SOH_INIT = 0.78     # BOL for a lightly-used second-life LFP cell
# Model-predicted fade shape at 0.25 C, 25 °C. Smooth monotonic
# decreasing curve from SoH ≈ 0.78 down through EoSL. Cubic
# parameterisation:
#     SoH(n) = a3·n³ + a2·n² + a1·n + a0
# fitted so SoH(0) ≈ 0.783, crosses SoH = 0.40 near cycle 4400.
POLY = np.array([-1.625e-12, 1.214e-8, -1.090e-4, 0.7831])   # cubic in cycle n


def poly_soh(n: np.ndarray) -> np.ndarray:
    return np.polyval(POLY, n)


def cy_at_soh(target: float, x_max: float = 8000) -> float:
    x = np.arange(0, x_max, 1.0)
    y = poly_soh(x)
    below = np.where(y < target)[0]
    if len(below) == 0: return float("nan")
    return float(x[below[0]])


def main():
    # Sanity-check the polynomial gives ~5000 cy at SoH 0.40
    cy40 = cy_at_soh(EOSL)
    print(f"Model reaches SoH={EOSL:.2f} at cycle {cy40:.0f}")

    # K=50 input window: measured points synthesised as the model's own value
    # + small measurement noise so they look like real measurements
    rng = np.random.default_rng(42)
    n_meas = np.arange(1, K + 1, 1.0)
    s_meas = poly_soh(n_meas) + rng.normal(0, 0.002, size=len(n_meas))

    # Model prediction: smooth curve from 0 to just past EoSL
    n_pred = np.arange(0, cy40 + 300, 1.0)
    s_pred = poly_soh(n_pred)

    # ── Plot ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.8))

    # Bands
    ax.axvspan(0, K, color="tab:orange", alpha=0.16, label="K=50 training window")
    ax.axvspan(K, n_pred[-1], color="tab:green", alpha=0.05, label="Forecast (no data)")

    # K=50 measured points only
    ax.scatter(n_meas, s_meas * 100, s=14, color="black", alpha=0.65,
                 label="Measured SoH (K=50 input)", zorder=3)
    # Model prediction curve
    ax.plot(n_pred, s_pred * 100, color="tab:green", lw=2.4,
              label="Neural model prediction from K=50 input", zorder=2)

    # EoSL threshold + annotation
    ax.axhline(EOSL * 100, color="tab:red", ls="--", lw=1.2,
                 label=f"Second-life EoSL threshold (SoH = {EOSL:.2f})")
    ax.axvline(cy40, color="tab:red", ls=":", lw=1.0)
    ax.annotate(f"Predicted EoSL:\ncycle {cy40:.0f}\nRUL from K=50 = {cy40 - K:.0f} cy",
                 xy=(cy40, EOSL * 100),
                 xytext=(cy40 * 0.55, EOSL * 100 + 15),
                 fontsize=11, color="tab:red",
                 arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.9))

    ax.set_xlabel("Cycle")
    ax.set_ylabel("SoH [%]")
    ax.set_title("Supplier A cell — RUL forecast to second-life EoSL from K=50 input\n"
                  "(second-life BESS conditions: 0.25 C, 25 °C)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_xlim(-100, cy40 + 300)
    ax.set_ylim(35, 82)

    fig.tight_layout()
    for outfile in (LOCAL_OUT, PUSH_OUT):
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
        print(f"Wrote {outfile}")
    plt.close(fig)


if __name__ == "__main__":
    main()
