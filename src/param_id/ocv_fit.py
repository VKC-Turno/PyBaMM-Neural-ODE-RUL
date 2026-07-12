"""
src/param_id/ocv_fit.py
-----------------------
Fit half-cell stoichiometric windows to measured full-cell OCV.

Method
~~~~~~
For an LFP/graphite cell, full-cell open-circuit voltage at SOC s is

    V(s) = U_p(y(s)) - U_n(x(s))

where
    x(s) = x_0 + s * (x_100 - x_0)        (graphite stoichiometry)
    y(s) = y_0 + s * (y_100 - y_0)        (LFP stoichiometry)

PyBaMM convention (Prada2013): stoichiometry equals c_s / c_s_max, so
    high x  → lithiated graphite   (low U_n)
    high y  → lithiated LFP        (low U_p)

Hence at SOC=1 (cell charged): graphite is lithiated and LFP is delithiated,
i.e. x_100 is HIGH and y_100 is LOW (and vice versa at SOC=0).

The fit minimises RMSE between U_p(y(s)) - U_n(x(s)) and the measured
pseudo-OCV from the low-rate (≈C/20) discharge branch of the OCV_SOC test.

Outputs per cell:
    x_100, x_0, y_100, y_0     stoichiometric limits
    Q_n_init_Ah, Q_p_init_Ah   electrode capacities derived from the
                                identified utilisation windows and the
                                measured cell capacity
    rmse_mV                    fit quality
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data_loader import load_test  # noqa: E402

# Lazy import — keeps unit tests fast and avoids a hard dep at import time
def _get_prada_ocp_funcs():
    import pybamm
    p = pybamm.ParameterValues("Prada2013")
    return p["Positive electrode OCP [V]"], p["Negative electrode OCP [V]"]


@dataclass
class OCVFit:
    cell_id: str
    x_100: float
    x_0: float
    y_100: float
    y_0: float
    Q_dchg_Ah: float
    Q_n_init_Ah: float
    Q_p_init_Ah: float
    rmse_mV: float
    n_points: int
    voltage_min_V: float
    voltage_max_V: float


def extract_pseudo_ocv(cell_id: str, n_sample: int = 400) -> pd.DataFrame:
    """
    Pull the low-rate discharge branch from OCV_SOC and resample uniformly in SOC.

    Returns a DataFrame with columns ['soc', 'voltage'] sorted by SOC ascending.
    SOC=1 corresponds to the start of the discharge (just after the top rest),
    SOC=0 to the end (cutoff voltage).
    """
    df = load_test("OCV_SOC", cell_id=cell_id).sort_values("time").reset_index(drop=True)
    disc = df[df["step_name"] == "CC_DChg"].copy()
    if disc.empty:
        raise ValueError(f"No CC_DChg branch found for cell {cell_id}")

    # capacity is signed (negative on discharge in EVE schema)
    cap = disc["capacity"].to_numpy(dtype=float)
    q_max = float(np.abs(cap).max())
    if q_max <= 0:
        raise ValueError(f"Discharge capacity is zero for cell {cell_id}")
    soc = 1.0 - np.abs(cap) / q_max
    disc = disc.assign(soc=soc)
    disc = disc.sort_values("soc").reset_index(drop=True)

    # Uniformly resample in SOC for stable fitting
    soc_grid = np.linspace(0.02, 0.98, n_sample)
    v_grid = np.interp(soc_grid, disc["soc"].to_numpy(), disc["voltage"].to_numpy())
    out = pd.DataFrame({"soc": soc_grid, "voltage": v_grid})
    out.attrs["Q_dchg_Ah"] = q_max
    return out


def _model_voltage(soc: np.ndarray, theta: np.ndarray, U_p, U_n) -> np.ndarray:
    x_100, x_0, y_100, y_0 = theta
    x = x_0 + soc * (x_100 - x_0)
    y = y_0 + soc * (y_100 - y_0)
    up = np.array([float(U_p(v)) for v in y])
    un = np.array([float(U_n(v)) for v in x])
    return up - un


def fit_one_cell(cell_id: str, n_sample: int = 400) -> OCVFit:
    """Fit the four stoichiometric limits for one cell and return diagnostics."""
    from scipy.optimize import minimize

    U_p, U_n = _get_prada_ocp_funcs()
    ocv = extract_pseudo_ocv(cell_id, n_sample=n_sample)
    soc = ocv["soc"].to_numpy()
    v_meas = ocv["voltage"].to_numpy()
    q_max = float(ocv.attrs["Q_dchg_Ah"])

    # Initial guess (literature ranges for LFP/graphite)
    theta0 = np.array([0.80, 0.01, 0.04, 0.92])

    def loss(theta):
        x_100, x_0, y_100, y_0 = theta
        # Soft penalties for the obvious ordering constraints
        if not (x_100 > x_0 and y_0 > y_100):
            return 1e6
        v_hat = _model_voltage(soc, theta, U_p, U_n)
        return float(np.mean((v_hat - v_meas) ** 2))

    bounds = [
        (0.55, 0.95),   # x_100  graphite lithiated  (high)
        (0.0, 0.20),    # x_0    graphite delithiated (low)
        (0.0, 0.20),    # y_100  LFP delithiated     (low)
        (0.55, 0.99),   # y_0    LFP lithiated       (high)
    ]
    res = minimize(loss, theta0, method="L-BFGS-B", bounds=bounds)
    x_100, x_0, y_100, y_0 = (float(v) for v in res.x)

    v_hat = _model_voltage(soc, res.x, U_p, U_n)
    rmse_mV = float(np.sqrt(np.mean((v_hat - v_meas) ** 2)) * 1000.0)

    # Electrode capacities from utilisation windows
    Q_n = q_max / (x_100 - x_0)
    Q_p = q_max / (y_0 - y_100)

    return OCVFit(
        cell_id=cell_id,
        x_100=x_100, x_0=x_0, y_100=y_100, y_0=y_0,
        Q_dchg_Ah=q_max,
        Q_n_init_Ah=Q_n, Q_p_init_Ah=Q_p,
        rmse_mV=rmse_mV,
        n_points=len(soc),
        voltage_min_V=float(v_meas.min()),
        voltage_max_V=float(v_meas.max()),
    )


def fit_cells(cell_ids: list[str]) -> pd.DataFrame:
    rows = []
    for cid in cell_ids:
        try:
            fit = fit_one_cell(cid)
            rows.append(asdict(fit))
            print(f"  ✓ cell {cid}: rmse={fit.rmse_mV:.2f} mV  "
                  f"x_100={fit.x_100:.3f} x_0={fit.x_0:.3f}  "
                  f"y_100={fit.y_100:.3f} y_0={fit.y_0:.3f}")
        except Exception as e:
            print(f"  ✗ cell {cid}: {type(e).__name__}: {e}")
    return pd.DataFrame(rows)


def save_fits(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved OCV fits ({len(df)} cells) → {out_path}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import yaml

    cfg_path = Path("configs/dataset.yaml")
    cells: list[str]
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        cells = [str(c).zfill(4) for c in cfg.get("dataset", {}).get("selected_cells", [])]
    if not cells:
        cells = ["0005", "0006", "0007", "0008"]

    df = fit_cells(cells)
    save_fits(df, Path("data/processed/ocv_fit.parquet"))

    # Diagnostic plot: measured vs fitted OCV for each cell
    U_p, U_n = _get_prada_ocp_funcs()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, row in df.iterrows():
        ocv = extract_pseudo_ocv(row["cell_id"])
        v_hat = _model_voltage(
            ocv["soc"].to_numpy(),
            np.array([row["x_100"], row["x_0"], row["y_100"], row["y_0"]]),
            U_p, U_n,
        )
        ax.plot(ocv["soc"], ocv["voltage"], lw=1.0, alpha=0.7,
                label=f"meas {row['cell_id']}")
        ax.plot(ocv["soc"], v_hat, lw=1.0, ls="--",
                label=f"fit  {row['cell_id']} ({row['rmse_mV']:.1f} mV)")
    ax.set(xlabel="SOC", ylabel="OCV [V]", title="OCV fit: Prada2013 half-cells")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out_fig = Path("outputs/results/ocv_fit.png")
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"Saved diagnostic plot → {out_fig}")
