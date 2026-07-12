"""
Phase 1: Per-cell BOL parameter identification for CALB cell 0020.

Adapted from verify_e2e_phase1.py (EVE 0008). Uses the same param_id
modules, restricted to CALB cell 0020.

Writes:
    data/synthetic/verification/calb_0020_bol_params.yaml
    data/synthetic/verification/calb_0020_phase1_log.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "/home/hj/Desktop/PINNs")

from src.param_id.ocv_fit import fit_one_cell as ocv_fit_one
from src.param_id.dcir_hppc import extract_pulses, aggregate_per_cell
from src.param_id.gitt_ds import extract_gitt_step_metrics
from src.param_id.sei_selfdisc import fit_one_cell as sei_fit_one


CELL = "0020"
CELL_TAG = "calb_0020"
OUT_DIR = Path("/home/hj/Desktop/PINNs/data/synthetic/verification")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def phase1() -> dict:
    log_lines = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    result: dict = {"cell_id": CELL, "notes": []}

    # ---- (a) OCV fit → x_100, x_0, y_100, y_0, Q_n, Q_p ----
    log("=== Phase 1a: OCV stoichiometry fit ===")
    ocv = ocv_fit_one(CELL)
    log(
        f"  x_100={ocv.x_100:.4f} x_0={ocv.x_0:.4f} "
        f"y_100={ocv.y_100:.4f} y_0={ocv.y_0:.4f}  "
        f"rmse={ocv.rmse_mV:.2f} mV"
    )
    log(f"  Q_dchg={ocv.Q_dchg_Ah:.3f} Ah, Q_n={ocv.Q_n_init_Ah:.3f} Ah, "
        f"Q_p={ocv.Q_p_init_Ah:.3f} Ah")
    result["stoichiometry"] = {
        "x_100": float(ocv.x_100),
        "x_0": float(ocv.x_0),
        "y_100": float(ocv.y_100),
        "y_0": float(ocv.y_0),
        "ocv_rmse_mV": float(ocv.rmse_mV),
        "_source": "src/param_id/ocv_fit.py against Prada2013 half-cells",
    }
    result["capacity"] = {
        "Q_dchg_measured_Ah": float(ocv.Q_dchg_Ah),
        "Q_n_init_Ah": float(ocv.Q_n_init_Ah),
        "Q_p_init_Ah": float(ocv.Q_p_init_Ah),
        "_source": "derived from OCV stoichiometry + measured OCVSOC discharge Q",
    }

    # ---- (b) HPPC RC pulse fits → R0, R1, tau, C1 ----
    log("\n=== Phase 1b: HPPC RC pulse fit ===")
    pulses = extract_pulses(CELL, "HPPC", Q_nominal_Ah=float(ocv.Q_dchg_Ah))
    pulse_df = pd.DataFrame([vars(p) for p in pulses])
    if pulse_df.empty:
        log("  NO HPPC pulses extracted — falling back to DCIR")
        pulses = extract_pulses(CELL, "DCIR", Q_nominal_Ah=float(ocv.Q_dchg_Ah))
        pulse_df = pd.DataFrame([vars(p) for p in pulses])
    if pulse_df.empty:
        raise RuntimeError(f"No resistance pulses could be extracted for cell {CELL}")
    disc = pulse_df[pulse_df["direction"] == "discharge"]
    if disc.empty:
        disc = pulse_df  # accept charge pulses if no discharge available
    R0 = float(disc["R0_Ohm"].median())
    R1 = float(disc["R1_Ohm"].median())
    tau = float(disc["tau_s"].median())
    C1 = float(disc["C1_F"].median())
    log(f"  n_pulses (discharge)={len(disc)}  R0={R0*1000:.3f} mOhm  "
        f"R1={R1*1000:.3f} mOhm  tau={tau:.1f} s  C1={C1:.0f} F  "
        f"SOC range=[{disc['SOC_est'].min():.3f},{disc['SOC_est'].max():.3f}]")
    result["resistance"] = {
        "R0_Ohm": R0,
        "R1_Ohm": R1,
        "tau_s": tau,
        "C1_F": C1,
        "n_pulses": int(len(disc)),
        "SOC_min": float(disc["SOC_est"].min()),
        "SOC_max": float(disc["SOC_est"].max()),
        "_source": "src/param_id/dcir_hppc.py (RC discharge pulses)",
        "_caveat": "HPPC probes only SOC ~0.97-1.00; R(SOC) outside this "
                   "window is not identified for this cell.",
    }

    # ---- (c) GITT step metrics ----
    log("\n=== Phase 1c: GITT diffusion timescale ===")
    try:
        gitt = extract_gitt_step_metrics(
            cell_id=CELL, Q_total_Ah=float(ocv.Q_dchg_Ah),
            diffusion_length_m=None,
        )
        if gitt.empty:
            log(f"  GITT metrics empty — cell {CELL} has no usable GITT data")
            result["diffusion"] = {"note": "no GITT metrics extracted"}
        else:
            dV = float(gitt["dV_dsqrt_t_V_sqrt_s"].median())
            tau_pulse = float(gitt["tau_s"].median())
            r2 = float(gitt["fit_r2"].median())
            log(f"  n_steps={len(gitt)}  dV/dsqrt(t) median={dV:.6f} V/sqrt(s)  "
                f"pulse tau median={tau_pulse:.1f} s  R2 median={r2:.4f}")
            result["diffusion"] = {
                "dV_dsqrt_t_V_per_sqrt_s_median": dV,
                "tau_pulse_s_median": tau_pulse,
                "gitt_fit_r2_median": r2,
                "n_steps": int(len(gitt)),
                "_source": "src/param_id/gitt_ds.py",
                "_caveat": "Full-cell GITT cannot separate D_s_n vs D_s_p. "
                           "PyBaMM will retain the Prada2013 default D values "
                           "in the deg-parameter fit; this metric documents the "
                           "measured diffusion timescale.",
            }
    except Exception as e:
        log(f"  GITT extraction failed: {type(e).__name__}: {e}")
        result["diffusion"] = {"error": f"{type(e).__name__}: {e}"}

    # ---- (d) Self-discharge → k_SEI ceiling ----
    log("\n=== Phase 1d: Self-discharge SEI ceiling ===")
    try:
        sd = sei_fit_one(CELL, Q_nominal_Ah=float(ocv.Q_dchg_Ah))
        log(f"  I_sd={sd.I_sd_uA:.1f} uA  dSOC/dt={sd.dSOC_dt_per_h*100:+.4f} "
            f"%SOC/h  k_SEI_max={sd.k_SEI_max_m_per_s:.3e} m/s")
        result["sei"] = {
            "I_sd_uA": float(sd.I_sd_uA),
            "dSOC_dt_per_h_pct": float(sd.dSOC_dt_per_h * 100),
            "k_SEI_max_m_per_s": float(sd.k_SEI_max_m_per_s),
            "dV_dt_uV_per_s": float(sd.dV_dt_uV_per_s),
            "_source": "src/param_id/sei_selfdisc.py (upper bound)",
            "_caveat": "Bound uses Prada2013 geometric area (0.18 m^2). Real "
                       "jelly-roll area for a 105 Ah cell is ~30x larger, so "
                       "the true k_SEI ceiling is ~30x smaller.",
        }
    except Exception as e:
        log(f"  Self-discharge fit failed: {type(e).__name__}: {e}")
        result["sei"] = {"error": f"{type(e).__name__}: {e}"}
        result["notes"].append("SEI ceiling not identified — using OKane2022 "
                                "default in Phase 2.")

    # Save
    out_yaml = OUT_DIR / f"{CELL_TAG}_bol_params.yaml"
    with open(out_yaml, "w") as f:
        yaml.safe_dump(result, f, sort_keys=False)
    log(f"\nWrote BOL params -> {out_yaml}")

    (OUT_DIR / f"{CELL_TAG}_phase1_log.txt").write_text("\n".join(log_lines))
    return result


if __name__ == "__main__":
    phase1()
