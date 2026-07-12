"""
PyBaMM cycle simulation wrapper.

Thin layer over pybamm.Simulation that:
  - Accepts a ParameterValues (built by parameters.build_pybamm_parameters)
  - Builds a standard CC-CV experiment from a CyclingProtocol
  - Runs N cycles with the robust IDAKLU solver
  - Extracts per-cycle features (SoH, Q_Ah, etc.) via the existing extractor
  - Caches results as parquet keyed by an md5 of the parameter+protocol JSON

The cache is what makes iteration cheap: tweak the OCV fit, re-run, get a
cache hit on identical configs; tweak a parameter, get a fresh sim.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pybamm


@dataclass(frozen=True)
class CyclingProtocol:
    c_rate: float = 0.25
    discharge_cut_V: float = 2.5
    charge_cut_V: float = 3.65
    cv_taper_to: str = "C/100"
    rest_minutes: float = 10.0
    label: str = "default"

    def steps(self) -> tuple[str, ...]:
        return (
            f"Discharge at {self.c_rate:.4f}C until {self.discharge_cut_V:.3f} V",
            f"Rest for {self.rest_minutes:.0f} minutes",
            f"Charge at {self.c_rate:.4f}C until {self.charge_cut_V:.3f} V",
            f"Hold at {self.charge_cut_V:.3f} V until {self.cv_taper_to}",
            f"Rest for {self.rest_minutes:.0f} minutes",
        )


def _fingerprint(parameters: pybamm.ParameterValues,
                  protocol: CyclingProtocol, n_cycles: int,
                  dfn_options: Optional[dict] = None) -> str:
    """Hash the run config to a short string for cache keys.

    `dfn_options` is included because two runs with identical ParameterValues
    can produce wildly different SoH trajectories under different SEI modes
    (solvent-diffusion vs reaction-limited, etc.) — and they were silently
    sharing cache entries before this was added.
    """
    # Only the keys we touched; ParameterValues serializes painfully otherwise
    safe = {}
    for k, v in parameters.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            safe[k] = v
    payload = {
        "params": safe,
        "protocol": asdict(protocol),
        "n_cycles": int(n_cycles),
        "dfn_options": {str(k): str(v) for k, v in (dfn_options or {}).items()},
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.md5(raw).hexdigest()[:12]


class Simulation:
    """
    Run a PyBaMM cycling simulation and return the per-cycle SoH trajectory.

    Examples
    --------
    >>> sim = Simulation(parameters, protocol=CyclingProtocol(c_rate=0.25),
    ...                  cache_dir=Path("Cell_to_Pack/outputs/sim_cache"))
    >>> df = sim.run(n_cycles=20)
    """

    def __init__(self,
                 parameters: pybamm.ParameterValues,
                 protocol: CyclingProtocol | None = None,
                 *,
                 cache_dir: Optional[Path] = None,
                 solver: str = "IDAKLU",
                 dfn_options: Optional[dict] = None):
        self.parameters = parameters
        self.protocol = protocol or CyclingProtocol()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.solver_name = solver
        self.dfn_options = dfn_options   # if None, build_dfn() defaults are used

    def _make_solver(self):
        if self.solver_name == "IDAKLU":
            try:
                return pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6)
            except Exception:
                pass
        return pybamm.CasadiSolver(mode="safe", dt_max=600.0)

    def cache_path(self, n_cycles: int) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        fp = _fingerprint(self.parameters, self.protocol, n_cycles,
                           dfn_options=self.dfn_options)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"sim_{fp}_{n_cycles}cy.parquet"

    def run(self, n_cycles: int, *, force: bool = False,
            timeout_s: int = 7200) -> pd.DataFrame:
        """Run the simulation; consult/populate the cache by default.

        Sets ``self.last_was_cached`` so callers (notably the calibrator)
        can track fresh vs cached evaluations for honest wall-time accounting.
        """
        cp = self.cache_path(n_cycles)
        if cp is not None and cp.exists() and not force:
            self.last_was_cached = True
            return pd.read_parquet(cp)
        self.last_was_cached = False

        import sys
        here = Path(__file__).resolve().parent
        project_root = here.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from src.simulation._pybamm_setup import build_dfn
        from src.simulation.extract_features import per_cycle_features

        t0 = time.time()
        model = build_dfn(options=self.dfn_options)
        experiment = pybamm.Experiment([self.protocol.steps()] * int(n_cycles))
        solver = self._make_solver()
        sim = pybamm.Simulation(model, parameter_values=self.parameters,
                                 experiment=experiment, solver=solver)
        sol = sim.solve()
        features = per_cycle_features(sol, params_used={"label": self.protocol.label})
        features["wall_time_s"] = time.time() - t0

        if cp is not None:
            features.to_parquet(cp)
        return features
