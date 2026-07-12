"""
pybamm_tuning
=============
Voltaris parameter-identification + simulation foundation.

A focused package for tuning PyBaMM parameters against real characterization
test data and validating predictions against longterm cycling measurements.
Serves as Step 1 (parameter ID) and Step 2 (synthetic-trajectory generation)
of the Voltaris physics-informed-neural-network workflow.

Public API:
    load_characterization(path, manufacturer=..., cell_id=..., aggregate=...)
        -> Characterization

    build_pybamm_parameters(char, base="Prada2013", aging_overrides=None,
                              fit_stoichiometry=True)
        -> pybamm.ParameterValues

    Simulation(parameters, c_rate, temperature_K).run(n_cycles, cache_path=None)
        -> pd.DataFrame   # per-cycle SoH trajectory

    load_longterm(cohort, cell_id) -> LongtermData

    validate(simulation_df, longterm) -> ValidationReport

    calibrate_k_sei(char, longterm_fade_rate_pp_per_100cy, base="Prada2013",
                    bracket=(1e-16, 1e-12), n_iter=5) -> float

All modules are independently importable for advanced use.
"""
from .characterization import (
    Characterization,
    load_characterization,
    list_available_cells,
)
from .parameters import (
    build_pybamm_parameters,
    apply_q_rpt_to_capacity,
    apply_r0_to_contact_resistance,
    apply_pre_aging,
    summarise_overrides,
)
from .ocv_fit import (
    fit_stoichiometry_from_ocv,
    StoichiometryResult,
)
from .simulation import Simulation, CyclingProtocol
from .longterm import LongtermData, load_longterm, compute_actual_fade_rate
from .validation import ValidationReport, validate
from .calibration import calibrate_k_sei, calibrate_sei_diffusivity, SEI_ONLY_DFN_OPTIONS

__all__ = [
    "Characterization",
    "load_characterization",
    "list_available_cells",
    "build_pybamm_parameters",
    "apply_q_rpt_to_capacity",
    "apply_r0_to_contact_resistance",
    "apply_pre_aging",
    "summarise_overrides",
    "fit_stoichiometry_from_ocv",
    "StoichiometryResult",
    "Simulation",
    "CyclingProtocol",
    "LongtermData",
    "load_longterm",
    "compute_actual_fade_rate",
    "ValidationReport",
    "validate",
    "calibrate_k_sei",
    "calibrate_sei_diffusivity",
    "SEI_ONLY_DFN_OPTIONS",
]
