"""Canonical request → public package API adapter. No scientific algorithms."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pennylane as qml

from qsvt.compatibility import qsvt_compatibility_report
from qsvt.hamiltonians import tight_binding_chain
from qsvt.stable import (
    design_workflow,
    hamiltonian_simulation_workflow,
    poisson_qsvt_workflow,
    spectral_filter_qsvt_workflow,
)

from .catalogue import catalogue, validate_request


def execute_request(raw: dict[str, Any]) -> dict[str, Any]:
    request = validate_request(raw)
    workflow = request["workflow"]
    settings = dict(request["settings"])
    if catalogue()["workflows"][workflow]["api"] == "design":
        solver = settings.pop("angle_solver")
        tolerance = settings.pop("phase_reconstruction_tolerance")
        reconstruction_points = settings.pop("reconstruction_num_points")
        attempt = settings["attempt_synthesis"]
        # The frozen design facade keeps its original root-finding default.
        # For another solver, ask the package compatibility API explicitly.
        if solver != "root-finding":
            settings["attempt_synthesis"] = False
        result = design_workflow(workflow, **settings)
        if solver != "root-finding":
            result = replace(
                result,
                compatibility=qsvt_compatibility_report(
                    result.coeffs,
                    bounded_num_points=settings["bounded_num_points"],
                    attempt_synthesis=attempt,
                    angle_solver=solver,
                ),
            )
        report = result.as_report()
        # These are additional public package reports, not studio estimates.
        report["resource_report"] = result.resource_report()
        if attempt:
            synthesis = result.synthesize(
                angle_solver=solver, reconstruction_num_points=reconstruction_points
            )
            report["synthesis"] = synthesis.as_report()
            report["synthesis_quality"] = synthesis.quality_report(tolerance)
        return report
    if "angle_solvers" in settings:
        settings["angle_solvers"] = tuple(settings["angle_solvers"])
    if workflow == "poisson":
        poisson = poisson_qsvt_workflow(**settings)
        report = poisson.as_report()
        report["synthesis_quality"] = poisson.synthesis.quality_report(
            settings["phase_reconstruction_tolerance"]
        )
        return report
    if workflow == "hamiltonian_simulation":
        # Keep the problem identical to examples/hamiltonian_simulation.py.
        matrix = tight_binding_chain(6)
        initial_state = np.zeros(6, dtype=complex)
        initial_state[1] = 1.0
        evolved = hamiltonian_simulation_workflow(matrix, initial_state, **settings)
        report = evolved.as_report()
        if settings["execute_qsvt"] and evolved.qsvt_execution is not None:
            report["component_synthesis_quality"] = {
                name: synthesis.quality_report(
                    settings["phase_reconstruction_tolerance"]
                )
                for name, synthesis in evolved.qsvt_execution.component_syntheses
            }
        return report
    # Exactly the published spectral_filter_qsvt.py problem, not a UI-created toy.
    operator = qml.dot([0.4, 0.3, 0.2], [qml.Z(0), qml.Z(1), qml.X(0)])
    filtered = spectral_filter_qsvt_workflow(operator, np.ones(4) / 2, **settings)
    report = filtered.as_report()
    report["synthesis_quality"] = filtered.synthesis.quality_report(
        settings["phase_reconstruction_tolerance"]
    )
    return report
