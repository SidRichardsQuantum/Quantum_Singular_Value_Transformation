"""Canonical request → public package API adapter. No scientific algorithms."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

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
                    attempt_synthesis=False,
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
            if solver != "root-finding":
                compat = cast(dict[str, Any], report["compatibility"]).copy()
                compat["attempted_pennylane_synthesis"] = True
                compat["pennylane_synthesis_succeeded"] = synthesis.succeeded
                if not synthesis.succeeded:
                    compat["compatible"] = False
                    reasons = list(compat.get("reasons", []))
                    if "synthesis_failed" not in reasons:
                        reasons.append("synthesis_failed")
                    compat["reasons"] = reasons
                    compat["synthesis_error_type"] = synthesis.error_type
                    compat["synthesis_error"] = synthesis.error
                report["compatibility"] = compat
        return report
    if "angle_solvers" in settings:
        settings["angle_solvers"] = tuple(settings["angle_solvers"])
    if workflow == "poisson":
        source_kind = settings.pop("source_kind")
        if source_kind != "sine":
            grid = np.linspace(0.0, settings["length"], settings["n_points"] + 2)[1:-1]
            if source_kind == "constant":
                settings["source"] = np.ones(settings["n_points"])
            else:
                center = settings["length"] / 2.0
                width = max(settings["length"] / 6.0, np.finfo(float).eps)
                settings["source"] = np.exp(-0.5 * ((grid - center) / width) ** 2)
        poisson = poisson_qsvt_workflow(**settings)
        report = poisson.as_report()
        report["synthesis_quality"] = poisson.synthesis.quality_report(
            settings["phase_reconstruction_tolerance"]
        )
        return report
    if workflow == "hamiltonian_simulation":
        n_sites = settings.pop("n_sites")
        initial_site = settings.pop("initial_site")
        hopping = settings.pop("hopping")
        onsite = settings.pop("onsite")
        periodic = settings.pop("periodic")
        matrix = tight_binding_chain(
            n_sites,
            hopping=hopping,
            onsite=np.full(n_sites, onsite),
            periodic=periodic,
        )
        initial_state = np.zeros(n_sites, dtype=complex)
        initial_state[initial_site] = 1.0
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
    coefficients = [
        settings.pop("z0_coefficient"),
        settings.pop("z1_coefficient"),
        settings.pop("x0_coefficient"),
    ]
    state_name = settings.pop("input_state")
    operator = qml.dot(coefficients, [qml.Z(0), qml.Z(1), qml.X(0)])
    if state_name == "uniform":
        state = np.ones(4) / 2
    else:
        state = np.zeros(4)
        state[int(state_name.removeprefix("basis-"), 2)] = 1.0
    filtered = spectral_filter_qsvt_workflow(operator, state, **settings)
    report = filtered.as_report()
    report["synthesis_quality"] = filtered.synthesis.quality_report(
        settings["phase_reconstruction_tolerance"]
    )
    return report
