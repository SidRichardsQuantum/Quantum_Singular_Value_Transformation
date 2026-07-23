"""Machine-readable acceptance contracts for the three flagship workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, cast

import numpy as np

FLAGSHIP_ACCEPTANCE_SCHEMA_NAME = "qsvt-flagship-acceptance"
FLAGSHIP_ACCEPTANCE_SCHEMA_VERSION = "1.0"

_MATRIX: dict[str, dict[str, object]] = {
    "poisson_qsvt": {
        "scope": "finite_qsvt",
        "criteria": (
            {
                "id": "classical_reference",
                "description": (
                    "The finite-difference system has direct and conjugate-gradient "
                    "classical references with a numerically valid direct residual."
                ),
                "required_for_scope": True,
                "required_for_full_qsvt": True,
            },
            {
                "id": "polynomial_accuracy",
                "description": (
                    "The selected inverse polynomial meets the caller's solution "
                    "relative-error tolerance."
                ),
                "required_for_scope": True,
                "required_for_full_qsvt": True,
            },
            {
                "id": "phase_synthesis",
                "description": (
                    "Phase synthesis succeeds and reconstruction meets the declared "
                    "phase tolerance."
                ),
                "required_for_scope": True,
                "required_for_full_qsvt": True,
            },
            {
                "id": "finite_qsvt_execution",
                "description": (
                    "The requested finite QNode executes and its logical solution "
                    "meets the workflow tolerance."
                ),
                "required_for_scope": True,
                "required_for_full_qsvt": True,
            },
            {
                "id": "diagnostics_and_resources",
                "description": (
                    "The report includes component errors, physical observables, "
                    "normalization, and an encoding-aware resource estimate."
                ),
                "required_for_scope": True,
                "required_for_full_qsvt": True,
            },
        ),
    },
    "spectral_filter_qsvt": {
        "scope": "finite_qsvt",
        "criteria": (
            {
                "id": "classical_reference",
                "description": (
                    "The exact spectral projector and postselected reference state "
                    "are present and finite."
                ),
                "required_for_scope": True,
                "required_for_full_qsvt": True,
            },
            {
                "id": "polynomial_accuracy",
                "description": (
                    "The selected band-filter polynomial meets the caller's operator "
                    "relative-error tolerance."
                ),
                "required_for_scope": True,
                "required_for_full_qsvt": True,
            },
            {
                "id": "phase_synthesis",
                "description": (
                    "Phase synthesis succeeds and reconstruction meets the declared "
                    "phase tolerance."
                ),
                "required_for_scope": True,
                "required_for_full_qsvt": True,
            },
            {
                "id": "finite_qsvt_execution",
                "description": (
                    "The requested Pauli-LCU QNode executes and agrees with the "
                    "polynomial output within the workflow tolerance."
                ),
                "required_for_scope": True,
                "required_for_full_qsvt": True,
            },
            {
                "id": "diagnostics_and_resources",
                "description": (
                    "The report includes success probabilities, observables, "
                    "component errors, normalization, and encoding-aware resources."
                ),
                "required_for_scope": True,
                "required_for_full_qsvt": True,
            },
        ),
    },
    "hamiltonian_simulation": {
        "scope": "polynomial_core",
        "criteria": (
            {
                "id": "classical_reference",
                "description": (
                    "The exact dense matrix exponential and evolved reference state "
                    "are present and finite."
                ),
                "required_for_scope": True,
                "required_for_full_qsvt": True,
            },
            {
                "id": "polynomial_accuracy",
                "description": (
                    "The cosine/sine polynomial pair meets the declared operator and "
                    "state error tolerance."
                ),
                "required_for_scope": True,
                "required_for_full_qsvt": True,
            },
            {
                "id": "norm_preservation",
                "description": (
                    "The evolved-state norm drift remains within the declared "
                    "acceptance tolerance."
                ),
                "required_for_scope": True,
                "required_for_full_qsvt": True,
            },
            {
                "id": "finite_qsvt_execution",
                "description": (
                    "Even/odd sequences are coherently combined and executed through "
                    "a finite block-encoded QSVT circuit."
                ),
                "required_for_scope": False,
                "required_for_full_qsvt": True,
            },
            {
                "id": "diagnostics_and_resources",
                "description": (
                    "A component error ledger and concrete encoding-aware circuit "
                    "resource ledger are present."
                ),
                "required_for_scope": False,
                "required_for_full_qsvt": True,
            },
        ),
    },
}

FLAGSHIP_ACCEPTANCE_MATRIX: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {workflow: MappingProxyType(contract) for workflow, contract in _MATRIX.items()}
)


def flagship_acceptance_matrix() -> dict[str, dict[str, object]]:
    """Return a mutable, machine-readable copy of the flagship criteria."""
    return {
        workflow: {
            "scope": contract["scope"],
            "criteria": [
                dict(criterion)
                for criterion in cast(
                    Sequence[Mapping[str, object]], contract["criteria"]
                )
            ],
        }
        for workflow, contract in FLAGSHIP_ACCEPTANCE_MATRIX.items()
    }


def evaluate_poisson_acceptance(result: Any) -> dict[str, object]:
    """Evaluate a :class:`qsvt.flagship.PoissonQSVTResult`."""
    tolerance = float(result.degree_search.tolerance)
    phase_tolerance = float(result.phase_reconstruction_tolerance)
    direct_residual = float(
        np.linalg.norm(result.matrix @ result.direct_solution - result.rhs)
    )
    reference_scale = max(1.0, float(np.linalg.norm(result.rhs)))
    execution_error = (
        None
        if result.execution is None
        else result.execution.logical_output_relative_error
    )
    checks = {
        "classical_reference": _observed_check(
            bool(
                np.all(np.isfinite(result.direct_solution))
                and direct_residual <= 1e-9 * reference_scale
                and bool(result.conjugate_gradient.get("converged", False))
            ),
            observed={
                "direct_residual_norm": direct_residual,
                "cg_converged": bool(result.conjugate_gradient.get("converged", False)),
            },
            threshold=1e-9 * reference_scale,
        ),
        "polynomial_accuracy": _observed_check(
            bool(
                result.degree_search.met_tolerance
                and result.polynomial_relative_error <= tolerance
            ),
            observed=result.polynomial_relative_error,
            threshold=tolerance,
        ),
        "phase_synthesis": _observed_check(
            bool(
                result.synthesis.succeeded
                and result.synthesis.reconstruction_max_error is not None
                and result.synthesis.reconstruction_max_error <= phase_tolerance
            ),
            observed=result.synthesis.reconstruction_max_error,
            threshold=phase_tolerance,
        ),
        "finite_qsvt_execution": _observed_check(
            bool(
                result.execution_requested
                and result.execution is not None
                and result.execution.succeeded
                and execution_error is not None
                and execution_error <= tolerance
                and result.circuit_relative_error is not None
                and result.circuit_relative_error <= tolerance
            ),
            observed={
                "execution_requested": result.execution_requested,
                "execution_succeeded": (
                    False if result.execution is None else result.execution.succeeded
                ),
                "circuit_vs_polynomial_error": execution_error,
                "circuit_solution_relative_error": result.circuit_relative_error,
            },
            threshold=tolerance,
        ),
        "diagnostics_and_resources": _observed_check(
            bool(
                result.resource_estimate.normalization_alpha > 0.0
                and result.error_budget
                and result.physical_observables
            ),
            observed={
                "normalization_alpha": result.resource_estimate.normalization_alpha,
                "error_components": sorted(result.error_budget),
                "observables": sorted(result.physical_observables),
            },
        ),
    }
    return _build_report("poisson_qsvt", checks)


def evaluate_spectral_filter_acceptance(result: Any) -> dict[str, object]:
    """Evaluate a :class:`qsvt.flagship.SpectralFilterQSVTResult`."""
    tolerance = float(result.degree_search.tolerance)
    phase_tolerance = float(result.phase_reconstruction_tolerance)
    execution_error = (
        None
        if result.execution is None
        else result.execution.logical_output_relative_error
    )
    checks = {
        "classical_reference": _observed_check(
            bool(
                np.all(np.isfinite(result.reference_projector))
                and np.all(np.isfinite(result.reference_state))
                and result.reference_success_probability > 0.0
            ),
            observed={
                "reference_success_probability": (result.reference_success_probability),
                "projector_shape": list(result.reference_projector.shape),
            },
        ),
        "polynomial_accuracy": _observed_check(
            bool(
                result.degree_search.met_tolerance
                and result.polynomial_operator_error <= tolerance
            ),
            observed=result.polynomial_operator_error,
            threshold=tolerance,
        ),
        "phase_synthesis": _observed_check(
            bool(
                result.synthesis.succeeded
                and result.synthesis.reconstruction_max_error is not None
                and result.synthesis.reconstruction_max_error <= phase_tolerance
            ),
            observed=result.synthesis.reconstruction_max_error,
            threshold=phase_tolerance,
        ),
        "finite_qsvt_execution": _observed_check(
            bool(
                result.execution_requested
                and result.execution is not None
                and result.execution.succeeded
                and execution_error is not None
                and execution_error <= tolerance
            ),
            observed={
                "execution_requested": result.execution_requested,
                "execution_succeeded": (
                    False if result.execution is None else result.execution.succeeded
                ),
                "circuit_vs_polynomial_error": execution_error,
            },
            threshold=tolerance,
        ),
        "diagnostics_and_resources": _observed_check(
            bool(
                result.resource_estimate.normalization_alpha > 0.0
                and result.error_budget
                and result.observable_values
                and result.polynomial_success_probability > 0.0
            ),
            observed={
                "normalization_alpha": result.resource_estimate.normalization_alpha,
                "polynomial_success_probability": (
                    result.polynomial_success_probability
                ),
                "error_components": sorted(result.error_budget),
                "observables": sorted(result.observable_values),
            },
        ),
    }
    return _build_report("spectral_filter_qsvt", checks)


def evaluate_hamiltonian_simulation_acceptance(result: Any) -> dict[str, object]:
    """Evaluate a dense polynomial-core Hamiltonian simulation result."""
    tolerance = float(result.acceptance_tolerance)
    checks = {
        "classical_reference": _observed_check(
            bool(
                np.all(np.isfinite(result.reference_unitary))
                and np.all(np.isfinite(result.reference_state))
            ),
            observed={"unitary_shape": list(result.reference_unitary.shape)},
        ),
        "polynomial_accuracy": _observed_check(
            bool(
                result.operator_relative_error <= tolerance
                and result.state_relative_error <= tolerance
            ),
            observed={
                "operator_relative_error": result.operator_relative_error,
                "state_relative_error": result.state_relative_error,
            },
            threshold=tolerance,
        ),
        "norm_preservation": _observed_check(
            bool(result.norm_drift <= tolerance),
            observed=result.norm_drift,
            threshold=tolerance,
        ),
        "finite_qsvt_execution": _observed_check(
            False,
            observed={
                "executed": False,
                "reason": (
                    "coherent even/odd QSVT sequence combination is not implemented"
                ),
            },
        ),
        "diagnostics_and_resources": _observed_check(
            False,
            observed={
                "component_error_ledger": False,
                "encoding_aware_circuit_resources": False,
            },
        ),
    }
    return _build_report("hamiltonian_simulation", checks)


def _observed_check(
    passed: bool,
    *,
    observed: object,
    threshold: float | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {"passed": bool(passed), "observed": observed}
    if threshold is not None:
        payload["threshold"] = float(threshold)
    return payload


def _build_report(
    workflow: str,
    observed_checks: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    contract = FLAGSHIP_ACCEPTANCE_MATRIX[workflow]
    criteria = {
        str(criterion["id"]): criterion
        for criterion in cast(Sequence[Mapping[str, object]], contract["criteria"])
    }
    checks: list[dict[str, object]] = []
    for criterion_id, criterion in criteria.items():
        observed = observed_checks[criterion_id]
        checks.append({**dict(criterion), **dict(observed)})

    required = [check for check in checks if bool(check["required_for_scope"])]
    full = [check for check in checks if bool(check["required_for_full_qsvt"])]
    accepted = all(bool(check["passed"]) for check in required)
    full_accepted = all(bool(check["passed"]) for check in full)
    return {
        "schema_name": FLAGSHIP_ACCEPTANCE_SCHEMA_NAME,
        "schema_version": FLAGSHIP_ACCEPTANCE_SCHEMA_VERSION,
        "workflow": workflow,
        "scope": contract["scope"],
        "status": (
            "accepted_for_stated_scope" if accepted else "acceptance_criteria_not_met"
        ),
        "accepted_for_stated_scope": accepted,
        "full_qsvt_acceptance": full_accepted,
        "checks": checks,
    }


__all__ = [
    "FLAGSHIP_ACCEPTANCE_MATRIX",
    "FLAGSHIP_ACCEPTANCE_SCHEMA_NAME",
    "FLAGSHIP_ACCEPTANCE_SCHEMA_VERSION",
    "evaluate_hamiltonian_simulation_acceptance",
    "evaluate_poisson_acceptance",
    "evaluate_spectral_filter_acceptance",
    "flagship_acceptance_matrix",
]
