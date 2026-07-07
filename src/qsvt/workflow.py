"""
High-level polynomial design workflows.

This module provides a small structured API for callers that want the designed
coefficients, approximation diagnostics, and QSVT compatibility report from a
single operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .algorithms import (
    ground_state_filtering_workflow,
    hamiltonian_simulation_workflow,
    linear_system_workflow,
    resolvent_workflow,
    singular_value_filtering_workflow,
    singular_value_pseudoinverse_workflow,
    spectral_thresholding_workflow,
)
from .compatibility import qsvt_compatibility_report
from .design import (
    design_filter_diagnostics,
    design_filter_polynomial,
    design_interval_projector_diagnostics,
    design_interval_projector_polynomial,
    design_inverse_diagnostics,
    design_inverse_polynomial,
    design_power_diagnostics,
    design_power_polynomial,
    design_projector_diagnostics,
    design_projector_polynomial,
    design_sign_diagnostics,
    design_sign_polynomial,
    design_sqrt_diagnostics,
    design_sqrt_polynomial,
)
from .resources import qsvt_resource_report
from .synthesis import PhaseSynthesisResult, SynthesisRoutine, synthesize_phases

DesignKind = Literal[
    "inverse",
    "sign",
    "projector",
    "sqrt",
    "power",
    "filter",
    "interval_projector",
]

ProblemWorkflowTarget = Literal[
    "linear_system",
    "spectral_projector",
    "ground_state_filter",
    "hamiltonian_simulation",
    "resolvent",
    "singular_value_filter",
    "singular_value_pseudoinverse",
]
_PROBLEM_WORKFLOW_TARGETS = (
    "linear_system",
    "spectral_projector",
    "ground_state_filter",
    "hamiltonian_simulation",
    "resolvent",
    "singular_value_filter",
    "singular_value_pseudoinverse",
)


@dataclass(frozen=True)
class DesignWorkflowResult:
    """
    Structured output from a design workflow.
    """

    kind: DesignKind
    builder: str
    coeffs: np.ndarray
    diagnostics: dict[str, object]
    compatibility: dict[str, object]

    def as_report(self) -> dict[str, object]:
        """
        Return a report-style dictionary for JSON conversion or persistence.
        """
        return {
            "mode": "design-workflow",
            "kind": self.kind,
            "builder": self.builder,
            "coeffs": self.coeffs,
            "diagnostics": self.diagnostics,
            "compatibility": self.compatibility,
        }

    def resource_report(
        self,
        *,
        matrix_dimension: int | None = None,
        encoding_qubits: int | None = None,
        block_encoding: str = "dense-block-encoding",
    ) -> dict[str, object]:
        """
        Return a resource proxy report carrying this workflow's diagnostics.
        """
        report = qsvt_resource_report(
            self.coeffs,
            matrix_dimension=matrix_dimension,
            encoding_qubits=encoding_qubits,
            block_encoding=block_encoding,
            attempt_synthesis=False,
            diagnostics=self.diagnostics,
        )
        report.update(
            {
                "kind": self.kind,
                "builder": self.builder,
                "compatibility": self.compatibility,
            }
        )
        return report

    def synthesize(
        self,
        *,
        routine: SynthesisRoutine = "QSVT",
        angle_solver: str = "root-finding",
        reconstruction_num_points: int = 257,
        **solver_kwargs: Any,
    ) -> PhaseSynthesisResult:
        """Synthesize and validate phases for this designed polynomial."""
        return synthesize_phases(
            self.coeffs,
            routine=routine,
            angle_solver=angle_solver,
            reconstruction_num_points=reconstruction_num_points,
            **solver_kwargs,
        )


@dataclass(frozen=True)
class QSVTProblemWorkflowResult:
    """
    Uniform wrapper for a high-level finite QSVT problem workflow.

    The wrapped ``result`` is one of the existing workflow dataclasses from
    :mod:`qsvt.algorithms`. This wrapper adds a stable top-level report shape
    for users who want the roadmap-style path from problem definition through
    polynomial design, validation, resource proxies, and classical reference
    comparison.
    """

    target: str
    input_kind: str
    result: Any
    resource_reports: tuple[dict[str, object], ...]

    def as_report(self) -> dict[str, object]:
        """
        Return a report-style dictionary for JSON conversion or persistence.
        """
        result_report = _as_result_report(self.result)
        return {
            "schema_name": "qsvt-problem-workflow",
            "schema_version": "1.0",
            "mode": "qsvt-problem-workflow",
            "target": self.target,
            "input_kind": self.input_kind,
            "implementation_kind": result_report.get(
                "implementation_kind",
                "finite-qsvt-style-workflow",
            ),
            "truth_contract": {
                "implemented_components": [
                    "finite_problem_definition",
                    "package_level_target_workflow",
                    "bounded_polynomial_design_or_matrix_function_approximation",
                    "finite_classical_reference_comparison",
                    "resource_proxy_report",
                ],
                "omitted_quantum_components": [
                    "problem_specific_scalable_block_encoding",
                    "problem_specific_state_preparation",
                    "application_specific_readout_or_tomography",
                    "amplitude_amplification_or_estimation",
                    "fault_tolerant_synthesis",
                    "hardware_compilation",
                ],
                "truth_status": (
                    "finite_workflow_with_explicit_classical_reference; "
                    "resource reports are polynomial-level proxies unless the "
                    "wrapped workflow states otherwise"
                ),
            },
            "result": result_report,
            "resource_reports": list(self.resource_reports),
        }


def design_workflow(
    kind: DesignKind,
    *,
    degree: int,
    gamma: float = 0.25,
    a: float = 0.2,
    alpha: float = 0.5,
    cutoff: float = 0.45,
    lower: float = -0.25,
    upper: float = 0.25,
    sharpness: float = 12.0,
    num_points: int = 2001,
    bounded_num_points: int = 4001,
    attempt_synthesis: bool = True,
) -> DesignWorkflowResult:
    """
    Build a design polynomial with diagnostics and compatibility metadata.
    """
    polynomial_builders = {
        "inverse": lambda: design_inverse_polynomial(
            gamma=gamma,
            degree=degree,
            num_points=num_points,
        ),
        "sign": lambda: design_sign_polynomial(
            gamma=gamma,
            degree=degree,
            num_points=num_points,
        ),
        "projector": lambda: design_projector_polynomial(
            gamma=gamma,
            degree=degree,
            num_points=num_points,
        ),
        "sqrt": lambda: design_sqrt_polynomial(
            a=a,
            degree=degree,
            num_points=num_points,
        ),
        "power": lambda: design_power_polynomial(
            alpha=alpha,
            degree=degree,
            a=a,
            num_points=num_points,
        ),
        "filter": lambda: design_filter_polynomial(
            cutoff=cutoff,
            degree=degree,
            sharpness=sharpness,
            num_points=num_points,
        ),
        "interval_projector": lambda: design_interval_projector_polynomial(
            lower=lower,
            upper=upper,
            degree=degree,
            sharpness=sharpness,
            num_points=num_points,
        ),
    }
    diagnostic_builders = {
        "inverse": lambda: design_inverse_diagnostics(
            gamma=gamma,
            degree=degree,
            num_points=num_points,
            bounded_num_points=bounded_num_points,
        ),
        "sign": lambda: design_sign_diagnostics(
            gamma=gamma,
            degree=degree,
            num_points=num_points,
            bounded_num_points=bounded_num_points,
        ),
        "projector": lambda: design_projector_diagnostics(
            gamma=gamma,
            degree=degree,
            num_points=num_points,
            bounded_num_points=bounded_num_points,
        ),
        "sqrt": lambda: design_sqrt_diagnostics(
            a=a,
            degree=degree,
            num_points=num_points,
            bounded_num_points=bounded_num_points,
        ),
        "power": lambda: design_power_diagnostics(
            alpha=alpha,
            degree=degree,
            a=a,
            num_points=num_points,
            bounded_num_points=bounded_num_points,
        ),
        "filter": lambda: design_filter_diagnostics(
            cutoff=cutoff,
            degree=degree,
            sharpness=sharpness,
            num_points=num_points,
            bounded_num_points=bounded_num_points,
        ),
        "interval_projector": lambda: design_interval_projector_diagnostics(
            lower=lower,
            upper=upper,
            degree=degree,
            sharpness=sharpness,
            num_points=num_points,
            bounded_num_points=bounded_num_points,
        ),
    }

    if kind not in polynomial_builders:
        choices = ", ".join(polynomial_builders)
        raise ValueError(f"kind must be one of: {choices}.")

    coeffs = polynomial_builders[kind]()
    diagnostics = diagnostic_builders[kind]()
    compatibility = qsvt_compatibility_report(
        coeffs,
        bounded_num_points=bounded_num_points,
        attempt_synthesis=attempt_synthesis,
    )

    return DesignWorkflowResult(
        kind=kind,
        builder=f"design_{kind}_polynomial",
        coeffs=coeffs,
        diagnostics=diagnostics,
        compatibility=compatibility,
    )


def qsvt_problem_workflow(
    target: str,
    matrix: np.ndarray,
    *,
    rhs: np.ndarray | None = None,
    state: np.ndarray | None = None,
    source: np.ndarray | None = None,
    degree: int,
    gamma: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
    cutoff: float | None = None,
    sharpness: float = 12.0,
    width: float = 0.25,
    center: float = -1.0,
    time: float | None = None,
    omega: float | None = None,
    eta: float | None = None,
    num_points: int = 2001,
    bounded_num_points: int = 4001,
    attempt_synthesis: bool = False,
    apply_qsvt: bool = False,
) -> QSVTProblemWorkflowResult:
    """
    Run a high-level finite QSVT problem workflow.

    This facade implements the roadmap user journey for small auditable
    examples: define a finite problem, choose a target transform, run the
    package-level workflow, compare against a finite classical reference, and
    expose resource proxy reports. It deliberately keeps scalable block
    encodings, state preparation, readout, and hardware execution out of scope
    unless the wrapped lower-level workflow implements and reports them.
    """
    A = np.asarray(matrix)
    if A.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")

    result: Any
    if target == "linear_system":
        result = linear_system_workflow(
            A,
            _require_array(rhs, "rhs"),
            degree=degree,
            gamma=gamma,
            num_points=num_points,
            bounded_num_points=bounded_num_points,
            attempt_synthesis=attempt_synthesis,
            apply_qsvt=apply_qsvt,
        )
    elif target == "spectral_projector":
        result = spectral_thresholding_workflow(
            A,
            lower=_require_float(lower, "lower"),
            upper=_require_float(upper, "upper"),
            degree=degree,
            sharpness=sharpness,
            state=state,
            num_points=num_points,
            bounded_num_points=bounded_num_points,
        )
    elif target == "ground_state_filter":
        result = ground_state_filtering_workflow(
            A,
            _require_array(state, "state"),
            degree=degree,
            width=width,
            center=center,
            num_points=num_points,
        )
    elif target == "hamiltonian_simulation":
        result = hamiltonian_simulation_workflow(
            A,
            _require_array(state, "state"),
            time=_require_float(time, "time"),
            degree=degree,
            num_points=num_points,
        )
    elif target == "resolvent":
        result = resolvent_workflow(
            A,
            omega=_require_float(omega, "omega"),
            eta=_require_float(eta, "eta"),
            degree=degree,
            source=source,
            num_points=num_points,
        )
    elif target == "singular_value_filter":
        result = singular_value_filtering_workflow(
            A,
            degree=degree,
            cutoff=_require_float(cutoff, "cutoff"),
            sharpness=sharpness,
            input_vector=state,
            num_points=num_points,
        )
    elif target == "singular_value_pseudoinverse":
        result = singular_value_pseudoinverse_workflow(
            A,
            _require_array(rhs, "rhs"),
            degree=degree,
            cutoff=_require_float(cutoff, "cutoff"),
            num_points=num_points,
        )
    else:
        choices = ", ".join(_PROBLEM_WORKFLOW_TARGETS)
        raise ValueError(f"target must be one of: {choices}.")

    return QSVTProblemWorkflowResult(
        target=target,
        input_kind=_input_kind(A),
        result=result,
        resource_reports=_resource_reports_for_result(result, A),
    )


def _require_array(value: np.ndarray | None, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is required for this workflow target.")
    return np.asarray(value)


def _require_float(value: float | None, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required for this workflow target.")
    return float(value)


def _input_kind(matrix: np.ndarray) -> str:
    if matrix.shape[0] == matrix.shape[1]:
        return "finite-square-matrix"
    return "finite-rectangular-matrix"


def _as_result_report(result: Any) -> dict[str, Any]:
    if hasattr(result, "as_report"):
        report = result.as_report()
        if isinstance(report, dict):
            return report
    return {"result": result}


def _resource_reports_for_result(
    result: Any,
    matrix: np.ndarray,
) -> tuple[dict[str, object], ...]:
    dimension = int(max(matrix.shape))
    reports: list[dict[str, object]] = []
    for name, coeffs in _coefficient_sets(result):
        report = qsvt_resource_report(
            coeffs,
            matrix_dimension=dimension,
            block_encoding="problem-workflow-unspecified-access-model",
            attempt_synthesis=False,
        )
        report["component"] = name
        reports.append(report)
    return tuple(reports)


def _coefficient_sets(result: Any) -> tuple[tuple[str, np.ndarray], ...]:
    names = (
        "coeffs",
        "cos_coeffs",
        "sin_coeffs",
        "real_coeffs",
        "imag_coeffs",
    )
    coefficient_sets: list[tuple[str, np.ndarray]] = []
    for name in names:
        value = getattr(result, name, None)
        if value is not None:
            coefficient_sets.append((name, np.asarray(value, dtype=float)))
    return tuple(coefficient_sets)


__all__ = [
    "DesignKind",
    "DesignWorkflowResult",
    "ProblemWorkflowTarget",
    "QSVTProblemWorkflowResult",
    "design_workflow",
    "qsvt_problem_workflow",
]
