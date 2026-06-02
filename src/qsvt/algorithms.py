"""
End-to-end QSVT-style algorithm workflows.

These helpers combine existing polynomial design, spectral rescaling, QSVT
application, and classical diagnostics into small executable workflows.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._algorithm_reports import algorithm_truth_contract, scaled_operator_report
from .block_encoding import BlockEncoding, block_encode_matrix, verify_block_encoding
from .compatibility import qsvt_compatibility_report
from .design import (
    design_interval_projector_diagnostics,
    design_interval_projector_polynomial,
    design_positive_inverse_diagnostics,
    design_positive_inverse_polynomial,
)
from .diagnostics import expectation_value, ground_state_overlap, operator_error
from .matrix import qsvt_matrix_transform
from .matrix_functions import (
    ScaledPolynomial,
    design_gaussian_window_polynomial,
    design_imaginary_time_polynomial,
    design_real_time_evolution_polynomials,
    design_resolvent_polynomials,
)
from .rescaling import (
    ScaledOperator,
    rescale_hermitian_to_unit_interval,
    rescale_positive_semidefinite,
)
from .spectral import (
    apply_function_to_hermitian,
    apply_polynomial_to_hermitian,
    eigh_hermitian,
)


@dataclass(frozen=True)
class LinearSystemWorkflowResult:
    """
    Structured output from a positive-definite linear-system workflow.
    """

    coeffs: np.ndarray
    scaled_operator: ScaledOperator
    rhs: np.ndarray
    classical_solution: np.ndarray
    polynomial_solution: np.ndarray
    polynomial_residual_norm: float
    polynomial_relative_error: float
    gamma: float
    degree: int
    diagnostics: dict[str, object]
    compatibility: dict[str, object]
    qsvt_solution: np.ndarray | None = None
    qsvt_residual_norm: float | None = None
    qsvt_relative_error: float | None = None
    qsvt_error: str | None = None

    def as_report(self) -> dict[str, Any]:
        """
        Return a report-style dictionary for JSON conversion or persistence.
        """
        qsvt_check = (
            "failed"
            if self.qsvt_error is not None
            else "succeeded"
            if self.qsvt_solution is not None
            else "not_attempted"
        )
        return {
            "mode": "linear-system-workflow",
            "truth_contract": algorithm_truth_contract(
                "linear-system-workflow",
                target="positive-definite linear-system inverse action",
                qsvt_check=qsvt_check,
            ),
            "gamma": self.gamma,
            "degree": self.degree,
            "coeffs": self.coeffs,
            "scaled_operator": scaled_operator_report(self.scaled_operator),
            "rhs": self.rhs,
            "classical_solution": self.classical_solution,
            "polynomial_solution": self.polynomial_solution,
            "polynomial_residual_norm": self.polynomial_residual_norm,
            "polynomial_relative_error": self.polynomial_relative_error,
            "qsvt_solution": self.qsvt_solution,
            "qsvt_residual_norm": self.qsvt_residual_norm,
            "qsvt_relative_error": self.qsvt_relative_error,
            "qsvt_error": self.qsvt_error,
            "diagnostics": self.diagnostics,
            "compatibility": self.compatibility,
        }


@dataclass(frozen=True)
class GroundStateFilteringWorkflowResult:
    """
    Structured output from a Gaussian ground-state filtering workflow.
    """

    coeffs: np.ndarray
    scaled_operator: ScaledOperator
    input_state: np.ndarray
    filtered_state: np.ndarray
    unnormalized_filtered_state: np.ndarray
    reference_filtered_state: np.ndarray
    ground_energy: float
    filtered_energy: float | complex
    ground_state_overlap: float
    reference_state_error: float
    center: float
    width: float
    degree: int
    polynomial_operator: np.ndarray
    reference_operator: np.ndarray
    operator_relative_error: float

    def as_report(self) -> dict[str, Any]:
        """
        Return a report-style dictionary for JSON conversion or persistence.
        """
        return {
            "mode": "ground-state-filtering-workflow",
            "truth_contract": algorithm_truth_contract(
                "ground-state-filtering-workflow",
                target="low-energy Gaussian spectral filtering",
            ),
            "degree": self.degree,
            "center": self.center,
            "width": self.width,
            "coeffs": self.coeffs,
            "scaled_operator": scaled_operator_report(self.scaled_operator),
            "input_state": self.input_state,
            "filtered_state": self.filtered_state,
            "unnormalized_filtered_state": self.unnormalized_filtered_state,
            "reference_filtered_state": self.reference_filtered_state,
            "ground_energy": self.ground_energy,
            "filtered_energy": self.filtered_energy,
            "ground_state_overlap": self.ground_state_overlap,
            "reference_state_error": self.reference_state_error,
            "polynomial_operator": self.polynomial_operator,
            "reference_operator": self.reference_operator,
            "operator_relative_error": self.operator_relative_error,
        }


@dataclass(frozen=True)
class HamiltonianSimulationWorkflowResult:
    """
    Structured output from a real-time Hamiltonian simulation workflow.
    """

    cos_coeffs: np.ndarray
    sin_coeffs: np.ndarray
    scaled_operator: ScaledOperator
    input_state: np.ndarray
    evolved_state: np.ndarray
    reference_state: np.ndarray
    polynomial_unitary: np.ndarray
    reference_unitary: np.ndarray
    time: float
    degree: int
    scaled_time: float
    state_relative_error: float
    operator_relative_error: float
    norm_drift: float

    def as_report(self) -> dict[str, Any]:
        """
        Return a report-style dictionary for JSON conversion or persistence.
        """
        return {
            "mode": "hamiltonian-simulation-workflow",
            "truth_contract": algorithm_truth_contract(
                "hamiltonian-simulation-workflow",
                target="real-time Hamiltonian matrix exponential action",
            ),
            "time": self.time,
            "degree": self.degree,
            "scaled_time": self.scaled_time,
            "cos_coeffs": self.cos_coeffs,
            "sin_coeffs": self.sin_coeffs,
            "scaled_operator": scaled_operator_report(self.scaled_operator),
            "input_state": self.input_state,
            "evolved_state": self.evolved_state,
            "reference_state": self.reference_state,
            "polynomial_unitary": self.polynomial_unitary,
            "reference_unitary": self.reference_unitary,
            "state_relative_error": self.state_relative_error,
            "operator_relative_error": self.operator_relative_error,
            "norm_drift": self.norm_drift,
        }


@dataclass(frozen=True)
class ResolventWorkflowResult:
    """
    Structured output from a Green's-function / resolvent workflow.
    """

    real_coeffs: np.ndarray
    imag_coeffs: np.ndarray
    scaled_operator: ScaledOperator
    source: np.ndarray | None
    polynomial_operator: np.ndarray
    reference_operator: np.ndarray
    polynomial_response: np.ndarray | None
    reference_response: np.ndarray | None
    omega: float
    eta: float
    degree: int
    operator_relative_error: float
    response_relative_error: float | None

    def as_report(self) -> dict[str, Any]:
        """
        Return a report-style dictionary for JSON conversion or persistence.
        """
        return {
            "mode": "resolvent-workflow",
            "truth_contract": algorithm_truth_contract(
                "resolvent-workflow",
                target="Green's-function resolvent matrix function",
            ),
            "omega": self.omega,
            "eta": self.eta,
            "degree": self.degree,
            "real_coeffs": self.real_coeffs,
            "imag_coeffs": self.imag_coeffs,
            "scaled_operator": scaled_operator_report(self.scaled_operator),
            "source": self.source,
            "polynomial_operator": self.polynomial_operator,
            "reference_operator": self.reference_operator,
            "polynomial_response": self.polynomial_response,
            "reference_response": self.reference_response,
            "operator_relative_error": self.operator_relative_error,
            "response_relative_error": self.response_relative_error,
        }


@dataclass(frozen=True)
class SpectralDensityWorkflowResult:
    """
    Structured output from a Gaussian spectral-density workflow.
    """

    centers: np.ndarray
    width: float
    degree: int
    coeffs_by_center: list[np.ndarray]
    scaled_operator: ScaledOperator
    polynomial_trace_density: np.ndarray
    reference_trace_density: np.ndarray
    trace_density_error: float
    state: np.ndarray | None = None
    polynomial_state_weights: np.ndarray | None = None
    reference_state_weights: np.ndarray | None = None
    state_weight_error: float | None = None

    def as_report(self) -> dict[str, Any]:
        """
        Return a report-style dictionary for JSON conversion or persistence.
        """
        return {
            "mode": "spectral-density-workflow",
            "truth_contract": algorithm_truth_contract(
                "spectral-density-workflow",
                target="Gaussian-window spectral-density estimate",
            ),
            "centers": self.centers,
            "width": self.width,
            "degree": self.degree,
            "coeffs_by_center": self.coeffs_by_center,
            "scaled_operator": scaled_operator_report(self.scaled_operator),
            "polynomial_trace_density": self.polynomial_trace_density,
            "reference_trace_density": self.reference_trace_density,
            "trace_density_error": self.trace_density_error,
            "state": self.state,
            "polynomial_state_weights": self.polynomial_state_weights,
            "reference_state_weights": self.reference_state_weights,
            "state_weight_error": self.state_weight_error,
        }


@dataclass(frozen=True)
class SpectralThresholdingWorkflowResult:
    """
    Structured output from a spectral interval-projector workflow.
    """

    coeffs: np.ndarray
    scaled_operator: ScaledOperator
    polynomial_projector: np.ndarray
    reference_projector: np.ndarray
    lower: float
    upper: float
    scaled_lower: float
    scaled_upper: float
    sharpness: float
    degree: int
    exact_rank: int
    polynomial_rank_proxy: float | complex
    operator_relative_error: float
    leakage_outside_interval: float
    diagnostics: dict[str, object]
    state: np.ndarray | None = None
    polynomial_state_weight: float | complex | None = None
    reference_state_weight: float | complex | None = None
    state_weight_error: float | None = None

    def as_report(self) -> dict[str, Any]:
        """
        Return a report-style dictionary for JSON conversion or persistence.
        """
        return {
            "mode": "spectral-thresholding-workflow",
            "truth_contract": algorithm_truth_contract(
                "spectral-thresholding-workflow",
                target="smooth spectral interval-projector approximation",
            ),
            "degree": self.degree,
            "lower": self.lower,
            "upper": self.upper,
            "scaled_lower": self.scaled_lower,
            "scaled_upper": self.scaled_upper,
            "sharpness": self.sharpness,
            "coeffs": self.coeffs,
            "scaled_operator": scaled_operator_report(self.scaled_operator),
            "polynomial_projector": self.polynomial_projector,
            "reference_projector": self.reference_projector,
            "exact_rank": self.exact_rank,
            "polynomial_rank_proxy": self.polynomial_rank_proxy,
            "operator_relative_error": self.operator_relative_error,
            "leakage_outside_interval": self.leakage_outside_interval,
            "diagnostics": self.diagnostics,
            "state": self.state,
            "polynomial_state_weight": self.polynomial_state_weight,
            "reference_state_weight": self.reference_state_weight,
            "state_weight_error": self.state_weight_error,
        }


@dataclass(frozen=True)
class ThermalGibbsWorkflowResult:
    """
    Structured output from an imaginary-time / Gibbs weighting workflow.
    """

    coeffs: np.ndarray
    prefactor: float
    scaled_operator: ScaledOperator
    polynomial_boltzmann_operator: np.ndarray
    reference_boltzmann_operator: np.ndarray
    polynomial_gibbs_state: np.ndarray
    reference_gibbs_state: np.ndarray
    beta: float
    degree: int
    polynomial_partition_function: float | complex
    reference_partition_function: float | complex
    operator_relative_error: float
    density_matrix_relative_error: float
    state: np.ndarray | None = None
    polynomial_weighted_state: np.ndarray | None = None
    reference_weighted_state: np.ndarray | None = None
    weighted_state_error: float | None = None

    def as_report(self) -> dict[str, Any]:
        """
        Return a report-style dictionary for JSON conversion or persistence.
        """
        return {
            "mode": "thermal-gibbs-workflow",
            "truth_contract": algorithm_truth_contract(
                "thermal-gibbs-workflow",
                target="imaginary-time Boltzmann weighting and Gibbs normalization",
            ),
            "beta": self.beta,
            "degree": self.degree,
            "coeffs": self.coeffs,
            "prefactor": self.prefactor,
            "scaled_operator": scaled_operator_report(self.scaled_operator),
            "polynomial_boltzmann_operator": self.polynomial_boltzmann_operator,
            "reference_boltzmann_operator": self.reference_boltzmann_operator,
            "polynomial_gibbs_state": self.polynomial_gibbs_state,
            "reference_gibbs_state": self.reference_gibbs_state,
            "polynomial_partition_function": self.polynomial_partition_function,
            "reference_partition_function": self.reference_partition_function,
            "operator_relative_error": self.operator_relative_error,
            "density_matrix_relative_error": self.density_matrix_relative_error,
            "state": self.state,
            "polynomial_weighted_state": self.polynomial_weighted_state,
            "reference_weighted_state": self.reference_weighted_state,
            "weighted_state_error": self.weighted_state_error,
        }


@dataclass(frozen=True)
class BlockEncodedQSVTWorkflowResult:
    """
    Structured output from a verified finite block-encoded QSVT workflow.
    """

    coeffs: np.ndarray
    block_encoding: BlockEncoding
    reference_operator: np.ndarray
    qsvt_operator: np.ndarray | None
    operator_relative_error: float | None
    verification: dict[str, object]
    degree: int
    state: np.ndarray | None = None
    reference_state: np.ndarray | None = None
    qsvt_state: np.ndarray | None = None
    state_relative_error: float | None = None
    qsvt_error: str | None = None

    def as_report(self) -> dict[str, Any]:
        """
        Return a report-style dictionary for JSON conversion or persistence.
        """
        qsvt_check = "failed" if self.qsvt_error is not None else "succeeded"
        return {
            "mode": "block-encoded-qsvt-workflow",
            "truth_contract": _block_encoded_qsvt_truth_contract(
                qsvt_check=qsvt_check,
            ),
            "degree": self.degree,
            "coeffs": self.coeffs,
            "block_encoding": self.block_encoding.as_report(),
            "verification": self.verification,
            "reference_operator": self.reference_operator,
            "qsvt_operator": self.qsvt_operator,
            "operator_relative_error": self.operator_relative_error,
            "state": self.state,
            "reference_state": self.reference_state,
            "qsvt_state": self.qsvt_state,
            "state_relative_error": self.state_relative_error,
            "qsvt_error": self.qsvt_error,
        }


def _block_encoded_qsvt_truth_contract(*, qsvt_check: str) -> dict[str, object]:
    if qsvt_check not in {"succeeded", "failed"}:
        raise ValueError("qsvt_check must be 'succeeded' or 'failed'.")

    return {
        "workflow": "block-encoded-qsvt-workflow",
        "target": "finite block-encoded positive-Hermitian polynomial transform",
        "implementation_kind": "verified-dense-block-encoded-qsvt-workflow",
        "truth_status": (
            "verified_block_encoding_and_qsvt_polynomial_transform"
            if qsvt_check == "succeeded"
            else "verified_block_encoding_qsvt_transform_failed"
        ),
        "is_end_to_end_quantum_algorithm": False,
        "implemented_components": [
            "explicit_dense_unitary_block_encoding",
            "top_left_block_verification",
            "unitarity_verification",
            "pennylane_qsvt_transform_when_synthesis_succeeds",
            "classical_spectral_polynomial_reference",
            "operator_and_optional_state_error_diagnostics",
        ],
        "pennylane_qsvt_check": qsvt_check,
        "conditional_qsvt_statement": (
            "The supplied finite matrix is encoded as the top-left block of an "
            "explicit unitary and transformed with a compatible QSVT polynomial "
            "on the normalized signal operator."
        ),
        "validation_scope": (
            "This validates a finite dense block encoding and QSVT transform. "
            "It does not make scalability, data-loading, readout, or hardware "
            "claims for larger problem families."
        ),
        "omitted_quantum_costs": [
            "scalable_oracle_or_sparse_block_encoding_construction",
            "input_state_preparation_or_data_loading",
            "measurement_or_readout_strategy",
            "amplitude_amplification_or_estimation",
            "fault_tolerant_synthesis",
            "hardware_compilation",
        ],
    }


def _validate_state(
    state: np.ndarray,
    dimension: int,
    name: str = "state",
) -> np.ndarray:
    vec = np.asarray(state, dtype=complex if np.iscomplexobj(state) else float)
    if vec.ndim != 1 or vec.shape[0] != dimension:
        raise ValueError(
            f"{name} must be a vector whose length matches matrix dimension."
        )
    if np.linalg.norm(vec) == 0.0:
        raise ValueError(f"{name} must be nonzero.")
    return vec


def _normalize_state(state: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(state)
    if norm == 0.0:
        raise ValueError("cannot normalize a zero state.")
    return state / norm


def _state_error(reference: np.ndarray, approximate: np.ndarray) -> float:
    phase = np.vdot(reference, approximate)
    if abs(phase) > 0.0:
        approximate = approximate * np.exp(-1j * np.angle(phase))
    return _relative_error(reference, approximate)


def _validate_linear_system_inputs(
    matrix: np.ndarray,
    rhs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.asarray(matrix)
    b = np.asarray(rhs, dtype=complex if np.iscomplexobj(rhs) else float)

    evals, _ = eigh_hermitian(A)
    if evals[0] <= 0.0:
        raise ValueError("matrix must be positive definite.")
    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError("rhs must be a vector whose length matches matrix dimension.")

    dtype = complex if np.iscomplexobj(A) or np.iscomplexobj(b) else float
    return A.astype(dtype, copy=False), b.astype(dtype, copy=False), evals


def _relative_error(reference: np.ndarray, approximate: np.ndarray) -> float:
    denom = np.linalg.norm(reference)
    diff = np.linalg.norm(approximate - reference)
    if denom == 0.0:
        return float(diff)
    return float(diff / denom)


def _state_weight_error(
    state: np.ndarray | None,
    reference: np.ndarray | None,
    approximate: np.ndarray | None,
) -> float | None:
    if state is None:
        return None
    if reference is None or approximate is None:
        raise ValueError("state-weight arrays are required when state is provided.")
    return _relative_error(reference, approximate)


def _gaussian_window_function(
    center: float,
    width: float,
) -> Callable[[np.ndarray], np.ndarray]:
    def gaussian_window(x: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * ((x - center) / width) ** 2)

    return gaussian_window


def linear_system_workflow(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    degree: int,
    gamma: float | None = None,
    num_points: int = 2001,
    bounded_num_points: int = 4001,
    attempt_synthesis: bool = True,
    apply_qsvt: bool = True,
) -> LinearSystemWorkflowResult:
    """
    Solve a positive-definite linear system with a QSVT-style inverse workflow.

    The input matrix is scaled by its largest eigenvalue so the scaled spectrum
    lies in ``[gamma_actual, 1]``. A bounded polynomial approximating
    ``gamma / x`` is designed on that interval, then rescaled back to an
    approximation of ``A^{-1} b``.
    """
    A, b, evals = _validate_linear_system_inputs(matrix, rhs)
    scaled = rescale_positive_semidefinite(A)
    scaled_min = float(evals[0] / scaled.scale)

    if gamma is None:
        gamma = min(scaled_min, 1.0 - 1e-12)
    gamma = float(gamma)
    if not (0.0 < gamma < 1.0 and gamma <= scaled_min + 1e-12):
        raise ValueError(
            "gamma must satisfy 0 < gamma < 1 and be no larger than the "
            "scaled minimum eigenvalue."
        )

    coeffs = design_positive_inverse_polynomial(
        gamma=gamma,
        degree=degree,
        num_points=num_points,
    )
    diagnostics = design_positive_inverse_diagnostics(
        gamma=gamma,
        degree=degree,
        num_points=num_points,
        bounded_num_points=bounded_num_points,
    )
    compatibility = qsvt_compatibility_report(
        coeffs,
        bounded_num_points=bounded_num_points,
        attempt_synthesis=attempt_synthesis,
    )

    classical_solution = np.linalg.solve(A, b)
    polynomial_operator = apply_polynomial_to_hermitian(scaled.matrix, coeffs)
    solution_scale = gamma * scaled.scale
    polynomial_solution = (polynomial_operator @ b) / solution_scale
    polynomial_residual = A @ polynomial_solution - b

    qsvt_solution = None
    qsvt_residual_norm = None
    qsvt_relative_error = None
    qsvt_error = None

    if apply_qsvt:
        try:
            raw_qsvt = qsvt_matrix_transform(scaled.matrix, coeffs) @ b
            qsvt_solution = raw_qsvt / solution_scale
            qsvt_residual = A @ qsvt_solution - b
            qsvt_residual_norm = float(np.linalg.norm(qsvt_residual))
            qsvt_relative_error = _relative_error(classical_solution, qsvt_solution)
        except Exception as exc:  # pragma: no cover - backend-dependent path
            qsvt_error = f"{type(exc).__name__}: {exc}"

    return LinearSystemWorkflowResult(
        coeffs=coeffs,
        scaled_operator=scaled,
        rhs=b,
        classical_solution=classical_solution,
        polynomial_solution=polynomial_solution,
        polynomial_residual_norm=float(np.linalg.norm(polynomial_residual)),
        polynomial_relative_error=_relative_error(
            classical_solution,
            polynomial_solution,
        ),
        gamma=gamma,
        degree=int(degree),
        diagnostics=diagnostics,
        compatibility=compatibility,
        qsvt_solution=qsvt_solution,
        qsvt_residual_norm=qsvt_residual_norm,
        qsvt_relative_error=qsvt_relative_error,
        qsvt_error=qsvt_error,
    )


def ground_state_filtering_workflow(
    matrix: np.ndarray,
    state: np.ndarray,
    *,
    degree: int,
    width: float = 0.25,
    center: float = -1.0,
    num_points: int = 2001,
) -> GroundStateFilteringWorkflowResult:
    """
    Apply a Gaussian low-energy filter to a trial state.

    The Hamiltonian is affinely mapped to ``[-1, 1]`` and filtered with a
    Gaussian window centered near the low-energy edge by default. The
    ``center`` and ``width`` parameters are expressed in the scaled spectral
    coordinate.
    """
    evals, _ = eigh_hermitian(matrix)
    scaled = rescale_hermitian_to_unit_interval(matrix)
    psi = _validate_state(state, scaled.matrix.shape[0])
    psi = _normalize_state(psi)

    if width <= 0.0:
        raise ValueError("width must be positive.")

    coeffs = design_gaussian_window_polynomial(
        center=center,
        width=width,
        degree=degree,
        num_points=num_points,
    )
    polynomial_operator = apply_polynomial_to_hermitian(scaled.matrix, coeffs)
    reference_operator = apply_function_to_hermitian(
        scaled.matrix,
        lambda x: np.exp(-0.5 * ((x - float(center)) / float(width)) ** 2),
    )

    unnormalized = polynomial_operator @ psi
    filtered = _normalize_state(unnormalized)
    reference_filtered = _normalize_state(reference_operator @ psi)

    return GroundStateFilteringWorkflowResult(
        coeffs=coeffs,
        scaled_operator=scaled,
        input_state=psi,
        filtered_state=filtered,
        unnormalized_filtered_state=unnormalized,
        reference_filtered_state=reference_filtered,
        ground_energy=float(evals[0]),
        filtered_energy=expectation_value(np.asarray(matrix), filtered),
        ground_state_overlap=ground_state_overlap(np.asarray(matrix), filtered),
        reference_state_error=_state_error(reference_filtered, filtered),
        center=float(center),
        width=float(width),
        degree=int(degree),
        polynomial_operator=polynomial_operator,
        reference_operator=reference_operator,
        operator_relative_error=operator_error(reference_operator, polynomial_operator),
    )


def hamiltonian_simulation_workflow(
    matrix: np.ndarray,
    state: np.ndarray,
    *,
    time: float,
    degree: int,
    num_points: int = 1001,
) -> HamiltonianSimulationWorkflowResult:
    """
    Approximate real-time evolution ``exp(-i H t)|psi>`` with polynomial pairs.
    """
    scaled = rescale_hermitian_to_unit_interval(matrix)
    psi = _normalize_state(_validate_state(state, scaled.matrix.shape[0]))
    polynomials = design_real_time_evolution_polynomials(
        time=time,
        scale=scaled.scale,
        degree=degree,
        num_points=num_points,
    )

    cos_op = apply_polynomial_to_hermitian(scaled.matrix, polynomials.cos_coeffs)
    sin_op = apply_polynomial_to_hermitian(scaled.matrix, polynomials.sin_coeffs)
    phase = np.exp(-1j * scaled.offset * float(time))
    polynomial_unitary = phase * (cos_op - 1j * sin_op)
    reference_unitary = apply_function_to_hermitian(
        np.asarray(matrix),
        lambda x: np.exp(-1j * float(time) * x),
    )
    evolved = polynomial_unitary @ psi
    reference = reference_unitary @ psi

    return HamiltonianSimulationWorkflowResult(
        cos_coeffs=polynomials.cos_coeffs,
        sin_coeffs=polynomials.sin_coeffs,
        scaled_operator=scaled,
        input_state=psi,
        evolved_state=evolved,
        reference_state=reference,
        polynomial_unitary=polynomial_unitary,
        reference_unitary=reference_unitary,
        time=float(time),
        degree=int(degree),
        scaled_time=polynomials.scaled_time,
        state_relative_error=_state_error(reference, evolved),
        operator_relative_error=operator_error(reference_unitary, polynomial_unitary),
        norm_drift=float(abs(np.linalg.norm(evolved) - 1.0)),
    )


def resolvent_workflow(
    matrix: np.ndarray,
    *,
    omega: float,
    eta: float,
    degree: int,
    source: np.ndarray | None = None,
    num_points: int = 2001,
) -> ResolventWorkflowResult:
    """
    Approximate the Green's-function resolvent ``(omega+i eta-H)^-1``.
    """
    scaled = rescale_hermitian_to_unit_interval(matrix)
    real_coeffs, imag_coeffs = design_resolvent_polynomials(
        omega=omega,
        eta=eta,
        scale=scaled.scale,
        offset=scaled.offset,
        degree=degree,
        num_points=num_points,
    )
    polynomial_operator = apply_polynomial_to_hermitian(
        scaled.matrix,
        real_coeffs,
    ) + 1j * apply_polynomial_to_hermitian(scaled.matrix, imag_coeffs)
    reference_operator = apply_function_to_hermitian(
        np.asarray(matrix),
        lambda x: 1.0 / (float(omega) + 1j * float(eta) - x),
    )

    source_vec = None
    polynomial_response = None
    reference_response = None
    response_error = None
    if source is not None:
        source_vec = _validate_state(source, scaled.matrix.shape[0], name="source")
        polynomial_response = polynomial_operator @ source_vec
        reference_response = reference_operator @ source_vec
        response_error = _relative_error(reference_response, polynomial_response)

    return ResolventWorkflowResult(
        real_coeffs=real_coeffs,
        imag_coeffs=imag_coeffs,
        scaled_operator=scaled,
        source=source_vec,
        polynomial_operator=polynomial_operator,
        reference_operator=reference_operator,
        polynomial_response=polynomial_response,
        reference_response=reference_response,
        omega=float(omega),
        eta=float(eta),
        degree=int(degree),
        operator_relative_error=operator_error(reference_operator, polynomial_operator),
        response_relative_error=response_error,
    )


def spectral_density_workflow(
    matrix: np.ndarray,
    centers: np.ndarray,
    *,
    width: float,
    degree: int,
    state: np.ndarray | None = None,
    num_points: int = 2001,
) -> SpectralDensityWorkflowResult:
    """
    Estimate Gaussian-window trace density and optional state spectral weights.

    The ``centers`` and ``width`` parameters are expressed in the original
    spectral coordinate of ``matrix``.
    """
    scaled = rescale_hermitian_to_unit_interval(matrix)
    center_values = np.asarray(centers, dtype=float)
    if center_values.ndim != 1:
        raise ValueError("centers must be a one-dimensional array.")
    if width <= 0.0:
        raise ValueError("width must be positive.")

    state_vec = None
    if state is not None:
        state_vec = _normalize_state(_validate_state(state, scaled.matrix.shape[0]))

    coeffs_by_center = []
    polynomial_density = []
    reference_density = []
    polynomial_weights = []
    reference_weights = []

    dimension = scaled.matrix.shape[0]
    for physical_center in center_values:
        scaled_center = (float(physical_center) - scaled.offset) / scaled.scale
        scaled_width = float(width) / scaled.scale
        coeffs = design_gaussian_window_polynomial(
            center=scaled_center,
            width=scaled_width,
            degree=degree,
            num_points=num_points,
        )
        coeffs_by_center.append(coeffs)
        polynomial_operator = apply_polynomial_to_hermitian(scaled.matrix, coeffs)
        reference_operator = apply_function_to_hermitian(
            np.asarray(matrix),
            _gaussian_window_function(float(physical_center), float(width)),
        )
        polynomial_density.append(np.trace(polynomial_operator) / dimension)
        reference_density.append(np.trace(reference_operator) / dimension)
        if state_vec is not None:
            polynomial_weights.append(
                np.vdot(state_vec, polynomial_operator @ state_vec)
            )
            reference_weights.append(np.vdot(state_vec, reference_operator @ state_vec))

    polynomial_density_arr = np.real_if_close(np.asarray(polynomial_density))
    reference_density_arr = np.real_if_close(np.asarray(reference_density))
    polynomial_weights_arr = (
        np.real_if_close(np.asarray(polynomial_weights))
        if state_vec is not None
        else None
    )
    reference_weights_arr = (
        np.real_if_close(np.asarray(reference_weights))
        if state_vec is not None
        else None
    )

    return SpectralDensityWorkflowResult(
        centers=center_values,
        width=float(width),
        degree=int(degree),
        coeffs_by_center=coeffs_by_center,
        scaled_operator=scaled,
        polynomial_trace_density=polynomial_density_arr,
        reference_trace_density=reference_density_arr,
        trace_density_error=_relative_error(
            reference_density_arr,
            polynomial_density_arr,
        ),
        state=state_vec,
        polynomial_state_weights=polynomial_weights_arr,
        reference_state_weights=reference_weights_arr,
        state_weight_error=_state_weight_error(
            state_vec,
            reference_weights_arr,
            polynomial_weights_arr,
        ),
    )


def spectral_thresholding_workflow(
    matrix: np.ndarray,
    *,
    lower: float,
    upper: float,
    degree: int,
    sharpness: float = 12.0,
    state: np.ndarray | None = None,
    num_points: int = 2001,
    bounded_num_points: int = 4001,
) -> SpectralThresholdingWorkflowResult:
    """
    Approximate a spectral interval projector with a QSVT-style polynomial.

    The physical interval ``[lower, upper]`` is mapped into the scaled
    ``[-1, 1]`` coordinate before designing a smooth interval-projector
    polynomial. The exact reference is the hard spectral projector onto
    eigenvalues of ``matrix`` that lie inside the physical interval.
    """
    lower = float(lower)
    upper = float(upper)
    if not lower < upper:
        raise ValueError("lower must be less than upper.")

    evals, _ = eigh_hermitian(matrix)
    if lower <= float(evals[0]) or upper >= float(evals[-1]):
        raise ValueError(
            "lower and upper must lie strictly inside the matrix spectral range."
        )

    scaled = rescale_hermitian_to_unit_interval(matrix)
    scaled_lower = (lower - scaled.offset) / scaled.scale
    scaled_upper = (upper - scaled.offset) / scaled.scale
    if not (-1.0 < scaled_lower < scaled_upper < 1.0):
        raise ValueError("scaled interval must lie strictly inside [-1, 1].")

    coeffs = design_interval_projector_polynomial(
        lower=scaled_lower,
        upper=scaled_upper,
        degree=degree,
        sharpness=sharpness,
        num_points=num_points,
    )
    diagnostics = design_interval_projector_diagnostics(
        lower=scaled_lower,
        upper=scaled_upper,
        degree=degree,
        sharpness=sharpness,
        num_points=num_points,
        bounded_num_points=bounded_num_points,
    )
    polynomial_projector = apply_polynomial_to_hermitian(scaled.matrix, coeffs)
    reference_projector = apply_function_to_hermitian(
        np.asarray(matrix),
        lambda x: np.where((lower <= x) & (x <= upper), 1.0, 0.0),
    )

    inside_mask = (evals >= lower) & (evals <= upper)
    exact_rank = int(np.count_nonzero(inside_mask))
    outside_projector = apply_function_to_hermitian(
        np.asarray(matrix),
        lambda x: np.where((x < lower) | (x > upper), 1.0, 0.0),
    )
    leakage = float(np.linalg.norm(outside_projector @ polynomial_projector))

    state_vec = None
    polynomial_weight = None
    reference_weight = None
    state_weight_error = None
    if state is not None:
        state_vec = _normalize_state(_validate_state(state, scaled.matrix.shape[0]))
        polynomial_weight = np.real_if_close(
            np.vdot(state_vec, polynomial_projector @ state_vec)
        ).item()
        reference_weight = np.real_if_close(
            np.vdot(state_vec, reference_projector @ state_vec)
        ).item()
        state_weight_error = float(abs(polynomial_weight - reference_weight))

    return SpectralThresholdingWorkflowResult(
        coeffs=coeffs,
        scaled_operator=scaled,
        polynomial_projector=polynomial_projector,
        reference_projector=reference_projector,
        lower=lower,
        upper=upper,
        scaled_lower=float(scaled_lower),
        scaled_upper=float(scaled_upper),
        sharpness=float(sharpness),
        degree=int(degree),
        exact_rank=exact_rank,
        polynomial_rank_proxy=np.real_if_close(np.trace(polynomial_projector)).item(),
        operator_relative_error=operator_error(
            reference_projector,
            polynomial_projector,
        ),
        leakage_outside_interval=leakage,
        diagnostics=diagnostics,
        state=state_vec,
        polynomial_state_weight=polynomial_weight,
        reference_state_weight=reference_weight,
        state_weight_error=state_weight_error,
    )


def thermal_gibbs_workflow(
    matrix: np.ndarray,
    *,
    beta: float,
    degree: int,
    state: np.ndarray | None = None,
    num_points: int = 2001,
) -> ThermalGibbsWorkflowResult:
    """
    Approximate ``exp(-beta H)`` and the normalized Gibbs density matrix.
    """
    if beta < 0.0:
        raise ValueError("beta must be non-negative.")

    scaled = rescale_hermitian_to_unit_interval(matrix)
    design: ScaledPolynomial = design_imaginary_time_polynomial(
        beta=beta,
        scale=scaled.scale,
        offset=scaled.offset,
        degree=degree,
        num_points=num_points,
    )
    polynomial_boltzmann = design.prefactor * apply_polynomial_to_hermitian(
        scaled.matrix,
        design.coeffs,
    )
    reference_boltzmann = apply_function_to_hermitian(
        np.asarray(matrix),
        lambda x: np.exp(-float(beta) * x),
    )
    polynomial_partition = np.trace(polynomial_boltzmann)
    reference_partition = np.trace(reference_boltzmann)
    polynomial_gibbs = polynomial_boltzmann / polynomial_partition
    reference_gibbs = reference_boltzmann / reference_partition

    state_vec = None
    polynomial_weighted = None
    reference_weighted = None
    weighted_error = None
    if state is not None:
        state_vec = _normalize_state(_validate_state(state, scaled.matrix.shape[0]))
        polynomial_weighted = polynomial_boltzmann @ state_vec
        reference_weighted = reference_boltzmann @ state_vec
        weighted_error = _relative_error(reference_weighted, polynomial_weighted)

    return ThermalGibbsWorkflowResult(
        coeffs=design.coeffs,
        prefactor=design.prefactor,
        scaled_operator=scaled,
        polynomial_boltzmann_operator=polynomial_boltzmann,
        reference_boltzmann_operator=reference_boltzmann,
        polynomial_gibbs_state=polynomial_gibbs,
        reference_gibbs_state=reference_gibbs,
        beta=float(beta),
        degree=int(degree),
        polynomial_partition_function=np.real_if_close(polynomial_partition).item(),
        reference_partition_function=np.real_if_close(reference_partition).item(),
        operator_relative_error=operator_error(
            reference_boltzmann,
            polynomial_boltzmann,
        ),
        density_matrix_relative_error=operator_error(reference_gibbs, polynomial_gibbs),
        state=state_vec,
        polynomial_weighted_state=polynomial_weighted,
        reference_weighted_state=reference_weighted,
        weighted_state_error=weighted_error,
    )


def block_encoded_qsvt_workflow(
    matrix: np.ndarray,
    coeffs: np.ndarray,
    *,
    alpha: float | None = None,
    state: np.ndarray | None = None,
    max_signal_norm: float = 0.8,
    block_atol: float = 1e-10,
    unitary_atol: float = 1e-10,
    encoding_wires: list[int] | None = None,
    wire_order: list[int] | None = None,
    block_encoding: str = "embedding",
    real_output: bool = True,
) -> BlockEncodedQSVTWorkflowResult:
    """
    Apply a QSVT polynomial to a verified finite block-encoded positive matrix.

    The workflow first constructs an explicit dense unitary whose top-left
    block is ``matrix / alpha``. QSVT is then applied to that normalized signal
    operator and compared with the exact spectral polynomial ``P(matrix /
    alpha)``. Polynomial coefficients are interpreted in the normalized signal
    coordinate. The direct PennyLane comparison used here is restricted to
    positive-semidefinite Hermitian signals, where the package's matrix-QSVT
    wrapper agrees with ordinary spectral polynomial functional calculus.
    """
    evals, _ = eigh_hermitian(matrix)
    if evals[0] < -1e-10:
        raise ValueError("matrix must be positive semidefinite for this workflow.")
    A = np.asarray(matrix, dtype=complex if np.iscomplexobj(matrix) else float)
    poly = np.asarray(coeffs, dtype=float)
    if poly.ndim != 1 or poly.size == 0:
        raise ValueError("coeffs must contain at least one coefficient.")
    if not np.all(np.isfinite(poly)):
        raise ValueError("coeffs must be finite.")
    if not (0.0 < max_signal_norm <= 1.0):
        raise ValueError("max_signal_norm must satisfy 0 < max_signal_norm <= 1.")

    if alpha is None:
        norm = float(np.linalg.norm(A, ord=2))
        alpha = max(1.0, norm / float(max_signal_norm))

    encoding = block_encode_matrix(A, alpha=alpha)
    verification = verify_block_encoding(
        encoding,
        block_atol=block_atol,
        unitary_atol=unitary_atol,
    )
    if not verification["block_encoding_verified"]:
        raise ValueError("block encoding failed top-left block verification.")
    if not verification["unitary_verified"]:
        raise ValueError("block encoding failed unitarity verification.")

    signal_evals = evals / encoding.alpha
    if np.max(np.abs(signal_evals)) > 1.0 + 1e-10:
        raise ValueError("normalized Hermitian spectrum must lie in [-1, 1].")

    reference_operator = apply_polynomial_to_hermitian(
        encoding.signal_operator,
        poly,
    )

    state_vec = None
    reference_state = None
    qsvt_state = None
    state_error = None
    if state is not None:
        state_vec = _validate_state(state, A.shape[0])
        reference_state = reference_operator @ state_vec

    qsvt_operator = None
    operator_relative = None
    qsvt_error = None
    try:
        qsvt_operator = qsvt_matrix_transform(
            encoding.signal_operator,
            poly,
            encoding_wires=encoding_wires,
            wire_order=wire_order,
            block_encoding=block_encoding,
            real_output=real_output,
        )
        operator_relative = operator_error(reference_operator, qsvt_operator)
        if state_vec is not None:
            qsvt_state = qsvt_operator @ state_vec
            state_error = _relative_error(reference_state, qsvt_state)
    except Exception as exc:  # pragma: no cover - backend-dependent path
        qsvt_error = f"{type(exc).__name__}: {exc}"

    return BlockEncodedQSVTWorkflowResult(
        coeffs=poly,
        block_encoding=encoding,
        reference_operator=reference_operator,
        qsvt_operator=qsvt_operator,
        operator_relative_error=operator_relative,
        verification=verification,
        degree=int(poly.size - 1),
        state=state_vec,
        reference_state=reference_state,
        qsvt_state=qsvt_state,
        state_relative_error=state_error,
        qsvt_error=qsvt_error,
    )


__all__ = [
    "BlockEncodedQSVTWorkflowResult",
    "GroundStateFilteringWorkflowResult",
    "HamiltonianSimulationWorkflowResult",
    "LinearSystemWorkflowResult",
    "ResolventWorkflowResult",
    "SpectralDensityWorkflowResult",
    "SpectralThresholdingWorkflowResult",
    "ThermalGibbsWorkflowResult",
    "block_encoded_qsvt_workflow",
    "ground_state_filtering_workflow",
    "hamiltonian_simulation_workflow",
    "linear_system_workflow",
    "resolvent_workflow",
    "spectral_density_workflow",
    "spectral_thresholding_workflow",
    "thermal_gibbs_workflow",
]
