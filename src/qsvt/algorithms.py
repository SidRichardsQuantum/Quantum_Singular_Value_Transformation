"""
End-to-end QSVT-style algorithm workflows.

These helpers combine existing polynomial design, spectral rescaling, QSVT
application, and classical diagnostics into small executable workflows.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._algorithm_reports import algorithm_truth_contract, scaled_operator_report
from .approximation import chebyshev_fit_function
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
from .polynomials import chebyshev_to_monomial, eval_polynomial
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


def _real_time_phase_function(time: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return the spectral phase function exp(-i time x).
    """

    def phase(eigenvalues: np.ndarray) -> np.ndarray:
        return np.exp(-1j * time * eigenvalues)

    return phase


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
    scaled_min_eigenvalue: float
    scaled_max_eigenvalue: float
    condition_number_2: float
    gamma_condition_proxy: float
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
        if self.qsvt_error is not None:
            qsvt_check = "failed"
        elif self.qsvt_solution is not None:
            qsvt_check = "succeeded"
        else:
            qsvt_check = "not_attempted"
        return {
            "mode": "linear-system-workflow",
            "implementation_kind": "dense-spectral-polynomial-workflow",
            "truth_contract": algorithm_truth_contract(
                "linear-system-workflow",
                target="positive-definite linear-system inverse action",
                qsvt_check=qsvt_check,
            ),
            "gamma": self.gamma,
            "scaled_min_eigenvalue": self.scaled_min_eigenvalue,
            "scaled_max_eigenvalue": self.scaled_max_eigenvalue,
            "condition_number_2": self.condition_number_2,
            "gamma_condition_proxy": self.gamma_condition_proxy,
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
            "resource_proxy": linear_system_resource_proxy(
                degree=self.degree,
                gamma=self.gamma,
                scaled_min_eigenvalue=self.scaled_min_eigenvalue,
                condition_number_2=self.condition_number_2,
                polynomial_residual_norm=self.polynomial_residual_norm,
                polynomial_relative_error=self.polynomial_relative_error,
                qsvt_check=qsvt_check,
                attempted_pennylane_synthesis=bool(
                    self.compatibility.get("attempted_pennylane_synthesis", False)
                ),
            ),
        }


@dataclass(frozen=True)
class LinearSystemComparisonResult:
    """
    Structured comparison for small positive-definite linear-system solvers.
    """

    workflow: LinearSystemWorkflowResult
    rows: tuple[dict[str, Any], ...]
    dense_solution: np.ndarray
    cg_solution: np.ndarray | None
    reference_solution: np.ndarray
    notes: tuple[str, ...] = ()

    def as_report(self) -> dict[str, Any]:
        """
        Return a report-style dictionary for JSON conversion or persistence.
        """
        workflow_report = self.workflow.as_report()
        return {
            "mode": "linear-system-comparison-workflow",
            "implementation_kind": "linear-system-solver-comparison",
            "truth_contract": algorithm_truth_contract(
                "linear-system-comparison-workflow",
                target="positive-definite linear-system solver comparison",
                qsvt_check=_qsvt_check_from_result(self.workflow),
            ),
            "rows": list(self.rows),
            "reference_solution": self.reference_solution,
            "dense_solution": self.dense_solution,
            "cg_solution": self.cg_solution,
            "linear_system_workflow": workflow_report,
            "resource_proxy": workflow_report["resource_proxy"],
            "notes": list(self.notes),
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
            "implementation_kind": "dense-spectral-polynomial-workflow",
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
            "implementation_kind": "dense-spectral-polynomial-workflow",
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
class QuantumWalkSearchWorkflowResult:
    """
    Structured output from a continuous-time quantum-walk search workflow.
    """

    adjacency: np.ndarray
    search_hamiltonian: np.ndarray
    marked_vertex: int
    gamma: float
    oracle_strength: float
    initial_state: np.ndarray
    times: np.ndarray
    marked_probabilities: np.ndarray
    best_time: float
    best_probability: float
    exact_best_state: np.ndarray
    polynomial_best_state: np.ndarray
    polynomial_marked_probability: float
    probability_error: float
    state_relative_error: float
    cos_coeffs: np.ndarray
    sin_coeffs: np.ndarray
    scaled_operator: ScaledOperator
    polynomial_unitary: np.ndarray
    reference_unitary: np.ndarray
    degree: int
    scaled_time: float
    operator_relative_error: float
    norm_drift: float

    def as_report(self) -> dict[str, Any]:
        """
        Return a report-style dictionary for JSON conversion or persistence.
        """
        return {
            "mode": "quantum-walk-search-workflow",
            "implementation_kind": "dense-spectral-polynomial-workflow",
            "truth_contract": algorithm_truth_contract(
                "quantum-walk-search-workflow",
                target=(
                    "continuous-time quantum-walk search and polynomial phase "
                    "approximation"
                ),
            ),
            "marked_vertex": self.marked_vertex,
            "gamma": self.gamma,
            "oracle_strength": self.oracle_strength,
            "degree": self.degree,
            "scaled_time": self.scaled_time,
            "adjacency": self.adjacency,
            "search_hamiltonian": self.search_hamiltonian,
            "scaled_operator": scaled_operator_report(self.scaled_operator),
            "initial_state": self.initial_state,
            "times": self.times,
            "marked_probabilities": self.marked_probabilities,
            "best_time": self.best_time,
            "best_probability": self.best_probability,
            "exact_best_state": self.exact_best_state,
            "polynomial_best_state": self.polynomial_best_state,
            "polynomial_marked_probability": self.polynomial_marked_probability,
            "probability_error": self.probability_error,
            "state_relative_error": self.state_relative_error,
            "operator_relative_error": self.operator_relative_error,
            "norm_drift": self.norm_drift,
            "cos_coeffs": self.cos_coeffs,
            "sin_coeffs": self.sin_coeffs,
            "polynomial_unitary": self.polynomial_unitary,
            "reference_unitary": self.reference_unitary,
            "resource_proxy": quantum_walk_search_resource_proxy(
                graph_vertices=self.adjacency.shape[0],
                degree=self.degree,
                time_grid_size=self.times.size,
                best_probability=self.best_probability,
                probability_error=self.probability_error,
            ),
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
            "implementation_kind": "dense-spectral-polynomial-workflow",
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
            "implementation_kind": "dense-spectral-polynomial-workflow",
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
            "implementation_kind": "dense-spectral-polynomial-workflow",
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
            "implementation_kind": "dense-spectral-polynomial-workflow",
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
            "implementation_kind": "verified-dense-block-encoded-qsvt-workflow",
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


@dataclass(frozen=True)
class SingularValueFilteringWorkflowResult:
    """
    Structured output from a rectangular singular-value filter workflow.
    """

    coeffs: np.ndarray
    matrix: np.ndarray
    singular_values: np.ndarray
    normalized_singular_values: np.ndarray
    polynomial_matrix: np.ndarray
    reference_matrix: np.ndarray
    cutoff: float
    sharpness: float
    scale: float
    degree: int
    operator_relative_error: float
    input_vector: np.ndarray | None = None
    polynomial_output: np.ndarray | None = None
    reference_output: np.ndarray | None = None
    output_relative_error: float | None = None

    def as_report(self) -> dict[str, Any]:
        return {
            "mode": "singular-value-filtering-workflow",
            "implementation_kind": "dense-svd-polynomial-workflow",
            "truth_contract": algorithm_truth_contract(
                "singular-value-filtering-workflow",
                target="rectangular singular-value filter transform",
            ),
            "degree": self.degree,
            "cutoff": self.cutoff,
            "sharpness": self.sharpness,
            "scale": self.scale,
            "coeffs": self.coeffs,
            "matrix": self.matrix,
            "singular_values": self.singular_values,
            "normalized_singular_values": self.normalized_singular_values,
            "polynomial_matrix": self.polynomial_matrix,
            "reference_matrix": self.reference_matrix,
            "operator_relative_error": self.operator_relative_error,
            "input_vector": self.input_vector,
            "polynomial_output": self.polynomial_output,
            "reference_output": self.reference_output,
            "output_relative_error": self.output_relative_error,
        }


@dataclass(frozen=True)
class SingularValuePseudoinverseWorkflowResult:
    """
    Structured output from a truncated SVD pseudoinverse workflow.
    """

    coeffs: np.ndarray
    matrix: np.ndarray
    rhs: np.ndarray
    singular_values: np.ndarray
    normalized_singular_values: np.ndarray
    polynomial_solution: np.ndarray
    reference_solution: np.ndarray
    polynomial_pseudoinverse: np.ndarray
    reference_pseudoinverse: np.ndarray
    cutoff: float
    scale: float
    degree: int
    residual_norm: float
    reference_residual_norm: float
    solution_relative_error: float
    operator_relative_error: float

    def as_report(self) -> dict[str, Any]:
        return {
            "mode": "singular-value-pseudoinverse-workflow",
            "implementation_kind": "dense-svd-polynomial-workflow",
            "truth_contract": algorithm_truth_contract(
                "singular-value-pseudoinverse-workflow",
                target="truncated singular-value pseudoinverse action",
            ),
            "degree": self.degree,
            "cutoff": self.cutoff,
            "scale": self.scale,
            "coeffs": self.coeffs,
            "matrix": self.matrix,
            "rhs": self.rhs,
            "singular_values": self.singular_values,
            "normalized_singular_values": self.normalized_singular_values,
            "polynomial_solution": self.polynomial_solution,
            "reference_solution": self.reference_solution,
            "polynomial_pseudoinverse": self.polynomial_pseudoinverse,
            "reference_pseudoinverse": self.reference_pseudoinverse,
            "residual_norm": self.residual_norm,
            "reference_residual_norm": self.reference_residual_norm,
            "solution_relative_error": self.solution_relative_error,
            "operator_relative_error": self.operator_relative_error,
        }


@dataclass(frozen=True)
class FermiDiracWorkflowResult:
    """
    Structured output from a Fermi-Dirac occupation workflow.
    """

    coeffs: np.ndarray
    scaled_operator: ScaledOperator
    polynomial_occupation_operator: np.ndarray
    reference_occupation_operator: np.ndarray
    chemical_potential: float
    beta: float
    degree: int
    particle_number: float | complex
    reference_particle_number: float | complex
    operator_relative_error: float
    state: np.ndarray | None = None
    polynomial_state_occupation: float | complex | None = None
    reference_state_occupation: float | complex | None = None
    state_occupation_error: float | None = None

    def as_report(self) -> dict[str, Any]:
        return {
            "mode": "fermi-dirac-occupation-workflow",
            "implementation_kind": "dense-spectral-polynomial-workflow",
            "truth_contract": algorithm_truth_contract(
                "fermi-dirac-occupation-workflow",
                target="finite-temperature Fermi-Dirac spectral occupation",
            ),
            "chemical_potential": self.chemical_potential,
            "beta": self.beta,
            "degree": self.degree,
            "coeffs": self.coeffs,
            "scaled_operator": scaled_operator_report(self.scaled_operator),
            "polynomial_occupation_operator": self.polynomial_occupation_operator,
            "reference_occupation_operator": self.reference_occupation_operator,
            "particle_number": self.particle_number,
            "reference_particle_number": self.reference_particle_number,
            "operator_relative_error": self.operator_relative_error,
            "state": self.state,
            "polynomial_state_occupation": self.polynomial_state_occupation,
            "reference_state_occupation": self.reference_state_occupation,
            "state_occupation_error": self.state_occupation_error,
        }


@dataclass(frozen=True)
class MatrixLogEntropyWorkflowResult:
    """
    Structured output from a regularized matrix-log and entropy workflow.
    """

    log_coeffs: np.ndarray
    entropy_coeffs: np.ndarray
    scaled_operator: ScaledOperator
    polynomial_log_operator: np.ndarray
    reference_log_operator: np.ndarray
    polynomial_entropy_operator: np.ndarray
    reference_entropy_operator: np.ndarray
    epsilon: float
    degree: int
    polynomial_entropy: float | complex
    reference_entropy: float | complex
    log_operator_relative_error: float
    entropy_operator_relative_error: float

    def as_report(self) -> dict[str, Any]:
        return {
            "mode": "matrix-log-entropy-workflow",
            "implementation_kind": "dense-spectral-polynomial-workflow",
            "truth_contract": algorithm_truth_contract(
                "matrix-log-entropy-workflow",
                target="regularized matrix logarithm and x log x entropy density",
            ),
            "epsilon": self.epsilon,
            "degree": self.degree,
            "log_coeffs": self.log_coeffs,
            "entropy_coeffs": self.entropy_coeffs,
            "scaled_operator": scaled_operator_report(self.scaled_operator),
            "polynomial_log_operator": self.polynomial_log_operator,
            "reference_log_operator": self.reference_log_operator,
            "polynomial_entropy_operator": self.polynomial_entropy_operator,
            "reference_entropy_operator": self.reference_entropy_operator,
            "polynomial_entropy": self.polynomial_entropy,
            "reference_entropy": self.reference_entropy,
            "log_operator_relative_error": self.log_operator_relative_error,
            "entropy_operator_relative_error": self.entropy_operator_relative_error,
        }


@dataclass(frozen=True)
class SpectralCountingWorkflowResult:
    """
    Structured output from a spectral interval-counting workflow.
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
    exact_count: int
    polynomial_count: float | complex
    count_error: float
    stochastic_count: float | None = None
    stochastic_count_error: float | None = None
    probe_count: int | None = None

    def as_report(self) -> dict[str, Any]:
        return {
            "mode": "spectral-counting-workflow",
            "implementation_kind": "dense-spectral-polynomial-workflow",
            "truth_contract": algorithm_truth_contract(
                "spectral-counting-workflow",
                target="spectral interval counting through smooth projector traces",
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
            "exact_count": self.exact_count,
            "polynomial_count": self.polynomial_count,
            "count_error": self.count_error,
            "stochastic_count": self.stochastic_count,
            "stochastic_count_error": self.stochastic_count_error,
            "probe_count": self.probe_count,
        }


@dataclass(frozen=True)
class FixedPointAmplificationWorkflowResult:
    """
    Structured output from a monotone fixed-point spectral amplification workflow.
    """

    coeffs: np.ndarray
    score_operator: np.ndarray
    input_state: np.ndarray
    amplified_state: np.ndarray
    reference_state: np.ndarray
    polynomial_operator: np.ndarray
    reference_operator: np.ndarray
    rounds: int
    degree: int
    initial_score: float | complex
    amplified_score: float | complex
    reference_score: float | complex
    state_relative_error: float
    operator_relative_error: float

    def as_report(self) -> dict[str, Any]:
        return {
            "mode": "fixed-point-amplification-workflow",
            "implementation_kind": "dense-spectral-polynomial-workflow",
            "truth_contract": algorithm_truth_contract(
                "fixed-point-amplification-workflow",
                target="monotone fixed-point polynomial amplification of scores",
            ),
            "rounds": self.rounds,
            "degree": self.degree,
            "coeffs": self.coeffs,
            "score_operator": self.score_operator,
            "input_state": self.input_state,
            "amplified_state": self.amplified_state,
            "reference_state": self.reference_state,
            "polynomial_operator": self.polynomial_operator,
            "reference_operator": self.reference_operator,
            "initial_score": self.initial_score,
            "amplified_score": self.amplified_score,
            "reference_score": self.reference_score,
            "state_relative_error": self.state_relative_error,
            "operator_relative_error": self.operator_relative_error,
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


def _validate_adjacency(adjacency: np.ndarray) -> np.ndarray:
    matrix = np.asarray(adjacency, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be a square 2D array.")
    if matrix.shape[0] < 2:
        raise ValueError("adjacency must describe at least two vertices.")
    if not np.allclose(matrix, matrix.conj().T):
        raise ValueError("adjacency must be Hermitian.")
    return np.real_if_close(matrix)


def _validate_marked_vertex(marked_vertex: int, dimension: int) -> int:
    marked = int(marked_vertex)
    if marked < 0 or marked >= dimension:
        raise ValueError("marked_vertex must be a valid vertex index.")
    return marked


def _validate_times(
    *,
    dimension: int,
    times: np.ndarray | None,
    max_time: float | None,
    num_time_points: int,
) -> np.ndarray:
    if times is not None:
        values = np.asarray(times, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("times must be a non-empty one-dimensional array.")
        if np.any(values < 0.0):
            raise ValueError("times must be non-negative.")
        return values

    points = int(num_time_points)
    if points < 2:
        raise ValueError("num_time_points must be at least 2.")
    end = float(max_time) if max_time is not None else float(np.pi * np.sqrt(dimension))
    if end <= 0.0:
        raise ValueError("max_time must be positive.")
    return np.linspace(0.0, end, points)


def _qsvt_check_from_result(result: LinearSystemWorkflowResult) -> str:
    if result.qsvt_error is not None:
        return "failed"
    if result.qsvt_solution is not None:
        return "succeeded"
    return "not_attempted"


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


def _validate_matrix_2d(matrix: np.ndarray, name: str = "matrix") -> np.ndarray:
    A = np.asarray(matrix, dtype=complex if np.iscomplexobj(matrix) else float)
    if A.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if min(A.shape) == 0:
        raise ValueError(f"{name} must have nonzero dimensions.")
    return A


def _svd_filter_matrix(
    matrix: np.ndarray,
    singular_weights: np.ndarray,
) -> np.ndarray:
    U, _, Vh = np.linalg.svd(matrix, full_matrices=False)
    return U @ np.diag(singular_weights) @ Vh


def _relative_matrix_error(reference: np.ndarray, approximate: np.ndarray) -> float:
    return _relative_error(np.ravel(reference), np.ravel(approximate))


def _stable_fermi_dirac(
    energies: np.ndarray,
    *,
    chemical_potential: float,
    beta: float,
) -> np.ndarray:
    z = np.clip(float(beta) * (energies - float(chemical_potential)), -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(z))


def singular_value_filtering_workflow(
    matrix: np.ndarray,
    *,
    degree: int,
    cutoff: float,
    sharpness: float = 20.0,
    input_vector: np.ndarray | None = None,
    num_points: int = 1001,
) -> SingularValueFilteringWorkflowResult:
    """
    Apply a smooth QSVT-style filter to the singular values of a matrix.

    The matrix is normalized by its largest singular value. The polynomial is
    designed on normalized singular values in ``[0, 1]`` and compared with the
    smooth reference filter
    ``0.5 * (1 + tanh(sharpness * (sigma - cutoff)))``.
    """
    A = _validate_matrix_2d(matrix)
    cutoff = float(cutoff)
    sharpness = float(sharpness)
    if not 0.0 < cutoff < 1.0:
        raise ValueError("cutoff must satisfy 0 < cutoff < 1.")
    if sharpness <= 0.0:
        raise ValueError("sharpness must be positive.")

    singular_values = np.linalg.svd(A, compute_uv=False)
    scale = float(np.max(singular_values))
    if scale <= 0.0:
        raise ValueError("matrix must have at least one nonzero singular value.")
    normalized = singular_values / scale

    def target(sigma: np.ndarray) -> np.ndarray:
        return 0.5 * (1.0 + np.tanh(sharpness * (sigma - cutoff)))

    cheb = chebyshev_fit_function(
        target,
        degree=degree,
        domain=(0.0, 1.0),
        num_points=num_points,
    )
    coeffs = chebyshev_to_monomial(cheb, domain=(0.0, 1.0))
    polynomial_weights = np.asarray(eval_polynomial(coeffs, normalized), dtype=float)
    reference_weights = target(normalized)
    polynomial_matrix = _svd_filter_matrix(A, polynomial_weights)
    reference_matrix = _svd_filter_matrix(A, reference_weights)

    vec = None
    polynomial_output = None
    reference_output = None
    output_error = None
    if input_vector is not None:
        vec = _validate_state(input_vector, A.shape[1], name="input_vector")
        polynomial_output = polynomial_matrix @ vec
        reference_output = reference_matrix @ vec
        output_error = _relative_error(reference_output, polynomial_output)

    return SingularValueFilteringWorkflowResult(
        coeffs=coeffs,
        matrix=A,
        singular_values=singular_values,
        normalized_singular_values=normalized,
        polynomial_matrix=polynomial_matrix,
        reference_matrix=reference_matrix,
        cutoff=cutoff,
        sharpness=sharpness,
        scale=scale,
        degree=int(degree),
        operator_relative_error=_relative_matrix_error(
            reference_matrix,
            polynomial_matrix,
        ),
        input_vector=vec,
        polynomial_output=polynomial_output,
        reference_output=reference_output,
        output_relative_error=output_error,
    )


def singular_value_pseudoinverse_workflow(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    degree: int,
    cutoff: float,
    num_points: int = 2001,
) -> SingularValuePseudoinverseWorkflowResult:
    """
    Approximate a truncated SVD pseudoinverse action with an inverse polynomial.

    ``cutoff`` is the normalized singular-value threshold. Singular values
    below the threshold are omitted from the dense reference pseudoinverse.
    """
    A = _validate_matrix_2d(matrix)
    b = _validate_state(rhs, A.shape[0], name="rhs")
    cutoff = float(cutoff)
    if not 0.0 < cutoff < 1.0:
        raise ValueError("cutoff must satisfy 0 < cutoff < 1.")

    U, singular_values, Vh = np.linalg.svd(A, full_matrices=False)
    scale = float(np.max(singular_values))
    if scale <= 0.0:
        raise ValueError("matrix must have at least one nonzero singular value.")
    normalized = singular_values / scale

    coeffs = design_positive_inverse_polynomial(
        gamma=cutoff,
        degree=degree,
        num_points=num_points,
    )
    polynomial_inverse_weights = eval_polynomial(coeffs, normalized) / (cutoff * scale)
    reference_inverse_weights = np.where(
        normalized >= cutoff,
        1.0 / singular_values,
        0.0,
    )
    polynomial_pinv = Vh.conj().T @ np.diag(polynomial_inverse_weights) @ U.conj().T
    reference_pinv = Vh.conj().T @ np.diag(reference_inverse_weights) @ U.conj().T
    polynomial_solution = polynomial_pinv @ b
    reference_solution = reference_pinv @ b

    return SingularValuePseudoinverseWorkflowResult(
        coeffs=coeffs,
        matrix=A,
        rhs=b,
        singular_values=singular_values,
        normalized_singular_values=normalized,
        polynomial_solution=polynomial_solution,
        reference_solution=reference_solution,
        polynomial_pseudoinverse=polynomial_pinv,
        reference_pseudoinverse=reference_pinv,
        cutoff=cutoff,
        scale=scale,
        degree=int(degree),
        residual_norm=float(np.linalg.norm(A @ polynomial_solution - b)),
        reference_residual_norm=float(np.linalg.norm(A @ reference_solution - b)),
        solution_relative_error=_relative_error(
            reference_solution, polynomial_solution
        ),
        operator_relative_error=_relative_matrix_error(reference_pinv, polynomial_pinv),
    )


def fermi_dirac_occupation_workflow(
    matrix: np.ndarray,
    *,
    chemical_potential: float,
    beta: float,
    degree: int,
    state: np.ndarray | None = None,
    num_points: int = 2001,
) -> FermiDiracWorkflowResult:
    """
    Approximate finite-temperature Fermi-Dirac occupations for a Hamiltonian.
    """
    if beta < 0.0:
        raise ValueError("beta must be non-negative.")
    scaled = rescale_hermitian_to_unit_interval(matrix)
    coeffs = chebyshev_to_monomial(
        chebyshev_fit_function(
            lambda x: _stable_fermi_dirac(
                scaled.offset + scaled.scale * x,
                chemical_potential=chemical_potential,
                beta=beta,
            ),
            degree=degree,
            num_points=num_points,
        )
    )
    polynomial_operator = apply_polynomial_to_hermitian(scaled.matrix, coeffs)
    reference_operator = apply_function_to_hermitian(
        np.asarray(matrix),
        lambda x: _stable_fermi_dirac(
            x,
            chemical_potential=chemical_potential,
            beta=beta,
        ),
    )

    state_vec = None
    polynomial_state_occupation = None
    reference_state_occupation = None
    state_error = None
    if state is not None:
        state_vec = _normalize_state(_validate_state(state, scaled.matrix.shape[0]))
        polynomial_state_occupation = np.real_if_close(
            np.vdot(state_vec, polynomial_operator @ state_vec)
        ).item()
        reference_state_occupation = np.real_if_close(
            np.vdot(state_vec, reference_operator @ state_vec)
        ).item()
        state_error = float(
            abs(polynomial_state_occupation - reference_state_occupation)
        )

    return FermiDiracWorkflowResult(
        coeffs=coeffs,
        scaled_operator=scaled,
        polynomial_occupation_operator=polynomial_operator,
        reference_occupation_operator=reference_operator,
        chemical_potential=float(chemical_potential),
        beta=float(beta),
        degree=int(degree),
        particle_number=np.real_if_close(np.trace(polynomial_operator)).item(),
        reference_particle_number=np.real_if_close(np.trace(reference_operator)).item(),
        operator_relative_error=operator_error(reference_operator, polynomial_operator),
        state=state_vec,
        polynomial_state_occupation=polynomial_state_occupation,
        reference_state_occupation=reference_state_occupation,
        state_occupation_error=state_error,
    )


def matrix_log_entropy_workflow(
    matrix: np.ndarray,
    *,
    degree: int,
    epsilon: float = 1e-8,
    num_points: int = 2001,
) -> MatrixLogEntropyWorkflowResult:
    """
    Approximate a regularized matrix logarithm and ``-x log(x)`` entropy term.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    evals, _ = eigh_hermitian(matrix)
    if evals[0] < -1e-10:
        raise ValueError("matrix must be positive semidefinite.")
    scaled = rescale_positive_semidefinite(np.asarray(matrix))

    def physical_from_scaled(x: np.ndarray) -> np.ndarray:
        return scaled.scale * x

    log_coeffs = chebyshev_to_monomial(
        chebyshev_fit_function(
            lambda x: np.log(physical_from_scaled(x) + float(epsilon)),
            degree=degree,
            domain=(0.0, 1.0),
            num_points=num_points,
        ),
        domain=(0.0, 1.0),
    )
    entropy_coeffs = chebyshev_to_monomial(
        chebyshev_fit_function(
            lambda x: (
                -physical_from_scaled(x)
                * np.log(physical_from_scaled(x) + float(epsilon))
            ),
            degree=degree,
            domain=(0.0, 1.0),
            num_points=num_points,
        ),
        domain=(0.0, 1.0),
    )
    polynomial_log = apply_polynomial_to_hermitian(scaled.matrix, log_coeffs)
    reference_log = apply_function_to_hermitian(
        np.asarray(matrix),
        lambda x: np.log(x + float(epsilon)),
    )
    polynomial_entropy_op = apply_polynomial_to_hermitian(
        scaled.matrix,
        entropy_coeffs,
    )
    reference_entropy_op = apply_function_to_hermitian(
        np.asarray(matrix),
        lambda x: -x * np.log(x + float(epsilon)),
    )

    return MatrixLogEntropyWorkflowResult(
        log_coeffs=log_coeffs,
        entropy_coeffs=entropy_coeffs,
        scaled_operator=scaled,
        polynomial_log_operator=polynomial_log,
        reference_log_operator=reference_log,
        polynomial_entropy_operator=polynomial_entropy_op,
        reference_entropy_operator=reference_entropy_op,
        epsilon=float(epsilon),
        degree=int(degree),
        polynomial_entropy=np.real_if_close(np.trace(polynomial_entropy_op)).item(),
        reference_entropy=np.real_if_close(np.trace(reference_entropy_op)).item(),
        log_operator_relative_error=operator_error(reference_log, polynomial_log),
        entropy_operator_relative_error=operator_error(
            reference_entropy_op,
            polynomial_entropy_op,
        ),
    )


def spectral_counting_workflow(
    matrix: np.ndarray,
    *,
    lower: float,
    upper: float,
    degree: int,
    sharpness: float = 12.0,
    num_points: int = 2001,
    probe_count: int | None = None,
    random_seed: int | None = None,
) -> SpectralCountingWorkflowResult:
    """
    Count eigenvalues in an interval using a smooth polynomial projector trace.
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
    coeffs = design_interval_projector_polynomial(
        lower=scaled_lower,
        upper=scaled_upper,
        degree=degree,
        sharpness=sharpness,
        num_points=num_points,
    )
    polynomial_projector = apply_polynomial_to_hermitian(scaled.matrix, coeffs)
    reference_projector = apply_function_to_hermitian(
        np.asarray(matrix),
        lambda x: np.where((lower <= x) & (x <= upper), 1.0, 0.0),
    )
    exact_count = int(np.count_nonzero((evals >= lower) & (evals <= upper)))
    polynomial_count = np.real_if_close(np.trace(polynomial_projector)).item()

    stochastic_count = None
    stochastic_error = None
    if probe_count is not None:
        probes = int(probe_count)
        if probes <= 0:
            raise ValueError("probe_count must be positive when provided.")
        rng = np.random.default_rng(random_seed)
        estimates = []
        dimension = scaled.matrix.shape[0]
        for _ in range(probes):
            signs = rng.choice([-1.0, 1.0], size=dimension)
            estimates.append(np.vdot(signs, polynomial_projector @ signs))
        stochastic_count = float(np.real_if_close(np.mean(estimates)).item())
        stochastic_error = float(abs(stochastic_count - exact_count))

    return SpectralCountingWorkflowResult(
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
        exact_count=exact_count,
        polynomial_count=polynomial_count,
        count_error=float(abs(polynomial_count - exact_count)),
        stochastic_count=stochastic_count,
        stochastic_count_error=stochastic_error,
        probe_count=probe_count,
    )


def fixed_point_amplification_workflow(
    score_operator: np.ndarray,
    state: np.ndarray,
    *,
    rounds: int,
) -> FixedPointAmplificationWorkflowResult:
    """
    Apply the monotone fixed-point polynomial ``1 - (1 - x)^rounds``.

    The score operator must be positive semidefinite with spectrum in
    ``[0, 1]``. This is a robust projector/score amplification primitive for
    finite spectral workflows, not a full Grover iterate implementation.
    """
    if rounds <= 0:
        raise ValueError("rounds must be positive.")
    evals, _ = eigh_hermitian(score_operator)
    if evals[0] < -1e-10 or evals[-1] > 1.0 + 1e-10:
        raise ValueError("score_operator spectrum must lie in [0, 1].")
    A = np.asarray(score_operator)
    psi = _normalize_state(_validate_state(state, A.shape[0]))

    coeffs = np.zeros(rounds + 1, dtype=float)
    for k in range(1, rounds + 1):
        coeffs[k] = ((-1.0) ** (k + 1)) * math.comb(rounds, k)

    polynomial_operator = apply_polynomial_to_hermitian(A, coeffs)
    reference_operator = apply_function_to_hermitian(
        A,
        lambda x: 1.0 - (1.0 - x) ** rounds,
    )
    amplified = _normalize_state(polynomial_operator @ psi)
    reference = _normalize_state(reference_operator @ psi)

    return FixedPointAmplificationWorkflowResult(
        coeffs=coeffs,
        score_operator=A,
        input_state=psi,
        amplified_state=amplified,
        reference_state=reference,
        polynomial_operator=polynomial_operator,
        reference_operator=reference_operator,
        rounds=int(rounds),
        degree=int(rounds),
        initial_score=np.real_if_close(np.vdot(psi, A @ psi)).item(),
        amplified_score=np.real_if_close(np.vdot(amplified, A @ amplified)).item(),
        reference_score=np.real_if_close(np.vdot(reference, A @ reference)).item(),
        state_relative_error=_state_error(reference, amplified),
        operator_relative_error=operator_error(reference_operator, polynomial_operator),
    )


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
    scaled_max = float(evals[-1] / scaled.scale)
    condition_number = float(evals[-1] / evals[0])

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
        scaled_min_eigenvalue=scaled_min,
        scaled_max_eigenvalue=scaled_max,
        condition_number_2=condition_number,
        gamma_condition_proxy=float(1.0 / gamma),
        degree=int(degree),
        diagnostics=diagnostics,
        compatibility=compatibility,
        qsvt_solution=qsvt_solution,
        qsvt_residual_norm=qsvt_residual_norm,
        qsvt_relative_error=qsvt_relative_error,
        qsvt_error=qsvt_error,
    )


def linear_system_resource_proxy(
    *,
    degree: int,
    gamma: float,
    scaled_min_eigenvalue: float,
    condition_number_2: float,
    polynomial_residual_norm: float,
    polynomial_relative_error: float,
    qsvt_check: str,
    attempted_pennylane_synthesis: bool,
) -> dict[str, object]:
    """
    Return machine-readable proxy metadata for a linear-system workflow.
    """
    return {
        "proxy_kind": "linear-system-qsvt-style-resource-proxy",
        "degree": int(degree),
        "gamma": float(gamma),
        "scaled_min_eigenvalue": float(scaled_min_eigenvalue),
        "gamma_condition_proxy": float(1.0 / gamma),
        "condition_number_2": float(condition_number_2),
        "polynomial_residual_norm": float(polynomial_residual_norm),
        "polynomial_relative_error": float(polynomial_relative_error),
        "pennylane_qsvt_check": qsvt_check,
        "attempted_pennylane_synthesis": bool(attempted_pennylane_synthesis),
        "requires_block_encoding": True,
        "requires_state_preparation": True,
        "requires_success_probability_management": True,
        "requires_readout_strategy": True,
        "omitted_layers": [
            "state_preparation_or_data_loading",
            "oracle_or_block_encoding_construction",
            "controlled_reflection_sequence_cost",
            "success_probability_management",
            "amplitude_amplification",
            "measurement_readout_or_tomography",
            "fault_tolerant_synthesis",
            "hardware_compilation",
        ],
        "limitations": (
            "This is a polynomial-degree and conditioning proxy for a finite "
            "dense instance. It is not a quantum runtime or an end-to-end "
            "linear-system algorithm cost."
        ),
    }


def quantum_walk_search_resource_proxy(
    *,
    graph_vertices: int,
    degree: int,
    time_grid_size: int,
    best_probability: float,
    probability_error: float,
) -> dict[str, object]:
    """
    Return machine-readable proxy metadata for quantum-walk search.
    """
    return {
        "proxy_kind": "quantum-walk-search-resource-proxy",
        "graph_vertices": int(graph_vertices),
        "degree": int(degree),
        "qsp_phase_count": int(degree) + 1,
        "signal_call_proxy": int(degree),
        "time_grid_size": int(time_grid_size),
        "best_probability": float(best_probability),
        "probability_error": float(probability_error),
        "requires_graph_oracle": True,
        "requires_marking_oracle": True,
        "requires_state_preparation": True,
        "requires_readout_strategy": True,
        "omitted_layers": [
            "graph_oracle_or_sparse_walk_construction",
            "marked_vertex_oracle_construction",
            "initial_state_preparation",
            "QSP_or_QSVT_phase_synthesis",
            "success_probability_estimation_or_sampling",
            "hardware_noise_and_fault_tolerance",
        ],
        "limitations": (
            "This is a finite dense spectral-polynomial workflow for a search "
            "Hamiltonian. It is not a quantum runtime or a scalable oracle "
            "construction cost."
        ),
    }


def linear_system_comparison_workflow(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    degree: int,
    gamma: float | None = None,
    num_points: int = 2001,
    bounded_num_points: int = 4001,
    attempt_synthesis: bool = True,
    apply_qsvt: bool = True,
    include_conjugate_gradient: bool = True,
    cg_tolerance: float = 1e-10,
    cg_max_iterations: int | None = None,
    include_hhl_execution: bool = False,
    hhl_num_phase_qubits: int = 2,
    hhl_evolution_time: float | None = None,
    hhl_rotation_scale_C: float | None = None,
    hhl_eigenvalue_lower_bound: float | None = None,
) -> LinearSystemComparisonResult:
    """
    Compare dense, iterative, QSVT-style, and optional HHL linear-system paths.

    This workflow is not a timing benchmark. It returns a compact numerical
    table using the dense solve as the reference solution. When
    ``include_hhl_execution`` is true, the HHL row reports finite PennyLane
    QNode execution diagnostics for power-of-two positive-definite systems.
    """
    result = linear_system_workflow(
        matrix,
        rhs,
        degree=degree,
        gamma=gamma,
        num_points=num_points,
        bounded_num_points=bounded_num_points,
        attempt_synthesis=attempt_synthesis,
        apply_qsvt=apply_qsvt,
    )
    A, b, _ = _validate_linear_system_inputs(matrix, rhs)
    reference = result.classical_solution
    dense_residual = A @ reference - b
    rows: list[dict[str, Any]] = [
        {
            "solver": "dense_solve",
            "implementation_kind": "classical-dense-reference",
            "residual_norm": float(np.linalg.norm(dense_residual)),
            "relative_solution_error": 0.0,
            "condition_number_2": result.condition_number_2,
        },
        {
            "solver": "qsvt_style_polynomial_inverse",
            "implementation_kind": "dense-spectral-polynomial-workflow",
            "degree": result.degree,
            "gamma": result.gamma,
            "residual_norm": result.polynomial_residual_norm,
            "relative_solution_error": result.polynomial_relative_error,
            "condition_number_2": result.condition_number_2,
        },
    ]

    cg_solution = None
    if include_conjugate_gradient:
        from .benchmarks import conjugate_gradient_solve

        cg = conjugate_gradient_solve(
            A,
            b,
            tolerance=cg_tolerance,
            max_iterations=cg_max_iterations,
        )
        cg_solution = np.asarray(cg["solution"])
        rows.insert(
            1,
            {
                "solver": "conjugate_gradient",
                "implementation_kind": "classical-iterative-reference",
                "iterations": cg["iterations"],
                "converged": cg["converged"],
                "residual_norm": cg["residual_norm"],
                "relative_residual_norm": cg["relative_residual_norm"],
                "relative_solution_error": _relative_error(reference, cg_solution),
                "condition_number_2": result.condition_number_2,
            },
        )

    if result.qsvt_solution is not None:
        rows.append(
            {
                "solver": "pennylane_qsvt_matrix_check",
                "implementation_kind": "pennylane-small-qsvt-verification",
                "degree": result.degree,
                "gamma": result.gamma,
                "residual_norm": result.qsvt_residual_norm,
                "relative_solution_error": result.qsvt_relative_error,
                "condition_number_2": result.condition_number_2,
            },
        )
    elif result.qsvt_error is not None:
        rows.append(
            {
                "solver": "pennylane_qsvt_matrix_check",
                "implementation_kind": "pennylane-small-qsvt-verification",
                "status": "failed",
                "error": result.qsvt_error,
            },
        )

    if include_hhl_execution:
        rows.append(
            _hhl_comparison_row(
                A,
                b,
                num_phase_qubits=hhl_num_phase_qubits,
                evolution_time=hhl_evolution_time,
                rotation_scale_C=hhl_rotation_scale_C,
                eigenvalue_lower_bound=hhl_eigenvalue_lower_bound,
            )
        )

    return LinearSystemComparisonResult(
        workflow=result,
        rows=tuple(rows),
        dense_solution=reference,
        cg_solution=cg_solution,
        reference_solution=reference,
        notes=(
            "The dense solve is the numerical reference for this finite instance.",
            (
                "The QSVT-style row measures polynomial inverse accuracy, "
                "not quantum runtime."
            ),
        ),
    )


def _hhl_comparison_row(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    num_phase_qubits: int,
    evolution_time: float | None,
    rotation_scale_C: float | None,
    eigenvalue_lower_bound: float | None,
) -> dict[str, Any]:
    from .hhl import execute_hhl_circuit

    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
        lower_bound = (
            float(np.min(eigenvalues))
            if eigenvalue_lower_bound is None
            else float(eigenvalue_lower_bound)
        )
        t = (
            float(np.pi / np.max(eigenvalues))
            if evolution_time is None
            else float(evolution_time)
        )
        C = lower_bound if rotation_scale_C is None else float(rotation_scale_C)
        hhl = execute_hhl_circuit(
            matrix,
            rhs,
            num_phase_qubits=num_phase_qubits,
            evolution_time=t,
            rotation_scale_C=C,
            eigenvalue_lower_bound=lower_bound,
            normalize_state=True,
        )
    except Exception as exc:
        return {
            "solver": "hhl_circuit_execution",
            "implementation_kind": "pennylane-qnode-hhl-execution",
            "status": "failed",
            "error": str(exc),
        }

    resources = hhl.resource_summary
    return {
        "solver": "hhl_circuit_execution",
        "implementation_kind": hhl.execution_kind,
        "status": "ok",
        "success_probability": hhl.success_probability,
        "fidelity": hhl.fidelity,
        "state_error": hhl.state_error,
        "phase_qubits": hhl.num_phase_qubits,
        "system_qubits": len(hhl.system_wires),
        "total_wires": len(hhl.wire_order),
        "num_gates": resources.get("num_gates"),
        "circuit_depth": resources.get("depth"),
        "uses_dense_time_evolution": hhl.uses_dense_time_evolution,
        "is_executable_hhl_circuit": True,
        "is_scalable_hhl_implementation": not hhl.uses_dense_time_evolution,
        "evolution_time": hhl.evolution_time,
        "rotation_scale_C": hhl.rotation_scale_C,
        "eigenvalue_lower_bound": hhl.eigenvalue_lower_bound,
    }


def linear_system_comparison_summary_table(
    report: LinearSystemComparisonResult | dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Convert a linear-system comparison report into compact table rows.
    """
    payload = (
        report.as_report()
        if isinstance(report, LinearSystemComparisonResult)
        else report
    )
    workflow = payload.get("linear_system_workflow", {})
    resource_proxy = payload.get("resource_proxy", {})
    rhs = workflow.get("rhs", [])
    rows = []
    for row in payload.get("rows", []):
        rows.append(
            {
                "solver": row.get("solver"),
                "implementation_kind": row.get("implementation_kind"),
                "matrix_dimension": len(rhs),
                "degree": row.get("degree", resource_proxy.get("degree")),
                "gamma": row.get("gamma", resource_proxy.get("gamma")),
                "condition_number_2": row.get(
                    "condition_number_2",
                    resource_proxy.get("condition_number_2"),
                ),
                "iterations": row.get("iterations"),
                "converged": row.get("converged"),
                "residual_norm": row.get("residual_norm"),
                "relative_residual_norm": row.get("relative_residual_norm"),
                "relative_solution_error": row.get("relative_solution_error"),
                "success_probability": row.get("success_probability"),
                "fidelity": row.get("fidelity"),
                "state_error": row.get("state_error"),
                "phase_qubits": row.get("phase_qubits"),
                "total_wires": row.get("total_wires"),
                "num_gates": row.get("num_gates"),
                "circuit_depth": row.get("circuit_depth"),
                "uses_dense_time_evolution": row.get("uses_dense_time_evolution"),
                "status": row.get("status", "ok"),
            }
        )
    return rows


def write_linear_system_comparison_csv(
    report: LinearSystemComparisonResult | dict[str, Any],
    path: str | Path,
) -> Path:
    """
    Write compact linear-system comparison rows to a CSV file.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = linear_system_comparison_summary_table(report)
    fieldnames = [
        "solver",
        "implementation_kind",
        "matrix_dimension",
        "degree",
        "gamma",
        "condition_number_2",
        "iterations",
        "converged",
        "residual_norm",
        "relative_residual_norm",
        "relative_solution_error",
        "success_probability",
        "fidelity",
        "state_error",
        "phase_qubits",
        "total_wires",
        "num_gates",
        "circuit_depth",
        "uses_dense_time_evolution",
        "status",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


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


def quantum_walk_search_workflow(
    adjacency: np.ndarray,
    marked_vertex: int,
    *,
    gamma: float | None = None,
    oracle_strength: float = 1.0,
    initial_state: np.ndarray | None = None,
    times: np.ndarray | None = None,
    max_time: float | None = None,
    num_time_points: int = 121,
    degree: int,
    num_points: int = 1001,
) -> QuantumWalkSearchWorkflowResult:
    """
    Run a simulator-scale continuous-time quantum-walk search workflow.

    The search Hamiltonian is ``H = -gamma A - oracle_strength |m><m|``.
    Exact dense spectral evolution is sampled over ``times`` to find the best
    marked-vertex probability. At the best sampled time, real and imaginary
    polynomial phase approximations validate the QSVT-style matrix-function
    view of the same search propagator.
    """
    graph = _validate_adjacency(adjacency)
    dimension = graph.shape[0]
    marked = _validate_marked_vertex(marked_vertex, dimension)

    if gamma is None:
        gamma = 1.0 / dimension
    gamma = float(gamma)
    if gamma <= 0.0:
        raise ValueError("gamma must be positive.")
    oracle_strength = float(oracle_strength)
    if oracle_strength <= 0.0:
        raise ValueError("oracle_strength must be positive.")

    if initial_state is None:
        state = np.ones(dimension, dtype=float) / np.sqrt(dimension)
    else:
        state = _normalize_state(_validate_state(initial_state, dimension))

    time_grid = _validate_times(
        dimension=dimension,
        times=times,
        max_time=max_time,
        num_time_points=num_time_points,
    )

    oracle = np.zeros((dimension, dimension), dtype=graph.dtype)
    oracle[marked, marked] = oracle_strength
    search_hamiltonian = -gamma * graph - oracle

    marked_probabilities = []
    for time in time_grid:
        unitary = apply_function_to_hermitian(
            search_hamiltonian,
            _real_time_phase_function(float(time)),
        )
        evolved = unitary @ state
        marked_probabilities.append(float(abs(evolved[marked]) ** 2))
    probability_arr = np.asarray(marked_probabilities, dtype=float)
    best_index = int(np.argmax(probability_arr))
    best_time = float(time_grid[best_index])
    best_probability = float(probability_arr[best_index])

    scaled = rescale_hermitian_to_unit_interval(search_hamiltonian)
    polynomials = design_real_time_evolution_polynomials(
        time=best_time,
        scale=scaled.scale,
        degree=degree,
        num_points=num_points,
    )
    cos_op = apply_polynomial_to_hermitian(scaled.matrix, polynomials.cos_coeffs)
    sin_op = apply_polynomial_to_hermitian(scaled.matrix, polynomials.sin_coeffs)
    phase = np.exp(-1j * scaled.offset * best_time)
    polynomial_unitary = phase * (cos_op - 1j * sin_op)
    reference_unitary = apply_function_to_hermitian(
        search_hamiltonian,
        lambda x: np.exp(-1j * best_time * x),
    )
    exact_best_state = reference_unitary @ state
    polynomial_best_state = polynomial_unitary @ state
    polynomial_probability = float(abs(polynomial_best_state[marked]) ** 2)

    return QuantumWalkSearchWorkflowResult(
        adjacency=graph,
        search_hamiltonian=search_hamiltonian,
        marked_vertex=marked,
        gamma=gamma,
        oracle_strength=oracle_strength,
        initial_state=state,
        times=time_grid,
        marked_probabilities=probability_arr,
        best_time=best_time,
        best_probability=best_probability,
        exact_best_state=exact_best_state,
        polynomial_best_state=polynomial_best_state,
        polynomial_marked_probability=polynomial_probability,
        probability_error=float(abs(best_probability - polynomial_probability)),
        state_relative_error=_state_error(exact_best_state, polynomial_best_state),
        cos_coeffs=polynomials.cos_coeffs,
        sin_coeffs=polynomials.sin_coeffs,
        scaled_operator=scaled,
        polynomial_unitary=polynomial_unitary,
        reference_unitary=reference_unitary,
        degree=int(degree),
        scaled_time=polynomials.scaled_time,
        operator_relative_error=operator_error(reference_unitary, polynomial_unitary),
        norm_drift=float(abs(np.linalg.norm(polynomial_best_state) - 1.0)),
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
        if state_vec is not None and reference_state is not None:
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
    "FermiDiracWorkflowResult",
    "FixedPointAmplificationWorkflowResult",
    "GroundStateFilteringWorkflowResult",
    "HamiltonianSimulationWorkflowResult",
    "LinearSystemComparisonResult",
    "LinearSystemWorkflowResult",
    "MatrixLogEntropyWorkflowResult",
    "QuantumWalkSearchWorkflowResult",
    "ResolventWorkflowResult",
    "SingularValueFilteringWorkflowResult",
    "SingularValuePseudoinverseWorkflowResult",
    "SpectralCountingWorkflowResult",
    "SpectralDensityWorkflowResult",
    "SpectralThresholdingWorkflowResult",
    "ThermalGibbsWorkflowResult",
    "block_encoded_qsvt_workflow",
    "fermi_dirac_occupation_workflow",
    "fixed_point_amplification_workflow",
    "ground_state_filtering_workflow",
    "hamiltonian_simulation_workflow",
    "linear_system_comparison_summary_table",
    "linear_system_comparison_workflow",
    "linear_system_workflow",
    "matrix_log_entropy_workflow",
    "quantum_walk_search_resource_proxy",
    "quantum_walk_search_workflow",
    "resolvent_workflow",
    "singular_value_filtering_workflow",
    "singular_value_pseudoinverse_workflow",
    "spectral_counting_workflow",
    "spectral_density_workflow",
    "spectral_thresholding_workflow",
    "thermal_gibbs_workflow",
    "write_linear_system_comparison_csv",
]
