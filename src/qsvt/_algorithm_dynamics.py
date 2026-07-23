"""Hamiltonian dynamics and quantum-walk workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._algorithm_reports import (
    algorithm_truth_contract,
    algorithm_workflow_schema_fields,
    scaled_operator_report,
)
from ._algorithm_shared import (
    _normalize_state,
    _state_error,
    _validate_state,
)
from .acceptance import evaluate_hamiltonian_simulation_acceptance
from .diagnostics import operator_error
from .matrix_functions import design_real_time_evolution_polynomials
from .rescaling import ScaledOperator, rescale_hermitian_to_unit_interval
from .spectral import apply_function_to_hermitian, apply_polynomial_to_hermitian


def _real_time_phase_function(time: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return the spectral phase function exp(-i time x).
    """

    def phase(eigenvalues: np.ndarray) -> np.ndarray:
        return np.exp(-1j * time * eigenvalues)

    return phase


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
    acceptance_tolerance: float

    def as_report(self) -> dict[str, Any]:
        """
        Return a report-style dictionary for JSON conversion or persistence.
        """
        return {
            **algorithm_workflow_schema_fields(),
            "mode": "hamiltonian-simulation-workflow",
            "implementation_kind": "dense-spectral-polynomial-workflow",
            "truth_contract": algorithm_truth_contract(
                "hamiltonian-simulation-workflow",
                target="real-time Hamiltonian matrix exponential action",
                polynomials={
                    "cosine": self.cos_coeffs,
                    "sine": self.sin_coeffs,
                },
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
            "acceptance_tolerance": self.acceptance_tolerance,
            "acceptance": evaluate_hamiltonian_simulation_acceptance(self),
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
            **algorithm_workflow_schema_fields(),
            "mode": "quantum-walk-search-workflow",
            "implementation_kind": "dense-spectral-polynomial-workflow",
            "truth_contract": algorithm_truth_contract(
                "quantum-walk-search-workflow",
                target=(
                    "continuous-time quantum-walk search and polynomial phase "
                    "approximation"
                ),
                polynomials={
                    "cosine": self.cos_coeffs,
                    "sine": self.sin_coeffs,
                },
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


def hamiltonian_simulation_workflow(
    matrix: np.ndarray,
    state: np.ndarray,
    *,
    time: float,
    degree: int,
    num_points: int = 1001,
    acceptance_tolerance: float = 1e-6,
) -> HamiltonianSimulationWorkflowResult:
    """
    Approximate real-time evolution ``exp(-i H t)|psi>`` with polynomial pairs.
    """
    acceptance_tolerance = float(acceptance_tolerance)
    if not np.isfinite(acceptance_tolerance) or acceptance_tolerance <= 0.0:
        raise ValueError("acceptance_tolerance must be positive and finite.")
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
        acceptance_tolerance=acceptance_tolerance,
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
