"""End-to-end finite flagship workflows for spectral filtering and Poisson PDEs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pennylane as qml

from .acceptance import (
    evaluate_poisson_acceptance,
    evaluate_spectral_filter_acceptance,
)
from .benchmarks import conjugate_gradient_solve
from .block_encoding import (
    BlockEncodingSpec,
    matrix_block_encoding_spec,
    pennylane_operator_block_encoding_spec,
)
from .degree import DegreeSearchResult, search_polynomial_degree
from .design import (
    design_interval_projector_polynomial,
    design_positive_inverse_polynomial,
)
from .diagnostics import expectation_value, operator_error
from .execution import BlockEncodingQSVTExecutionResult, execute_qsvt_from_spec
from .pde import dirichlet_laplacian_1d
from .resources import EncodingAwareResourceEstimate, estimate_encoding_aware_resources
from .spectral import apply_polynomial_to_hermitian
from .synthesis import (
    PhaseSynthesisResult,
    available_phase_solver_adapters,
    classify_polynomial_realizability,
    synthesize_phases_cached,
    synthesize_phases_with_adapter,
)


@dataclass(frozen=True)
class SpectralFilterQSVTResult:
    """Pauli-Hamiltonian band filter from polynomial design through execution."""

    operator: qml.operation.Operator
    matrix: np.ndarray
    input_state: np.ndarray
    lower: float
    upper: float
    block_encoding_spec: BlockEncodingSpec
    degree_search: DegreeSearchResult
    coeffs: np.ndarray
    synthesis: PhaseSynthesisResult
    resource_estimate: EncodingAwareResourceEstimate
    polynomial_operator: np.ndarray
    reference_projector: np.ndarray
    polynomial_state: np.ndarray
    reference_state: np.ndarray
    polynomial_success_probability: float
    reference_success_probability: float
    polynomial_operator_error: float
    polynomial_state_error: float
    execution_requested: bool
    phase_reconstruction_tolerance: float
    execution: BlockEncodingQSVTExecutionResult | None
    observable_values: dict[str, dict[str, float | complex | None]]
    error_budget: dict[str, float | None]

    def as_report(self) -> dict[str, object]:
        """Return the complete spectral-filter workflow report."""
        return {
            "mode": "spectral-filter-qsvt-flagship",
            "implementation_kind": "pauli-lcu-block-encoded-qsvt-workflow",
            "physical_interval": (self.lower, self.upper),
            "matrix": self.matrix,
            "input_state": self.input_state,
            "block_encoding_spec": self.block_encoding_spec.as_report(),
            "degree_search": self.degree_search.as_report(),
            "coeffs": self.coeffs,
            "synthesis": self.synthesis.as_report(),
            "resources": self.resource_estimate.as_report(),
            "polynomial_operator": self.polynomial_operator,
            "reference_projector": self.reference_projector,
            "polynomial_state": self.polynomial_state,
            "reference_state": self.reference_state,
            "polynomial_success_probability": self.polynomial_success_probability,
            "reference_success_probability": self.reference_success_probability,
            "polynomial_operator_error": self.polynomial_operator_error,
            "polynomial_state_error": self.polynomial_state_error,
            "execution_requested": self.execution_requested,
            "phase_reconstruction_tolerance": (self.phase_reconstruction_tolerance),
            "execution": None if self.execution is None else self.execution.as_report(),
            "observable_values": self.observable_values,
            "error_budget": self.error_budget,
            "acceptance": evaluate_spectral_filter_acceptance(self),
            "truth_contract": {
                "implemented_components": [
                    "pauli_hamiltonian_matrix_reference",
                    "prepselprep_or_qubitization_access_model",
                    "tolerance_driven_bounded_polynomial_design",
                    "phase_synthesis_and_reconstruction_validation",
                    "finite_qnode_qsvt_execution_when_requested",
                    "postselected_state_and_observable_validation",
                    "encoding_aware_resource_estimate",
                ],
                "full_state_outputs_are_simulator_validation_data": True,
                "omitted_components": [
                    "application_state_preparation_cost",
                    "amplitude_amplification",
                    "hardware_native_compilation",
                    "large_scale_observable_estimation",
                ],
            },
        }


@dataclass(frozen=True)
class PoissonQSVTResult:
    """One-dimensional Poisson solve with classical and block-encoded QSVT paths."""

    grid: np.ndarray
    matrix: np.ndarray
    rhs: np.ndarray
    direct_solution: np.ndarray
    conjugate_gradient: dict[str, Any]
    access_model: str
    block_encoding_spec: BlockEncodingSpec
    gamma: float
    condition_number_2: float
    degree_search: DegreeSearchResult
    coeffs: np.ndarray
    polynomial_solution: np.ndarray
    polynomial_residual_norm: float
    polynomial_relative_error: float
    synthesis: PhaseSynthesisResult
    resource_estimate: EncodingAwareResourceEstimate
    execution_requested: bool
    phase_reconstruction_tolerance: float
    execution: BlockEncodingQSVTExecutionResult | None
    circuit_solution: np.ndarray | None
    circuit_relative_error: float | None
    physical_observables: dict[str, dict[str, float | complex | None]]
    continuum_relative_error: float | None
    error_budget: dict[str, float | None]

    def as_report(self) -> dict[str, object]:
        """Return the complete Poisson workflow report."""
        return {
            "mode": "poisson-qsvt-flagship",
            "implementation_kind": "finite-difference-block-encoded-qsvt-workflow",
            "grid": self.grid,
            "matrix": self.matrix,
            "rhs": self.rhs,
            "direct_solution": self.direct_solution,
            "conjugate_gradient": self.conjugate_gradient,
            "access_model": self.access_model,
            "block_encoding_spec": self.block_encoding_spec.as_report(),
            "gamma": self.gamma,
            "condition_number_2": self.condition_number_2,
            "degree_search": self.degree_search.as_report(),
            "coeffs": self.coeffs,
            "polynomial_solution": self.polynomial_solution,
            "polynomial_residual_norm": self.polynomial_residual_norm,
            "polynomial_relative_error": self.polynomial_relative_error,
            "synthesis": self.synthesis.as_report(),
            "resources": self.resource_estimate.as_report(),
            "execution_requested": self.execution_requested,
            "phase_reconstruction_tolerance": (self.phase_reconstruction_tolerance),
            "execution": None if self.execution is None else self.execution.as_report(),
            "circuit_solution": self.circuit_solution,
            "circuit_relative_error": self.circuit_relative_error,
            "physical_observables": self.physical_observables,
            "continuum_relative_error": self.continuum_relative_error,
            "error_budget": self.error_budget,
            "acceptance": evaluate_poisson_acceptance(self),
            "truth_contract": {
                "implemented_components": [
                    "dirichlet_finite_difference_discretization",
                    "dense_direct_and_conjugate_gradient_baselines",
                    "access_model_specific_normalization",
                    "tolerance_driven_inverse_polynomial_design",
                    "phase_synthesis_and_encoding_aware_resources",
                    "finite_qnode_qsvt_execution_when_requested",
                ],
                "full_solution_vector_is_simulator_validation_data": True,
                "quantum_output_is_a_postselected_solution_state": True,
                "omitted_components": [
                    "scalable_rhs_state_preparation",
                    "amplitude_amplification",
                    "solution_norm_estimation",
                    "full_vector_tomography",
                    "provider_native_compilation",
                ],
            },
        }


def spectral_filter_qsvt_workflow(
    operator: qml.operation.Operator,
    state: np.ndarray | Sequence[complex],
    *,
    lower: float,
    upper: float,
    tolerance: float = 2e-2,
    min_degree: int = 2,
    max_degree: int = 24,
    degree_step: int = 2,
    sharpness: float = 8.0,
    block_encoding: Literal["prepselprep", "qubitization"] = "prepselprep",
    encoding_wires: Sequence[Any] | None = None,
    angle_solvers: tuple[str, ...] = ("root-finding", "iterative"),
    phase_reconstruction_tolerance: float = 1e-6,
    execute: bool = True,
    device_name: str = "default.qubit",
    shots: int | None = None,
    observables: Mapping[str, np.ndarray] | None = None,
    num_points: int = 2001,
    gate_set: tuple[str, ...] | None = None,
) -> SpectralFilterQSVTResult:
    """Filter a physical energy band using a Pauli-LCU QSVT circuit.

    The physical Hamiltonian is encoded as ``H / alpha``. Consequently the
    polynomial interval is ``[lower / alpha, upper / alpha]``; the report keeps
    this normalization and its impact on degree and success probability visible.
    """
    if not isinstance(operator, qml.operation.Operator):
        raise TypeError("operator must be a PennyLane Operator.")
    system_wires = list(operator.wires)
    matrix = np.asarray(qml.matrix(operator, wire_order=system_wires), dtype=complex)
    if not np.allclose(matrix, matrix.conj().T, atol=1e-10, rtol=1e-10):
        raise ValueError("operator must be Hermitian.")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    lower, upper = float(lower), float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("lower and upper must be finite with lower < upper.")
    psi = _normalized_vector(state, matrix.shape[0], "state")
    if encoding_wires is None:
        term_count = _operator_term_count(operator)
        encoding_wires = _fresh_integer_wires(
            max(1, int((term_count - 1).bit_length())),
            occupied=system_wires,
        )
    spec = pennylane_operator_block_encoding_spec(
        operator,
        encoding_wires=list(encoding_wires),
        block_encoding=block_encoding,
    )
    scaled_lower = lower / spec.alpha
    scaled_upper = upper / spec.alpha
    if not -1.0 < scaled_lower < scaled_upper < 1.0:
        raise ValueError("encoded filter interval must lie strictly inside [-1, 1].")
    encoded_matrix = matrix / spec.alpha
    inside = (eigenvalues >= lower) & (eigenvalues <= upper)
    reference_projector = (eigenvectors * inside.astype(float)) @ eigenvectors.conj().T

    def builder(degree: int) -> np.ndarray:
        coeffs = design_interval_projector_polynomial(
            lower=scaled_lower,
            upper=scaled_upper,
            degree=degree,
            sharpness=sharpness,
            num_points=num_points,
        )
        classification = classify_polynomial_realizability(coeffs)
        if not classification.single_sequence_realizable:
            raise ValueError(
                "candidate is not realizable by a single QSVT phase sequence"
            )
        return coeffs

    def evaluator(
        coeffs: np.ndarray,
        degree: int,
    ) -> tuple[float, dict[str, object]]:
        del degree
        polynomial = apply_polynomial_to_hermitian(encoded_matrix, coeffs)
        return operator_error(reference_projector, polynomial), {
            "state_error": _filtered_state_error(reference_projector, polynomial, psi),
        }

    search = search_polynomial_degree(
        builder,
        evaluator,
        tolerance=tolerance,
        degrees=range(min_degree, max_degree + 1, degree_step),
        metric="hard_projector_operator_relative_error",
    )
    if search.chosen_coeffs is None:
        raise ValueError("no spectral-filter degree candidate was usable.")
    coeffs = search.chosen_coeffs
    synthesis = _synthesize_with_fallback(
        coeffs,
        angle_solvers,
        reconstruction_tolerance=phase_reconstruction_tolerance,
    )
    resources = estimate_encoding_aware_resources(spec, coeffs, gate_set=gate_set)
    polynomial_operator = apply_polynomial_to_hermitian(encoded_matrix, coeffs)
    polynomial_raw = polynomial_operator @ psi
    reference_raw = reference_projector @ psi
    polynomial_probability = float(np.linalg.norm(polynomial_raw) ** 2)
    reference_probability = float(np.linalg.norm(reference_raw) ** 2)
    if polynomial_probability == 0.0 or reference_probability == 0.0:
        raise ValueError(
            "input state must overlap both polynomial and reference bands."
        )
    polynomial_state = polynomial_raw / np.sqrt(polynomial_probability)
    reference_state = reference_raw / np.sqrt(reference_probability)

    execution_result = None
    if execute:
        _require_execution_quality(synthesis, phase_reconstruction_tolerance)
        execution_result = execute_qsvt_from_spec(
            spec,
            coeffs,
            psi,
            projectors=_projectors(spec, synthesis),
            device_name=device_name,
            shots=shots,
            normalize_state=True,
        )
    observable_map = {"hamiltonian": np.real_if_close(matrix), **(observables or {})}
    observable_values = _filter_observable_values(
        observable_map,
        reference_state,
        polynomial_state,
        execution_result,
    )
    state_error = _phase_invariant_state_error(reference_state, polynomial_state)
    error_budget = _component_error_budget(
        approximation=operator_error(reference_projector, polynomial_operator),
        synthesis=synthesis.reconstruction_max_error,
        execution=(
            None
            if execution_result is None
            else execution_result.logical_output_relative_error
        ),
        sampling=(
            None
            if execution_result is None
            else execution_result.maximum_probability_standard_error
        ),
    )
    return SpectralFilterQSVTResult(
        operator=operator,
        matrix=np.real_if_close(matrix),
        input_state=psi,
        lower=lower,
        upper=upper,
        block_encoding_spec=spec,
        degree_search=search,
        coeffs=coeffs,
        synthesis=synthesis,
        resource_estimate=resources,
        polynomial_operator=np.real_if_close(polynomial_operator),
        reference_projector=np.real_if_close(reference_projector),
        polynomial_state=np.real_if_close(polynomial_state),
        reference_state=np.real_if_close(reference_state),
        polynomial_success_probability=polynomial_probability,
        reference_success_probability=reference_probability,
        polynomial_operator_error=operator_error(
            reference_projector, polynomial_operator
        ),
        polynomial_state_error=state_error,
        execution_requested=bool(execute),
        phase_reconstruction_tolerance=float(phase_reconstruction_tolerance),
        execution=execution_result,
        observable_values=observable_values,
        error_budget=error_budget,
    )


def poisson_qsvt_workflow(
    n_points: int = 4,
    *,
    length: float = 1.0,
    source: (
        np.ndarray | Sequence[float] | Callable[[np.ndarray], np.ndarray] | None
    ) = None,
    analytic_solution: Callable[[np.ndarray], np.ndarray] | None = None,
    tolerance: float = 2e-1,
    min_degree: int = 3,
    max_degree: int = 31,
    degree_step: int = 2,
    access_model: Literal["dense", "fable", "prepselprep", "qubitization"] = (
        "prepselprep"
    ),
    angle_solvers: tuple[str, ...] = ("root-finding", "iterative"),
    phase_reconstruction_tolerance: float = 1e-6,
    execute: bool = True,
    device_name: str = "default.qubit",
    shots: int | None = None,
    num_points: int = 2001,
    gate_set: tuple[str, ...] | None = None,
) -> PoissonQSVTResult:
    """Solve ``-u'' = f`` with Dirichlet boundaries and an explicit QSVT model."""
    grid, matrix = dirichlet_laplacian_1d(n_points, length=length)
    if source is None:
        rhs = np.sin(np.pi * grid / length)
        if analytic_solution is None:
            analytic_solution = lambda x: (  # noqa: E731
                (length / np.pi) ** 2 * np.sin(np.pi * x / length)
            )
    elif callable(source):
        rhs = np.asarray(source(grid), dtype=float)
    else:
        rhs = np.asarray(source, dtype=float)
    if rhs.shape != (n_points,) or not np.all(np.isfinite(rhs)):
        raise ValueError(
            "source must produce one finite value per interior grid point."
        )
    if np.linalg.norm(rhs) == 0.0:
        raise ValueError("source must be nonzero.")

    direct = np.linalg.solve(matrix, rhs)
    cg = conjugate_gradient_solve(matrix, rhs, tolerance=min(tolerance, 1e-10))
    spec = _poisson_block_encoding_spec(matrix, access_model)
    encoded_matrix = matrix / spec.alpha
    eigenvalues = np.linalg.eigvalsh(matrix)
    minimum_eigenvalue = float(eigenvalues[0])
    gamma = minimum_eigenvalue / spec.alpha

    def builder(degree: int) -> np.ndarray:
        coeffs = design_positive_inverse_polynomial(
            gamma=gamma,
            degree=degree,
            num_points=num_points,
        )
        classification = classify_polynomial_realizability(coeffs)
        if not classification.single_sequence_realizable:
            raise ValueError(
                "candidate is not realizable by a single QSVT phase sequence"
            )
        return coeffs

    def evaluator(
        coeffs: np.ndarray,
        degree: int,
    ) -> tuple[float, dict[str, object]]:
        del degree
        polynomial = apply_polynomial_to_hermitian(encoded_matrix, coeffs)
        solution = (polynomial @ rhs) / minimum_eigenvalue
        relative = _relative_error(direct, solution)
        return relative, {
            "residual_norm": float(np.linalg.norm(matrix @ solution - rhs)),
            "gamma": gamma,
        }

    search = search_polynomial_degree(
        builder,
        evaluator,
        tolerance=tolerance,
        degrees=range(min_degree, max_degree + 1, degree_step),
        metric="solution_relative_error",
    )
    if search.chosen_coeffs is None:
        raise ValueError("no Poisson inverse-polynomial degree candidate was usable.")
    coeffs = search.chosen_coeffs
    polynomial_operator = apply_polynomial_to_hermitian(encoded_matrix, coeffs)
    polynomial_solution = (polynomial_operator @ rhs) / minimum_eigenvalue
    polynomial_error = _relative_error(direct, polynomial_solution)
    synthesis = _synthesize_with_fallback(
        coeffs,
        angle_solvers,
        reconstruction_tolerance=phase_reconstruction_tolerance,
    )
    resources = estimate_encoding_aware_resources(spec, coeffs, gate_set=gate_set)

    execution_result = None
    circuit_solution = None
    circuit_error = None
    if execute:
        _require_execution_quality(synthesis, phase_reconstruction_tolerance)
        rhs_norm = float(np.linalg.norm(rhs))
        execution_result = execute_qsvt_from_spec(
            spec,
            coeffs,
            rhs / rhs_norm,
            projectors=_projectors(spec, synthesis),
            device_name=device_name,
            shots=shots,
            normalize_state=True,
        )
        if execution_result.logical_output is not None:
            circuit_solution = (
                np.real(np.asarray(execution_result.logical_output))
                * rhs_norm
                / minimum_eigenvalue
            )
            circuit_error = _relative_error(direct, circuit_solution)

    dx = length / (n_points + 1)
    physical_observables = {
        "solution_integral": {
            "direct": np.real_if_close(dx * np.sum(direct)).item(),
            "polynomial": np.real_if_close(dx * np.sum(polynomial_solution)).item(),
            "circuit_validation": (
                None
                if circuit_solution is None
                else np.real_if_close(dx * np.sum(circuit_solution)).item()
            ),
        },
        "source_solution_energy": {
            "direct": np.real_if_close(dx * np.vdot(rhs, direct)).item(),
            "polynomial": np.real_if_close(
                dx * np.vdot(rhs, polynomial_solution)
            ).item(),
            "circuit_validation": (
                None
                if circuit_solution is None
                else np.real_if_close(dx * np.vdot(rhs, circuit_solution)).item()
            ),
        },
    }
    continuum_error = None
    if analytic_solution is not None:
        continuum = np.asarray(analytic_solution(grid), dtype=float)
        if continuum.shape != direct.shape or not np.all(np.isfinite(continuum)):
            raise ValueError(
                "analytic_solution must produce one finite value per grid point."
            )
        continuum_error = _relative_error(continuum, direct)
    error_budget = _component_error_budget(
        approximation=polynomial_error,
        synthesis=synthesis.reconstruction_max_error,
        execution=(
            None
            if execution_result is None
            else execution_result.logical_output_relative_error
        ),
        sampling=(
            None
            if execution_result is None
            else execution_result.maximum_probability_standard_error
        ),
        discretization=continuum_error,
    )
    return PoissonQSVTResult(
        grid=grid,
        matrix=matrix,
        rhs=rhs,
        direct_solution=direct,
        conjugate_gradient=cg,
        access_model=access_model,
        block_encoding_spec=spec,
        gamma=gamma,
        condition_number_2=float(np.linalg.cond(matrix)),
        degree_search=search,
        coeffs=coeffs,
        polynomial_solution=np.real_if_close(polynomial_solution),
        polynomial_residual_norm=float(
            np.linalg.norm(matrix @ polynomial_solution - rhs)
        ),
        polynomial_relative_error=polynomial_error,
        synthesis=synthesis,
        resource_estimate=resources,
        execution_requested=bool(execute),
        phase_reconstruction_tolerance=float(phase_reconstruction_tolerance),
        execution=execution_result,
        circuit_solution=(
            None if circuit_solution is None else np.real_if_close(circuit_solution)
        ),
        circuit_relative_error=circuit_error,
        physical_observables=physical_observables,
        continuum_relative_error=continuum_error,
        error_budget=error_budget,
    )


def _poisson_block_encoding_spec(
    matrix: np.ndarray,
    access_model: str,
) -> BlockEncodingSpec:
    if access_model == "dense":
        return matrix_block_encoding_spec(matrix, block_encoding="embedding")
    if access_model == "fable":
        alpha = max(
            float(np.linalg.norm(matrix, ord=2)),
            float(max(matrix.shape) * np.max(np.abs(matrix))),
        )
        return matrix_block_encoding_spec(
            matrix,
            alpha=alpha,
            block_encoding="fable",
        )
    if access_model not in {"prepselprep", "qubitization"}:
        raise ValueError(
            "access_model must be dense, fable, prepselprep, or qubitization."
        )
    dimension = matrix.shape[0]
    system_qubits = int(np.log2(dimension))
    if 2**system_qubits != dimension:
        raise ValueError("Pauli-LCU Poisson access requires a power-of-two n_points.")
    provisional = qml.pauli_decompose(matrix, wire_order=list(range(system_qubits)))
    term_count = _operator_term_count(provisional)
    control_count = max(1, int((term_count - 1).bit_length()))
    controls = list(range(control_count))
    system_wires = list(range(control_count, control_count + system_qubits))
    operator = qml.pauli_decompose(matrix, wire_order=system_wires)
    return pennylane_operator_block_encoding_spec(
        operator,
        encoding_wires=controls,
        block_encoding=cast(Any, access_model),
    )


def _operator_term_count(operator: qml.operation.Operator) -> int:
    try:
        coefficients, _ = operator.terms()
    except (AttributeError, NotImplementedError):
        return 1
    return max(1, len(coefficients))


def _fresh_integer_wires(count: int, *, occupied: Sequence[Any]) -> list[int]:
    used = set(occupied)
    wires: list[int] = []
    candidate = 0
    while len(wires) < count:
        if candidate not in used:
            wires.append(candidate)
        candidate += 1
    return wires


def _normalized_vector(
    values: np.ndarray | Sequence[complex],
    dimension: int,
    name: str,
) -> np.ndarray:
    vector = np.asarray(values, dtype=complex)
    if vector.shape != (dimension,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector of length {dimension}.")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(f"{name} must be nonzero.")
    return vector / norm


def _filtered_state_error(
    reference_projector: np.ndarray,
    polynomial_operator: np.ndarray,
    state: np.ndarray,
) -> float:
    reference = reference_projector @ state
    approximate = polynomial_operator @ state
    if np.linalg.norm(reference) == 0.0 or np.linalg.norm(approximate) == 0.0:
        return float("inf")
    return _phase_invariant_state_error(
        reference / np.linalg.norm(reference),
        approximate / np.linalg.norm(approximate),
    )


def _phase_invariant_state_error(
    reference: np.ndarray, approximate: np.ndarray
) -> float:
    phase = np.vdot(reference, approximate)
    aligned = approximate
    if abs(phase) > 0.0:
        aligned = approximate * np.exp(-1j * np.angle(phase))
    return _relative_error(reference, aligned)


def _relative_error(reference: np.ndarray, approximate: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    numerator = float(np.linalg.norm(np.asarray(approximate) - np.asarray(reference)))
    return numerator if denominator == 0.0 else numerator / denominator


def _synthesize_with_fallback(
    coeffs: np.ndarray,
    solvers: tuple[str, ...],
    *,
    reconstruction_tolerance: float,
) -> PhaseSynthesisResult:
    if not solvers:
        raise ValueError("angle_solvers must contain at least one solver.")
    if reconstruction_tolerance <= 0.0:
        raise ValueError("phase_reconstruction_tolerance must be positive.")
    attempts: list[PhaseSynthesisResult] = []
    adapters = set(available_phase_solver_adapters())
    for solver in solvers:
        result = (
            synthesize_phases_with_adapter(
                coeffs,
                adapter=solver,
                reconstruction_num_points=129,
            )
            if solver in adapters
            else synthesize_phases_cached(
                coeffs,
                angle_solver=solver,
                reconstruction_num_points=129,
            )
        )
        attempts.append(result)
        if (
            result.succeeded
            and result.reconstruction_max_error is not None
            and result.reconstruction_max_error <= reconstruction_tolerance
        ):
            return result
    succeeded = [attempt for attempt in attempts if attempt.succeeded]
    if succeeded:
        return min(
            succeeded,
            key=lambda attempt: (
                np.inf
                if attempt.reconstruction_max_error is None
                else attempt.reconstruction_max_error
            ),
        )
    return attempts[-1]


def _projectors(
    spec: BlockEncodingSpec,
    synthesis: PhaseSynthesisResult,
) -> list[qml.operation.Operator]:
    if synthesis.angles is None:
        raise ValueError("synthesis did not produce projector phases.")
    if spec.kind == "pennylane-operator":
        source = cast(Any, spec.source)
        wires = list(spec.encoding_wires) + list(source.wires)
        dimensions = [spec.logical_shape[0]] * len(synthesis.angles)
    elif spec.block_encoding == "fable":
        wires = list(spec.encoding_wires)
        dimensions = [spec.logical_shape[0]] * len(synthesis.angles)
    else:
        wires = list(spec.encoding_wires)
        rows, columns = spec.logical_shape
        dimensions = [
            rows if index % 2 == 0 else columns
            for index in range(len(synthesis.angles))
        ]
    return [
        qml.PCPhase(float(angle), dim=int(dimension), wires=wires)
        for angle, dimension in zip(synthesis.angles, dimensions, strict=True)
    ]


def _require_execution_quality(
    synthesis: PhaseSynthesisResult,
    reconstruction_tolerance: float,
) -> None:
    if not synthesis.succeeded:
        raise ValueError(synthesis.error or "phase synthesis failed")
    if (
        synthesis.reconstruction_max_error is None
        or synthesis.reconstruction_max_error > reconstruction_tolerance
    ):
        raise ValueError("phase synthesis did not meet phase_reconstruction_tolerance.")


def _filter_observable_values(
    observables: Mapping[str, np.ndarray],
    reference_state: np.ndarray,
    polynomial_state: np.ndarray,
    execution: BlockEncodingQSVTExecutionResult | None,
) -> dict[str, dict[str, float | complex | None]]:
    values: dict[str, dict[str, float | complex | None]] = {}
    circuit_state = None
    circuit_probabilities = None
    if execution is not None and execution.logical_output is not None:
        raw = np.real(np.asarray(execution.logical_output, dtype=complex))
        if np.linalg.norm(raw) > 0.0:
            circuit_state = raw / np.linalg.norm(raw)
    elif (
        execution is not None
        and execution.logical_probabilities is not None
        and execution.logical_success_probability is not None
        and execution.logical_success_probability > 0.0
    ):
        circuit_probabilities = (
            execution.logical_probabilities / execution.logical_success_probability
        )
    for name, observable in observables.items():
        matrix = np.asarray(observable)
        if matrix.shape != (reference_state.size, reference_state.size):
            raise ValueError(f"observable {name!r} has the wrong shape.")
        if not np.allclose(matrix, matrix.conj().T, atol=1e-10, rtol=1e-10):
            raise ValueError(f"observable {name!r} must be Hermitian.")
        circuit_value = None
        if circuit_state is not None:
            circuit_value = expectation_value(matrix, circuit_state)
        elif circuit_probabilities is not None and np.allclose(
            matrix,
            np.diag(np.diag(matrix)),
            atol=1e-12,
        ):
            circuit_value = np.real_if_close(
                np.dot(np.diag(matrix), circuit_probabilities)
            ).item()
        values[name] = {
            "reference": expectation_value(matrix, reference_state),
            "polynomial": expectation_value(matrix, polynomial_state),
            "circuit_or_shot_estimate": circuit_value,
        }
    return values


def _component_error_budget(
    *,
    approximation: float | None,
    synthesis: float | None,
    execution: float | None,
    sampling: float | None,
    discretization: float | None = None,
) -> dict[str, float | None]:
    budget = {
        "discretization_error": discretization,
        "polynomial_approximation_error": approximation,
        "phase_reconstruction_error": synthesis,
        "circuit_vs_polynomial_error": execution,
        "maximum_sampling_standard_error": sampling,
    }
    values = [float(value) for value in budget.values() if value is not None]
    budget["additive_error_proxy"] = sum(values) if values else None
    return budget


__all__ = [
    "PoissonQSVTResult",
    "SpectralFilterQSVTResult",
    "poisson_qsvt_workflow",
    "spectral_filter_qsvt_workflow",
]
