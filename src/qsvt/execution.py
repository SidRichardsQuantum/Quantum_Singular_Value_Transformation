"""
Circuit-level QSVT execution helpers.

This module keeps a clear boundary between dense reference calculations and
quantum-circuit execution. The helpers here run a PennyLane QNode with state
preparation, a queued QSVT operator, and a measurement. They do not materialize
the QSVT unitary with ``qml.matrix``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence, Sized
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pennylane as qml

from .block_encoding import BlockEncodingSpec, build_block_encoding_operator
from .operators import (
    _as_numeric_operator,
    _default_wire_order_from_operator,
    _validate_wire_inputs,
)
from .polynomials import polynomial_degree
from .spectral import apply_polynomial_to_hermitian
from .synthesis import (
    PhaseSynthesisResult,
    certify_polynomial_boundedness,
    classify_polynomial_realizability,
    parity_components,
    synthesize_phases,
)


@dataclass(frozen=True)
class QSVTCircuitExecutionResult:
    """
    Result from executing a QSVT circuit through a PennyLane QNode.
    """

    operator: np.ndarray
    poly: np.ndarray
    input_state: np.ndarray
    encoding_wires: tuple[int, ...]
    wire_order: tuple[int, ...]
    block_encoding: str
    angle_solver: str
    device_name: str
    shots: int | None
    final_state: np.ndarray | None
    probabilities: np.ndarray
    logical_output: np.ndarray | None
    logical_probabilities: np.ndarray
    logical_success_probability: float
    classical_reference_output: np.ndarray
    execution_kind: str
    resource_summary: dict[str, object]

    def as_report(self) -> dict[str, object]:
        """
        Return a report preserving the circuit-execution claim boundary.
        """
        return {
            "mode": "qsvt-circuit-execution-report",
            "implementation_kind": self.execution_kind,
            "is_end_to_end_quantum_algorithm": False,
            "implemented_components": [
                "qnode_state_preparation",
                "queued_pennylane_qsvt_operator",
                "qnode_measurement",
                "circuit_resource_summary",
            ],
            "reference_components": [
                "classical_polynomial_output_for_validation",
            ],
            "omitted_quantum_components": [
                "problem_specific_scalable_block_encoding",
                "problem_specific_state_preparation_circuit",
                "postselection_or_amplitude_amplification_strategy",
                "application_specific_readout_or_tomography",
                "hardware_compilation",
            ],
            "operator": self.operator,
            "poly": self.poly,
            "input_state": self.input_state,
            "encoding_wires": self.encoding_wires,
            "wire_order": self.wire_order,
            "block_encoding": self.block_encoding,
            "angle_solver": self.angle_solver,
            "device_name": self.device_name,
            "shots": self.shots,
            "final_state": self.final_state,
            "probabilities": self.probabilities,
            "logical_output": self.logical_output,
            "logical_probabilities": self.logical_probabilities,
            "logical_success_probability": self.logical_success_probability,
            "classical_reference_output": self.classical_reference_output,
            "resource_summary": self.resource_summary,
        }


@dataclass(frozen=True)
class BlockEncodingQSVTExecutionResult:
    """Result from executing QSVT from a :class:`BlockEncodingSpec`."""

    spec: BlockEncodingSpec
    poly: np.ndarray
    input_state: np.ndarray
    wire_order: tuple[Any, ...]
    projector_source: str
    angle_solver: str
    device_name: str
    shots: int | None
    succeeded: bool
    final_state: np.ndarray | None
    probabilities: np.ndarray | None
    logical_output: np.ndarray | None
    logical_probabilities: np.ndarray | None
    logical_success_probability: float | None
    classical_reference_output: np.ndarray | None
    logical_output_absolute_error: float | None
    logical_output_relative_error: float | None
    complex_leakage_norm: float | None
    logical_subspace_leakage_probability: float | None
    probability_normalization_error: float | None
    statevector_normalization_error: float | None
    logical_success_standard_error: float | None
    maximum_probability_standard_error: float | None
    resource_summary: dict[str, object]
    error_type: str | None = None
    error: str | None = None

    def as_report(self) -> dict[str, object]:
        """Return execution data with access-model and claim-boundary metadata."""
        return {
            "schema_name": "block-encoding-qsvt-execution",
            "schema_version": "1.0",
            "mode": "block-encoding-qsvt-execution-report",
            "implementation_kind": "pennylane-lower-level-qsvt-execution",
            "is_end_to_end_quantum_algorithm": False,
            "succeeded": self.succeeded,
            "block_encoding_spec": self.spec.as_report(),
            "poly": self.poly,
            "input_state": self.input_state,
            "wire_order": self.wire_order,
            "projector_source": self.projector_source,
            "projector_convention": (
                "PennyLane QSVT projector-phase convention; rectangular "
                "embedding/custom paths alternate output and input signal dimensions."
            ),
            "angle_solver": self.angle_solver,
            "device_name": self.device_name,
            "shots": self.shots,
            "final_state": self.final_state,
            "probabilities": self.probabilities,
            "logical_output": self.logical_output,
            "logical_probabilities": self.logical_probabilities,
            "logical_success_probability": self.logical_success_probability,
            "classical_reference_output": self.classical_reference_output,
            "logical_output_absolute_error": self.logical_output_absolute_error,
            "logical_output_relative_error": self.logical_output_relative_error,
            "complex_leakage_norm": self.complex_leakage_norm,
            "logical_subspace_leakage_probability": (
                self.logical_subspace_leakage_probability
            ),
            "probability_normalization_error": self.probability_normalization_error,
            "statevector_normalization_error": self.statevector_normalization_error,
            "logical_success_standard_error": self.logical_success_standard_error,
            "maximum_probability_standard_error": (
                self.maximum_probability_standard_error
            ),
            "resource_summary": self.resource_summary,
            "error_type": self.error_type,
            "error": self.error,
            "truth_contract": {
                "implemented_components": [
                    "caller_selected_block_encoding_access_model",
                    "qnode_state_preparation",
                    "lower_level_pennylane_qsvt_operator",
                    "statevector_or_probability_measurement",
                    "encoding_specific_resource_summary",
                ],
                "reference_components": (
                    []
                    if self.classical_reference_output is None
                    else ["finite_dense_polynomial_reference"]
                ),
                "omitted_quantum_components": [
                    "application_specific_scalable_oracle_construction",
                    "application_state_preparation_cost",
                    "postselection_or_amplitude_amplification_strategy",
                    "application_specific_readout_or_tomography",
                    "fault_tolerant_synthesis",
                    "hardware_compilation",
                ],
            },
        }


@dataclass(frozen=True)
class CoherentQSVTComponent:
    """One real definite-parity polynomial in a coherent QSVT combination."""

    name: str
    coeffs: np.ndarray
    coefficient: complex = 1.0 + 0.0j


@dataclass(frozen=True)
class CoherentQSVTExecutionResult:
    """Result of coherently combining real QSVT polynomial components."""

    spec: BlockEncodingSpec
    components: tuple[CoherentQSVTComponent, ...]
    normalized_components: tuple[CoherentQSVTComponent, ...]
    component_weights: tuple[tuple[str, float], ...]
    component_syntheses: tuple[tuple[str, PhaseSynthesisResult], ...]
    input_state: np.ndarray
    wire_order: tuple[Any, ...]
    selection_wires: tuple[Any, ...]
    angle_solver: str
    device_name: str
    shots: int | None
    succeeded: bool
    lcu_normalization: float
    selection_state_amplitudes: np.ndarray
    final_state: np.ndarray | None
    probabilities: np.ndarray | None
    postselected_logical_output: np.ndarray | None
    logical_output: np.ndarray | None
    selection_success_probability: float | None
    logical_success_probability: float | None
    selection_success_standard_error: float | None
    logical_success_standard_error: float | None
    maximum_probability_standard_error: float | None
    classical_reference_output: np.ndarray | None
    logical_output_absolute_error: float | None
    logical_output_relative_error: float | None
    probability_normalization_error: float | None
    statevector_normalization_error: float | None
    resource_summary: dict[str, object]
    error_type: str | None = None
    error: str | None = None

    def as_report(self) -> dict[str, object]:
        """Return the versioned coherent-combination execution contract."""
        normalized = {
            component.name: component for component in self.normalized_components
        }
        weights = dict(self.component_weights)
        syntheses = dict(self.component_syntheses)
        return {
            "schema_name": "coherent-qsvt-execution",
            "schema_version": "1.0",
            "mode": "coherent-qsvt-execution-report",
            "implementation_kind": "pennylane-coherent-real-part-lcu-qsvt-execution",
            "is_end_to_end_quantum_algorithm": False,
            "succeeded": self.succeeded,
            "block_encoding_spec": self.spec.as_report(),
            "components": {
                component.name: {
                    "coeffs": component.coeffs,
                    "coefficient": component.coefficient,
                    "extrema_weight": weights.get(component.name),
                    "normalized_coeffs": (
                        None
                        if component.name not in normalized
                        else normalized[component.name].coeffs
                    ),
                    "synthesis": (
                        None
                        if component.name not in syntheses
                        else syntheses[component.name].as_report()
                    ),
                }
                for component in self.components
            },
            "input_state": self.input_state,
            "wire_order": self.wire_order,
            "selection_wires": self.selection_wires,
            "angle_solver": self.angle_solver,
            "device_name": self.device_name,
            "shots": self.shots,
            "lcu_normalization": self.lcu_normalization,
            "selection_state_amplitudes": self.selection_state_amplitudes,
            "final_state": self.final_state,
            "probabilities": self.probabilities,
            "postselected_logical_output": self.postselected_logical_output,
            "logical_output": self.logical_output,
            "selection_success_probability": self.selection_success_probability,
            "logical_success_probability": self.logical_success_probability,
            "selection_success_standard_error": (self.selection_success_standard_error),
            "logical_success_standard_error": self.logical_success_standard_error,
            "maximum_probability_standard_error": (
                self.maximum_probability_standard_error
            ),
            "classical_reference_output": self.classical_reference_output,
            "logical_output_absolute_error": self.logical_output_absolute_error,
            "logical_output_relative_error": self.logical_output_relative_error,
            "probability_normalization_error": self.probability_normalization_error,
            "statevector_normalization_error": self.statevector_normalization_error,
            "resource_summary": self.resource_summary,
            "error_type": self.error_type,
            "error": self.error,
            "truth_contract": {
                "implemented_components": [
                    "definite_parity_component_phase_synthesis",
                    "real_part_extraction_with_forward_and_adjoint_sequences",
                    "lcu_selector_state_preparation",
                    "controlled_selection_between_qsvt_sequences",
                    "selector_uncomputation",
                    "finite_statevector_or_probability_execution",
                    "measured_postselection_probability",
                    "component_and_combination_resource_ledger",
                ],
                "combination_identity": (
                    "sum_j coefficient_j * weight_j * " "(U_j + U_j_adjoint) / 2"
                ),
                "lcu_circuit_implemented": True,
                "postselection_probability_is_proxy": False,
                "recovered_output_rescaled_by_lcu_normalization": True,
                "omitted_components": [
                    "application_specific_scalable_oracle_construction",
                    "application_state_preparation_cost",
                    "amplitude_amplification",
                    "application_specific_readout_or_tomography",
                    "fault_tolerant_synthesis",
                    "hardware_compilation",
                ],
            },
        }


def execute_qsvt_circuit(
    operator: np.ndarray,
    poly: Iterable[float],
    state: Iterable[float | complex],
    *,
    encoding_wires: Iterable[int] | None = None,
    wire_order: Iterable[int] | None = None,
    block_encoding: str = "embedding",
    angle_solver: str = "root-finding",
    device_name: str = "default.qubit",
    shots: int | None = None,
    normalize_state: bool = False,
) -> QSVTCircuitExecutionResult:
    """
    Execute a finite QSVT circuit with explicit state preparation.

    The logical input state occupies the first ``n`` amplitudes of the full
    QNode state, where ``n`` is the matrix dimension. With ``shots=None`` this
    returns the final statevector and exact probabilities from the simulator.
    With finite ``shots`` it returns measured probabilities only.

    This is still simulator-scale for dense matrix inputs, but it is a true
    circuit execution path: QSVT is queued inside a QNode and measured instead
    of being applied by eigendecomposition or explicit unitary multiplication.
    """
    A = _validate_execution_operator(operator)
    coeffs = _validate_coefficients(poly)
    logical_state = _validate_logical_state(
        state,
        dimension=A.shape[0],
        normalize=normalize_state,
    )

    if encoding_wires is None:
        encoding_wires = _default_wire_order_from_operator(A)
    enc, order = _validate_wire_inputs(encoding_wires, wire_order)
    if not set(enc).issubset(set(order)):
        raise ValueError("wire_order must contain all encoding_wires.")

    full_dimension = 2 ** len(order)
    if full_dimension < A.shape[0]:
        raise ValueError("wire_order does not provide enough amplitudes.")

    prepared = np.zeros(full_dimension, dtype=complex)
    prepared[: A.shape[0]] = logical_state

    if shots is not None:
        shots = int(shots)
        if shots <= 0:
            raise ValueError("shots must be positive when supplied.")

    probabilities = _execute_probability_qnode(
        A,
        coeffs,
        prepared,
        enc,
        order,
        block_encoding=block_encoding,
        angle_solver=angle_solver,
        device_name=device_name,
        shots=shots,
    )

    final_state = None
    logical_output = None
    if shots is None:
        final_state = _execute_state_qnode(
            A,
            coeffs,
            prepared,
            enc,
            order,
            block_encoding=block_encoding,
            angle_solver=angle_solver,
            device_name=device_name,
        )
        logical_output = final_state[: A.shape[0]]

    logical_probabilities = probabilities[: A.shape[0]]
    reference = apply_polynomial_to_hermitian(A, coeffs) @ logical_state
    execution_kind = (
        "pennylane-qnode-statevector-qsvt-execution"
        if shots is None
        else "pennylane-qnode-shot-qsvt-execution"
    )

    return QSVTCircuitExecutionResult(
        operator=A,
        poly=coeffs,
        input_state=logical_state,
        encoding_wires=tuple(enc),
        wire_order=tuple(order),
        block_encoding=block_encoding,
        angle_solver=angle_solver,
        device_name=device_name,
        shots=shots,
        final_state=final_state,
        probabilities=probabilities,
        logical_output=logical_output,
        logical_probabilities=logical_probabilities,
        logical_success_probability=float(np.sum(logical_probabilities)),
        classical_reference_output=np.real_if_close(reference),
        execution_kind=execution_kind,
        resource_summary=_qsvt_circuit_resource_summary(
            A,
            coeffs,
            prepared,
            enc,
            order,
            block_encoding=block_encoding,
            angle_solver=angle_solver,
            device_name=device_name,
            shots=shots,
        ),
    )


def execute_qsvt_from_spec(
    spec: BlockEncodingSpec,
    poly: Iterable[float],
    state: Iterable[float | complex],
    *,
    wire_order: Iterable[Any] | None = None,
    projectors: Sequence[qml.operation.Operator] | None = None,
    angle_solver: str = "root-finding",
    device_name: str = "default.qubit",
    shots: int | None = None,
    normalize_state: bool = False,
    raise_on_failure: bool = False,
) -> BlockEncodingQSVTExecutionResult:
    """
    Execute QSVT from a block-encoding access-model specification.

    The logical input has length ``spec.logical_shape[1]`` and is embedded in
    the first amplitudes of ``wire_order``. The first
    ``spec.logical_shape[0]`` output amplitudes define the reported logical
    output subspace. This ordering matches the package's matrix specifications
    and the default control-zero subspace for PennyLane operator encodings.

    When ``projectors`` is omitted, QSVT phases are synthesized with
    :func:`pennylane.poly_to_angles` and converted to ``PCPhase`` operations.
    Explicit projectors allow callers to control the signal-subspace convention
    for custom encodings.

    Backend or construction failures are returned as structured result data
    unless ``raise_on_failure=True``.
    """
    if not isinstance(spec, BlockEncodingSpec):
        raise TypeError("spec must be a BlockEncodingSpec.")
    coeffs = _validate_coefficients(poly)
    rows, columns = spec.logical_shape
    logical_state = _validate_logical_state(
        state,
        dimension=columns,
        normalize=normalize_state,
    )
    order = _resolve_spec_wire_order(spec, wire_order)
    full_dimension = 2 ** len(order)
    if full_dimension < max(rows, columns):
        raise ValueError("wire_order does not provide enough amplitudes.")
    if shots is not None:
        shots = int(shots)
        if shots <= 0:
            raise ValueError("shots must be positive when supplied.")

    prepared = np.zeros(full_dimension, dtype=complex)
    prepared[:columns] = logical_state
    explicit_projectors = None if projectors is None else tuple(projectors)
    projector_source = (
        "pennylane-poly-to-angles"
        if explicit_projectors is None
        else "caller-supplied-projectors"
    )
    phase_count = (
        int(coeffs.size) if explicit_projectors is None else len(explicit_projectors)
    )

    final_state: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    logical_output: np.ndarray | None = None
    logical_probabilities: np.ndarray | None = None
    logical_success_probability: float | None = None
    reference = _spec_classical_reference_output(spec, coeffs, logical_state)
    absolute_error: float | None = None
    relative_error: float | None = None
    complex_leakage_norm: float | None = None
    logical_subspace_leakage_probability: float | None = None
    probability_normalization_error: float | None = None
    statevector_normalization_error: float | None = None
    logical_success_standard_error: float | None = None
    maximum_probability_standard_error: float | None = None
    resource_summary = _base_spec_resource_summary(
        spec,
        coeffs,
        order,
        projector_source=projector_source,
        phase_count=phase_count,
        shots=shots,
    )
    error_type: str | None = None
    error: str | None = None

    try:
        angles = (
            None
            if explicit_projectors is not None
            else np.asarray(
                qml.poly_to_angles(
                    coeffs,
                    "QSVT",
                    angle_solver=angle_solver,
                ),
                dtype=float,
            )
        )
        operation_factory = _spec_qsvt_operation_factory(
            spec,
            angles=angles,
            projectors=explicit_projectors,
        )
        probabilities = _execute_spec_probability_qnode(
            operation_factory,
            prepared,
            order,
            device_name=device_name,
            shots=shots,
        )
        if shots is None:
            final_state = _execute_spec_state_qnode(
                operation_factory,
                prepared,
                order,
                device_name=device_name,
            )
            logical_output = final_state[:rows]
        logical_probabilities = probabilities[:rows]
        logical_success_probability = float(np.sum(logical_probabilities))
        logical_subspace_leakage_probability = max(
            0.0,
            1.0 - logical_success_probability,
        )
        probability_normalization_error = abs(float(np.sum(probabilities)) - 1.0)
        if shots is not None:
            logical_success_standard_error = float(
                np.sqrt(
                    logical_success_probability
                    * max(0.0, 1.0 - logical_success_probability)
                    / shots
                )
            )
            maximum_probability_standard_error = float(
                np.max(np.sqrt(probabilities * (1.0 - probabilities) / shots))
            )
        if final_state is not None:
            statevector_normalization_error = abs(
                float(np.linalg.norm(final_state)) - 1.0
            )
            complex_leakage_norm = float(
                np.linalg.norm(np.imag(cast(np.ndarray, logical_output)))
            )
        resource_summary = _spec_qsvt_resource_summary(
            spec,
            coeffs,
            operation_factory,
            prepared,
            order,
            projector_source=projector_source,
            phase_count=phase_count,
            device_name=device_name,
            shots=shots,
        )
        if logical_output is not None and reference is not None:
            absolute_error, relative_error = _real_output_errors(
                logical_output,
                reference,
            )
    except Exception as exc:
        error_type = type(exc).__name__
        error = str(exc)
        if raise_on_failure:
            raise

    return BlockEncodingQSVTExecutionResult(
        spec=spec,
        poly=coeffs,
        input_state=logical_state,
        wire_order=tuple(order),
        projector_source=projector_source,
        angle_solver=str(angle_solver),
        device_name=device_name,
        shots=shots,
        succeeded=error is None,
        final_state=final_state,
        probabilities=probabilities,
        logical_output=logical_output,
        logical_probabilities=logical_probabilities,
        logical_success_probability=logical_success_probability,
        classical_reference_output=reference,
        logical_output_absolute_error=absolute_error,
        logical_output_relative_error=relative_error,
        complex_leakage_norm=complex_leakage_norm,
        logical_subspace_leakage_probability=logical_subspace_leakage_probability,
        probability_normalization_error=probability_normalization_error,
        statevector_normalization_error=statevector_normalization_error,
        logical_success_standard_error=logical_success_standard_error,
        maximum_probability_standard_error=maximum_probability_standard_error,
        resource_summary=resource_summary,
        error_type=error_type,
        error=error,
    )


def execute_qsvt_component_lcu_from_spec(
    spec: BlockEncodingSpec,
    components: Sequence[CoherentQSVTComponent],
    state: Iterable[float | complex],
    *,
    wire_order: Iterable[Any] | None = None,
    selection_wires: Iterable[Any] | None = None,
    angle_solver: str = "root-finding",
    device_name: str = "default.qubit",
    shots: int | None = None,
    normalize_state: bool = False,
    reconstruction_num_points: int = 257,
    phase_reconstruction_tolerance: float = 1e-6,
    raise_on_failure: bool = False,
) -> CoherentQSVTExecutionResult:
    """Execute a coherent LCU of real, definite-parity QSVT components.

    Each component polynomial is normalized by its certified extrema norm and
    synthesized independently. The circuit selects both its QSVT sequence and
    its adjoint, implementing the Hermitian part ``(U + U†) / 2`` that contains
    the requested real polynomial. A selector LCU then combines those Hermitian
    parts with the supplied complex component coefficients.
    """
    if not isinstance(spec, BlockEncodingSpec):
        raise TypeError("spec must be a BlockEncodingSpec.")
    if reconstruction_num_points < 2:
        raise ValueError("reconstruction_num_points must be at least 2.")
    phase_tolerance = float(phase_reconstruction_tolerance)
    if not np.isfinite(phase_tolerance) or phase_tolerance < 0.0:
        raise ValueError(
            "phase_reconstruction_tolerance must be finite and non-negative."
        )

    rows, columns = spec.logical_shape
    logical_state = _validate_logical_state(
        state,
        dimension=columns,
        normalize=normalize_state,
    )
    order = _resolve_spec_wire_order(spec, wire_order)
    data_dimension = 2 ** len(order)
    if data_dimension < max(rows, columns):
        raise ValueError("wire_order does not provide enough amplitudes.")
    if shots is not None:
        shots = int(shots)
        if shots <= 0:
            raise ValueError("shots must be positive when supplied.")

    prepared = np.zeros(data_dimension, dtype=complex)
    prepared[:columns] = logical_state
    raw_components = tuple(components)
    if not raw_components:
        raise ValueError("components must contain at least one nonzero polynomial.")

    normalized_components: list[CoherentQSVTComponent] = []
    component_weights: list[tuple[str, float]] = []
    component_syntheses: list[tuple[str, PhaseSynthesisResult]] = []
    operation_factories: list[tuple[str, object, int, int, complex]] = []
    seen_names: set[str] = set()
    error_type: str | None = None
    error: str | None = None

    try:
        if rows != columns:
            raise ValueError(
                "coherent component-LCU execution currently requires a square "
                "logical transform."
            )
        if spec.kind in {"dense-matrix", "sparse-matrix"} and not bool(
            spec.metadata.get("hermitian", False)
        ):
            raise ValueError(
                "coherent component-LCU execution currently requires a Hermitian "
                "matrix specification."
            )
        for component in raw_components:
            name = str(component.name).strip()
            if not name:
                raise ValueError("component names must be non-empty.")
            if name in seen_names:
                raise ValueError(f"duplicate coherent component name: {name!r}.")
            seen_names.add(name)
            coeffs = _validate_coefficients(component.coeffs)
            coefficient = complex(component.coefficient)
            if not np.isfinite(coefficient.real) or not np.isfinite(coefficient.imag):
                raise ValueError(f"component {name!r} coefficient must be finite.")
            classification = classify_polynomial_realizability(coeffs)
            if not classification.single_sequence_realizable:
                raise ValueError(
                    f"component {name!r} must have definite parity and be "
                    "single-sequence realizable."
                )
            extrema_weight = float(certify_polynomial_boundedness(coeffs).max_abs_value)
            if coefficient == 0.0 or extrema_weight <= 1e-15:
                continue
            # Leave a small synthesis margin at |P| = 1. The compensating LCU
            # weight preserves the requested polynomial exactly while avoiding
            # root-finding failures at a numerically saturated boundary.
            weight = extrema_weight / (1.0 - 1e-8)
            normalized_coeffs = coeffs / weight
            synthesis = _synthesize_component_with_fallback(
                normalized_coeffs,
                angle_solver=angle_solver,
                reconstruction_num_points=reconstruction_num_points,
                phase_reconstruction_tolerance=phase_tolerance,
            )
            if not synthesis.succeeded or synthesis.angles is None:
                raise ValueError(
                    f"phase synthesis failed for component {name!r}: "
                    f"{synthesis.error or synthesis.error_type}"
                )
            if (
                synthesis.reconstruction_max_error is None
                or synthesis.reconstruction_max_error > phase_tolerance
            ):
                raise ValueError(
                    f"phase reconstruction for component {name!r} exceeds "
                    f"the tolerance {phase_tolerance}."
                )
            normalized_component = CoherentQSVTComponent(
                name=name,
                coeffs=normalized_coeffs,
                coefficient=coefficient,
            )
            normalized_components.append(normalized_component)
            component_weights.append((name, weight))
            component_syntheses.append((name, synthesis))
            operation_factories.append(
                (
                    name,
                    _spec_qsvt_operation_factory(
                        spec,
                        angles=np.asarray(synthesis.angles, dtype=float),
                        projectors=None,
                    ),
                    polynomial_degree(normalized_coeffs),
                    int(synthesis.angles.size),
                    coefficient * weight,
                )
            )
        if not operation_factories:
            raise ValueError("components must define a nonzero coherent transform.")
    except Exception as exc:
        error_type = type(exc).__name__
        error = str(exc)
        if raise_on_failure:
            raise

    if error is not None:
        return _failed_coherent_execution_result(
            spec,
            raw_components,
            tuple(normalized_components),
            tuple(component_weights),
            tuple(component_syntheses),
            logical_state,
            order,
            angle_solver=angle_solver,
            device_name=device_name,
            shots=shots,
            error_type=error_type,
            error=error,
        )

    term_coefficients: list[complex] = []
    term_factories: list[tuple[object, bool]] = []
    component_ledger: list[dict[str, object]] = []
    for name, factory, degree, phase_count, weighted_coefficient in operation_factories:
        half_coefficient = weighted_coefficient / 2.0
        term_coefficients.extend((half_coefficient, half_coefficient))
        term_factories.extend(((factory, False), (factory, True)))
        component_ledger.append(
            {
                "name": name,
                "polynomial_degree": degree,
                "phase_count_per_sequence": phase_count,
                "selected_unitary_branches": 2,
                "forward_signal_operator_calls": degree,
                "adjoint_signal_operator_calls": degree,
                "total_signal_operator_calls": 2 * degree,
            }
        )
    lcu_normalization = float(sum(abs(value) for value in term_coefficients))
    selector_count = max(1, (len(term_coefficients) - 1).bit_length())
    selectors = _resolve_selection_wires(order, selector_count, selection_wires)
    selector_dimension = 2**selector_count
    selection_state = np.zeros(selector_dimension, dtype=complex)
    branch_phases = np.zeros(selector_dimension, dtype=float)
    for index, coefficient in enumerate(term_coefficients):
        selection_state[index] = np.sqrt(abs(coefficient) / lcu_normalization)
        branch_phases[index] = float(np.angle(coefficient))

    final_state: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    postselected_output: np.ndarray | None = None
    logical_output: np.ndarray | None = None
    selection_success_probability: float | None = None
    logical_success_probability: float | None = None
    selection_success_standard_error: float | None = None
    logical_success_standard_error: float | None = None
    maximum_probability_standard_error: float | None = None
    reference = _coherent_classical_reference(
        spec,
        raw_components,
        logical_state,
    )
    absolute_error: float | None = None
    relative_error: float | None = None
    probability_normalization_error: float | None = None
    statevector_normalization_error: float | None = None
    resource_summary = _base_coherent_resource_summary(
        spec,
        order,
        selectors,
        component_ledger,
        lcu_normalization=lcu_normalization,
        shots=shots,
    )

    try:
        probabilities = _execute_coherent_probability_qnode(
            term_factories,
            selection_state,
            branch_phases,
            prepared,
            selectors,
            order,
            device_name=device_name,
            shots=shots,
        )
        selection_success_probability = float(np.sum(probabilities[:data_dimension]))
        logical_success_probability = float(np.sum(probabilities[:rows]))
        probability_normalization_error = abs(float(np.sum(probabilities)) - 1.0)
        if shots is not None:
            selection_success_standard_error = float(
                np.sqrt(
                    selection_success_probability
                    * max(0.0, 1.0 - selection_success_probability)
                    / shots
                )
            )
            logical_success_standard_error = float(
                np.sqrt(
                    logical_success_probability
                    * max(0.0, 1.0 - logical_success_probability)
                    / shots
                )
            )
            maximum_probability_standard_error = float(
                np.max(np.sqrt(probabilities * (1.0 - probabilities) / shots))
            )
        if shots is None:
            final_state = _execute_coherent_state_qnode(
                term_factories,
                selection_state,
                branch_phases,
                prepared,
                selectors,
                order,
                device_name=device_name,
            )
            statevector_normalization_error = abs(
                float(np.linalg.norm(final_state)) - 1.0
            )
            postselected_output = final_state[:rows]
            logical_output = lcu_normalization * postselected_output
            if reference is not None:
                absolute_error, relative_error = _complex_output_errors(
                    logical_output,
                    reference,
                )
        resource_summary = _coherent_qsvt_resource_summary(
            spec,
            term_factories,
            selection_state,
            branch_phases,
            prepared,
            selectors,
            order,
            component_ledger,
            lcu_normalization=lcu_normalization,
            device_name=device_name,
            shots=shots,
        )
        resource_summary.update(
            {
                "selection_success_probability": selection_success_probability,
                "logical_success_probability": logical_success_probability,
                "amplitude_amplification_overhead_proxy": (
                    None
                    if logical_success_probability <= 0.0
                    else float(1.0 / np.sqrt(logical_success_probability))
                ),
            }
        )
    except Exception as exc:
        error_type = type(exc).__name__
        error = str(exc)
        if raise_on_failure:
            raise

    return CoherentQSVTExecutionResult(
        spec=spec,
        components=raw_components,
        normalized_components=tuple(normalized_components),
        component_weights=tuple(component_weights),
        component_syntheses=tuple(component_syntheses),
        input_state=logical_state,
        wire_order=tuple(order),
        selection_wires=tuple(selectors),
        angle_solver=str(angle_solver),
        device_name=device_name,
        shots=shots,
        succeeded=error is None,
        lcu_normalization=lcu_normalization,
        selection_state_amplitudes=selection_state,
        final_state=final_state,
        probabilities=probabilities,
        postselected_logical_output=postselected_output,
        logical_output=logical_output,
        selection_success_probability=selection_success_probability,
        logical_success_probability=logical_success_probability,
        selection_success_standard_error=selection_success_standard_error,
        logical_success_standard_error=logical_success_standard_error,
        maximum_probability_standard_error=maximum_probability_standard_error,
        classical_reference_output=reference,
        logical_output_absolute_error=absolute_error,
        logical_output_relative_error=relative_error,
        probability_normalization_error=probability_normalization_error,
        statevector_normalization_error=statevector_normalization_error,
        resource_summary=resource_summary,
        error_type=error_type,
        error=error,
    )


def execute_mixed_parity_qsvt_from_spec(
    spec: BlockEncodingSpec,
    poly: Iterable[float],
    state: Iterable[float | complex],
    **kwargs: Any,
) -> CoherentQSVTExecutionResult:
    """Execute a real polynomial through coherent even/odd QSVT combination."""
    coeffs = _validate_coefficients(poly)
    even, odd = parity_components(coeffs)
    components = tuple(
        component
        for component in (
            CoherentQSVTComponent("even", even),
            CoherentQSVTComponent("odd", odd),
        )
        if np.any(np.abs(component.coeffs) > 1e-15)
    )
    return execute_qsvt_component_lcu_from_spec(
        spec,
        components,
        state,
        **kwargs,
    )


def qsvt_circuit_truth_contract() -> dict[str, object]:
    """
    Return the claim boundary for circuit-level QSVT execution helpers.
    """
    return {
        "implementation_kind": "pennylane-qnode-qsvt-execution",
        "truth_status": "queued_qsvt_circuit_execution",
        "is_end_to_end_quantum_algorithm": False,
        "implemented_components": [
            "state_preparation_on_qnode_register",
            "pennylane_qsvt_operation_queued_in_qnode",
            "statevector_or_probability_measurement",
            "circuit_resource_summary",
        ],
        "reference_components": [
            "optional_dense_classical_polynomial_output_for_validation",
        ],
        "omitted_quantum_components": [
            "scalable_problem_oracle_or_block_encoding",
            "application_state_preparation_cost",
            "postselection_success_management",
            "readout_or_tomography_cost",
            "fault_tolerant_synthesis",
            "hardware_compilation",
        ],
    }


def _resolve_spec_wire_order(
    spec: BlockEncodingSpec,
    wire_order: Iterable[Any] | None,
) -> list[Any]:
    if wire_order is not None:
        order = list(wire_order)
    elif spec.kind == "pennylane-operator":
        source = cast(Any, spec.source)
        order = list(spec.encoding_wires) + [
            wire for wire in source.wires if wire not in spec.encoding_wires
        ]
    elif spec.kind == "custom-circuit":
        operation = build_block_encoding_operator(spec)
        order = list(operation.wires)
    else:
        order = list(spec.encoding_wires)
    if not order:
        raise ValueError("wire_order must contain at least one wire.")
    if len(set(order)) != len(order):
        raise ValueError("wire_order must contain distinct wires.")
    if not set(spec.encoding_wires).issubset(set(order)):
        raise ValueError("wire_order must contain all encoding_wires.")
    return order


def _spec_qsvt_operation_factory(
    spec: BlockEncodingSpec,
    *,
    angles: np.ndarray | None,
    projectors: Sequence[qml.operation.Operator] | None,
):
    def operation():
        if spec.block_encoding == "fable" and not bool(
            spec.metadata.get("fable_compatible", False)
        ):
            raise ValueError(
                "The normalized matrix does not satisfy PennyLane's FABLE "
                "normalization condition."
            )
        block_encoding = _execution_block_encoding_operator(spec)
        resolved_projectors = (
            list(projectors)
            if projectors is not None
            else _projectors_from_angles(spec, cast(np.ndarray, angles))
        )
        return qml.QSVT(block_encoding, resolved_projectors)

    return operation


def _execution_block_encoding_operator(
    spec: BlockEncodingSpec,
) -> qml.operation.Operator:
    """Build an encoding without PennyLane's implicit dense-matrix rescaling.

    ``qml.BlockEncode`` applies an additional infinity-norm normalization for
    some non-diagonal contractions. That changes the encoded matrix relative to
    ``BlockEncodingSpec.alpha``. The execution path therefore uses the exact
    unitary dilation for dense embedding specifications, preserving the
    normalization contract recorded by the spec.
    """
    if spec.kind not in {"dense-matrix", "sparse-matrix"}:
        return build_block_encoding_operator(spec)
    if spec.block_encoding != "embedding":
        return build_block_encoding_operator(spec)

    matrix = np.asarray(spec.dense_matrix(), dtype=complex) / spec.alpha
    rows, columns = matrix.shape
    left = _positive_semidefinite_sqrt(
        np.eye(rows, dtype=complex) - matrix @ matrix.conj().T
    )
    right = _positive_semidefinite_sqrt(
        np.eye(columns, dtype=complex) - matrix.conj().T @ matrix
    )
    dilation = np.block(
        [
            [matrix, left],
            [right, -matrix.conj().T],
        ]
    )
    full_dimension = 2 ** len(spec.encoding_wires)
    if dilation.shape[0] < full_dimension:
        padding = full_dimension - dilation.shape[0]
        dilation = np.block(
            [
                [
                    dilation,
                    np.zeros((dilation.shape[0], padding), dtype=complex),
                ],
                [
                    np.zeros((padding, dilation.shape[1]), dtype=complex),
                    np.eye(padding, dtype=complex),
                ],
            ]
        )
    return qml.QubitUnitary(dilation, wires=spec.encoding_wires)


def _positive_semidefinite_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Return a stable square root for a numerically positive matrix."""
    hermitian = (matrix + matrix.conj().T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    if float(np.min(eigenvalues)) < -1e-9:
        raise ValueError(
            "the normalized matrix is not a contraction under the declared alpha."
        )
    clipped = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * np.sqrt(clipped)) @ eigenvectors.conj().T


def _projectors_from_angles(
    spec: BlockEncodingSpec,
    angles: np.ndarray,
) -> list[qml.operation.Operator]:
    rows, columns = spec.logical_shape
    if spec.kind == "pennylane-operator":
        source = cast(Any, spec.source)
        wires = list(spec.encoding_wires) + list(source.wires)
        dimensions = [rows] * len(angles)
    elif spec.block_encoding == "fable":
        wires = list(spec.encoding_wires)
        dimensions = [rows] * len(angles)
    else:
        wires = list(spec.encoding_wires)
        dimensions = [
            rows if index % 2 == 0 else columns for index in range(len(angles))
        ]
    return [
        qml.PCPhase(float(angle), dim=int(dimension), wires=wires)
        for angle, dimension in zip(angles, dimensions, strict=True)
    ]


def _execute_spec_state_qnode(
    operation_factory,
    prepared: np.ndarray,
    wire_order: list[Any],
    *,
    device_name: str,
) -> np.ndarray:
    dev = qml.device(device_name, wires=wire_order)

    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(prepared, wires=wire_order)
        operation_factory()
        return qml.state()

    return np.asarray(circuit(), dtype=complex)


def _execute_spec_probability_qnode(
    operation_factory,
    prepared: np.ndarray,
    wire_order: list[Any],
    *,
    device_name: str,
    shots: int | None,
) -> np.ndarray:
    dev = qml.device(device_name, wires=wire_order)

    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(prepared, wires=wire_order)
        operation_factory()
        return qml.probs(wires=wire_order)

    executable = qml.set_shots(circuit, shots=shots) if shots is not None else circuit
    return np.asarray(executable(), dtype=float)


def _resolve_selection_wires(
    data_wires: Sequence[Any],
    count: int,
    selection_wires: Iterable[Any] | None,
) -> list[Any]:
    if selection_wires is not None:
        resolved = list(selection_wires)
        if len(resolved) != count:
            raise ValueError(
                f"selection_wires must contain exactly {count} wire labels."
            )
    else:
        occupied = set(data_wires)
        resolved = []
        index = 0
        while len(resolved) < count:
            candidate = f"__qsvt_lcu_{index}__"
            if candidate not in occupied:
                resolved.append(candidate)
            index += 1
    if len(set(resolved)) != len(resolved):
        raise ValueError("selection_wires must contain distinct wire labels.")
    if set(resolved).intersection(data_wires):
        raise ValueError("selection_wires must not overlap block-encoding wires.")
    return resolved


def _apply_coherent_lcu(
    term_factories: Sequence[tuple[object, bool]],
    selection_state: np.ndarray,
    branch_phases: np.ndarray,
    selection_wires: Sequence[Any],
) -> None:
    qml.StatePrep(selection_state, wires=selection_wires)
    qml.DiagonalQubitUnitary(
        np.exp(1j * branch_phases),
        wires=selection_wires,
    )
    selector_count = len(selection_wires)
    for index, (factory, use_adjoint) in enumerate(term_factories):
        control_values = tuple(int(bit) for bit in format(index, f"0{selector_count}b"))
        operation = qml.adjoint(factory) if use_adjoint else factory
        qml.ctrl(
            operation,
            control=selection_wires,
            control_values=control_values,
        )()
    qml.adjoint(qml.StatePrep)(selection_state, wires=selection_wires)


def _execute_coherent_state_qnode(
    term_factories: Sequence[tuple[object, bool]],
    selection_state: np.ndarray,
    branch_phases: np.ndarray,
    prepared: np.ndarray,
    selection_wires: Sequence[Any],
    data_wires: Sequence[Any],
    *,
    device_name: str,
) -> np.ndarray:
    full_order = list(selection_wires) + list(data_wires)
    dev = qml.device(device_name, wires=full_order)

    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(prepared, wires=data_wires)
        _apply_coherent_lcu(
            term_factories,
            selection_state,
            branch_phases,
            selection_wires,
        )
        return qml.state()

    return np.asarray(circuit(), dtype=complex)


def _execute_coherent_probability_qnode(
    term_factories: Sequence[tuple[object, bool]],
    selection_state: np.ndarray,
    branch_phases: np.ndarray,
    prepared: np.ndarray,
    selection_wires: Sequence[Any],
    data_wires: Sequence[Any],
    *,
    device_name: str,
    shots: int | None,
) -> np.ndarray:
    full_order = list(selection_wires) + list(data_wires)
    dev = qml.device(device_name, wires=full_order)

    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(prepared, wires=data_wires)
        _apply_coherent_lcu(
            term_factories,
            selection_state,
            branch_phases,
            selection_wires,
        )
        return qml.probs(wires=full_order)

    executable = qml.set_shots(circuit, shots=shots) if shots is not None else circuit
    return np.asarray(executable(), dtype=float)


def _base_coherent_resource_summary(
    spec: BlockEncodingSpec,
    data_wires: Sequence[Any],
    selection_wires: Sequence[Any],
    component_ledger: Sequence[dict[str, object]],
    *,
    lcu_normalization: float,
    shots: int | None,
) -> dict[str, object]:
    return {
        "block_encoding_kind": spec.kind,
        "block_encoding_method": spec.block_encoding,
        "execution_block_encoding_operation": _execution_block_encoding_label(spec),
        "normalization_alpha": spec.alpha,
        "logical_shape": spec.logical_shape,
        "encoding_wire_count": len(spec.encoding_wires),
        "data_wire_count": len(data_wires),
        "selection_ancilla_count": len(selection_wires),
        "total_wire_count": len(data_wires) + len(selection_wires),
        "component_sequence_count": len(component_ledger),
        "selected_unitary_branch_count": 2 * len(component_ledger),
        "real_part_extraction": "(U + U_adjoint) / 2",
        "lcu_normalization": lcu_normalization,
        "component_resource_ledger": list(component_ledger),
        "total_phase_count": sum(
            2 * int(cast(Any, component["phase_count_per_sequence"]))
            for component in component_ledger
        ),
        "total_signal_operator_calls": sum(
            int(cast(Any, component["total_signal_operator_calls"]))
            for component in component_ledger
        ),
        "selector_state_preparations": 2,
        "shots": shots,
        "num_gates": None,
        "depth": None,
        "gate_types": {},
        "measurement_count": None,
        "selection_success_probability": None,
        "logical_success_probability": None,
        "amplitude_amplification_overhead_proxy": None,
        "omitted_costs": [
            "application_state_preparation",
            "amplitude_amplification",
            "application_readout_or_tomography",
            "provider_compilation_and_routing",
            "error_correction_cycle_time",
        ],
    }


def _coherent_qsvt_resource_summary(
    spec: BlockEncodingSpec,
    term_factories: Sequence[tuple[object, bool]],
    selection_state: np.ndarray,
    branch_phases: np.ndarray,
    prepared: np.ndarray,
    selection_wires: Sequence[Any],
    data_wires: Sequence[Any],
    component_ledger: Sequence[dict[str, object]],
    *,
    lcu_normalization: float,
    device_name: str,
    shots: int | None,
) -> dict[str, object]:
    full_order = list(selection_wires) + list(data_wires)
    dev = qml.device(device_name, wires=full_order)

    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(prepared, wires=data_wires)
        _apply_coherent_lcu(
            term_factories,
            selection_state,
            branch_phases,
            selection_wires,
        )
        return qml.probs(wires=full_order)

    executable = qml.set_shots(circuit, shots=shots) if shots is not None else circuit
    specs = qml.specs(executable)()
    resources = _spec_value(specs, "resources")
    summary = _base_coherent_resource_summary(
        spec,
        data_wires,
        selection_wires,
        component_ledger,
        lcu_normalization=lcu_normalization,
        shots=shots,
    )
    summary.update(
        {
            "device_name": _spec_value(specs, "device_name", device_name),
            "num_device_wires": _object_to_int(
                _spec_value(specs, "num_device_wires", len(full_order))
            ),
            "num_gates": _object_to_int(_resource_value(resources, "num_gates", 0)),
            "depth": _object_to_int(_resource_value(resources, "depth", 0)),
            "gate_types": _object_to_dict(_resource_value(resources, "gate_types", {})),
            "measurement_count": _object_len(
                _resource_value(resources, "measurements", [])
            ),
        }
    )
    return summary


def _coherent_classical_reference(
    spec: BlockEncodingSpec,
    components: Sequence[CoherentQSVTComponent],
    state: np.ndarray,
) -> np.ndarray | None:
    outputs: list[np.ndarray] = []
    for component in components:
        if component.coefficient == 0.0 or not np.any(np.abs(component.coeffs) > 1e-15):
            continue
        output = _spec_classical_reference_output(spec, component.coeffs, state)
        if output is None:
            return None
        outputs.append(
            complex(component.coefficient) * np.asarray(output, dtype=complex)
        )
    if not outputs:
        return np.zeros(spec.logical_shape[0], dtype=complex)
    return np.real_if_close(sum(outputs, np.zeros_like(outputs[0])))


def _synthesize_component_with_fallback(
    coeffs: np.ndarray,
    *,
    angle_solver: str,
    reconstruction_num_points: int,
    phase_reconstruction_tolerance: float,
) -> PhaseSynthesisResult:
    attempts: list[PhaseSynthesisResult] = []
    solvers = tuple(dict.fromkeys((str(angle_solver), "root-finding", "iterative")))
    for solver in solvers:
        result = synthesize_phases(
            coeffs,
            angle_solver=solver,
            reconstruction_num_points=reconstruction_num_points,
        )
        attempts.append(result)
        if (
            result.succeeded
            and result.angles is not None
            and result.reconstruction_max_error is not None
            and result.reconstruction_max_error <= phase_reconstruction_tolerance
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


def _complex_output_errors(
    logical_output: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, float]:
    difference = np.asarray(logical_output, dtype=complex) - np.asarray(
        reference,
        dtype=complex,
    )
    denominator = float(np.linalg.norm(reference))
    numerator = float(np.linalg.norm(difference))
    relative = numerator if denominator == 0.0 else numerator / denominator
    return numerator, relative


def _failed_coherent_execution_result(
    spec: BlockEncodingSpec,
    components: tuple[CoherentQSVTComponent, ...],
    normalized_components: tuple[CoherentQSVTComponent, ...],
    component_weights: tuple[tuple[str, float], ...],
    component_syntheses: tuple[tuple[str, PhaseSynthesisResult], ...],
    logical_state: np.ndarray,
    wire_order: Sequence[Any],
    *,
    angle_solver: str,
    device_name: str,
    shots: int | None,
    error_type: str | None,
    error: str,
) -> CoherentQSVTExecutionResult:
    return CoherentQSVTExecutionResult(
        spec=spec,
        components=components,
        normalized_components=normalized_components,
        component_weights=component_weights,
        component_syntheses=component_syntheses,
        input_state=logical_state,
        wire_order=tuple(wire_order),
        selection_wires=(),
        angle_solver=str(angle_solver),
        device_name=device_name,
        shots=shots,
        succeeded=False,
        lcu_normalization=0.0,
        selection_state_amplitudes=np.array([], dtype=complex),
        final_state=None,
        probabilities=None,
        postselected_logical_output=None,
        logical_output=None,
        selection_success_probability=None,
        logical_success_probability=None,
        selection_success_standard_error=None,
        logical_success_standard_error=None,
        maximum_probability_standard_error=None,
        classical_reference_output=_coherent_classical_reference(
            spec,
            components,
            logical_state,
        ),
        logical_output_absolute_error=None,
        logical_output_relative_error=None,
        probability_normalization_error=None,
        statevector_normalization_error=None,
        resource_summary=_base_coherent_resource_summary(
            spec,
            wire_order,
            (),
            (),
            lcu_normalization=0.0,
            shots=shots,
        ),
        error_type=error_type,
        error=error,
    )


def _base_spec_resource_summary(
    spec: BlockEncodingSpec,
    coeffs: np.ndarray,
    wire_order: list[Any],
    *,
    projector_source: str,
    phase_count: int,
    shots: int | None,
) -> dict[str, object]:
    return {
        "block_encoding_kind": spec.kind,
        "block_encoding_method": spec.block_encoding,
        "execution_block_encoding_operation": _execution_block_encoding_label(spec),
        "normalization_alpha": spec.alpha,
        "logical_shape": spec.logical_shape,
        "encoding_wire_count": len(spec.encoding_wires),
        "total_wire_count": len(wire_order),
        "polynomial_degree": int(coeffs.size - 1),
        "phase_count": phase_count,
        "signal_operator_calls": max(0, phase_count - 1),
        "projector_source": projector_source,
        "shots": shots,
        "num_gates": None,
        "depth": None,
        "gate_types": {},
        "measurement_count": None,
    }


def _execution_block_encoding_label(spec: BlockEncodingSpec) -> str:
    if (
        spec.kind in {"dense-matrix", "sparse-matrix"}
        and spec.block_encoding == "embedding"
    ):
        return "exact-unitary-dilation"
    return str(spec.block_encoding)


def _spec_qsvt_resource_summary(
    spec: BlockEncodingSpec,
    coeffs: np.ndarray,
    operation_factory,
    prepared: np.ndarray,
    wire_order: list[Any],
    *,
    projector_source: str,
    phase_count: int,
    device_name: str,
    shots: int | None,
) -> dict[str, object]:
    dev = qml.device(device_name, wires=wire_order)

    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(prepared, wires=wire_order)
        operation_factory()
        return qml.probs(wires=wire_order)

    executable = qml.set_shots(circuit, shots=shots) if shots is not None else circuit
    specs = qml.specs(executable)()
    resources = _spec_value(specs, "resources")
    summary = _base_spec_resource_summary(
        spec,
        coeffs,
        wire_order,
        projector_source=projector_source,
        phase_count=phase_count,
        shots=shots,
    )
    summary.update(
        {
            "device_name": _spec_value(specs, "device_name", device_name),
            "num_device_wires": _object_to_int(
                _spec_value(specs, "num_device_wires", len(wire_order))
            ),
            "num_gates": _object_to_int(_resource_value(resources, "num_gates", 0)),
            "depth": _object_to_int(_resource_value(resources, "depth", 0)),
            "gate_types": _object_to_dict(_resource_value(resources, "gate_types", {})),
            "measurement_count": _object_len(
                _resource_value(resources, "measurements", [])
            ),
        }
    )
    return summary


def _spec_classical_reference_output(
    spec: BlockEncodingSpec,
    coeffs: np.ndarray,
    state: np.ndarray,
) -> np.ndarray | None:
    matrix: np.ndarray | None = None
    if spec.kind in {"dense-matrix", "sparse-matrix"}:
        matrix = spec.dense_matrix() / spec.alpha
    elif spec.kind == "pennylane-operator":
        source = cast(Any, spec.source)
        system_wires = list(source.wires)
        matrix = (
            np.asarray(
                qml.matrix(source, wire_order=system_wires),
                dtype=complex,
            )
            / spec.alpha
        )
    if matrix is None:
        return None
    if matrix.shape[0] == matrix.shape[1] and np.allclose(
        matrix,
        matrix.conj().T,
        atol=1e-10,
        rtol=1e-10,
    ):
        transformed = apply_polynomial_to_hermitian(matrix, coeffs)
    else:
        left, singular_values, right_adjoint = np.linalg.svd(
            matrix,
            full_matrices=False,
        )
        transformed = (
            left * np.polynomial.polynomial.polyval(singular_values, coeffs)
        ) @ right_adjoint
    return np.real_if_close(transformed @ state)


def _real_output_errors(
    logical_output: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, float]:
    difference = np.real(logical_output) - np.real(reference)
    denominator = float(np.linalg.norm(reference))
    numerator = float(np.linalg.norm(difference))
    relative = numerator if denominator == 0.0 else numerator / denominator
    return numerator, relative


def _execute_state_qnode(
    operator: np.ndarray,
    coeffs: np.ndarray,
    prepared: np.ndarray,
    encoding_wires: list[int],
    wire_order: list[int],
    *,
    block_encoding: str,
    angle_solver: str,
    device_name: str,
) -> np.ndarray:
    dev = qml.device(device_name, wires=wire_order)

    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(prepared, wires=wire_order)
        qml.qsvt(
            operator,
            coeffs,
            encoding_wires=encoding_wires,
            block_encoding=block_encoding,
            angle_solver=angle_solver,
        )
        return qml.state()

    return np.asarray(circuit(), dtype=complex)


def _execute_probability_qnode(
    operator: np.ndarray,
    coeffs: np.ndarray,
    prepared: np.ndarray,
    encoding_wires: list[int],
    wire_order: list[int],
    *,
    block_encoding: str,
    angle_solver: str,
    device_name: str,
    shots: int | None,
) -> np.ndarray:
    dev = qml.device(device_name, wires=wire_order)

    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(prepared, wires=wire_order)
        qml.qsvt(
            operator,
            coeffs,
            encoding_wires=encoding_wires,
            block_encoding=block_encoding,
            angle_solver=angle_solver,
        )
        return qml.probs(wires=wire_order)

    executable = qml.set_shots(circuit, shots=shots) if shots is not None else circuit
    return np.asarray(executable(), dtype=float)


def _qsvt_circuit_resource_summary(
    operator: np.ndarray,
    coeffs: np.ndarray,
    prepared: np.ndarray,
    encoding_wires: list[int],
    wire_order: list[int],
    *,
    block_encoding: str,
    angle_solver: str,
    device_name: str,
    shots: int | None,
) -> dict[str, object]:
    dev = qml.device(device_name, wires=wire_order)

    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(prepared, wires=wire_order)
        qml.qsvt(
            operator,
            coeffs,
            encoding_wires=encoding_wires,
            block_encoding=block_encoding,
            angle_solver=angle_solver,
        )
        return qml.probs(wires=wire_order)

    executable = qml.set_shots(circuit, shots=shots) if shots is not None else circuit
    specs = qml.specs(executable)()
    resources = _spec_value(specs, "resources")
    return {
        "device_name": _spec_value(specs, "device_name", device_name),
        "num_device_wires": _object_to_int(
            _spec_value(specs, "num_device_wires", len(wire_order))
        ),
        "shots": shots,
        "num_gates": _object_to_int(_resource_value(resources, "num_gates", 0)),
        "depth": _object_to_int(_resource_value(resources, "depth", 0)),
        "gate_types": _object_to_dict(_resource_value(resources, "gate_types", {})),
        "measurement_count": _object_len(
            _resource_value(resources, "measurements", [])
        ),
        "polynomial_degree": int(coeffs.size - 1),
        "encoding_wire_count": len(encoding_wires),
    }


def _spec_value(specs: object, key: str, default: object | None = None) -> object:
    if hasattr(specs, key):
        return getattr(specs, key)
    if isinstance(specs, dict):
        return specs.get(key, default)
    get_value = getattr(specs, "get", None)
    if callable(get_value):
        return get_value(key, default)
    to_dict = getattr(specs, "to_dict", None)
    if callable(to_dict):
        return to_dict().get(key, default)
    return default


def _resource_value(
    resources: object,
    key: str,
    default: object | None = None,
) -> object:
    if hasattr(resources, key):
        return getattr(resources, key)
    if isinstance(resources, dict):
        return resources.get(key, default)
    get_value = getattr(resources, "get", None)
    if callable(get_value):
        return get_value(key, default)
    return default


def _object_to_int(value: object) -> int:
    return int(cast(Any, value))


def _object_to_dict(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    return dict(cast(Any, value))


def _object_len(value: object) -> int:
    if isinstance(value, Sized):
        return len(value)
    return len(cast(Any, value))


def _validate_execution_operator(operator: np.ndarray) -> np.ndarray:
    if np.isscalar(operator):
        raise ValueError("operator must be a square Hermitian matrix, not a scalar.")
    A = _as_numeric_operator(operator)
    if not np.all(np.isfinite(A)):
        raise ValueError("operator entries must be finite.")
    if not np.allclose(A, A.conj().T, atol=1e-10, rtol=1e-10):
        raise ValueError("operator must be Hermitian / symmetric.")
    evals = np.linalg.eigvalsh(A)
    if np.max(np.abs(evals)) > 1.0 + 1e-12:
        raise ValueError("operator eigenvalues must lie in [-1, 1].")
    return A


def _validate_coefficients(poly: Iterable[float]) -> np.ndarray:
    coeffs = np.asarray(list(poly), dtype=float)
    if coeffs.ndim != 1 or coeffs.size == 0:
        raise ValueError("poly must contain at least one coefficient.")
    if not np.all(np.isfinite(coeffs)):
        raise ValueError("polynomial coefficients must be finite.")
    return coeffs


def _validate_logical_state(
    state: Iterable[float | complex],
    *,
    dimension: int,
    normalize: bool,
) -> np.ndarray:
    logical = np.asarray(list(state), dtype=complex)
    if logical.shape != (dimension,):
        raise ValueError("state length must match the operator dimension.")
    if not np.all(np.isfinite(logical)):
        raise ValueError("state entries must be finite.")
    norm = float(np.linalg.norm(logical))
    if norm == 0.0:
        raise ValueError("state must be nonzero.")
    if normalize:
        return logical / norm
    if not np.isclose(norm, 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError("state must be normalized, or pass normalize_state=True.")
    return logical


__all__ = [
    "BlockEncodingQSVTExecutionResult",
    "CoherentQSVTComponent",
    "CoherentQSVTExecutionResult",
    "QSVTCircuitExecutionResult",
    "execute_qsvt_circuit",
    "execute_mixed_parity_qsvt_from_spec",
    "execute_qsvt_component_lcu_from_spec",
    "execute_qsvt_from_spec",
    "qsvt_circuit_truth_contract",
]
