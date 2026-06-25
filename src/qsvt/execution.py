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
from .spectral import apply_polynomial_to_hermitian


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
        block_encoding = build_block_encoding_operator(spec)
        resolved_projectors = (
            list(projectors)
            if projectors is not None
            else _projectors_from_angles(spec, cast(np.ndarray, angles))
        )
        return qml.QSVT(block_encoding, resolved_projectors)

    return operation


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
    "QSVTCircuitExecutionResult",
    "execute_qsvt_circuit",
    "execute_qsvt_from_spec",
    "qsvt_circuit_truth_contract",
]
