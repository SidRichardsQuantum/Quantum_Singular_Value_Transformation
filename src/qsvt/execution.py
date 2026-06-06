"""
Circuit-level QSVT execution helpers.

This module keeps a clear boundary between dense reference calculations and
quantum-circuit execution. The helpers here run a PennyLane QNode with state
preparation, a queued QSVT operator, and a measurement. They do not materialize
the QSVT unitary with ``qml.matrix``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sized
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pennylane as qml

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


def execute_qsvt_circuit(
    operator: np.ndarray,
    poly: Iterable[float],
    state: Iterable[float | complex],
    *,
    encoding_wires: Iterable[int] | None = None,
    wire_order: Iterable[int] | None = None,
    block_encoding: str = "embedding",
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
            device_name=device_name,
            shots=shots,
        ),
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


def _execute_state_qnode(
    operator: np.ndarray,
    coeffs: np.ndarray,
    prepared: np.ndarray,
    encoding_wires: list[int],
    wire_order: list[int],
    *,
    block_encoding: str,
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
    "QSVTCircuitExecutionResult",
    "execute_qsvt_circuit",
    "qsvt_circuit_truth_contract",
]
