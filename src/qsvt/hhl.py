"""
Circuit-level HHL execution helpers.

The helpers in this module queue the HHL algorithmic primitives inside a
PennyLane QNode: state preparation, quantum phase estimation, controlled
reciprocal rotation, inverse phase estimation, and measurement/postselection.

Dense matrix inputs use explicit ``QubitUnitary`` time-evolution matrices so
that examples are executable on ``default.qubit``. For scalable use, callers
can provide ``controlled_time_evolution`` to queue problem-specific sparse
Hamiltonian simulation or oracle blocks without materializing dense powers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pennylane as qml

from .execution import (
    _object_len,
    _object_to_dict,
    _object_to_int,
    _resource_value,
    _spec_value,
)
from .spectral import _as_hermitian_matrix

ControlledTimeEvolution = Callable[[int, int, tuple[int, ...]], None]


@dataclass(frozen=True)
class HHLCircuitExecutionResult:
    """
    Result from executing an HHL circuit through a PennyLane QNode.
    """

    matrix: np.ndarray | None
    input_state: np.ndarray
    phase_wires: tuple[int, ...]
    system_wires: tuple[int, ...]
    ancilla_wire: int
    wire_order: tuple[int, ...]
    num_phase_qubits: int
    evolution_time: float
    rotation_scale_C: float
    eigenvalue_lower_bound: float
    device_name: str
    shots: int | None
    final_state: np.ndarray | None
    probabilities: np.ndarray
    postselected_solution_state: np.ndarray | None
    success_probability: float
    classical_solution_state: np.ndarray | None
    fidelity: float | None
    state_error: float | None
    execution_kind: str
    resource_summary: dict[str, object]
    uses_dense_time_evolution: bool

    def as_report(self) -> dict[str, object]:
        """
        Return a report preserving the circuit-execution claim boundary.
        """
        return {
            "mode": "hhl-circuit-execution-report",
            "implementation_kind": self.execution_kind,
            "is_executable_hhl_circuit": True,
            "is_scalable_hhl_implementation": not self.uses_dense_time_evolution,
            "implemented_components": [
                "qnode_state_preparation",
                "coherent_quantum_phase_estimation",
                "controlled_reciprocal_eigenvalue_rotation",
                "inverse_quantum_phase_estimation",
                "ancilla_postselection",
                "circuit_resource_summary",
            ],
            "omitted_quantum_components": [
                "problem_specific_state_preparation_cost",
                "amplitude_amplification",
                "application_specific_readout_or_tomography",
                "fault_tolerant_synthesis",
                "hardware_compilation",
            ],
            "scalability_note": (
                "Dense matrix inputs execute a real HHL circuit on a simulator "
                "but are not scalable. Supplying controlled_time_evolution "
                "moves Hamiltonian simulation behind an oracle-compatible "
                "interface."
            ),
            "phase_wires": self.phase_wires,
            "system_wires": self.system_wires,
            "ancilla_wire": self.ancilla_wire,
            "wire_order": self.wire_order,
            "num_phase_qubits": self.num_phase_qubits,
            "evolution_time": self.evolution_time,
            "rotation_scale_C": self.rotation_scale_C,
            "eigenvalue_lower_bound": self.eigenvalue_lower_bound,
            "device_name": self.device_name,
            "shots": self.shots,
            "final_state": self.final_state,
            "probabilities": self.probabilities,
            "postselected_solution_state": self.postselected_solution_state,
            "success_probability": self.success_probability,
            "classical_solution_state": self.classical_solution_state,
            "fidelity": self.fidelity,
            "state_error": self.state_error,
            "resource_summary": self.resource_summary,
        }


def execute_hhl_circuit(
    matrix: np.ndarray | None,
    state: Iterable[float | complex],
    *,
    num_phase_qubits: int,
    evolution_time: float,
    rotation_scale_C: float,
    eigenvalue_lower_bound: float | None = None,
    phase_wires: Iterable[int] | None = None,
    system_wires: Iterable[int] | None = None,
    ancilla_wire: int | None = None,
    wire_order: Iterable[int] | None = None,
    controlled_time_evolution: ControlledTimeEvolution | None = None,
    device_name: str = "default.qubit",
    shots: int | None = None,
    normalize_state: bool = False,
) -> HHLCircuitExecutionResult:
    """
    Execute HHL as a PennyLane circuit.

    Parameters
    ----------
    matrix
        Hermitian positive-definite matrix used by the dense simulator path.
        May be ``None`` only when ``controlled_time_evolution`` and
        ``system_wires`` are supplied.
    state
        Normalized right-hand-side state ``|b>``.
    num_phase_qubits
        Number of phase-estimation qubits.
    evolution_time
        Time ``t`` in ``exp(i A t)``.
    rotation_scale_C
        Reciprocal-rotation scale. It must satisfy
        ``0 < C <= eigenvalue_lower_bound``.
    eigenvalue_lower_bound
        Known lower spectral bound for the occupied eigenvalue range. Phase
        labels whose inferred eigenvalue is below this bound receive no
        reciprocal rotation, avoiding non-unitary amplitudes above one. Defaults
        to ``rotation_scale_C``.
    controlled_time_evolution
        Optional oracle hook. The callable receives ``(power, control_wire,
        system_wires)`` and must queue controlled ``exp(i A t * power)``.

    Notes
    -----
    This is a true circuit execution path: the HHL primitives are queued and
    run in a QNode. Dense matrix input remains simulator-scale because it
    materializes controlled time-evolution unitaries; scalable deployments
    should supply ``controlled_time_evolution``.
    """
    m = _validate_num_phase_qubits(num_phase_qubits)
    t = _validate_positive_float(evolution_time, "evolution_time")
    C = _validate_positive_float(rotation_scale_C, "rotation_scale_C")
    lower_bound = (
        C
        if eigenvalue_lower_bound is None
        else _validate_positive_float(eigenvalue_lower_bound, "eigenvalue_lower_bound")
    )
    if C > lower_bound + 1e-12:
        raise ValueError("rotation_scale_C must not exceed eigenvalue_lower_bound.")

    A = None if matrix is None else _validate_hhl_matrix(matrix)
    logical_state = _validate_hhl_state(
        state,
        dimension=None if A is None else A.shape[0],
        normalize=normalize_state,
    )
    system_dimension = logical_state.size
    system_qubit_count = _required_qubits(system_dimension)

    if system_wires is None:
        if A is None:
            raise ValueError("system_wires are required when matrix is None.")
        system_wires = range(system_qubit_count)
    sys_wires = tuple(int(wire) for wire in system_wires)
    if len(sys_wires) != system_qubit_count:
        raise ValueError("system_wires count must match the state dimension.")

    if phase_wires is None:
        phase_wires = range(len(sys_wires), len(sys_wires) + m)
    ph_wires = tuple(int(wire) for wire in phase_wires)
    if len(ph_wires) != m:
        raise ValueError("phase_wires count must equal num_phase_qubits.")

    if ancilla_wire is None:
        ancilla_wire = max((*sys_wires, *ph_wires), default=-1) + 1
    anc = int(ancilla_wire)

    if wire_order is None:
        wire_order = (*ph_wires, anc, *sys_wires)
    order = tuple(int(wire) for wire in wire_order)
    _validate_distinct_wires((*ph_wires, anc, *sys_wires), order)

    shots = _validate_shots(shots)
    prepared = _embedded_state(logical_state, len(sys_wires))
    uses_dense = controlled_time_evolution is None
    if uses_dense and A is None:
        raise ValueError("matrix is required when controlled_time_evolution is absent.")

    probabilities = _execute_hhl_probability_qnode(
        A,
        prepared,
        ph_wires,
        sys_wires,
        anc,
        order,
        t=t,
        C=C,
        lower_bound=lower_bound,
        controlled_time_evolution=controlled_time_evolution,
        device_name=device_name,
        shots=shots,
    )

    final_state = None
    postselected = None
    success_probability = float(
        np.sum(
            _postselection_probabilities(
                probabilities,
                ph_wires,
                sys_wires,
                anc,
                order,
            )
        )
    )
    if shots is None:
        final_state = _execute_hhl_state_qnode(
            A,
            prepared,
            ph_wires,
            sys_wires,
            anc,
            order,
            t=t,
            C=C,
            lower_bound=lower_bound,
            controlled_time_evolution=controlled_time_evolution,
            device_name=device_name,
        )
        raw_postselected = _postselected_system_amplitudes(
            final_state,
            ph_wires,
            sys_wires,
            anc,
            order,
        )[:system_dimension]
        success_probability = float(np.vdot(raw_postselected, raw_postselected).real)
        postselected = _normalize_optional(raw_postselected)

    classical_state = None
    fidelity = None
    state_error = None
    if A is not None and postselected is not None:
        classical_state = _normalize(np.linalg.solve(A, logical_state))
        aligned = _align_global_phase(classical_state, postselected)
        fidelity = float(abs(np.vdot(classical_state, postselected)) ** 2)
        state_error = float(np.linalg.norm(classical_state - aligned))

    execution_kind = (
        "pennylane-qnode-statevector-hhl-execution"
        if shots is None
        else "pennylane-qnode-shot-hhl-execution"
    )

    return HHLCircuitExecutionResult(
        matrix=A,
        input_state=logical_state,
        phase_wires=ph_wires,
        system_wires=sys_wires,
        ancilla_wire=anc,
        wire_order=order,
        num_phase_qubits=m,
        evolution_time=t,
        rotation_scale_C=C,
        eigenvalue_lower_bound=lower_bound,
        device_name=device_name,
        shots=shots,
        final_state=final_state,
        probabilities=probabilities,
        postselected_solution_state=postselected,
        success_probability=success_probability,
        classical_solution_state=classical_state,
        fidelity=fidelity,
        state_error=state_error,
        execution_kind=execution_kind,
        resource_summary=_hhl_circuit_resource_summary(
            A,
            prepared,
            ph_wires,
            sys_wires,
            anc,
            order,
            t=t,
            C=C,
            lower_bound=lower_bound,
            controlled_time_evolution=controlled_time_evolution,
            device_name=device_name,
            shots=shots,
        ),
        uses_dense_time_evolution=uses_dense,
    )


def hhl_circuit_truth_contract() -> dict[str, object]:
    """
    Return the claim boundary for circuit-level HHL execution helpers.
    """
    return {
        "implementation_kind": "pennylane-qnode-hhl-execution",
        "truth_status": "queued_hhl_circuit_execution",
        "is_executable_hhl_circuit": True,
        "is_scalable_hhl_implementation": "requires_oracle_time_evolution",
        "implemented_components": [
            "state_preparation_on_qnode_register",
            "coherent_quantum_phase_estimation",
            "controlled_reciprocal_rotation",
            "inverse_quantum_phase_estimation",
            "ancilla_postselection",
            "statevector_or_probability_measurement",
            "circuit_resource_summary",
        ],
        "dense_fallback_components": [
            "explicit_qubitunitary_hamiltonian_simulation_for_small_matrices",
        ],
        "omitted_quantum_components": [
            "problem_specific_state_preparation_cost",
            "amplitude_amplification",
            "application_specific_readout_or_tomography",
            "fault_tolerant_synthesis",
            "hardware_compilation",
        ],
    }


def _queue_hhl_circuit(
    matrix: np.ndarray | None,
    prepared: np.ndarray,
    phase_wires: tuple[int, ...],
    system_wires: tuple[int, ...],
    ancilla_wire: int,
    *,
    t: float,
    C: float,
    lower_bound: float,
    controlled_time_evolution: ControlledTimeEvolution | None,
) -> None:
    qml.StatePrep(prepared, wires=system_wires)
    for wire in phase_wires:
        qml.Hadamard(wire)
    for index, wire in enumerate(phase_wires):
        power = 2 ** (len(phase_wires) - 1 - index)
        if controlled_time_evolution is None:
            assert matrix is not None
            unitary = _dense_time_evolution_power(matrix, t, power)
            qml.ctrl(qml.QubitUnitary, control=wire)(unitary, wires=system_wires)
        else:
            controlled_time_evolution(power, wire, system_wires)
    qml.adjoint(qml.QFT)(wires=phase_wires)
    _queue_reciprocal_rotations(
        phase_wires,
        ancilla_wire,
        C=C,
        t=t,
        lower_bound=lower_bound,
    )
    qml.QFT(wires=phase_wires)
    for index, wire in reversed(list(enumerate(phase_wires))):
        power = 2 ** (len(phase_wires) - 1 - index)
        if controlled_time_evolution is None:
            assert matrix is not None
            unitary = _dense_time_evolution_power(matrix, t, -power)
            qml.ctrl(qml.QubitUnitary, control=wire)(unitary, wires=system_wires)
        else:
            controlled_time_evolution(-power, wire, system_wires)
    for wire in phase_wires:
        qml.Hadamard(wire)


def _queue_reciprocal_rotations(
    phase_wires: tuple[int, ...],
    ancilla_wire: int,
    *,
    C: float,
    t: float,
    lower_bound: float,
) -> None:
    grid_size = 2 ** len(phase_wires)
    for label in range(1, grid_size):
        estimated_eigenvalue = 2.0 * np.pi * label / (grid_size * t)
        if estimated_eigenvalue < lower_bound - 1e-12:
            continue
        amplitude = C / estimated_eigenvalue
        if amplitude > 1.0 + 1e-12:
            raise ValueError("rotation amplitude exceeds one.")
        theta = 2.0 * np.arcsin(min(1.0, amplitude))
        qml.ctrl(
            qml.RY,
            control=phase_wires,
            control_values=_label_bits(label, len(phase_wires)),
        )(theta, wires=ancilla_wire)


def _execute_hhl_state_qnode(
    matrix: np.ndarray | None,
    prepared: np.ndarray,
    phase_wires: tuple[int, ...],
    system_wires: tuple[int, ...],
    ancilla_wire: int,
    wire_order: tuple[int, ...],
    *,
    t: float,
    C: float,
    lower_bound: float,
    controlled_time_evolution: ControlledTimeEvolution | None,
    device_name: str,
) -> np.ndarray:
    dev = qml.device(device_name, wires=wire_order)

    @qml.qnode(dev)
    def circuit():
        _queue_hhl_circuit(
            matrix,
            prepared,
            phase_wires,
            system_wires,
            ancilla_wire,
            t=t,
            C=C,
            lower_bound=lower_bound,
            controlled_time_evolution=controlled_time_evolution,
        )
        return qml.state()

    return np.asarray(circuit(), dtype=complex)


def _execute_hhl_probability_qnode(
    matrix: np.ndarray | None,
    prepared: np.ndarray,
    phase_wires: tuple[int, ...],
    system_wires: tuple[int, ...],
    ancilla_wire: int,
    wire_order: tuple[int, ...],
    *,
    t: float,
    C: float,
    lower_bound: float,
    controlled_time_evolution: ControlledTimeEvolution | None,
    device_name: str,
    shots: int | None,
) -> np.ndarray:
    dev = qml.device(device_name, wires=wire_order)

    @qml.qnode(dev)
    def circuit():
        _queue_hhl_circuit(
            matrix,
            prepared,
            phase_wires,
            system_wires,
            ancilla_wire,
            t=t,
            C=C,
            lower_bound=lower_bound,
            controlled_time_evolution=controlled_time_evolution,
        )
        return qml.probs(wires=wire_order)

    executable = qml.set_shots(circuit, shots=shots) if shots is not None else circuit
    return np.asarray(executable(), dtype=float)


def _hhl_circuit_resource_summary(
    matrix: np.ndarray | None,
    prepared: np.ndarray,
    phase_wires: tuple[int, ...],
    system_wires: tuple[int, ...],
    ancilla_wire: int,
    wire_order: tuple[int, ...],
    *,
    t: float,
    C: float,
    lower_bound: float,
    controlled_time_evolution: ControlledTimeEvolution | None,
    device_name: str,
    shots: int | None,
) -> dict[str, object]:
    dev = qml.device(device_name, wires=wire_order)

    @qml.qnode(dev)
    def circuit():
        _queue_hhl_circuit(
            matrix,
            prepared,
            phase_wires,
            system_wires,
            ancilla_wire,
            t=t,
            C=C,
            lower_bound=lower_bound,
            controlled_time_evolution=controlled_time_evolution,
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
        "phase_qubits": len(phase_wires),
        "system_qubits": len(system_wires),
        "uses_dense_time_evolution": controlled_time_evolution is None,
    }


def _dense_time_evolution_power(matrix: np.ndarray, t: float, power: int) -> np.ndarray:
    evals, evecs = np.linalg.eigh(matrix)
    phases = np.exp(1j * evals * t * power)
    return evecs @ np.diag(phases) @ evecs.conj().T


def _postselected_system_amplitudes(
    state: np.ndarray,
    phase_wires: tuple[int, ...],
    system_wires: tuple[int, ...],
    ancilla_wire: int,
    wire_order: tuple[int, ...],
) -> np.ndarray:
    tensor = np.asarray(state).reshape((2,) * len(wire_order))
    index: list[int | slice] = []
    for wire in wire_order:
        if wire in phase_wires:
            index.append(0)
        elif wire == ancilla_wire:
            index.append(1)
        elif wire in system_wires:
            index.append(slice(None))
        else:  # pragma: no cover - wire validation prevents this.
            index.append(slice(None))
    return np.asarray(tensor[tuple(index)], dtype=complex).reshape(-1)


def _postselection_probabilities(
    probabilities: np.ndarray,
    phase_wires: tuple[int, ...],
    system_wires: tuple[int, ...],
    ancilla_wire: int,
    wire_order: tuple[int, ...],
) -> np.ndarray:
    tensor = np.asarray(probabilities).reshape((2,) * len(wire_order))
    index: list[int | slice] = []
    for wire in wire_order:
        if wire in phase_wires:
            index.append(0)
        elif wire == ancilla_wire:
            index.append(1)
        elif wire in system_wires:
            index.append(slice(None))
        else:  # pragma: no cover - wire validation prevents this.
            index.append(slice(None))
    return np.asarray(tensor[tuple(index)], dtype=float).reshape(-1)


def _validate_hhl_matrix(matrix: np.ndarray) -> np.ndarray:
    A = _as_hermitian_matrix(matrix).astype(complex, copy=False)
    dimension = A.shape[0]
    if dimension & (dimension - 1):
        raise ValueError("matrix dimension must be a power of two.")
    evals = np.linalg.eigvalsh(A)
    if np.any(evals <= 0.0):
        raise ValueError("HHL requires a positive-definite Hermitian matrix.")
    return A


def _validate_hhl_state(
    state: Iterable[float | complex],
    *,
    dimension: int | None,
    normalize: bool,
) -> np.ndarray:
    vector = np.asarray(list(state), dtype=complex)
    if vector.ndim != 1:
        raise ValueError("state must be a one-dimensional vector.")
    if dimension is not None and vector.size != dimension:
        raise ValueError("state length must match the matrix dimension.")
    if vector.size == 0 or vector.size & (vector.size - 1):
        raise ValueError("state length must be a positive power of two.")
    if not np.all(np.isfinite(vector)):
        raise ValueError("state entries must be finite.")
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError("state must be nonzero.")
    if normalize:
        return vector / norm
    if not np.isclose(norm, 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError("state must be normalized unless normalize_state=True.")
    return vector


def _embedded_state(state: np.ndarray, system_qubit_count: int) -> np.ndarray:
    dimension = 2**system_qubit_count
    prepared = np.zeros(dimension, dtype=complex)
    prepared[: state.size] = state
    return prepared


def _required_qubits(dimension: int) -> int:
    return int(np.log2(dimension))


def _validate_num_phase_qubits(value: int) -> int:
    m = int(value)
    if m <= 0:
        raise ValueError("num_phase_qubits must be positive.")
    return m


def _validate_positive_float(value: float, name: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _validate_shots(shots: int | None) -> int | None:
    if shots is None:
        return None
    result = int(shots)
    if result <= 0:
        raise ValueError("shots must be positive when supplied.")
    return result


def _validate_distinct_wires(
    required_wires: tuple[int, ...],
    wire_order: tuple[int, ...],
) -> None:
    if len(set(required_wires)) != len(required_wires):
        raise ValueError("phase, system, and ancilla wires must be distinct.")
    if len(set(wire_order)) != len(wire_order):
        raise ValueError("wire_order must not contain duplicate wires.")
    if not set(required_wires).issubset(set(wire_order)):
        raise ValueError("wire_order must contain phase, system, and ancilla wires.")


def _label_bits(label: int, width: int) -> list[int]:
    return [int(bit) for bit in f"{label:0{width}b}"]


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError("cannot normalize the zero vector.")
    return vector / norm


def _normalize_optional(vector: np.ndarray) -> np.ndarray | None:
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return None
    return vector / norm


def _align_global_phase(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    overlap = np.vdot(reference, candidate)
    if abs(overlap) == 0.0:
        return candidate
    return candidate * np.conj(overlap) / abs(overlap)


__all__ = [
    "ControlledTimeEvolution",
    "HHLCircuitExecutionResult",
    "execute_hhl_circuit",
    "hhl_circuit_truth_contract",
]
