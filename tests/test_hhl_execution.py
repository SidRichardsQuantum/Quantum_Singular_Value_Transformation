import numpy as np
import pennylane as qml
import pytest

from qsvt.comparisons import execute_hhl_circuit, hhl_circuit_truth_contract
from qsvt.reports import report_to_jsonable


def test_execute_hhl_circuit_exact_phase_grid_matches_solution_state():
    matrix = np.diag([1.0, 2.0])
    state = np.array([1.0, 1.0]) / np.sqrt(2.0)

    result = execute_hhl_circuit(
        matrix,
        state,
        num_phase_qubits=2,
        evolution_time=np.pi / 2.0,
        rotation_scale_C=1.0,
        phase_wires=[0, 1],
        system_wires=[2],
        ancilla_wire=3,
    )
    report = report_to_jsonable(result.as_report())

    assert result.execution_kind == "pennylane-qnode-statevector-hhl-execution"
    assert result.postselected_solution_state is not None
    assert result.classical_solution_state is not None
    assert np.allclose(
        result.postselected_solution_state,
        result.classical_solution_state,
        atol=1e-10,
    )
    assert np.isclose(result.success_probability, 0.625, atol=1e-10)
    assert result.state_error is not None
    assert result.state_error < 1e-10
    assert result.fidelity is not None
    assert np.isclose(result.fidelity, 1.0, atol=1e-10)
    assert result.resource_summary["gate_types"]["Adjoint(QFT)"] == 1
    assert result.resource_summary["gate_types"]["QFT"] == 1
    assert result.resource_summary["uses_dense_time_evolution"] is True
    assert report["is_executable_hhl_circuit"] is True
    assert report["is_scalable_hhl_implementation"] is False


def test_execute_hhl_circuit_with_shots_returns_probabilities_only():
    result = execute_hhl_circuit(
        np.diag([1.0, 2.0]),
        [1.0, 0.0],
        num_phase_qubits=2,
        evolution_time=np.pi / 2.0,
        rotation_scale_C=1.0,
        shots=200,
    )

    assert result.execution_kind == "pennylane-qnode-shot-hhl-execution"
    assert result.shots == 200
    assert result.final_state is None
    assert result.postselected_solution_state is None
    assert result.probabilities.shape == (16,)
    assert np.isclose(np.sum(result.probabilities), 1.0, atol=1e-12)


def test_execute_hhl_circuit_accepts_oracle_time_evolution_hook():
    t = np.pi / 2.0
    calls = []

    def controlled_time_evolution(power, control_wire, system_wires):
        calls.append((power, control_wire, system_wires))
        unitary = np.diag(
            [
                np.exp(1j * 1.0 * t * power),
                np.exp(1j * 2.0 * t * power),
            ]
        )
        qml.ctrl(qml.QubitUnitary, control=control_wire)(
            unitary,
            wires=system_wires,
        )

    result = execute_hhl_circuit(
        None,
        [1.0, 0.0],
        num_phase_qubits=2,
        evolution_time=t,
        rotation_scale_C=1.0,
        phase_wires=[0, 1],
        system_wires=[2],
        ancilla_wire=3,
        controlled_time_evolution=controlled_time_evolution,
    )
    report = result.as_report()

    expected_calls = [
        (2, 0, (2,)),
        (1, 1, (2,)),
        (-1, 1, (2,)),
        (-2, 0, (2,)),
    ]
    assert len(calls) >= len(expected_calls)
    assert len(calls) % len(expected_calls) == 0
    for offset in range(0, len(calls), len(expected_calls)):
        assert calls[offset : offset + len(expected_calls)] == expected_calls
    assert result.uses_dense_time_evolution is False
    assert result.resource_summary["uses_dense_time_evolution"] is False
    assert report["is_scalable_hhl_implementation"] is True
    assert result.postselected_solution_state is not None
    assert np.allclose(result.postselected_solution_state, [1.0, 0.0], atol=1e-10)


def test_execute_hhl_circuit_validates_rotation_scale_against_phase_grid():
    with pytest.raises(ValueError, match="rotation_scale_C"):
        execute_hhl_circuit(
            np.diag([1.0, 2.0]),
            [1.0, 0.0],
            num_phase_qubits=2,
            evolution_time=np.pi / 2.0,
            rotation_scale_C=1.1,
            eigenvalue_lower_bound=1.0,
        )


def test_hhl_circuit_truth_contract_marks_oracle_requirement():
    contract = hhl_circuit_truth_contract()

    assert contract["truth_status"] == "queued_hhl_circuit_execution"
    assert contract["is_executable_hhl_circuit"] is True
    assert contract["is_scalable_hhl_implementation"] == (
        "requires_oracle_time_evolution"
    )
    assert "coherent_quantum_phase_estimation" in contract["implemented_components"]
