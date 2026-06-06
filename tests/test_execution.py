import numpy as np
import pennylane as qml
import pytest

from qsvt.execution import execute_qsvt_circuit, qsvt_circuit_truth_contract
from qsvt.reports import report_to_jsonable


def test_execute_qsvt_circuit_statevector_runs_qnode_without_matrix_extraction(
    monkeypatch,
):
    matrix = np.diag([0.2, 0.8])
    coeffs = [0.0, 0.0, 1.0]
    state = np.array([1.0, 0.0])
    original_matrix = qml.matrix

    def fail_qsvt_matrix(*args, **kwargs):  # pragma: no cover - regression guard
        operation = args[0] if args else None
        operation_name = operation.__class__.__name__
        if operation_name in {"BlockEncode", "QSVT"}:
            raise AssertionError("execute_qsvt_circuit must not extract a QSVT matrix")
        return original_matrix(*args, **kwargs)

    monkeypatch.setattr(qml, "matrix", fail_qsvt_matrix)

    result = execute_qsvt_circuit(matrix, coeffs, state, encoding_wires=[0, 1])
    report = report_to_jsonable(result.as_report())

    assert result.execution_kind == "pennylane-qnode-statevector-qsvt-execution"
    assert result.final_state is not None
    assert result.final_state.shape == (4,)
    assert result.logical_output is not None
    assert np.allclose(
        np.real(result.logical_output),
        result.classical_reference_output,
        atol=1e-10,
    )
    assert result.logical_success_probability <= 1.0 + 1e-12
    assert result.resource_summary["gate_types"]["QSVT"] == 1
    assert result.resource_summary["gate_types"]["StatePrep"] == 1
    assert report["implementation_kind"] == (
        "pennylane-qnode-statevector-qsvt-execution"
    )
    assert report["is_end_to_end_quantum_algorithm"] is False


def test_execute_qsvt_circuit_with_shots_returns_measured_probabilities_only():
    result = execute_qsvt_circuit(
        np.diag([0.2, 0.8]),
        [0.0, 0.0, 1.0],
        [1.0, 0.0],
        encoding_wires=[0, 1],
        shots=200,
    )

    assert result.execution_kind == "pennylane-qnode-shot-qsvt-execution"
    assert result.shots == 200
    assert result.final_state is None
    assert result.logical_output is None
    assert result.probabilities.shape == (4,)
    assert np.isclose(np.sum(result.probabilities), 1.0, atol=1e-12)


def test_execute_qsvt_circuit_requires_normalized_state_unless_requested():
    matrix = np.diag([0.2, 0.8])

    with pytest.raises(ValueError, match="state must be normalized"):
        execute_qsvt_circuit(matrix, [0.0, 0.0, 1.0], [2.0, 0.0])

    result = execute_qsvt_circuit(
        matrix,
        [0.0, 0.0, 1.0],
        [2.0, 0.0],
        normalize_state=True,
    )

    assert np.allclose(result.input_state, [1.0, 0.0])


def test_qsvt_circuit_truth_contract_marks_missing_algorithm_layers():
    contract = qsvt_circuit_truth_contract()

    assert contract["truth_status"] == "queued_qsvt_circuit_execution"
    assert contract["is_end_to_end_quantum_algorithm"] is False
    assert (
        "scalable_problem_oracle_or_block_encoding"
        in contract["omitted_quantum_components"]
    )
