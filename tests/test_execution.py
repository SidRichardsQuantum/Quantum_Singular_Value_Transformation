from types import SimpleNamespace

import numpy as np
import pennylane as qml
import pytest

import qsvt.execution as execution_module
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


@pytest.mark.parametrize(
    ("matrix", "state", "message"),
    [
        (0.5, [1.0], "not a scalar"),
        (np.array([[np.inf, 0.0], [0.0, 0.2]]), [1.0, 0.0], "entries must be finite"),
        (np.array([[0.0, 0.2], [0.0, 0.0]]), [1.0, 0.0], "Hermitian"),
        (np.diag([1.2, 0.1]), [1.0, 0.0], r"\[-1, 1\]"),
    ],
)
def test_execute_qsvt_circuit_rejects_invalid_operators(matrix, state, message):
    with pytest.raises(ValueError, match=message):
        execute_qsvt_circuit(matrix, [0.0, 1.0], state)


@pytest.mark.parametrize(
    ("poly", "message"),
    [
        ([], "poly must contain"),
        ([np.inf], "polynomial coefficients must be finite"),
    ],
)
def test_execute_qsvt_circuit_rejects_invalid_polynomials(poly, message):
    with pytest.raises(ValueError, match=message):
        execute_qsvt_circuit(np.diag([0.2, 0.8]), poly, [1.0, 0.0])


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ([1.0], "state length"),
        ([np.nan, 0.0], "state entries must be finite"),
        ([0.0, 0.0], "state must be nonzero"),
    ],
)
def test_execute_qsvt_circuit_rejects_invalid_states(state, message):
    with pytest.raises(ValueError, match=message):
        execute_qsvt_circuit(np.diag([0.2, 0.8]), [0.0, 1.0], state)


def test_execute_qsvt_circuit_validates_wires_and_shots():
    matrix = np.diag([0.2, 0.8])

    with pytest.raises(ValueError, match="contain all encoding_wires"):
        execute_qsvt_circuit(
            matrix,
            [0.0, 1.0],
            [1.0, 0.0],
            encoding_wires=[0, 1],
            wire_order=[0],
        )

    with pytest.raises(ValueError, match="shots must be positive"):
        execute_qsvt_circuit(matrix, [0.0, 1.0], [1.0, 0.0], shots=0)


def test_execution_spec_resource_adapters_support_mapping_and_object_shapes():
    class DictLike:
        def __init__(self, payload):
            self.payload = payload

        def get(self, key, default=None):
            return self.payload.get(key, default)

    class ToDictOnly:
        def to_dict(self):
            return {"device_name": "custom.device"}

    resource_object = SimpleNamespace(num_gates=2, depth=3, gate_types={"QSVT": 1})
    resource_mapping = {"num_gates": 4, "measurements": ["probs"]}

    assert execution_module._spec_value({"resources": resource_mapping}, "resources")
    assert (
        execution_module._spec_value(
            DictLike({"device_name": "dictlike.device"}),
            "device_name",
        )
        == "dictlike.device"
    )
    assert execution_module._spec_value(ToDictOnly(), "device_name") == "custom.device"
    assert execution_module._spec_value(object(), "missing", "fallback") == "fallback"
    assert execution_module._resource_value(resource_object, "depth") == 3
    assert execution_module._resource_value(resource_mapping, "num_gates") == 4
    assert execution_module._resource_value(DictLike({"depth": 5}), "depth") == 5
    assert execution_module._resource_value(object(), "missing", "fallback") == (
        "fallback"
    )
    assert execution_module._object_to_dict([("Hadamard", 1)]) == {"Hadamard": 1}
    with pytest.raises(TypeError):
        execution_module._object_len(iter([1, 2, 3]))


def test_qsvt_circuit_truth_contract_marks_missing_algorithm_layers():
    contract = qsvt_circuit_truth_contract()

    assert contract["truth_status"] == "queued_qsvt_circuit_execution"
    assert contract["is_end_to_end_quantum_algorithm"] is False
    assert (
        "scalable_problem_oracle_or_block_encoding"
        in contract["omitted_quantum_components"]
    )
