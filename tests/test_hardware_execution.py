import numpy as np
import pennylane as qml
import pytest

from qsvt.block_encoding import matrix_block_encoding_spec
from qsvt.hardware import (
    execute_qsvt_on_device,
    qsvt_hardware_circuit_report,
    qsvt_hardware_preflight,
    qsvt_hardware_truth_contract,
    qsvt_provider_plugin_report,
)
from qsvt.reports import report_to_jsonable


def _basis_zero_preparation():
    return None


def test_execute_qsvt_on_device_runs_finite_shot_probability_path():
    spec = matrix_block_encoding_spec(np.diag([0.2, 0.8]), alpha=1.0)
    device = qml.device("default.qubit", wires=spec.encoding_wires)

    result = execute_qsvt_on_device(
        spec,
        [0.0, 0.0, 1.0],
        _basis_zero_preparation,
        device,
        shots=200,
    )
    report = report_to_jsonable(result.as_report())

    assert result.succeeded is True
    assert result.shots == 200
    assert result.probabilities is not None
    assert result.probabilities.shape == (4,)
    assert np.isclose(np.sum(result.probabilities), 1.0, atol=1e-12)
    assert result.logical_probabilities is not None
    assert result.logical_success_probability is not None
    assert result.logical_success_standard_error is not None
    assert result.maximum_probability_standard_error is not None
    assert result.preflight.passed is True
    assert result.resource_summary["compiled_depth"] is None
    assert result.resource_summary["compilation_status"] == "not_requested"
    assert result.resource_summary["logical_gate_types"]["QSVT"] == 1
    assert report["schema_name"] == "hardware-qsvt-execution"
    assert report["schema_version"] == "1.0"
    assert report["truth_contract"]["implemented_components"] == [
        "caller_supplied_pennylane_device",
        "caller_supplied_preparation_circuit",
        "caller_selected_block_encoding_access_model",
        "finite_shot_probability_measurement",
        "preflight_device_compatibility_checks",
        "logical_circuit_resource_summary",
    ]


def test_hardware_preflight_requires_finite_positive_shots():
    spec = matrix_block_encoding_spec(np.diag([0.2, 0.8]), alpha=1.0)
    device = qml.device("default.qubit", wires=spec.encoding_wires)

    with pytest.raises(ValueError, match="finite shot"):
        qsvt_hardware_preflight(
            spec,
            [0.0, 1.0],
            _basis_zero_preparation,
            device,
            shots=None,
        )

    with pytest.raises(ValueError, match="shots must be positive"):
        execute_qsvt_on_device(
            spec,
            [0.0, 1.0],
            _basis_zero_preparation,
            device,
            shots=0,
        )


def test_hardware_preflight_rejects_stateprep_by_default():
    spec = matrix_block_encoding_spec(np.diag([0.2, 0.8]), alpha=1.0)
    device = qml.device("default.qubit", wires=spec.encoding_wires)

    def stateprep_preparation():
        qml.StatePrep(np.array([1.0, 0.0, 0.0, 0.0]), wires=spec.encoding_wires)

    result = execute_qsvt_on_device(
        spec,
        [0.0, 1.0],
        stateprep_preparation,
        device,
        shots=100,
    )

    assert result.succeeded is False
    assert result.error_type == "HardwarePreflightError"
    assert result.probabilities is None
    assert result.preflight.checks["stateprep_allowed"] is False
    assert "StatePrep-style operations" in result.error


def test_hardware_preflight_reports_wire_mismatch_without_execution():
    spec = matrix_block_encoding_spec(np.diag([0.2, 0.8]), alpha=1.0)
    device = qml.device("default.qubit", wires=[0])

    with pytest.raises(ValueError, match="contain all encoding_wires"):
        qsvt_hardware_preflight(
            spec,
            [0.0, 1.0],
            _basis_zero_preparation,
            device,
            wire_order=[0],
            shots=100,
        )


def test_provider_plugin_report_reads_fake_backend_metadata():
    device = qml.device("default.qubit", wires=[0, 1])
    device.provider_name = "ExampleProvider"
    device.backend_name = "fake_two_qubit_backend"
    device.backend_version = "2026.7"
    device.plugin_name = "example-pennylane-plugin"
    device.plugin_version = "0.1"
    device.native_gate_set = ("Hadamard", "QSVT")
    device.min_shots = 10
    device.max_shots = 1000
    device.is_fake_backend = True

    report = qsvt_provider_plugin_report(
        device,
        plugin_packages=("pennylane", "definitely-not-installed-qsvt-plugin"),
    )
    payload = report.as_report()

    assert report.provider_name == "ExampleProvider"
    assert report.backend_name == "fake_two_qubit_backend"
    assert report.plugin_name == "example-pennylane-plugin"
    assert report.plugin_version == "0.1"
    assert report.native_gate_set == ("Hadamard", "QSVT")
    assert report.min_shots == 10
    assert report.max_shots == 1000
    assert report.is_fake_backend is True
    assert payload["installed_packages"]["pennylane"] is not None
    assert payload["installed_packages"]["definitely-not-installed-qsvt-plugin"] is None


def test_fake_backend_metadata_is_recorded_on_successful_execution():
    spec = matrix_block_encoding_spec(np.diag([0.2, 0.8]), alpha=1.0)
    device = qml.device("default.qubit", wires=spec.encoding_wires)
    device.provider_name = "ExampleProvider"
    device.backend_name = "fake_qsvt_backend"
    device.plugin_name = "example-pennylane-plugin"
    device.plugin_version = "0.1"
    device.native_gate_set = ("QSVT",)
    device.max_shots = 500
    device.is_fake_backend = True

    result = execute_qsvt_on_device(
        spec,
        [0.0, 0.0, 1.0],
        _basis_zero_preparation,
        device,
        shots=200,
    )

    provider = result.preflight.metadata["provider_plugin"]
    resource_provider = result.resource_summary["provider_plugin"]
    assert result.succeeded is True
    assert result.preflight.checks["shots_within_device_limits"] is True
    assert provider["provider_name"] == "ExampleProvider"
    assert provider["backend_name"] == "fake_qsvt_backend"
    assert provider["is_fake_backend"] is True
    assert resource_provider["plugin_name"] == "example-pennylane-plugin"


def test_fake_backend_preflight_rejects_native_gate_and_shot_mismatches():
    spec = matrix_block_encoding_spec(np.diag([0.2, 0.8]), alpha=1.0)
    device = qml.device("default.qubit", wires=spec.encoding_wires)
    device.provider_name = "ExampleProvider"
    device.backend_name = "fake_restricted_backend"
    device.native_gate_set = ("Hadamard",)
    device.max_shots = 50
    device.is_fake_backend = True

    result = execute_qsvt_on_device(
        spec,
        [0.0, 1.0],
        _basis_zero_preparation,
        device,
        shots=100,
    )

    assert result.succeeded is False
    assert result.error_type == "HardwarePreflightError"
    assert result.preflight.checks["shots_within_device_limits"] is False
    assert result.preflight.checks["operations_supported_when_known"] is False
    assert result.preflight.unsupported_operations == ("QSVT",)
    assert "device limits" in result.error
    assert "unsupported operations" in result.error


def test_hardware_circuit_report_audits_decomposition_without_execution():
    spec = matrix_block_encoding_spec(np.diag([0.2, 0.8]), alpha=1.0)
    device = qml.device("default.qubit", wires=spec.encoding_wires)
    device.provider_name = "ExampleProvider"
    device.backend_name = "fake_decomposed_backend"
    device.native_gate_set = ("PCPhase", "BlockEncode")
    device.max_shots = 1000
    device.is_fake_backend = True

    report = qsvt_hardware_circuit_report(
        spec,
        [0.0, 1.0],
        _basis_zero_preparation,
        device,
        shots=100,
    )
    payload = report_to_jsonable(report.as_report())

    assert report.logical_operations == ("QSVT",)
    assert report.decomposed_operations == ("PCPhase", "BlockEncode", "PCPhase")
    assert report.unsupported_logical_operations == ("QSVT",)
    assert report.unsupported_decomposed_operations == ()
    assert report.decomposition_status == "succeeded"
    assert report.logical_resource_summary["gate_types"] == {"QSVT": 1}
    assert report.decomposed_resource_summary["gate_types"] == {
        "BlockEncode": 1,
        "PCPhase": 2,
    }
    assert payload["executed"] is False
    assert payload["schema_name"] == "hardware-qsvt-circuit"
    assert payload["provider_plugin"]["backend_name"] == "fake_decomposed_backend"


def test_hardware_circuit_report_shows_remaining_unsupported_decomposed_ops():
    spec = matrix_block_encoding_spec(np.diag([0.2, 0.8]), alpha=1.0)
    device = qml.device("default.qubit", wires=spec.encoding_wires)
    device.provider_name = "ExampleProvider"
    device.backend_name = "fake_restricted_decomposed_backend"
    device.native_gate_set = ("Hadamard",)
    device.is_fake_backend = True

    report = qsvt_hardware_circuit_report(
        spec,
        [0.0, 1.0],
        _basis_zero_preparation,
        device,
        shots=100,
    )

    assert report.unsupported_logical_operations == ("QSVT",)
    assert report.unsupported_decomposed_operations == ("BlockEncode", "PCPhase")
    assert report.preflight.passed is False
    assert report.preflight.unsupported_operations == ("QSVT",)


def test_hardware_circuit_report_can_disable_decomposition():
    spec = matrix_block_encoding_spec(np.diag([0.2, 0.8]), alpha=1.0)
    device = qml.device("default.qubit", wires=spec.encoding_wires)

    report = qsvt_hardware_circuit_report(
        spec,
        [0.0, 1.0],
        _basis_zero_preparation,
        device,
        shots=100,
        max_decomposition_depth=0,
    )

    assert report.decomposed_operations == ("QSVT",)
    assert report.decomposition_status == "succeeded"


def test_qsvt_hardware_truth_contract_marks_omitted_provider_layers():
    contract = qsvt_hardware_truth_contract()

    assert contract["truth_status"] == "finite_shot_device_execution_with_preflight"
    assert contract["is_end_to_end_quantum_algorithm"] is False
    assert "provider_credentials" in contract["omitted_quantum_components"]
