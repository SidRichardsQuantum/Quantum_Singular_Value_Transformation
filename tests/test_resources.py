import pytest

import qsvt
from qsvt.resources import estimate_qsvt_resources, qsvt_resource_report
from qsvt.workflow import design_workflow


def test_estimate_qsvt_resources_infers_width_from_dimension():
    estimate = estimate_qsvt_resources([0.0, 0.0, 1.0], matrix_dimension=3)
    report = estimate.as_report()

    assert report["degree"] == 2
    assert report["coefficient_count"] == 3
    assert report["qsp_phase_count"] == 3
    assert report["signal_operator_calls"] == 2
    assert report["inverse_signal_operator_calls"] == 2
    assert report["encoding_qubits"] == 2
    assert report["total_qubits"] == 3
    assert report["estimate_kind"] == "proxy"
    assert report["requires_block_encoding"] is True
    assert report["requires_state_preparation"] is True
    assert report["fault_tolerant_estimate"] is False
    assert "block_encoding_construction" in report["omitted_costs"]
    assert "unused basis states" in " ".join(report["notes"])


def test_qsvt_resource_report_combines_resources_and_compatibility():
    report = qsvt_resource_report(
        [0.0, 0.0, 1.0],
        matrix_dimension=2,
        attempt_synthesis=False,
    )

    assert report["mode"] == "resource-report"
    assert report["resources"]["degree"] == 2
    assert report["resources"]["encoding_qubits"] == 1
    assert report["compatibility"]["attempted_pennylane_synthesis"] is False
    assert report["estimate_kind"] == "proxy"
    assert report["resources"]["estimate_kind"] == "proxy"
    assert report["requires_block_encoding"] is True
    assert report["requires_state_preparation"] is True
    assert report["fault_tolerant_estimate"] is False
    assert "state_preparation" in report["omitted_costs"]
    assert report["limitations"]


def test_resource_estimate_rejects_invalid_width():
    with pytest.raises(ValueError, match="cannot represent"):
        estimate_qsvt_resources(
            [0.0, 1.0],
            matrix_dimension=5,
            encoding_qubits=2,
        )


def test_top_level_exports_resource_helpers():
    assert qsvt.ResourceEstimate is not None
    assert qsvt.estimate_qsvt_resources is estimate_qsvt_resources
    assert qsvt.qsvt_resource_report is qsvt_resource_report
    assert "qsvt_resource_report" in qsvt.__all__


def test_design_workflow_can_emit_resource_report_with_diagnostics():
    result = design_workflow(
        "sign",
        gamma=0.25,
        degree=5,
        num_points=51,
        bounded_num_points=101,
        attempt_synthesis=False,
    )

    report = result.resource_report(matrix_dimension=2)

    assert report["mode"] == "resource-report"
    assert report["kind"] == "sign"
    assert report["builder"] == "design_sign_polynomial"
    assert report["diagnostics"]["builder"] == "design_sign_polynomial"
    assert report["resources"]["matrix_dimension"] == 2
    assert report["compatibility"]["attempted_pennylane_synthesis"] is False
