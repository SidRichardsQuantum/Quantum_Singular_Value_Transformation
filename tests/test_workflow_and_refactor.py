import numpy as np

from qsvt.compatibility import qsvt_compatibility_report
from qsvt.diagonal import qsvt_diagonal_transform
from qsvt.matrix import qsvt_matrix_transform_report
from qsvt.operators import qsvt_unitary
from qsvt.qsvt import qsvt_scalar_output
from qsvt.reports import report_to_jsonable
from qsvt.workflow import DesignWorkflowResult, design_workflow


def test_split_modules_preserve_qsvt_compatibility_imports():
    assert np.isclose(qsvt_scalar_output(0.5, [0.0, 0.0, 1.0]), 0.25)
    values = qsvt_diagonal_transform([0.2, 0.4], [0.0, 1.0])
    assert np.allclose(values, [0.2, 0.4], atol=1e-10)


def test_qsvt_unitary_default_wires_support_non_power_of_two_matrix():
    matrix = np.diag([0.1, 0.3, 0.5])
    unitary = qsvt_unitary(matrix, [0.0, 1.0])

    assert unitary.shape == (8, 8)


def test_matrix_report_serializes_complex_hermitian_payloads():
    matrix = np.array([[0.0, 0.2j], [-0.2j, 0.0]], dtype=complex)
    report = qsvt_matrix_transform_report(matrix, [0.0, 1.0])
    payload = report_to_jsonable(report)

    assert report["comparison_basis"] == "full_complex"
    assert payload["input"]["real"][0][1] == 0.0
    assert payload["input"]["imag"][0][1] == 0.2
    assert payload["classical"]["real"][0][1] == 0.0
    assert np.isclose(payload["classical"]["imag"][0][1], 0.2)


def test_compatibility_report_handles_non_finite_coefficients_without_synthesis():
    report = qsvt_compatibility_report(
        [0.0, np.inf],
        attempt_synthesis=False,
    )

    assert report["compatible"] is False
    assert "non_finite_coefficients" in report["reasons"]
    assert report["pennylane_synthesis_succeeded"] is None


def test_design_workflow_returns_coefficients_reports_and_compatibility():
    result = design_workflow(
        "sign",
        degree=5,
        gamma=0.25,
        num_points=101,
        bounded_num_points=201,
        attempt_synthesis=False,
    )
    report = result.as_report()

    assert isinstance(result, DesignWorkflowResult)
    assert result.kind == "sign"
    assert result.builder == "design_sign_polynomial"
    assert result.coeffs.ndim == 1
    assert result.diagnostics["builder"] == "design_sign_polynomial"
    assert result.compatibility["attempted_pennylane_synthesis"] is False
    assert report["mode"] == "design-workflow"
