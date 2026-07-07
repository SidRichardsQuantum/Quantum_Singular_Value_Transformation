import numpy as np

import qsvt
from qsvt.compatibility import qsvt_compatibility_report
from qsvt.diagonal import qsvt_diagonal_transform
from qsvt.matrix import qsvt_matrix_transform_report
from qsvt.operators import qsvt_unitary
from qsvt.qsvt import qsvt_scalar_output
from qsvt.reports import report_to_jsonable
from qsvt.workflow import (
    DesignWorkflowResult,
    QSVTProblemWorkflowResult,
    design_workflow,
    qsvt_problem_workflow,
)


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


def test_design_workflow_supports_interval_projector():
    result = design_workflow(
        "interval_projector",
        lower=-0.2,
        upper=0.4,
        degree=18,
        num_points=301,
        bounded_num_points=601,
        attempt_synthesis=False,
    )

    assert result.builder == "design_interval_projector_polynomial"
    assert result.diagnostics["lower"] == -0.2
    assert result.diagnostics["upper"] == 0.4
    assert result.compatibility["attempted_pennylane_synthesis"] is False


def test_problem_workflow_wraps_linear_system_user_journey():
    matrix = np.diag([1.0, 2.0])
    rhs = np.array([1.0, 1.0])

    result = qsvt_problem_workflow(
        "linear_system",
        matrix,
        rhs=rhs,
        degree=8,
        num_points=301,
        bounded_num_points=601,
    )
    report = result.as_report()

    assert isinstance(result, QSVTProblemWorkflowResult)
    assert result.target == "linear_system"
    assert report["schema_name"] == "qsvt-problem-workflow"
    assert report["schema_version"] == "1.0"
    assert report["target"] == "linear_system"
    assert report["input_kind"] == "finite-square-matrix"
    assert set(report) == {
        "schema_name",
        "schema_version",
        "mode",
        "target",
        "input_kind",
        "implementation_kind",
        "truth_contract",
        "result",
        "resource_reports",
    }
    assert (
        "finite_problem_definition"
        in report["truth_contract"]["implemented_components"]
    )
    assert (
        "problem_specific_scalable_block_encoding"
        in report["truth_contract"]["omitted_quantum_components"]
    )
    assert report["result"]["mode"] == "linear-system-workflow"
    assert report["resource_reports"][0]["component"] == "coeffs"
    assert report["resource_reports"][0]["resources"]["matrix_dimension"] == 2


def test_problem_workflow_reports_multiple_polynomial_components():
    matrix = np.diag([-0.5, 0.5])

    result = qsvt_problem_workflow(
        "resolvent",
        matrix,
        omega=0.2,
        eta=0.4,
        degree=6,
        num_points=301,
    )
    components = {report["component"] for report in result.resource_reports}

    assert result.as_report()["result"]["mode"] == "resolvent-workflow"
    assert components == {"real_coeffs", "imag_coeffs"}


def test_problem_workflow_validates_required_inputs():
    matrix = np.diag([1.0, 2.0])

    try:
        qsvt_problem_workflow("linear_system", matrix, degree=4)
    except ValueError as exc:
        assert "rhs is required" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing rhs to fail")


def test_problem_workflow_is_exported_as_stable_root_api():
    assert qsvt.qsvt_problem_workflow is qsvt_problem_workflow
    assert qsvt.api_status("qsvt_problem_workflow") == "stable"
    assert qsvt.api_status("QSVTProblemWorkflowResult") == "stable"
