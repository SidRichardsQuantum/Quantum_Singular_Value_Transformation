import numpy as np
import pennylane as qml
import pytest

import qsvt.block_encoding as block_encoding_module
from qsvt.algorithms import block_encoded_qsvt_workflow
from qsvt.block_encoding import (
    BlockEncodingSpec,
    block_encode_matrix,
    block_encoding_report,
    build_block_encoding_operator,
    circuit_block_encoding_spec,
    extract_block_encoded_operator,
    matrix_block_encoding_spec,
    pennylane_operator_block_encoding_spec,
    qsvt_operator_from_block_encoding,
    verify_block_encoding,
)
from qsvt.matrices import rotated_diagonal
from qsvt.reports import report_to_jsonable, validate_report_schema
from qsvt.spectral import apply_polynomial_to_hermitian


def test_unitary_dilation_block_encoding_recovers_operator():
    matrix = np.array([[2.0, 0.5], [0.5, -1.0]])

    encoding = block_encode_matrix(matrix)
    verification = verify_block_encoding(encoding)

    assert encoding.logical_dimension == 2
    assert encoding.unitary_dimension == 4
    assert encoding.ancilla_dimension == 2
    assert verification["block_encoding_verified"] is True
    assert verification["unitary_verified"] is True
    assert encoding.block_error() < 1e-12
    assert encoding.unitarity_error() < 1e-12
    assert np.allclose(encoding.reconstruction(), matrix, atol=1e-12)
    assert np.allclose(
        extract_block_encoded_operator(
            encoding.unitary,
            encoding.logical_dimension,
            alpha=encoding.alpha,
        ),
        matrix,
        atol=1e-12,
    )


def test_block_encoding_report_and_zero_operator_defaults():
    report = block_encoding_report(np.zeros((2, 2)))

    assert report["mode"] == "block-encoding-report"
    assert report["alpha"] == 1.0
    assert report["logical_dimension"] == 2
    assert report["unitary_dimension"] == 4
    assert report["block_error"] == pytest.approx(0.0)
    assert report["unitarity_error"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.ones((1, 2)), "square 2D matrix"),
        (np.array([[np.inf]]), "entries must be finite"),
    ],
)
def test_block_encode_matrix_rejects_invalid_matrices(matrix, message):
    with pytest.raises(ValueError, match=message):
        block_encode_matrix(matrix)


@pytest.mark.parametrize("alpha", [0.0, -1.0, np.inf])
def test_block_encode_matrix_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="positive and finite"):
        block_encode_matrix(np.eye(2), alpha=alpha)


def test_block_encode_matrix_rejects_alpha_below_spectral_norm():
    with pytest.raises(ValueError, match="at least the spectral norm"):
        block_encode_matrix(np.eye(2), alpha=0.5)


def test_extract_block_encoded_operator_validates_dimension_and_alpha():
    unitary = np.eye(2)

    with pytest.raises(ValueError, match="logical_dimension must be positive"):
        extract_block_encoded_operator(unitary, 0)
    with pytest.raises(ValueError, match="cannot exceed unitary dimension"):
        extract_block_encoded_operator(unitary, 3)
    with pytest.raises(ValueError, match="positive and finite"):
        extract_block_encoded_operator(unitary, 1, alpha=0.0)


def test_hermitian_psd_sqrt_rejects_negative_spectrum():
    with pytest.raises(ValueError, match="positive semidefinite"):
        block_encoding_module._hermitian_psd_sqrt(np.diag([-0.1, 1.0]))


def test_matrix_block_encoding_spec_supports_rectangular_matrices():
    matrix = np.array([[0.2, 0.1, 0.0], [0.0, 0.3, 0.1]])
    spec = matrix_block_encoding_spec(matrix, alpha=0.5)
    report = spec.as_report()

    assert isinstance(spec, BlockEncodingSpec)
    assert spec.logical_shape == (2, 3)
    assert spec.is_rectangular is True
    assert spec.execution_supported is False
    assert report["kind"] == "dense-matrix"
    assert report["high_level_qsvt_supported"] is False
    assert report["lower_level_qsvt_supported"] is True
    assert np.allclose(spec.dense_matrix(), matrix)
    assert isinstance(build_block_encoding_operator(spec), qml.BlockEncode)

    with pytest.raises(NotImplementedError, match="square Hermitian"):
        qsvt_operator_from_block_encoding(spec, [0.0, 1.0])


def test_matrix_block_encoding_spec_accepts_sparse_like_inputs():
    class SparseLike:
        def toarray(self):
            return np.diag([0.2, 0.8])

    spec = matrix_block_encoding_spec(SparseLike())

    assert spec.kind == "sparse-matrix"
    assert spec.execution_supported is True
    assert np.allclose(spec.dense_matrix(), np.diag([0.2, 0.8]))


def test_fable_spec_reports_normalization_constraint():
    spec = matrix_block_encoding_spec(
        np.eye(2),
        alpha=1.0,
        block_encoding="fable",
    )

    assert spec.metadata["fable_compatible"] is False
    assert spec.execution_supported is False
    assert "FABLE" in spec.execution_reason


def test_pennylane_operator_block_encoding_spec_builds_qsvt_adapter():
    operator = qml.dot([0.3, 0.7], [qml.Z(1), qml.X(1)])
    spec = pennylane_operator_block_encoding_spec(
        operator,
        encoding_wires=[0],
        block_encoding="prepselprep",
    )

    assert spec.kind == "pennylane-operator"
    assert spec.alpha == pytest.approx(1.0)
    assert spec.execution_supported is True
    assert isinstance(build_block_encoding_operator(spec), qml.PrepSelPrep)
    assert qsvt_operator_from_block_encoding(spec, [0.0, 1.0]).name == "QSVT"


def test_custom_circuit_block_encoding_spec_queues_factory():
    spec = circuit_block_encoding_spec(
        lambda: qml.Hadamard(0),
        logical_shape=(1, 1),
        encoding_wires=[0],
        metadata={"signal_projector": "caller-supplied"},
    )

    assert spec.kind == "custom-circuit"
    assert build_block_encoding_operator(spec).name == "Hadamard"
    assert spec.as_report()["metadata"]["signal_projector"] == "caller-supplied"
    assert spec.as_report()["lower_level_qsvt_supported"] is True


def test_block_encoded_qsvt_workflow_matches_spectral_reference():
    matrix = rotated_diagonal([0.25, 0.8], theta=0.31)
    coeffs = np.array([0.0, 0.0, 1.0])
    state = np.array([1.0, -0.25])

    result = block_encoded_qsvt_workflow(matrix, coeffs, state=state)
    reference = apply_polynomial_to_hermitian(
        result.block_encoding.signal_operator,
        coeffs,
    )
    report = report_to_jsonable(result.as_report())

    assert validate_report_schema(report, require_schema=True).supported is True
    assert report["mode"] == "block-encoded-qsvt-workflow"
    assert report["implementation_kind"] == (
        "verified-dense-block-encoded-qsvt-workflow"
    )
    assert report["truth_contract"]["implementation_kind"] == (
        "verified-dense-block-encoded-qsvt-workflow"
    )
    assert report["truth_contract"]["pennylane_qsvt_check"] == "succeeded"
    assert report["verification"]["block_encoding_verified"] is True
    assert report["verification"]["unitary_verified"] is True
    assert np.allclose(result.reference_operator, reference, atol=1e-12)
    assert result.qsvt_operator is not None
    assert np.allclose(result.qsvt_operator, reference, atol=1e-10)
    assert result.operator_relative_error is not None
    assert result.operator_relative_error < 1e-10
    assert result.qsvt_state is not None
    assert result.reference_state is not None
    assert result.state_relative_error is not None
    assert result.state_relative_error < 1e-10
