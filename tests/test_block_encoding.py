import numpy as np

from qsvt.algorithms import block_encoded_qsvt_workflow
from qsvt.block_encoding import (
    block_encode_matrix,
    extract_block_encoded_operator,
    verify_block_encoding,
)
from qsvt.matrices import rotated_diagonal
from qsvt.reports import report_to_jsonable
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
