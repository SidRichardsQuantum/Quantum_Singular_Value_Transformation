# test_qsvt_smoke.py

import numpy as np

from qsvt.matrices import rotated_diagonal
from qsvt.polynomials import chebyshev_t, polynomial_parity
from qsvt.qsvt import (
    compare_qsvt_vs_classical_matrix,
    qsvt_compatibility_report,
    qsvt_diagonal_transform,
    qsvt_matrix_transform,
    qsvt_matrix_transform_report,
    qsvt_scalar_output,
    qsvt_transform_report,
)
from qsvt.spectral import apply_polynomial_to_hermitian


def test_chebyshev_t3_at_half():
    assert np.isclose(chebyshev_t(3, 0.5), -1.0)


def test_polynomial_parity_x_squared():
    assert polynomial_parity([0, 0, 1]) == "even"


def test_qsvt_scalar_x_squared():
    out = qsvt_scalar_output(0.5, [0, 0, 1], encoding_wires=[0])
    assert np.isclose(out, 0.25, atol=1e-10)


def test_qsvt_diagonal_x_squared():
    vals = qsvt_diagonal_transform(
        [1.0, 0.7, 0.3, 0.1],
        [0, 0, 1],
        encoding_wires=[0, 1, 2],
    )
    expected = np.array([1.0, 0.49, 0.09, 0.01])
    assert np.allclose(vals, expected, atol=1e-10)


def test_qsvt_matrix_transform_matches_rotated_hermitian_reference():
    matrix = rotated_diagonal([0.2, 0.8], theta=0.45)
    qsvt_block = qsvt_matrix_transform(matrix, [0, 0, 1])
    classical = apply_polynomial_to_hermitian(matrix, [0, 0, 1])

    assert np.allclose(qsvt_block, classical, atol=1e-10)


def test_compare_qsvt_vs_classical_matrix_tracks_imaginary_block():
    matrix = rotated_diagonal([0.2, 0.8], theta=0.45)
    comparison = compare_qsvt_vs_classical_matrix(matrix, [0, 0, 1])

    assert comparison["comparison_basis"] == "real_part"
    assert np.allclose(comparison["qsvt"], comparison["classical"], atol=1e-10)
    assert np.max(comparison["abs_error"]) < 1e-10
    assert comparison["qsvt_imag"].shape == matrix.shape
    assert np.max(np.abs(comparison["qsvt_imag"])) > 0.0


def test_compare_qsvt_vs_classical_matrix_supports_complex_hermitian_inputs():
    matrix = np.array([[0.0, 0.2j], [-0.2j, 0.0]], dtype=complex)
    comparison = compare_qsvt_vs_classical_matrix(matrix, [0, 1])

    assert comparison["comparison_basis"] == "full_complex"
    assert np.allclose(comparison["qsvt"], comparison["classical"], atol=1e-6)
    assert np.max(comparison["abs_error"]) < 1e-6


def test_qsvt_transform_report_matches_classical_polynomial():
    report = qsvt_transform_report(
        [1.0, 0.7, 0.3, 0.1],
        [0, 0, 1],
        encoding_wires=[0, 1, 2],
    )

    assert report["mode"] == "qsvt-transform-report"
    assert report["truth_contract"]["implementation_kind"] == (
        "pennylane-small-qsvt-verification"
    )
    assert report["truth_contract"]["pennylane_qsvt_check"] == "succeeded"
    assert report["qsvt_succeeded"] is True
    assert report["polynomial_degree"] == 2
    assert report["matrix_dimension"] == 4
    assert np.allclose(report["classical"], [1.0, 0.49, 0.09, 0.01])
    assert np.allclose(report["qsvt"], report["classical"], atol=1e-10)
    assert report["max_error"] < 1e-10


def test_qsvt_matrix_transform_report_matches_classical_polynomial():
    matrix = rotated_diagonal([0.2, 0.8], theta=0.45)
    report = qsvt_matrix_transform_report(matrix, [0, 0, 1])

    assert report["mode"] == "qsvt-matrix-transform-report"
    assert report["truth_contract"]["truth_status"] == (
        "verified_against_classical_polynomial"
    )
    assert report["qsvt_succeeded"] is True
    assert report["polynomial_degree"] == 2
    assert report["matrix_dimension"] == 2
    assert report["unitary_dimension"] == 4
    assert np.allclose(report["eigenvalues"], [0.2, 0.8])
    assert np.allclose(report["qsvt"], report["classical"], atol=1e-10)
    assert report["max_error"] < 1e-10
    assert report["max_imag_abs"] > 0.0


def test_qsvt_matrix_transform_report_supports_complex_hermitian_inputs():
    matrix = np.array([[0.0, 0.2j], [-0.2j, 0.0]], dtype=complex)
    report = qsvt_matrix_transform_report(matrix, [0, 1])

    assert report["comparison_basis"] == "full_complex"
    assert np.allclose(report["qsvt"], report["classical"], atol=1e-6)
    assert report["max_error"] < 1e-6


def test_qsvt_compatibility_report_accepts_x_squared():
    report = qsvt_compatibility_report([0, 0, 1])

    assert report["mode"] == "qsvt-compatibility-report"
    assert report["compatible"] is True
    assert report["parity"] == "even"
    assert report["is_bounded"] is True
    assert report["pennylane_synthesis_succeeded"] is True
    assert report["reasons"] == []


def test_qsvt_compatibility_report_explains_mixed_unbounded_poly():
    report = qsvt_compatibility_report([1, 1], attempt_synthesis=False)

    assert report["compatible"] is False
    assert report["parity"] == "mixed"
    assert "mixed_parity" in report["reasons"]
    assert "out_of_bounds" in report["reasons"]
