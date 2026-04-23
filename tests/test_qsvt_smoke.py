# test_qsvt_smoke.py

import numpy as np

from qsvt.polynomials import chebyshev_t, polynomial_parity
from qsvt.qsvt import (
    qsvt_diagonal_transform,
    qsvt_compatibility_report,
    qsvt_scalar_output,
    qsvt_transform_report,
)


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


def test_qsvt_transform_report_matches_classical_polynomial():
    report = qsvt_transform_report(
        [1.0, 0.7, 0.3, 0.1],
        [0, 0, 1],
        encoding_wires=[0, 1, 2],
    )

    assert report["mode"] == "qsvt-transform-report"
    assert report["qsvt_succeeded"] is True
    assert report["polynomial_degree"] == 2
    assert report["matrix_dimension"] == 4
    assert np.allclose(report["classical"], [1.0, 0.49, 0.09, 0.01])
    assert np.allclose(report["qsvt"], report["classical"], atol=1e-10)
    assert report["max_error"] < 1e-10


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
