# test_qsvt_smoke.py

import numpy as np

from qsvt.polynomials import chebyshev_t, polynomial_parity
from qsvt.qsvt import qsvt_scalar_output, qsvt_diagonal_transform


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
