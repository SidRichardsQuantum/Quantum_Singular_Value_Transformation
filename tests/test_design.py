# test_design.py

import numpy as np

from qsvt.design import (
    design_filter_polynomial,
    design_inverse_polynomial,
    design_power_polynomial,
    design_projector_polynomial,
    design_sign_polynomial,
    design_sqrt_polynomial,
)
from qsvt.polynomials import is_bounded_on_interval, polynomial_parity


def test_design_functions_return_numpy_arrays():
    funcs = [
        lambda: design_inverse_polynomial(gamma=0.25, degree=11),
        lambda: design_sign_polynomial(gamma=0.2, degree=13),
        lambda: design_projector_polynomial(gamma=0.2, degree=13),
        lambda: design_sqrt_polynomial(a=0.2, degree=12),
        lambda: design_power_polynomial(alpha=0.5, degree=12, a=0.2),
        lambda: design_filter_polynomial(cutoff=0.45, degree=10),
    ]

    for build in funcs:
        coeffs = build()
        assert isinstance(coeffs, np.ndarray)
        assert coeffs.ndim == 1
        assert coeffs.size >= 1
        assert np.all(np.isfinite(coeffs))


def test_design_polynomials_are_bounded_on_minus_one_one():
    coeff_sets = [
        design_inverse_polynomial(gamma=0.25, degree=11),
        design_sign_polynomial(gamma=0.2, degree=13),
        design_projector_polynomial(gamma=0.2, degree=13),
        design_sqrt_polynomial(a=0.2, degree=12),
        design_power_polynomial(alpha=0.5, degree=12, a=0.2),
        design_filter_polynomial(cutoff=0.45, degree=10),
    ]

    for coeffs in coeff_sets:
        assert is_bounded_on_interval(
            coeffs,
            lower=-1.0,
            upper=1.0,
            bound=1.0,
            num_points=4001,
            tol=1e-8,
        )


def test_expected_parity_for_inverse_sign_and_filter_designs():
    inverse_coeffs = design_inverse_polynomial(gamma=0.25, degree=11)
    sign_coeffs = design_sign_polynomial(gamma=0.2, degree=13)
    filter_coeffs = design_filter_polynomial(cutoff=0.45, degree=10)

    assert polynomial_parity(inverse_coeffs, tol=1e-8) == "odd"
    assert polynomial_parity(sign_coeffs, tol=1e-8) == "odd"
    assert polynomial_parity(filter_coeffs, tol=1e-8) == "even"


def test_inverse_and_sign_are_reasonable_away_from_zero():
    gamma = 0.25
    xs = np.array([-1.0, -0.75, -0.5, -gamma, gamma, 0.5, 0.75, 1.0])

    inverse_coeffs = design_inverse_polynomial(gamma=gamma, degree=15)
    sign_coeffs = design_sign_polynomial(gamma=gamma, degree=15)

    inverse_vals = np.polynomial.polynomial.polyval(xs, inverse_coeffs)
    sign_vals = np.polynomial.polynomial.polyval(xs, sign_coeffs)

    assert np.max(np.abs(inverse_vals - (gamma / xs))) < 0.2
    assert np.max(np.abs(sign_vals - np.sign(xs))) < 0.2


def test_projector_sqrt_and_power_have_basic_shape_properties():
    xs = np.linspace(-1.0, 1.0, 2001)

    projector_coeffs = design_projector_polynomial(gamma=0.2, degree=13)
    sqrt_coeffs = design_sqrt_polynomial(a=0.2, degree=12)
    power_coeffs = design_power_polynomial(alpha=0.5, degree=12, a=0.2)

    projector_vals = np.polynomial.polynomial.polyval(xs, projector_coeffs)
    sqrt_vals = np.polynomial.polynomial.polyval(xs, sqrt_coeffs)
    power_vals = np.polynomial.polynomial.polyval(xs, power_coeffs)

    assert projector_vals[0] < 0.25
    assert projector_vals[-1] > 0.75

    pos_mask = xs >= 0.2

    assert np.min(sqrt_vals[pos_mask]) >= -1e-6
    assert np.min(power_vals[pos_mask]) >= -1e-6

    assert np.max(np.abs(sqrt_vals[pos_mask] - np.sqrt(xs[pos_mask]))) < 0.1
    assert np.max(np.abs(power_vals[pos_mask] - xs[pos_mask] ** 0.5)) < 0.1
