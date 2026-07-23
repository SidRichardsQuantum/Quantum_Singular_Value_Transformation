# test_presets.py

import numpy as np

from qsvt.polynomials import is_bounded_on_interval, polynomial_parity
from qsvt.presets import (
    exponential_approximation_diagnostics,
    exponential_approximation_polynomial,
    inverse_like_diagnostics,
    inverse_like_polynomial,
    sign_approximation_diagnostics,
    sign_approximation_polynomial,
    soft_threshold_filter_diagnostics,
    soft_threshold_filter_polynomial,
    sqrt_approximation_diagnostics,
    sqrt_approximation_polynomial,
)


def test_preset_functions_return_numpy_arrays():
    funcs = [
        lambda: inverse_like_polynomial(7),
        lambda: sign_approximation_polynomial(9),
        lambda: soft_threshold_filter_polynomial(10),
        lambda: sqrt_approximation_polynomial(8),
        lambda: exponential_approximation_polynomial(5),
    ]

    for build in funcs:
        coeffs = build()
        assert isinstance(coeffs, np.ndarray)
        assert coeffs.ndim == 1
        assert coeffs.size >= 1
        assert np.all(np.isfinite(coeffs))


def test_preset_polynomials_are_bounded_on_minus_one_one():
    coeff_sets = [
        inverse_like_polynomial(7, mu=0.3),
        sign_approximation_polynomial(9, sharpness=8.0),
        soft_threshold_filter_polynomial(10, threshold=0.45, sharpness=10.0),
        sqrt_approximation_polynomial(8),
        exponential_approximation_polynomial(6, beta=1.5),
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


def test_parity_smoke_for_odd_and_even_presets():
    odd_inverse = inverse_like_polynomial(7)
    odd_sign = sign_approximation_polynomial(9)
    even_filter = soft_threshold_filter_polynomial(10)

    assert polynomial_parity(odd_inverse, tol=1e-8) == "odd"
    assert polynomial_parity(odd_sign, tol=1e-8) == "odd"
    assert polynomial_parity(even_filter, tol=1e-8) == "even"


def test_filter_and_sqrt_presets_have_expected_basic_shape():
    filter_coeffs = soft_threshold_filter_polynomial(12, threshold=0.5)
    sqrt_coeffs = sqrt_approximation_polynomial(10)

    xs = np.linspace(-1.0, 1.0, 2001)

    filter_vals = np.polynomial.polynomial.polyval(xs, filter_coeffs)
    sqrt_vals = np.polynomial.polynomial.polyval(xs, sqrt_coeffs)

    assert filter_vals[0] > filter_vals[len(xs) // 2]
    assert filter_vals[-1] > filter_vals[len(xs) // 2]

    assert sqrt_vals[0] >= -1e-6
    assert sqrt_vals[-1] <= 1.0 + 1e-6


def test_preset_diagnostics_report_fit_and_boundedness():
    report = inverse_like_diagnostics(7, mu=0.3)
    coeffs = inverse_like_polynomial(7, mu=0.3)

    assert report["builder"] == "inverse_like_polynomial"
    assert np.allclose(report["coeffs"], coeffs)
    assert report["max_error"] < 0.2
    assert report["bounded_margin"] >= -1e-8
    assert (
        report["xs"].shape
        == report["target_values"].shape
        == report["polynomial_values"].shape
    )
    assert report["errors"].shape == report["xs"].shape


def test_preset_diagnostics_cover_sign_filter_sqrt_and_exponential():
    sign_report = sign_approximation_diagnostics(9, sharpness=8.0)
    filter_report = soft_threshold_filter_diagnostics(
        10,
        threshold=0.45,
        sharpness=10.0,
    )
    sqrt_report = sqrt_approximation_diagnostics(8)
    exp_report = exponential_approximation_diagnostics(6, beta=1.5)

    assert sign_report["builder"] == "sign_approximation_polynomial"
    assert filter_report["builder"] == "soft_threshold_filter_polynomial"
    assert sqrt_report["builder"] == "sqrt_approximation_polynomial"
    assert exp_report["builder"] == "exponential_approximation_polynomial"
    assert sign_report["max_error"] < 0.35
    assert filter_report["max_error"] < 0.35
    assert sqrt_report["bounded_margin"] >= -1e-8
    assert exp_report["bounded_margin"] >= -1e-8
