"""
qsvt.presets
------------

Named bounded polynomial presets for QSVT/QSP-style experiments.

This module provides small polynomial families that are commonly useful when
experimenting with singular-value transforms, filters, and matrix-function
surrogates.

Design goals
------------
- simple coefficient-form outputs
- ascending degree ordering
- bounded on [-1, 1]
- lightweight Chebyshev-based construction where appropriate
- readable implementations suitable for notebooks and small examples

Conventions
-----------
All returned coefficient arrays use ascending monomial order:

    coeffs = [c0, c1, c2, ...]
    P(x) = c0 + c1 x + c2 x^2 + ...

Notes
-----
These presets are intended as practical educational starting points rather
than minimax-optimal constructions. They are useful for prototyping and
intuition-building before moving to more specialised QSVT polynomial design.
"""

from __future__ import annotations

import numpy as np

from ._polyfit import fit_bounded_monomial, validate_degree, validate_num_points
from .approximation import approximation_quality_report


def _inverse_like_target(x: np.ndarray, mu: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (2.0 * mu * x) / (x**2 + mu**2)


def _sign_target(x: np.ndarray, sharpness: float) -> np.ndarray:
    return np.tanh(sharpness * np.asarray(x, dtype=float))


def _soft_threshold_target(
    x: np.ndarray,
    threshold: float,
    sharpness: float,
) -> np.ndarray:
    return 0.5 * (
        1.0
        + np.tanh(
            sharpness * (np.abs(np.asarray(x, dtype=float)) - threshold),
        )
    )


def _sqrt_target(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(0.5 * (np.asarray(x, dtype=float) + 1.0), 0.0, 1.0))


def _exponential_target(x: np.ndarray, beta: float) -> np.ndarray:
    return np.exp(beta * np.asarray(x, dtype=float) - abs(beta))


def _fit_template(
    func,
    degree: int,
    *,
    parity: str | None = None,
    num_points: int = 2001,
) -> np.ndarray:
    """
    Fit a bounded template polynomial on [-1, 1].

    The target function is sampled on [-1, 1], approximated in the Chebyshev
    basis, optionally projected to even/odd parity, then converted to standard
    monomial coefficients in ascending degree order.

    Parameters
    ----------
    func
        Target scalar function defined on [-1, 1].
    degree
        Polynomial degree.
    parity
        Optional parity constraint: "even", "odd", or None.
    num_points
        Number of grid points used in the fit.

    Returns
    -------
    numpy.ndarray
        Polynomial coefficients in ascending monomial order.

    Raises
    ------
    ValueError
        If parity is invalid.
    """
    return fit_bounded_monomial(
        func,
        degree=degree,
        parity=parity,
        num_points=num_points,
        bound_num_points=4001,
    )


def inverse_like_polynomial(
    degree: int,
    mu: float = 0.25,
    num_points: int = 2001,
) -> np.ndarray:
    """
    Build a bounded inverse-like odd polynomial on [-1, 1].

    The fitted target is the smooth rational function

        f(x) = 2 mu x / (x^2 + mu^2),

    which is odd and satisfies |f(x)| <= 1 on [-1, 1]. For |x| >> mu, this
    behaves like a scaled inverse, making it useful as a small regularised
    inverse-like QSVT template.

    Parameters
    ----------
    degree
        Polynomial degree.
    mu
        Positive regularisation scale. Smaller values produce a sharper
        inverse-like transition near zero.
    num_points
        Number of fitting grid points.

    Returns
    -------
    numpy.ndarray
        Polynomial coefficients in ascending degree order.

    Raises
    ------
    ValueError
        If mu is not positive.

    Examples
    --------
    >>> coeffs = inverse_like_polynomial(7, mu=0.3)
    >>> coeffs.ndim
    1
    >>> abs(coeffs[0]) < 1e-8
    True
    """
    degree = validate_degree(degree)
    num_points = validate_num_points(num_points)

    if mu <= 0.0:
        raise ValueError("mu must be positive.")

    return _fit_template(
        lambda x: _inverse_like_target(x, mu),
        degree,
        parity="odd",
        num_points=num_points,
    )


def sign_approximation_polynomial(
    degree: int,
    sharpness: float = 6.0,
    num_points: int = 2001,
) -> np.ndarray:
    """
    Build an odd bounded polynomial approximating sign(x).

    The target function is

        f(x) = tanh(sharpness * x),

    which is smooth, odd, and bounded by 1 in magnitude. Increasing
    `sharpness` makes the transition near zero steeper.

    Parameters
    ----------
    degree
        Polynomial degree.
    sharpness
        Positive steepness parameter for the smooth sign surrogate.
    num_points
        Number of fitting grid points.

    Returns
    -------
    numpy.ndarray
        Polynomial coefficients in ascending degree order.

    Raises
    ------
    ValueError
        If sharpness is not positive.

    Examples
    --------
    >>> coeffs = sign_approximation_polynomial(9, sharpness=8.0)
    >>> coeffs.ndim
    1
    >>> abs(coeffs[0]) < 1e-8
    True
    """
    degree = validate_degree(degree)
    num_points = validate_num_points(num_points)

    if sharpness <= 0.0:
        raise ValueError("sharpness must be positive.")

    return _fit_template(
        lambda x: _sign_target(x, sharpness),
        degree,
        parity="odd",
        num_points=num_points,
    )


def soft_threshold_filter_polynomial(
    degree: int,
    threshold: float = 0.5,
    sharpness: float = 12.0,
    num_points: int = 2001,
) -> np.ndarray:
    """
    Build an even bounded soft-threshold filter polynomial.

    The fitted target is

        f(x) = 0.5 * (1 + tanh(sharpness * (|x| - threshold))),

    which is near 0 for small |x| and near 1 for larger |x|. This is a useful
    singular-value filter template when experimenting with pass/reject
    behaviour.

    Parameters
    ----------
    degree
        Polynomial degree.
    threshold
        Threshold in [0, 1] controlling where the filter turns on.
    sharpness
        Positive transition steepness.
    num_points
        Number of fitting grid points.

    Returns
    -------
    numpy.ndarray
        Polynomial coefficients in ascending degree order.

    Raises
    ------
    ValueError
        If threshold is outside [0, 1] or sharpness is not positive.

    Examples
    --------
    >>> coeffs = soft_threshold_filter_polynomial(10, threshold=0.4)
    >>> coeffs.ndim
    1
    >>> coeffs[0] >= 0.0
    True
    """
    degree = validate_degree(degree)
    num_points = validate_num_points(num_points)

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1].")
    if sharpness <= 0.0:
        raise ValueError("sharpness must be positive.")

    return _fit_template(
        lambda x: _soft_threshold_target(x, threshold, sharpness),
        degree,
        parity="even",
        num_points=num_points,
    )


def sqrt_approximation_polynomial(
    degree: int,
    num_points: int = 2001,
) -> np.ndarray:
    """
    Build a bounded polynomial approximating a shifted square-root profile.

    The fitted target is

        f(x) = sqrt((x + 1) / 2),

    which maps [-1, 1] to [0, 1]. This provides a simple square-root-like
    template that stays bounded on the full canonical interval.

    Parameters
    ----------
    degree
        Polynomial degree.
    num_points
        Number of fitting grid points.

    Returns
    -------
    numpy.ndarray
        Polynomial coefficients in ascending degree order.

    Examples
    --------
    >>> coeffs = sqrt_approximation_polynomial(8)
    >>> coeffs.ndim
    1
    >>> coeffs[0] > 0.0
    True
    """
    degree = validate_degree(degree)
    num_points = validate_num_points(num_points)

    return _fit_template(_sqrt_target, degree, num_points=num_points)


def exponential_approximation_polynomial(
    degree: int,
    beta: float = 1.0,
    num_points: int = 2001,
) -> np.ndarray:
    """
    Build a bounded exponential-like polynomial on [-1, 1].

    The fitted target is

        f(x) = exp(beta * x - |beta|),

    which is strictly positive and bounded by 1 on [-1, 1]. This is a useful
    low-degree smooth weighting template for experiments involving spectral
    damping or amplification profiles.

    Parameters
    ----------
    degree
        Polynomial degree.
    beta
        Exponential slope parameter.
    num_points
        Number of fitting grid points.

    Returns
    -------
    numpy.ndarray
        Polynomial coefficients in ascending degree order.

    Examples
    --------
    >>> coeffs = exponential_approximation_polynomial(5, beta=1.5)
    >>> coeffs.ndim
    1
    >>> np.all(np.isfinite(coeffs))
    True
    """
    degree = validate_degree(degree)
    num_points = validate_num_points(num_points)

    return _fit_template(
        lambda x: _exponential_target(x, beta),
        degree,
        num_points=num_points,
    )


def _template_quality_report(
    target,
    coeffs: np.ndarray,
    *,
    domain: tuple[float, float] = (-1.0, 1.0),
    num_points: int = 2001,
    bounded_num_points: int = 4001,
) -> dict[str, object]:
    """
    Assemble a quality report for a template polynomial.
    """
    coeffs = np.asarray(coeffs, dtype=float)

    def approx(x: np.ndarray) -> np.ndarray:
        return np.polynomial.polynomial.polyval(x, coeffs)

    return approximation_quality_report(
        target,
        approx,
        domain=domain,
        num_points=num_points,
        bounded_domain=(-1.0, 1.0),
        bounded_num_points=bounded_num_points,
        coeffs=coeffs,
    )


def inverse_like_diagnostics(
    degree: int,
    mu: float = 0.25,
    num_points: int = 2001,
    bounded_num_points: int = 4001,
) -> dict[str, object]:
    """
    Build a report for the inverse-like template polynomial.
    """
    coeffs = inverse_like_polynomial(degree, mu=mu, num_points=num_points)
    report = _template_quality_report(
        lambda x: _inverse_like_target(x, mu),
        coeffs,
        num_points=num_points,
        bounded_num_points=bounded_num_points,
    )
    report.update(
        {
            "builder": "inverse_like_polynomial",
            "mu": float(mu),
            "degree": int(degree),
        }
    )
    return report


def sign_approximation_diagnostics(
    degree: int,
    sharpness: float = 6.0,
    num_points: int = 2001,
    bounded_num_points: int = 4001,
) -> dict[str, object]:
    """
    Build a report for the sign template polynomial.
    """
    coeffs = sign_approximation_polynomial(
        degree,
        sharpness=sharpness,
        num_points=num_points,
    )
    report = _template_quality_report(
        lambda x: _sign_target(x, sharpness),
        coeffs,
        num_points=num_points,
        bounded_num_points=bounded_num_points,
    )
    report.update(
        {
            "builder": "sign_approximation_polynomial",
            "sharpness": float(sharpness),
            "degree": int(degree),
        }
    )
    return report


def soft_threshold_filter_diagnostics(
    degree: int,
    threshold: float = 0.5,
    sharpness: float = 12.0,
    num_points: int = 2001,
    bounded_num_points: int = 4001,
) -> dict[str, object]:
    """
    Build a report for the soft-threshold template polynomial.
    """
    coeffs = soft_threshold_filter_polynomial(
        degree,
        threshold=threshold,
        sharpness=sharpness,
        num_points=num_points,
    )
    report = _template_quality_report(
        lambda x: _soft_threshold_target(x, threshold, sharpness),
        coeffs,
        num_points=num_points,
        bounded_num_points=bounded_num_points,
    )
    report.update(
        {
            "builder": "soft_threshold_filter_polynomial",
            "threshold": float(threshold),
            "sharpness": float(sharpness),
            "degree": int(degree),
        }
    )
    return report


def sqrt_approximation_diagnostics(
    degree: int,
    num_points: int = 2001,
    bounded_num_points: int = 4001,
) -> dict[str, object]:
    """
    Build a report for the shifted square-root template polynomial.
    """
    coeffs = sqrt_approximation_polynomial(degree, num_points=num_points)
    report = _template_quality_report(
        _sqrt_target,
        coeffs,
        num_points=num_points,
        bounded_num_points=bounded_num_points,
    )
    report.update({"builder": "sqrt_approximation_polynomial", "degree": int(degree)})
    return report


def exponential_approximation_diagnostics(
    degree: int,
    beta: float = 1.0,
    num_points: int = 2001,
    bounded_num_points: int = 4001,
) -> dict[str, object]:
    """
    Build a report for the exponential template polynomial.
    """
    coeffs = exponential_approximation_polynomial(
        degree,
        beta=beta,
        num_points=num_points,
    )
    report = _template_quality_report(
        lambda x: _exponential_target(x, beta),
        coeffs,
        num_points=num_points,
        bounded_num_points=bounded_num_points,
    )
    report.update(
        {
            "builder": "exponential_approximation_polynomial",
            "beta": float(beta),
            "degree": int(degree),
        }
    )
    return report


__all__ = [
    "inverse_like_diagnostics",
    "sign_approximation_diagnostics",
    "soft_threshold_filter_diagnostics",
    "sqrt_approximation_diagnostics",
    "exponential_approximation_diagnostics",
    "inverse_like_polynomial",
    "sign_approximation_polynomial",
    "soft_threshold_filter_polynomial",
    "sqrt_approximation_polynomial",
    "exponential_approximation_polynomial",
]
