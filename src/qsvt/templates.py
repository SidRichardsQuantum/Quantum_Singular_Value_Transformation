"""
qsvt.templates
--------------

Ready-to-use bounded polynomial templates for QSVT/QSP-style experiments.

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
These templates are intended as practical educational starting points rather
than minimax-optimal constructions. They are useful for prototyping and
intuition-building before moving to more specialised QSVT polynomial design.
"""

from __future__ import annotations

import numpy as np

from .polynomials import normalize_coefficients


def _validate_degree(degree: int) -> int:
    """
    Validate a polynomial degree.

    Parameters
    ----------
    degree
        Non-negative polynomial degree.

    Returns
    -------
    int
        Validated degree.

    Raises
    ------
    ValueError
        If degree is negative.
    """
    if degree < 0:
        raise ValueError("degree must be non-negative.")
    return int(degree)


def _validate_num_points(num_points: int) -> int:
    """
    Validate a sampling-grid size.

    Parameters
    ----------
    num_points
        Number of fitting/evaluation grid points.

    Returns
    -------
    int
        Validated number of points.

    Raises
    ------
    ValueError
        If num_points is too small.
    """
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")
    return int(num_points)


def _grid_max_abs(coeffs: np.ndarray, num_points: int = 4001) -> float:
    """
    Estimate max |P(x)| on [-1, 1] using a dense grid.

    Parameters
    ----------
    coeffs
        Polynomial coefficients in ascending monomial order.
    num_points
        Number of grid points used in the check.

    Returns
    -------
    float
        Maximum sampled absolute value.
    """
    xs = np.linspace(-1.0, 1.0, num_points)
    values = np.polynomial.polynomial.polyval(xs, coeffs)
    return float(np.max(np.abs(values)))


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
    degree = _validate_degree(degree)
    num_points = _validate_num_points(num_points)

    xs = np.linspace(-1.0, 1.0, num_points)
    ys = np.asarray(func(xs), dtype=float)

    cheb_coeffs = np.polynomial.chebyshev.chebfit(xs, ys, degree)

    if parity == "odd":
        cheb_coeffs[::2] = 0.0
    elif parity == "even":
        cheb_coeffs[1::2] = 0.0
    elif parity is not None:
        raise ValueError("parity must be None, 'even', or 'odd'.")

    poly = np.polynomial.Chebyshev(cheb_coeffs, domain=[-1.0, 1.0])
    coeffs = np.asarray(poly.convert(kind=np.polynomial.Polynomial).coef, dtype=float)

    max_abs = _grid_max_abs(coeffs, num_points=max(4001, num_points))
    if max_abs > 1.0:
        coeffs = coeffs / max_abs

    return normalize_coefficients(coeffs)


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
    degree = _validate_degree(degree)
    num_points = _validate_num_points(num_points)

    if mu <= 0.0:
        raise ValueError("mu must be positive.")

    def target(x: np.ndarray) -> np.ndarray:
        return (2.0 * mu * x) / (x**2 + mu**2)

    return _fit_template(target, degree, parity="odd", num_points=num_points)


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
    degree = _validate_degree(degree)
    num_points = _validate_num_points(num_points)

    if sharpness <= 0.0:
        raise ValueError("sharpness must be positive.")

    def target(x: np.ndarray) -> np.ndarray:
        return np.tanh(sharpness * x)

    return _fit_template(target, degree, parity="odd", num_points=num_points)


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
    degree = _validate_degree(degree)
    num_points = _validate_num_points(num_points)

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1].")
    if sharpness <= 0.0:
        raise ValueError("sharpness must be positive.")

    def target(x: np.ndarray) -> np.ndarray:
        return 0.5 * (1.0 + np.tanh(sharpness * (np.abs(x) - threshold)))

    return _fit_template(target, degree, parity="even", num_points=num_points)


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
    degree = _validate_degree(degree)
    num_points = _validate_num_points(num_points)

    def target(x: np.ndarray) -> np.ndarray:
        return np.sqrt(np.clip(0.5 * (x + 1.0), 0.0, 1.0))

    return _fit_template(target, degree, num_points=num_points)


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
    degree = _validate_degree(degree)
    num_points = _validate_num_points(num_points)

    def target(x: np.ndarray) -> np.ndarray:
        return np.exp(beta * x - abs(beta))

    return _fit_template(target, degree, num_points=num_points)


__all__ = [
    "inverse_like_polynomial",
    "sign_approximation_polynomial",
    "soft_threshold_filter_polynomial",
    "sqrt_approximation_polynomial",
    "exponential_approximation_polynomial",
]
