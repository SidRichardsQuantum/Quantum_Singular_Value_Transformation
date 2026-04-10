"""
qsvt.polynomials
----------------

Polynomial utilities for QSVT/QSP-style demonstrations and small-scale
classical validation.

This module focuses on:

- Chebyshev polynomials of the first kind
- coefficient-based polynomial evaluation
- simple parity inspection for real polynomials

Conventions
-----------
Polynomial coefficients are ordered in ascending degree:

    coeffs = [c0, c1, c2, ...]
    P(x) = c0 + c1 x + c2 x^2 + ...

This matches the simple coefficient style used throughout the notebooks.

Notes
-----
These helpers are intentionally lightweight and notebook-friendly.
They are not intended to be a full symbolic polynomial framework.
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np

ArrayLike = float | np.ndarray


def chebyshev_t(n: int, x: ArrayLike) -> ArrayLike:
    """
    Evaluate the Chebyshev polynomial of the first kind T_n(x).

    The definition used is:

        T_n(x) = cos(n arccos(x))

    Parameters
    ----------
    n
        Non-negative polynomial degree.
    x
        Scalar or NumPy array of evaluation points. Values are expected to lie
        in [-1, 1] for the trigonometric definition used here.

    Returns
    -------
    float or numpy.ndarray
        T_n(x), with shape matching the input.

    Raises
    ------
    ValueError
        If n is negative.

    Examples
    --------
    >>> chebyshev_t(0, 0.3)
    1.0
    >>> chebyshev_t(1, 0.3)
    0.3
    >>> chebyshev_t(3, 0.5)
    -1.0
    """
    if n < 0:
        raise ValueError("n must be non-negative.")

    x_arr = np.asarray(x, dtype=float)
    values = np.cos(n * np.arccos(x_arr))

    if np.isscalar(x):
        return float(values)
    return values


def chebyshev_t3(x: ArrayLike) -> ArrayLike:
    """
    Evaluate the third Chebyshev polynomial T_3(x).

    This uses the explicit polynomial form:

        T_3(x) = 4x^3 - 3x

    Parameters
    ----------
    x
        Scalar or NumPy array of evaluation points.

    Returns
    -------
    float or numpy.ndarray
        T_3(x), with shape matching the input.

    Examples
    --------
    >>> chebyshev_t3(0.5)
    -1.0
    >>> chebyshev_t3(-0.5)
    1.0
    """
    x_arr = np.asarray(x, dtype=float)
    values = 4.0 * x_arr**3 - 3.0 * x_arr

    if np.isscalar(x):
        return float(values)
    return values


def eval_polynomial(coeffs: Iterable[float], x: ArrayLike) -> ArrayLike:
    """
    Evaluate a polynomial with ascending-order coefficients.

    The coefficient convention is:

        coeffs = [c0, c1, ..., cn]
        P(x) = c0 + c1 x + ... + cn x^n

    Parameters
    ----------
    coeffs
        Polynomial coefficients in ascending degree order.
    x
        Scalar or NumPy array of evaluation points.

    Returns
    -------
    float or numpy.ndarray
        Polynomial values at x.

    Examples
    --------
    >>> eval_polynomial([0, 0, 1], 0.5)
    0.25
    >>> eval_polynomial([0, 1], np.array([-1.0, 0.0, 1.0]))
    array([-1.,  0.,  1.])
    """
    coeffs_arr = np.asarray(list(coeffs), dtype=float)
    x_arr = np.asarray(x, dtype=float)

    values = np.zeros_like(x_arr, dtype=float)
    for degree, coeff in enumerate(coeffs_arr):
        if coeff != 0.0:
            values = values + coeff * x_arr**degree

    if np.isscalar(x):
        return float(values)
    return values


def polynomial_degree(coeffs: Iterable[float], tol: float = 1e-12) -> int:
    """
    Return the effective degree of a polynomial.

    Trailing coefficients with absolute value <= tol are ignored.

    Parameters
    ----------
    coeffs
        Polynomial coefficients in ascending degree order.
    tol
        Numerical tolerance used to identify trailing zeros.

    Returns
    -------
    int
        Effective polynomial degree. Returns 0 for the zero polynomial.

    Examples
    --------
    >>> polynomial_degree([1, 0, 0])
    0
    >>> polynomial_degree([0, 1, 0, 0])
    1
    >>> polynomial_degree([0, 0, 0])
    0
    """
    coeffs_arr = np.asarray(list(coeffs), dtype=float)

    nz = np.where(np.abs(coeffs_arr) > tol)[0]
    if len(nz) == 0:
        return 0
    return int(nz[-1])


def polynomial_parity(
    coeffs: Iterable[float], tol: float = 1e-12
) -> Literal["even", "odd", "mixed", "zero"]:
    """
    Determine the parity of a real polynomial from its coefficients.

    A polynomial is:
    - even if only even-degree coefficients are nonzero,
    - odd if only odd-degree coefficients are nonzero,
    - mixed otherwise,
    - zero if all coefficients vanish within tolerance.

    Parameters
    ----------
    coeffs
        Polynomial coefficients in ascending degree order.
    tol
        Numerical tolerance for deciding whether a coefficient is zero.

    Returns
    -------
    {"even", "odd", "mixed", "zero"}
        Detected parity class.

    Examples
    --------
    >>> polynomial_parity([0, 0, 1])
    'even'
    >>> polynomial_parity([0, 1])
    'odd'
    >>> polynomial_parity([1, 1])
    'mixed'
    """
    coeffs_arr = np.asarray(list(coeffs), dtype=float)

    nonzero_degrees = [i for i, c in enumerate(coeffs_arr) if abs(c) > tol]
    if not nonzero_degrees:
        return "zero"

    has_even = any(i % 2 == 0 for i in nonzero_degrees)
    has_odd = any(i % 2 == 1 for i in nonzero_degrees)

    if has_even and has_odd:
        return "mixed"
    if has_even:
        return "even"
    return "odd"


def is_bounded_on_interval(
    coeffs: Iterable[float],
    lower: float = -1.0,
    upper: float = 1.0,
    bound: float = 1.0,
    num_points: int = 2001,
    tol: float = 1e-9,
) -> bool:
    """
    Check whether a polynomial is numerically bounded on an interval.

    This is a grid-based numerical check, useful for lightweight validation
    of notebook/demo polynomials used in QSVT examples.

    Parameters
    ----------
    coeffs
        Polynomial coefficients in ascending degree order.
    lower
        Left endpoint of the interval.
    upper
        Right endpoint of the interval.
    bound
        Upper bound for |P(x)|.
    num_points
        Number of grid points used for the numerical check.
    tol
        Numerical slack added to the bound.

    Returns
    -------
    bool
        True if max |P(x)| <= bound + tol on the sampled grid.

    Examples
    --------
    >>> is_bounded_on_interval([0, 1], -1.0, 1.0, 1.0)
    True
    >>> is_bounded_on_interval([0, 0, 2], -1.0, 1.0, 1.0)
    False
    """
    if upper < lower:
        raise ValueError("upper must be greater than or equal to lower.")
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")

    xs = np.linspace(lower, upper, num_points)
    values = eval_polynomial(coeffs, xs)
    return bool(np.max(np.abs(values)) <= bound + tol)


def normalize_coefficients(coeffs: Iterable[float], tol: float = 1e-12) -> np.ndarray:
    """
    Return coefficients with tiny values zeroed out and trailing zeros removed.

    Parameters
    ----------
    coeffs
        Polynomial coefficients in ascending degree order.
    tol
        Values with absolute magnitude <= tol are set to zero.

    Returns
    -------
    numpy.ndarray
        Cleaned coefficient array. The zero polynomial is returned as [0.0].

    Examples
    --------
    >>> normalize_coefficients([0, 1e-14, 1.0, 0.0])
    array([0., 0., 1.])
    """
    arr = np.asarray(list(coeffs), dtype=float).copy()
    arr[np.abs(arr) <= tol] = 0.0

    if arr.size == 0:
        return np.array([0.0], dtype=float)

    nz = np.where(arr != 0.0)[0]
    if len(nz) == 0:
        return np.array([0.0], dtype=float)

    return arr[: nz[-1] + 1]


__all__ = [
    "chebyshev_t",
    "chebyshev_t3",
    "eval_polynomial",
    "polynomial_degree",
    "polynomial_parity",
    "is_bounded_on_interval",
    "normalize_coefficients",
]
