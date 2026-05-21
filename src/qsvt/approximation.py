"""
qsvt.approximation
------------------

Approximation helpers for QSVT/QSP-style workflows.

This module provides lightweight utilities for constructing and evaluating
Chebyshev polynomial approximations to scalar functions on bounded intervals.
The main intended use-cases are:

- approximating smooth bounded functions on subintervals of [-1, 1],
- building inverse-like, root-like, or filter-like polynomial surrogates,
- numerically assessing approximation quality before passing a polynomial
  into QSVT/QSP routines.

Notes
-----
NumPy's Chebyshev routines work most naturally on the canonical interval
[-1, 1]. To support a general interval [a, b], this module uses an affine
map between x in [a, b] and t in [-1, 1].

Unless otherwise stated, coefficient arrays returned here are Chebyshev-basis
coefficients, not monomial coefficients.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeAlias

import numpy as np

RealArray: TypeAlias = float | np.ndarray


def _validate_domain(domain: tuple[float, float]) -> tuple[float, float]:
    """
    Validate and normalize a closed interval domain.

    Parameters
    ----------
    domain
        Tuple (a, b) with a < b.

    Returns
    -------
    tuple[float, float]
        Validated domain.

    Raises
    ------
    ValueError
        If the interval is invalid.
    """
    if len(domain) != 2:
        raise ValueError("domain must be a 2-tuple (lower, upper).")

    lower, upper = float(domain[0]), float(domain[1])

    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("domain endpoints must be finite.")
    if upper <= lower:
        raise ValueError("domain must satisfy lower < upper.")

    return lower, upper


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


def _validate_num_points(num_points: int, name: str = "num_points") -> int:
    """
    Validate a positive grid/sample count.

    Parameters
    ----------
    num_points
        Number of sampling points.
    name
        Parameter name for error messages.

    Returns
    -------
    int
        Validated positive integer.

    Raises
    ------
    ValueError
        If num_points is too small.
    """
    if num_points < 2:
        raise ValueError(f"{name} must be at least 2.")
    return int(num_points)


def scale_to_chebyshev_domain(x: RealArray, domain: tuple[float, float]) -> RealArray:
    """
    Map x from a general interval [a, b] to t in [-1, 1].

    The affine transformation is:

        t = 2 * (x - a) / (b - a) - 1

    Parameters
    ----------
    x
        Scalar or NumPy array on the physical domain [a, b].
    domain
        Approximation interval (a, b).

    Returns
    -------
    float or numpy.ndarray
        Mapped coordinate(s) in [-1, 1].
    """
    a, b = _validate_domain(domain)
    x_arr = np.asarray(x, dtype=float)
    t = 2.0 * (x_arr - a) / (b - a) - 1.0

    if np.isscalar(x):
        return float(t)
    return t


def scale_from_chebyshev_domain(t: RealArray, domain: tuple[float, float]) -> RealArray:
    """
    Map t from [-1, 1] back to x in a general interval [a, b].

    The affine transformation is:

        x = a + (t + 1) * (b - a) / 2

    Parameters
    ----------
    t
        Scalar or NumPy array on [-1, 1].
    domain
        Physical interval (a, b).

    Returns
    -------
    float or numpy.ndarray
        Mapped coordinate(s) in [a, b].
    """
    a, b = _validate_domain(domain)
    t_arr = np.asarray(t, dtype=float)
    x = a + 0.5 * (t_arr + 1.0) * (b - a)

    if np.isscalar(t):
        return float(x)
    return x


def chebyshev_fit_function(
    func: Callable[[np.ndarray], np.ndarray | float],
    degree: int,
    domain: tuple[float, float] = (-1.0, 1.0),
    num_points: int = 500,
) -> np.ndarray:
    """
    Fit a scalar function with a Chebyshev polynomial on a closed interval.

    The returned coefficients are in the Chebyshev basis on the canonical
    variable t in [-1, 1], where t is obtained from the physical variable x
    through affine scaling of the given domain.

    Parameters
    ----------
    func
        Target scalar function. It should accept a NumPy array and return
        values of matching shape, or be compatible with NumPy broadcasting.
    degree
        Chebyshev degree of the approximation.
    domain
        Approximation interval (lower, upper).
    num_points
        Number of sample points used in the least-squares fit.

    Returns
    -------
    numpy.ndarray
        Chebyshev-basis coefficients c such that:

            P(t) = c_0 T_0(t) + c_1 T_1(t) + ... + c_n T_n(t)

        approximates func(x), with x corresponding to t via the affine map.

    Notes
    -----
    This uses `numpy.polynomial.chebyshev.chebfit` on a dense sampled grid.
    For educational and lightweight package use, this is usually sufficient.

    Examples
    --------
    >>> coeffs = chebyshev_fit_function(np.sqrt, degree=6, domain=(0.2, 1.0))
    >>> coeffs.ndim
    1
    """
    degree = _validate_degree(degree)
    num_points = _validate_num_points(num_points)
    domain = _validate_domain(domain)

    xs = np.linspace(domain[0], domain[1], num_points)
    ts = np.asarray(scale_to_chebyshev_domain(xs, domain), dtype=float)

    ys = np.asarray(func(xs), dtype=float)
    coeffs = np.polynomial.chebyshev.chebfit(ts, ys, degree)

    return np.asarray(coeffs, dtype=float)


def chebyshev_eval(
    coeffs: Iterable[float],
    x: RealArray,
    domain: tuple[float, float] = (-1.0, 1.0),
) -> RealArray:
    """
    Evaluate a Chebyshev-series approximation on a physical interval.

    Parameters
    ----------
    coeffs
        Chebyshev-basis coefficients on the scaled variable t in [-1, 1].
    x
        Scalar or NumPy array of evaluation points in the physical domain.
    domain
        Physical interval (lower, upper) used when fitting the approximation.

    Returns
    -------
    float or numpy.ndarray
        Approximation values at x.

    Examples
    --------
    >>> coeffs = chebyshev_fit_function(lambda z: z**2, degree=2)
    >>> chebyshev_eval(coeffs, 0.5)
    0.25
    """
    domain = _validate_domain(domain)
    coeffs_arr = np.asarray(list(coeffs), dtype=float)

    t = scale_to_chebyshev_domain(x, domain)
    values = np.polynomial.chebyshev.chebval(t, coeffs_arr)

    if np.isscalar(x):
        return float(values)
    return values


def chebyshev_approximant(
    coeffs: Iterable[float],
    domain: tuple[float, float] = (-1.0, 1.0),
) -> Callable[[RealArray], RealArray]:
    """
    Build a callable approximant from Chebyshev coefficients.

    Parameters
    ----------
    coeffs
        Chebyshev-basis coefficients on the scaled variable t in [-1, 1].
    domain
        Physical interval (lower, upper).

    Returns
    -------
    Callable
        Function P(x) evaluating the Chebyshev approximation on the domain.

    Examples
    --------
    >>> coeffs = chebyshev_fit_function(np.sqrt, degree=4, domain=(0.2, 1.0))
    >>> P = chebyshev_approximant(coeffs, domain=(0.2, 1.0))
    >>> isinstance(P(0.5), float)
    True
    """
    coeffs_arr = np.asarray(list(coeffs), dtype=float)
    domain = _validate_domain(domain)

    def approximant(x: RealArray) -> RealArray:
        return chebyshev_eval(coeffs_arr, x, domain=domain)

    return approximant


def max_error(
    func: Callable[[np.ndarray], np.ndarray | float],
    approx: Callable[[np.ndarray], np.ndarray | float],
    domain: tuple[float, float] = (-1.0, 1.0),
    num_points: int = 1000,
) -> float:
    """
    Compute the maximum absolute approximation error on a sampled grid.

    Parameters
    ----------
    func
        Target function.
    approx
        Approximation function.
    domain
        Interval on which to measure the error.
    num_points
        Number of grid points used for the numerical estimate.

    Returns
    -------
    float
        Maximum sampled absolute error.

    Examples
    --------
    >>> coeffs = chebyshev_fit_function(lambda z: z**2, degree=2)
    >>> P = chebyshev_approximant(coeffs)
    >>> max_error(lambda z: z**2, P) < 1e-10
    True
    """
    domain = _validate_domain(domain)
    num_points = _validate_num_points(num_points)

    xs = np.linspace(domain[0], domain[1], num_points)
    true_vals = np.asarray(func(xs), dtype=float)
    approx_vals = np.asarray(approx(xs), dtype=float)

    return float(np.max(np.abs(true_vals - approx_vals)))


def rms_error(
    func: Callable[[np.ndarray], np.ndarray | float],
    approx: Callable[[np.ndarray], np.ndarray | float],
    domain: tuple[float, float] = (-1.0, 1.0),
    num_points: int = 1000,
) -> float:
    """
    Compute the root-mean-square approximation error on a sampled grid.

    Parameters
    ----------
    func
        Target function.
    approx
        Approximation function.
    domain
        Interval on which to measure the error.
    num_points
        Number of grid points used for the numerical estimate.

    Returns
    -------
    float
        RMS sampled absolute error.
    """
    domain = _validate_domain(domain)
    num_points = _validate_num_points(num_points)

    xs = np.linspace(domain[0], domain[1], num_points)
    true_vals = np.asarray(func(xs), dtype=float)
    approx_vals = np.asarray(approx(xs), dtype=float)

    err = true_vals - approx_vals
    return float(np.sqrt(np.mean(err**2)))


def fit_and_build_approximant(
    func: Callable[[np.ndarray], np.ndarray | float],
    degree: int,
    domain: tuple[float, float] = (-1.0, 1.0),
    num_points: int = 500,
) -> tuple[np.ndarray, Callable[[RealArray], RealArray]]:
    """
    Fit a Chebyshev approximation and return both coefficients and callable.

    Parameters
    ----------
    func
        Target function.
    degree
        Chebyshev approximation degree.
    domain
        Approximation interval.
    num_points
        Number of fit samples.

    Returns
    -------
    tuple[numpy.ndarray, Callable]
        `(coeffs, approximant)` pair.

    Examples
    --------
    >>> coeffs, P = fit_and_build_approximant(np.sqrt, 6, domain=(0.2, 1.0))
    >>> len(coeffs) == 7
    True
    """
    coeffs = chebyshev_fit_function(
        func=func,
        degree=degree,
        domain=domain,
        num_points=num_points,
    )
    return coeffs, chebyshev_approximant(coeffs, domain=domain)


def sample_approximation(
    func: Callable[[np.ndarray], np.ndarray | float],
    approx: Callable[[np.ndarray], np.ndarray | float],
    domain: tuple[float, float] = (-1.0, 1.0),
    num_points: int = 400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample a target function and approximation on a shared grid.

    Parameters
    ----------
    func
        Target function.
    approx
        Approximation function.
    domain
        Sampling interval.
    num_points
        Number of sample points.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        `(xs, func_values, approx_values)` on the sampled grid.
    """
    domain = _validate_domain(domain)
    num_points = _validate_num_points(num_points)

    xs = np.linspace(domain[0], domain[1], num_points)
    ys_true = np.asarray(func(xs), dtype=float)
    ys_approx = np.asarray(approx(xs), dtype=float)

    return xs, ys_true, ys_approx


def approximation_quality_report(
    func: Callable[[np.ndarray], np.ndarray | float],
    approx: Callable[[np.ndarray], np.ndarray | float],
    domain: tuple[float, float] = (-1.0, 1.0),
    num_points: int = 1000,
    bound: float = 1.0,
    *,
    bounded_domain: tuple[float, float] | None = None,
    bounded_num_points: int | None = None,
    coeffs: Iterable[float] | None = None,
) -> dict[str, object]:
    """
    Build a compact approximation-quality report for a sampled polynomial fit.

    The report includes sampled target/polynomial values, fit error statistics,
    and a boundedness check on a second interval. By default the fit and
    boundedness domains are the same, but callers can request a broader
    boundedness interval while measuring fit error on a smaller design window.

    Parameters
    ----------
    func
        Target function.
    approx
        Approximation function.
    domain
        Interval on which to measure approximation error.
    num_points
        Number of sample points used for the error estimate.
    bound
        Absolute-value bound used when computing the boundedness margin.
    bounded_domain
        Optional interval on which to assess boundedness. Defaults to `domain`.
    bounded_num_points
        Optional grid size for the boundedness check. Defaults to `num_points`.
    coeffs
        Optional coefficient array to include in the returned report.

    Returns
    -------
    dict[str, object]
        A dictionary containing sampled values and summary metrics.
    """
    domain = _validate_domain(domain)
    num_points = _validate_num_points(num_points)

    if bounded_domain is None:
        bounded_domain = domain
    bounded_domain = _validate_domain(bounded_domain)

    if bounded_num_points is None:
        bounded_num_points = num_points
    bounded_num_points = _validate_num_points(
        bounded_num_points,
        name="bounded_num_points",
    )

    xs, ys_true, ys_approx = sample_approximation(
        func,
        approx,
        domain=domain,
        num_points=num_points,
    )
    errors = ys_true - ys_approx
    abs_errors = np.abs(errors)

    bounded_xs = np.linspace(bounded_domain[0], bounded_domain[1], bounded_num_points)
    bounded_values = np.asarray(approx(bounded_xs), dtype=float)
    max_abs_value = float(np.max(np.abs(bounded_values)))

    report: dict[str, object] = {
        "domain": domain,
        "fit_domain": domain,
        "num_points": num_points,
        "fit_num_points": num_points,
        "bounded_domain": bounded_domain,
        "bounded_num_points": bounded_num_points,
        "bound": float(bound),
        "xs": xs,
        "target_values": ys_true,
        "polynomial_values": ys_approx,
        "errors": errors,
        "max_error": float(np.max(abs_errors)),
        "rms_error": float(np.sqrt(np.mean(errors**2))),
        "bounded_xs": bounded_xs,
        "bounded_polynomial_values": bounded_values,
        "max_abs_value": max_abs_value,
        "bounded_margin": float(bound - max_abs_value),
        "is_bounded": bool(max_abs_value <= float(bound) + 1e-12),
    }

    if coeffs is not None:
        report["coeffs"] = np.asarray(list(coeffs), dtype=float)

    return report


__all__ = [
    "scale_to_chebyshev_domain",
    "scale_from_chebyshev_domain",
    "chebyshev_fit_function",
    "chebyshev_eval",
    "chebyshev_approximant",
    "max_error",
    "rms_error",
    "fit_and_build_approximant",
    "sample_approximation",
    "approximation_quality_report",
]
