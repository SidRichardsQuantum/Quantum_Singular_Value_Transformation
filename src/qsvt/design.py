"""
qsvt.design
-----------

High-level bounded polynomial design helpers for QSVT/QSP workflows.

This module provides small, explicit utilities for constructing practical
polynomial surrogates for common QSVT-style tasks:

- inverse-like transforms away from zero,
- sign approximations,
- projector-style polynomials derived from sign surrogates,
- square-root approximations on a positive interval,
- positive power functions on a positive interval,
- smooth threshold/filter polynomials.

Design goals
------------
- return coefficients in ascending monomial order,
- stay bounded on [-1, 1] for QSVT compatibility,
- use only NumPy,
- reuse the lightweight approximation helpers already present in the package,
- keep the implementations readable and notebook-friendly.

Notes
-----
These routines are intended as practical educational builders rather than
minimax-optimal polynomial synthesis methods.
"""

from __future__ import annotations

import numpy as np

from .approximation import approximation_quality_report, chebyshev_fit_function
from .polynomials import normalize_coefficients

_DEF_NUM_POINTS = 2001
_DEF_BOUND_GRID = 4001


def _design_inverse_target(x: np.ndarray, gamma: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.sign(x)
    mask = np.abs(x) >= gamma
    out[mask] = gamma / x[mask]
    out[x == 0.0] = 0.0
    return out


def _design_sign_target(x: np.ndarray, gamma: float) -> np.ndarray:
    sharpness = _tanh_sharpness_from_margin(gamma, target_value=0.98)
    return np.tanh(sharpness * np.asarray(x, dtype=float))


def _design_projector_target(x: np.ndarray, gamma: float) -> np.ndarray:
    return 0.5 * (1.0 + _design_sign_target(x, gamma))


def _design_sqrt_target(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    mask = x >= 0.0
    out[mask] = np.sqrt(np.clip(x[mask], 0.0, 1.0))
    return out


def _design_power_target(x: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.clip(x, 0.0, 1.0) ** alpha


def _design_filter_target(x: np.ndarray, cutoff: float, sharpness: float) -> np.ndarray:
    return 0.5 * (
        1.0
        + np.tanh(
            sharpness * (np.abs(np.asarray(x, dtype=float)) - cutoff),
        )
    )


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
    Validate a fitting-grid size.

    Parameters
    ----------
    num_points
        Number of fitting/evaluation points.

    Returns
    -------
    int
        Validated point count.

    Raises
    ------
    ValueError
        If num_points is too small.
    """
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")
    return int(num_points)


def _validate_unit_interval_parameter(value: float, name: str) -> float:
    """
    Validate a parameter constrained to the open unit interval.

    Parameters
    ----------
    value
        Parameter value.
    name
        Parameter name for error messages.

    Returns
    -------
    float
        Validated value.

    Raises
    ------
    ValueError
        If value is not strictly between 0 and 1.
    """
    value = float(value)
    if not (0.0 < value < 1.0):
        raise ValueError(f"{name} must satisfy 0 < {name} < 1.")
    return value


def _tanh_sharpness_from_margin(gamma: float, target_value: float = 0.98) -> float:
    """
    Choose a tanh sharpness so that tanh(sharpness * gamma) ~= target_value.

    Parameters
    ----------
    gamma
        Positive transition-width parameter.
    target_value
        Desired magnitude at x = gamma.

    Returns
    -------
    float
        Sharpness parameter for tanh.
    """
    gamma = _validate_unit_interval_parameter(gamma, "gamma")
    target_value = float(target_value)

    if not (0.0 < target_value < 1.0):
        raise ValueError("target_value must lie strictly between 0 and 1.")

    return float(np.arctanh(target_value) / gamma)


def _chebyshev_to_monomial(cheb_coeffs: np.ndarray) -> np.ndarray:
    """
    Convert Chebyshev-basis coefficients on [-1, 1] to monomial coefficients.

    Parameters
    ----------
    cheb_coeffs
        Chebyshev-basis coefficients.

    Returns
    -------
    numpy.ndarray
        Monomial coefficients in ascending degree order.
    """
    poly = np.polynomial.Chebyshev(cheb_coeffs, domain=[-1.0, 1.0])
    coeffs = np.asarray(poly.convert(kind=np.polynomial.Polynomial).coef, dtype=float)
    return normalize_coefficients(coeffs)


def _enforce_boundedness(
    coeffs: np.ndarray,
    *,
    num_points: int = _DEF_BOUND_GRID,
) -> np.ndarray:
    """
    Rescale a polynomial if necessary so that it is numerically bounded by 1
    on [-1, 1].

    Parameters
    ----------
    coeffs
        Polynomial coefficients in ascending monomial order.
    num_points
        Number of grid points used in the boundedness check.

    Returns
    -------
    numpy.ndarray
        Possibly rescaled coefficient array.
    """
    xs = np.linspace(-1.0, 1.0, num_points)
    values = np.polynomial.polynomial.polyval(xs, coeffs)
    max_abs = float(np.max(np.abs(values)))

    if max_abs > 1.0:
        coeffs = coeffs / max_abs

    return normalize_coefficients(coeffs)


def _fit_on_canonical_interval(
    func,
    *,
    degree: int,
    parity: str | None = None,
    num_points: int = _DEF_NUM_POINTS,
) -> np.ndarray:
    """
    Fit a target on [-1, 1], optionally enforce parity, convert to monomials,
    and numerically enforce boundedness.

    Parameters
    ----------
    func
        Target scalar function defined on [-1, 1].
    degree
        Polynomial degree.
    parity
        Optional parity constraint: "even", "odd", or None.
    num_points
        Number of fitting points.

    Returns
    -------
    numpy.ndarray
        Monomial coefficients in ascending degree order.
    """
    degree = _validate_degree(degree)
    num_points = _validate_num_points(num_points)

    cheb_coeffs = chebyshev_fit_function(
        func,
        degree=degree,
        domain=(-1.0, 1.0),
        num_points=num_points,
    )

    if parity == "odd":
        cheb_coeffs[::2] = 0.0
    elif parity == "even":
        cheb_coeffs[1::2] = 0.0
    elif parity is not None:
        raise ValueError("parity must be None, 'even', or 'odd'.")

    coeffs = _chebyshev_to_monomial(cheb_coeffs)
    return _enforce_boundedness(coeffs, num_points=max(_DEF_BOUND_GRID, num_points))


def _fit_on_interval(
    func,
    domain: tuple[float, float],
    *,
    degree: int,
    num_points: int = _DEF_NUM_POINTS,
) -> np.ndarray:
    """
    Fit a target on a general interval, convert to monomials, and bound it.
    """
    degree = _validate_degree(degree)
    num_points = _validate_num_points(num_points)

    cheb_coeffs = chebyshev_fit_function(
        func,
        degree=degree,
        domain=domain,
        num_points=num_points,
    )
    poly = np.polynomial.Chebyshev(cheb_coeffs, domain=domain)
    coeffs = np.asarray(poly.convert(kind=np.polynomial.Polynomial).coef, dtype=float)
    coeffs = normalize_coefficients(coeffs)
    return _enforce_boundedness(coeffs, num_points=max(_DEF_BOUND_GRID, num_points))


def design_inverse_polynomial(
    gamma: float,
    degree: int,
    num_points: int = _DEF_NUM_POINTS,
) -> np.ndarray:
    """
    Construct a bounded odd inverse-like polynomial on
    [-1, -gamma] ∪ [gamma, 1].

    The fitted target is the clipped normalized inverse

        f(x) = gamma / x     for |x| >= gamma,
        f(x) = sign(x)       for |x| < gamma,

    which is odd and bounded by 1 in magnitude on [-1, 1]. Away from the
    origin it matches the normalized inverse gamma / x. To recover an
    approximation to 1/x itself, divide the polynomial output by gamma.

    Parameters
    ----------
    gamma
        Positive spectral-gap parameter with 0 < gamma < 1.
    degree
        Polynomial degree.
    num_points
        Number of fitting points.

    Returns
    -------
    numpy.ndarray
        Polynomial coefficients in ascending monomial degree order.

    Examples
    --------
    >>> coeffs = design_inverse_polynomial(gamma=0.25, degree=11)
    >>> coeffs.ndim
    1
    """
    gamma = _validate_unit_interval_parameter(gamma, "gamma")
    return _fit_on_canonical_interval(
        lambda x: _design_inverse_target(x, gamma),
        degree=degree,
        parity="odd",
        num_points=num_points,
    )


def design_sign_polynomial(
    gamma: float,
    degree: int,
    num_points: int = _DEF_NUM_POINTS,
) -> np.ndarray:
    """
    Construct a bounded odd polynomial approximating sign(x) away from zero.

    A smooth bounded tanh surrogate is fit on [-1, 1], with sharpness chosen
    so that the target is already close to ±1 at |x| = gamma.

    Parameters
    ----------
    gamma
        Positive transition-width parameter with 0 < gamma < 1.
    degree
        Polynomial degree.
    num_points
        Number of fitting points.

    Returns
    -------
    numpy.ndarray
        Polynomial coefficients in ascending monomial degree order.

    Examples
    --------
    >>> coeffs = design_sign_polynomial(gamma=0.2, degree=13)
    >>> coeffs.ndim
    1
    """
    gamma = _validate_unit_interval_parameter(gamma, "gamma")

    return _fit_on_canonical_interval(
        lambda x: _design_sign_target(x, gamma),
        degree=degree,
        parity="odd",
        num_points=num_points,
    )


def design_projector_polynomial(
    gamma: float,
    degree: int,
    num_points: int = _DEF_NUM_POINTS,
) -> np.ndarray:
    """
    Construct a bounded polynomial approximating

        (1 + sign(x)) / 2.

    This is useful for projector-style spectral constructions such as
    (I + sign(A)) / 2.

    Parameters
    ----------
    gamma
        Positive transition-width parameter with 0 < gamma < 1.
    degree
        Polynomial degree.
    num_points
        Number of fitting points.

    Returns
    -------
    numpy.ndarray
        Polynomial coefficients in ascending monomial degree order.

    Examples
    --------
    >>> coeffs = design_projector_polynomial(gamma=0.2, degree=13)
    >>> coeffs[0] > 0.0
    True
    """
    sign_coeffs = design_sign_polynomial(
        gamma=gamma,
        degree=degree,
        num_points=num_points,
    )
    coeffs = 0.5 * sign_coeffs
    coeffs[0] += 0.5
    return _enforce_boundedness(coeffs)


def design_sqrt_polynomial(
    a: float,
    degree: int,
    num_points: int = _DEF_NUM_POINTS,
) -> np.ndarray:
    """
    Construct a bounded polynomial approximating sqrt(x) on [a, 1].

    The fitting target is extended to all of [-1, 1] by setting it to zero
    for non-positive inputs. This keeps the designed polynomial bounded on the
    full QSVT interval while still approximating sqrt(x) on [a, 1].

    Parameters
    ----------
    a
        Lower endpoint of the positive design interval, with 0 <= a < 1.
    degree
        Polynomial degree.
    num_points
        Number of fitting points.

    Returns
    -------
    numpy.ndarray
        Polynomial coefficients in ascending monomial degree order.

    Examples
    --------
    >>> coeffs = design_sqrt_polynomial(a=0.2, degree=12)
    >>> coeffs.ndim
    1
    """
    a = float(a)
    if not (0.0 <= a < 1.0):
        raise ValueError("a must satisfy 0 <= a < 1.")

    return _fit_on_canonical_interval(
        _design_sqrt_target,
        degree=degree,
        parity=None,
        num_points=num_points,
    )


def design_power_polynomial(
    alpha: float,
    degree: int,
    a: float = 0.0,
    num_points: int = _DEF_NUM_POINTS,
) -> np.ndarray:
    """
    Construct a bounded polynomial approximating x**alpha on [a, 1].

    This helper is intended for non-negative powers. Negative powers are better
    handled through `design_inverse_polynomial`.

    The fitting target is extended to all of [-1, 1] by setting it to zero
    for non-positive inputs.

    Parameters
    ----------
    alpha
        Non-negative exponent.
    degree
        Polynomial degree.
    a
        Lower endpoint of the positive design interval, with 0 <= a < 1.
    num_points
        Number of fitting points.

    Returns
    -------
    numpy.ndarray
        Polynomial coefficients in ascending monomial degree order.

    Examples
    --------
    >>> coeffs = design_power_polynomial(alpha=0.5, degree=12, a=0.2)
    >>> coeffs.ndim
    1
    """
    alpha = float(alpha)
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative.")

    a = float(a)
    if not (0.0 <= a < 1.0):
        raise ValueError("a must satisfy 0 <= a < 1.")

    return _fit_on_canonical_interval(
        lambda x: _design_power_target(x, alpha),
        degree=degree,
        parity=None,
        num_points=num_points,
    )


def design_filter_polynomial(
    cutoff: float,
    degree: int,
    sharpness: float = 12.0,
    num_points: int = _DEF_NUM_POINTS,
) -> np.ndarray:
    """
    Construct a bounded even soft-threshold/filter polynomial on [-1, 1].

    The target is the smooth pass filter

        f(x) = 0.5 * (1 + tanh(sharpness * (|x| - cutoff))).

    Parameters
    ----------
    cutoff
        Filter cutoff satisfying 0 < cutoff < 1.
    degree
        Polynomial degree.
    sharpness
        Positive steepness parameter.
    num_points
        Number of fitting points.

    Returns
    -------
    numpy.ndarray
        Polynomial coefficients in ascending monomial degree order.

    Examples
    --------
    >>> coeffs = design_filter_polynomial(cutoff=0.45, degree=10)
    >>> coeffs.ndim
    1
    """
    cutoff = _validate_unit_interval_parameter(cutoff, "cutoff")
    sharpness = float(sharpness)

    if sharpness <= 0.0:
        raise ValueError("sharpness must be positive.")

    return _fit_on_canonical_interval(
        lambda x: _design_filter_target(x, cutoff, sharpness),
        degree=degree,
        parity="even",
        num_points=num_points,
    )


def _design_quality_report(
    target,
    coeffs: np.ndarray,
    *,
    fit_domain: tuple[float, float] = (-1.0, 1.0),
    fit_num_points: int = _DEF_NUM_POINTS,
    bounded_domain: tuple[float, float] = (-1.0, 1.0),
    bounded_num_points: int = _DEF_BOUND_GRID,
    bound: float = 1.0,
) -> dict[str, object]:
    """
    Assemble a quality report for a design polynomial.
    """
    coeffs = np.asarray(coeffs, dtype=float)

    def approx(x: np.ndarray) -> np.ndarray:
        return np.polynomial.polynomial.polyval(x, coeffs)

    return approximation_quality_report(
        target,
        approx,
        domain=fit_domain,
        num_points=fit_num_points,
        bound=bound,
        bounded_domain=bounded_domain,
        bounded_num_points=bounded_num_points,
        coeffs=coeffs,
    )


def design_inverse_diagnostics(
    gamma: float,
    degree: int,
    num_points: int = _DEF_NUM_POINTS,
    bounded_num_points: int = _DEF_BOUND_GRID,
) -> dict[str, object]:
    """
    Build a report for the inverse-like design polynomial.
    """
    gamma = _validate_unit_interval_parameter(gamma, "gamma")
    coeffs = design_inverse_polynomial(
        gamma=gamma,
        degree=degree,
        num_points=num_points,
    )
    report = _design_quality_report(
        lambda x: _design_inverse_target(x, gamma),
        coeffs,
        fit_num_points=num_points,
        bounded_num_points=bounded_num_points,
    )
    report.update(
        {
            "builder": "design_inverse_polynomial",
            "gamma": gamma,
            "degree": int(degree),
        }
    )
    return report


def design_sign_diagnostics(
    gamma: float,
    degree: int,
    num_points: int = _DEF_NUM_POINTS,
    bounded_num_points: int = _DEF_BOUND_GRID,
) -> dict[str, object]:
    """
    Build a report for the sign design polynomial.
    """
    gamma = _validate_unit_interval_parameter(gamma, "gamma")
    coeffs = design_sign_polynomial(gamma=gamma, degree=degree, num_points=num_points)
    report = _design_quality_report(
        lambda x: _design_sign_target(x, gamma),
        coeffs,
        fit_num_points=num_points,
        bounded_num_points=bounded_num_points,
    )
    report.update(
        {
            "builder": "design_sign_polynomial",
            "gamma": gamma,
            "degree": int(degree),
        }
    )
    return report


def design_projector_diagnostics(
    gamma: float,
    degree: int,
    num_points: int = _DEF_NUM_POINTS,
    bounded_num_points: int = _DEF_BOUND_GRID,
) -> dict[str, object]:
    """
    Build a report for the projector-style design polynomial.
    """
    gamma = _validate_unit_interval_parameter(gamma, "gamma")
    coeffs = design_projector_polynomial(
        gamma=gamma,
        degree=degree,
        num_points=num_points,
    )
    report = _design_quality_report(
        lambda x: _design_projector_target(x, gamma),
        coeffs,
        fit_num_points=num_points,
        bounded_num_points=bounded_num_points,
    )
    report.update(
        {
            "builder": "design_projector_polynomial",
            "gamma": gamma,
            "degree": int(degree),
        }
    )
    return report


def design_sqrt_diagnostics(
    a: float,
    degree: int,
    num_points: int = _DEF_NUM_POINTS,
    bounded_num_points: int = _DEF_BOUND_GRID,
) -> dict[str, object]:
    """
    Build a report for the sqrt design polynomial.
    """
    a = float(a)
    if not (0.0 <= a < 1.0):
        raise ValueError("a must satisfy 0 <= a < 1.")

    coeffs = design_sqrt_polynomial(a=a, degree=degree, num_points=num_points)
    report = _design_quality_report(
        _design_sqrt_target,
        coeffs,
        fit_domain=(a, 1.0),
        fit_num_points=num_points,
        bounded_domain=(-1.0, 1.0),
        bounded_num_points=bounded_num_points,
    )
    report.update({"builder": "design_sqrt_polynomial", "a": a, "degree": int(degree)})
    return report


def design_power_diagnostics(
    alpha: float,
    degree: int,
    a: float = 0.0,
    num_points: int = _DEF_NUM_POINTS,
    bounded_num_points: int = _DEF_BOUND_GRID,
) -> dict[str, object]:
    """
    Build a report for the positive-power design polynomial.
    """
    alpha = float(alpha)
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative.")

    a = float(a)
    if not (0.0 <= a < 1.0):
        raise ValueError("a must satisfy 0 <= a < 1.")

    coeffs = design_power_polynomial(
        alpha=alpha,
        degree=degree,
        a=a,
        num_points=num_points,
    )
    report = _design_quality_report(
        lambda x: _design_power_target(x, alpha),
        coeffs,
        fit_domain=(a, 1.0),
        fit_num_points=num_points,
        bounded_domain=(-1.0, 1.0),
        bounded_num_points=bounded_num_points,
    )
    report.update(
        {
            "builder": "design_power_polynomial",
            "alpha": alpha,
            "a": a,
            "degree": int(degree),
        }
    )
    return report


def design_filter_diagnostics(
    cutoff: float,
    degree: int,
    sharpness: float = 12.0,
    num_points: int = _DEF_NUM_POINTS,
    bounded_num_points: int = _DEF_BOUND_GRID,
) -> dict[str, object]:
    """
    Build a report for the threshold-filter design polynomial.
    """
    cutoff = _validate_unit_interval_parameter(cutoff, "cutoff")
    sharpness = float(sharpness)
    if sharpness <= 0.0:
        raise ValueError("sharpness must be positive.")

    coeffs = design_filter_polynomial(
        cutoff=cutoff,
        degree=degree,
        sharpness=sharpness,
        num_points=num_points,
    )
    report = _design_quality_report(
        lambda x: _design_filter_target(x, cutoff, sharpness),
        coeffs,
        fit_num_points=num_points,
        bounded_num_points=bounded_num_points,
    )
    report.update(
        {
            "builder": "design_filter_polynomial",
            "cutoff": cutoff,
            "sharpness": sharpness,
            "degree": int(degree),
        }
    )
    return report


__all__ = [
    "design_inverse_diagnostics",
    "design_sign_diagnostics",
    "design_projector_diagnostics",
    "design_sqrt_diagnostics",
    "design_power_diagnostics",
    "design_filter_diagnostics",
    "design_filter_polynomial",
    "design_inverse_polynomial",
    "design_power_polynomial",
    "design_projector_polynomial",
    "design_sign_polynomial",
    "design_sqrt_polynomial",
]
