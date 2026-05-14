"""
Shared polynomial-fitting internals.

These helpers are private to the package. They centralize the repeated
Chebyshev fitting, parity projection, monomial conversion, and sampled
boundedness enforcement used by the design and template modules.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .approximation import chebyshev_fit_function
from .polynomials import normalize_coefficients


def validate_degree(degree: int) -> int:
    """
    Validate a non-negative polynomial degree.
    """
    if degree < 0:
        raise ValueError("degree must be non-negative.")
    return int(degree)


def validate_num_points(num_points: int, name: str = "num_points") -> int:
    """
    Validate a fitting or boundedness grid size.
    """
    if num_points < 2:
        raise ValueError(f"{name} must be at least 2.")
    return int(num_points)


def chebyshev_to_monomial(
    cheb_coeffs: np.ndarray,
    *,
    domain: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """
    Convert Chebyshev-basis coefficients to ascending monomial coefficients.
    """
    poly = np.polynomial.Chebyshev(cheb_coeffs, domain=domain)
    coeffs = np.asarray(poly.convert(kind=np.polynomial.Polynomial).coef, dtype=float)
    return normalize_coefficients(coeffs)


def grid_max_abs(coeffs: np.ndarray, num_points: int = 4001) -> float:
    """
    Estimate max |P(x)| on [-1, 1] using a dense grid.
    """
    num_points = validate_num_points(num_points)
    xs = np.linspace(-1.0, 1.0, num_points)
    values = np.polynomial.polynomial.polyval(xs, coeffs)
    return float(np.max(np.abs(values)))


def enforce_boundedness(
    coeffs: np.ndarray,
    *,
    num_points: int = 4001,
) -> np.ndarray:
    """
    Rescale coefficients when sampled values exceed unit magnitude.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    max_abs = grid_max_abs(coeffs, num_points=num_points)
    if max_abs > 1.0:
        coeffs = coeffs / max_abs
    return normalize_coefficients(coeffs)


def apply_parity_projection(
    cheb_coeffs: np.ndarray,
    parity: str | None,
) -> np.ndarray:
    """
    Project Chebyshev coefficients to even or odd parity when requested.
    """
    coeffs = np.asarray(cheb_coeffs, dtype=float).copy()
    if parity == "odd":
        coeffs[::2] = 0.0
    elif parity == "even":
        coeffs[1::2] = 0.0
    elif parity is not None:
        raise ValueError("parity must be None, 'even', or 'odd'.")
    return coeffs


def fit_bounded_monomial(
    func: Callable[[np.ndarray], np.ndarray | float],
    *,
    degree: int,
    parity: str | None = None,
    num_points: int = 2001,
    bound_num_points: int = 4001,
    domain: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """
    Fit a target, optionally enforce parity, convert to monomials, and bound.
    """
    degree = validate_degree(degree)
    num_points = validate_num_points(num_points)

    cheb_coeffs = chebyshev_fit_function(
        func,
        degree=degree,
        domain=domain,
        num_points=num_points,
    )
    cheb_coeffs = apply_parity_projection(cheb_coeffs, parity)
    coeffs = chebyshev_to_monomial(cheb_coeffs, domain=domain)
    return enforce_boundedness(
        coeffs,
        num_points=max(validate_num_points(bound_num_points), num_points),
    )


__all__ = [
    "apply_parity_projection",
    "chebyshev_to_monomial",
    "enforce_boundedness",
    "fit_bounded_monomial",
    "grid_max_abs",
    "validate_degree",
    "validate_num_points",
]
