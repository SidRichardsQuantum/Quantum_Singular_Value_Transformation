"""
Private target functions and fitting helpers for bounded polynomial design.
"""

from __future__ import annotations

import numpy as np

from ._polyfit import fit_bounded_monomial

DEF_NUM_POINTS = 2001
DEF_BOUND_GRID = 4001


def validate_unit_interval_parameter(value: float, name: str) -> float:
    """
    Validate a parameter constrained to the open unit interval.
    """
    value = float(value)
    if not (0.0 < value < 1.0):
        raise ValueError(f"{name} must satisfy 0 < {name} < 1.")
    return value


def tanh_sharpness_from_margin(gamma: float, target_value: float = 0.98) -> float:
    """
    Choose tanh sharpness so that tanh(sharpness * gamma) ~= target_value.
    """
    gamma = validate_unit_interval_parameter(gamma, "gamma")
    target_value = float(target_value)

    if not (0.0 < target_value < 1.0):
        raise ValueError("target_value must lie strictly between 0 and 1.")

    return float(np.arctanh(target_value) / gamma)


def design_inverse_target(x: np.ndarray, gamma: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.sign(x)
    mask = np.abs(x) >= gamma
    out[mask] = gamma / x[mask]
    out[x == 0.0] = 0.0
    return out


def design_positive_inverse_target(
    x: np.ndarray,
    gamma: float,
    extension: str,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if extension == "even":
        return gamma / np.maximum(np.abs(x), gamma)
    if extension == "flat":
        out = np.ones_like(x, dtype=float)
        mask = x >= gamma
        out[mask] = gamma / x[mask]
        return out
    raise ValueError("extension must be 'auto', 'even', or 'flat'.")


def design_sign_target(x: np.ndarray, gamma: float) -> np.ndarray:
    sharpness = tanh_sharpness_from_margin(gamma, target_value=0.98)
    return np.tanh(sharpness * np.asarray(x, dtype=float))


def design_projector_target(x: np.ndarray, gamma: float) -> np.ndarray:
    return 0.5 * (1.0 + design_sign_target(x, gamma))


def design_sqrt_target(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    mask = x >= 0.0
    out[mask] = np.sqrt(np.clip(x[mask], 0.0, 1.0))
    return out


def design_power_target(x: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.clip(x, 0.0, 1.0) ** alpha


def design_filter_target(x: np.ndarray, cutoff: float, sharpness: float) -> np.ndarray:
    return 0.5 * (
        1.0
        + np.tanh(
            sharpness * (np.abs(np.asarray(x, dtype=float)) - cutoff),
        )
    )


def design_interval_projector_target(
    x: np.ndarray,
    lower: float,
    upper: float,
    sharpness: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    left = 0.5 * (1.0 + np.tanh(sharpness * (x - lower)))
    right = 0.5 * (1.0 + np.tanh(sharpness * (upper - x)))
    return left * right


def fit_on_canonical_interval(
    func,
    *,
    degree: int,
    parity: str | None = None,
    num_points: int = DEF_NUM_POINTS,
) -> np.ndarray:
    """
    Fit a target on [-1, 1], optionally enforce parity, and bound it.
    """
    return fit_bounded_monomial(
        func,
        degree=degree,
        parity=parity,
        num_points=num_points,
        bound_num_points=DEF_BOUND_GRID,
    )
