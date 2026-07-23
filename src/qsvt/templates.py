"""Compatibility imports for the renamed :mod:`qsvt.presets` module.

New code should import named polynomial families from :mod:`qsvt.presets`.
"""

from __future__ import annotations

from .presets import (
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

__all__ = [
    "exponential_approximation_diagnostics",
    "exponential_approximation_polynomial",
    "inverse_like_diagnostics",
    "inverse_like_polynomial",
    "sign_approximation_diagnostics",
    "sign_approximation_polynomial",
    "soft_threshold_filter_diagnostics",
    "soft_threshold_filter_polynomial",
    "sqrt_approximation_diagnostics",
    "sqrt_approximation_polynomial",
]
