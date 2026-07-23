"""Compatibility imports for matrix-function designs now exposed by
:mod:`qsvt.design`.
"""

from __future__ import annotations

from ._matrix_function_designs import (
    RealTimeEvolutionPolynomials,
    ScaledPolynomial,
    design_gaussian_window_polynomial,
    design_imaginary_time_polynomial,
    design_low_energy_projector_polynomial,
    design_positive_inverse_matrix_polynomial,
    design_real_time_evolution_polynomials,
    design_resolvent_polynomials,
)

__all__ = [
    "RealTimeEvolutionPolynomials",
    "ScaledPolynomial",
    "design_gaussian_window_polynomial",
    "design_imaginary_time_polynomial",
    "design_low_energy_projector_polynomial",
    "design_positive_inverse_matrix_polynomial",
    "design_real_time_evolution_polynomials",
    "design_resolvent_polynomials",
]
