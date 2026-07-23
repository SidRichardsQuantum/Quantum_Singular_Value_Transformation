"""
General polynomial builders for physics matrix functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .approximation import chebyshev_fit_function
from .polynomials import chebyshev_to_monomial


@dataclass(frozen=True)
class RealTimeEvolutionPolynomials:
    """
    Polynomial pair for exp(-i H t) = cos(Ht) - i sin(Ht).
    """

    cos_coeffs: np.ndarray
    sin_coeffs: np.ndarray
    scaled_time: float


@dataclass(frozen=True)
class ScaledPolynomial:
    """
    Polynomial coefficients with an optional scalar prefactor.
    """

    coeffs: np.ndarray
    prefactor: float = 1.0


def design_real_time_evolution_polynomials(
    time: float,
    scale: float,
    *,
    degree: int,
    num_points: int = 500,
) -> RealTimeEvolutionPolynomials:
    """
    Design monomial polynomials for cos(scale*time*x) and sin(scale*time*x).
    """
    tau = float(scale) * float(time)
    cos_cheb = chebyshev_fit_function(
        lambda x: np.cos(tau * x),
        degree=degree,
        num_points=num_points,
    )
    sin_cheb = chebyshev_fit_function(
        lambda x: np.sin(tau * x),
        degree=degree,
        num_points=num_points,
    )
    cos_coeffs = chebyshev_to_monomial(cos_cheb)
    sin_coeffs = chebyshev_to_monomial(sin_cheb)
    # The targets have exact even/odd parity. Remove conversion roundoff in the
    # opposite-parity coefficients so downstream QSVT realizability checks see
    # the mathematical construction rather than numerical noise.
    cos_coeffs[1::2] = 0.0
    sin_coeffs[0::2] = 0.0
    return RealTimeEvolutionPolynomials(
        cos_coeffs=cos_coeffs,
        sin_coeffs=sin_coeffs,
        scaled_time=tau,
    )


def design_imaginary_time_polynomial(
    beta: float,
    scale: float,
    *,
    offset: float = 0.0,
    degree: int,
    num_points: int = 2001,
) -> ScaledPolynomial:
    """
    Design a bounded polynomial for exp(-beta * (offset + scale*x)).
    """
    beta_scaled = -float(beta) * float(scale)
    prefactor = float(np.exp(-float(beta) * float(offset) + abs(beta_scaled)))
    cheb = chebyshev_fit_function(
        lambda x: np.exp(beta_scaled * x - abs(beta_scaled)),
        degree=degree,
        num_points=num_points,
    )
    return ScaledPolynomial(chebyshev_to_monomial(cheb), prefactor=prefactor)


def design_resolvent_polynomials(
    omega: float,
    eta: float,
    scale: float,
    *,
    offset: float = 0.0,
    degree: int,
    num_points: int = 2001,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Design real and imaginary monomial polynomials for (omega+i eta-H)^-1.
    """
    if eta <= 0.0:
        raise ValueError("eta must be positive.")

    def denom(x: np.ndarray) -> np.ndarray:
        shifted = float(omega) - (float(offset) + float(scale) * x)
        return shifted**2 + float(eta) ** 2

    real_cheb = chebyshev_fit_function(
        lambda x: (float(omega) - (float(offset) + float(scale) * x)) / denom(x),
        degree=degree,
        num_points=num_points,
    )
    imag_cheb = chebyshev_fit_function(
        lambda x: -float(eta) / denom(x),
        degree=degree,
        num_points=num_points,
    )
    return chebyshev_to_monomial(real_cheb), chebyshev_to_monomial(imag_cheb)


def design_gaussian_window_polynomial(
    center: float,
    width: float,
    *,
    degree: int,
    num_points: int = 2001,
) -> np.ndarray:
    """
    Design a smooth Gaussian spectral-window polynomial on [-1, 1].
    """
    if width <= 0.0:
        raise ValueError("width must be positive.")
    cheb = chebyshev_fit_function(
        lambda x: np.exp(-0.5 * ((x - center) / width) ** 2),
        degree=degree,
        num_points=num_points,
    )
    return chebyshev_to_monomial(cheb)


def design_low_energy_projector_polynomial(
    gap: float,
    *,
    degree: int,
    num_points: int = 2001,
) -> np.ndarray:
    """
    Design a projector polynomial for spectra shifted so low energy is positive.
    """
    from .design import design_projector_polynomial

    return design_projector_polynomial(
        gamma=gap,
        degree=degree,
        num_points=num_points,
    )


def design_positive_inverse_matrix_polynomial(
    gamma: float,
    *,
    degree: int,
    num_points: int = 2001,
) -> np.ndarray:
    """
    Design a positive-definite inverse polynomial for spectra in [gamma, 1].
    """
    from .design import design_positive_inverse_polynomial

    return design_positive_inverse_polynomial(
        gamma=gamma,
        degree=degree,
        num_points=num_points,
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
