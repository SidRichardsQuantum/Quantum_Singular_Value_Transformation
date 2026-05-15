"""
Spectral rescaling helpers for QSVT-style matrix-function workflows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectral import eigh_hermitian


@dataclass(frozen=True)
class ScaledOperator:
    """
    Hermitian operator rescaled by an affine map.
    """

    matrix: np.ndarray
    offset: float
    scale: float
    eigenvalue_bounds: tuple[float, float]

    def original_from_scaled_value(
        self,
        value: np.ndarray | float,
    ) -> np.ndarray | float:
        """
        Map scaled eigenvalues back to the original spectral coordinate.
        """
        return self.offset + self.scale * value


def spectral_bounds(matrix: np.ndarray) -> tuple[float, float]:
    """
    Return the minimum and maximum eigenvalues of a Hermitian matrix.
    """
    evals, _ = eigh_hermitian(matrix)
    return float(evals[0]), float(evals[-1])


def rescale_hermitian_to_unit_interval(matrix: np.ndarray) -> ScaledOperator:
    """
    Affinely map a Hermitian matrix spectrum to approximately [-1, 1].
    """
    lower, upper = spectral_bounds(matrix)
    offset = 0.5 * (lower + upper)
    scale = 0.5 * (upper - lower)

    if scale <= 0.0:
        raise ValueError("matrix must have nonzero spectral width.")

    A = np.asarray(matrix)
    return ScaledOperator(
        matrix=(A - offset * np.eye(A.shape[0])) / scale,
        offset=offset,
        scale=scale,
        eigenvalue_bounds=(lower, upper),
    )


def rescale_hermitian_about_cutoff(
    matrix: np.ndarray,
    cutoff: float,
    *,
    low_energy_positive: bool = True,
) -> ScaledOperator:
    """
    Shift a Hermitian matrix around a cutoff and scale it into [-1, 1].

    When `low_energy_positive=True`, eigenvalues below the cutoff map to
    positive scaled values, which is convenient for low-energy projectors.
    """
    A = np.asarray(matrix)
    evals, _ = eigh_hermitian(A)
    scale = float(np.max(np.abs(evals - cutoff)))
    if scale <= 0.0:
        raise ValueError("cutoff must not equal every eigenvalue.")

    sign = -1.0 if low_energy_positive else 1.0
    return ScaledOperator(
        matrix=sign * (A - cutoff * np.eye(A.shape[0])) / scale,
        offset=float(cutoff),
        scale=sign * scale,
        eigenvalue_bounds=(float(evals[0]), float(evals[-1])),
    )


def rescale_positive_semidefinite(matrix: np.ndarray) -> ScaledOperator:
    """
    Scale a positive semidefinite Hermitian matrix so its largest eigenvalue is 1.
    """
    lower, upper = spectral_bounds(matrix)
    if lower < -1e-12:
        raise ValueError("matrix must be positive semidefinite.")
    if upper <= 0.0:
        raise ValueError("matrix must have a positive maximum eigenvalue.")

    return ScaledOperator(
        matrix=np.asarray(matrix) / upper,
        offset=0.0,
        scale=upper,
        eigenvalue_bounds=(lower, upper),
    )


__all__ = [
    "ScaledOperator",
    "rescale_hermitian_about_cutoff",
    "rescale_hermitian_to_unit_interval",
    "rescale_positive_semidefinite",
    "spectral_bounds",
]
