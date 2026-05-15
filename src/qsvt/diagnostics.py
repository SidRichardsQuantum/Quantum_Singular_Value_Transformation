"""
Diagnostics for state, operator, and physics-oriented examples.
"""

from __future__ import annotations

import numpy as np

from .spectral import eigh_hermitian


def relative_state_error(reference: np.ndarray, approximate: np.ndarray) -> float:
    """
    Compute ||approximate - reference|| / ||reference||.
    """
    denom = np.linalg.norm(reference)
    if denom == 0.0:
        raise ValueError("reference state must be nonzero.")
    return float(np.linalg.norm(approximate - reference) / denom)


def operator_error(
    reference: np.ndarray,
    approximate: np.ndarray,
    *,
    relative: bool = True,
) -> float:
    """
    Compute absolute or relative Frobenius error between operators.
    """
    err = np.linalg.norm(approximate - reference)
    if not relative:
        return float(err)
    denom = np.linalg.norm(reference)
    if denom == 0.0:
        raise ValueError("reference operator must be nonzero.")
    return float(err / denom)


def expectation_value(operator: np.ndarray, state: np.ndarray) -> float | complex:
    """
    Compute <state|operator|state>.
    """
    value = np.vdot(state, np.asarray(operator) @ state)
    return np.real_if_close(value).item()


def ground_state_overlap(hamiltonian: np.ndarray, state: np.ndarray) -> float:
    """
    Return overlap probability with the ground-state eigenvector.
    """
    _, evecs = eigh_hermitian(hamiltonian)
    ground = evecs[:, 0]
    return float(abs(np.vdot(ground, state)) ** 2)


def spectral_weights(operator: np.ndarray, state: np.ndarray) -> np.ndarray:
    """
    Return probabilities of a state in the eigenbasis of a Hermitian operator.
    """
    _, evecs = eigh_hermitian(operator)
    return np.abs(evecs.conj().T @ state) ** 2


def density_matrix_error(reference: np.ndarray, approximate: np.ndarray) -> float:
    """
    Compute relative Frobenius error between density matrices.
    """
    return operator_error(reference, approximate, relative=True)


__all__ = [
    "density_matrix_error",
    "expectation_value",
    "ground_state_overlap",
    "operator_error",
    "relative_state_error",
    "spectral_weights",
]
