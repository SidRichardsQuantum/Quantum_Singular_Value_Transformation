"""Shared numerical validation for algorithm workflows."""

from __future__ import annotations

import numpy as np


def _validate_state(
    state: np.ndarray,
    dimension: int,
    name: str = "state",
) -> np.ndarray:
    vec = np.asarray(state, dtype=complex if np.iscomplexobj(state) else float)
    if vec.ndim != 1 or vec.shape[0] != dimension:
        raise ValueError(
            f"{name} must be a vector whose length matches matrix dimension."
        )
    if np.linalg.norm(vec) == 0.0:
        raise ValueError(f"{name} must be nonzero.")
    return vec


def _normalize_state(state: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(state)
    if norm == 0.0:
        raise ValueError("cannot normalize a zero state.")
    return state / norm


def _state_error(reference: np.ndarray, approximate: np.ndarray) -> float:
    phase = np.vdot(reference, approximate)
    if abs(phase) > 0.0:
        approximate = approximate * np.exp(-1j * np.angle(phase))
    return _relative_error(reference, approximate)


def _relative_error(reference: np.ndarray, approximate: np.ndarray) -> float:
    denom = np.linalg.norm(reference)
    diff = np.linalg.norm(approximate - reference)
    if denom == 0.0:
        return float(diff)
    return float(diff / denom)
