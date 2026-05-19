"""
End-to-end QSVT-style algorithm workflows.

These helpers combine existing polynomial design, spectral rescaling, QSVT
application, and classical diagnostics into small executable workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .compatibility import qsvt_compatibility_report
from .design import (
    design_positive_inverse_diagnostics,
    design_positive_inverse_polynomial,
)
from .matrix import qsvt_matrix_transform
from .rescaling import ScaledOperator, rescale_positive_semidefinite
from .spectral import apply_polynomial_to_hermitian, eigh_hermitian


@dataclass(frozen=True)
class LinearSystemWorkflowResult:
    """
    Structured output from a positive-definite linear-system workflow.
    """

    coeffs: np.ndarray
    scaled_operator: ScaledOperator
    rhs: np.ndarray
    classical_solution: np.ndarray
    polynomial_solution: np.ndarray
    polynomial_residual_norm: float
    polynomial_relative_error: float
    gamma: float
    degree: int
    diagnostics: dict[str, object]
    compatibility: dict[str, object]
    qsvt_solution: np.ndarray | None = None
    qsvt_residual_norm: float | None = None
    qsvt_relative_error: float | None = None
    qsvt_error: str | None = None

    def as_report(self) -> dict[str, Any]:
        """
        Return a report-style dictionary for JSON conversion or persistence.
        """
        return {
            "mode": "linear-system-workflow",
            "gamma": self.gamma,
            "degree": self.degree,
            "coeffs": self.coeffs,
            "scaled_operator": {
                "matrix": self.scaled_operator.matrix,
                "offset": self.scaled_operator.offset,
                "scale": self.scaled_operator.scale,
                "eigenvalue_bounds": self.scaled_operator.eigenvalue_bounds,
            },
            "rhs": self.rhs,
            "classical_solution": self.classical_solution,
            "polynomial_solution": self.polynomial_solution,
            "polynomial_residual_norm": self.polynomial_residual_norm,
            "polynomial_relative_error": self.polynomial_relative_error,
            "qsvt_solution": self.qsvt_solution,
            "qsvt_residual_norm": self.qsvt_residual_norm,
            "qsvt_relative_error": self.qsvt_relative_error,
            "qsvt_error": self.qsvt_error,
            "diagnostics": self.diagnostics,
            "compatibility": self.compatibility,
        }


def _validate_linear_system_inputs(
    matrix: np.ndarray,
    rhs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.asarray(matrix)
    b = np.asarray(rhs, dtype=complex if np.iscomplexobj(rhs) else float)

    evals, _ = eigh_hermitian(A)
    if evals[0] <= 0.0:
        raise ValueError("matrix must be positive definite.")
    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError("rhs must be a vector whose length matches matrix dimension.")

    dtype = complex if np.iscomplexobj(A) or np.iscomplexobj(b) else float
    return A.astype(dtype, copy=False), b.astype(dtype, copy=False), evals


def _relative_error(reference: np.ndarray, approximate: np.ndarray) -> float:
    denom = np.linalg.norm(reference)
    diff = np.linalg.norm(approximate - reference)
    if denom == 0.0:
        return float(diff)
    return float(diff / denom)


def linear_system_workflow(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    degree: int,
    gamma: float | None = None,
    num_points: int = 2001,
    bounded_num_points: int = 4001,
    attempt_synthesis: bool = True,
    apply_qsvt: bool = True,
) -> LinearSystemWorkflowResult:
    """
    Solve a positive-definite linear system with a QSVT-style inverse workflow.

    The input matrix is scaled by its largest eigenvalue so the scaled spectrum
    lies in ``[gamma_actual, 1]``. A bounded polynomial approximating
    ``gamma / x`` is designed on that interval, then rescaled back to an
    approximation of ``A^{-1} b``.
    """
    A, b, evals = _validate_linear_system_inputs(matrix, rhs)
    scaled = rescale_positive_semidefinite(A)
    scaled_min = float(evals[0] / scaled.scale)

    if gamma is None:
        gamma = min(scaled_min, 1.0 - 1e-12)
    gamma = float(gamma)
    if not (0.0 < gamma < 1.0 and gamma <= scaled_min + 1e-12):
        raise ValueError(
            "gamma must satisfy 0 < gamma < 1 and be no larger than the "
            "scaled minimum eigenvalue."
        )

    coeffs = design_positive_inverse_polynomial(
        gamma=gamma,
        degree=degree,
        num_points=num_points,
    )
    diagnostics = design_positive_inverse_diagnostics(
        gamma=gamma,
        degree=degree,
        num_points=num_points,
        bounded_num_points=bounded_num_points,
    )
    compatibility = qsvt_compatibility_report(
        coeffs,
        bounded_num_points=bounded_num_points,
        attempt_synthesis=attempt_synthesis,
    )

    classical_solution = np.linalg.solve(A, b)
    polynomial_operator = apply_polynomial_to_hermitian(scaled.matrix, coeffs)
    solution_scale = gamma * scaled.scale
    polynomial_solution = (polynomial_operator @ b) / solution_scale
    polynomial_residual = A @ polynomial_solution - b

    qsvt_solution = None
    qsvt_residual_norm = None
    qsvt_relative_error = None
    qsvt_error = None

    if apply_qsvt:
        try:
            raw_qsvt = qsvt_matrix_transform(scaled.matrix, coeffs) @ b
            qsvt_solution = raw_qsvt / solution_scale
            qsvt_residual = A @ qsvt_solution - b
            qsvt_residual_norm = float(np.linalg.norm(qsvt_residual))
            qsvt_relative_error = _relative_error(classical_solution, qsvt_solution)
        except Exception as exc:  # pragma: no cover - backend-dependent path
            qsvt_error = f"{type(exc).__name__}: {exc}"

    return LinearSystemWorkflowResult(
        coeffs=coeffs,
        scaled_operator=scaled,
        rhs=b,
        classical_solution=classical_solution,
        polynomial_solution=polynomial_solution,
        polynomial_residual_norm=float(np.linalg.norm(polynomial_residual)),
        polynomial_relative_error=_relative_error(
            classical_solution,
            polynomial_solution,
        ),
        gamma=gamma,
        degree=int(degree),
        diagnostics=diagnostics,
        compatibility=compatibility,
        qsvt_solution=qsvt_solution,
        qsvt_residual_norm=qsvt_residual_norm,
        qsvt_relative_error=qsvt_relative_error,
        qsvt_error=qsvt_error,
    )


__all__ = ["LinearSystemWorkflowResult", "linear_system_workflow"]
