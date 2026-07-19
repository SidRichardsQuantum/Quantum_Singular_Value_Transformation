"""Singular-value filtering and inverse workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._algorithm_reports import (
    algorithm_truth_contract,
    algorithm_workflow_schema_fields,
)
from ._algorithm_shared import (
    _relative_error,
    _validate_state,
)
from .approximation import chebyshev_fit_function
from .design import design_positive_inverse_polynomial
from .polynomials import chebyshev_to_monomial, eval_polynomial


@dataclass(frozen=True)
class SingularValueFilteringWorkflowResult:
    """
    Structured output from a rectangular singular-value filter workflow.
    """

    coeffs: np.ndarray
    matrix: np.ndarray
    singular_values: np.ndarray
    normalized_singular_values: np.ndarray
    polynomial_matrix: np.ndarray
    reference_matrix: np.ndarray
    cutoff: float
    sharpness: float
    scale: float
    degree: int
    operator_relative_error: float
    input_vector: np.ndarray | None = None
    polynomial_output: np.ndarray | None = None
    reference_output: np.ndarray | None = None
    output_relative_error: float | None = None

    def as_report(self) -> dict[str, Any]:
        return {
            **algorithm_workflow_schema_fields(),
            "mode": "singular-value-filtering-workflow",
            "implementation_kind": "dense-svd-polynomial-workflow",
            "truth_contract": algorithm_truth_contract(
                "singular-value-filtering-workflow",
                target="rectangular singular-value filter transform",
            ),
            "degree": self.degree,
            "cutoff": self.cutoff,
            "sharpness": self.sharpness,
            "scale": self.scale,
            "coeffs": self.coeffs,
            "matrix": self.matrix,
            "singular_values": self.singular_values,
            "normalized_singular_values": self.normalized_singular_values,
            "polynomial_matrix": self.polynomial_matrix,
            "reference_matrix": self.reference_matrix,
            "operator_relative_error": self.operator_relative_error,
            "input_vector": self.input_vector,
            "polynomial_output": self.polynomial_output,
            "reference_output": self.reference_output,
            "output_relative_error": self.output_relative_error,
        }


@dataclass(frozen=True)
class SingularValuePseudoinverseWorkflowResult:
    """
    Structured output from a truncated SVD pseudoinverse workflow.
    """

    coeffs: np.ndarray
    matrix: np.ndarray
    rhs: np.ndarray
    singular_values: np.ndarray
    normalized_singular_values: np.ndarray
    polynomial_solution: np.ndarray
    reference_solution: np.ndarray
    polynomial_pseudoinverse: np.ndarray
    reference_pseudoinverse: np.ndarray
    cutoff: float
    scale: float
    degree: int
    residual_norm: float
    reference_residual_norm: float
    solution_relative_error: float
    operator_relative_error: float

    def as_report(self) -> dict[str, Any]:
        return {
            **algorithm_workflow_schema_fields(),
            "mode": "singular-value-pseudoinverse-workflow",
            "implementation_kind": "dense-svd-polynomial-workflow",
            "truth_contract": algorithm_truth_contract(
                "singular-value-pseudoinverse-workflow",
                target="truncated singular-value pseudoinverse action",
            ),
            "degree": self.degree,
            "cutoff": self.cutoff,
            "scale": self.scale,
            "coeffs": self.coeffs,
            "matrix": self.matrix,
            "rhs": self.rhs,
            "singular_values": self.singular_values,
            "normalized_singular_values": self.normalized_singular_values,
            "polynomial_solution": self.polynomial_solution,
            "reference_solution": self.reference_solution,
            "polynomial_pseudoinverse": self.polynomial_pseudoinverse,
            "reference_pseudoinverse": self.reference_pseudoinverse,
            "residual_norm": self.residual_norm,
            "reference_residual_norm": self.reference_residual_norm,
            "solution_relative_error": self.solution_relative_error,
            "operator_relative_error": self.operator_relative_error,
        }


def _validate_matrix_2d(matrix: np.ndarray, name: str = "matrix") -> np.ndarray:
    A = np.asarray(matrix, dtype=complex if np.iscomplexobj(matrix) else float)
    if A.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if min(A.shape) == 0:
        raise ValueError(f"{name} must have nonzero dimensions.")
    return A


def _svd_filter_matrix(
    matrix: np.ndarray,
    singular_weights: np.ndarray,
) -> np.ndarray:
    U, _, Vh = np.linalg.svd(matrix, full_matrices=False)
    return U @ np.diag(singular_weights) @ Vh


def _relative_matrix_error(reference: np.ndarray, approximate: np.ndarray) -> float:
    return _relative_error(np.ravel(reference), np.ravel(approximate))


def singular_value_filtering_workflow(
    matrix: np.ndarray,
    *,
    degree: int,
    cutoff: float,
    sharpness: float = 20.0,
    input_vector: np.ndarray | None = None,
    num_points: int = 1001,
) -> SingularValueFilteringWorkflowResult:
    """
    Apply a smooth QSVT-style filter to the singular values of a matrix.

    The matrix is normalized by its largest singular value. The polynomial is
    designed on normalized singular values in ``[0, 1]`` and compared with the
    smooth reference filter
    ``0.5 * (1 + tanh(sharpness * (sigma - cutoff)))``.
    """
    A = _validate_matrix_2d(matrix)
    cutoff = float(cutoff)
    sharpness = float(sharpness)
    if not 0.0 < cutoff < 1.0:
        raise ValueError("cutoff must satisfy 0 < cutoff < 1.")
    if sharpness <= 0.0:
        raise ValueError("sharpness must be positive.")

    singular_values = np.linalg.svd(A, compute_uv=False)
    scale = float(np.max(singular_values))
    if scale <= 0.0:
        raise ValueError("matrix must have at least one nonzero singular value.")
    normalized = singular_values / scale

    def target(sigma: np.ndarray) -> np.ndarray:
        return 0.5 * (1.0 + np.tanh(sharpness * (sigma - cutoff)))

    cheb = chebyshev_fit_function(
        target,
        degree=degree,
        domain=(0.0, 1.0),
        num_points=num_points,
    )
    coeffs = chebyshev_to_monomial(cheb, domain=(0.0, 1.0))
    polynomial_weights = np.asarray(eval_polynomial(coeffs, normalized), dtype=float)
    reference_weights = target(normalized)
    polynomial_matrix = _svd_filter_matrix(A, polynomial_weights)
    reference_matrix = _svd_filter_matrix(A, reference_weights)

    vec = None
    polynomial_output = None
    reference_output = None
    output_error = None
    if input_vector is not None:
        vec = _validate_state(input_vector, A.shape[1], name="input_vector")
        polynomial_output = polynomial_matrix @ vec
        reference_output = reference_matrix @ vec
        output_error = _relative_error(reference_output, polynomial_output)

    return SingularValueFilteringWorkflowResult(
        coeffs=coeffs,
        matrix=A,
        singular_values=singular_values,
        normalized_singular_values=normalized,
        polynomial_matrix=polynomial_matrix,
        reference_matrix=reference_matrix,
        cutoff=cutoff,
        sharpness=sharpness,
        scale=scale,
        degree=int(degree),
        operator_relative_error=_relative_matrix_error(
            reference_matrix,
            polynomial_matrix,
        ),
        input_vector=vec,
        polynomial_output=polynomial_output,
        reference_output=reference_output,
        output_relative_error=output_error,
    )


def singular_value_pseudoinverse_workflow(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    degree: int,
    cutoff: float,
    num_points: int = 2001,
) -> SingularValuePseudoinverseWorkflowResult:
    """
    Approximate a truncated SVD pseudoinverse action with an inverse polynomial.

    ``cutoff`` is the normalized singular-value threshold. Singular values
    below the threshold are omitted from the dense reference pseudoinverse.
    """
    A = _validate_matrix_2d(matrix)
    b = _validate_state(rhs, A.shape[0], name="rhs")
    cutoff = float(cutoff)
    if not 0.0 < cutoff < 1.0:
        raise ValueError("cutoff must satisfy 0 < cutoff < 1.")

    U, singular_values, Vh = np.linalg.svd(A, full_matrices=False)
    scale = float(np.max(singular_values))
    if scale <= 0.0:
        raise ValueError("matrix must have at least one nonzero singular value.")
    normalized = singular_values / scale

    coeffs = design_positive_inverse_polynomial(
        gamma=cutoff,
        degree=degree,
        num_points=num_points,
    )
    polynomial_inverse_weights = eval_polynomial(coeffs, normalized) / (cutoff * scale)
    reference_inverse_weights = np.where(
        normalized >= cutoff,
        1.0 / singular_values,
        0.0,
    )
    polynomial_pinv = Vh.conj().T @ np.diag(polynomial_inverse_weights) @ U.conj().T
    reference_pinv = Vh.conj().T @ np.diag(reference_inverse_weights) @ U.conj().T
    polynomial_solution = polynomial_pinv @ b
    reference_solution = reference_pinv @ b

    return SingularValuePseudoinverseWorkflowResult(
        coeffs=coeffs,
        matrix=A,
        rhs=b,
        singular_values=singular_values,
        normalized_singular_values=normalized,
        polynomial_solution=polynomial_solution,
        reference_solution=reference_solution,
        polynomial_pseudoinverse=polynomial_pinv,
        reference_pseudoinverse=reference_pinv,
        cutoff=cutoff,
        scale=scale,
        degree=int(degree),
        residual_norm=float(np.linalg.norm(A @ polynomial_solution - b)),
        reference_residual_norm=float(np.linalg.norm(A @ reference_solution - b)),
        solution_relative_error=_relative_error(
            reference_solution, polynomial_solution
        ),
        operator_relative_error=_relative_matrix_error(reference_pinv, polynomial_pinv),
    )
