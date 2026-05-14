"""
qsvt.matrix
-----------

Hermitian-matrix QSVT transform and report helpers.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .operators import (
    _as_numeric_operator,
    _default_wire_order_from_operator,
    _validate_wire_inputs,
    qsvt_top_left_block,
)
from .spectral import apply_polynomial_to_hermitian, eigh_hermitian


def qsvt_matrix_transform(
    operator: np.ndarray,
    poly: Iterable[float],
    *,
    encoding_wires: Iterable[int] | None = None,
    wire_order: Iterable[int] | None = None,
    block_encoding: str = "embedding",
    real_output: bool = True,
) -> np.ndarray:
    """
    Apply QSVT to a small Hermitian matrix and return the logical block.

    This generalizes `qsvt_diagonal_transform` to non-diagonal Hermitian test
    matrices. PennyLane's QSVT matrix can include convention-dependent complex
    phases in the extracted block; when `real_output=True`, this returns the
    real part, which is the quantity used in this package's standard
    real-symmetric report comparisons.

    Parameters
    ----------
    operator
        Square Hermitian matrix with spectrum in [-1, 1].
    poly
        Polynomial coefficients in ascending degree order.
    encoding_wires
        Wires used for the QSVT encoding.
    wire_order
        Wire order passed to `qml.matrix`.
    block_encoding
        PennyLane block-encoding mode.
    real_output
        If True, return the real part of the logical block.

    Returns
    -------
    numpy.ndarray
        Logical QSVT block, or its real part.

    Raises
    ------
    ValueError
        If the operator is not a finite Hermitian matrix with spectrum in
        [-1, 1].
    """
    A = _validate_hermitian_qsvt_matrix(operator)
    block = qsvt_top_left_block(
        A,
        poly,
        encoding_wires=encoding_wires,
        wire_order=wire_order,
        block_encoding=block_encoding,
    )
    return np.real(block).astype(float) if real_output else block


def compare_qsvt_vs_classical_matrix(
    operator: np.ndarray,
    poly: Iterable[float],
    *,
    encoding_wires: Iterable[int] | None = None,
    wire_order: Iterable[int] | None = None,
    block_encoding: str = "embedding",
) -> dict[str, object]:
    """
    Compare a non-diagonal QSVT block against a classical spectral polynomial.

    Parameters
    ----------
    operator
        Square Hermitian matrix with spectrum in [-1, 1].
    poly
        Polynomial coefficients in ascending degree order.
    encoding_wires
        Wires used for the QSVT encoding.
    wire_order
        Wire order passed to `qml.matrix`.
    block_encoding
        PennyLane block-encoding mode.

    Returns
    -------
    dict[str, object]
        Dictionary with keys:
        - "input"
        - "qsvt"
        - "qsvt_imag"
        - "classical"
        - "comparison_basis"
        - "abs_error"
    """
    A = _validate_hermitian_qsvt_matrix(operator)
    block = qsvt_matrix_transform(
        A,
        poly,
        encoding_wires=encoding_wires,
        wire_order=wire_order,
        block_encoding=block_encoding,
        real_output=False,
    )
    classical = apply_polynomial_to_hermitian(A, poly)
    qsvt_reference, comparison_basis = _matrix_qsvt_reference(block, classical)

    return {
        "input": A,
        "qsvt": qsvt_reference,
        "qsvt_imag": np.imag(block).astype(float),
        "classical": classical,
        "comparison_basis": comparison_basis,
        "abs_error": np.abs(qsvt_reference - classical),
    }


def qsvt_matrix_transform_report(
    operator: np.ndarray,
    poly: Iterable[float],
    *,
    encoding_wires: Iterable[int] | None = None,
    wire_order: Iterable[int] | None = None,
    block_encoding: str = "embedding",
    allow_qsvt_failure: bool = False,
) -> dict[str, object]:
    """
    Build a report comparing a non-diagonal QSVT block with P(A).

    The input matrix must be finite, Hermitian/symmetric, and have eigenvalues
    in [-1, 1]. The classical reference is computed spectrally as P(A).

    Parameters
    ----------
    operator
        Square Hermitian matrix with spectrum in [-1, 1].
    poly
        Polynomial coefficients in ascending degree order.
    encoding_wires
        Wires used for the QSVT encoding.
    wire_order
        Wire order passed to `qml.matrix`.
    block_encoding
        PennyLane block-encoding mode.
    allow_qsvt_failure
        If True, return a report with classical values and synthesis error
        details when PennyLane cannot synthesize the QSVT transform.

    Returns
    -------
    dict[str, object]
        Report containing the input matrix, eigenvalues, coefficients, QSVT
        real block, QSVT imaginary block, classical P(A), error metrics, and
        basic wire/dimension metadata.
    """
    A = _validate_hermitian_qsvt_matrix(operator)
    coeffs = np.asarray(list(poly), dtype=float)

    if coeffs.ndim != 1 or coeffs.size == 0:
        raise ValueError("poly must contain at least one coefficient.")
    if not np.all(np.isfinite(coeffs)):
        raise ValueError("polynomial coefficients must be finite.")

    eigenvalues, _ = eigh_hermitian(A)
    if encoding_wires is None:
        encoding_wires = _default_wire_order_from_operator(A)
    enc, order = _validate_wire_inputs(encoding_wires, wire_order)

    classical = apply_polynomial_to_hermitian(A, coeffs)

    report: dict[str, object] = {
        "mode": "qsvt-matrix-transform-report",
        "input": A,
        "eigenvalues": eigenvalues,
        "poly": coeffs,
        "classical": classical,
        "encoding_wires": enc,
        "wire_order": order,
        "block_encoding": block_encoding,
        "matrix_dimension": int(A.shape[0]),
        "unitary_dimension": int(2 ** len(order)),
        "polynomial_degree": int(coeffs.size - 1),
    }

    try:
        block = qsvt_matrix_transform(
            A,
            coeffs,
            encoding_wires=enc,
            wire_order=order,
            block_encoding=block_encoding,
            real_output=False,
        )
    except Exception as exc:
        if not allow_qsvt_failure:
            raise
        report.update(
            {
                "qsvt_succeeded": False,
                "qsvt": None,
                "qsvt_imag": None,
                "abs_error": None,
                "max_error": None,
                "rms_error": None,
                "frobenius_error": None,
                "max_imag_abs": None,
                "qsvt_error_type": type(exc).__name__,
                "qsvt_error": str(exc),
            }
        )
        return report

    qsvt_reference, comparison_basis = _matrix_qsvt_reference(block, classical)
    qsvt_imag = np.imag(block).astype(float)
    abs_error = np.abs(qsvt_reference - classical)
    report.update(
        {
            "qsvt_succeeded": True,
            "qsvt": qsvt_reference,
            "qsvt_imag": qsvt_imag,
            "comparison_basis": comparison_basis,
            "abs_error": abs_error,
            "max_error": float(np.max(abs_error)),
            "rms_error": float(np.sqrt(np.mean(abs_error**2))),
            "frobenius_error": float(np.linalg.norm(qsvt_reference - classical)),
            "max_imag_abs": float(np.max(np.abs(qsvt_imag))),
        }
    )
    return report


def _validate_hermitian_qsvt_matrix(operator: np.ndarray) -> np.ndarray:
    """
    Validate a finite Hermitian matrix for small QSVT matrix workflows.
    """
    if np.isscalar(operator):
        raise ValueError("operator must be a square Hermitian matrix, not a scalar.")

    A = _as_numeric_operator(operator)
    if not np.all(np.isfinite(A)):
        raise ValueError("operator entries must be finite.")

    eigenvalues, _ = eigh_hermitian(A)
    if np.max(np.abs(eigenvalues)) > 1.0 + 1e-12:
        raise ValueError("QSVT matrix eigenvalues must lie in [-1, 1].")

    return A


def _matrix_qsvt_reference(
    block: np.ndarray,
    classical: np.ndarray,
    *,
    imag_tol: float = 1e-10,
) -> tuple[np.ndarray, str]:
    """
    Choose the QSVT block representation used for matrix-report comparisons.

    For effectively real classical references, compare against the real part of
    the extracted QSVT block to preserve the existing report convention. For
    genuinely complex Hermitian references, compare against the full complex
    block.
    """
    if np.max(np.abs(np.imag(classical))) <= imag_tol:
        return np.real(block).astype(float), "real_part"
    return block, "full_complex"


__all__ = [
    "qsvt_matrix_transform",
    "compare_qsvt_vs_classical_matrix",
    "qsvt_matrix_transform_report",
]
