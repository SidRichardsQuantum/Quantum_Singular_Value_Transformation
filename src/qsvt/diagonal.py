"""
qsvt.diagonal
-------------

Scalar and diagonal QSVT transform helpers.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ._algorithm_reports import qsvt_verification_truth_contract
from .operators import (
    _default_wire_order_from_operator,
    _validate_wire_inputs,
    qsvt_top_left_block,
    qsvt_unitary,
)
from .polynomials import eval_polynomial


def qsvt_scalar_output(
    x: float,
    poly: Iterable[float],
    *,
    encoding_wires: Iterable[int] | None = None,
    wire_order: Iterable[int] | None = None,
    block_encoding: str = "embedding",
    real_output: bool = True,
) -> float | complex:
    """
    Apply QSVT to a scalar and return the top-left matrix element.

    This is the standard scalar-QSP-style notebook pattern.

    Parameters
    ----------
    x
        Scalar input, typically in [-1, 1].
    poly
        Polynomial coefficients in ascending degree order.
    encoding_wires
        Wires used for the QSVT encoding. Defaults to [0].
    wire_order
        Wire order passed to `qml.matrix`. Defaults to encoding_wires.
    block_encoding
        PennyLane block-encoding mode.
    real_output
        If True, return the real part as a float. If False, return the
        complex matrix entry directly.

    Returns
    -------
    float or complex
        Top-left matrix entry of the resulting QSVT unitary.

    Examples
    --------
    >>> qsvt_scalar_output(0.5, [0, 0, 1], encoding_wires=[0])
    0.25
    """
    U = qsvt_unitary(
        float(x),
        poly,
        encoding_wires=encoding_wires,
        wire_order=wire_order,
        block_encoding=block_encoding,
    )
    value = U[0, 0]
    return float(np.real(value)) if real_output else value


def qsvt_scalar_scan(
    xs: Iterable[float],
    poly: Iterable[float],
    *,
    encoding_wires: Iterable[int] | None = None,
    wire_order: Iterable[int] | None = None,
    block_encoding: str = "embedding",
) -> np.ndarray:
    """
    Evaluate scalar QSVT outputs over a collection of scalar inputs.

    Parameters
    ----------
    xs
        Scalar inputs.
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
    numpy.ndarray
        Real QSVT outputs for each input scalar.
    """
    x_arr = np.asarray(list(xs), dtype=float)

    return np.array(
        [
            qsvt_scalar_output(
                x,
                poly,
                encoding_wires=encoding_wires,
                wire_order=wire_order,
                block_encoding=block_encoding,
                real_output=True,
            )
            for x in x_arr
        ],
        dtype=float,
    )


def qsvt_diagonal_transform(
    diagonal: Iterable[float],
    poly: Iterable[float],
    *,
    encoding_wires: Iterable[int] | None = None,
    wire_order: Iterable[int] | None = None,
    block_encoding: str = "embedding",
    real_output: bool = True,
) -> np.ndarray:
    """
    Apply QSVT to a diagonal matrix and return the transformed diagonal entries.

    This is a compact helper for the common notebook pattern:
    build A = diag(singular_values), run QSVT, extract the top-left block,
    and read off its diagonal.

    Parameters
    ----------
    diagonal
        Diagonal entries of the matrix.
    poly
        Polynomial coefficients in ascending degree order.
    encoding_wires
        Wires used for the QSVT encoding.
    wire_order
        Wire order passed to `qml.matrix`.
    block_encoding
        PennyLane block-encoding mode.
    real_output
        If True, return the real part of the transformed diagonal.

    Returns
    -------
    numpy.ndarray
        Transformed diagonal entries.

    Examples
    --------
    >>> vals = qsvt_diagonal_transform(
    ...     [1.0, 0.7, 0.3, 0.1], [0, 0, 1], encoding_wires=[0, 1, 2]
    ... )
    >>> vals.shape
    (4,)
    """
    diag_vals = np.asarray(list(diagonal), dtype=float)
    A = np.diag(diag_vals)

    block = qsvt_top_left_block(
        A,
        poly,
        encoding_wires=encoding_wires,
        wire_order=wire_order,
        block_encoding=block_encoding,
    )

    transformed = np.diagonal(block)
    return np.real(transformed).astype(float) if real_output else transformed


def classical_diagonal_polynomial_transform(
    diagonal: Iterable[float],
    poly: Iterable[float],
) -> np.ndarray:
    """
    Apply a polynomial classically to a list of diagonal entries.

    Parameters
    ----------
    diagonal
        Diagonal entries.
    poly
        Polynomial coefficients in ascending degree order.

    Returns
    -------
    numpy.ndarray
        Classical polynomial transform P(diagonal).
    """
    diag_vals = np.asarray(list(diagonal), dtype=float)
    return np.asarray(eval_polynomial(poly, diag_vals), dtype=float)


def compare_qsvt_vs_classical_diagonal(
    diagonal: Iterable[float],
    poly: Iterable[float],
    *,
    encoding_wires: Iterable[int] | None = None,
    wire_order: Iterable[int] | None = None,
    block_encoding: str = "embedding",
) -> dict[str, np.ndarray]:
    """
    Compare QSVT-transformed diagonal entries against the direct classical
    polynomial transform.

    Parameters
    ----------
    diagonal
        Diagonal entries of the input matrix.
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
    dict[str, numpy.ndarray]
        Dictionary with keys:
        - "input"
        - "qsvt"
        - "classical"
        - "abs_error"
    """
    diag_vals = np.asarray(list(diagonal), dtype=float)

    qsvt_vals = qsvt_diagonal_transform(
        diag_vals,
        poly,
        encoding_wires=encoding_wires,
        wire_order=wire_order,
        block_encoding=block_encoding,
        real_output=True,
    )
    classical_vals = classical_diagonal_polynomial_transform(diag_vals, poly)

    return {
        "input": diag_vals,
        "qsvt": qsvt_vals,
        "classical": classical_vals,
        "abs_error": np.abs(qsvt_vals - classical_vals),
    }


def qsvt_transform_report(
    diagonal: Iterable[float],
    poly: Iterable[float],
    *,
    encoding_wires: Iterable[int] | None = None,
    wire_order: Iterable[int] | None = None,
    block_encoding: str = "embedding",
    allow_qsvt_failure: bool = False,
) -> dict[str, object]:
    """
    Build a report comparing QSVT and classical diagonal polynomial transforms.

    Parameters
    ----------
    diagonal
        Diagonal entries of the input matrix.
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
        Report containing input values, coefficients, QSVT output, classical
        output, error metrics, and basic wire/dimension metadata.
    """
    diag_vals = np.asarray(list(diagonal), dtype=float)
    coeffs = np.asarray(list(poly), dtype=float)

    if diag_vals.ndim != 1 or diag_vals.size == 0:
        raise ValueError("diagonal must contain at least one value.")
    if coeffs.ndim != 1 or coeffs.size == 0:
        raise ValueError("poly must contain at least one coefficient.")
    if not np.all(np.isfinite(diag_vals)):
        raise ValueError("diagonal values must be finite.")
    if not np.all(np.isfinite(coeffs)):
        raise ValueError("polynomial coefficients must be finite.")
    if np.max(np.abs(diag_vals)) > 1.0 + 1e-12:
        raise ValueError("QSVT diagonal values must lie in [-1, 1].")

    if encoding_wires is None:
        encoding_wires = _default_wire_order_from_operator(np.diag(diag_vals))
    enc, order = _validate_wire_inputs(encoding_wires, wire_order)

    classical_vals = classical_diagonal_polynomial_transform(diag_vals, coeffs)

    report: dict[str, object] = {
        "mode": "qsvt-transform-report",
        "truth_contract": qsvt_verification_truth_contract(
            "qsvt-transform-report",
            target="diagonal polynomial transform",
            qsvt_check="failed" if allow_qsvt_failure else "succeeded",
        ),
        "input": diag_vals,
        "poly": coeffs,
        "classical": classical_vals,
        "encoding_wires": enc,
        "wire_order": order,
        "block_encoding": block_encoding,
        "num_values": int(diag_vals.size),
        "matrix_dimension": int(diag_vals.size),
        "unitary_dimension": int(2 ** len(order)),
        "polynomial_degree": int(coeffs.size - 1),
    }

    try:
        qsvt_vals = qsvt_diagonal_transform(
            diag_vals,
            coeffs,
            encoding_wires=enc,
            wire_order=order,
            block_encoding=block_encoding,
            real_output=True,
        )
    except Exception as exc:
        if not allow_qsvt_failure:
            raise
        report.update(
            {
                "truth_contract": qsvt_verification_truth_contract(
                    "qsvt-transform-report",
                    target="diagonal polynomial transform",
                    qsvt_check="failed",
                ),
                "qsvt_succeeded": False,
                "qsvt": None,
                "abs_error": None,
                "max_error": None,
                "rms_error": None,
                "qsvt_error_type": type(exc).__name__,
                "qsvt_error": str(exc),
            }
        )
        return report

    abs_error = np.abs(qsvt_vals - classical_vals)
    report.update(
        {
            "truth_contract": qsvt_verification_truth_contract(
                "qsvt-transform-report",
                target="diagonal polynomial transform",
                qsvt_check="succeeded",
            ),
            "qsvt_succeeded": True,
            "qsvt": qsvt_vals,
            "abs_error": abs_error,
            "max_error": float(np.max(abs_error)),
            "rms_error": float(np.sqrt(np.mean(abs_error**2))),
        }
    )
    return report


__all__ = [
    "qsvt_scalar_output",
    "qsvt_scalar_scan",
    "qsvt_diagonal_transform",
    "classical_diagonal_polynomial_transform",
    "compare_qsvt_vs_classical_diagonal",
    "qsvt_transform_report",
]
