"""
qsvt.qsvt
---------

Thin PennyLane-facing helpers for Quantum Singular Value Transformation (QSVT)
and scalar-QSP-style examples.

This module wraps the repeated notebook pattern:

- build a QSVT operator with `qml.qsvt(...)`
- convert it to a matrix with `qml.matrix(...)`
- extract the logical top-left block or top-left scalar entry

The goal is not to hide PennyLane, but to provide a small, stable utility
layer for:

- scalar QSVT / QSP demonstrations
- diagonal-matrix singular-value transforms
- block extraction for small explicit examples
- simple classical-vs-QSVT validation workflows

Notes
-----
These helpers assume the "embedding" block-encoding mode used throughout the
notebooks unless explicitly overridden.

Polynomial coefficients are assumed to be in ascending degree order:

    coeffs = [c0, c1, ..., cn]
    P(x) = c0 + c1 x + ... + cn x^n
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pennylane as qml

from .polynomials import eval_polynomial, polynomial_parity
from .spectral import apply_polynomial_to_hermitian, eigh_hermitian


def _as_numeric_operator(operator: float | np.ndarray) -> float | np.ndarray:
    """
    Normalize a scalar or matrix operator into a numeric form suitable for QSVT.

    Parameters
    ----------
    operator
        Scalar or square matrix.

    Returns
    -------
    float or numpy.ndarray
        Normalized scalar or matrix representation.

    Raises
    ------
    ValueError
        If a matrix input is not square.
    """
    if np.isscalar(operator):
        return float(operator)

    arr = np.asarray(operator, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("operator must be a scalar or a square 2D array.")

    return arr


def _default_wire_order_from_operator(operator: float | np.ndarray) -> list[int]:
    """
    Choose a simple default wire order for a scalar or matrix operator.

    For scalar examples we use one wire.
    For an n x n matrix in PennyLane's embedding mode, we choose the smallest
    number of qubits such that 2**num_wires >= 2*n.

    Parameters
    ----------
    operator
        Scalar or square matrix.

    Returns
    -------
    list[int]
        Default wire order.
    """
    if np.isscalar(operator):
        return [0]

    n = np.asarray(operator).shape[0]
    num_wires = max(1, int(np.ceil(np.log2(2 * n))))
    return list(range(num_wires))


def _validate_wire_inputs(
    encoding_wires: Iterable[int],
    wire_order: Iterable[int] | None,
) -> tuple[list[int], list[int]]:
    """
    Validate and normalize wire inputs.

    Parameters
    ----------
    encoding_wires
        Wires used for the QSVT encoding.
    wire_order
        Wires passed to `qml.matrix`. If omitted, defaults to encoding_wires.

    Returns
    -------
    tuple[list[int], list[int]]
        `(encoding_wires, wire_order)` as concrete lists.

    Raises
    ------
    ValueError
        If no encoding wires are provided.
    """
    enc = list(encoding_wires)
    if not enc:
        raise ValueError("encoding_wires must contain at least one wire.")

    order = list(wire_order) if wire_order is not None else enc

    return enc, order


def qsvt_operator(
    operator: float | np.ndarray,
    poly: Iterable[float],
    *,
    encoding_wires: Iterable[int] | None = None,
    block_encoding: str = "embedding",
):
    """
    Build a PennyLane QSVT operator.

    Parameters
    ----------
    operator
        Scalar or square matrix to transform.
    poly
        Polynomial coefficients in ascending degree order.
    encoding_wires
        Wires used for the QSVT encoding. If omitted, a simple default is used.
    block_encoding
        PennyLane block-encoding mode. Defaults to "embedding".

    Returns
    -------
    pennylane.operation.Operator
        QSVT operator object returned by `qml.qsvt`.

    Examples
    --------
    >>> op = qsvt_operator(0.5, [0, 0, 1], encoding_wires=[0])
    """
    op_in = _as_numeric_operator(operator)
    coeffs = np.asarray(list(poly), dtype=float)

    if encoding_wires is None:
        encoding_wires = _default_wire_order_from_operator(op_in)

    enc, _ = _validate_wire_inputs(encoding_wires, None)

    return qml.qsvt(
        op_in,
        coeffs,
        encoding_wires=enc,
        block_encoding=block_encoding,
    )


def qsvt_unitary(
    operator: float | np.ndarray,
    poly: Iterable[float],
    *,
    encoding_wires: Iterable[int] | None = None,
    wire_order: Iterable[int] | None = None,
    block_encoding: str = "embedding",
) -> np.ndarray:
    """
    Compute the explicit matrix of a QSVT transform.

    Parameters
    ----------
    operator
        Scalar or square matrix to transform.
    poly
        Polynomial coefficients in ascending degree order.
    encoding_wires
        Wires used for the QSVT encoding. If omitted, a simple default is used.
    wire_order
        Wire order used when extracting the matrix via `qml.matrix`.
        If omitted, defaults to `encoding_wires`.
    block_encoding
        PennyLane block-encoding mode. Defaults to "embedding".

    Returns
    -------
    numpy.ndarray
        Full unitary matrix representation of the QSVT operator.

    Examples
    --------
    >>> U = qsvt_unitary(0.5, [0, 0, 1], encoding_wires=[0])
    >>> U.shape
    (2, 2)
    """
    op_in = _as_numeric_operator(operator)
    coeffs = np.asarray(list(poly), dtype=float)

    if encoding_wires is None:
        encoding_wires = _default_wire_order_from_operator(op_in)

    enc, order = _validate_wire_inputs(encoding_wires, wire_order)

    return np.asarray(
        qml.matrix(
            qml.qsvt(
                op_in,
                coeffs,
                encoding_wires=enc,
                block_encoding=block_encoding,
            ),
            wire_order=order,
        ),
        dtype=complex,
    )


def qsvt_top_left_block(
    operator: np.ndarray,
    poly: Iterable[float],
    *,
    encoding_wires: Iterable[int] | None = None,
    wire_order: Iterable[int] | None = None,
    block_encoding: str = "embedding",
) -> np.ndarray:
    """
    Extract the logical top-left block from a matrix QSVT unitary.

    If the input operator is n x n, this returns the top-left n x n block
    of the full QSVT unitary.

    Parameters
    ----------
    operator
        Square matrix to transform.
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
        Top-left logical block.

    Raises
    ------
    ValueError
        If the input is scalar. Use `qsvt_scalar_output` for scalar inputs.

    Examples
    --------
    >>> A = np.diag([1.0, 0.7, 0.3, 0.1])
    >>> block = qsvt_top_left_block(A, [0, 0, 1], encoding_wires=[0, 1, 2])
    >>> block.shape
    (4, 4)
    """
    if np.isscalar(operator):
        raise ValueError(
            "qsvt_top_left_block expects a matrix input. "
            "Use qsvt_scalar_output for scalar inputs."
        )

    A = _as_numeric_operator(operator)
    n = A.shape[0]
    U = qsvt_unitary(
        A,
        poly,
        encoding_wires=encoding_wires,
        wire_order=wire_order,
        block_encoding=block_encoding,
    )
    return U[:n, :n]


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
    real part, which is the matrix compared against the classical spectral
    polynomial reference in this package's reports.

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


def apply_qsvt_to_embedded_vector(
    operator: np.ndarray,
    vector: Iterable[float | complex],
    poly: Iterable[float],
    *,
    encoding_wires: Iterable[int] | None = None,
    wire_order: Iterable[int] | None = None,
    block_encoding: str = "embedding",
    normalize_output: bool = False,
) -> np.ndarray:
    """
    Embed a logical vector, apply the full QSVT unitary, and extract the
    logical output components.

    If the input operator is n x n, the logical state is assumed to occupy
    the first n amplitudes of the enlarged Hilbert space.

    Parameters
    ----------
    operator
        Square matrix to transform.
    vector
        Logical input vector of length n.
    poly
        Polynomial coefficients in ascending degree order.
    encoding_wires
        Wires used for the QSVT encoding.
    wire_order
        Wire order passed to `qml.matrix`.
    block_encoding
        PennyLane block-encoding mode.
    normalize_output
        If True, normalize the extracted logical output.

    Returns
    -------
    numpy.ndarray
        Logical output vector extracted from the enlarged Hilbert-space action.

    Raises
    ------
    ValueError
        If the vector length does not match the operator dimension.
    """
    A = _as_numeric_operator(operator)
    if np.isscalar(A):
        raise ValueError(
            "apply_qsvt_to_embedded_vector expects a matrix operator, not a scalar."
        )

    n = A.shape[0]
    vec = np.asarray(list(vector), dtype=complex)

    if vec.shape != (n,):
        raise ValueError("vector length must match the operator dimension.")

    U = qsvt_unitary(
        A,
        poly,
        encoding_wires=encoding_wires,
        wire_order=wire_order,
        block_encoding=block_encoding,
    )

    full_dim = U.shape[0]
    embedded = np.zeros(full_dim, dtype=complex)
    embedded[:n] = vec

    out = U @ embedded
    logical = out[:n]

    if normalize_output:
        norm = np.linalg.norm(logical)
        if norm == 0.0:
            raise ValueError("Cannot normalize a zero output vector.")
        logical = logical / norm

    return logical


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


def compare_qsvt_vs_classical_matrix(
    operator: np.ndarray,
    poly: Iterable[float],
    *,
    encoding_wires: Iterable[int] | None = None,
    wire_order: Iterable[int] | None = None,
    block_encoding: str = "embedding",
) -> dict[str, np.ndarray]:
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
    dict[str, numpy.ndarray]
        Dictionary with keys:
        - "input"
        - "qsvt"
        - "qsvt_imag"
        - "classical"
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
    qsvt_real = np.real(block).astype(float)
    classical = apply_polynomial_to_hermitian(A, poly)

    return {
        "input": A,
        "qsvt": qsvt_real,
        "qsvt_imag": np.imag(block).astype(float),
        "classical": classical,
        "abs_error": np.abs(qsvt_real - classical),
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
            "qsvt_succeeded": True,
            "qsvt": qsvt_vals,
            "abs_error": abs_error,
            "max_error": float(np.max(abs_error)),
            "rms_error": float(np.sqrt(np.mean(abs_error**2))),
        }
    )
    return report


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

    qsvt_real = np.real(block).astype(float)
    qsvt_imag = np.imag(block).astype(float)
    abs_error = np.abs(qsvt_real - classical)
    report.update(
        {
            "qsvt_succeeded": True,
            "qsvt": qsvt_real,
            "qsvt_imag": qsvt_imag,
            "abs_error": abs_error,
            "max_error": float(np.max(abs_error)),
            "rms_error": float(np.sqrt(np.mean(abs_error**2))),
            "frobenius_error": float(np.linalg.norm(qsvt_real - classical)),
            "max_imag_abs": float(np.max(np.abs(qsvt_imag))),
        }
    )
    return report


def qsvt_compatibility_report(
    poly: Iterable[float],
    *,
    bounded_domain: tuple[float, float] = (-1.0, 1.0),
    bounded_num_points: int = 4001,
    bound: float = 1.0,
    parity_tol: float = 1e-10,
    attempt_synthesis: bool = True,
    block_encoding: str = "embedding",
) -> dict[str, object]:
    """
    Check whether polynomial coefficients are suitable for PennyLane QSVT.

    This report separates lightweight structural checks from PennyLane's phase
    synthesis attempt. A bounded polynomial can still fail synthesis if it does
    not meet parity or solver-specific requirements.

    Parameters
    ----------
    poly
        Polynomial coefficients in ascending degree order.
    bounded_domain
        Domain used for sampled boundedness checks.
    bounded_num_points
        Number of grid points used for sampled boundedness checks.
    bound
        Absolute-value bound expected by QSVT.
    parity_tol
        Tolerance used for parity classification.
    attempt_synthesis
        If True, attempt PennyLane QSVT synthesis on a scalar input.
    block_encoding
        PennyLane block-encoding mode used for the synthesis attempt.

    Returns
    -------
    dict[str, object]
        Compatibility report with reasons for any failed checks.
    """
    coeffs = np.asarray(list(poly), dtype=float)

    if coeffs.ndim != 1 or coeffs.size == 0:
        raise ValueError("poly must contain at least one coefficient.")
    if bounded_num_points < 2:
        raise ValueError("bounded_num_points must be at least 2.")
    lower, upper = float(bounded_domain[0]), float(bounded_domain[1])
    if upper <= lower:
        raise ValueError("bounded_domain must satisfy lower < upper.")

    coeffs_finite = bool(np.all(np.isfinite(coeffs)))
    reasons: list[str] = []
    if not coeffs_finite:
        reasons.append("non_finite_coefficients")

    parity = polynomial_parity(coeffs, tol=parity_tol)
    has_definite_parity = parity in {"even", "odd", "zero"}
    if not has_definite_parity:
        reasons.append("mixed_parity")

    if coeffs_finite:
        xs = np.linspace(lower, upper, int(bounded_num_points))
        values = np.asarray(eval_polynomial(coeffs, xs), dtype=float)
        max_abs_value: float | None = float(np.max(np.abs(values)))
        bounded_margin: float | None = float(bound - max_abs_value)
        is_bounded = bool(max_abs_value <= float(bound) + 1e-12)
    else:
        max_abs_value = None
        bounded_margin = None
        is_bounded = False

    if not is_bounded:
        reasons.append("out_of_bounds")

    report: dict[str, object] = {
        "mode": "qsvt-compatibility-report",
        "poly": coeffs,
        "polynomial_degree": int(coeffs.size - 1),
        "coefficients_finite": coeffs_finite,
        "parity": parity,
        "has_definite_parity": has_definite_parity,
        "bounded_domain": (lower, upper),
        "bounded_num_points": int(bounded_num_points),
        "bound": float(bound),
        "max_abs_value": max_abs_value,
        "bounded_margin": bounded_margin,
        "is_bounded": is_bounded,
        "attempted_pennylane_synthesis": bool(attempt_synthesis),
    }

    synthesis_succeeded: bool | None = None
    if attempt_synthesis and coeffs_finite:
        try:
            qml.qsvt(
                0.5,
                coeffs,
                encoding_wires=[0],
                block_encoding=block_encoding,
            )
        except Exception as exc:
            synthesis_succeeded = False
            reasons.append("synthesis_failed")
            report.update(
                {
                    "pennylane_synthesis_succeeded": False,
                    "pennylane_error_type": type(exc).__name__,
                    "pennylane_error": str(exc),
                }
            )
        else:
            synthesis_succeeded = True
            report["pennylane_synthesis_succeeded"] = True
    else:
        report["pennylane_synthesis_succeeded"] = synthesis_succeeded

    report["reasons"] = reasons
    report["compatible"] = bool(
        coeffs_finite
        and has_definite_parity
        and is_bounded
        and (synthesis_succeeded is not False)
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


__all__ = [
    "qsvt_operator",
    "qsvt_unitary",
    "qsvt_top_left_block",
    "qsvt_scalar_output",
    "qsvt_scalar_scan",
    "qsvt_diagonal_transform",
    "qsvt_matrix_transform",
    "apply_qsvt_to_embedded_vector",
    "classical_diagonal_polynomial_transform",
    "compare_qsvt_vs_classical_diagonal",
    "compare_qsvt_vs_classical_matrix",
    "qsvt_compatibility_report",
    "qsvt_transform_report",
    "qsvt_matrix_transform_report",
]
