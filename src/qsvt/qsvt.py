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

from .polynomials import eval_polynomial


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
    For an n x n matrix, we choose the smallest number of qubits such that
    2**num_wires >= n.

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
    num_wires = max(1, int(np.ceil(np.log2(n))))
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
    >>> vals = qsvt_diagonal_transform([1.0, 0.7, 0.3, 0.1], [0, 0, 1], encoding_wires=[0,1,2])
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


__all__ = [
    "qsvt_operator",
    "qsvt_unitary",
    "qsvt_top_left_block",
    "qsvt_scalar_output",
    "qsvt_scalar_scan",
    "qsvt_diagonal_transform",
    "apply_qsvt_to_embedded_vector",
    "classical_diagonal_polynomial_transform",
    "compare_qsvt_vs_classical_diagonal",
]
