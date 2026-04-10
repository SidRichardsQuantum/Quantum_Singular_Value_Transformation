"""
qsvt.spectral
-------------

Spectral matrix-function utilities for small Hermitian test problems.

This module provides lightweight helpers for applying scalar functions to the
spectrum of a Hermitian matrix. The core viewpoint is:

    A = V diag(lambda) V^T
    f(A) = V diag(f(lambda)) V^T

These routines are intended for:

- classical reference calculations for QSVT demos
- validating spectral transforms on small matrices
- illustrating matrix functions such as powers, roots, filters, and sign maps

All routines assume real symmetric / Hermitian inputs and return NumPy arrays.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np


def eigh_hermitian(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the eigendecomposition of a Hermitian matrix.

    Parameters
    ----------
    matrix
        Square Hermitian matrix.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        `(eigenvalues, eigenvectors)` as returned by `numpy.linalg.eigh`,
        with eigenvectors stored column-wise.

    Raises
    ------
    ValueError
        If the input is not square or not Hermitian within numerical tolerance.
    """
    A = np.asarray(matrix, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("matrix must be a square 2D array.")

    if not np.allclose(A, A.T, atol=1e-10, rtol=1e-10):
        raise ValueError("matrix must be Hermitian / symmetric.")

    evals, evecs = np.linalg.eigh(A)
    return evals, evecs


def matrix_from_eigendecomposition(
    eigenvalues: Iterable[float],
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct a Hermitian matrix from spectral data.

    Parameters
    ----------
    eigenvalues
        Eigenvalues of the matrix.
    eigenvectors
        Matrix whose columns are orthonormal eigenvectors.

    Returns
    -------
    numpy.ndarray
        Reconstructed matrix.

    Raises
    ------
    ValueError
        If dimensions are inconsistent.
    """
    vals = np.asarray(list(eigenvalues), dtype=float)
    vecs = np.asarray(eigenvectors, dtype=float)

    if vecs.ndim != 2 or vecs.shape[0] != vecs.shape[1]:
        raise ValueError("eigenvectors must be a square 2D array.")

    if len(vals) != vecs.shape[0]:
        raise ValueError("Mismatch between number of eigenvalues and matrix size.")

    return vecs @ np.diag(vals) @ vecs.T


def apply_function_to_hermitian(
    matrix: np.ndarray,
    func: Callable[[np.ndarray], np.ndarray | float],
) -> np.ndarray:
    """
    Apply a scalar function spectrally to a Hermitian matrix.

    If

        A = V diag(lambda) V^T,

    then this routine returns

        f(A) = V diag(f(lambda)) V^T.

    Parameters
    ----------
    matrix
        Square Hermitian matrix.
    func
        Scalar function applied elementwise to the eigenvalues.
        It should accept a NumPy array or be NumPy-broadcast compatible.

    Returns
    -------
    numpy.ndarray
        The matrix function f(A).

    Examples
    --------
    >>> A = np.diag([0.2, 0.8])
    >>> apply_function_to_hermitian(A, lambda x: x**2)
    array([[0.04, 0.  ],
           [0.  , 0.64]])
    """
    evals, evecs = eigh_hermitian(matrix)
    transformed = np.asarray(func(evals), dtype=float)

    if transformed.shape != evals.shape:
        raise ValueError("func must return values with the same shape as eigenvalues.")

    return evecs @ np.diag(transformed) @ evecs.T


def apply_polynomial_to_hermitian(
    matrix: np.ndarray,
    coeffs: Iterable[float],
) -> np.ndarray:
    """
    Apply a monomial-basis polynomial to a Hermitian matrix spectrally.

    Coefficients are interpreted in ascending degree order:

        coeffs = [c0, c1, ..., cn]
        P(x) = c0 + c1 x + ... + cn x^n

    Parameters
    ----------
    matrix
        Square Hermitian matrix.
    coeffs
        Polynomial coefficients in ascending degree order.

    Returns
    -------
    numpy.ndarray
        P(A) computed via the eigendecomposition.
    """
    coeffs_arr = np.asarray(list(coeffs), dtype=float)

    def poly(x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=float)
        for degree, coeff in enumerate(coeffs_arr):
            if coeff != 0.0:
                out = out + coeff * x**degree
        return out

    return apply_function_to_hermitian(matrix, poly)


def matrix_power_spectral(matrix: np.ndarray, power: int) -> np.ndarray:
    """
    Compute an integer power of a Hermitian matrix via the spectrum.

    Parameters
    ----------
    matrix
        Square Hermitian matrix.
    power
        Integer power. May be zero or positive.

    Returns
    -------
    numpy.ndarray
        A^power computed spectrally.

    Raises
    ------
    ValueError
        If power is negative.
    """
    if power < 0:
        raise ValueError("power must be non-negative.")

    return apply_function_to_hermitian(matrix, lambda x: x**power)


def matrix_fractional_power(
    matrix: np.ndarray,
    power: float,
    *,
    require_nonnegative_spectrum: bool = True,
) -> np.ndarray:
    """
    Compute a fractional matrix power A^power spectrally.

    Parameters
    ----------
    matrix
        Square Hermitian matrix.
    power
        Real exponent.
    require_nonnegative_spectrum
        If True, raise an error when the matrix has negative eigenvalues.
        This is the safest default for real-valued outputs.

    Returns
    -------
    numpy.ndarray
        Fractional power of the matrix.

    Raises
    ------
    ValueError
        If negative eigenvalues are present while
        `require_nonnegative_spectrum=True`.
    """
    evals, evecs = eigh_hermitian(matrix)

    if require_nonnegative_spectrum and np.any(evals < -1e-12):
        raise ValueError(
            "matrix has negative eigenvalues; fractional power would not be real."
        )

    transformed = np.power(np.clip(evals, 0.0, None), power)
    return evecs @ np.diag(transformed) @ evecs.T


def matrix_square_root(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the principal square root of a positive semidefinite Hermitian matrix.

    Parameters
    ----------
    matrix
        Square Hermitian positive semidefinite matrix.

    Returns
    -------
    numpy.ndarray
        Principal square root of the matrix.
    """
    return matrix_fractional_power(matrix, 0.5, require_nonnegative_spectrum=True)


def matrix_sign(matrix: np.ndarray, zero_tol: float = 1e-12) -> np.ndarray:
    """
    Compute the matrix sign function spectrally.

    For eigenvalues lambda_i, the sign map used is:

    - +1 for lambda_i > zero_tol
    - -1 for lambda_i < -zero_tol
    -  0 otherwise

    Parameters
    ----------
    matrix
        Square Hermitian matrix.
    zero_tol
        Numerical tolerance for treating eigenvalues as zero.

    Returns
    -------
    numpy.ndarray
        sign(A).
    """
    evals, evecs = eigh_hermitian(matrix)

    signed = np.zeros_like(evals)
    signed[evals > zero_tol] = 1.0
    signed[evals < -zero_tol] = -1.0

    return evecs @ np.diag(signed) @ evecs.T


def spectral_projector_positive(
    matrix: np.ndarray,
    *,
    zero_tol: float = 1e-12,
) -> np.ndarray:
    """
    Construct the projector onto the positive-eigenvalue subspace.

    Uses the exact spectral projector obtained from the eigendecomposition.

    Parameters
    ----------
    matrix
        Square Hermitian matrix.
    zero_tol
        Eigenvalues greater than zero_tol are treated as positive.

    Returns
    -------
    numpy.ndarray
        Projector onto the positive-eigenvalue subspace.
    """
    evals, evecs = eigh_hermitian(matrix)
    mask = (evals > zero_tol).astype(float)
    return evecs @ np.diag(mask) @ evecs.T


def spectral_projector_negative(
    matrix: np.ndarray,
    *,
    zero_tol: float = 1e-12,
) -> np.ndarray:
    """
    Construct the projector onto the negative-eigenvalue subspace.

    Parameters
    ----------
    matrix
        Square Hermitian matrix.
    zero_tol
        Eigenvalues less than -zero_tol are treated as negative.

    Returns
    -------
    numpy.ndarray
        Projector onto the negative-eigenvalue subspace.
    """
    evals, evecs = eigh_hermitian(matrix)
    mask = (evals < -zero_tol).astype(float)
    return evecs @ np.diag(mask) @ evecs.T


def positive_projector_from_sign(
    matrix: np.ndarray,
    *,
    zero_tol: float = 1e-12,
) -> np.ndarray:
    """
    Construct the positive spectral projector using the sign function identity.

        Pi_+ = (I + sign(A)) / 2

    This expression is exact when the matrix has no zero eigenvalues. If zero
    eigenvalues are present, the zero-eigenspace contributes 1/2 in this form,
    so the exact eigenspace projector is better obtained with
    `spectral_projector_positive`.

    Parameters
    ----------
    matrix
        Square Hermitian matrix.
    zero_tol
        Numerical tolerance used in the sign function.

    Returns
    -------
    numpy.ndarray
        Sign-based positive projector surrogate.
    """
    A = np.asarray(matrix, dtype=float)
    sign_A = matrix_sign(A, zero_tol=zero_tol)
    return 0.5 * (np.eye(A.shape[0], dtype=float) + sign_A)


def negative_projector_from_sign(
    matrix: np.ndarray,
    *,
    zero_tol: float = 1e-12,
) -> np.ndarray:
    """
    Construct the negative spectral projector using the sign function identity.

        Pi_- = (I - sign(A)) / 2

    Parameters
    ----------
    matrix
        Square Hermitian matrix.
    zero_tol
        Numerical tolerance used in the sign function.

    Returns
    -------
    numpy.ndarray
        Sign-based negative projector surrogate.
    """
    A = np.asarray(matrix, dtype=float)
    sign_A = matrix_sign(A, zero_tol=zero_tol)
    return 0.5 * (np.eye(A.shape[0], dtype=float) - sign_A)


def transformed_eigenvalues(
    matrix: np.ndarray,
    func: Callable[[np.ndarray], np.ndarray | float],
) -> np.ndarray:
    """
    Apply a scalar function directly to the eigenvalues of a Hermitian matrix.

    Parameters
    ----------
    matrix
        Square Hermitian matrix.
    func
        Scalar function applied to the eigenvalues.

    Returns
    -------
    numpy.ndarray
        Transformed eigenvalues.
    """
    evals, _ = eigh_hermitian(matrix)
    transformed = np.asarray(func(evals), dtype=float)

    if transformed.shape != evals.shape:
        raise ValueError("func must return values with the same shape as eigenvalues.")

    return transformed


__all__ = [
    "eigh_hermitian",
    "matrix_from_eigendecomposition",
    "apply_function_to_hermitian",
    "apply_polynomial_to_hermitian",
    "matrix_power_spectral",
    "matrix_fractional_power",
    "matrix_square_root",
    "matrix_sign",
    "spectral_projector_positive",
    "spectral_projector_negative",
    "positive_projector_from_sign",
    "negative_projector_from_sign",
    "transformed_eigenvalues",
]
