"""
qsvt.matrices
-------------

Small matrix construction utilities used across QSVT demonstrations.

These helpers provide simple, explicit Hermitian test matrices whose spectra
are easy to understand analytically. They are intentionally lightweight and
designed for:

- pedagogical examples
- reproducible small test cases
- spectral intuition building
- validating polynomial transforms before QSVT use

All matrices are NumPy arrays with dtype=float.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def diagonal_matrix(values: Iterable[float]) -> np.ndarray:
    """
    Construct a diagonal matrix from eigenvalues.

    Parameters
    ----------
    values
        Iterable of diagonal entries (eigenvalues).

    Returns
    -------
    numpy.ndarray
        Square diagonal matrix.

    Examples
    --------
    >>> diagonal_matrix([1.0, -1.0])
    array([[ 1.,  0.],
           [ 0., -1.]])
    """
    vals = np.asarray(list(values), dtype=float)
    return np.diag(vals)


def identity(n: int) -> np.ndarray:
    """
    Construct an identity matrix.

    Parameters
    ----------
    n
        Matrix dimension.

    Returns
    -------
    numpy.ndarray
        n x n identity matrix.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    return np.eye(n, dtype=float)


def pauli_x() -> np.ndarray:
    """
    Pauli X matrix.

    Returns
    -------
    numpy.ndarray
        2x2 Pauli-X operator.

    Examples
    --------
    >>> pauli_x()
    array([[0., 1.],
           [1., 0.]])
    """
    return np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )


def pauli_z() -> np.ndarray:
    """
    Pauli Z matrix.

    Returns
    -------
    numpy.ndarray
        2x2 Pauli-Z operator.
    """
    return np.array(
        [
            [1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=float,
    )


def rotation(theta: float) -> np.ndarray:
    """
    2x2 real rotation matrix.

    This matrix is orthogonal and diagonalizes many simple
    symmetric 2x2 test Hamiltonians.

    R(theta) =
        [[ cos(theta), -sin(theta)],
         [ sin(theta),  cos(theta)]]

    Parameters
    ----------
    theta
        Rotation angle in radians.

    Returns
    -------
    numpy.ndarray
        2x2 orthogonal matrix.

    Examples
    --------
    >>> rotation(0.0)
    array([[ 1., -0.],
           [ 0.,  1.]])
    """
    c = float(np.cos(theta))
    s = float(np.sin(theta))

    return np.array(
        [
            [c, -s],
            [s, c],
        ],
        dtype=float,
    )


def rotated_diagonal(
    eigenvalues: Iterable[float],
    theta: float,
) -> np.ndarray:
    """
    Construct a symmetric matrix with known eigenvalues.

    A = R(theta) diag(eigenvalues) R(theta)^T

    Useful for generating Hermitian matrices with non-trivial
    eigenvectors but analytically known spectrum.

    Parameters
    ----------
    eigenvalues
        Iterable of eigenvalues.
    theta
        Rotation angle in radians.

    Returns
    -------
    numpy.ndarray
        Symmetric matrix with specified eigenvalues.

    Examples
    --------
    >>> A = rotated_diagonal([0.2, 0.8], 0.5)
    >>> np.linalg.eigvalsh(A)
    array([0.2, 0.8])
    """
    eigvals = np.asarray(list(eigenvalues), dtype=float)

    if eigvals.ndim != 1:
        raise ValueError("eigenvalues must be a 1D iterable.")

    R = rotation(theta)

    if len(eigvals) != 2:
        raise ValueError("rotated_diagonal currently supports 2x2 matrices only.")

    return R @ np.diag(eigvals) @ R.T


def hermitian_from_eigendecomposition(
    eigenvalues: Iterable[float],
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """
    Construct a Hermitian matrix from spectral components.

    A = V diag(lambda) V^T

    Parameters
    ----------
    eigenvalues
        Iterable of eigenvalues.
    eigenvectors
        Orthogonal matrix whose columns are eigenvectors.

    Returns
    -------
    numpy.ndarray
        Hermitian matrix.

    Notes
    -----
    eigenvectors should be orthonormal:

        V^T V = I
    """
    eigvals = np.asarray(list(eigenvalues), dtype=float)
    V = np.asarray(eigenvectors, dtype=float)

    if V.shape[0] != V.shape[1]:
        raise ValueError("eigenvectors must form a square matrix.")

    if len(eigvals) != V.shape[0]:
        raise ValueError("Mismatch between eigenvalues and eigenvector dimension.")

    return V @ np.diag(eigvals) @ V.T


def involutory_diagonal(sign_pattern: Iterable[int | float]) -> np.ndarray:
    """
    Construct a diagonal involutory matrix.

    An involutory matrix satisfies:

        A^2 = I

    This holds when diagonal entries are ±1.

    Parameters
    ----------
    sign_pattern
        Iterable of ±1 values.

    Returns
    -------
    numpy.ndarray
        Diagonal involutory matrix.

    Examples
    --------
    >>> involutory_diagonal([1, -1, 1])
    array([[ 1.,  0.,  0.],
           [ 0., -1.,  0.],
           [ 0.,  0.,  1.]])
    """
    vals = np.asarray(list(sign_pattern), dtype=float)

    if not np.all(np.isin(vals, [-1.0, 1.0])):
        raise ValueError("Entries must be ±1.")

    return np.diag(vals)


def normalized_vector(values: Iterable[float]) -> np.ndarray:
    """
    Normalize a real vector.

    Parameters
    ----------
    values
        Iterable of components.

    Returns
    -------
    numpy.ndarray
        Unit-norm vector.

    Raises
    ------
    ValueError
        If the vector has zero norm.
    """
    v = np.asarray(list(values), dtype=float)

    norm = np.linalg.norm(v)

    if norm == 0:
        raise ValueError("Cannot normalize zero vector.")

    return v / norm


def embed_vector(vec: Iterable[float], dimension: int) -> np.ndarray:
    """
    Embed a vector into a higher-dimensional Hilbert space.

    The vector occupies the leading entries and remaining components are zero.

    Parameters
    ----------
    vec
        Vector to embed.
    dimension
        Target dimension.

    Returns
    -------
    numpy.ndarray
        Embedded vector.

    Examples
    --------
    >>> embed_vector([1,2], 4)
    array([1., 2., 0., 0.])
    """
    v = np.asarray(list(vec), dtype=float)

    if dimension < len(v):
        raise ValueError("dimension must be >= len(vec).")

    out = np.zeros(dimension, dtype=float)
    out[: len(v)] = v

    return out


__all__ = [
    "diagonal_matrix",
    "identity",
    "pauli_x",
    "pauli_z",
    "rotation",
    "rotated_diagonal",
    "hermitian_from_eigendecomposition",
    "involutory_diagonal",
    "normalized_vector",
    "embed_vector",
]
