"""
Finite block-encoding helpers.

The routines in this module construct explicit dense block encodings for
small matrices. They are useful for validating QSVT algorithm structure on
finite instances: the encoded unitary is real quantum data for the supplied
matrix, while scalability, sparse-oracle access, and state preparation remain
separate modeling assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BlockEncoding:
    """
    Explicit dense block encoding of a finite matrix.

    The top-left logical block of ``unitary`` is ``operator / alpha``.
    """

    operator: np.ndarray
    alpha: float
    signal_operator: np.ndarray
    unitary: np.ndarray
    method: str = "unitary-dilation"

    @property
    def logical_dimension(self) -> int:
        """
        Dimension of the encoded logical operator.
        """
        return int(self.operator.shape[0])

    @property
    def unitary_dimension(self) -> int:
        """
        Dimension of the dense block-encoding unitary.
        """
        return int(self.unitary.shape[0])

    @property
    def ancilla_dimension(self) -> int:
        """
        Dimension multiplier introduced by the block encoding.
        """
        return int(self.unitary_dimension // self.logical_dimension)

    def top_left_block(self) -> np.ndarray:
        """
        Return the logical top-left block of the encoded unitary.
        """
        n = self.logical_dimension
        return self.unitary[:n, :n]

    def reconstruction(self) -> np.ndarray:
        """
        Reconstruct the encoded operator from the top-left block.
        """
        return self.alpha * self.top_left_block()

    def block_error(self) -> float:
        """
        Frobenius error between the requested operator and encoded block.
        """
        return float(np.linalg.norm(self.reconstruction() - self.operator))

    def unitarity_error(self) -> float:
        """
        Frobenius error in ``U^dagger U = I`` for the dense unitary.
        """
        ident = np.eye(self.unitary_dimension, dtype=complex)
        return float(np.linalg.norm(self.unitary.conj().T @ self.unitary - ident))

    def as_report(self) -> dict[str, object]:
        """
        Return a report dictionary for JSON conversion or persistence.
        """
        return {
            "mode": "block-encoding-report",
            "method": self.method,
            "operator": self.operator,
            "alpha": self.alpha,
            "signal_operator": self.signal_operator,
            "unitary": self.unitary,
            "logical_dimension": self.logical_dimension,
            "unitary_dimension": self.unitary_dimension,
            "ancilla_dimension": self.ancilla_dimension,
            "top_left_block": self.top_left_block(),
            "reconstruction": self.reconstruction(),
            "block_error": self.block_error(),
            "unitarity_error": self.unitarity_error(),
        }


def block_encode_matrix(
    operator: np.ndarray,
    *,
    alpha: float | None = None,
    atol: float = 1e-12,
) -> BlockEncoding:
    """
    Construct a dense unitary block encoding for a finite matrix.

    The construction uses the standard unitary dilation for a contraction
    ``B = operator / alpha``:

    ``[[B, sqrt(I - B B^dagger)], [sqrt(I - B^dagger B), -B^dagger]]``.

    Parameters
    ----------
    operator
        Square finite matrix to encode.
    alpha
        Positive normalization satisfying ``alpha >= ||operator||_2``. If
        omitted, the spectral norm is used, with ``alpha=1`` for the zero
        matrix.
    atol
        Numerical tolerance used for contraction validation.
    """
    A = _validate_square_matrix(operator)
    norm = float(np.linalg.norm(A, ord=2))

    if alpha is None:
        alpha = norm if norm > 0.0 else 1.0
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be positive and finite.")
    if norm > alpha + atol:
        raise ValueError("alpha must be at least the spectral norm of operator.")

    B = A / alpha
    left = _hermitian_psd_sqrt(np.eye(A.shape[0], dtype=complex) - B @ B.conj().T)
    right = _hermitian_psd_sqrt(np.eye(A.shape[0], dtype=complex) - B.conj().T @ B)
    unitary = np.block([[B, left], [right, -B.conj().T]])

    return BlockEncoding(
        operator=np.real_if_close(A),
        alpha=alpha,
        signal_operator=np.real_if_close(B),
        unitary=np.real_if_close(unitary),
    )


def block_encoding_report(
    operator: np.ndarray,
    *,
    alpha: float | None = None,
    atol: float = 1e-12,
) -> dict[str, object]:
    """
    Build a report for a dense matrix block encoding.
    """
    return block_encode_matrix(operator, alpha=alpha, atol=atol).as_report()


def extract_block_encoded_operator(
    unitary: np.ndarray,
    logical_dimension: int,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Extract ``alpha`` times the top-left logical block from an encoding unitary.
    """
    U = _validate_square_matrix(unitary)
    n = int(logical_dimension)
    if n <= 0:
        raise ValueError("logical_dimension must be positive.")
    if U.shape[0] < n:
        raise ValueError("logical_dimension cannot exceed unitary dimension.")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be positive and finite.")
    return np.real_if_close(float(alpha) * U[:n, :n])


def verify_block_encoding(
    encoding: BlockEncoding,
    *,
    block_atol: float = 1e-10,
    unitary_atol: float = 1e-10,
) -> dict[str, object]:
    """
    Verify the block and unitarity errors for a finite block encoding.
    """
    block_error = encoding.block_error()
    unitarity_error = encoding.unitarity_error()
    return {
        "mode": "block-encoding-verification",
        "method": encoding.method,
        "alpha": encoding.alpha,
        "logical_dimension": encoding.logical_dimension,
        "unitary_dimension": encoding.unitary_dimension,
        "ancilla_dimension": encoding.ancilla_dimension,
        "block_error": block_error,
        "unitarity_error": unitarity_error,
        "block_atol": float(block_atol),
        "unitary_atol": float(unitary_atol),
        "block_encoding_verified": bool(block_error <= block_atol),
        "unitary_verified": bool(unitarity_error <= unitary_atol),
    }


def _validate_square_matrix(operator: np.ndarray) -> np.ndarray:
    A = np.asarray(operator)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("operator must be a square 2D matrix.")
    if not np.all(np.isfinite(A)):
        raise ValueError("operator entries must be finite.")
    dtype = complex if np.iscomplexobj(A) else float
    return A.astype(dtype, copy=False)


def _hermitian_psd_sqrt(matrix: np.ndarray, *, atol: float = 1e-12) -> np.ndarray:
    H = 0.5 * (matrix + matrix.conj().T)
    evals, evecs = np.linalg.eigh(H)
    if np.min(evals) < -atol:
        raise ValueError("matrix is not positive semidefinite within tolerance.")
    clipped = np.clip(evals, 0.0, None)
    return (evecs * np.sqrt(clipped)) @ evecs.conj().T


__all__ = [
    "BlockEncoding",
    "block_encode_matrix",
    "block_encoding_report",
    "extract_block_encoded_operator",
    "verify_block_encoding",
]
