"""
Finite block-encoding helpers.

The routines in this module construct explicit dense block encodings for
small matrices. They are useful for validating QSVT algorithm structure on
finite instances: the encoded unitary is real quantum data for the supplied
matrix, while scalability, sparse-oracle access, and state preparation remain
separate modeling assumptions.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable, Literal, cast

import numpy as np
import pennylane as qml

BlockEncodingKind = Literal[
    "dense-matrix",
    "sparse-matrix",
    "pennylane-operator",
    "custom-circuit",
]


@dataclass(frozen=True)
class BlockEncodingSpec:
    """
    Research-facing description of a block encoding and its access model.

    A specification can represent dense or sparse matrices, PennyLane
    operators/Hamiltonians, and user-provided circuit factories. Representation
    does not imply that every backend can directly execute QSVT for that source.
    """

    kind: BlockEncodingKind
    source: object
    alpha: float
    logical_shape: tuple[int, int]
    encoding_wires: tuple[Any, ...]
    block_encoding: str
    execution_supported: bool
    execution_reason: str
    metadata: dict[str, object]

    @property
    def is_rectangular(self) -> bool:
        """Whether the logical operator is rectangular."""
        return self.logical_shape[0] != self.logical_shape[1]

    @property
    def is_matrix_source(self) -> bool:
        """Whether the source can be converted to a finite dense matrix."""
        return self.kind in {"dense-matrix", "sparse-matrix"}

    def dense_matrix(self) -> np.ndarray:
        """Return the finite matrix represented by this specification."""
        if not self.is_matrix_source:
            raise TypeError("this block-encoding specification is not matrix-backed.")
        return _as_dense_or_sparse_matrix(self.source)

    def as_report(self) -> dict[str, object]:
        """Return metadata without serializing an opaque operator or callable."""
        return {
            "mode": "block-encoding-specification",
            "kind": self.kind,
            "alpha": self.alpha,
            "logical_shape": self.logical_shape,
            "is_rectangular": self.is_rectangular,
            "encoding_wires": self.encoding_wires,
            "block_encoding": self.block_encoding,
            "execution_supported": self.execution_supported,
            "execution_reason": self.execution_reason,
            "source_type": type(self.source).__name__,
            "metadata": self.metadata,
            "truth_contract": {
                "represents_block_encoding_access_model": True,
                "is_verified_unitary": False,
                "is_scalable_implementation": False,
                "opaque_source_omitted_from_report": True,
            },
        }


def matrix_block_encoding_spec(
    operator: object,
    *,
    alpha: float | None = None,
    encoding_wires: tuple[Any, ...] | list[Any] | None = None,
    block_encoding: Literal["embedding", "fable"] = "embedding",
    metadata: dict[str, object] | None = None,
) -> BlockEncodingSpec:
    """
    Describe a dense or sparse matrix block encoding.

    Rectangular matrices are accepted because block encoding and singular-value
    transformation are naturally rectangular. PennyLane's high-level
    ``qml.qsvt`` matrix path currently requires Hermitian input, so rectangular
    specifications are representable but marked as not directly executable by
    :func:`qsvt_operator_from_block_encoding`.
    """
    matrix = _as_dense_or_sparse_matrix(operator)
    rows, columns = map(int, matrix.shape)
    norm = float(np.linalg.norm(matrix, ord=2))
    resolved_alpha = norm if alpha is None and norm > 0.0 else alpha
    if resolved_alpha is None:
        resolved_alpha = 1.0
    resolved_alpha = _validate_alpha(resolved_alpha, norm)

    if encoding_wires is None:
        if block_encoding == "embedding":
            wire_count = max(1, int(np.ceil(np.log2(rows + columns))))
        else:
            dimension = max(rows, columns)
            wire_count = max(1, 2 * int(np.ceil(np.log2(dimension))) + 1)
        encoding_wires = tuple(range(wire_count))
    wires = _validate_spec_wires(encoding_wires)

    sparse = not isinstance(operator, np.ndarray) and hasattr(operator, "toarray")
    hermitian = bool(
        rows == columns and np.allclose(matrix, matrix.conj().T, atol=1e-10)
    )
    normalized_matrix = matrix / resolved_alpha
    fable_condition_value = float(
        max(rows, columns) * np.linalg.norm(normalized_matrix, ord="fro") ** 2
    )
    fable_compatible = bool(fable_condition_value <= 1.0 + 1e-12)
    execution_supported = bool(
        rows == columns
        and hermitian
        and (block_encoding != "fable" or fable_compatible)
    )
    if rows != columns or not hermitian:
        execution_reason = (
            "Representable as a block encoding, but the package's high-level "
            "PennyLane QSVT adapter requires a square Hermitian matrix."
        )
    elif block_encoding == "fable" and not fable_compatible:
        execution_reason = (
            "The normalized matrix does not satisfy PennyLane's FABLE "
            "normalization condition."
        )
    else:
        execution_reason = (
            "Supported by PennyLane's high-level Hermitian matrix QSVT path."
        )
    return BlockEncodingSpec(
        kind="sparse-matrix" if sparse else "dense-matrix",
        source=operator,
        alpha=resolved_alpha,
        logical_shape=(rows, columns),
        encoding_wires=wires,
        block_encoding=block_encoding,
        execution_supported=execution_supported,
        execution_reason=execution_reason,
        metadata={
            "spectral_norm": norm,
            "normalized_operator_norm": norm / resolved_alpha,
            "hermitian": hermitian,
            "fable_condition_value": fable_condition_value,
            "fable_compatible": fable_compatible,
            **(metadata or {}),
        },
    )


def pennylane_operator_block_encoding_spec(
    operator: qml.operation.Operator,
    *,
    encoding_wires: tuple[Any, ...] | list[Any],
    block_encoding: Literal["prepselprep", "qubitization"] = "prepselprep",
    alpha: float | None = None,
    metadata: dict[str, object] | None = None,
) -> BlockEncodingSpec:
    """Describe a PennyLane operator/Hamiltonian block-encoding adapter."""
    if not isinstance(operator, qml.operation.Operator):
        raise TypeError("operator must be a PennyLane Operator.")
    wires = _validate_spec_wires(encoding_wires)
    inferred_alpha = _operator_lcu_normalization(operator)
    resolved_alpha = _validate_alpha(
        inferred_alpha if alpha is None else alpha,
    )
    system_wire_count = len(operator.wires)
    dimension = 2**system_wire_count
    return BlockEncodingSpec(
        kind="pennylane-operator",
        source=operator,
        alpha=resolved_alpha,
        logical_shape=(dimension, dimension),
        encoding_wires=wires,
        block_encoding=block_encoding,
        execution_supported=True,
        execution_reason="Supported by PennyLane's Hamiltonian QSVT path.",
        metadata={
            "operator_name": operator.name,
            "system_wires": tuple(operator.wires),
            "inferred_lcu_normalization": inferred_alpha,
            **(metadata or {}),
        },
    )


def circuit_block_encoding_spec(
    circuit_factory: Callable[[], qml.operation.Operator],
    *,
    logical_shape: tuple[int, int],
    encoding_wires: tuple[Any, ...] | list[Any],
    alpha: float = 1.0,
    metadata: dict[str, object] | None = None,
) -> BlockEncodingSpec:
    """
    Describe a user-provided block-encoding operation factory.

    The factory should return a PennyLane operation when called in a queuing
    context. Custom projector definitions are problem-specific, so this source
    is not passed through PennyLane's high-level ``qml.qsvt`` convenience API.
    """
    if not callable(circuit_factory):
        raise TypeError("circuit_factory must be callable.")
    rows, columns = _validate_logical_shape(logical_shape)
    wires = _validate_spec_wires(encoding_wires)
    return BlockEncodingSpec(
        kind="custom-circuit",
        source=circuit_factory,
        alpha=_validate_alpha(alpha),
        logical_shape=(rows, columns),
        encoding_wires=wires,
        block_encoding="custom",
        execution_supported=False,
        execution_reason=(
            "The custom block encoding can be queued, but QSVT projectors and "
            "signal subspaces must be supplied by the caller."
        ),
        metadata=dict(metadata or {}),
    )


def build_block_encoding_operator(spec: BlockEncodingSpec):
    """Build or queue the PennyLane block-encoding operation for a specification."""
    if spec.kind in {"dense-matrix", "sparse-matrix"}:
        matrix = spec.dense_matrix() / spec.alpha
        if spec.block_encoding == "embedding":
            return qml.BlockEncode(matrix, wires=spec.encoding_wires)
        if spec.block_encoding == "fable":
            return qml.FABLE(matrix, wires=spec.encoding_wires)
        raise ValueError("matrix block_encoding must be 'embedding' or 'fable'.")
    if spec.kind == "pennylane-operator":
        if spec.block_encoding == "prepselprep":
            return qml.PrepSelPrep(spec.source, control=spec.encoding_wires)
        if spec.block_encoding == "qubitization":
            return qml.Qubitization(spec.source, control=spec.encoding_wires)
        raise ValueError(
            "PennyLane operator block_encoding must be 'prepselprep' or 'qubitization'."
        )
    if spec.kind == "custom-circuit":
        factory = spec.source
        if not callable(factory):  # defensive guard for manually-built specs
            raise TypeError("custom-circuit source must be callable.")
        return factory()
    raise ValueError(f"unsupported block-encoding kind: {spec.kind}")


def qsvt_operator_from_block_encoding(
    spec: BlockEncodingSpec,
    poly: Iterable[float],
    *,
    angle_solver: str = "root-finding",
):
    """Build PennyLane's high-level QSVT operator from a supported specification."""
    if not spec.execution_supported:
        raise NotImplementedError(spec.execution_reason)
    source: Any
    if spec.kind in {"dense-matrix", "sparse-matrix"}:
        source = spec.dense_matrix() / spec.alpha
    elif spec.kind == "pennylane-operator":
        source = spec.source
    else:
        raise NotImplementedError(spec.execution_reason)
    return qml.qsvt(
        source,
        np.asarray(list(poly), dtype=float),
        encoding_wires=spec.encoding_wires,
        block_encoding=spec.block_encoding,
        angle_solver=angle_solver,
    )


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


def _as_dense_or_sparse_matrix(operator: object) -> np.ndarray:
    source = operator.toarray() if hasattr(operator, "toarray") else operator
    matrix = np.asarray(source)
    if matrix.ndim != 2 or min(matrix.shape) <= 0:
        raise ValueError("operator must be a non-empty 2D matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("operator entries must be finite.")
    dtype = complex if np.iscomplexobj(matrix) else float
    return matrix.astype(dtype, copy=False)


def _validate_alpha(alpha: float, norm: float | None = None) -> float:
    value = float(alpha)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("alpha must be positive and finite.")
    if norm is not None and norm > value + 1e-12:
        raise ValueError("alpha must be at least the spectral norm of operator.")
    return value


def _validate_spec_wires(wires: object) -> tuple[Any, ...]:
    normalized: tuple[Any, ...] = tuple(cast(Iterable[Any], wires))
    if not normalized:
        raise ValueError("encoding_wires must contain at least one wire.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("encoding_wires must be distinct.")
    return normalized


def _validate_logical_shape(shape: tuple[int, int]) -> tuple[int, int]:
    rows, columns = int(shape[0]), int(shape[1])
    if rows <= 0 or columns <= 0:
        raise ValueError("logical_shape must contain two positive dimensions.")
    return rows, columns


def _operator_lcu_normalization(operator: qml.operation.Operator) -> float:
    try:
        coefficients, _ = operator.terms()
    except (AttributeError, NotImplementedError):
        return 1.0
    values = np.asarray(coefficients, dtype=complex)
    normalization = float(np.sum(np.abs(values)))
    return normalization if normalization > 0.0 else 1.0


def _hermitian_psd_sqrt(matrix: np.ndarray, *, atol: float = 1e-12) -> np.ndarray:
    H = 0.5 * (matrix + matrix.conj().T)
    evals, evecs = np.linalg.eigh(H)
    if np.min(evals) < -atol:
        raise ValueError("matrix is not positive semidefinite within tolerance.")
    clipped = np.clip(evals, 0.0, None)
    return (evecs * np.sqrt(clipped)) @ evecs.conj().T


__all__ = [
    "BlockEncodingKind",
    "BlockEncodingSpec",
    "BlockEncoding",
    "build_block_encoding_operator",
    "block_encode_matrix",
    "block_encoding_report",
    "circuit_block_encoding_spec",
    "extract_block_encoded_operator",
    "matrix_block_encoding_spec",
    "pennylane_operator_block_encoding_spec",
    "qsvt_operator_from_block_encoding",
    "verify_block_encoding",
]
