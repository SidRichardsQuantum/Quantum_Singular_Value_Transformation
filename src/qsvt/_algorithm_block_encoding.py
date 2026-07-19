"""Block-encoded QSVT verification workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._algorithm_reports import algorithm_workflow_schema_fields
from ._algorithm_shared import (
    _relative_error,
    _validate_state,
)
from .block_encoding import BlockEncoding, block_encode_matrix, verify_block_encoding
from .diagnostics import operator_error
from .matrix import qsvt_matrix_transform
from .spectral import apply_polynomial_to_hermitian, eigh_hermitian


@dataclass(frozen=True)
class BlockEncodedQSVTWorkflowResult:
    """
    Structured output from a verified finite block-encoded QSVT workflow.
    """

    coeffs: np.ndarray
    block_encoding: BlockEncoding
    reference_operator: np.ndarray
    qsvt_operator: np.ndarray | None
    operator_relative_error: float | None
    verification: dict[str, object]
    degree: int
    state: np.ndarray | None = None
    reference_state: np.ndarray | None = None
    qsvt_state: np.ndarray | None = None
    state_relative_error: float | None = None
    qsvt_error: str | None = None

    def as_report(self) -> dict[str, Any]:
        """
        Return a report-style dictionary for JSON conversion or persistence.
        """
        qsvt_check = "failed" if self.qsvt_error is not None else "succeeded"
        return {
            **algorithm_workflow_schema_fields(),
            "mode": "block-encoded-qsvt-workflow",
            "implementation_kind": "verified-dense-block-encoded-qsvt-workflow",
            "truth_contract": _block_encoded_qsvt_truth_contract(
                qsvt_check=qsvt_check,
            ),
            "degree": self.degree,
            "coeffs": self.coeffs,
            "block_encoding": self.block_encoding.as_report(),
            "verification": self.verification,
            "reference_operator": self.reference_operator,
            "qsvt_operator": self.qsvt_operator,
            "operator_relative_error": self.operator_relative_error,
            "state": self.state,
            "reference_state": self.reference_state,
            "qsvt_state": self.qsvt_state,
            "state_relative_error": self.state_relative_error,
            "qsvt_error": self.qsvt_error,
        }


def _block_encoded_qsvt_truth_contract(*, qsvt_check: str) -> dict[str, object]:
    if qsvt_check not in {"succeeded", "failed"}:
        raise ValueError("qsvt_check must be 'succeeded' or 'failed'.")

    return {
        "workflow": "block-encoded-qsvt-workflow",
        "target": "finite block-encoded positive-Hermitian polynomial transform",
        "implementation_kind": "verified-dense-block-encoded-qsvt-workflow",
        "truth_status": (
            "verified_block_encoding_and_qsvt_polynomial_transform"
            if qsvt_check == "succeeded"
            else "verified_block_encoding_qsvt_transform_failed"
        ),
        "is_end_to_end_quantum_algorithm": False,
        "implemented_components": [
            "explicit_dense_unitary_block_encoding",
            "top_left_block_verification",
            "unitarity_verification",
            "pennylane_qsvt_transform_when_synthesis_succeeds",
            "classical_spectral_polynomial_reference",
            "operator_and_optional_state_error_diagnostics",
        ],
        "pennylane_qsvt_check": qsvt_check,
        "conditional_qsvt_statement": (
            "The supplied finite matrix is encoded as the top-left block of an "
            "explicit unitary and transformed with a compatible QSVT polynomial "
            "on the normalized signal operator."
        ),
        "validation_scope": (
            "This validates a finite dense block encoding and QSVT transform. "
            "It does not make scalability, data-loading, readout, or hardware "
            "claims for larger problem families."
        ),
        "omitted_quantum_costs": [
            "scalable_oracle_or_sparse_block_encoding_construction",
            "input_state_preparation_or_data_loading",
            "measurement_or_readout_strategy",
            "amplitude_amplification_or_estimation",
            "fault_tolerant_synthesis",
            "hardware_compilation",
        ],
    }


def block_encoded_qsvt_workflow(
    matrix: np.ndarray,
    coeffs: np.ndarray,
    *,
    alpha: float | None = None,
    state: np.ndarray | None = None,
    max_signal_norm: float = 0.8,
    block_atol: float = 1e-10,
    unitary_atol: float = 1e-10,
    encoding_wires: list[int] | None = None,
    wire_order: list[int] | None = None,
    block_encoding: str = "embedding",
    real_output: bool = True,
) -> BlockEncodedQSVTWorkflowResult:
    """
    Apply a QSVT polynomial to a verified finite block-encoded positive matrix.

    The workflow first constructs an explicit dense unitary whose top-left
    block is ``matrix / alpha``. QSVT is then applied to that normalized signal
    operator and compared with the exact spectral polynomial ``P(matrix /
    alpha)``. Polynomial coefficients are interpreted in the normalized signal
    coordinate. The direct PennyLane comparison used here is restricted to
    positive-semidefinite Hermitian signals, where the package's matrix-QSVT
    wrapper agrees with ordinary spectral polynomial functional calculus.
    """
    evals, _ = eigh_hermitian(matrix)
    if evals[0] < -1e-10:
        raise ValueError("matrix must be positive semidefinite for this workflow.")
    A = np.asarray(matrix, dtype=complex if np.iscomplexobj(matrix) else float)
    poly = np.asarray(coeffs, dtype=float)
    if poly.ndim != 1 or poly.size == 0:
        raise ValueError("coeffs must contain at least one coefficient.")
    if not np.all(np.isfinite(poly)):
        raise ValueError("coeffs must be finite.")
    if not (0.0 < max_signal_norm <= 1.0):
        raise ValueError("max_signal_norm must satisfy 0 < max_signal_norm <= 1.")

    if alpha is None:
        norm = float(np.linalg.norm(A, ord=2))
        alpha = max(1.0, norm / float(max_signal_norm))

    encoding = block_encode_matrix(A, alpha=alpha)
    verification = verify_block_encoding(
        encoding,
        block_atol=block_atol,
        unitary_atol=unitary_atol,
    )
    if not verification["block_encoding_verified"]:
        raise ValueError("block encoding failed top-left block verification.")
    if not verification["unitary_verified"]:
        raise ValueError("block encoding failed unitarity verification.")

    signal_evals = evals / encoding.alpha
    if np.max(np.abs(signal_evals)) > 1.0 + 1e-10:
        raise ValueError("normalized Hermitian spectrum must lie in [-1, 1].")

    reference_operator = apply_polynomial_to_hermitian(
        encoding.signal_operator,
        poly,
    )

    state_vec = None
    reference_state = None
    qsvt_state = None
    state_error = None
    if state is not None:
        state_vec = _validate_state(state, A.shape[0])
        reference_state = reference_operator @ state_vec

    qsvt_operator = None
    operator_relative = None
    qsvt_error = None
    try:
        qsvt_operator = qsvt_matrix_transform(
            encoding.signal_operator,
            poly,
            encoding_wires=encoding_wires,
            wire_order=wire_order,
            block_encoding=block_encoding,
            real_output=real_output,
        )
        operator_relative = operator_error(reference_operator, qsvt_operator)
        if state_vec is not None and reference_state is not None:
            qsvt_state = qsvt_operator @ state_vec
            state_error = _relative_error(reference_state, qsvt_state)
    except Exception as exc:  # pragma: no cover - backend-dependent path
        qsvt_error = f"{type(exc).__name__}: {exc}"

    return BlockEncodedQSVTWorkflowResult(
        coeffs=poly,
        block_encoding=encoding,
        reference_operator=reference_operator,
        qsvt_operator=qsvt_operator,
        operator_relative_error=operator_relative,
        verification=verification,
        degree=int(poly.size - 1),
        state=state_vec,
        reference_state=reference_state,
        qsvt_state=qsvt_state,
        state_relative_error=state_error,
        qsvt_error=qsvt_error,
    )
