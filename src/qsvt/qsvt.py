"""
qsvt.qsvt
---------

Compatibility re-export module for QSVT helpers.

The implementation is split across ``qsvt.operators``, ``qsvt.diagonal``,
``qsvt.matrix``, and ``qsvt.compatibility``. Existing imports from
``qsvt.qsvt`` remain supported.
"""

from __future__ import annotations

from .compatibility import qsvt_compatibility_report
from .diagonal import (
    classical_diagonal_polynomial_transform,
    compare_qsvt_vs_classical_diagonal,
    qsvt_diagonal_transform,
    qsvt_scalar_output,
    qsvt_scalar_scan,
    qsvt_transform_report,
)
from .execution import (
    QSVTCircuitExecutionResult,
    execute_qsvt_circuit,
    qsvt_circuit_truth_contract,
)
from .matrix import (
    compare_qsvt_vs_classical_matrix,
    qsvt_matrix_transform,
    qsvt_matrix_transform_report,
)
from .operators import (
    apply_qsvt_to_embedded_vector,
    qsvt_operator,
    qsvt_top_left_block,
    qsvt_unitary,
)

__all__ = [
    "QSVTCircuitExecutionResult",
    "execute_qsvt_circuit",
    "qsvt_circuit_truth_contract",
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
