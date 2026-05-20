"""
Lightweight QSVT resource proxy reports.

These helpers summarize polynomial degree, phase-count, width, call-count, and
diagnostic metadata for small educational QSVT workflows. They are intentionally
not fault-tolerant or hardware resource estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .compatibility import qsvt_compatibility_report
from .polynomials import polynomial_degree


@dataclass(frozen=True)
class ResourceEstimate:
    """
    Compact proxy resource estimate for a QSVT-style polynomial transform.
    """

    degree: int
    coefficient_count: int
    qsp_phase_count: int
    signal_operator_calls: int
    inverse_signal_operator_calls: int
    matrix_dimension: int | None = None
    encoding_qubits: int | None = None
    total_qubits: int | None = None
    block_encoding: str = "unspecified"
    notes: tuple[str, ...] = ()

    def as_report(self) -> dict[str, object]:
        """
        Return a JSON-friendly report dictionary.
        """
        return {
            "degree": self.degree,
            "coefficient_count": self.coefficient_count,
            "qsp_phase_count": self.qsp_phase_count,
            "signal_operator_calls": self.signal_operator_calls,
            "inverse_signal_operator_calls": self.inverse_signal_operator_calls,
            "matrix_dimension": self.matrix_dimension,
            "encoding_qubits": self.encoding_qubits,
            "total_qubits": self.total_qubits,
            "block_encoding": self.block_encoding,
            "notes": list(self.notes),
        }


def _ceil_log2(value: int) -> int:
    if value < 1:
        raise ValueError("matrix_dimension must be positive.")
    return int((value - 1).bit_length())


def estimate_qsvt_resources(
    coeffs: np.ndarray | list[float],
    *,
    matrix_dimension: int | None = None,
    encoding_qubits: int | None = None,
    block_encoding: str = "dense-block-encoding",
) -> ResourceEstimate:
    """
    Estimate high-level resource proxies for a polynomial QSVT transform.

    The estimate uses the polynomial degree as the signal-processing sequence
    length proxy. If ``matrix_dimension`` is supplied and ``encoding_qubits`` is
    omitted, the encoding width is inferred as ``ceil(log2(matrix_dimension))``.
    One extra signal/control qubit is included in ``total_qubits`` when an
    encoding width is known.
    """
    coeff_arr = np.asarray(coeffs, dtype=float)
    if coeff_arr.ndim != 1 or coeff_arr.size == 0:
        raise ValueError("coeffs must be a non-empty one-dimensional sequence.")
    if not np.all(np.isfinite(coeff_arr)):
        raise ValueError("coeffs must contain only finite values.")

    if matrix_dimension is not None:
        matrix_dimension = int(matrix_dimension)
        inferred_qubits = _ceil_log2(matrix_dimension)
    else:
        inferred_qubits = None

    if encoding_qubits is None:
        encoding_qubits = inferred_qubits
    elif encoding_qubits < 0:
        raise ValueError("encoding_qubits must be non-negative.")
    else:
        encoding_qubits = int(encoding_qubits)

    degree = polynomial_degree(coeff_arr)
    notes = [
        "Proxy estimate based on polynomial degree; not a hardware resource model.",
        "Signal-call counts assume one forward and one inverse query per QSVT step.",
    ]
    if matrix_dimension is not None and encoding_qubits is not None:
        capacity = 2**encoding_qubits
        if capacity < matrix_dimension:
            raise ValueError("encoding_qubits cannot represent matrix_dimension.")
        if capacity != matrix_dimension:
            notes.append("Encoding width includes unused basis states.")

    return ResourceEstimate(
        degree=degree,
        coefficient_count=int(coeff_arr.size),
        qsp_phase_count=degree + 1,
        signal_operator_calls=degree,
        inverse_signal_operator_calls=degree,
        matrix_dimension=matrix_dimension,
        encoding_qubits=encoding_qubits,
        total_qubits=(encoding_qubits + 1 if encoding_qubits is not None else None),
        block_encoding=block_encoding,
        notes=tuple(notes),
    )


def qsvt_resource_report(
    coeffs: np.ndarray | list[float],
    *,
    matrix_dimension: int | None = None,
    encoding_qubits: int | None = None,
    block_encoding: str = "dense-block-encoding",
    bounded_num_points: int = 4001,
    attempt_synthesis: bool = True,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, object]:
    """
    Build a combined resource, compatibility, and optional diagnostics report.
    """
    coeff_arr = np.asarray(coeffs, dtype=float)
    estimate = estimate_qsvt_resources(
        coeff_arr,
        matrix_dimension=matrix_dimension,
        encoding_qubits=encoding_qubits,
        block_encoding=block_encoding,
    )
    compatibility = qsvt_compatibility_report(
        coeff_arr,
        bounded_num_points=bounded_num_points,
        attempt_synthesis=attempt_synthesis,
    )

    return {
        "mode": "resource-report",
        "coeffs": coeff_arr,
        "resources": estimate.as_report(),
        "compatibility": compatibility,
        "diagnostics": diagnostics or {},
        "limitations": [
            "No block-encoding construction cost is included.",
            "No state-preparation, amplitude-amplification, error-correction, "
            "or hardware compilation cost is included.",
            "Use the report for comparing small polynomial workflows, not for "
            "claiming end-to-end quantum runtime.",
        ],
    }


__all__ = [
    "ResourceEstimate",
    "estimate_qsvt_resources",
    "qsvt_resource_report",
]
