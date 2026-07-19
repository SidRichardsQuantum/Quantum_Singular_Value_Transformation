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

from .block_encoding import BlockEncodingSpec
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
    estimate_kind: str = "proxy"
    omitted_costs: tuple[str, ...] = (
        "block_encoding_construction",
        "state_preparation",
        "amplitude_amplification",
        "error_correction",
        "hardware_compilation",
    )
    requires_block_encoding: bool = True
    requires_state_preparation: bool = True
    fault_tolerant_estimate: bool = False

    def as_report(self) -> dict[str, object]:
        """
        Return a JSON-friendly report dictionary.
        """
        return {
            "estimate_kind": self.estimate_kind,
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
            "omitted_costs": list(self.omitted_costs),
            "requires_block_encoding": self.requires_block_encoding,
            "requires_state_preparation": self.requires_state_preparation,
            "fault_tolerant_estimate": self.fault_tolerant_estimate,
        }


@dataclass(frozen=True)
class EncodingAwareResourceEstimate:
    """Logical and gate-set resource data for a concrete block-encoding spec."""

    degree: int
    encoding_kind: str
    encoding_method: str
    normalization_alpha: float
    logical_shape: tuple[int, int]
    encoding_wire_count: int
    signal_operator_calls: int
    inverse_signal_operator_calls: int
    estimator_available: bool
    estimator_kind: str
    estimator_model: str
    gate_set: tuple[str, ...] | None
    total_wires: int | None
    total_gates: int | None
    gate_counts: dict[str, int]
    assumptions: tuple[str, ...]
    omitted_costs: tuple[str, ...]
    error_type: str | None = None
    error: str | None = None

    def as_report(self) -> dict[str, object]:
        """Return a portable encoding-aware resource report."""
        return {
            "mode": "encoding-aware-qsvt-resource-report",
            "implementation_kind": "encoding-aware-logical-resource-estimate",
            "degree": self.degree,
            "encoding_kind": self.encoding_kind,
            "encoding_method": self.encoding_method,
            "normalization_alpha": self.normalization_alpha,
            "logical_shape": self.logical_shape,
            "encoding_wire_count": self.encoding_wire_count,
            "signal_operator_calls": self.signal_operator_calls,
            "inverse_signal_operator_calls": self.inverse_signal_operator_calls,
            "estimator_available": self.estimator_available,
            "estimator_kind": self.estimator_kind,
            "estimator_model": self.estimator_model,
            "gate_set": self.gate_set,
            "total_wires": self.total_wires,
            "total_gates": self.total_gates,
            "gate_counts": self.gate_counts,
            "assumptions": list(self.assumptions),
            "omitted_costs": list(self.omitted_costs),
            "error_type": self.error_type,
            "error": self.error,
            "truth_contract": {
                "is_executed_circuit_measurement": False,
                "is_fault_tolerant_estimate": False,
                "is_logical_algorithm_estimate": (
                    self.estimator_available and self.total_gates is not None
                ),
                "includes_encoding_normalization": True,
                "includes_selected_block_encoding_model": self.total_gates is not None,
                "omitted_costs": list(self.omitted_costs),
            },
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
        "estimate_kind": estimate.estimate_kind,
        "truth_contract": {
            "implementation_kind": "polynomial-resource-proxy",
            "truth_status": "proxy_only",
            "is_end_to_end_quantum_resource_estimate": False,
            "reported_components": [
                "polynomial_degree",
                "coefficient_count",
                "qsp_phase_count_proxy",
                "signal_operator_call_proxy",
                "matrix_register_width_when_dimension_is_supplied",
                "sampled_compatibility_checks",
            ],
            "conditional_qsvt_statement": (
                "The proxy is relevant only after a valid block encoding, "
                "state-preparation model, and readout strategy have been specified."
            ),
            "validation_scope": (
                "The report compares polynomial-level quantities and sampled "
                "compatibility checks; it does not estimate wall-clock runtime."
            ),
        },
        "coeffs": coeff_arr,
        "resources": estimate.as_report(),
        "compatibility": compatibility,
        "diagnostics": diagnostics or {},
        "requires_block_encoding": estimate.requires_block_encoding,
        "requires_state_preparation": estimate.requires_state_preparation,
        "fault_tolerant_estimate": estimate.fault_tolerant_estimate,
        "omitted_costs": list(estimate.omitted_costs),
        "limitations": [
            "No block-encoding construction cost is included.",
            "No state-preparation, amplitude-amplification, error-correction, "
            "or hardware compilation cost is included.",
            "Use the report for comparing small polynomial workflows, not for "
            "claiming end-to-end quantum runtime.",
        ],
    }


def estimate_encoding_aware_resources(
    spec: BlockEncodingSpec,
    coeffs: np.ndarray | list[float],
    *,
    gate_set: tuple[str, ...] | list[str] | None = None,
) -> EncodingAwareResourceEstimate:
    """Estimate QSVT resources for a concrete block-encoding access model.

    PennyLane's logical estimator is used when it is available. Matrix and
    opaque custom encodings are modeled as arbitrary unitaries on the declared
    encoding wires. Pauli-operator PrepSelPrep/qubitization specifications use
    a Pauli-LCU qubitization model. These choices are recorded explicitly so
    callers can distinguish a concrete model from an executed-circuit count.
    """
    if not isinstance(spec, BlockEncodingSpec):
        raise TypeError("spec must be a BlockEncodingSpec.")
    coeff_arr = np.asarray(coeffs, dtype=float)
    if coeff_arr.ndim != 1 or coeff_arr.size == 0:
        raise ValueError("coeffs must be a non-empty one-dimensional sequence.")
    if not np.all(np.isfinite(coeff_arr)):
        raise ValueError("coeffs must contain only finite values.")

    degree = polynomial_degree(coeff_arr)
    normalized_gate_set = None if gate_set is None else tuple(map(str, gate_set))
    assumptions = [
        "The block-encoding normalization alpha is included explicitly.",
        "QSVT query counts follow the alternating forward/adjoint sequence.",
    ]
    omitted = (
        "application_state_preparation",
        "postselection_or_amplitude_amplification",
        "application_readout_or_tomography",
        "provider_compilation_and_routing",
        "error_correction_cycle_time",
    )
    estimator_available = False
    estimator_model = "degree-and-access-model-only"
    total_wires: int | None = None
    total_gates: int | None = None
    gate_counts: dict[str, int] = {}
    error_type: str | None = None
    error: str | None = None

    try:
        import pennylane.estimator as qre

        if not hasattr(qre, "QSVT") or not hasattr(qre, "estimate"):
            raise AttributeError("PennyLane logical QSVT estimator is unavailable.")
        estimator_available = True
        resource_block, estimator_model, model_assumptions = (
            _pennylane_resource_block_encoding(spec, qre)
        )
        assumptions.extend(model_assumptions)
        resource_qsvt = qre.QSVT(
            resource_block,
            encoding_dims=spec.logical_shape,
            poly_deg=degree,
        )
        estimate_kwargs: dict[str, object] = {}
        if normalized_gate_set is not None:
            estimate_kwargs["gate_set"] = set(normalized_gate_set)
        estimated = qre.estimate(resource_qsvt, **estimate_kwargs)
        total_wires = int(estimated.total_wires)
        total_gates = int(estimated.total_gates)
        gate_counts = {
            str(getattr(operation, "name", operation)): int(count)
            for operation, count in estimated.gate_types.items()
        }
    except Exception as exc:
        error_type = type(exc).__name__
        error = str(exc)

    return EncodingAwareResourceEstimate(
        degree=degree,
        encoding_kind=spec.kind,
        encoding_method=spec.block_encoding,
        normalization_alpha=float(spec.alpha),
        logical_shape=spec.logical_shape,
        encoding_wire_count=len(spec.encoding_wires),
        signal_operator_calls=(degree + 1) // 2,
        inverse_signal_operator_calls=degree // 2,
        estimator_available=estimator_available,
        estimator_kind="pennylane.estimator.QSVT",
        estimator_model=estimator_model,
        gate_set=normalized_gate_set,
        total_wires=total_wires,
        total_gates=total_gates,
        gate_counts=gate_counts,
        assumptions=tuple(assumptions),
        omitted_costs=omitted,
        error_type=error_type,
        error=error,
    )


def _pennylane_resource_block_encoding(
    spec: BlockEncodingSpec,
    qre: Any,
) -> tuple[Any, str, tuple[str, ...]]:
    if spec.kind == "pennylane-operator":
        source = spec.source
        pauli_rep = getattr(source, "pauli_rep", None)
        if pauli_rep:
            pauli_terms: dict[str, int] = {}
            identity_terms = 0
            for word in pauli_rep:
                letters = "".join(
                    str(letter) for _, letter in sorted(dict(word).items())
                )
                if not letters:
                    identity_terms += 1
                    continue
                label = letters
                pauli_terms[label] = pauli_terms.get(label, 0) + 1
            if pauli_terms:
                system_qubits = _ceil_log2(spec.logical_shape[0])
                hamiltonian = qre.PauliHamiltonian(
                    num_qubits=system_qubits,
                    pauli_terms=pauli_terms,
                    one_norm=float(spec.alpha),
                )
                select = qre.SelectPauli(hamiltonian)
                preparation = qre.QROMStatePreparation(
                    num_state_qubits=max(
                        1,
                        _ceil_log2(sum(pauli_terms.values()) + identity_terms),
                    ),
                )
                return (
                    qre.Qubitization(preparation, select),
                    "pauli-lcu-qubitization",
                    (
                        (
                            "Pauli terms are grouped by Pauli-word shape for logical "
                            "costing."
                        ),
                        (
                            f"{identity_terms} identity term(s) add normalization but "
                            "no SELECT gates."
                        ),
                        (
                            "PrepSelPrep is costed with the available Pauli-LCU "
                            "qubitization resource model."
                            if spec.block_encoding == "prepselprep"
                            else (
                                "The declared qubitization access model is costed "
                                "directly."
                            )
                        ),
                    ),
                )

    wire_count = max(1, len(spec.encoding_wires))
    return (
        qre.QubitUnitary(num_wires=wire_count),
        "arbitrary-unitary-block-encoding",
        (
            "The declared block encoding is modeled as a generic unitary; "
            "structured oracle savings are not assumed.",
        ),
    )


__all__ = [
    "EncodingAwareResourceEstimate",
    "ResourceEstimate",
    "estimate_encoding_aware_resources",
    "estimate_qsvt_resources",
    "qsvt_resource_report",
]
