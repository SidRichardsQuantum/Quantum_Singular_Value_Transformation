"""
Shared report helpers for algorithm workflow result dataclasses.
"""

from __future__ import annotations

from .rescaling import ScaledOperator

ALGORITHM_WORKFLOW_SCHEMA_NAME = "qsvt-algorithm-workflow"
ALGORITHM_WORKFLOW_SCHEMA_VERSION = "1.0"


def algorithm_workflow_schema_fields() -> dict[str, str]:
    """Return the versioned envelope shared by stable algorithm reports."""
    return {
        "schema_name": ALGORITHM_WORKFLOW_SCHEMA_NAME,
        "schema_version": ALGORITHM_WORKFLOW_SCHEMA_VERSION,
    }


def scaled_operator_report(scaled: ScaledOperator) -> dict[str, object]:
    """
    Return a report dictionary for a scaled dense operator.
    """
    return {
        "matrix": scaled.matrix,
        "offset": scaled.offset,
        "scale": scaled.scale,
        "eigenvalue_bounds": scaled.eigenvalue_bounds,
    }


def algorithm_truth_contract(
    workflow: str,
    *,
    target: str,
    qsvt_check: str = "not_attempted",
) -> dict[str, object]:
    """
    Return the explicit implementation contract for an algorithm workflow.

    The workflows in :mod:`qsvt.algorithms` are useful because they validate
    the spectral-polynomial core of QSVT-style methods on concrete dense
    matrices. This helper keeps reports honest about the boundary between what
    the package implements and what a full quantum algorithm would still need.
    """
    if qsvt_check not in {"not_attempted", "succeeded", "failed"}:
        raise ValueError(
            "qsvt_check must be 'not_attempted', 'succeeded', or 'failed'."
        )

    return {
        "workflow": workflow,
        "target": target,
        "implementation_kind": "dense-spectral-polynomial-workflow",
        "truth_status": "validated_polynomial_core",
        "is_end_to_end_quantum_algorithm": False,
        "implemented_components": [
            "input_validation",
            "spectral_rescaling",
            "bounded_polynomial_design_or_application",
            "dense_classical_spectral_reference",
            "numerical_error_diagnostics",
        ],
        "conditional_qsvt_statement": (
            "If a valid block encoding and input-state preparation are supplied, "
            "a compatible bounded polynomial is the QSVT-transform core for "
            "this target."
        ),
        "pennylane_qsvt_check": qsvt_check,
        "assumed_quantum_components": [
            "block_encoding_construction",
            "input_state_preparation_or_data_loading",
            "success_probability_management",
            "measurement_or_readout_strategy",
        ],
        "omitted_quantum_costs": [
            "block_encoding_query_cost",
            "state_preparation_cost",
            "amplitude_amplification",
            "amplitude_estimation_or_tomography",
            "fault_tolerant_synthesis",
            "error_correction",
            "hardware_compilation",
        ],
        "validation_scope": (
            "The report validates numerical agreement between the polynomial "
            "workflow and a dense classical reference for the supplied finite instance."
        ),
        "research_use": (
            "Use this report to study approximation error, degree scaling, "
            "conditioning, spectral gaps, and proxy resources before making "
            "separate assumptions about quantum data access."
        ),
    }


def qsvt_verification_truth_contract(
    workflow: str,
    *,
    target: str,
    qsvt_check: str,
) -> dict[str, object]:
    """
    Return the implementation contract for a direct QSVT verification report.
    """
    if qsvt_check not in {"succeeded", "failed"}:
        raise ValueError("qsvt_check must be 'succeeded' or 'failed'.")

    return {
        "workflow": workflow,
        "target": target,
        "implementation_kind": "pennylane-small-qsvt-verification",
        "truth_status": (
            "verified_against_classical_polynomial"
            if qsvt_check == "succeeded"
            else "classical_reference_only_qsvt_failed"
        ),
        "is_end_to_end_quantum_algorithm": False,
        "implemented_components": [
            "explicit_pennylane_qsvt_operator_when_synthesis_succeeds",
            "logical_block_extraction",
            "classical_polynomial_reference",
            "absolute_error_diagnostics",
        ],
        "pennylane_qsvt_check": qsvt_check,
        "validation_scope": (
            "The report checks a finite simulator instance against direct "
            "classical polynomial evaluation. It does not include scalable "
            "block-encoding construction, data loading, readout, or hardware costs."
        ),
        "omitted_quantum_costs": [
            "block_encoding_construction",
            "state_preparation",
            "measurement_or_readout",
            "fault_tolerant_synthesis",
            "hardware_compilation",
        ],
    }


def benchmark_truth_contract() -> dict[str, object]:
    """
    Return the implementation contract for a classical benchmark report.
    """
    return {
        "implementation_kind": "classical-baseline-with-optional-qsvt-proxy",
        "truth_status": "classical_timing_reference",
        "is_quantum_runtime_benchmark": False,
        "implemented_components": [
            "classical_dense_or_iterative_numpy_computation",
            "wall_clock_timing_for_the_classical_path",
            "residual_or_matrix_function_diagnostics",
            "optional_polynomial_resource_proxy",
        ],
        "validation_scope": (
            "The timing fields measure only the classical baseline path in this "
            "Python/NumPy environment. Attached QSVT fields are polynomial "
            "proxies and are not quantum runtimes."
        ),
    }
