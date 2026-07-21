"""
Shared report helpers for algorithm workflow result dataclasses.
"""

from __future__ import annotations

from collections.abc import Mapping

from .rescaling import ScaledOperator
from .synthesis import classify_polynomial_realizability

ALGORITHM_WORKFLOW_SCHEMA_NAME = "qsvt-algorithm-workflow"
ALGORITHM_WORKFLOW_SCHEMA_VERSION = "1.1"


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
    polynomials: Mapping[str, object] | None = None,
    polynomial_design_domains: Mapping[str, tuple[float, float]] | None = None,
    polynomial_output_prefactors: Mapping[str, object] | None = None,
    qnode_executed: bool = False,
    physical_device_executed: bool = False,
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

    evidence = polynomial_truth_evidence(
        polynomials or {},
        design_domains=polynomial_design_domains,
        output_prefactors=polynomial_output_prefactors,
    )
    execution_tier = _execution_tier(
        qsvt_check=qsvt_check,
        qnode_executed=qnode_executed,
        physical_device_executed=physical_device_executed,
        qsvt_realizable=bool(evidence["all_single_sequence_realizable"]),
    )
    truth_status = _polynomial_truth_status(
        qsvt_check=qsvt_check,
        all_single_sequence_realizable=bool(evidence["all_single_sequence_realizable"]),
        requires_parity_decomposition=bool(evidence["requires_parity_decomposition"]),
    )

    return {
        "workflow": workflow,
        "target": target,
        "implementation_kind": "dense-spectral-polynomial-workflow",
        "truth_status": truth_status,
        "execution_tier": execution_tier,
        "qnode_executed": bool(qnode_executed),
        "physical_device_executed": bool(physical_device_executed),
        "circuit_evaluated": qsvt_check == "succeeded" or qnode_executed,
        "resource_completeness": "partial",
        "is_end_to_end_quantum_algorithm": False,
        "implemented_components": [
            "input_validation",
            "spectral_rescaling",
            "polynomial_design_or_application_with_realizability_classification",
            "dense_classical_spectral_reference",
            "numerical_error_diagnostics",
        ],
        "conditional_qsvt_statement": (
            "If a valid block encoding and input-state preparation are supplied, "
            "a compatible bounded polynomial is the QSVT-transform core for "
            "this target."
        ),
        "pennylane_qsvt_check": qsvt_check,
        "polynomial_evidence": evidence,
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


def polynomial_truth_evidence(
    polynomials: Mapping[str, object],
    *,
    design_domains: Mapping[str, tuple[float, float]] | None = None,
    output_prefactors: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Classify the exact polynomials attached to one workflow result."""
    domains = design_domains or {}
    prefactors = output_prefactors or {}
    components: dict[str, dict[str, object]] = {}
    for name, coeffs in polynomials.items():
        classification = classify_polynomial_realizability(coeffs)
        component = classification.as_report()
        component.update(
            {
                "name": name,
                "design_domain": tuple(domains.get(name, (-1.0, 1.0))),
                "qsvt_certification_domain": (-1.0, 1.0),
                "output_prefactor": prefactors.get(name, 1.0),
            }
        )
        components[name] = component

    values = list(components.values())
    return {
        "components": components,
        "component_count": len(components),
        "all_bounded": bool(values) and all(bool(item["bounded"]) for item in values),
        "all_single_sequence_realizable": bool(values)
        and all(bool(item["single_sequence_realizable"]) for item in values),
        "requires_parity_decomposition": any(
            bool(item["requires_parity_decomposition"]) for item in values
        ),
        "contains_classical_only_component": any(
            item["realizability_kind"]
            in {"classical-polynomial-only", "invalid-polynomial"}
            for item in values
        ),
    }


def algorithm_truth_contract_issues(contract: Mapping[str, object]) -> tuple[str, ...]:
    """Return semantic contradictions in a current algorithm truth contract."""
    issues: list[str] = []
    tier = contract.get("execution_tier")
    qsvt_check = contract.get("pennylane_qsvt_check")
    if qsvt_check not in {"not_attempted", "succeeded", "failed"}:
        issues.append("invalid_or_missing_qsvt_check")
        qsvt_check = "not_attempted"
    assert isinstance(qsvt_check, str)
    if tier not in {
        "classical_reference",
        "polynomial_core",
        "qsvt_circuit",
        "hardware_execution",
    }:
        issues.append("invalid_or_missing_execution_tier")
    if tier == "hardware_execution" and not contract.get("physical_device_executed"):
        issues.append("hardware_tier_without_physical_execution")
    if tier == "qsvt_circuit" and not contract.get("circuit_evaluated"):
        issues.append("qsvt_circuit_tier_without_circuit_evaluation")
    if contract.get("qnode_executed") and tier not in {
        "qsvt_circuit",
        "hardware_execution",
    }:
        issues.append("qnode_execution_mislabeled")

    evidence = contract.get("polynomial_evidence")
    if not isinstance(evidence, Mapping):
        issues.append("missing_polynomial_evidence")
        return tuple(issues)
    components = evidence.get("components")
    if not isinstance(components, Mapping) or not components:
        issues.append("missing_polynomial_components")
        return tuple(issues)

    recomputed_components: list[Mapping[str, object]] = []
    for name, component in components.items():
        if not isinstance(component, Mapping):
            issues.append(f"invalid_polynomial_component:{name}")
            continue
        try:
            expected = classify_polynomial_realizability(
                component.get("coeffs")
            ).as_report()
        except (TypeError, ValueError):
            issues.append(f"unclassifiable_polynomial_component:{name}")
            continue
        recomputed_components.append(expected)
        for field in (
            "parity",
            "bounded",
            "realizability_kind",
            "single_sequence_realizable",
            "requires_parity_decomposition",
        ):
            if component.get(field) != expected.get(field):
                issues.append(f"polynomial_evidence_mismatch:{name}:{field}")
    expected_summary = {
        "component_count": len(recomputed_components),
        "all_bounded": bool(recomputed_components)
        and all(bool(item["bounded"]) for item in recomputed_components),
        "all_single_sequence_realizable": bool(recomputed_components)
        and all(
            bool(item["single_sequence_realizable"]) for item in recomputed_components
        ),
        "requires_parity_decomposition": any(
            bool(item["requires_parity_decomposition"])
            for item in recomputed_components
        ),
        "contains_classical_only_component": any(
            item["realizability_kind"]
            in {"classical-polynomial-only", "invalid-polynomial"}
            for item in recomputed_components
        ),
    }
    for field, expected_value in expected_summary.items():
        if evidence.get(field) != expected_value:
            issues.append(f"polynomial_evidence_summary_mismatch:{field}")
    expected_tier = _execution_tier(
        qsvt_check=qsvt_check,
        qnode_executed=bool(contract.get("qnode_executed")),
        physical_device_executed=bool(contract.get("physical_device_executed")),
        qsvt_realizable=bool(expected_summary["all_single_sequence_realizable"]),
    )
    if tier != expected_tier:
        issues.append("execution_tier_mismatch")
    expected_circuit_evaluated = qsvt_check == "succeeded" or bool(
        contract.get("qnode_executed")
    )
    if contract.get("circuit_evaluated") != expected_circuit_evaluated:
        issues.append("circuit_evaluation_status_mismatch")
    if contract.get("resource_completeness") != "partial":
        issues.append("resource_completeness_mismatch")
    expected_status = _polynomial_truth_status(
        qsvt_check=qsvt_check,
        all_single_sequence_realizable=bool(
            expected_summary["all_single_sequence_realizable"]
        ),
        requires_parity_decomposition=bool(
            expected_summary["requires_parity_decomposition"]
        ),
    )
    if contract.get("truth_status") != expected_status:
        issues.append("truth_status_mismatch")
    if (
        tier in {"qsvt_circuit", "hardware_execution"}
        and not expected_summary["all_single_sequence_realizable"]
    ):
        issues.append("circuit_tier_with_nonrealizable_polynomial")
    return tuple(issues)


def _execution_tier(
    *,
    qsvt_check: str,
    qnode_executed: bool,
    physical_device_executed: bool,
    qsvt_realizable: bool,
) -> str:
    if physical_device_executed:
        return "hardware_execution"
    if qsvt_realizable and (qnode_executed or qsvt_check == "succeeded"):
        return "qsvt_circuit"
    return "polynomial_core"


def _polynomial_truth_status(
    *,
    qsvt_check: str,
    all_single_sequence_realizable: bool,
    requires_parity_decomposition: bool,
) -> str:
    if qsvt_check == "succeeded" and all_single_sequence_realizable:
        return "verified_finite_qsvt_circuit"
    if qsvt_check == "succeeded":
        return "circuit_evaluated_without_qsvt_realizability_certificate"
    if all_single_sequence_realizable:
        return "validated_qsvt_compatible_polynomial_core"
    if requires_parity_decomposition:
        return "validated_polynomial_core_requires_parity_combination"
    return "validated_classical_polynomial_core"


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
