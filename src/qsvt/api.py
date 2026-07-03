"""
Public API status labels for qsvt.

The labels are intentionally coarse during the 0.x series. They help users
distinguish documented workflow entry points from lower-level utilities that
may still move as the package approaches a 1.0 API.
"""

from __future__ import annotations

from types import MappingProxyType

API_STATUS_STABLE = "stable"
API_STATUS_EXPERIMENTAL = "experimental"

_STABLE_NAMES = frozenset(
    {
        "ClassicalBenchmarkResult",
        "BoundednessCertificate",
        "DesignWorkflowResult",
        "MixedParitySynthesisResult",
        "PhaseSynthesisResult",
        "PhaseSolverBenchmarkResult",
        "PolynomialRealizability",
        "FermiDiracWorkflowResult",
        "FixedPointAmplificationWorkflowResult",
        "GroundStateFilteringWorkflowResult",
        "LinearSystemComparisonResult",
        "LinearSystemWorkflowResult",
        "MatrixLogEntropyWorkflowResult",
        "ResourceEstimate",
        "SingularValueFilteringWorkflowResult",
        "SingularValuePseudoinverseWorkflowResult",
        "SpectralCountingWorkflowResult",
        "ThermalGibbsWorkflowResult",
        "approximation_quality_report",
        "benchmark_environment_report",
        "benchmark_phase_solvers",
        "benchmark_summary_table",
        "block_encoded_qsvt_workflow",
        "chebyshev_fit_function",
        "certify_polynomial_boundedness",
        "classify_polynomial_realizability",
        "conjugate_gradient_benchmark",
        "conjugate_gradient_solve",
        "dense_eigendecomposition_benchmark",
        "dense_linear_solve_benchmark",
        "design_workflow",
        "estimate_qsvt_resources",
        "fermi_dirac_occupation_workflow",
        "fixed_point_amplification_workflow",
        "ground_state_filtering_workflow",
        "hamiltonian_simulation_workflow",
        "linear_system_comparison_summary_table",
        "linear_system_comparison_workflow",
        "linear_system_workflow",
        "load_report",
        "matrix_log_entropy_workflow",
        "parity_components",
        "plot_benchmark_timings",
        "plot_qsvt_proxy_resources",
        "qsvt_compatibility_report",
        "qsvt_resource_report",
        "qsvt_transform_report",
        "quantum_walk_search_resource_proxy",
        "quantum_walk_search_workflow",
        "report_to_jsonable",
        "resolvent_workflow",
        "save_report",
        "save_report_plot",
        "synthesize",
        "synthesize_mixed_parity",
        "synthesize_phases",
        "synthesis_workflow",
        "singular_value_filtering_workflow",
        "singular_value_pseudoinverse_workflow",
        "spectral_counting_workflow",
        "spectral_density_workflow",
        "spectral_thresholding_workflow",
        "thermal_gibbs_workflow",
        "write_benchmark_summary_csv",
        "write_linear_system_comparison_csv",
    }
)

_EXPERIMENTAL_NAMES = frozenset(
    {
        "BlockEncodedQSVTWorkflowResult",
        "BlockEncoding",
        "BlockEncodingSpec",
        "BlockEncodingQSVTExecutionResult",
        "HHLCircuitExecutionResult",
        "HardwareQSVTCircuitReport",
        "HardwareQSVTExecutionResult",
        "HardwareQSVTPreflightResult",
        "ProviderPluginReport",
        "QSVTCircuitExecutionResult",
        "execute_hhl_circuit",
        "execute_qsvt_circuit",
        "execute_qsvt_from_spec",
        "execute_qsvt_on_device",
        "hhl_circuit_truth_contract",
        "qsvt_hardware_circuit_report",
        "qsvt_circuit_truth_contract",
        "qsvt_hardware_preflight",
        "qsvt_hardware_truth_contract",
        "qsvt_provider_plugin_report",
        "build_block_encoding_operator",
        "circuit_block_encoding_spec",
        "matrix_block_encoding_spec",
        "pennylane_operator_block_encoding_spec",
        "qsvt_operator_from_block_encoding",
    }
)

__api_statuses__ = MappingProxyType(
    {
        **{name: API_STATUS_STABLE for name in _STABLE_NAMES},
        **{name: API_STATUS_EXPERIMENTAL for name in _EXPERIMENTAL_NAMES},
    }
)


def api_status(name: str) -> str:
    """
    Return the coarse public API status for an exported name.

    Names not listed explicitly are treated as experimental during the 0.x
    series. This keeps the root API importable while making the stability
    boundary clear to users.
    """
    return __api_statuses__.get(name, API_STATUS_EXPERIMENTAL)


__all__ = [
    "API_STATUS_EXPERIMENTAL",
    "API_STATUS_STABLE",
    "__api_statuses__",
    "api_status",
]
