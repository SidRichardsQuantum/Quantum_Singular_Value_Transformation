"""Public API stability labels and compatibility policy for :mod:`qsvt`."""

from __future__ import annotations

from types import MappingProxyType

API_STATUS_STABLE = "stable"
API_STATUS_COMPATIBILITY = "compatibility"
API_STATUS_EXPERIMENTAL = "experimental"

_STABLE_NAMES = frozenset(
    {
        "BlockEncodingSpec",
        "QSVTExecutionConfig",
        "QSVTProblemSpec",
        "QSVTTransformSpec",
        "certify_polynomial_boundedness",
        "classify_polynomial_realizability",
        "design_workflow",
        "estimate_encoding_aware_resources",
        "hamiltonian_simulation_workflow",
        "load_report_with_schema",
        "plan_qsvt",
        "poisson_qsvt_workflow",
        "qsvt_problem_workflow",
        "report_to_jsonable",
        "save_report",
        "synthesize_phases",
        "supported_report_schemas",
        "spectral_filter_qsvt_workflow",
        "run_qsvt_plan",
        "validate_report_schema",
    }
)

_PREVIOUS_STABLE_NAMES = frozenset(
    {
        "ClassicalBenchmarkResult",
        "BoundednessCertificate",
        "DesignWorkflowResult",
        "DegreeSearchCandidate",
        "DegreeSearchResult",
        "EncodingAwareResourceEstimate",
        "MixedParitySynthesisResult",
        "PhaseSynthesisResult",
        "PhaseSolverBenchmarkResult",
        "PoissonQSVTResult",
        "PolynomialRealizability",
        "QSVTProblemWorkflowResult",
        "QSVTExecutionConfig",
        "QSVTPlan",
        "QSVTPlanRunResult",
        "QSVTProblemSpec",
        "QSVTTransformSpec",
        "ReportSchemaCompatibility",
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
        "SpectralFilterQSVTResult",
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
        "estimate_encoding_aware_resources",
        "fermi_dirac_occupation_workflow",
        "fixed_point_amplification_workflow",
        "ground_state_filtering_workflow",
        "hamiltonian_simulation_workflow",
        "linear_system_comparison_summary_table",
        "linear_system_comparison_workflow",
        "linear_system_workflow",
        "load_report",
        "load_report_with_schema",
        "migrate_algorithm_workflow_report",
        "matrix_log_entropy_workflow",
        "parity_components",
        "plot_benchmark_timings",
        "plot_qsvt_proxy_resources",
        "plan_qsvt",
        "poisson_qsvt_workflow",
        "qsvt_compatibility_report",
        "qsvt_resource_report",
        "qsvt_problem_workflow",
        "qsvt_transform_report",
        "quantum_walk_search_resource_proxy",
        "quantum_walk_search_workflow",
        "report_schema_manifest",
        "report_to_jsonable",
        "resolvent_workflow",
        "save_report",
        "save_report_plot",
        "search_design_degree",
        "search_polynomial_degree",
        "synthesize",
        "synthesize_mixed_parity",
        "synthesize_phases",
        "synthesis_workflow",
        "supported_report_schemas",
        "singular_value_filtering_workflow",
        "singular_value_pseudoinverse_workflow",
        "spectral_counting_workflow",
        "spectral_filter_qsvt_workflow",
        "spectral_density_workflow",
        "spectral_thresholding_workflow",
        "thermal_gibbs_workflow",
        "run_qsvt_plan",
        "validate_report_schema",
        "write_report_schema_manifest_csv",
        "write_benchmark_summary_csv",
        "write_linear_system_comparison_csv",
    }
)

_COMPATIBILITY_NAMES = _PREVIOUS_STABLE_NAMES - _STABLE_NAMES

_EXPERIMENTAL_NAMES = frozenset(
    {
        "BlockEncodedQSVTWorkflowResult",
        "BlockEncoding",
        "BlockEncodingSpec",
        "BlockEncodingQSVTExecutionResult",
        "AccuracyResourceFrontierEvaluator",
        "AccuracyResourceFrontierResult",
        "PhaseSolverAdapter",
        "HHLCircuitExecutionResult",
        "HardwareQSVTCircuitReport",
        "HardwareQSVTExecutionResult",
        "HardwareQSVTPreflightResult",
        "ProviderPluginReport",
        "QSVTCircuitExecutionResult",
        "ResearchOperatorSpec",
        "ResearchSweepResult",
        "ResearchSweepSpec",
        "ResearchTargetSpec",
        "ResearchTrial",
        "accuracy_resource_frontier_rows",
        "accuracy_resource_frontier_spec",
        "execute_hhl_circuit",
        "execute_qsvt_circuit",
        "execute_qsvt_from_spec",
        "execute_qsvt_on_device",
        "expand_research_sweep",
        "hhl_circuit_truth_contract",
        "qsvt_hardware_circuit_report",
        "qsvt_circuit_truth_contract",
        "qsvt_hardware_preflight",
        "qsvt_hardware_truth_contract",
        "qsvt_provider_plugin_report",
        "load_research_sweep_spec",
        "research_summary_rows",
        "run_accuracy_resource_frontier",
        "run_research_sweep",
        "save_research_sweep_spec",
        "available_phase_solver_adapters",
        "build_block_encoding_operator",
        "circuit_block_encoding_spec",
        "matrix_block_encoding_spec",
        "pennylane_operator_block_encoding_spec",
        "qsvt_operator_from_block_encoding",
        "clear_phase_synthesis_cache",
        "phase_synthesis_cache_info",
        "register_phase_solver_adapter",
        "synthesize_phases_cached",
        "synthesize_phases_with_adapter",
        "unregister_phase_solver_adapter",
        "write_accuracy_resource_pareto_csv",
        "write_research_summary_csv",
    }
)

STABLE_API_NAMES = tuple(sorted(_STABLE_NAMES))
COMPATIBILITY_API_NAMES = tuple(sorted(_COMPATIBILITY_NAMES))

DEPRECATION_POLICY = (
    "The names exported by qsvt.stable are frozen for the remainder of the 0.x "
    "series. Compatibility names remain importable from their documented modules "
    "and the qsvt package root. A compatibility name must be announced as "
    "deprecated in the changelog and emit DeprecationWarning for at least two "
    "minor releases before removal. Experimental names may change between minor "
    "releases, but incompatible changes must still be documented."
)

__api_statuses__ = MappingProxyType(
    {
        **{name: API_STATUS_EXPERIMENTAL for name in _EXPERIMENTAL_NAMES},
        **{name: API_STATUS_COMPATIBILITY for name in _COMPATIBILITY_NAMES},
        **{name: API_STATUS_STABLE for name in _STABLE_NAMES},
    }
)


def api_status(name: str) -> str:
    """
    Return the coarse public API status for an exported name.

    ``stable`` names are frozen in :mod:`qsvt.stable`. ``compatibility`` names
    retain the documented two-minor-release deprecation window. Names not
    listed explicitly are treated as experimental during the 0.x series.
    """
    return __api_statuses__.get(name, API_STATUS_EXPERIMENTAL)


__all__ = [
    "API_STATUS_COMPATIBILITY",
    "API_STATUS_EXPERIMENTAL",
    "API_STATUS_STABLE",
    "COMPATIBILITY_API_NAMES",
    "DEPRECATION_POLICY",
    "STABLE_API_NAMES",
    "__api_statuses__",
    "api_status",
]
