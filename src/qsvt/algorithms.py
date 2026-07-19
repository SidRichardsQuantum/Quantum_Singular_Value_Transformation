"""Public facade for the QSVT algorithm workflow families."""

from __future__ import annotations

from ._algorithm_block_encoding import (
    BlockEncodedQSVTWorkflowResult,
    block_encoded_qsvt_workflow,
)
from ._algorithm_dynamics import (
    HamiltonianSimulationWorkflowResult,
    QuantumWalkSearchWorkflowResult,
    hamiltonian_simulation_workflow,
    quantum_walk_search_resource_proxy,
    quantum_walk_search_workflow,
)
from ._algorithm_linear_systems import (
    LinearSystemComparisonResult,
    LinearSystemWorkflowResult,
    linear_system_comparison_summary_table,
    linear_system_comparison_workflow,
    linear_system_resource_proxy,
    linear_system_workflow,
    write_linear_system_comparison_csv,
)
from ._algorithm_singular_values import (
    SingularValueFilteringWorkflowResult,
    SingularValuePseudoinverseWorkflowResult,
    singular_value_filtering_workflow,
    singular_value_pseudoinverse_workflow,
)
from ._algorithm_spectral import (
    FermiDiracWorkflowResult,
    FixedPointAmplificationWorkflowResult,
    GroundStateFilteringWorkflowResult,
    MatrixLogEntropyWorkflowResult,
    ResolventWorkflowResult,
    SpectralCountingWorkflowResult,
    SpectralDensityWorkflowResult,
    SpectralThresholdingWorkflowResult,
    ThermalGibbsWorkflowResult,
    fermi_dirac_occupation_workflow,
    fixed_point_amplification_workflow,
    ground_state_filtering_workflow,
    matrix_log_entropy_workflow,
    resolvent_workflow,
    spectral_counting_workflow,
    spectral_density_workflow,
    spectral_thresholding_workflow,
    thermal_gibbs_workflow,
)

__all__ = [
    "BlockEncodedQSVTWorkflowResult",
    "FermiDiracWorkflowResult",
    "FixedPointAmplificationWorkflowResult",
    "GroundStateFilteringWorkflowResult",
    "HamiltonianSimulationWorkflowResult",
    "LinearSystemComparisonResult",
    "LinearSystemWorkflowResult",
    "MatrixLogEntropyWorkflowResult",
    "QuantumWalkSearchWorkflowResult",
    "ResolventWorkflowResult",
    "SingularValueFilteringWorkflowResult",
    "SingularValuePseudoinverseWorkflowResult",
    "SpectralCountingWorkflowResult",
    "SpectralDensityWorkflowResult",
    "SpectralThresholdingWorkflowResult",
    "ThermalGibbsWorkflowResult",
    "block_encoded_qsvt_workflow",
    "fermi_dirac_occupation_workflow",
    "fixed_point_amplification_workflow",
    "ground_state_filtering_workflow",
    "hamiltonian_simulation_workflow",
    "linear_system_comparison_summary_table",
    "linear_system_comparison_workflow",
    "linear_system_resource_proxy",
    "linear_system_workflow",
    "matrix_log_entropy_workflow",
    "quantum_walk_search_resource_proxy",
    "quantum_walk_search_workflow",
    "resolvent_workflow",
    "singular_value_filtering_workflow",
    "singular_value_pseudoinverse_workflow",
    "spectral_counting_workflow",
    "spectral_density_workflow",
    "spectral_thresholding_workflow",
    "thermal_gibbs_workflow",
    "write_linear_system_comparison_csv",
]
