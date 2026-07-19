"""Private facade for spectral algorithm workflow families."""

from __future__ import annotations

from ._algorithm_response import (
    ResolventWorkflowResult,
    SpectralDensityWorkflowResult,
    resolvent_workflow,
    spectral_density_workflow,
)
from ._algorithm_spectral_filters import (
    FixedPointAmplificationWorkflowResult,
    GroundStateFilteringWorkflowResult,
    SpectralCountingWorkflowResult,
    SpectralThresholdingWorkflowResult,
    fixed_point_amplification_workflow,
    ground_state_filtering_workflow,
    spectral_counting_workflow,
    spectral_thresholding_workflow,
)
from ._algorithm_thermal import (
    FermiDiracWorkflowResult,
    MatrixLogEntropyWorkflowResult,
    ThermalGibbsWorkflowResult,
    fermi_dirac_occupation_workflow,
    matrix_log_entropy_workflow,
    thermal_gibbs_workflow,
)

__all__ = [
    "FermiDiracWorkflowResult",
    "FixedPointAmplificationWorkflowResult",
    "GroundStateFilteringWorkflowResult",
    "MatrixLogEntropyWorkflowResult",
    "ResolventWorkflowResult",
    "SpectralCountingWorkflowResult",
    "SpectralDensityWorkflowResult",
    "SpectralThresholdingWorkflowResult",
    "ThermalGibbsWorkflowResult",
    "fermi_dirac_occupation_workflow",
    "fixed_point_amplification_workflow",
    "ground_state_filtering_workflow",
    "matrix_log_entropy_workflow",
    "resolvent_workflow",
    "spectral_counting_workflow",
    "spectral_density_workflow",
    "spectral_thresholding_workflow",
    "thermal_gibbs_workflow",
]
