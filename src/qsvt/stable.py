"""Frozen, compact public facade for the remainder of the qsvt 0.x series.

Existing imports from :mod:`qsvt` and its documented submodules remain
available. This module is the deliberately small surface whose signatures and
documented behavior receive the compatibility guarantee in
``qsvt.DEPRECATION_POLICY``.
"""

from .algorithms import hamiltonian_simulation_workflow
from .block_encoding import BlockEncodingSpec
from .flagship import poisson_qsvt_workflow, spectral_filter_qsvt_workflow
from .planning import (
    QSVTExecutionConfig,
    QSVTProblemSpec,
    QSVTTransformSpec,
    plan_qsvt,
    run_qsvt_plan,
)
from .reports import (
    load_report_with_schema,
    report_to_jsonable,
    save_report,
    supported_report_schemas,
    validate_report_schema,
)
from .resources import estimate_encoding_aware_resources
from .synthesis import (
    certify_polynomial_boundedness,
    classify_polynomial_realizability,
    synthesize_phases,
)
from .workflow import design_workflow, qsvt_problem_workflow

__all__ = [
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
    "run_qsvt_plan",
    "save_report",
    "spectral_filter_qsvt_workflow",
    "supported_report_schemas",
    "synthesize_phases",
    "validate_report_schema",
]
