"""Command-line facade and parser for qsvt-pennylane."""

from __future__ import annotations

import argparse
from typing import Iterable

from ._cli_benchmark_commands import (
    cmd_benchmark_cg_solve,
    cmd_benchmark_dense_solve,
    cmd_benchmark_eigh,
    cmd_benchmark_polynomial,
    cmd_benchmark_spectral_function,
    register_benchmark_commands,
)
from ._cli_core_commands import (
    cmd_cheb,
    cmd_compare_report,
    cmd_diag,
    cmd_execute_spec,
    cmd_matrix_report,
    cmd_poly,
    cmd_scalar,
    register_core_commands,
    register_core_report_commands,
)
from ._cli_design_commands import (
    cmd_apply_design,
    cmd_design_compatibility,
    cmd_design_report,
    cmd_design_sweep,
    cmd_design_workflow,
    cmd_template_report,
    register_design_application_commands,
    register_design_commands,
)
from ._cli_flagship_commands import (
    cmd_degree_search,
    cmd_plan_qsvt,
    cmd_poisson_qsvt,
    cmd_spectral_filter_qsvt,
    register_flagship_commands,
)
from ._cli_research_commands import (
    cmd_accuracy_resource_frontier,
    cmd_research_sweep,
    register_research_commands,
)
from ._cli_synthesis_commands import (
    cmd_boundedness_certificate,
    cmd_compatibility_report,
    cmd_mixed_parity_synthesis,
    cmd_phase_solver_benchmark,
    cmd_phase_synthesis,
    cmd_report_schema_manifest,
    cmd_resource_report,
    register_synthesis_commands,
)
from ._cli_utils import emit_cli_result
from ._cli_workflow_commands import (
    cmd_linear_system_compare,
    cmd_problem_workflow,
    cmd_threshold_workflow,
    register_workflow_commands,
)

DESIGN_KINDS = [
    "inverse",
    "sign",
    "projector",
    "sqrt",
    "power",
    "filter",
    "interval_projector",
]


TEMPLATE_KINDS = ["inverse", "sign", "filter", "sqrt", "exponential"]


BENCHMARK_COMMANDS = [
    "eigh",
    "dense-solve",
    "cg-solve",
    "polynomial",
    "spectral-function",
]


PROBLEM_WORKFLOW_TARGETS = [
    "linear_system",
    "spectral_projector",
    "ground_state_filter",
    "hamiltonian_simulation",
    "resolvent",
    "singular_value_filter",
    "singular_value_pseudoinverse",
]


__all__ = [
    "BENCHMARK_COMMANDS",
    "DESIGN_KINDS",
    "PROBLEM_WORKFLOW_TARGETS",
    "TEMPLATE_KINDS",
    "build_parser",
    "cmd_apply_design",
    "cmd_accuracy_resource_frontier",
    "cmd_benchmark_cg_solve",
    "cmd_benchmark_dense_solve",
    "cmd_benchmark_eigh",
    "cmd_benchmark_polynomial",
    "cmd_benchmark_spectral_function",
    "cmd_boundedness_certificate",
    "cmd_cheb",
    "cmd_compare_report",
    "cmd_compatibility_report",
    "cmd_design_compatibility",
    "cmd_design_report",
    "cmd_design_sweep",
    "cmd_design_workflow",
    "cmd_degree_search",
    "cmd_diag",
    "cmd_examples",
    "cmd_execute_spec",
    "cmd_linear_system_compare",
    "cmd_matrix_report",
    "cmd_mixed_parity_synthesis",
    "cmd_phase_solver_benchmark",
    "cmd_phase_synthesis",
    "cmd_plan_qsvt",
    "cmd_poly",
    "cmd_poisson_qsvt",
    "cmd_problem_workflow",
    "cmd_report_schema_manifest",
    "cmd_research_sweep",
    "cmd_resource_report",
    "cmd_scalar",
    "cmd_spectral_filter_qsvt",
    "cmd_template_report",
    "cmd_threshold_workflow",
    "main",
]


def cmd_examples(args: argparse.Namespace) -> dict:
    """
    Return compact CLI discovery metadata and copy-pasteable examples.
    """
    return {
        "mode": "examples",
        "design_kinds": DESIGN_KINDS,
        "template_kinds": TEMPLATE_KINDS,
        "benchmark_commands": BENCHMARK_COMMANDS,
        "workflow_commands": [
            "design-workflow",
            "design-sweep",
            "problem-workflow",
            "plan-workflow",
            "spectral-filter-qsvt",
            "poisson-qsvt",
            "degree-search",
            "research-sweep",
            "accuracy-resource-frontier",
            "linear-system-compare",
            "threshold-workflow",
            "resource-report",
            "report-schema-manifest",
            "execute-spec",
        ],
        "examples": [
            'qsvt scalar --x 0.5 --poly "0,0,1"',
            "qsvt design-workflow --kind sign --gamma 0.2 --degree 13 --no-synthesis",
            'qsvt design-sweep --kind sign --degrees "5,9,13" --gamma 0.2 '
            "--no-synthesis --output sign-degree-sweep.json",
            'qsvt resource-report --poly "0,0,1" --matrix-dimension 4 --no-synthesis',
            "qsvt report-schema-manifest --path "
            "tests/fixtures/reports/qsvt_problem_workflow_v1.json",
            'qsvt execute-spec --kind matrix --matrix "0.2,0;0,0.8" '
            '--poly "0,0,1" --state "1,0"',
            'qsvt linear-system-compare --matrix "2,0.25;0.25,1.25" '
            '--rhs "1,-0.5" --degree 8 --no-synthesis --no-qsvt',
            'qsvt benchmark cg-solve --matrix "4,1;1,3" --rhs "1,2" --qsvt-poly "0,1"',
            "qsvt accuracy-resource-frontier --degrees 3,5 "
            "--output-dir results/research/frontier",
            'qsvt threshold-workflow --matrix "-1,0,0;0,0,0;0,0,1" '
            "--lower -0.25 --upper 0.25 --degree 24",
            'qsvt problem-workflow --target linear_system --matrix "2,0;0,1" '
            '--rhs "1,1" --degree 8 --no-synthesis --no-qsvt',
            'qsvt plan-workflow --target linear_system --matrix "2,0;0,1" '
            '--rhs "1,1" --tolerance 0.2 --no-execute',
            'qsvt spectral-filter-qsvt --pauli-terms "0.4:ZI,0.3:IZ,0.2:XI" '
            '--state "0.5,0.5,0.5,0.5" --lower -0.4 --upper 0.4 '
            "--tolerance 0.16",
            "qsvt poisson-qsvt --n-points 4 --tolerance 0.4",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qsvt",
        description="Minimal CLI for qsvt-pennylane",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_examples = sub.add_parser(
        "examples",
        help="List available workflow families and compact CLI examples",
    )
    p_examples.set_defaults(func=cmd_examples)

    register_core_commands(sub)
    register_design_commands(sub, DESIGN_KINDS, TEMPLATE_KINDS)
    register_core_report_commands(sub)
    register_synthesis_commands(sub)
    register_design_application_commands(sub, DESIGN_KINDS)
    register_workflow_commands(sub, PROBLEM_WORKFLOW_TARGETS)
    register_flagship_commands(sub, PROBLEM_WORKFLOW_TARGETS, DESIGN_KINDS)
    register_benchmark_commands(sub)
    register_research_commands(sub)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(None if argv is None else list(argv))

    result = args.func(args)
    emit_cli_result(args, result)
