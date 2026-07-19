"""High-level algorithm workflow CLI commands."""

from __future__ import annotations

import argparse

import numpy as np

from ._cli_utils import add_report_output_args, parse_complex_list, parse_matrix
from .algorithms import (
    linear_system_comparison_workflow,
    spectral_thresholding_workflow,
    write_linear_system_comparison_csv,
)
from .workflow import qsvt_problem_workflow


def cmd_threshold_workflow(args: argparse.Namespace) -> dict:
    """
    Build a spectral interval-projector workflow report.
    """
    state = np.asarray(parse_complex_list(args.state)) if args.state else None
    result = spectral_thresholding_workflow(
        np.asarray(parse_matrix(args.matrix)),
        lower=args.lower,
        upper=args.upper,
        degree=args.degree,
        sharpness=args.sharpness,
        state=state,
        num_points=args.num_points,
        bounded_num_points=args.bounded_num_points,
    )
    return result.as_report()


def cmd_linear_system_compare(args: argparse.Namespace) -> dict:
    """
    Build a linear-system solver comparison workflow report.
    """
    result = linear_system_comparison_workflow(
        np.asarray(parse_matrix(args.matrix)),
        np.asarray(parse_complex_list(args.rhs)),
        degree=args.degree,
        gamma=args.gamma,
        num_points=args.num_points,
        bounded_num_points=args.bounded_num_points,
        attempt_synthesis=args.attempt_synthesis,
        apply_qsvt=args.apply_qsvt,
        include_conjugate_gradient=not args.no_cg,
        cg_tolerance=args.cg_tolerance,
        cg_max_iterations=args.cg_max_iterations,
    )
    report = result.as_report()
    rows_output = getattr(args, "rows_output", None)
    if rows_output:
        write_linear_system_comparison_csv(report, rows_output)
    return report


def cmd_problem_workflow(args: argparse.Namespace) -> dict:
    """
    Build a high-level finite problem workflow report.
    """
    state = np.asarray(parse_complex_list(args.state)) if args.state else None
    rhs = np.asarray(parse_complex_list(args.rhs)) if args.rhs else None
    source = np.asarray(parse_complex_list(args.source)) if args.source else None
    result = qsvt_problem_workflow(
        args.target,
        np.asarray(parse_matrix(args.matrix)),
        rhs=rhs,
        state=state,
        source=source,
        degree=args.degree,
        gamma=args.gamma,
        lower=args.lower,
        upper=args.upper,
        cutoff=args.cutoff,
        sharpness=args.sharpness,
        width=args.width,
        center=args.center,
        time=args.time,
        omega=args.omega,
        eta=args.eta,
        num_points=args.num_points,
        bounded_num_points=args.bounded_num_points,
        attempt_synthesis=args.attempt_synthesis,
        apply_qsvt=args.apply_qsvt,
    )
    return result.as_report()


def register_workflow_commands(sub, problem_workflow_targets) -> None:
    p_linear_compare = sub.add_parser(
        "linear-system-compare",
        help="Compare dense, CG, and QSVT-style positive linear-system solves",
    )

    p_linear_compare.add_argument(
        "--matrix",
        type=str,
        required=True,
        help='Rows separated by semicolons, e.g. "2,0.25;0.25,1.25"',
    )

    p_linear_compare.add_argument(
        "--rhs",
        type=str,
        required=True,
        help='Right-hand side vector, e.g. "1,-0.5"',
    )

    p_linear_compare.add_argument("--degree", type=int, required=True)

    p_linear_compare.add_argument("--gamma", type=float, default=None)

    p_linear_compare.add_argument(
        "--num-points",
        dest="num_points",
        type=int,
        default=2001,
    )

    p_linear_compare.add_argument(
        "--bounded-num-points",
        dest="bounded_num_points",
        type=int,
        default=4001,
    )

    p_linear_compare.add_argument(
        "--no-synthesis",
        dest="attempt_synthesis",
        action="store_false",
        help="Skip the PennyLane synthesis attempt.",
    )

    p_linear_compare.add_argument(
        "--no-qsvt",
        dest="apply_qsvt",
        action="store_false",
        help="Skip the PennyLane QSVT matrix check.",
    )

    p_linear_compare.add_argument(
        "--no-cg",
        action="store_true",
        help="Omit the conjugate-gradient comparison row.",
    )

    p_linear_compare.add_argument("--cg-tolerance", type=float, default=1e-10)

    p_linear_compare.add_argument("--cg-max-iterations", type=int, default=None)

    p_linear_compare.add_argument(
        "--rows-output",
        type=str,
        help="Optional path for writing compact comparison rows as CSV.",
    )

    add_report_output_args(p_linear_compare)

    p_linear_compare.set_defaults(func=cmd_linear_system_compare)

    p_problem_workflow = sub.add_parser(
        "problem-workflow",
        help="Run a high-level finite QSVT problem workflow",
    )

    p_problem_workflow.add_argument(
        "--target",
        choices=problem_workflow_targets,
        required=True,
        help="Problem workflow target transform.",
    )

    p_problem_workflow.add_argument(
        "--matrix",
        type=str,
        required=True,
        help='Rows separated by semicolons, e.g. "2,0;0,1".',
    )

    p_problem_workflow.add_argument(
        "--rhs",
        type=str,
        default=None,
        help='Optional right-hand side vector, e.g. "1,1".',
    )

    p_problem_workflow.add_argument(
        "--state",
        type=str,
        default=None,
        help='Optional input state vector, e.g. "1,0".',
    )

    p_problem_workflow.add_argument(
        "--source",
        type=str,
        default=None,
        help='Optional source vector for response workflows, e.g. "1,0".',
    )

    p_problem_workflow.add_argument("--degree", type=int, required=True)

    p_problem_workflow.add_argument("--gamma", type=float, default=None)

    p_problem_workflow.add_argument("--lower", type=float, default=None)

    p_problem_workflow.add_argument("--upper", type=float, default=None)

    p_problem_workflow.add_argument("--cutoff", type=float, default=None)

    p_problem_workflow.add_argument("--sharpness", type=float, default=12.0)

    p_problem_workflow.add_argument("--width", type=float, default=0.25)

    p_problem_workflow.add_argument("--center", type=float, default=-1.0)

    p_problem_workflow.add_argument("--time", type=float, default=None)

    p_problem_workflow.add_argument("--omega", type=float, default=None)

    p_problem_workflow.add_argument("--eta", type=float, default=None)

    p_problem_workflow.add_argument(
        "--num-points",
        dest="num_points",
        type=int,
        default=2001,
    )

    p_problem_workflow.add_argument(
        "--bounded-num-points",
        dest="bounded_num_points",
        type=int,
        default=4001,
    )

    p_problem_workflow.add_argument(
        "--no-synthesis",
        dest="attempt_synthesis",
        action="store_false",
        help="Skip the PennyLane synthesis attempt where applicable.",
    )

    p_problem_workflow.add_argument(
        "--apply-qsvt",
        dest="apply_qsvt",
        action="store_true",
        help="Attempt the optional PennyLane QSVT matrix check where supported.",
    )

    p_problem_workflow.add_argument(
        "--no-qsvt",
        dest="apply_qsvt",
        action="store_false",
        help="Skip the optional PennyLane QSVT matrix check.",
    )

    p_problem_workflow.set_defaults(apply_qsvt=False)

    add_report_output_args(p_problem_workflow)

    p_problem_workflow.set_defaults(func=cmd_problem_workflow)

    p_threshold = sub.add_parser(
        "threshold-workflow",
        help="Build a spectral thresholding / interval-projector workflow report",
    )

    p_threshold.add_argument(
        "--matrix",
        type=str,
        required=True,
        help='Rows separated by semicolons, e.g. "-1,0,0;0,0,0;0,0,1"',
    )

    p_threshold.add_argument(
        "--lower",
        type=float,
        required=True,
        help="Lower physical eigenvalue threshold for the selected interval.",
    )

    p_threshold.add_argument(
        "--upper",
        type=float,
        required=True,
        help="Upper physical eigenvalue threshold for the selected interval.",
    )

    p_threshold.add_argument("--degree", type=int, required=True)

    p_threshold.add_argument("--sharpness", type=float, default=12.0)

    p_threshold.add_argument(
        "--state",
        type=str,
        default=None,
        help='Optional state vector, e.g. "1,0,0".',
    )

    p_threshold.add_argument(
        "--num-points",
        dest="num_points",
        type=int,
        default=2001,
    )

    p_threshold.add_argument(
        "--bounded-num-points",
        dest="bounded_num_points",
        type=int,
        default=4001,
    )

    add_report_output_args(p_threshold)

    p_threshold.set_defaults(func=cmd_threshold_workflow)
