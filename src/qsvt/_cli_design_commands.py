"""Polynomial design and preset CLI commands."""

from __future__ import annotations

import argparse

from ._cli_utils import add_report_output_args, parse_float_list, parse_int_list
from .design import (
    design_filter_diagnostics,
    design_filter_polynomial,
    design_interval_projector_diagnostics,
    design_interval_projector_polynomial,
    design_inverse_diagnostics,
    design_inverse_polynomial,
    design_power_diagnostics,
    design_power_polynomial,
    design_projector_diagnostics,
    design_projector_polynomial,
    design_sign_diagnostics,
    design_sign_polynomial,
    design_sqrt_diagnostics,
    design_sqrt_polynomial,
)
from .presets import (
    exponential_approximation_diagnostics,
    inverse_like_diagnostics,
    sign_approximation_diagnostics,
    soft_threshold_filter_diagnostics,
    sqrt_approximation_diagnostics,
)
from .qsvt import qsvt_compatibility_report, qsvt_transform_report
from .workflow import design_workflow


def cmd_design_report(args: argparse.Namespace) -> dict:
    """
    Build a diagnostics report for a design polynomial.
    """
    builders = {
        "inverse": lambda: design_inverse_diagnostics(
            gamma=args.gamma,
            degree=args.degree,
            num_points=args.num_points,
            bounded_num_points=args.bounded_num_points,
        ),
        "sign": lambda: design_sign_diagnostics(
            gamma=args.gamma,
            degree=args.degree,
            num_points=args.num_points,
            bounded_num_points=args.bounded_num_points,
        ),
        "projector": lambda: design_projector_diagnostics(
            gamma=args.gamma,
            degree=args.degree,
            num_points=args.num_points,
            bounded_num_points=args.bounded_num_points,
        ),
        "sqrt": lambda: design_sqrt_diagnostics(
            a=args.a,
            degree=args.degree,
            num_points=args.num_points,
            bounded_num_points=args.bounded_num_points,
        ),
        "power": lambda: design_power_diagnostics(
            alpha=args.alpha,
            degree=args.degree,
            a=args.a,
            num_points=args.num_points,
            bounded_num_points=args.bounded_num_points,
        ),
        "filter": lambda: design_filter_diagnostics(
            cutoff=args.cutoff,
            degree=args.degree,
            sharpness=args.sharpness,
            num_points=args.num_points,
            bounded_num_points=args.bounded_num_points,
        ),
        "interval_projector": lambda: design_interval_projector_diagnostics(
            lower=args.lower,
            upper=args.upper,
            degree=args.degree,
            sharpness=args.sharpness,
            num_points=args.num_points,
            bounded_num_points=args.bounded_num_points,
        ),
    }

    report = builders[args.kind]()
    return {
        "mode": "design-report",
        "kind": args.kind,
        **report,
    }


def cmd_design_workflow(args: argparse.Namespace) -> dict:
    """
    Build a complete design workflow report.
    """
    result = design_workflow(
        args.kind,
        degree=args.degree,
        gamma=args.gamma,
        a=args.a,
        alpha=args.alpha,
        cutoff=args.cutoff,
        lower=args.lower,
        upper=args.upper,
        sharpness=args.sharpness,
        num_points=args.num_points,
        bounded_num_points=args.bounded_num_points,
        attempt_synthesis=args.attempt_synthesis,
    )
    return result.as_report()


def cmd_design_sweep(args: argparse.Namespace) -> dict:
    """
    Build a compact manifest for a degree/error/boundedness design sweep.
    """
    degrees = parse_int_list(args.degrees)
    rows = []
    for degree in degrees:
        result = design_workflow(
            args.kind,
            degree=degree,
            gamma=args.gamma,
            a=args.a,
            alpha=args.alpha,
            cutoff=args.cutoff,
            lower=args.lower,
            upper=args.upper,
            sharpness=args.sharpness,
            num_points=args.num_points,
            bounded_num_points=args.bounded_num_points,
            attempt_synthesis=args.attempt_synthesis,
        )
        diagnostics = result.diagnostics
        compatibility = result.compatibility
        rows.append(
            {
                "degree": int(degree),
                "builder": result.builder,
                "max_error": diagnostics.get("max_error"),
                "rms_error": diagnostics.get("rms_error"),
                "bounded": diagnostics.get("bounded"),
                "bounded_margin": diagnostics.get("bounded_margin"),
                "compatible": compatibility.get("compatible"),
                "compatibility_reasons": compatibility.get("reasons", []),
                "attempted_pennylane_synthesis": compatibility.get(
                    "attempted_pennylane_synthesis",
                ),
            }
        )

    return {
        "mode": "design-sweep",
        "kind": args.kind,
        "degrees": degrees,
        "parameters": {
            "gamma": args.gamma,
            "a": args.a,
            "alpha": args.alpha,
            "cutoff": args.cutoff,
            "lower": args.lower,
            "upper": args.upper,
            "sharpness": args.sharpness,
            "num_points": args.num_points,
            "bounded_num_points": args.bounded_num_points,
            "attempt_synthesis": args.attempt_synthesis,
        },
        "rows": rows,
    }


def _build_preset_report(args: argparse.Namespace, *, mode: str) -> dict:
    """
    Build a diagnostics report for a named preset polynomial.
    """
    builders = {
        "inverse": lambda: inverse_like_diagnostics(
            degree=args.degree,
            mu=args.mu,
            num_points=args.num_points,
            bounded_num_points=args.bounded_num_points,
        ),
        "sign": lambda: sign_approximation_diagnostics(
            degree=args.degree,
            sharpness=args.sharpness,
            num_points=args.num_points,
            bounded_num_points=args.bounded_num_points,
        ),
        "filter": lambda: soft_threshold_filter_diagnostics(
            degree=args.degree,
            threshold=args.threshold,
            sharpness=args.sharpness,
            num_points=args.num_points,
            bounded_num_points=args.bounded_num_points,
        ),
        "sqrt": lambda: sqrt_approximation_diagnostics(
            degree=args.degree,
            num_points=args.num_points,
            bounded_num_points=args.bounded_num_points,
        ),
        "exponential": lambda: exponential_approximation_diagnostics(
            degree=args.degree,
            beta=args.beta,
            num_points=args.num_points,
            bounded_num_points=args.bounded_num_points,
        ),
    }

    report = builders[args.kind]()
    return {
        "mode": mode,
        "kind": args.kind,
        **report,
    }


def cmd_preset_report(args: argparse.Namespace) -> dict:
    """Build a diagnostics report for a named preset polynomial."""
    return _build_preset_report(args, mode="preset-report")


def cmd_template_report(args: argparse.Namespace) -> dict:
    """Compatibility handler for the legacy ``template-report`` command."""
    return _build_preset_report(args, mode="template-report")


def cmd_design_compatibility(args: argparse.Namespace) -> dict:
    """
    Build a design polynomial and check QSVT compatibility.
    """
    builders = {
        "inverse": lambda: design_inverse_polynomial(
            gamma=args.gamma,
            degree=args.degree,
        ),
        "sign": lambda: design_sign_polynomial(
            gamma=args.gamma,
            degree=args.degree,
        ),
        "projector": lambda: design_projector_polynomial(
            gamma=args.gamma,
            degree=args.degree,
        ),
        "sqrt": lambda: design_sqrt_polynomial(
            a=args.a,
            degree=args.degree,
        ),
        "power": lambda: design_power_polynomial(
            alpha=args.alpha,
            degree=args.degree,
            a=args.a,
        ),
        "filter": lambda: design_filter_polynomial(
            cutoff=args.cutoff,
            degree=args.degree,
            sharpness=args.sharpness,
        ),
        "interval_projector": lambda: design_interval_projector_polynomial(
            lower=args.lower,
            upper=args.upper,
            degree=args.degree,
            sharpness=args.sharpness,
        ),
    }

    coeffs = builders[args.kind]()
    report = qsvt_compatibility_report(
        coeffs,
        bounded_num_points=args.bounded_num_points,
        attempt_synthesis=args.attempt_synthesis,
    )
    report.update(
        {
            "mode": "design-compatibility",
            "kind": args.kind,
            "builder": f"design_{args.kind}_polynomial",
        }
    )
    return report


def cmd_apply_design(args: argparse.Namespace) -> dict:
    """
    Build a design polynomial and compare its QSVT transform against classical.
    """
    builders = {
        "inverse": lambda: design_inverse_polynomial(
            gamma=args.gamma,
            degree=args.degree,
        ),
        "sign": lambda: design_sign_polynomial(
            gamma=args.gamma,
            degree=args.degree,
        ),
        "projector": lambda: design_projector_polynomial(
            gamma=args.gamma,
            degree=args.degree,
        ),
        "sqrt": lambda: design_sqrt_polynomial(
            a=args.a,
            degree=args.degree,
        ),
        "power": lambda: design_power_polynomial(
            alpha=args.alpha,
            degree=args.degree,
            a=args.a,
        ),
        "filter": lambda: design_filter_polynomial(
            cutoff=args.cutoff,
            degree=args.degree,
            sharpness=args.sharpness,
        ),
        "interval_projector": lambda: design_interval_projector_polynomial(
            lower=args.lower,
            upper=args.upper,
            degree=args.degree,
            sharpness=args.sharpness,
        ),
    }

    values = parse_float_list(args.values)
    coeffs = builders[args.kind]()
    compatibility = qsvt_compatibility_report(coeffs)
    report = qsvt_transform_report(
        values,
        coeffs,
        encoding_wires=list(range(args.wires)),
        allow_qsvt_failure=True,
    )
    report.update(
        {
            "mode": "apply-design",
            "kind": args.kind,
            "builder": f"design_{args.kind}_polynomial",
            "compatibility": compatibility,
        }
    )
    return report


def _add_preset_report_arguments(parser, preset_kinds) -> None:
    parser.add_argument(
        "--kind",
        choices=preset_kinds,
        required=True,
    )

    parser.add_argument("--degree", type=int, required=True)

    parser.add_argument("--mu", type=float, default=0.25)

    parser.add_argument("--sharpness", type=float, default=6.0)

    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--beta", type=float, default=1.0)

    parser.add_argument(
        "--num-points",
        dest="num_points",
        type=int,
        default=2001,
    )

    parser.add_argument(
        "--bounded-num-points",
        dest="bounded_num_points",
        type=int,
        default=4001,
    )

    add_report_output_args(parser, include_plot=True)


def register_design_commands(sub, design_kinds, preset_kinds) -> None:
    p_design_report = sub.add_parser(
        "design-report",
        help="Build a design diagnostics report",
    )

    p_design_report.add_argument(
        "--kind",
        choices=design_kinds,
        required=True,
    )

    p_design_report.add_argument("--degree", type=int, required=True)

    p_design_report.add_argument("--gamma", type=float, default=0.25)

    p_design_report.add_argument("--a", type=float, default=0.2)

    p_design_report.add_argument("--alpha", type=float, default=0.5)

    p_design_report.add_argument("--cutoff", type=float, default=0.45)

    p_design_report.add_argument("--lower", type=float, default=-0.25)

    p_design_report.add_argument("--upper", type=float, default=0.25)

    p_design_report.add_argument("--sharpness", type=float, default=12.0)

    p_design_report.add_argument(
        "--num-points",
        dest="num_points",
        type=int,
        default=2001,
    )

    p_design_report.add_argument(
        "--bounded-num-points",
        dest="bounded_num_points",
        type=int,
        default=4001,
    )

    add_report_output_args(p_design_report, include_plot=True)

    p_design_report.set_defaults(func=cmd_design_report)

    p_design_workflow = sub.add_parser(
        "design-workflow",
        help="Build coefficients, diagnostics, and compatibility metadata",
    )

    p_design_workflow.add_argument(
        "--kind",
        choices=design_kinds,
        required=True,
    )

    p_design_workflow.add_argument("--degree", type=int, required=True)

    p_design_workflow.add_argument("--gamma", type=float, default=0.25)

    p_design_workflow.add_argument("--a", type=float, default=0.2)

    p_design_workflow.add_argument("--alpha", type=float, default=0.5)

    p_design_workflow.add_argument("--cutoff", type=float, default=0.45)

    p_design_workflow.add_argument("--lower", type=float, default=-0.25)

    p_design_workflow.add_argument("--upper", type=float, default=0.25)

    p_design_workflow.add_argument("--sharpness", type=float, default=12.0)

    p_design_workflow.add_argument(
        "--num-points",
        dest="num_points",
        type=int,
        default=2001,
    )

    p_design_workflow.add_argument(
        "--bounded-num-points",
        dest="bounded_num_points",
        type=int,
        default=4001,
    )

    p_design_workflow.add_argument(
        "--no-synthesis",
        dest="attempt_synthesis",
        action="store_false",
        help="Skip the PennyLane synthesis attempt.",
    )

    add_report_output_args(p_design_workflow)

    p_design_workflow.set_defaults(func=cmd_design_workflow)

    p_design_sweep = sub.add_parser(
        "design-sweep",
        help="Build a compact degree/error/boundedness sweep manifest",
    )

    p_design_sweep.add_argument(
        "--kind",
        choices=design_kinds,
        required=True,
    )

    p_design_sweep.add_argument(
        "--degrees",
        type=str,
        required=True,
        help='Comma-separated polynomial degrees, e.g. "5,9,13"',
    )

    p_design_sweep.add_argument("--gamma", type=float, default=0.25)

    p_design_sweep.add_argument("--a", type=float, default=0.2)

    p_design_sweep.add_argument("--alpha", type=float, default=0.5)

    p_design_sweep.add_argument("--cutoff", type=float, default=0.45)

    p_design_sweep.add_argument("--lower", type=float, default=-0.25)

    p_design_sweep.add_argument("--upper", type=float, default=0.25)

    p_design_sweep.add_argument("--sharpness", type=float, default=12.0)

    p_design_sweep.add_argument(
        "--num-points",
        dest="num_points",
        type=int,
        default=2001,
    )

    p_design_sweep.add_argument(
        "--bounded-num-points",
        dest="bounded_num_points",
        type=int,
        default=4001,
    )

    p_design_sweep.add_argument(
        "--no-synthesis",
        dest="attempt_synthesis",
        action="store_false",
        help="Skip the PennyLane synthesis attempt.",
    )

    add_report_output_args(p_design_sweep)

    p_design_sweep.set_defaults(func=cmd_design_sweep)

    p_preset_report = sub.add_parser(
        "preset-report",
        help="Build a named-preset diagnostics report",
    )
    _add_preset_report_arguments(p_preset_report, preset_kinds)
    p_preset_report.set_defaults(func=cmd_preset_report)

    p_template_report = sub.add_parser(
        "template-report",
        help="Compatibility alias for preset-report",
    )
    _add_preset_report_arguments(p_template_report, preset_kinds)
    p_template_report.set_defaults(func=cmd_template_report)


def register_design_application_commands(sub, design_kinds) -> None:
    p_design_compatibility = sub.add_parser(
        "design-compatibility",
        help="Build a design polynomial and check QSVT compatibility",
    )

    p_design_compatibility.add_argument(
        "--kind",
        choices=design_kinds,
        required=True,
    )

    p_design_compatibility.add_argument("--degree", type=int, required=True)

    p_design_compatibility.add_argument("--gamma", type=float, default=0.25)

    p_design_compatibility.add_argument("--a", type=float, default=0.2)

    p_design_compatibility.add_argument("--alpha", type=float, default=0.5)

    p_design_compatibility.add_argument("--cutoff", type=float, default=0.45)

    p_design_compatibility.add_argument("--lower", type=float, default=-0.25)

    p_design_compatibility.add_argument("--upper", type=float, default=0.25)

    p_design_compatibility.add_argument("--sharpness", type=float, default=12.0)

    p_design_compatibility.add_argument(
        "--bounded-num-points",
        dest="bounded_num_points",
        type=int,
        default=4001,
    )

    p_design_compatibility.add_argument(
        "--no-synthesis",
        dest="attempt_synthesis",
        action="store_false",
        help="Skip the PennyLane synthesis attempt.",
    )

    add_report_output_args(p_design_compatibility)

    p_design_compatibility.set_defaults(func=cmd_design_compatibility)

    p_apply_design = sub.add_parser(
        "apply-design",
        help="Build a design polynomial and run a QSVT comparison report",
    )

    p_apply_design.add_argument(
        "--kind",
        choices=design_kinds,
        required=True,
    )

    p_apply_design.add_argument(
        "--values",
        type=str,
        required=True,
        help='Diagonal entries, e.g. "-0.8,-0.3,0.3,0.8"',
    )

    p_apply_design.add_argument("--degree", type=int, required=True)

    p_apply_design.add_argument("--gamma", type=float, default=0.25)

    p_apply_design.add_argument("--a", type=float, default=0.2)

    p_apply_design.add_argument("--alpha", type=float, default=0.5)

    p_apply_design.add_argument("--cutoff", type=float, default=0.45)

    p_apply_design.add_argument("--lower", type=float, default=-0.25)

    p_apply_design.add_argument("--upper", type=float, default=0.25)

    p_apply_design.add_argument("--sharpness", type=float, default=12.0)

    p_apply_design.add_argument(
        "--wires",
        type=int,
        default=3,
        help="Number of qubits for block encoding",
    )

    add_report_output_args(p_apply_design)

    p_apply_design.set_defaults(func=cmd_apply_design)
