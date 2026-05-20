"""
qsvt.cli
--------

Minimal CLI entry point for qsvt-pennylane.

This CLI is intentionally lightweight and focuses on quick verification
and demonstration of core functionality:

    python -m qsvt scalar --x 0.5 --poly "0,0,1"
    python -m qsvt diag --values "1.0,0.7,0.3,0.1" --poly "0,0,1"
    python -m qsvt cheb --degree 3 --x 0.5
    python -m qsvt design-report --kind sign --gamma 0.2 --degree 13
    python -m qsvt design-report --kind sign --gamma 0.2 --degree 13 \
        --output sign.json
    python -m qsvt template-report --kind inverse --degree 7 --mu 0.3
    python -m qsvt compatibility-report --poly "0,0,1"
    python -m qsvt design-compatibility --kind sign --degree 13 --gamma 0.2
    python -m qsvt compare-report --values "1.0,0.7,0.3,0.1" --poly "0,0,1"
    python -m qsvt matrix-report --matrix "0.3135,-0.235;-0.235,0.6865" \
        --poly "0,0,1"
    python -m qsvt apply-design --kind sign --values="-0.8,-0.3,0.3,0.8" \
        --degree 13
    python -m qsvt design-sweep --kind sign --degrees "5,9,13" \
        --gamma 0.2 --output sweep.json

The CLI is not intended to replace notebooks; it provides simple smoke
tests and reproducible command-line demonstrations.
"""

from __future__ import annotations

import argparse
from typing import Iterable

from ._cli_utils import (
    add_report_output_args,
    emit_cli_result,
    parse_complex_list,
    parse_float_list,
    parse_int_list,
    parse_matrix,
    parse_poly,
)
from .benchmarks import (
    conjugate_gradient_benchmark,
    dense_eigendecomposition_benchmark,
    dense_linear_solve_benchmark,
    polynomial_matrix_function_benchmark,
    spectral_matrix_function_benchmark,
)
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
from .polynomials import chebyshev_t, eval_polynomial
from .qsvt import (
    compare_qsvt_vs_classical_diagonal,
    qsvt_compatibility_report,
    qsvt_matrix_transform_report,
    qsvt_scalar_output,
    qsvt_transform_report,
)
from .resources import qsvt_resource_report
from .templates import (
    exponential_approximation_diagnostics,
    inverse_like_diagnostics,
    sign_approximation_diagnostics,
    soft_threshold_filter_diagnostics,
    sqrt_approximation_diagnostics,
)
from .workflow import design_workflow


def cmd_scalar(args: argparse.Namespace) -> dict:
    """
    Evaluate scalar QSVT output.
    """
    poly = parse_poly(args.poly)

    result = qsvt_scalar_output(
        args.x,
        poly,
        encoding_wires=[0],
    )

    classical = eval_polynomial(poly, args.x)

    return {
        "mode": "scalar",
        "input": args.x,
        "poly": poly,
        "qsvt": result,
        "classical": classical,
        "abs_error": abs(result - classical),
    }


def cmd_diag(args: argparse.Namespace) -> dict:
    """
    Apply QSVT to a diagonal matrix.
    """
    values = parse_float_list(args.values)
    poly = parse_poly(args.poly)

    comparison = compare_qsvt_vs_classical_diagonal(
        values,
        poly,
        encoding_wires=list(range(args.wires)),
    )

    return {
        "mode": "diagonal",
        "input": values,
        "poly": poly,
        **comparison,
    }


def cmd_cheb(args: argparse.Namespace) -> dict:
    """
    Evaluate Chebyshev polynomial T_n(x).
    """
    value = chebyshev_t(args.degree, args.x)

    return {
        "mode": "chebyshev",
        "degree": args.degree,
        "x": args.x,
        "value": float(value),
    }


def cmd_poly(args: argparse.Namespace) -> dict:
    """
    Evaluate polynomial directly.
    """
    poly = parse_poly(args.poly)

    value = eval_polynomial(poly, args.x)

    return {
        "mode": "polynomial",
        "poly": poly,
        "x": args.x,
        "value": float(value),
    }


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


def cmd_template_report(args: argparse.Namespace) -> dict:
    """
    Build a diagnostics report for a template polynomial.
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
        "mode": "template-report",
        "kind": args.kind,
        **report,
    }


def cmd_compare_report(args: argparse.Namespace) -> dict:
    """
    Build a QSVT-vs-classical transform report for explicit coefficients.
    """
    values = parse_float_list(args.values)
    poly = parse_poly(args.poly)

    return qsvt_transform_report(
        values,
        poly,
        encoding_wires=list(range(args.wires)),
    )


def cmd_matrix_report(args: argparse.Namespace) -> dict:
    """
    Build a QSVT-vs-classical transform report for a full Hermitian matrix.
    """
    matrix = parse_matrix(args.matrix)
    poly = parse_poly(args.poly)

    return qsvt_matrix_transform_report(
        matrix,
        poly,
        encoding_wires=list(range(args.wires)),
    )


def cmd_compatibility_report(args: argparse.Namespace) -> dict:
    """
    Build a QSVT compatibility report for explicit coefficients.
    """
    poly = parse_poly(args.poly)

    return qsvt_compatibility_report(
        poly,
        bounded_num_points=args.bounded_num_points,
        attempt_synthesis=args.attempt_synthesis,
    )


def cmd_resource_report(args: argparse.Namespace) -> dict:
    """
    Build a QSVT resource proxy report for explicit coefficients.
    """
    poly = parse_poly(args.poly)
    return qsvt_resource_report(
        poly,
        matrix_dimension=args.matrix_dimension,
        encoding_qubits=args.encoding_qubits,
        block_encoding=args.block_encoding,
        bounded_num_points=args.bounded_num_points,
        attempt_synthesis=args.attempt_synthesis,
    )


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


def cmd_benchmark_eigh(args: argparse.Namespace) -> dict:
    """
    Benchmark dense Hermitian eigendecomposition.
    """
    return dense_eigendecomposition_benchmark(
        parse_matrix(args.matrix),
        repeats=args.repeats,
    )


def cmd_benchmark_dense_solve(args: argparse.Namespace) -> dict:
    """
    Benchmark a dense direct linear solve.
    """
    qsvt_coeffs = parse_poly(args.qsvt_poly) if args.qsvt_poly else None
    return dense_linear_solve_benchmark(
        parse_matrix(args.matrix),
        parse_complex_list(args.rhs),
        repeats=args.repeats,
        qsvt_coeffs=qsvt_coeffs,
    )


def cmd_benchmark_cg_solve(args: argparse.Namespace) -> dict:
    """
    Benchmark conjugate gradients for a positive-definite system.
    """
    qsvt_coeffs = parse_poly(args.qsvt_poly) if args.qsvt_poly else None
    return conjugate_gradient_benchmark(
        parse_matrix(args.matrix),
        parse_complex_list(args.rhs),
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        repeats=args.repeats,
        qsvt_coeffs=qsvt_coeffs,
    )


def cmd_benchmark_polynomial(args: argparse.Namespace) -> dict:
    """
    Benchmark classical polynomial matrix-function evaluation.
    """
    return polynomial_matrix_function_benchmark(
        parse_matrix(args.matrix),
        parse_poly(args.poly),
        repeats=args.repeats,
        include_qsvt_proxy=not args.no_qsvt_proxy,
    )


def cmd_benchmark_spectral_function(args: argparse.Namespace) -> dict:
    """
    Benchmark a dense spectral matrix function.
    """
    return spectral_matrix_function_benchmark(
        parse_matrix(args.matrix),
        args.function,
        repeats=args.repeats,
        beta=args.beta,
        shift=args.shift,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qsvt",
        description="Minimal CLI for qsvt-pennylane",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_scalar = sub.add_parser(
        "scalar",
        help="Evaluate scalar QSVT polynomial",
    )
    p_scalar.add_argument("--x", type=float, required=True)
    p_scalar.add_argument(
        "--poly",
        type=str,
        required=True,
        help='Polynomial coefficients, e.g. "0,0,1"',
    )
    p_scalar.set_defaults(func=cmd_scalar)

    p_diag = sub.add_parser(
        "diag",
        help="Apply QSVT to diagonal matrix",
    )
    p_diag.add_argument(
        "--values",
        type=str,
        required=True,
        help='Diagonal entries, e.g. "1.0,0.7,0.3,0.1"',
    )
    p_diag.add_argument(
        "--poly",
        type=str,
        required=True,
        help='Polynomial coefficients, e.g. "0,0,1"',
    )
    p_diag.add_argument(
        "--wires",
        type=int,
        default=3,
        help="Number of qubits for block encoding",
    )
    p_diag.set_defaults(func=cmd_diag)

    p_cheb = sub.add_parser(
        "cheb",
        help="Evaluate Chebyshev polynomial",
    )
    p_cheb.add_argument("--degree", type=int, required=True)
    p_cheb.add_argument("--x", type=float, required=True)
    p_cheb.set_defaults(func=cmd_cheb)

    p_poly = sub.add_parser(
        "poly",
        help="Evaluate polynomial directly",
    )
    p_poly.add_argument("--x", type=float, required=True)
    p_poly.add_argument(
        "--poly",
        type=str,
        required=True,
    )
    p_poly.set_defaults(func=cmd_poly)

    p_design_report = sub.add_parser(
        "design-report",
        help="Build a design diagnostics report",
    )
    p_design_report.add_argument(
        "--kind",
        choices=[
            "inverse",
            "sign",
            "projector",
            "sqrt",
            "power",
            "filter",
            "interval_projector",
        ],
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
        choices=[
            "inverse",
            "sign",
            "projector",
            "sqrt",
            "power",
            "filter",
            "interval_projector",
        ],
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
        choices=[
            "inverse",
            "sign",
            "projector",
            "sqrt",
            "power",
            "filter",
            "interval_projector",
        ],
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

    p_template_report = sub.add_parser(
        "template-report",
        help="Build a template diagnostics report",
    )
    p_template_report.add_argument(
        "--kind",
        choices=["inverse", "sign", "filter", "sqrt", "exponential"],
        required=True,
    )
    p_template_report.add_argument("--degree", type=int, required=True)
    p_template_report.add_argument("--mu", type=float, default=0.25)
    p_template_report.add_argument("--sharpness", type=float, default=6.0)
    p_template_report.add_argument("--threshold", type=float, default=0.5)
    p_template_report.add_argument("--beta", type=float, default=1.0)
    p_template_report.add_argument(
        "--num-points",
        dest="num_points",
        type=int,
        default=2001,
    )
    p_template_report.add_argument(
        "--bounded-num-points",
        dest="bounded_num_points",
        type=int,
        default=4001,
    )
    add_report_output_args(p_template_report, include_plot=True)
    p_template_report.set_defaults(func=cmd_template_report)

    p_compare_report = sub.add_parser(
        "compare-report",
        help="Compare QSVT and classical transforms for explicit coefficients",
    )
    p_compare_report.add_argument(
        "--values",
        type=str,
        required=True,
        help='Diagonal entries, e.g. "1.0,0.7,0.3,0.1"',
    )
    p_compare_report.add_argument(
        "--poly",
        type=str,
        required=True,
        help='Polynomial coefficients, e.g. "0,0,1"',
    )
    p_compare_report.add_argument(
        "--wires",
        type=int,
        default=3,
        help="Number of qubits for block encoding",
    )
    add_report_output_args(p_compare_report)
    p_compare_report.set_defaults(func=cmd_compare_report)

    p_matrix_report = sub.add_parser(
        "matrix-report",
        help="Compare QSVT and classical transforms for a Hermitian matrix",
    )
    p_matrix_report.add_argument(
        "--matrix",
        type=str,
        required=True,
        help='Rows separated by semicolons, e.g. "0.5,0.1;0.1,0.3"',
    )
    p_matrix_report.add_argument(
        "--poly",
        type=str,
        required=True,
        help='Polynomial coefficients, e.g. "0,0,1"',
    )
    p_matrix_report.add_argument(
        "--wires",
        type=int,
        default=2,
        help="Number of qubits for block encoding",
    )
    add_report_output_args(p_matrix_report)
    p_matrix_report.set_defaults(func=cmd_matrix_report)

    p_compatibility = sub.add_parser(
        "compatibility-report",
        help="Check QSVT compatibility for explicit coefficients",
    )
    p_compatibility.add_argument(
        "--poly",
        type=str,
        required=True,
        help='Polynomial coefficients, e.g. "0,0,1"',
    )
    p_compatibility.add_argument(
        "--bounded-num-points",
        dest="bounded_num_points",
        type=int,
        default=4001,
    )
    p_compatibility.add_argument(
        "--no-synthesis",
        dest="attempt_synthesis",
        action="store_false",
        help="Skip the PennyLane synthesis attempt.",
    )
    add_report_output_args(p_compatibility)
    p_compatibility.set_defaults(func=cmd_compatibility_report)

    p_resource = sub.add_parser(
        "resource-report",
        help="Build a QSVT resource proxy report for explicit coefficients",
    )
    p_resource.add_argument(
        "--poly",
        type=str,
        required=True,
        help='Polynomial coefficients, e.g. "0,0,1"',
    )
    p_resource.add_argument(
        "--matrix-dimension",
        dest="matrix_dimension",
        type=int,
        default=None,
        help="Optional encoded matrix dimension.",
    )
    p_resource.add_argument(
        "--encoding-qubits",
        dest="encoding_qubits",
        type=int,
        default=None,
        help="Optional number of matrix-register encoding qubits.",
    )
    p_resource.add_argument(
        "--block-encoding",
        dest="block_encoding",
        type=str,
        default="dense-block-encoding",
        help="Descriptive block-encoding label for the report.",
    )
    p_resource.add_argument(
        "--bounded-num-points",
        dest="bounded_num_points",
        type=int,
        default=4001,
    )
    p_resource.add_argument(
        "--no-synthesis",
        dest="attempt_synthesis",
        action="store_false",
        help="Skip the PennyLane synthesis attempt.",
    )
    add_report_output_args(p_resource)
    p_resource.set_defaults(func=cmd_resource_report)

    p_design_compatibility = sub.add_parser(
        "design-compatibility",
        help="Build a design polynomial and check QSVT compatibility",
    )
    p_design_compatibility.add_argument(
        "--kind",
        choices=[
            "inverse",
            "sign",
            "projector",
            "sqrt",
            "power",
            "filter",
            "interval_projector",
        ],
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
        choices=[
            "inverse",
            "sign",
            "projector",
            "sqrt",
            "power",
            "filter",
            "interval_projector",
        ],
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

    p_benchmark = sub.add_parser(
        "benchmark",
        help="Classical benchmark baselines for QSVT-oriented workflows",
    )
    benchmark_sub = p_benchmark.add_subparsers(
        dest="benchmark_command",
        required=True,
    )

    p_bench_eigh = benchmark_sub.add_parser(
        "eigh",
        help="Benchmark dense Hermitian eigendecomposition",
    )
    p_bench_eigh.add_argument(
        "--matrix",
        type=str,
        required=True,
        help='Rows separated by semicolons, e.g. "2,0;0,3"',
    )
    p_bench_eigh.add_argument("--repeats", type=int, default=3)
    add_report_output_args(p_bench_eigh)
    p_bench_eigh.set_defaults(func=cmd_benchmark_eigh)

    p_bench_dense_solve = benchmark_sub.add_parser(
        "dense-solve",
        help="Benchmark a dense direct linear solve",
    )
    p_bench_dense_solve.add_argument("--matrix", type=str, required=True)
    p_bench_dense_solve.add_argument(
        "--rhs",
        type=str,
        required=True,
        help='Right-hand side vector, e.g. "1,0"',
    )
    p_bench_dense_solve.add_argument("--repeats", type=int, default=3)
    p_bench_dense_solve.add_argument(
        "--qsvt-poly",
        type=str,
        help="Optional polynomial coefficients for an attached QSVT resource proxy.",
    )
    add_report_output_args(p_bench_dense_solve)
    p_bench_dense_solve.set_defaults(func=cmd_benchmark_dense_solve)

    p_bench_cg = benchmark_sub.add_parser(
        "cg-solve",
        help="Benchmark conjugate gradients for a positive-definite system",
    )
    p_bench_cg.add_argument("--matrix", type=str, required=True)
    p_bench_cg.add_argument("--rhs", type=str, required=True)
    p_bench_cg.add_argument("--tolerance", type=float, default=1e-10)
    p_bench_cg.add_argument("--max-iterations", type=int, default=None)
    p_bench_cg.add_argument("--repeats", type=int, default=3)
    p_bench_cg.add_argument(
        "--qsvt-poly",
        type=str,
        help="Optional polynomial coefficients for an attached QSVT resource proxy.",
    )
    add_report_output_args(p_bench_cg)
    p_bench_cg.set_defaults(func=cmd_benchmark_cg_solve)

    p_bench_polynomial = benchmark_sub.add_parser(
        "polynomial",
        help="Benchmark classical polynomial matrix-function evaluation",
    )
    p_bench_polynomial.add_argument("--matrix", type=str, required=True)
    p_bench_polynomial.add_argument("--poly", type=str, required=True)
    p_bench_polynomial.add_argument("--repeats", type=int, default=3)
    p_bench_polynomial.add_argument(
        "--no-qsvt-proxy",
        action="store_true",
        help="Do not attach a QSVT resource proxy to the benchmark report.",
    )
    add_report_output_args(p_bench_polynomial)
    p_bench_polynomial.set_defaults(func=cmd_benchmark_polynomial)

    p_bench_spectral = benchmark_sub.add_parser(
        "spectral-function",
        help="Benchmark dense spectral matrix-function evaluation",
    )
    p_bench_spectral.add_argument("--matrix", type=str, required=True)
    p_bench_spectral.add_argument(
        "--function",
        choices=[
            "inverse",
            "sqrt",
            "sign",
            "exponential",
            "imaginary_time",
            "positive_projector",
        ],
        required=True,
    )
    p_bench_spectral.add_argument("--beta", type=float, default=1.0)
    p_bench_spectral.add_argument("--shift", type=float, default=0.0)
    p_bench_spectral.add_argument("--repeats", type=int, default=3)
    add_report_output_args(p_bench_spectral)
    p_bench_spectral.set_defaults(func=cmd_benchmark_spectral_function)

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = args.func(args)
    emit_cli_result(args, result)


if __name__ == "__main__":
    main()
