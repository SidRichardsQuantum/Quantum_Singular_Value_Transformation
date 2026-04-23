"""
qsvt.__main__
-------------

Minimal CLI entry point for qsvt-pennylane.

This CLI is intentionally lightweight and focuses on quick verification
and demonstration of core functionality:

    python -m qsvt scalar --x 0.5 --poly "0,0,1"
    python -m qsvt diag --values "1.0,0.7,0.3,0.1" --poly "0,0,1"
    python -m qsvt cheb --degree 3 --x 0.5
    python -m qsvt design-report --kind sign --gamma 0.2 --degree 13
    python -m qsvt template-report --kind inverse --degree 7 --mu 0.3

The CLI is not intended to replace notebooks; it provides simple smoke
tests and reproducible command-line demonstrations.
"""

from __future__ import annotations

import argparse
import json
from typing import Iterable

import numpy as np

from .design import (
    design_filter_diagnostics,
    design_inverse_diagnostics,
    design_power_diagnostics,
    design_projector_diagnostics,
    design_sign_diagnostics,
    design_sqrt_diagnostics,
)
from .polynomials import chebyshev_t, eval_polynomial
from .qsvt import compare_qsvt_vs_classical_diagonal, qsvt_scalar_output
from .templates import (
    exponential_approximation_diagnostics,
    inverse_like_diagnostics,
    sign_approximation_diagnostics,
    soft_threshold_filter_diagnostics,
    sqrt_approximation_diagnostics,
)


def _parse_float_list(text: str) -> list[float]:
    """
    Parse a comma-separated list of floats.
    """
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_poly(text: str) -> list[float]:
    """
    Parse polynomial coefficients.

    Example
    -------
    "0,0,1"
    """
    return _parse_float_list(text)


def _to_jsonable(obj):
    """
    Recursively convert NumPy/Python mixed objects into JSON-serializable types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def cmd_scalar(args: argparse.Namespace) -> dict:
    """
    Evaluate scalar QSVT output.
    """
    poly = _parse_poly(args.poly)

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
    values = _parse_float_list(args.values)
    poly = _parse_poly(args.poly)

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
    poly = _parse_poly(args.poly)

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
    }

    report = builders[args.kind]()
    return {
        "mode": "design-report",
        "kind": args.kind,
        **report,
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
        choices=["inverse", "sign", "projector", "sqrt", "power", "filter"],
        required=True,
    )
    p_design_report.add_argument("--degree", type=int, required=True)
    p_design_report.add_argument("--gamma", type=float, default=0.25)
    p_design_report.add_argument("--a", type=float, default=0.2)
    p_design_report.add_argument("--alpha", type=float, default=0.5)
    p_design_report.add_argument("--cutoff", type=float, default=0.45)
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
    p_design_report.set_defaults(func=cmd_design_report)

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
    p_template_report.set_defaults(func=cmd_template_report)

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = args.func(args)
    print(json.dumps(_to_jsonable(result), indent=2))


if __name__ == "__main__":
    main()
