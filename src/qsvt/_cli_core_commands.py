"""Core polynomial and finite-execution CLI commands."""

from __future__ import annotations

import argparse

import numpy as np

from ._cli_utils import (
    add_report_output_args,
    parse_complex_list,
    parse_float_list,
    parse_matrix,
    parse_poly,
)
from .block_encoding import matrix_block_encoding_spec
from .execution import execute_qsvt_from_spec
from .polynomials import chebyshev_t, eval_polynomial
from .qsvt import (
    compare_qsvt_vs_classical_diagonal,
    qsvt_matrix_transform_report,
    qsvt_scalar_output,
    qsvt_transform_report,
)


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
        np.asarray(matrix),
        poly,
        encoding_wires=list(range(args.wires)),
    )


def cmd_execute_spec(args: argparse.Namespace) -> dict:
    """Execute QSVT from a serializable block-encoding specification."""
    if args.kind != "matrix":  # defensive guard for direct function callers
        raise ValueError("execute-spec currently supports only --kind matrix.")
    matrix = np.asarray(parse_matrix(args.matrix))
    state = np.asarray(parse_complex_list(args.state))
    spec = matrix_block_encoding_spec(
        matrix,
        alpha=args.alpha,
        block_encoding=args.block_encoding,
    )
    result = execute_qsvt_from_spec(
        spec,
        parse_poly(args.poly),
        state,
        angle_solver=args.angle_solver,
        device_name=args.device,
        shots=args.shots,
        normalize_state=args.normalize_state,
    )
    return result.as_report()


def register_core_commands(sub) -> None:
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


def register_core_report_commands(sub) -> None:
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

    p_execute_spec = sub.add_parser(
        "execute-spec",
        help="Execute QSVT from a matrix block-encoding specification",
    )

    p_execute_spec.add_argument(
        "--kind",
        choices=["matrix"],
        default="matrix",
        help="Serializable block-encoding specification kind.",
    )

    p_execute_spec.add_argument(
        "--matrix",
        type=str,
        required=True,
        help='Rows separated by semicolons, e.g. "0.2,0;0,0.8".',
    )

    p_execute_spec.add_argument(
        "--poly",
        type=str,
        required=True,
        help='Polynomial coefficients, e.g. "0,0,1".',
    )

    p_execute_spec.add_argument(
        "--state",
        type=str,
        required=True,
        help='Logical input state, e.g. "1,0".',
    )

    p_execute_spec.add_argument("--alpha", type=float, default=None)

    p_execute_spec.add_argument(
        "--block-encoding",
        choices=["embedding", "fable"],
        default="embedding",
    )

    p_execute_spec.add_argument(
        "--angle-solver",
        choices=["root-finding", "iterative", "iterative-optax"],
        default="root-finding",
    )

    p_execute_spec.add_argument("--device", type=str, default="default.qubit")

    p_execute_spec.add_argument("--shots", type=int, default=None)

    p_execute_spec.add_argument(
        "--normalize-state",
        action="store_true",
        help="Normalize the supplied logical state before execution.",
    )

    add_report_output_args(p_execute_spec)

    p_execute_spec.set_defaults(func=cmd_execute_spec)
