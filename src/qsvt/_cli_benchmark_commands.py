"""Classical benchmark CLI commands."""

from __future__ import annotations

import argparse

from ._cli_utils import (
    add_report_output_args,
    parse_complex_list,
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


def register_benchmark_commands(sub) -> None:
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
