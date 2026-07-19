"""Synthesis, resource, and schema CLI commands."""

from __future__ import annotations

import argparse

from ._cli_utils import add_report_output_args, parse_poly
from .qsvt import qsvt_compatibility_report
from .reports import (
    report_schema_manifest,
    supported_report_schemas,
    write_report_schema_manifest_csv,
)
from .resources import qsvt_resource_report
from .synthesis import (
    benchmark_phase_solvers,
    certify_polynomial_boundedness,
    synthesize_mixed_parity,
    synthesize_phases,
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


def cmd_phase_synthesis(args: argparse.Namespace) -> dict:
    """Synthesize QSP/QSVT phases for explicit polynomial coefficients."""
    result = synthesize_phases(
        parse_poly(args.poly),
        routine=args.routine,
        angle_solver=args.angle_solver,
        bounded_num_points=args.bounded_num_points,
        reconstruction_num_points=args.reconstruction_num_points,
    )
    return result.as_report()


def cmd_boundedness_certificate(args: argparse.Namespace) -> dict:
    """Certify polynomial boundedness from endpoints and derivative roots."""
    return certify_polynomial_boundedness(
        parse_poly(args.poly),
        domain=(args.lower, args.upper),
        bound=args.bound,
        tolerance=args.tolerance,
    ).as_report()


def cmd_phase_solver_benchmark(args: argparse.Namespace) -> dict:
    """Benchmark one or more PennyLane phase solvers."""
    solvers = tuple(
        solver.strip() for solver in args.solvers.split(",") if solver.strip()
    )
    return benchmark_phase_solvers(
        parse_poly(args.poly),
        solvers=solvers,
        routine=args.routine,
        repeats=args.repeats,
        reconstruction_num_points=args.reconstruction_num_points,
    ).as_report()


def cmd_mixed_parity_synthesis(args: argparse.Namespace) -> dict:
    """Synthesize separate parity components and report an LCU model."""
    return synthesize_mixed_parity(
        parse_poly(args.poly),
        angle_solver=args.angle_solver,
        reconstruction_num_points=args.reconstruction_num_points,
    ).as_report()


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


def cmd_report_schema_manifest(args: argparse.Namespace) -> dict:
    """
    Build a compatibility manifest for saved JSON report files.
    """
    rows = report_schema_manifest(args.paths)
    if args.csv_output:
        write_report_schema_manifest_csv(rows, args.csv_output)
        args.rows_output = args.csv_output
    all_supported = all(row["supported"] for row in rows)
    if args.fail_on_unsupported and not all_supported:
        failed = [row for row in rows if not row["supported"]]
        details = "; ".join(f"{row['path']}: {row['message']}" for row in failed)
        raise SystemExit(f"unsupported report schemas: {details}")
    return {
        "mode": "report-schema-manifest",
        "supported_schemas": supported_report_schemas(),
        "checked_paths": list(args.paths),
        "all_supported": all_supported,
        "rows": rows,
    }


def register_synthesis_commands(sub) -> None:
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

    p_synthesis = sub.add_parser(
        "phase-synthesis",
        help="Classify a polynomial and synthesize QSP/QSVT phase angles",
    )

    p_synthesis.add_argument(
        "--poly",
        type=str,
        required=True,
        help='Polynomial coefficients, e.g. "0,1,0,-0.5,0,0.333333"',
    )

    p_synthesis.add_argument(
        "--routine",
        choices=["QSP", "QSVT"],
        default="QSVT",
    )

    p_synthesis.add_argument(
        "--angle-solver",
        choices=["root-finding", "iterative", "iterative-optax"],
        default="root-finding",
    )

    p_synthesis.add_argument("--bounded-num-points", type=int, default=4001)

    p_synthesis.add_argument("--reconstruction-num-points", type=int, default=257)

    add_report_output_args(p_synthesis, include_plot=False)

    p_synthesis.set_defaults(func=cmd_phase_synthesis)

    p_certificate = sub.add_parser(
        "boundedness-certificate",
        help="Check polynomial extrema at endpoints and derivative roots",
    )

    p_certificate.add_argument("--poly", type=str, required=True)

    p_certificate.add_argument("--lower", type=float, default=-1.0)

    p_certificate.add_argument("--upper", type=float, default=1.0)

    p_certificate.add_argument("--bound", type=float, default=1.0)

    p_certificate.add_argument("--tolerance", type=float, default=1e-10)

    add_report_output_args(p_certificate, include_plot=False)

    p_certificate.set_defaults(func=cmd_boundedness_certificate)

    p_solver_benchmark = sub.add_parser(
        "phase-solver-benchmark",
        help="Compare phase-synthesis solvers for one polynomial",
    )

    p_solver_benchmark.add_argument("--poly", type=str, required=True)

    p_solver_benchmark.add_argument(
        "--solvers",
        type=str,
        default="root-finding,iterative",
        help="Comma-separated PennyLane angle solvers.",
    )

    p_solver_benchmark.add_argument(
        "--routine",
        choices=["QSP", "QSVT"],
        default="QSVT",
    )

    p_solver_benchmark.add_argument("--repeats", type=int, default=3)

    p_solver_benchmark.add_argument(
        "--reconstruction-num-points",
        type=int,
        default=65,
    )

    add_report_output_args(p_solver_benchmark, include_plot=False)

    p_solver_benchmark.set_defaults(func=cmd_phase_solver_benchmark)

    p_mixed = sub.add_parser(
        "mixed-parity-synthesis",
        help="Synthesize even and odd sequences with an LCU cost model",
    )

    p_mixed.add_argument("--poly", type=str, required=True)

    p_mixed.add_argument(
        "--angle-solver",
        choices=["root-finding", "iterative", "iterative-optax"],
        default="root-finding",
    )

    p_mixed.add_argument("--reconstruction-num-points", type=int, default=257)

    add_report_output_args(p_mixed, include_plot=False)

    p_mixed.set_defaults(func=cmd_mixed_parity_synthesis)

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

    p_report_schema = sub.add_parser(
        "report-schema-manifest",
        help="Audit saved JSON reports against supported schema metadata",
    )

    p_report_schema.add_argument(
        "--path",
        dest="paths",
        action="append",
        required=True,
        help="Report JSON path to audit. Repeat for multiple reports.",
    )

    p_report_schema.add_argument(
        "--csv-output",
        type=str,
        default=None,
        help="Optional path for writing compact manifest rows as CSV.",
    )

    p_report_schema.add_argument(
        "--fail-on-unsupported",
        action="store_true",
        help="Exit nonzero if any report is unsupported or malformed.",
    )

    add_report_output_args(p_report_schema, include_plot=False)

    p_report_schema.set_defaults(func=cmd_report_schema_manifest)
