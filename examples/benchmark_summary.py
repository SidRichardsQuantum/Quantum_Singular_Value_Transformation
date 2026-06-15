"""Cookbook example: run small classical baselines and export summary rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qsvt.benchmarks import (
    benchmark_summary_table,
    conjugate_gradient_benchmark,
    dense_linear_solve_benchmark,
    write_benchmark_summary_csv,
)
from qsvt.reports import save_report


def build_report() -> dict[str, object]:
    """Build a compact benchmark report bundle for a 2x2 linear system."""
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.array([1.0, 2.0])
    qsvt_coeffs = [0.0, 1.0]

    reports = [
        dense_linear_solve_benchmark(
            matrix,
            rhs,
            repeats=1,
            qsvt_coeffs=qsvt_coeffs,
        ),
        conjugate_gradient_benchmark(
            matrix,
            rhs,
            repeats=1,
            qsvt_coeffs=qsvt_coeffs,
        ),
    ]
    return {
        "example": "benchmark-summary",
        "mode": "benchmark-summary-bundle",
        "reports": reports,
        "summary_rows": benchmark_summary_table(reports),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/examples/benchmark_summary.json"),
        help="Destination JSON report path.",
    )
    parser.add_argument(
        "--rows-output",
        type=Path,
        default=Path("results/examples/benchmark_summary.csv"),
        help="Destination CSV summary table path.",
    )
    args = parser.parse_args(argv)

    report = build_report()
    written = save_report(report, args.output)
    rows_written = write_benchmark_summary_csv(
        report["summary_rows"],  # type: ignore[arg-type]
        args.rows_output,
    )
    print(written)
    print(rows_written)


if __name__ == "__main__":
    main()
