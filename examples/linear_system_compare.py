"""Cookbook example: compare linear-system solver paths."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qsvt.algorithms import (
    linear_system_comparison_workflow,
    write_linear_system_comparison_csv,
)
from qsvt.reports import save_report


def build_comparison():
    """Run a small positive-definite linear-system comparison."""
    matrix = np.array(
        [
            [2.0, 0.25],
            [0.25, 1.25],
        ]
    )
    rhs = np.array([1.0, -0.5])
    return linear_system_comparison_workflow(
        matrix,
        rhs,
        degree=8,
        num_points=401,
        bounded_num_points=801,
        attempt_synthesis=False,
        apply_qsvt=False,
        cg_tolerance=1e-12,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/examples/linear_system_compare.json"),
        help="Destination JSON report path.",
    )
    parser.add_argument(
        "--rows-output",
        type=Path,
        default=Path("results/examples/linear_system_compare.csv"),
        help="Destination CSV summary path.",
    )
    args = parser.parse_args(argv)

    comparison = build_comparison()
    report = comparison.as_report()
    report["example"] = "linear-system-compare"
    json_path = save_report(report, args.output)
    csv_path = write_linear_system_comparison_csv(comparison, args.rows_output)
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
