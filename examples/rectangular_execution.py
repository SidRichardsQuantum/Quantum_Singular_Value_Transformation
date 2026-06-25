"""Cookbook example: execute a rectangular singular-value transformation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qsvt.block_encoding import matrix_block_encoding_spec
from qsvt.execution import execute_qsvt_from_spec
from qsvt.reports import save_report


def build_report() -> dict[str, object]:
    """Execute the identity singular-value polynomial on a 2-by-3 matrix."""
    matrix = np.array(
        [
            [0.2, 0.1, 0.0],
            [0.0, 0.3, 0.1],
        ]
    )
    spec = matrix_block_encoding_spec(matrix, alpha=1.0)
    result = execute_qsvt_from_spec(
        spec,
        [0.0, 1.0],
        [1.0, 0.0, 0.0],
    )
    report = result.as_report()
    report["example"] = "rectangular-execution"
    report["interpretation"] = (
        "The lower-level QSVT path alternates output and input signal-projector "
        "dimensions and validates the logical output against an SVD reference."
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/examples/rectangular_execution.json"),
        help="Destination JSON report path.",
    )
    args = parser.parse_args(argv)

    written = save_report(build_report(), args.output)
    print(written)


if __name__ == "__main__":
    main()
