"""Cookbook example: run a finite block-encoded QSVT workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qsvt.algorithms import block_encoded_qsvt_workflow
from qsvt.matrices import rotated_diagonal
from qsvt.reports import save_report


def build_report() -> dict[str, object]:
    """Build a compact verified dense block-encoding workflow report."""
    matrix = rotated_diagonal([0.25, 0.8], theta=0.31)
    coeffs = np.array([0.0, 0.0, 1.0])
    state = np.array([1.0, -0.25])

    result = block_encoded_qsvt_workflow(matrix, coeffs, state=state)
    report = result.as_report()
    report["example"] = "block-encoded-workflow"
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/examples/block_encoded_workflow.json"),
        help="Destination JSON report path.",
    )
    args = parser.parse_args(argv)

    written = save_report(build_report(), args.output)
    print(written)


if __name__ == "__main__":
    main()
