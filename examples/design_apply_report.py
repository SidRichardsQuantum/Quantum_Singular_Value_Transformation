"""Cookbook example: design a polynomial, apply it, and save diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qsvt.reports import save_report
from qsvt.spectral import apply_polynomial_to_hermitian
from qsvt.workflow import design_workflow


def build_report() -> dict[str, object]:
    """Build a compact sign-polynomial workflow report."""
    matrix = np.diag([-0.8, -0.35, 0.35, 0.8])
    result = design_workflow(
        "sign",
        gamma=0.25,
        degree=13,
        num_points=401,
        bounded_num_points=801,
        attempt_synthesis=False,
    )
    transformed = apply_polynomial_to_hermitian(matrix, result.coeffs)

    report = result.as_report()
    report.update(
        {
            "example": "design-apply-report",
            "input_matrix": matrix,
            "transformed_matrix": transformed,
            "transformed_diagonal": np.diag(transformed),
        }
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/examples/design_apply_report.json"),
        help="Destination JSON report path.",
    )
    args = parser.parse_args(argv)

    written = save_report(build_report(), args.output)
    print(written)


if __name__ == "__main__":
    main()
