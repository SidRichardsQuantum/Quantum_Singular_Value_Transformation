"""Cookbook example: apply a spectral interval threshold workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qsvt.algorithms import spectral_thresholding_workflow
from qsvt.reports import save_report


def build_report() -> dict[str, object]:
    """Build a report for a small spectral interval projector."""
    matrix = np.diag([-1.0, -0.2, 0.15, 0.9])
    state = np.array([0.5, 0.5, 0.5, 0.5])
    result = spectral_thresholding_workflow(
        matrix,
        lower=-0.25,
        upper=0.25,
        degree=18,
        sharpness=10.0,
        state=state,
        num_points=401,
        bounded_num_points=801,
    )

    report = result.as_report()
    report["example"] = "threshold-filter"
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/examples/threshold_filter.json"),
        help="Destination JSON report path.",
    )
    args = parser.parse_args(argv)

    written = save_report(build_report(), args.output)
    print(written)


if __name__ == "__main__":
    main()
