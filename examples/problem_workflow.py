"""Cookbook example: run high-level finite QSVT problem workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qsvt.reports import save_report
from qsvt.workflow import qsvt_problem_workflow


def build_report() -> dict[str, object]:
    """Build a compact multi-target problem-workflow report."""
    linear = qsvt_problem_workflow(
        "linear_system",
        np.array([[2.0, 0.0], [0.0, 1.0]]),
        rhs=np.array([1.0, 1.0]),
        degree=8,
        num_points=401,
        bounded_num_points=801,
        attempt_synthesis=False,
        apply_qsvt=False,
    )
    resolvent = qsvt_problem_workflow(
        "resolvent",
        np.diag([-0.5, 0.5]),
        omega=0.2,
        eta=0.4,
        degree=6,
        num_points=401,
    )

    return {
        "example": "problem-workflow",
        "mode": "problem-workflow-cookbook",
        "workflows": {
            "linear_system": linear.as_report(),
            "resolvent": resolvent.as_report(),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/examples/problem_workflow.json"),
        help="Destination JSON report path.",
    )
    args = parser.parse_args(argv)

    written = save_report(build_report(), args.output)
    print(written)


if __name__ == "__main__":
    main()
