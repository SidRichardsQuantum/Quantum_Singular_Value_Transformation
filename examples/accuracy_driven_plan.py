"""Cookbook example: plan and execute QSVT from a requested accuracy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qsvt.planning import (
    QSVTExecutionConfig,
    QSVTProblemSpec,
    QSVTTransformSpec,
    plan_qsvt,
    run_qsvt_plan,
)
from qsvt.reports import save_report


def build_report(*, execute: bool = True) -> dict[str, object]:
    """Build an accuracy-driven linear-system plan and optional execution."""
    problem = QSVTProblemSpec(
        np.diag([1.0, 2.0]),
        rhs=np.ones(2),
        observables={"population_0": np.diag([1.0, 0.0])},
        name="two-level-linear-system",
    )
    transform = QSVTTransformSpec(
        "linear_system",
        tolerance=0.4,
        min_degree=3,
        max_degree=9,
        degree_step=2,
        parameters={"num_points": 201, "bounded_num_points": 401},
    )
    config = QSVTExecutionConfig(
        execute=execute,
        angle_solvers=("root-finding", "iterative"),
        reconstruction_num_points=65,
    )
    plan = plan_qsvt(problem, transform, config)
    execution = run_qsvt_plan(plan)
    candidates = [candidate.as_report() for candidate in plan.degree_candidates]

    return {
        "example": "accuracy-driven-plan",
        "mode": "accuracy-driven-plan-cookbook",
        "summary": {
            "selected_degree": plan.selected_degree,
            "achieved_error": plan.achieved_error,
            "met_tolerance": plan.met_tolerance,
            "access_model_status": plan.access_model_status,
            "execution_ready": plan.execution_ready,
            "execution_succeeded": execution.succeeded,
        },
        "degree_candidates": candidates,
        "plan": plan.as_report(),
        "execution": execution.as_report(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/examples/accuracy_driven_plan.json"),
        help="Destination JSON report path.",
    )
    parser.add_argument("--no-execute", dest="execute", action="store_false")
    parser.set_defaults(execute=True)
    args = parser.parse_args(argv)

    written = save_report(build_report(execute=args.execute), args.output)
    print(written)


if __name__ == "__main__":
    main()
