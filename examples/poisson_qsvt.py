"""Cookbook example: compare direct, CG, and QSVT Poisson solve paths."""

from __future__ import annotations

import argparse
from pathlib import Path

from qsvt.stable import poisson_qsvt_workflow, save_report


def build_report(*, execute: bool = True) -> dict[str, object]:
    """Build a four-point Dirichlet Poisson workflow report."""
    result = poisson_qsvt_workflow(
        4,
        tolerance=0.4,
        min_degree=5,
        max_degree=5,
        access_model="prepselprep",
        num_points=401,
        execute=execute,
    )
    report = result.as_report()
    report["example"] = "poisson-qsvt"
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/examples/poisson_qsvt.json"),
        help="Destination JSON report path.",
    )
    parser.add_argument("--no-execute", dest="execute", action="store_false")
    parser.set_defaults(execute=True)
    args = parser.parse_args(argv)

    report = build_report(execute=args.execute)
    written = save_report(report, args.output)
    print(written)
    acceptance = report["acceptance"]
    print(
        "acceptance: "
        f"{acceptance['status']} "
        f"(scope={acceptance['scope']}, "
        f"full_qsvt={acceptance['full_qsvt_acceptance']})"
    )


if __name__ == "__main__":
    main()
