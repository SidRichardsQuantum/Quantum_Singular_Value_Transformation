"""Cookbook example: diagnose QSVT polynomial compatibility failures."""

from __future__ import annotations

import argparse
from pathlib import Path

from qsvt.compatibility import qsvt_compatibility_report
from qsvt.reports import save_report


def build_report() -> dict[str, object]:
    """Build a report for a bounded but mixed-parity polynomial."""
    report = qsvt_compatibility_report(
        [0.25, 0.25],
        bounded_num_points=801,
        attempt_synthesis=False,
    )
    report["example"] = "compatibility-report"
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/examples/compatibility_report.json"),
        help="Destination JSON report path.",
    )
    args = parser.parse_args(argv)

    written = save_report(build_report(), args.output)
    print(written)


if __name__ == "__main__":
    main()
