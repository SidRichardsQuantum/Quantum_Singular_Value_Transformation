"""Cookbook example: execute an accuracy-driven Pauli-LCU spectral filter."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pennylane as qml

from qsvt.flagship import spectral_filter_qsvt_workflow
from qsvt.reports import save_report


def build_report(*, execute: bool = True) -> dict[str, object]:
    """Build a band-filter report with a hard-projector reference."""
    operator = qml.dot(
        [0.4, 0.3, 0.2],
        [qml.Z(0), qml.Z(1), qml.X(0)],
    )
    result = spectral_filter_qsvt_workflow(
        operator,
        np.ones(4) / 2.0,
        lower=-0.4,
        upper=0.4,
        tolerance=0.16,
        min_degree=2,
        max_degree=4,
        num_points=401,
        execute=execute,
    )
    report = result.as_report()
    report["example"] = "spectral-filter-qsvt"
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/examples/spectral_filter_qsvt.json"),
        help="Destination JSON report path.",
    )
    parser.add_argument("--no-execute", dest="execute", action="store_false")
    parser.set_defaults(execute=True)
    args = parser.parse_args(argv)

    written = save_report(build_report(execute=args.execute), args.output)
    print(written)


if __name__ == "__main__":
    main()
