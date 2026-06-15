"""Cookbook example: execute a simulator-scale PennyLane QSVT QNode."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qsvt.execution import execute_qsvt_circuit
from qsvt.reports import save_report


def build_report() -> dict[str, object]:
    """Build a statevector QNode execution report for a small diagonal signal."""
    result = execute_qsvt_circuit(
        np.diag([0.2, 0.8]),
        [0.0, 0.0, 1.0],
        [1.0, 0.0],
        encoding_wires=[0, 1],
    )

    report = result.as_report()
    report["example"] = "circuit-execution"
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/examples/circuit_execution.json"),
        help="Destination JSON report path.",
    )
    args = parser.parse_args(argv)

    written = save_report(build_report(), args.output)
    print(written)


if __name__ == "__main__":
    main()
