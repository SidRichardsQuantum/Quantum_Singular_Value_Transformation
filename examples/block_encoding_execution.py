"""Cookbook example: execute QSVT from a block-encoding specification."""

from __future__ import annotations

import argparse
from pathlib import Path

import pennylane as qml

from qsvt.block_encoding import pennylane_operator_block_encoding_spec
from qsvt.execution import execute_qsvt_from_spec
from qsvt.reports import save_report


def build_report() -> dict[str, object]:
    """Execute an LCU Hamiltonian through the specification-based QSVT path."""
    operator = qml.dot([0.3, 0.7], [qml.Z(1), qml.X(1)])
    spec = pennylane_operator_block_encoding_spec(
        operator,
        encoding_wires=[0],
        block_encoding="prepselprep",
    )
    result = execute_qsvt_from_spec(
        spec,
        [0.0, 1.0],
        [1.0, 0.0],
    )
    report = result.as_report()
    report["example"] = "block-encoding-execution"
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/examples/block_encoding_execution.json"),
        help="Destination JSON report path.",
    )
    args = parser.parse_args(argv)

    written = save_report(build_report(), args.output)
    print(written)


if __name__ == "__main__":
    main()
