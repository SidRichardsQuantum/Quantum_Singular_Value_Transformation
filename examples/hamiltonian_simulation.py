"""Cookbook example: validate Hamiltonian simulation's polynomial core."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qsvt.hamiltonians import tight_binding_chain
from qsvt.stable import hamiltonian_simulation_workflow, save_report


def build_report() -> dict[str, object]:
    """Build a six-site real-time evolution acceptance report."""
    hamiltonian = tight_binding_chain(6)
    initial_state = np.zeros(6, dtype=complex)
    initial_state[1] = 1.0
    result = hamiltonian_simulation_workflow(
        hamiltonian,
        initial_state,
        time=1.4,
        degree=12,
        num_points=401,
        acceptance_tolerance=1e-6,
    )
    report = result.as_report()
    report["example"] = "hamiltonian-simulation"
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/examples/hamiltonian_simulation.json"),
        help="Destination JSON report path.",
    )
    args = parser.parse_args(argv)

    report = build_report()
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
