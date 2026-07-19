"""Cookbook example: compare ideal and finite-shot QSVT on a local device."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pennylane as qml

from qsvt.block_encoding import matrix_block_encoding_spec
from qsvt.execution import execute_qsvt_from_spec
from qsvt.hardware import execute_qsvt_on_device
from qsvt.reports import save_report


def build_report(*, shots: int = 2000, seed: int = 12345) -> dict[str, object]:
    """Run a credential-free finite-shot FABLE validation."""
    matrix = np.diag([0.2, 0.8])
    spec = matrix_block_encoding_spec(
        matrix,
        alpha=1.6,
        block_encoding="fable",
        metadata={"example_input_basis_index": 1},
    )
    poly = np.array([0.0, 1.0])

    def prepare_basis_one() -> None:
        qml.PauliX(spec.encoding_wires[-1])

    device = qml.device(
        "default.qubit",
        wires=spec.encoding_wires,
        seed=seed,
    )
    sampled = execute_qsvt_on_device(
        spec,
        poly,
        prepare_basis_one,
        device,
        shots=shots,
    )
    ideal = execute_qsvt_from_spec(spec, poly, [0.0, 1.0])
    if not sampled.succeeded or sampled.logical_probabilities is None:
        raise RuntimeError(sampled.error or "finite-shot execution failed")
    if not ideal.succeeded or ideal.logical_probabilities is None:
        raise RuntimeError(ideal.error or "ideal execution failed")

    probability_error = float(
        np.linalg.norm(sampled.logical_probabilities - ideal.logical_probabilities)
    )
    return {
        "example": "finite-shot-qsvt",
        "mode": "finite-shot-qsvt-cookbook",
        "shots": shots,
        "seed": seed,
        "sampled_execution": sampled.as_report(),
        "ideal_execution": ideal.as_report(),
        "comparison": {
            "logical_probability_l2_error": probability_error,
            "ideal_logical_success_probability": ideal.logical_success_probability,
            "sampled_logical_success_probability": (
                sampled.logical_success_probability
            ),
            "reported_success_standard_error": (sampled.logical_success_standard_error),
        },
        "truth_contract": {
            "uses_local_simulator": True,
            "uses_finite_shots": True,
            "uses_real_hardware": False,
            "requires_provider_credentials": False,
            "purpose": (
                "Validate preflight, finite-shot probabilities, uncertainty, and "
                "ideal-reference agreement before selecting a provider device."
            ),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/examples/finite_shot_qsvt.json"),
        help="Destination JSON report path.",
    )
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args(argv)

    written = save_report(
        build_report(shots=args.shots, seed=args.seed),
        args.output,
    )
    print(written)


if __name__ == "__main__":
    main()
