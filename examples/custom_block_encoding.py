"""Cookbook example: execute QSVT with a custom block-encoding circuit."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pennylane as qml

from qsvt.block_encoding import circuit_block_encoding_spec
from qsvt.execution import execute_qsvt_from_spec
from qsvt.reports import save_report


def build_report() -> dict[str, object]:
    """Execute an identity polynomial on a known one-qubit signal block."""
    signal_value = 0.4
    rotation_angle = 2.0 * np.arccos(signal_value)

    def block_encoding_circuit():
        return qml.RY(rotation_angle, wires=0)

    spec = circuit_block_encoding_spec(
        block_encoding_circuit,
        logical_shape=(1, 1),
        encoding_wires=[0],
        alpha=1.0,
        metadata={
            "known_signal_value": signal_value,
            "signal_subspace": "leading computational-basis amplitude",
        },
    )
    poly = np.array([0.0, 1.0])
    angles = qml.poly_to_angles(poly, "QSVT")
    projectors = [
        qml.PCPhase(float(angle), dim=1, wires=spec.encoding_wires) for angle in angles
    ]
    result = execute_qsvt_from_spec(
        spec,
        poly,
        [1.0],
        projectors=projectors,
    )
    if not result.succeeded or result.logical_output is None:
        raise RuntimeError(result.error or "custom block-encoding execution failed")

    encoded_unitary = np.asarray(qml.matrix(qml.RY(rotation_angle, wires=0)))
    expected_output = np.array([signal_value], dtype=complex)
    report = result.as_report()
    report.update(
        {
            "example": "custom-block-encoding",
            "known_block_validation": {
                "encoded_top_left": encoded_unitary[0, 0],
                "expected_signal": signal_value,
                "block_absolute_error": abs(encoded_unitary[0, 0] - signal_value),
                "expected_logical_output": expected_output,
                "logical_output_absolute_error": float(
                    np.linalg.norm(result.logical_output - expected_output)
                ),
                "unitarity_error": float(
                    np.linalg.norm(
                        encoded_unitary.conj().T @ encoded_unitary
                        - np.eye(encoded_unitary.shape[0])
                    )
                ),
            },
            "caller_supplied_projector_angles": angles,
        }
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/examples/custom_block_encoding.json"),
        help="Destination JSON report path.",
    )
    args = parser.parse_args(argv)

    written = save_report(build_report(), args.output)
    print(written)


if __name__ == "__main__":
    main()
