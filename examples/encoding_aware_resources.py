"""Cookbook example: compare encoding-aware logical QSVT resources."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pennylane as qml

from qsvt.block_encoding import (
    matrix_block_encoding_spec,
    pennylane_operator_block_encoding_spec,
)
from qsvt.reports import save_report
from qsvt.resources import estimate_encoding_aware_resources


def build_report() -> dict[str, object]:
    """Sweep one operator across four access models and five degrees."""
    operator = qml.dot(
        [0.4, 0.3, 0.2],
        [qml.Z(2), qml.Z(3), qml.X(2)],
    )
    matrix = np.asarray(
        np.real_if_close(qml.matrix(operator, wire_order=[2, 3])),
        dtype=float,
    )
    fable_alpha = matrix.shape[0] * float(np.max(np.abs(matrix)))
    specs = {
        "dense embedding": matrix_block_encoding_spec(
            matrix,
            block_encoding="embedding",
        ),
        "FABLE": matrix_block_encoding_spec(
            matrix,
            alpha=fable_alpha,
            block_encoding="fable",
        ),
        "PrepSelPrep": pennylane_operator_block_encoding_spec(
            operator,
            encoding_wires=[0, 1],
            block_encoding="prepselprep",
        ),
        "qubitization": pennylane_operator_block_encoding_spec(
            operator,
            encoding_wires=[0, 1],
            block_encoding="qubitization",
        ),
    }
    degrees = [1, 3, 5, 7, 9]
    rows: list[dict[str, object]] = []
    reports: dict[str, list[dict[str, object]]] = {}

    for access_model, spec in specs.items():
        reports[access_model] = []
        for degree in degrees:
            coeffs = np.zeros(degree + 1)
            coeffs[-1] = 0.8
            estimate = estimate_encoding_aware_resources(spec, coeffs)
            reports[access_model].append(estimate.as_report())
            rows.append(
                {
                    "access_model": access_model,
                    "degree": degree,
                    "normalization_alpha": estimate.normalization_alpha,
                    "signal_operator_calls": estimate.signal_operator_calls,
                    "inverse_signal_operator_calls": (
                        estimate.inverse_signal_operator_calls
                    ),
                    "total_wires": estimate.total_wires,
                    "total_gates": estimate.total_gates,
                    "estimator_model": estimate.estimator_model,
                    "estimator_available": estimate.estimator_available,
                }
            )

    return {
        "example": "encoding-aware-resources",
        "mode": "encoding-aware-resource-cookbook",
        "operator_matrix": matrix,
        "degrees": degrees,
        "rows": rows,
        "reports": reports,
        "truth_contract": {
            "is_executed_runtime_benchmark": False,
            "is_fault_tolerant_estimate": False,
            "includes_normalization_and_access_model": True,
            "omitted_components": [
                "application_state_preparation",
                "postselection_or_amplitude_amplification",
                "application_readout_or_tomography",
                "provider_compilation_and_routing",
                "error_correction",
            ],
        },
    }


def write_rows(report: dict[str, object], path: Path) -> Path:
    """Write the compact resource rows from ``build_report`` to CSV."""
    rows = report["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("resource report must contain at least one row")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/examples/encoding_aware_resources.json"),
        help="Destination JSON report path.",
    )
    parser.add_argument(
        "--rows-output",
        type=Path,
        default=Path("results/examples/encoding_aware_resources.csv"),
        help="Destination CSV summary path.",
    )
    args = parser.parse_args(argv)

    report = build_report()
    written = save_report(report, args.output)
    rows_written = write_rows(report, args.rows_output)
    print(written)
    print(rows_written)


if __name__ == "__main__":
    main()
