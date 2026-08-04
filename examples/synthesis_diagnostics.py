"""Cookbook example: certify, classify, and synthesize QSVT polynomials."""

from __future__ import annotations

import argparse
from pathlib import Path

from qsvt.stable import (
    certify_polynomial_boundedness,
    classify_polynomial_realizability,
    save_report,
    synthesize_phases,
)

POLYNOMIAL_CASES = {
    "single-sequence-odd": [0.0, 0.75, 0.0, -0.25],
    "bounded-mixed-parity": [0.25, 0.5],
    "interior-peak-violation": [0.996, 0.1, -0.5],
}


def build_report() -> dict[str, object]:
    """Build success and structured-failure phase-synthesis diagnostics."""
    cases: dict[str, dict[str, object]] = {}
    successful = 0
    for name, coeffs in POLYNOMIAL_CASES.items():
        certificate = certify_polynomial_boundedness(coeffs)
        realizability = classify_polynomial_realizability(coeffs)
        synthesis = synthesize_phases(
            coeffs,
            routine="QSVT",
            angle_solver="root-finding",
            reconstruction_num_points=129,
        )
        cases[name] = {
            "coeffs": coeffs,
            "boundedness": certificate.as_report(),
            "realizability": realizability.as_report(),
            "synthesis": synthesis.as_report(),
        }
        successful += int(synthesis.succeeded)
    return {
        "example": "synthesis-diagnostics",
        "mode": "phase-synthesis-diagnostics-cookbook",
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "successful_syntheses": successful,
            "structured_failures": len(cases) - successful,
        },
        "truth_contract": {
            "measured_component": "classical_phase_synthesis",
            "executes_qsvt_circuit": False,
            "uses_hardware": False,
            "coefficient_order": "ascending-monomial",
            "phase_convention": "PennyLane QSVT projector-phase convention",
            "purpose": (
                "Distinguish extrema boundedness, structural realizability, and "
                "numerical phase synthesis without claiming circuit execution."
            ),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/examples/synthesis_diagnostics.json"),
        help="Destination JSON report path.",
    )
    args = parser.parse_args(argv)

    written = save_report(build_report(), args.output)
    print(written)


if __name__ == "__main__":
    main()
