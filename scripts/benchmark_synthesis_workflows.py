"""Compare phase completion and reconstruction on reproducible workflow designs.

Run from an installed source checkout. Timing snapshots are written only to the
requested output; existing notebook benchmark artifacts are never refreshed.
"""

from __future__ import annotations

import argparse
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np

from qsvt.matrix_functions import design_real_time_evolution_polynomials
from qsvt.reports import save_report
from qsvt.synthesis import benchmark_phase_solver_stress_matrix
from qsvt.workflow import DesignKind, design_workflow


def polynomial_cases() -> dict[str, np.ndarray]:
    """Keep original coefficients, including near-boundary fitting residuals."""
    cases = {}
    designs: tuple[tuple[DesignKind, tuple[int, ...]], ...] = (
        ("sign", (13, 25)),
        ("inverse", (13, 25)),
        ("filter", (10, 24)),
    )
    for kind, degrees in designs:
        for degree in degrees:
            cases[f"{kind}-{degree}"] = design_workflow(
                kind, degree=degree, num_points=401, attempt_synthesis=False
            ).coeffs
    for degree in (12, 24):
        evolution = design_real_time_evolution_polynomials(
            time=1.4, scale=1.0, degree=degree, num_points=401
        )
        cases[f"hamiltonian-cos-{degree}"] = evolution.cos_coeffs
        cases[f"hamiltonian-sin-{degree}"] = evolution.sin_coeffs
    for degree in (16, 32):
        for margin in (0.05, 1e-8):
            cheb = np.zeros(degree + 1)
            cheb[-1] = 1.0 - margin
            cases[f"chebyshev-{degree}-margin-{margin:g}"] = (
                np.polynomial.chebyshev.cheb2poly(cheb)
            )
    cases["constant-zero"] = np.array([0.0])
    cases["constant-boundary"] = np.array([1.0])
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    benchmark = benchmark_phase_solver_stress_matrix(
        polynomial_cases(),
        solvers=("root-finding", "iterative"),
        repeats=args.repeats,
        reconstruction_num_points=257,
        reconstruction_tolerance=1e-6,
    )
    report = benchmark.as_report()
    report["environment"] = {
        "python": platform.python_version(),
        **{
            name: version(name)
            for name in ("numpy", "scipy", "pennylane", "qsvt-pennylane")
        },
    }
    report["design_settings"] = {"num_points": 401, "time": 1.4, "scale": 1.0}
    save_report(report, args.output)
    for name, case in benchmark.cases:
        for row in case.rows:
            print(
                f"{name}: {row['angle_solver']} "
                f"returned={row['successes']}/{row['attempts']} "
                f"validated={row['validated_successes']}/{row['attempts']} "
                f"error={row['max_reconstruction_error']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
