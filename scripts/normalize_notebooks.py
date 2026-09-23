#!/usr/bin/env python3
"""Normalize the visible and machine-readable contract for repository notebooks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

NOTEBOOK_DIRS = (
    Path("notebooks/tutorials"),
    Path("notebooks/real_examples"),
    Path("notebooks/benchmarks"),
)

FOCUS = {
    "tutorials/01_QSVT_Scalar_and_Diagonal_Matrix.ipynb": (
        "apply and validate scalar and diagonal polynomial transforms"
    ),
    "tutorials/02_QSVT_Singular_Value_Filter.ipynb": (
        "build and apply a soft singular-value filter"
    ),
    "tutorials/03_QSP_Polynomial_Demo.ipynb": (
        "connect QSP circuit behavior with Chebyshev polynomials"
    ),
    "tutorials/04_QSVT_Exact_Linear_Solver_Toy_Cases.ipynb": (
        "compare exact toy inverse transforms with linear-system solutions"
    ),
    "tutorials/05_QSVT_Polynomial_Design_and_Approximation.ipynb": (
        "design Chebyshev approximations and measure their error"
    ),
    "tutorials/06_QSVT_Matrix_Functions_Powers_and_Roots.ipynb": (
        "apply polynomial approximations to matrix powers and roots"
    ),
    "tutorials/07_QSVT_Sign_Function_and_Projectors.ipynb": (
        "construct sign approximations and spectral projectors"
    ),
    "tutorials/08_QSVT_Design_and_Presets.ipynb": (
        "compare named presets with task-oriented polynomial design"
    ),
    "tutorials/09_QSVT_Algorithm_Workflows.ipynb": (
        "run algorithm workflows and interpret their acceptance boundaries"
    ),
    "tutorials/10_QSVT_Reports_CLI_and_Artifacts.ipynb": (
        "create reproducible reports through Python and the CLI"
    ),
    "tutorials/11_QSVT_Design_Tradeoffs.ipynb": (
        "study degree, approximation error, and boundedness tradeoffs"
    ),
    "tutorials/12_QSVT_Resource_Proxy_Limits.ipynb": (
        "interpret QSVT resource proxies and their omitted costs"
    ),
    "tutorials/13_Block_Encoded_QSVT_Workflow.ipynb": (
        "construct, execute, and validate a finite block-encoded QSVT workflow"
    ),
    "tutorials/14_Sparse_Oracle_Assumptions.ipynb": (
        "compare sparse access models and their resource assumptions"
    ),
    "tutorials/15_QSVT_Compatibility_Failure_Cases.ipynb": (
        "diagnose boundedness, parity, and synthesis failures"
    ),
    "tutorials/16_QSVT_Linear_System_Comparisons.ipynb": (
        "compare classical and QSVT-style linear-system results"
    ),
    "tutorials/17_HHL_Linear_System_Solver.ipynb": (
        "execute finite HHL and compare it with QSVT-style inversion"
    ),
    "tutorials/18_Quantum_Walk_Search_Workflow.ipynb": (
        "run a quantum-walk search comparison and inspect its proxy costs"
    ),
    "tutorials/19_Accuracy_Driven_QSVT_Planning.ipynb": (
        "select degree and resources from an accuracy requirement"
    ),
    "tutorials/20_Finite_Shot_Device_Preflight_and_Circuit_Audit.ipynb": (
        "preflight, audit, and execute a finite-shot local circuit"
    ),
    "real_examples/01_poisson_equation_pde.ipynb": (
        "solve finite Poisson systems with polynomial and circuit references"
    ),
    "real_examples/02_hamiltonian_simulation_schrodinger_dynamics.ipynb": (
        "simulate finite real-time dynamics with coherent QSVT"
    ),
    "real_examples/03_greens_function_response.ipynb": (
        "approximate a resolvent and validate its response function"
    ),
    "real_examples/04_ising_phase_transition_filtering.ipynb": (
        "filter a finite Ising spectrum through a Pauli-LCU workflow"
    ),
    "real_examples/05_fermi_dirac_electronic_occupations.ipynb": (
        "approximate finite-temperature electronic occupations"
    ),
    "real_examples/06_topological_band_projector_chern_marker.ipynb": (
        "approximate a band projector and evaluate a Chern marker"
    ),
    "real_examples/07_singular_value_pseudoinverse_deblurring.ipynb": (
        "apply a regularized singular-value pseudoinverse to deblurring"
    ),
    "real_examples/08_matrix_log_entropy_graph_laplacian.ipynb": (
        "approximate a graph matrix logarithm and entropy"
    ),
    "real_examples/09_phonon_density_of_states.ipynb": (
        "estimate the phonon density of states of a mass-spring chain"
    ),
    "real_examples/10_thermal_heisenberg_chain.ipynb": (
        "estimate thermal observables of a finite Heisenberg spin chain"
    ),
    "real_examples/11_disordered_transport_localization.ipynb": (
        "validate finite-chain transport observables with coherent QSVT"
    ),
    "benchmarks/01_linear_system_classical_vs_qsvt_proxy.ipynb": (
        "compare linear-system baselines, finite HHL, and QSVT proxies"
    ),
    "benchmarks/02_matrix_functions_spectral_baselines.ipynb": (
        "compare spectral and polynomial matrix-function baselines"
    ),
    "benchmarks/03_scaling_sweeps.ipynb": (
        "inspect smoke-scale dimension and inverse-degree trends"
    ),
    "benchmarks/04_classical_baseline_assumptions.ipynb": (
        "make classical timing and QSVT proxy assumptions explicit"
    ),
    "benchmarks/05_quantum_walk_search_scaling.ipynb": (
        "compare quantum-walk accuracy and signal-call trends"
    ),
    "benchmarks/06_encoding_aware_resources.ipynb": (
        "compare logical resources across block-encoding access models"
    ),
    "benchmarks/07_phase_synthesis_stress_matrix.ipynb": (
        "stress phase synthesis across conditioning regimes"
    ),
}

ROLE_DETAILS = {
    "tutorial": {
        "prerequisites": (
            "an editable repository install plus basic NumPy and linear-algebra "
            "familiarity"
        ),
        "boundary": (
            "Treat simulator-scale circuits and resource proxies according to "
            "the truth contract stated in the notebook."
        ),
    },
    "real-example": {
        "prerequisites": (
            "an editable repository install, the relevant tutorial workflow, "
            "and basic spectral linear algebra"
        ),
        "boundary": (
            "This is a finite, simulator-scale application client; input, "
            "readout, and scalability assumptions remain explicit."
        ),
    },
    "benchmark": {
        "prerequisites": (
            "an editable repository install and familiarity with classical "
            "baselines and QSVT resource proxies"
        ),
        "boundary": (
            "Environment-specific timings and logical proxies are evidence for "
            "different claims and are not quantum runtime measurements."
        ),
    },
}


def _cell_id(path: Path, purpose: str) -> str:
    payload = f"{path.as_posix()}:{purpose}".encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def _role(path: Path) -> str:
    if path.parent.name == "tutorials":
        return "tutorial"
    if path.parent.name == "real_examples":
        return "real-example"
    return "benchmark"


def _runtime(path: Path) -> str:
    slower = {
        "01_poisson_equation_pde.ipynb",
        "04_ising_phase_transition_filtering.ipynb",
        "17_HHL_Linear_System_Solver.ipynb",
        "19_Accuracy_Driven_QSVT_Planning.ipynb",
        "20_Finite_Shot_Device_Preflight_and_Circuit_Audit.ipynb",
        "07_phase_synthesis_stress_matrix.ipynb",
    }
    return "about 3–5 minutes" if path.name in slower else "about 1–3 minutes"


def _navigation(paths: list[Path], index: int) -> str:
    links = ["[collection index](README.md)"]
    if index:
        links.append(f"[previous]({paths[index - 1].name})")
    if index + 1 < len(paths):
        links.append(f"[next]({paths[index + 1].name})")
    return " · ".join(links)


def _markdown_cell(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def _guide(path: Path, paths: list[Path], index: int) -> dict:
    key = f"{path.parent.name}/{path.name}"
    role = _role(path)
    details = ROLE_DETAILS[role]
    source = (
        "## Notebook guide\n\n"
        f"- **Learning objective:** {FOCUS[key]}.\n"
        f"- **Prerequisites:** {details['prerequisites']}.\n"
        f"- **Estimated runtime:** {_runtime(path)} on a local CPU.\n"
        f"- **Navigation:** {_navigation(paths, index)}.\n"
    )
    return _markdown_cell(_cell_id(path, "guide"), source)


def _takeaways(path: Path, paths: list[Path], index: int) -> dict:
    key = f"{path.parent.name}/{path.name}"
    role = _role(path)
    source = (
        "## Takeaways and next steps\n\n"
        f"- **Result:** The workflow shows how to {FOCUS[key]}.\n"
        f"- **Interpretation boundary:** {ROLE_DETAILS[role]['boundary']}\n"
        f"- **Continue:** {_navigation(paths, index)}.\n"
    )
    return _markdown_cell(_cell_id(path, "takeaways"), source)


def normalize_notebook(
    path: Path,
    paths: list[Path],
    index: int,
    *,
    write: bool = True,
) -> bool:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    original = json.dumps(notebook, sort_keys=True)

    notebook["cells"] = [
        cell
        for cell in notebook["cells"]
        if cell.get("id") not in {_cell_id(path, "guide"), _cell_id(path, "takeaways")}
    ]
    notebook["cells"].insert(1, _guide(path, paths, index))
    notebook["cells"].append(_takeaways(path, paths, index))
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python"}
    notebook["metadata"]["qsvt_notebook"] = {
        "role": _role(path),
        "schema_version": "1.0",
    }

    changed = json.dumps(notebook, sort_keys=True) != original
    if changed and write:
        path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report notebooks that require normalization without writing them.",
    )
    args = parser.parse_args()

    changed = []
    for directory in NOTEBOOK_DIRS:
        paths = sorted(directory.glob("*.ipynb"))
        for index, path in enumerate(paths):
            needs_change = normalize_notebook(
                path,
                paths,
                index,
                write=not args.check,
            )
            if needs_change:
                changed.append(path)

    if args.check and changed:
        for path in changed:
            print(path)
        return 1
    if not args.check:
        print(f"Normalized {len(changed)} notebook(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
