"Versioned, explicit configurations; never inherit live package defaults."

from __future__ import annotations

import copy

# These literals pin all exposed scientific settings, including inactive controls.
DESIGN = {
    "degree": 13,
    "num_points": 401,
    "bounded_num_points": 801,
    "attempt_synthesis": False,
    "angle_solver": "root-finding",
    "phase_reconstruction_tolerance": 1e-6,
    "reconstruction_num_points": 257,
}
POISSON = {
    "n_points": 4,
    "length": 1.0,
    "tolerance": 0.2,
    "min_degree": 3,
    "max_degree": 31,
    "degree_step": 2,
    "num_points": 2001,
    "phase_reconstruction_tolerance": 1e-6,
    "angle_solvers": ["root-finding", "iterative"],
    "execute": True,
    "device_name": "default.qubit",
    "access_model": "prepselprep",
    "source_kind": "sine",
    "shots": None,
    "sampling_tolerance": 0.05,
    "sampling_confidence": 0.95,
}
PAULI = {
    "lower": -0.4,
    "upper": 0.4,
    "sharpness": 8.0,
    "tolerance": 0.02,
    "min_degree": 2,
    "max_degree": 24,
    "degree_step": 2,
    "num_points": 2001,
    "phase_reconstruction_tolerance": 1e-6,
    "angle_solvers": ["root-finding", "iterative"],
    "execute": True,
    "device_name": "default.qubit",
    "block_encoding": "prepselprep",
    "z0_coefficient": 0.4,
    "z1_coefficient": 0.3,
    "x0_coefficient": 0.2,
    "input_state": "uniform",
    "shots": None,
    "sampling_tolerance": 0.05,
    "sampling_confidence": 0.95,
}
HAMILTONIAN = {
    "n_sites": 6,
    "initial_site": 1,
    "hopping": 1.0,
    "onsite": 0.0,
    "periodic": False,
    "time": 1.4,
    "degree": 12,
    "num_points": 401,
    "acceptance_tolerance": 1e-6,
    "phase_reconstruction_tolerance": 1e-6,
    "angle_solver": "root-finding",
    "execute_qsvt": True,
    "block_encoding": "embedding",
    "device_name": "default.qubit",
    "shots": None,
    "sampling_tolerance": 0.05,
    "sampling_confidence": 0.95,
}


def presets():
    entries = []

    def add(
        id,
        name,
        workflow,
        purpose,
        settings,
        description,
        *,
        source=None,
        recommended=False,
        expected="design",
    ):
        entries.append(
            {
                "id": id,
                "revision": (
                    3
                    if workflow
                    in {"poisson", "spectral_filter", "hamiltonian_simulation"}
                    else 1
                ),
                "name": name,
                "workflow": workflow,
                "purpose": purpose,
                "settings": copy.deepcopy(settings),
                "description": description,
                "source": source or "studio/presets.py",
                "recommended": recommended,
                "expected": expected,
            }
        )

    # Retain the published examples and their original display names.
    add(
        "sign-cookbook",
        "Sign · cookbook",
        "sign",
        "Quick demonstration",
        {**DESIGN, "gamma": 0.25},
        "Explore the fitting target and bounded polynomial; no synthesis or QNode.",
        source="examples/design_apply_report.py",
        recommended=True,
    )
    add(
        "poisson-cookbook",
        "Poisson · four-point finite QNode",
        "poisson",
        "Quick demonstration",
        {
            **POISSON,
            "tolerance": 0.4,
            "min_degree": 5,
            "max_degree": 5,
            "num_points": 401,
        },
        (
            "Published fixed-degree demonstration with a loose 40% "
            "solution-error target."
        ),
        source="examples/poisson_qsvt.py",
        expected="finite_qsvt",
    )
    add(
        "pauli-cookbook",
        "Pauli band filter · finite QNode",
        "spectral_filter",
        "Quick demonstration",
        {**PAULI, "tolerance": 0.16, "max_degree": 4, "num_points": 401},
        (
            "Published small-degree demonstration with a 16% hard- "
            "projector error target."
        ),
        source="examples/spectral_filter_qsvt.py",
        expected="finite_qsvt",
    )
    add(
        "hamiltonian-cookbook",
        "Hamiltonian · six-site coherent QNode",
        "hamiltonian_simulation",
        "Accuracy study",
        HAMILTONIAN,
        (
            "Validate coherent cosine/sine evolution against the dense "
            "exponential at tolerance 1e-6."
        ),
        source="examples/hamiltonian_simulation.py",
        recommended=True,
        expected="finite_qsvt",
    )

    for workflow, title, target in (
        ("sign", "Sign", {"gamma": 0.25}),
        ("inverse", "Normalized reciprocal", {"gamma": 0.25}),
        ("filter", "Soft filter", {"cutoff": 0.45, "sharpness": 12.0}),
        (
            "interval_projector",
            "Interval projector",
            {"lower": -0.25, "upper": 0.25, "sharpness": 12.0},
        ),
    ):
        base = {
            **DESIGN,
            **target,
            "degree": 13 if workflow in {"sign", "inverse"} else 10,
        }
        if workflow != "sign":
            add(
                f"{workflow}-quick",
                f"{title} · quick design",
                workflow,
                "Quick demonstration",
                base,
                (
                    "Inspect sampled fitting error and boundedness. No synthesis "
                    "or circuit acceptance is claimed."
                ),
                recommended=True,
            )
        add(
            f"{workflow}-phases",
            f"{title} · validated phases",
            workflow,
            "Accuracy study",
            {**base, "attempt_synthesis": True, "angle_solver": "iterative"},
            (
                "Study phase reconstruction of the same polynomial at "
                "tolerance 1e-6; fitting error is separate."
            ),
            expected="phases",
        )

    add(
        "poisson-accuracy",
        "Poisson · accuracy study",
        "poisson",
        "Accuracy study",
        POISSON,
        (
            "Search degrees 3–31 for a 20% solution-error target, then "
            "validate the finite circuit."
        ),
        recommended=True,
        expected="finite_qsvt",
    )
    add(
        "pauli-accuracy",
        "Pauli band filter · accuracy study",
        "spectral_filter",
        "Accuracy study",
        PAULI,
        (
            "Search even degrees 2–24 for a 2% hard-projector error "
            "target, then validate the finite circuit."
        ),
        recommended=True,
        expected="finite_qsvt",
    )
    add(
        "hamiltonian-quick",
        "Hamiltonian · quick polynomial study",
        "hamiltonian_simulation",
        "Quick demonstration",
        {**HAMILTONIAN, "execute_qsvt": False},
        (
            "Compare polynomial evolution with the dense exponential; "
            "phases and circuit resources are unavailable."
        ),
        expected="polynomial",
    )
    add(
        "interval-mixed",
        "Interval projector · expected synthesis failure",
        "interval_projector",
        "Expected failure",
        {
            **DESIGN,
            "degree": 10,
            "lower": 0.1,
            "upper": 0.6,
            "sharpness": 12.0,
            "attempt_synthesis": True,
        },
        (
            "An asymmetric interval produces mixed parity: one QSVT phase "
            "sequence cannot realize it."
        ),
        expected="synthesis_failure",
    )
    add(
        "hamiltonian-underresolved",
        "Hamiltonian · expected accuracy failure",
        "hamiltonian_simulation",
        "Expected failure",
        {**HAMILTONIAN, "degree": 2, "execute_qsvt": False},
        (
            "Degree 2 under-resolves evolution at time 1.4. The run "
            "completes but fails polynomial acceptance."
        ),
        expected="acceptance_failure",
    )
    return entries
