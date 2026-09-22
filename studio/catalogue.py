"One catalogue for request validation, defaults, controls, and API dispatch."

from __future__ import annotations

import copy
import inspect
import math
from collections.abc import Callable
from typing import Any

from qsvt.stable import (
    design_workflow,
    hamiltonian_simulation_workflow,
    poisson_qsvt_workflow,
    spectral_filter_qsvt_workflow,
)

from .presets import presets

SCHEMA_VERSION = "1.2"
FUNCTIONS: dict[str, Callable[..., Any]] = {
    "design": design_workflow,
    "hamiltonian_simulation": hamiltonian_simulation_workflow,
    "poisson": poisson_qsvt_workflow,
    "spectral_filter": spectral_filter_qsvt_workflow,
}


def field(api: str, name: str, group: str, *, default: Any = None, **rules: Any):
    """Use API defaults unless a required argument needs a catalogue choice."""
    parameter = inspect.signature(FUNCTIONS[api]).parameters[name]
    value = parameter.default
    required = value is inspect.Parameter.empty
    if required:
        value = default
    if isinstance(value, tuple):
        value = list(value)
    return {
        "label": name.replace("_", " ").capitalize(),
        "group": group,
        "default": value,
        "required": required,
        "default_source": "catalogue" if required else "package signature",
        **rules,
    }


def number(api, name, group, low, high, *, integer=False, default=None):
    return field(
        api,
        name,
        group,
        default=default,
        control="number",
        min=low,
        max=high,
        integer=integer,
    )


def catalogue() -> dict[str, Any]:
    workflows: dict[str, dict[str, Any]] = {}
    for kind, title, target_fields in (
        ("sign", "Sign approximation", {"gamma": (0.001, 0.999)}),
        ("inverse", "Normalized reciprocal", {"gamma": (0.001, 0.999)}),
        (
            "filter",
            "Soft spectral filter",
            {"cutoff": (0.001, 0.999), "sharpness": (0.1, 40)},
        ),
        (
            "interval_projector",
            "Interval projector design",
            {
                "lower": (-0.999, 0.999),
                "upper": (-0.999, 0.999),
                "sharpness": (0.1, 40),
            },
        ),
    ):
        settings = {
            name: number("design", name, "Target", *limits)
            for name, limits in target_fields.items()
        }
        settings.update(
            {
                "degree": number(
                    "design",
                    "degree",
                    "Polynomial",
                    1,
                    48,
                    integer=True,
                    default=13 if kind in {"sign", "inverse"} else 10,
                ),
                "num_points": number(
                    "design", "num_points", "Polynomial", 101, 4001, integer=True
                ),
                "bounded_num_points": number(
                    "design",
                    "bounded_num_points",
                    "Validation",
                    101,
                    8001,
                    integer=True,
                ),
                "attempt_synthesis": field(
                    "design", "attempt_synthesis", "Validation", control="boolean"
                ),
            }
        )
        workflows[kind] = {
            "id": kind,
            "name": title,
            "api": "design",
            "settings": settings,
            "description": "Bounded polynomial design on normalized [-1, 1]. "
            "Sampled errors use the package's fitting target; no QNode is executed.",
            "circuit_execution": False,
            "resource_estimation": "polynomial proxy",
            "truth_metadata": True,
            "acceptance_metadata": False,
            "diagnostics": [
                "target and polynomial",
                "sampled error",
                "compatibility",
                "phase reconstruction",
            ],
            "comparison_fields": list(target_fields),
        }
    for api, title in (
        ("poisson", "Poisson inversion"),
        ("spectral_filter", "Pauli spectral filtering"),
    ):
        settings = {
            "tolerance": number(api, "tolerance", "Polynomial", 1e-8, 1),
            "min_degree": number(api, "min_degree", "Polynomial", 1, 31, integer=True),
            "max_degree": number(api, "max_degree", "Polynomial", 1, 31, integer=True),
            "degree_step": number(api, "degree_step", "Polynomial", 1, 8, integer=True),
            "num_points": number(
                api, "num_points", "Polynomial", 101, 4001, integer=True
            ),
            "phase_reconstruction_tolerance": number(
                api, "phase_reconstruction_tolerance", "Validation", 1e-10, 1e-3
            ),
            "execute": field(api, "execute", "Execution", control="boolean"),
            "shots": {
                "label": "Sampling",
                "group": "Execution",
                "control": "select",
                "values": [None, 100, 1000, 10000],
                "value_labels": [
                    "Analytic statevector",
                    "100 shots",
                    "1,000 shots",
                    "10,000 shots",
                ],
                "default": None,
                "default_source": "package signature",
                "help": (
                    "Finite shots return sampled probabilities; analytic execution "
                    "returns amplitudes."
                ),
            },
            "device_name": field(
                api,
                "device_name",
                "Execution",
                control="select",
                values=["default.qubit"],
            ),
        }
        if api == "poisson":
            settings.update(
                {
                    "n_points": field(
                        api, "n_points", "Problem", control="select", values=[2, 4, 8]
                    ),
                    "length": number(api, "length", "Problem", 0.1, 10),
                    "access_model": field(
                        api,
                        "access_model",
                        "Block encoding",
                        control="select",
                        values=["dense", "fable", "prepselprep", "qubitization"],
                    ),
                    "source_kind": {
                        "label": "Source profile",
                        "group": "Problem",
                        "control": "select",
                        "values": ["sine", "constant", "gaussian"],
                        "value_labels": [
                            "Sine (published example)",
                            "Constant",
                            "Centered Gaussian",
                        ],
                        "default": "sine",
                        "default_source": "studio bounded problem family",
                    },
                }
            )
            problem = "Dirichlet 1D Laplacian with the package's default sine source."
            comparison_fields = ["n_points", "length", "source_kind"]
        else:
            settings.update(
                {
                    "lower": number(api, "lower", "Target", -0.89, 0.89, default=-0.4),
                    "upper": number(api, "upper", "Target", -0.89, 0.89, default=0.4),
                    "sharpness": number(api, "sharpness", "Polynomial", 0.1, 40),
                    "block_encoding": field(
                        api,
                        "block_encoding",
                        "Block encoding",
                        control="select",
                        values=["prepselprep", "qubitization"],
                    ),
                    "z0_coefficient": {
                        "label": "Z₀ coefficient",
                        "group": "Problem",
                        "control": "number",
                        "min": -1.0,
                        "max": 1.0,
                        "default": 0.4,
                        "default_source": "published example",
                    },
                    "z1_coefficient": {
                        "label": "Z₁ coefficient",
                        "group": "Problem",
                        "control": "number",
                        "min": -1.0,
                        "max": 1.0,
                        "default": 0.3,
                        "default_source": "published example",
                    },
                    "x0_coefficient": {
                        "label": "X₀ coefficient",
                        "group": "Problem",
                        "control": "number",
                        "min": -1.0,
                        "max": 1.0,
                        "default": 0.2,
                        "default_source": "published example",
                    },
                    "input_state": {
                        "label": "Input state",
                        "group": "Problem",
                        "control": "select",
                        "values": [
                            "uniform",
                            "basis-00",
                            "basis-01",
                            "basis-10",
                            "basis-11",
                        ],
                        "value_labels": ["Uniform", "|00⟩", "|01⟩", "|10⟩", "|11⟩"],
                        "default": "uniform",
                        "default_source": "published example",
                    },
                }
            )
            problem = (
                "Cookbook Hamiltonian 0.4 Z₀ + 0.3 Z₁ + 0.2 X₀; "
                "uniform four-component input state."
            )
            comparison_fields = [
                "lower",
                "upper",
                "z0_coefficient",
                "z1_coefficient",
                "x0_coefficient",
                "input_state",
            ]
        workflows[api] = {
            "id": api,
            "name": title,
            "api": api,
            "settings": settings,
            "description": problem
            + " Execute enables local analytic or finite-shot QNode validation.",
            "execution_setting": "execute",
            "no_execution_label": "Polynomial + synthesis · no QNode requested",
            "circuit_execution": True,
            "resource_estimation": "encoding-aware logical estimate",
            "truth_metadata": True,
            "acceptance_metadata": True,
            "diagnostics": [
                "degree search",
                "phase reconstruction",
                "finite reference",
                "acceptance",
                "logical resources",
            ],
            "comparison_fields": comparison_fields,
        }
    api = "hamiltonian_simulation"
    workflows[api] = {
        "id": api,
        "name": "Hamiltonian simulation",
        "api": api,
        "description": (
            "Published six-site tight-binding chain, initially at site 1 "
            "(zero-based). "
            "Cosine/sine polynomial evolution with optional coherent finite QNode "
            "execution; default.qubit, analytic or finite-shot sampling, dense "
            "embedding."
        ),
        "settings": {
            "n_sites": {
                "label": "Chain sites",
                "group": "Problem",
                "control": "number",
                "min": 2,
                "max": 8,
                "integer": True,
                "default": 6,
                "default_source": "published example",
            },
            "initial_site": {
                "label": "Initial site",
                "group": "Problem",
                "control": "number",
                "min": 0,
                "max": 7,
                "integer": True,
                "default": 1,
                "default_source": "published example",
            },
            "hopping": {
                "label": "Hopping",
                "group": "Problem",
                "control": "number",
                "min": -2.0,
                "max": 2.0,
                "default": 1.0,
                "default_source": "qsvt.hamiltonians.tight_binding_chain",
            },
            "onsite": {
                "label": "Uniform onsite energy",
                "group": "Problem",
                "control": "number",
                "min": -1.0,
                "max": 1.0,
                "default": 0.0,
                "default_source": "studio bounded problem family",
            },
            "periodic": {
                "label": "Periodic boundary",
                "group": "Problem",
                "control": "boolean",
                "default": False,
                "default_source": "qsvt.hamiltonians.tight_binding_chain",
            },
            "time": number(api, "time", "Problem", -5, 5, default=1.4),
            "degree": number(
                api, "degree", "Polynomial", 1, 24, integer=True, default=12
            ),
            "num_points": number(
                api, "num_points", "Polynomial", 101, 4001, integer=True
            ),
            "acceptance_tolerance": number(
                api, "acceptance_tolerance", "Validation", 1e-10, 1e-2
            ),
            "phase_reconstruction_tolerance": number(
                api, "phase_reconstruction_tolerance", "Validation", 0, 1e-3
            ),
            "execute_qsvt": field(api, "execute_qsvt", "Execution", control="boolean"),
            "shots": {
                "label": "Sampling",
                "group": "Execution",
                "control": "select",
                "values": [None, 100, 1000, 10000],
                "value_labels": [
                    "Analytic statevector",
                    "100 shots",
                    "1,000 shots",
                    "10,000 shots",
                ],
                "default": None,
                "default_source": "package signature",
                "help": (
                    "Finite shots return sampled probabilities; analytic execution "
                    "returns amplitudes."
                ),
            },
            "block_encoding": field(
                api,
                "block_encoding",
                "Block encoding",
                control="select",
                values=["embedding"],
            ),
            "device_name": field(
                api,
                "device_name",
                "Execution",
                control="select",
                values=["default.qubit"],
            ),
        },
        "execution_setting": "execute_qsvt",
        "no_execution_label": "Polynomial core · no QNode or phase synthesis requested",
        "circuit_execution": True,
        "resource_estimation": "finite coherent circuit ledger when executed",
        "truth_metadata": True,
        "acceptance_metadata": True,
        "diagnostics": [
            "complex evolved state",
            "dense exponential reference",
            "component phases",
            "coherent execution",
            "acceptance",
            "component error ledger",
            "circuit resource ledger",
        ],
        "comparison_fields": [
            "time",
            "n_sites",
            "initial_site",
            "hopping",
            "onsite",
            "periodic",
        ],
    }
    for entry in workflows.values():
        fields = entry["settings"]
        api = entry["api"]
        if api == "design":
            fields["angle_solver"] = {
                "label": "Phase solver",
                "group": "Synthesis",
                "control": "select",
                "values": ["root-finding", "iterative"],
                "default": "root-finding",
                "default_source": "DesignWorkflowResult.synthesize",
                "advanced": True,
                "help": (
                    "Root finding or iterative optimization of the same "
                    "polynomial; accuracy is checked separately."
                ),
            }
            fields["phase_reconstruction_tolerance"] = {
                "label": "Phase reconstruction tolerance",
                "group": "Synthesis",
                "control": "number",
                "min": 1e-10,
                "max": 1e-3,
                "default": 1e-6,
                "default_source": "PhaseSynthesisResult.quality_report",
                "advanced": True,
            }
            fields["reconstruction_num_points"] = {
                "label": "Reconstruction sample count",
                "group": "Synthesis",
                "control": "number",
                "min": 33,
                "max": 2001,
                "integer": True,
                "default": 257,
                "default_source": "DesignWorkflowResult.synthesize",
                "advanced": True,
                "help": (
                    "Samples for phase reconstruction on [-1, 1]; this is not a "
                    "formal error bound."
                ),
            }
            if entry["id"] in {"sign", "inverse", "filter"}:
                fields["degree"].update(step=2, min=2 if entry["id"] == "filter" else 1)
                fields["degree"]["help"] = (
                    "Even degrees only."
                    if entry["id"] == "filter"
                    else "Odd degrees only."
                )
        elif api in {"poisson", "spectral_filter"}:
            fields["angle_solvers"] = field(
                api,
                "angle_solvers",
                "Synthesis",
                control="select",
                advanced=True,
                values=[["root-finding", "iterative"], ["root-finding"], ["iterative"]],
                value_labels=[
                    "Root finding → iterative fallback",
                    "Root finding only",
                    "Iterative only",
                ],
                help=(
                    "Fallback tests reconstruction tolerance before accepting a "
                    "solver result. The report identifies the selected solver."
                ),
            )
        else:
            fields["angle_solver"] = field(
                api,
                "angle_solver",
                "Synthesis",
                control="select",
                advanced=True,
                values=["root-finding", "iterative"],
                help=(
                    "One solver is used for both cosine and sine components; each "
                    "component is validated."
                ),
            )
        for name, spec in fields.items():
            if name in {
                "num_points",
                "bounded_num_points",
                "phase_reconstruction_tolerance",
                "degree_step",
            }:
                spec["advanced"] = True
            if spec["control"] == "select" and len(spec["values"]) == 1:
                spec["fixed"] = True
            if name == "phase_reconstruction_tolerance":
                spec["help"] = (
                    "Maximum sampled absolute error between the synthesized phase "
                    "response and the polynomial; separate from target "
                    "approximation error."
                )
            elif name == "num_points":
                spec["help"] = (
                    "Fitting/diagnostic grid size. More points do not increase "
                    "polynomial degree."
                )
            elif name == "bounded_num_points":
                spec["help"] = (
                    "Package compatibility setting retained for reproducibility. "
                    "Boundedness certification uses numerical extrema, not this "
                    "grid."
                )
            elif name == "acceptance_tolerance":
                spec["help"] = (
                    "Package threshold for polynomial evolution accuracy, norm "
                    "drift, and finite execution checks; not a solver stopping "
                    "criterion."
                )
            elif name == "tolerance":
                spec["help"] = (
                    "Relative solution error against the dense direct solve."
                    if api == "poisson"
                    else "Relative operator error against the hard spectral projector."
                )
            elif name == "gamma":
                spec["help"] = (
                    "Excluded central region: interpret sign/reciprocal accuracy "
                    "on |x| ≥ gamma; the full-domain fitting target differs "
                    "there."
                )

    choices = presets()
    for entry in workflows.values():
        entry["recommended_preset"] = next(
            p["id"]
            for p in choices
            if p["workflow"] == entry["id"] and p["recommended"]
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "workflows": workflows,
        "presets": choices,
    }


def validate_request(raw: Any) -> dict[str, Any]:
    """Resolve exposed defaults and reject unsupported or oversized work."""
    if not isinstance(raw, dict) or set(raw) - {
        "schema_version",
        "workflow",
        "settings",
    }:
        raise ValueError(
            "Request must contain only schema_version, workflow, settings."
        )
    if not isinstance(raw.get("schema_version"), str) or raw.get(
        "schema_version"
    ) not in {"1.0", "1.1", SCHEMA_VERSION}:
        raise ValueError(
            "Unsupported request schema_version; expected 1.0, 1.1, or 1.2."
        )
    workflow = raw.get("workflow")
    entries = catalogue()["workflows"]
    if not isinstance(workflow, str) or workflow not in entries:
        raise ValueError("Unknown workflow.")
    settings = raw.get("settings", {})
    fields = entries[workflow]["settings"]
    if not isinstance(settings, dict) or set(settings) - set(fields):
        raise ValueError("Unsupported settings for this workflow.")
    if raw["schema_version"] in {"1.0", "1.1"}:
        settings = dict(settings)
        additions = (
            (
                {
                    "angle_solver": "root-finding",
                    "phase_reconstruction_tolerance": 1e-6,
                    "reconstruction_num_points": 257,
                }
                if entries[workflow]["api"] == "design"
                else (
                    {"angle_solver": "root-finding"}
                    if workflow == "hamiltonian_simulation"
                    else {"angle_solvers": ["root-finding", "iterative"]}
                )
            )
            if raw["schema_version"] == "1.0"
            else {}
        )
        for name, value in additions.items():
            if name in settings:
                raise ValueError(f"{name} requires request schema 1.1.")
            settings[name] = value
        modern_defaults: dict[str, dict[str, Any]] = {
            "poisson": {"source_kind": "sine", "shots": None},
            "spectral_filter": {
                "z0_coefficient": 0.4,
                "z1_coefficient": 0.3,
                "x0_coefficient": 0.2,
                "input_state": "uniform",
                "shots": None,
            },
            "hamiltonian_simulation": {
                "n_sites": 6,
                "initial_site": 1,
                "hopping": 1.0,
                "onsite": 0.0,
                "periodic": False,
                "shots": None,
            },
        }
        for name, value in modern_defaults.get(workflow, {}).items():
            if name in settings:
                raise ValueError(f"{name} requires request schema 1.2.")
            settings[name] = value
    resolved = {}
    for name, spec in fields.items():
        value = settings.get(name, spec["default"])
        if spec["control"] == "boolean":
            valid = type(value) is bool
        elif spec["control"] == "select":
            valid = any(
                type(value) is type(choice) and value == choice
                for choice in spec["values"]
            )
        else:
            valid = type(value) in (int, float) and math.isfinite(value)
            valid = valid and spec["min"] <= value <= spec["max"]
            if spec.get("integer"):
                valid = valid and type(value) is int
        if not valid:
            raise ValueError(
                f"Invalid {name}: check the catalogue type and allowed range."
            )
        resolved[name] = value
    if "lower" in resolved and resolved["lower"] >= resolved["upper"]:
        raise ValueError("lower must be smaller than upper.")
    if "min_degree" in resolved and resolved["min_degree"] > resolved["max_degree"]:
        raise ValueError("min_degree must not exceed max_degree.")
    if workflow in {"sign", "inverse"} and resolved["degree"] % 2 != 1:
        raise ValueError("This odd design requires an odd degree.")
    if workflow == "filter" and resolved["degree"] % 2:
        raise ValueError("This even filter design requires an even degree.")
    if (
        workflow == "hamiltonian_simulation"
        and resolved["initial_site"] >= resolved["n_sites"]
    ):
        raise ValueError("initial_site must be smaller than n_sites.")
    if workflow == "spectral_filter" and not any(
        resolved[name] != 0
        for name in ("z0_coefficient", "z1_coefficient", "x0_coefficient")
    ):
        raise ValueError("At least one Pauli coefficient must be non-zero.")
    return copy.deepcopy(
        {"schema_version": SCHEMA_VERSION, "workflow": workflow, "settings": resolved}
    )
