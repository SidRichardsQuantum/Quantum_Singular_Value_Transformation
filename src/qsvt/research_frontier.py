"""Accuracy-resource frontier studies built on the research sweep runner."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pennylane as qml

from .approximation import chebyshev_fit_function
from .block_encoding import (
    BlockEncodingSpec,
    matrix_block_encoding_spec,
    pennylane_operator_block_encoding_spec,
)
from .design import design_positive_inverse_polynomial
from .hamiltonians import ising_hamiltonian
from .pde import dirichlet_laplacian_1d
from .polynomials import chebyshev_to_monomial, polynomial_degree
from .reports import report_to_jsonable, save_report
from .research import (
    ResearchOperatorSpec,
    ResearchSweepResult,
    ResearchSweepSpec,
    ResearchTargetSpec,
    ResearchTrial,
    research_summary_rows,
    run_research_sweep,
)
from .resources import estimate_encoding_aware_resources
from .synthesis import synthesize_phases_with_adapter

_FRONTIER_ACCESS_MODELS = (
    "embedding",
    "fable",
    "prepselprep",
    "qubitization",
)
_FRONTIER_TARGETS = (
    "inverse",
    "projector",
    "band_filter",
    "resolvent",
)


@dataclass(frozen=True)
class AccuracyResourceFrontierResult:
    """Sweep results plus cross-configuration Pareto classifications."""

    sweep: ResearchSweepResult
    frontier_rows: tuple[dict[str, object], ...]
    output_dir: Path | None = None

    @property
    def pareto_rows(self) -> tuple[dict[str, object], ...]:
        """Return only rows not dominated on error, gates, and wires."""
        return tuple(row for row in self.frontier_rows if row["pareto_optimal"])

    def as_report(self) -> dict[str, object]:
        """Return a compact research-study manifest."""
        output_dir = self.output_dir
        return {
            "schema_name": "qsvt-accuracy-resource-frontier",
            "schema_version": "1.0",
            "mode": "accuracy-resource-frontier",
            "sweep": self.sweep.as_report(),
            "frontier_row_count": len(self.frontier_rows),
            "pareto_row_count": len(self.pareto_rows),
            "artifacts": {
                "frontier_json": (
                    None if output_dir is None else str(output_dir / "frontier.json")
                ),
                "pareto_csv": (
                    None if output_dir is None else str(output_dir / "pareto.csv")
                ),
            },
            "truth_contract": {
                "is_executed_hardware_benchmark": False,
                "accuracy_uses_finite_spectral_references": True,
                "resources_are_encoding_aware_logical_estimates": True,
                "normalization_is_access_model_specific": True,
                "pareto_objectives": [
                    "operator_relative_error",
                    "total_gates",
                    "total_wires",
                ],
                "omitted_components": [
                    "application_state_preparation",
                    "postselection_or_amplitude_amplification",
                    "provider_compilation_and_routing",
                    "hardware_noise",
                    "fault_tolerant_error_correction",
                ],
            },
        }


@dataclass(frozen=True)
class _PreparedOperator:
    name: str
    kind: str
    original_matrix: np.ndarray
    matrix: np.ndarray
    positive_shift: float
    metadata: dict[str, object]


@dataclass(frozen=True)
class _TargetDesign:
    components: tuple[tuple[str, np.ndarray], ...]
    exact_values: np.ndarray
    metadata: dict[str, object]


class AccuracyResourceFrontierEvaluator:
    """Cached evaluator for the built-in accuracy-resource frontier study."""

    def __init__(self) -> None:
        self._operators: dict[str, _PreparedOperator] = {}
        self._encodings: dict[tuple[str, str], BlockEncodingSpec] = {}
        self._designs: dict[tuple[str, str, str, int], _TargetDesign] = {}
        self._resources: dict[tuple[str, str, str, int], tuple[dict[str, Any], ...]] = (
            {}
        )

    def __call__(self, trial: ResearchTrial) -> dict[str, object]:
        """Evaluate one ideal logical-resource trial."""
        if trial.shots is not None or trial.noise_model != "ideal":
            return {
                "status": "skipped",
                "error_type": "UnsupportedResearchFactor",
                "error": (
                    "The accuracy-resource evaluator is ideal and resource-only; "
                    "finite shots and noise models are reserved for "
                    "noise-aware studies."
                ),
                "summary": {
                    "requested_shots": trial.shots,
                    "requested_noise_model": trial.noise_model,
                },
            }

        operator_key = _canonical_key(trial.operator.as_report())
        prepared = self._operators.get(operator_key)
        if prepared is None:
            prepared = _prepare_operator(trial.operator)
            self._operators[operator_key] = prepared
        encoding_key = (operator_key, trial.access_model)
        encoding = self._encodings.get(encoding_key)
        if encoding is None:
            encoding = _build_access_model(prepared, trial.access_model)
            self._encodings[encoding_key] = encoding

        target_key = _canonical_key(trial.target.as_report())
        design_key = (operator_key, target_key, trial.access_model, trial.degree)
        design = self._designs.get(design_key)
        if design is None:
            design = _design_target(prepared, encoding, trial.target, trial.degree)
            self._designs[design_key] = design

        resource_reports = self._resources.get(design_key)
        if resource_reports is None:
            resource_reports = tuple(
                estimate_encoding_aware_resources(encoding, coeffs).as_report()
                for _, coeffs in design.components
            )
            self._resources[design_key] = resource_reports

        signal_eigenvalues = np.linalg.eigvalsh(prepared.matrix) / encoding.alpha
        approximate = _evaluate_components(design.components, signal_eigenvalues)
        difference = approximate - design.exact_values
        max_error = float(np.max(np.abs(difference)))
        rms_error = float(np.sqrt(np.mean(np.abs(difference) ** 2)))
        reference_norm = float(np.linalg.norm(design.exact_values))
        relative_error = float(
            np.linalg.norm(difference) / max(reference_norm, np.finfo(float).eps)
        )
        grid = np.linspace(-1.0, 1.0, 2001)
        boundedness_max = float(
            max(
                np.max(np.abs(np.polynomial.polynomial.polyval(grid, coeffs)))
                for _, coeffs in design.components
            )
        )

        synthesis_reports: list[dict[str, object]] = []
        if trial.attempt_synthesis:
            adapter = (
                trial.phase_solver
                if trial.phase_solver.startswith("pennylane:")
                else f"pennylane:{trial.phase_solver}"
            )
            for component_name, coeffs in design.components:
                synthesis = synthesize_phases_with_adapter(
                    coeffs,
                    adapter=adapter,
                    reconstruction_num_points=65,
                )
                synthesis_reports.append(
                    {"component": component_name, **synthesis.as_report()}
                )

        total_gates = _sum_optional_ints(
            report.get("total_gates") for report in resource_reports
        )
        total_wires = _max_optional_ints(
            report.get("total_wires") for report in resource_reports
        )
        signal_calls = sum(
            int(report["signal_operator_calls"]) for report in resource_reports
        )
        inverse_calls = sum(
            int(report["inverse_signal_operator_calls"]) for report in resource_reports
        )
        estimator_models = sorted(
            {str(report["estimator_model"]) for report in resource_reports}
        )
        effective_degree = max(
            polynomial_degree(coeffs) for _, coeffs in design.components
        )
        condition_number = float(np.linalg.cond(prepared.matrix))
        synthesis_succeeded = (
            None
            if not synthesis_reports
            else all(bool(report["succeeded"]) for report in synthesis_reports)
        )

        return {
            "status": "completed",
            "summary": {
                "matrix_dimension": prepared.matrix.shape[0],
                "requested_degree": trial.degree,
                "effective_degree": effective_degree,
                "normalization_alpha": float(encoding.alpha),
                "condition_number_2": condition_number,
                "minimum_signal_value": float(np.min(signal_eigenvalues)),
                "maximum_signal_value": float(np.max(signal_eigenvalues)),
                "approximation_max_error": max_error,
                "approximation_rms_error": rms_error,
                "operator_relative_error": relative_error,
                "met_tolerance": relative_error <= trial.tolerance,
                "boundedness_max": boundedness_max,
                "boundedness_margin": 1.0 - boundedness_max,
                "polynomial_component_count": len(design.components),
                "signal_operator_calls": signal_calls,
                "inverse_signal_operator_calls": inverse_calls,
                "total_wires": total_wires,
                "total_gates": total_gates,
                "logical_depth": None,
                "logical_success_probability": None,
                "estimator_models": estimator_models,
                "phase_synthesis_succeeded": synthesis_succeeded,
            },
            "operator": {
                "name": prepared.name,
                "kind": prepared.kind,
                "positive_shift": prepared.positive_shift,
                "matrix": prepared.matrix,
                "metadata": prepared.metadata,
            },
            "target_design": {
                "kind": trial.target.kind,
                "metadata": design.metadata,
                "exact_spectral_values": design.exact_values,
            },
            "access_model": encoding.as_report(),
            "components": [
                {
                    "name": name,
                    "coeffs": coeffs,
                    "resource_estimate": resource,
                }
                for (name, coeffs), resource in zip(
                    design.components,
                    resource_reports,
                    strict=True,
                )
            ],
            "synthesis": synthesis_reports,
            "truth_contract": {
                "accuracy_reference": "finite_eigendecomposition",
                "resource_layer": "encoding-aware-logical-estimate",
                "finite_shots_executed": False,
                "hardware_executed": False,
                "normalization_changes_polynomial_signal_domain": True,
                "logical_depth_available": False,
                "success_probability_available": False,
                "complex_components_are_separate_sequences": (
                    len(design.components) > 1
                ),
                "omitted_components": [
                    "coherent_combination_of_complex_components",
                    "application_state_preparation",
                    "postselection_or_amplitude_amplification",
                    "provider_compilation_and_routing",
                ],
            },
        }


def accuracy_resource_frontier_spec(
    *,
    degrees: tuple[int, ...] = (3, 5, 7),
    tolerances: tuple[float, ...] = (0.2,),
    attempt_synthesis: bool = False,
) -> ResearchSweepSpec:
    """Return the built-in Poisson/Ising/graph frontier experiment."""
    return ResearchSweepSpec(
        name="qsvt-accuracy-resource-frontier",
        study="accuracy-resource-frontier",
        operators=(
            ResearchOperatorSpec(
                "poisson-1d-4",
                "poisson_1d",
                {"size": 4},
            ),
            ResearchOperatorSpec(
                "transverse-field-ising-2",
                "ising",
                {"n_spins": 2, "coupling": 1.0, "transverse_field": 0.6},
            ),
            ResearchOperatorSpec(
                "path-graph-laplacian-4",
                "graph_laplacian",
                {"size": 4, "topology": "path"},
            ),
        ),
        targets=tuple(
            ResearchTargetSpec(name=kind.replace("_", "-"), kind=kind)
            for kind in _FRONTIER_TARGETS
        ),
        access_models=_FRONTIER_ACCESS_MODELS,
        degrees=degrees,
        tolerances=tolerances,
        phase_solvers=("root-finding",),
        shots=(None,),
        seeds=(0,),
        noise_models=("ideal",),
        attempt_synthesis=attempt_synthesis,
        metadata={
            "purpose": (
                "Compare accuracy and logical resources under access-model-specific "
                "normalization for the same finite operators."
            ),
            "resource_claim": "logical estimate, not hardware runtime",
        },
    )


def run_accuracy_resource_frontier(
    spec: ResearchSweepSpec | None = None,
    *,
    output_dir: str | Path | None = None,
    resume: bool = True,
    fail_fast: bool = False,
) -> AccuracyResourceFrontierResult:
    """Run a configured or built-in accuracy-resource frontier study."""
    resolved_spec = accuracy_resource_frontier_spec() if spec is None else spec
    if resolved_spec.study != "accuracy-resource-frontier":
        raise ValueError("frontier spec study must be 'accuracy-resource-frontier'.")
    evaluator = AccuracyResourceFrontierEvaluator()
    sweep = run_research_sweep(
        resolved_spec,
        evaluator,
        output_dir=output_dir,
        resume=resume,
        fail_fast=fail_fast,
    )
    rows = tuple(accuracy_resource_frontier_rows(sweep.trial_reports))
    resolved_output = None if output_dir is None else Path(output_dir)
    result = AccuracyResourceFrontierResult(
        sweep=sweep,
        frontier_rows=rows,
        output_dir=resolved_output,
    )
    if resolved_output is not None:
        save_report(
            {
                "schema_name": "qsvt-accuracy-resource-frontier-rows",
                "schema_version": "1.0",
                "mode": "accuracy-resource-frontier-rows",
                "rows": rows,
                "truth_contract": result.as_report()["truth_contract"],
            },
            resolved_output / "frontier.json",
        )
        write_accuracy_resource_pareto_csv(rows, resolved_output / "pareto.csv")
        save_report(result.as_report(), resolved_output / "frontier-manifest.json")
    return result


def accuracy_resource_frontier_rows(
    trial_reports: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> list[dict[str, object]]:
    """Classify completed sweep rows by three-objective Pareto dominance."""
    rows = research_summary_rows(trial_reports)
    grouping_fields = (
        "operator",
        "target",
        "tolerance",
        "phase_solver",
        "shots",
        "seed",
        "noise_model",
    )
    for row in rows:
        row["pareto_optimal"] = False
    for row in rows:
        if not _pareto_eligible(row):
            continue
        group = tuple(row.get(field) for field in grouping_fields)
        competitors = [
            candidate
            for candidate in rows
            if _pareto_eligible(candidate)
            and tuple(candidate.get(field) for field in grouping_fields) == group
        ]
        row["pareto_optimal"] = not any(
            _dominates(candidate, row)
            for candidate in competitors
            if candidate is not row
        )
    return rows


def write_accuracy_resource_pareto_csv(
    frontier_rows: tuple[dict[str, object], ...] | list[dict[str, object]],
    path: str | Path,
) -> Path:
    """Write Pareto-optimal configurations as a compact CSV table."""
    rows = [row for row in frontier_rows if row.get("pareto_optimal")]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    leading = [
        "trial_id",
        "operator",
        "target",
        "access_model",
        "requested_degree",
        "tolerance",
        "met_tolerance",
        "operator_relative_error",
        "normalization_alpha",
        "signal_operator_calls",
        "total_wires",
        "total_gates",
        "pareto_optimal",
    ]
    discovered = sorted({key for row in rows for key in row} - set(leading))
    fieldnames = leading + discovered
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})
    return output_path


def _prepare_operator(spec: ResearchOperatorSpec) -> _PreparedOperator:
    parameters = spec.parameters
    if spec.kind == "poisson_1d":
        size = _as_int(parameters.get("size", 4), "size")
        _, original = dirichlet_laplacian_1d(size)
        construction = {"size": size, "boundary": "Dirichlet"}
    elif spec.kind == "ising":
        n_spins = _as_int(parameters.get("n_spins", 2), "n_spins")
        coupling = _as_float(parameters.get("coupling", 1.0), "coupling")
        field = _as_float(
            parameters.get("transverse_field", 0.6),
            "transverse_field",
        )
        periodic = bool(parameters.get("periodic", False))
        original = ising_hamiltonian(
            n_spins,
            coupling=coupling,
            transverse_field=field,
            periodic=periodic,
        )
        construction = {
            "n_spins": n_spins,
            "coupling": coupling,
            "transverse_field": field,
            "periodic": periodic,
        }
    elif spec.kind == "graph_laplacian":
        size = _as_int(parameters.get("size", 4), "size")
        topology = str(parameters.get("topology", "path"))
        original = _graph_laplacian(size, topology)
        construction = {"size": size, "topology": topology}
    elif spec.kind == "sparse_structured":
        size = _as_int(parameters.get("size", 4), "size")
        diagonal = _as_float(parameters.get("diagonal", 2.0), "diagonal")
        off_diagonal = _as_float(
            parameters.get("off_diagonal", -0.5),
            "off_diagonal",
        )
        original = np.diag(np.full(size, diagonal))
        original += np.diag(np.full(size - 1, off_diagonal), 1)
        original += np.diag(np.full(size - 1, off_diagonal), -1)
        construction = {
            "size": size,
            "diagonal": diagonal,
            "off_diagonal": off_diagonal,
        }
    elif spec.kind == "matrix":
        original = np.asarray(parameters.get("matrix"), dtype=complex)
        construction = {"source": "explicit-matrix"}
    else:
        raise ValueError(
            "operator kind must be poisson_1d, ising, graph_laplacian, "
            "sparse_structured, or matrix."
        )
    matrix = np.asarray(np.real_if_close(original))
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("research operators must resolve to a square matrix.")
    if not np.allclose(matrix, matrix.conj().T, atol=1e-10):
        raise ValueError("research operators must be Hermitian.")
    if matrix.shape[0] < 2 or matrix.shape[0] & (matrix.shape[0] - 1):
        raise ValueError(
            "frontier operator dimensions must be powers of two and at least two."
        )
    eigenvalues = np.linalg.eigvalsh(matrix)
    width = float(np.max(eigenvalues) - np.min(eigenvalues))
    floor_fraction = _as_float(
        parameters.get("positive_floor_fraction", 0.05),
        "positive_floor_fraction",
    )
    if not 0.0 < floor_fraction < 1.0:
        raise ValueError("positive_floor_fraction must lie between zero and one.")
    target_floor = floor_fraction * max(width, 1.0)
    shift = max(0.0, target_floor - float(np.min(eigenvalues)))
    positive_matrix = matrix + shift * np.eye(matrix.shape[0])
    positive_eigenvalues = np.linalg.eigvalsh(positive_matrix)
    return _PreparedOperator(
        name=spec.name,
        kind=spec.kind,
        original_matrix=matrix,
        matrix=np.real_if_close(positive_matrix),
        positive_shift=float(shift),
        metadata={
            "construction": construction,
            "original_min_eigenvalue": float(np.min(eigenvalues)),
            "original_max_eigenvalue": float(np.max(eigenvalues)),
            "positive_min_eigenvalue": float(np.min(positive_eigenvalues)),
            "positive_max_eigenvalue": float(np.max(positive_eigenvalues)),
            "positive_floor_fraction": floor_fraction,
        },
    )


def _build_access_model(
    prepared: _PreparedOperator,
    access_model: str,
) -> BlockEncodingSpec:
    matrix = prepared.matrix
    metadata = {
        "research_operator": prepared.name,
        "positive_shift": prepared.positive_shift,
    }
    if access_model == "embedding":
        return matrix_block_encoding_spec(
            matrix,
            block_encoding="embedding",
            metadata=metadata,
        )
    if access_model == "fable":
        alpha = max(
            float(np.linalg.norm(matrix, ord=2)),
            matrix.shape[0] * float(np.max(np.abs(matrix))),
        )
        return matrix_block_encoding_spec(
            matrix,
            alpha=alpha,
            block_encoding="fable",
            metadata=metadata,
        )
    if access_model in {"prepselprep", "qubitization"}:
        system_qubits = (matrix.shape[0] - 1).bit_length()
        system_wires = tuple(range(system_qubits))
        operator = qml.pauli_decompose(matrix, wire_order=system_wires)
        coefficients, _ = operator.terms()
        control_count = max(1, (len(coefficients) - 1).bit_length())
        controls = tuple(range(system_qubits, system_qubits + control_count))
        return pennylane_operator_block_encoding_spec(
            operator,
            encoding_wires=controls,
            block_encoding=cast(
                Literal["prepselprep", "qubitization"],
                access_model,
            ),
            metadata={**metadata, "pauli_term_count": len(coefficients)},
        )
    raise ValueError(
        "access_model must be embedding, fable, prepselprep, or qubitization."
    )


def _design_target(
    prepared: _PreparedOperator,
    encoding: BlockEncodingSpec,
    target: ResearchTargetSpec,
    degree: int,
) -> _TargetDesign:
    eigenvalues = np.linalg.eigvalsh(prepared.matrix) / encoding.alpha
    low = float(np.min(eigenvalues))
    high = float(np.max(eigenvalues))
    span = max(high - low, 1e-8)
    num_points = _as_int(target.parameters.get("num_points", 1001), "num_points")
    if num_points < degree + 1:
        raise ValueError("target num_points must be at least degree + 1.")

    components: tuple[tuple[str, np.ndarray], ...]
    if target.kind == "inverse":
        gamma = _as_float(target.parameters.get("gamma", low), "gamma")
        gamma = min(max(gamma, 1e-6), 1.0 - 1e-6)
        coeffs = design_positive_inverse_polynomial(
            gamma=gamma,
            degree=degree,
            num_points=num_points,
        )
        exact = gamma / eigenvalues
        components = (("real", coeffs),)
        metadata = {"gamma": gamma, "target": "gamma/x"}
    elif target.kind == "projector":
        cutoff_fraction = _as_float(
            target.parameters.get("cutoff_fraction", 0.4),
            "cutoff_fraction",
        )
        sharpness = _as_float(
            target.parameters.get("sharpness", 12.0),
            "sharpness",
        )
        cutoff = low + cutoff_fraction * span
        slope = sharpness / span

        def projector(x: np.ndarray) -> np.ndarray:
            return 0.5 * (1.0 - np.tanh(slope * (x - cutoff)))

        coeffs = _fit_monomial(projector, degree, num_points)
        exact = projector(eigenvalues)
        components = (("real", coeffs),)
        metadata = {
            "cutoff": cutoff,
            "cutoff_fraction": cutoff_fraction,
            "sharpness": sharpness,
        }
    elif target.kind == "band_filter":
        center_fraction = _as_float(
            target.parameters.get("center_fraction", 0.5),
            "center_fraction",
        )
        width_fraction = _as_float(
            target.parameters.get("width_fraction", 0.2),
            "width_fraction",
        )
        center = low + center_fraction * span
        width = max(width_fraction * span, 1e-6)

        def band_filter(x: np.ndarray) -> np.ndarray:
            return np.exp(-0.5 * ((x - center) / width) ** 2)

        coeffs = _fit_monomial(band_filter, degree, num_points)
        exact = band_filter(eigenvalues)
        components = (("real", coeffs),)
        metadata = {
            "center": center,
            "center_fraction": center_fraction,
            "width": width,
            "width_fraction": width_fraction,
        }
    elif target.kind == "resolvent":
        omega_fraction = _as_float(
            target.parameters.get("omega_fraction", 0.5),
            "omega_fraction",
        )
        eta_fraction = _as_float(
            target.parameters.get("eta_fraction", 0.15),
            "eta_fraction",
        )
        omega = low + omega_fraction * span
        eta = max(eta_fraction * span, 1e-6)

        def resolvent(x: np.ndarray) -> np.ndarray:
            return eta / (omega + 1j * eta - x)

        real_coeffs = _fit_monomial(lambda x: resolvent(x).real, degree, num_points)
        imag_coeffs = _fit_monomial(lambda x: resolvent(x).imag, degree, num_points)
        exact = resolvent(eigenvalues)
        components = (("real", real_coeffs), ("imag", imag_coeffs))
        metadata = {
            "omega": omega,
            "omega_fraction": omega_fraction,
            "eta": eta,
            "eta_fraction": eta_fraction,
            "scaling": "eta/(omega+i*eta-x)",
        }
    else:
        raise ValueError(
            "target kind must be inverse, projector, band_filter, or resolvent."
        )
    return _TargetDesign(
        components=tuple(
            (name, np.asarray(coeffs, dtype=float)) for name, coeffs in components
        ),
        exact_values=np.asarray(exact),
        metadata={
            **metadata,
            "signal_domain_min": low,
            "signal_domain_max": high,
            "requested_degree": degree,
        },
    )


def _fit_monomial(function: Any, degree: int, num_points: int) -> np.ndarray:
    chebyshev = chebyshev_fit_function(
        function,
        degree=degree,
        num_points=num_points,
    )
    return chebyshev_to_monomial(chebyshev)


def _evaluate_components(
    components: tuple[tuple[str, np.ndarray], ...],
    eigenvalues: np.ndarray,
) -> np.ndarray:
    values = np.zeros(eigenvalues.shape, dtype=complex)
    for name, coeffs in components:
        component = np.polynomial.polynomial.polyval(eigenvalues, coeffs)
        values += component if name == "real" else 1j * component
    return np.real_if_close(values)


def _graph_laplacian(size: int, topology: str) -> np.ndarray:
    if size < 2:
        raise ValueError("graph size must be at least two.")
    if topology not in {"path", "cycle"}:
        raise ValueError("graph topology must be 'path' or 'cycle'.")
    adjacency = np.zeros((size, size), dtype=float)
    for index in range(size - 1):
        adjacency[index, index + 1] = 1.0
        adjacency[index + 1, index] = 1.0
    if topology == "cycle" and size > 2:
        adjacency[0, -1] = 1.0
        adjacency[-1, 0] = 1.0
    return np.diag(np.sum(adjacency, axis=1)) - adjacency


def _pareto_eligible(row: dict[str, object]) -> bool:
    return bool(
        row.get("status") == "completed"
        and _is_finite_number(row.get("operator_relative_error"))
        and _is_finite_number(row.get("total_gates"))
        and _is_finite_number(row.get("total_wires"))
    )


def _dominates(left: dict[str, object], right: dict[str, object]) -> bool:
    left_values = (
        _as_float(left["operator_relative_error"], "operator_relative_error"),
        _as_float(left["total_gates"], "total_gates"),
        _as_float(left["total_wires"], "total_wires"),
    )
    right_values = (
        _as_float(right["operator_relative_error"], "operator_relative_error"),
        _as_float(right["total_gates"], "total_gates"),
        _as_float(right["total_wires"], "total_wires"),
    )
    return all(a <= b for a, b in zip(left_values, right_values, strict=True)) and any(
        a < b for a, b in zip(left_values, right_values, strict=True)
    )


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and bool(np.isfinite(value))


def _as_int(value: object, name: str) -> int:
    if isinstance(value, (str, bytes, bytearray, int, float)):
        return int(value)
    raise TypeError(f"{name} must be an integer-compatible value.")


def _as_float(value: object, name: str) -> float:
    if isinstance(value, (str, bytes, bytearray, int, float)):
        return float(value)
    raise TypeError(f"{name} must be a real-number-compatible value.")


def _sum_optional_ints(values: Any) -> int | None:
    normalized = list(values)
    if any(value is None for value in normalized):
        return None
    return sum(int(value) for value in normalized)


def _max_optional_ints(values: Any) -> int | None:
    normalized = list(values)
    if any(value is None for value in normalized):
        return None
    return max(int(value) for value in normalized)


def _canonical_key(value: object) -> str:
    return json.dumps(
        report_to_jsonable({"value": value})["value"],
        sort_keys=True,
        separators=(",", ":"),
    )


def _csv_value(value: object) -> object:
    converted = report_to_jsonable({"value": value})["value"]
    if isinstance(converted, (dict, list)):
        return json.dumps(converted, sort_keys=True, separators=(",", ":"))
    return converted


__all__ = [
    "AccuracyResourceFrontierEvaluator",
    "AccuracyResourceFrontierResult",
    "accuracy_resource_frontier_rows",
    "accuracy_resource_frontier_spec",
    "run_accuracy_resource_frontier",
    "write_accuracy_resource_pareto_csv",
]
