"""Accuracy-driven planning and execution for finite QSVT workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import pennylane as qml

from .block_encoding import BlockEncodingSpec, matrix_block_encoding_spec
from .degree import DegreeSearchCandidate
from .diagnostics import expectation_value
from .execution import (
    BlockEncodingQSVTExecutionResult,
    CoherentQSVTComponent,
    CoherentQSVTExecutionResult,
    execute_qsvt_component_lcu_from_spec,
    execute_qsvt_from_spec,
)
from .polynomials import polynomial_degree
from .resources import EncodingAwareResourceEstimate, estimate_encoding_aware_resources
from .synthesis import (
    PhaseSynthesisResult,
    available_phase_solver_adapters,
    certify_polynomial_boundedness,
    synthesize_phases_cached,
    synthesize_phases_with_adapter,
)
from .workflow import QSVTProblemWorkflowResult, qsvt_problem_workflow

PlanningTarget = Literal[
    "linear_system",
    "spectral_projector",
    "ground_state_filter",
    "hamiltonian_simulation",
    "resolvent",
    "singular_value_filter",
    "singular_value_pseudoinverse",
]


@dataclass(frozen=True)
class QSVTProblemSpec:
    """Finite problem definition plus data-loading and observable metadata."""

    operator: object
    rhs: np.ndarray | None = None
    state: np.ndarray | None = None
    source: np.ndarray | None = None
    reference_matrix: np.ndarray | None = None
    observables: dict[str, np.ndarray] = field(default_factory=dict)
    name: str = "finite-qsvt-problem"
    metadata: dict[str, object] = field(default_factory=dict)

    def as_report(self) -> dict[str, object]:
        """Return metadata without serializing an opaque operator or callable."""
        return {
            "name": self.name,
            "operator_type": type(self.operator).__name__,
            "has_rhs": self.rhs is not None,
            "has_state": self.state is not None,
            "has_source": self.source is not None,
            "has_reference_matrix": self.reference_matrix is not None,
            "observables": sorted(self.observables),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class QSVTTransformSpec:
    """Target transform and accuracy-driven degree-search configuration."""

    target: PlanningTarget
    tolerance: float = 1e-2
    min_degree: int = 2
    max_degree: int = 32
    degree_step: int = 2
    degree: int | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def degrees(self) -> tuple[int, ...]:
        """Return the explicit or searched requested degrees."""
        if self.degree is not None:
            if self.degree < 0:
                raise ValueError("degree must be non-negative.")
            return (int(self.degree),)
        if self.degree_step <= 0:
            raise ValueError("degree_step must be positive.")
        if self.min_degree < 0 or self.max_degree < self.min_degree:
            raise ValueError(
                "degree bounds must satisfy 0 <= min_degree <= max_degree."
            )
        return tuple(range(self.min_degree, self.max_degree + 1, self.degree_step))


@dataclass(frozen=True)
class QSVTExecutionConfig:
    """Planning policy for synthesis, access-model fallback, and execution."""

    execute: bool = False
    device_name: str = "default.qubit"
    shots: int | None = None
    block_encoding: Literal["embedding", "fable"] = "embedding"
    angle_solvers: tuple[str, ...] = ("root-finding", "iterative")
    allow_matrix_fallback: bool = True
    gate_set: tuple[str, ...] | None = None
    reconstruction_num_points: int = 129
    phase_reconstruction_tolerance: float = 1e-6


@dataclass(frozen=True)
class QSVTPlan:
    """Selected finite workflow, synthesis results, access model, and resources."""

    problem: QSVTProblemSpec
    transform: QSVTTransformSpec
    execution_config: QSVTExecutionConfig
    input_kind: str
    matrix: np.ndarray
    workflow: QSVTProblemWorkflowResult
    degree_candidates: tuple[DegreeSearchCandidate, ...]
    selected_degree: int
    achieved_error: float
    error_metric: str
    met_tolerance: bool
    coefficient_sets: tuple[tuple[str, np.ndarray], ...]
    synthesis_results: tuple[tuple[str, PhaseSynthesisResult], ...]
    block_encoding_spec: BlockEncodingSpec | None
    access_model_status: str
    access_model_reason: str
    resource_estimates: tuple[tuple[str, EncodingAwareResourceEstimate], ...]
    coherent_resource_estimate: dict[str, object] | None
    execution_ready: bool
    planning_warnings: tuple[str, ...]

    def as_report(self) -> dict[str, object]:
        """Return the complete auditable plan."""
        return {
            "mode": "qsvt-execution-plan",
            "implementation_kind": "accuracy-driven-finite-qsvt-plan",
            "problem": self.problem.as_report(),
            "target": self.transform.target,
            "tolerance": self.transform.tolerance,
            "input_kind": self.input_kind,
            "selected_degree": self.selected_degree,
            "achieved_error": self.achieved_error,
            "error_metric": self.error_metric,
            "met_tolerance": self.met_tolerance,
            "degree_candidates": [
                candidate.as_report() for candidate in self.degree_candidates
            ],
            "coefficient_sets": {
                name: coeffs for name, coeffs in self.coefficient_sets
            },
            "workflow": self.workflow.as_report(),
            "synthesis": {
                name: result.as_report() for name, result in self.synthesis_results
            },
            "block_encoding_spec": (
                None
                if self.block_encoding_spec is None
                else self.block_encoding_spec.as_report()
            ),
            "access_model_status": self.access_model_status,
            "access_model_reason": self.access_model_reason,
            "resources": {
                name: estimate.as_report() for name, estimate in self.resource_estimates
            },
            "coherent_resource_estimate": self.coherent_resource_estimate,
            "execution_ready": self.execution_ready,
            "planning_warnings": list(self.planning_warnings),
            "truth_contract": {
                "implemented_components": [
                    "finite_problem_resolution",
                    "tolerance_driven_degree_search",
                    "polynomial_and_classical_reference_validation",
                    "phase_synthesis_with_fallback",
                    "block_encoding_access_model_selection",
                    "encoding_aware_resource_estimation",
                ],
                "is_executed": False,
                "matrix_fallback_is_scalable_access_model": False,
            },
        }


@dataclass(frozen=True)
class QSVTPlanRunResult:
    """Execution results, observable values, and a component error ledger."""

    plan: QSVTPlan
    executions: tuple[
        tuple[
            str,
            BlockEncodingQSVTExecutionResult | CoherentQSVTExecutionResult,
        ],
        ...,
    ]
    observables: dict[str, dict[str, float | complex | None]]
    error_budget: dict[str, float | None]
    succeeded: bool
    execution_reason: str

    def as_report(self) -> dict[str, object]:
        """Return plan and execution results with explicit error components."""
        return {
            "mode": "qsvt-plan-execution",
            "implementation_kind": "planned-block-encoding-qsvt-execution",
            "succeeded": self.succeeded,
            "execution_reason": self.execution_reason,
            "plan": self.plan.as_report(),
            "executions": {
                name: result.as_report() for name, result in self.executions
            },
            "observables": self.observables,
            "error_budget": self.error_budget,
            "truth_contract": {
                "is_end_to_end_quantum_algorithm": False,
                "executed_components": [name for name, _ in self.executions],
                "full_state_outputs_are_simulator_validation_data": True,
                "omitted_components": [
                    "scalable_application_state_preparation",
                    "postselection_or_amplitude_amplification",
                    "tomographic_readout_of_full_solution_vectors",
                    "provider_native_compilation",
                ],
            },
        }


def plan_qsvt(
    problem: QSVTProblemSpec,
    transform: QSVTTransformSpec,
    execution: QSVTExecutionConfig | None = None,
) -> QSVTPlan:
    """Plan a finite QSVT workflow from target tolerance through resources."""
    if not isinstance(problem, QSVTProblemSpec):
        raise TypeError("problem must be a QSVTProblemSpec.")
    if not isinstance(transform, QSVTTransformSpec):
        raise TypeError("transform must be a QSVTTransformSpec.")
    config = execution or QSVTExecutionConfig()
    tolerance = float(transform.tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite.")
    reserved_parameters = {
        "apply_qsvt",
        "attempt_synthesis",
        "degree",
        "rhs",
        "source",
        "state",
    }
    conflicts = sorted(reserved_parameters.intersection(transform.parameters))
    if conflicts:
        raise ValueError(
            "transform.parameters must not override planner-managed arguments: "
            + ", ".join(conflicts)
        )
    matrix, provided_spec, input_kind = _resolve_problem_operator(problem)

    candidates: list[DegreeSearchCandidate] = []
    successful: list[tuple[DegreeSearchCandidate, QSVTProblemWorkflowResult]] = []
    selected: tuple[DegreeSearchCandidate, QSVTProblemWorkflowResult] | None = None
    metric_name = _metric_name(transform.target, problem)
    for degree in transform.degrees():
        try:
            workflow = qsvt_problem_workflow(
                transform.target,
                matrix,
                rhs=problem.rhs,
                state=problem.state,
                source=problem.source,
                degree=degree,
                attempt_synthesis=False,
                apply_qsvt=False,
                **transform.parameters,
            )
            error = _workflow_error(workflow.result, metric_name)
            coeff_sets = _coefficient_sets(workflow.result)
            single_sequence = all(
                _single_sequence_realizable(coeffs) for _, coeffs in coeff_sets
            )
            usable = single_sequence or not config.execute
            candidate = DegreeSearchCandidate(
                requested_degree=int(degree),
                polynomial_degree=max(
                    polynomial_degree(coeffs) for _, coeffs in coeff_sets
                ),
                error=error,
                met_tolerance=error <= tolerance and usable,
                metadata={
                    "coefficient_components": [name for name, _ in coeff_sets],
                    "single_sequence_realizable": single_sequence,
                },
                error_type=(None if usable else "PolynomialRealizabilityError"),
                error_message=(
                    None
                    if usable
                    else "execution requires single-sequence-realizable components"
                ),
            )
            candidates.append(candidate)
            if usable:
                successful.append((candidate, workflow))
            if candidate.met_tolerance:
                selected = (candidate, workflow)
                break
        except Exception as exc:
            candidates.append(
                DegreeSearchCandidate(
                    requested_degree=int(degree),
                    polynomial_degree=None,
                    error=None,
                    met_tolerance=False,
                    metadata={},
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )

    if selected is None and successful:
        selected = min(
            successful,
            key=lambda item: (
                cast(float, item[0].error),
                item[0].requested_degree,
            ),
        )
    if selected is None:
        failures = "; ".join(
            f"degree {item.requested_degree}: {item.error_message}"
            for item in candidates
        )
        raise ValueError(f"no degree candidate produced a workflow: {failures}")

    selected_candidate, workflow = selected
    coeff_sets = _coefficient_sets(workflow.result)
    synthesis_results = tuple(
        (name, _synthesize_with_fallback(coeffs, config)) for name, coeffs in coeff_sets
    )
    selected_spec, access_status, access_reason = _select_execution_spec(
        workflow.result,
        provided_spec,
        config,
    )
    resources = (
        tuple(
            (
                name,
                estimate_encoding_aware_resources(
                    selected_spec,
                    coeffs,
                    gate_set=config.gate_set,
                ),
            )
            for name, coeffs in coeff_sets
        )
        if selected_spec is not None
        else ()
    )
    coherent_resources = _coherent_resource_plan(
        transform.target,
        coeff_sets,
        synthesis_results,
        resources,
    )
    warnings: list[str] = []
    if not selected_candidate.met_tolerance:
        warnings.append(
            "No candidate met the requested tolerance; the lowest-error candidate "
            "was selected."
        )
    synthesis_ok = all(
        result.succeeded
        and result.reconstruction_max_error is not None
        and result.reconstruction_max_error <= config.phase_reconstruction_tolerance
        for _, result in synthesis_results
    )
    if not synthesis_ok:
        warnings.append(
            "At least one polynomial component did not meet the phase "
            "reconstruction tolerance."
        )
    if access_status == "matrix-fallback":
        warnings.append(
            "Execution uses a finite matrix block encoding rather than the supplied "
            "application access model."
        )
    state = _execution_input(problem, transform.target)
    ready = bool(selected_spec is not None and state is not None and synthesis_ok)
    return QSVTPlan(
        problem=problem,
        transform=transform,
        execution_config=config,
        input_kind=input_kind,
        matrix=matrix,
        workflow=workflow,
        degree_candidates=tuple(candidates),
        selected_degree=selected_candidate.requested_degree,
        achieved_error=cast(float, selected_candidate.error),
        error_metric=metric_name,
        met_tolerance=selected_candidate.met_tolerance,
        coefficient_sets=coeff_sets,
        synthesis_results=synthesis_results,
        block_encoding_spec=selected_spec,
        access_model_status=access_status,
        access_model_reason=access_reason,
        resource_estimates=resources,
        coherent_resource_estimate=coherent_resources,
        execution_ready=ready,
        planning_warnings=tuple(warnings),
    )


def run_qsvt_plan(plan: QSVTPlan) -> QSVTPlanRunResult:
    """Execute every synthesizable polynomial component in a prepared plan."""
    if not isinstance(plan, QSVTPlan):
        raise TypeError("plan must be a QSVTPlan.")
    if not plan.execution_config.execute:
        return QSVTPlanRunResult(
            plan=plan,
            executions=(),
            observables={},
            error_budget=_error_budget(plan, ()),
            succeeded=False,
            execution_reason="execution_config.execute is false",
        )
    if not plan.execution_ready or plan.block_encoding_spec is None:
        return QSVTPlanRunResult(
            plan=plan,
            executions=(),
            observables={},
            error_budget=_error_budget(plan, ()),
            succeeded=False,
            execution_reason=plan.access_model_reason,
        )
    raw_state = _execution_input(plan.problem, plan.transform.target)
    if raw_state is None:
        raise ValueError(
            "the selected target does not provide an execution input state."
        )
    state = np.asarray(raw_state, dtype=complex)
    state = state / np.linalg.norm(state)
    synth_by_name = dict(plan.synthesis_results)
    executions: list[
        tuple[
            str,
            BlockEncodingQSVTExecutionResult | CoherentQSVTExecutionResult,
        ]
    ] = []
    if plan.transform.target == "hamiltonian_simulation":
        components = _hamiltonian_coherent_components(plan)
        coherent_result = execute_qsvt_component_lcu_from_spec(
            plan.block_encoding_spec,
            components,
            state,
            angle_solver=_builtin_solver_name(
                next(iter(synth_by_name.values())).angle_solver
            ),
            device_name=plan.execution_config.device_name,
            shots=plan.execution_config.shots,
            normalize_state=True,
            reconstruction_num_points=(plan.execution_config.reconstruction_num_points),
            phase_reconstruction_tolerance=(
                plan.execution_config.phase_reconstruction_tolerance
            ),
            raise_on_failure=False,
        )
        executions.append(("coherent_cosine_sine", coherent_result))
    else:
        for name, coeffs in plan.coefficient_sets:
            synthesis = synth_by_name[name]
            projectors = _projectors_from_synthesis(plan.block_encoding_spec, synthesis)
            component_result = execute_qsvt_from_spec(
                plan.block_encoding_spec,
                coeffs,
                state,
                projectors=projectors,
                angle_solver=_builtin_solver_name(synthesis.angle_solver),
                device_name=plan.execution_config.device_name,
                shots=plan.execution_config.shots,
                normalize_state=True,
                raise_on_failure=False,
            )
            executions.append((name, component_result))

    succeeded = bool(executions) and all(result.succeeded for _, result in executions)
    return QSVTPlanRunResult(
        plan=plan,
        executions=tuple(executions),
        observables=_observable_reports(plan.problem, tuple(executions)),
        error_budget=_error_budget(plan, tuple(executions)),
        succeeded=succeeded,
        execution_reason=(
            "all planned polynomial components executed"
            if succeeded
            else "one or more planned polynomial components failed"
        ),
    )


def _resolve_problem_operator(
    problem: QSVTProblemSpec,
) -> tuple[np.ndarray, BlockEncodingSpec | None, str]:
    source = problem.operator
    if isinstance(source, BlockEncodingSpec):
        if source.kind in {"dense-matrix", "sparse-matrix"}:
            return source.dense_matrix(), source, f"block-encoding:{source.kind}"
        if source.kind == "pennylane-operator":
            operator = cast(Any, source.source)
            matrix = np.asarray(
                qml.matrix(operator, wire_order=list(operator.wires)),
                dtype=complex,
            )
            return np.real_if_close(matrix), source, "block-encoding:pennylane-operator"
        if problem.reference_matrix is None:
            raise ValueError(
                "opaque custom block encodings require reference_matrix for planning."
            )
        return np.asarray(problem.reference_matrix), source, "block-encoding:custom"
    if isinstance(source, qml.operation.Operator):
        matrix = np.asarray(qml.matrix(source, wire_order=list(source.wires)))
        return np.real_if_close(matrix), None, "pennylane-operator"
    matrix = np.asarray(source)
    if matrix.ndim != 2:
        raise ValueError("problem operator must resolve to a two-dimensional matrix.")
    return matrix, None, "finite-matrix"


def _metric_name(target: PlanningTarget, problem: QSVTProblemSpec) -> str:
    return {
        "linear_system": "polynomial_relative_error",
        "spectral_projector": (
            "state_weight_error"
            if problem.state is not None
            else "operator_relative_error"
        ),
        "ground_state_filter": "reference_state_error",
        "hamiltonian_simulation": "state_relative_error",
        "resolvent": (
            "response_relative_error"
            if problem.source is not None
            else "operator_relative_error"
        ),
        "singular_value_filter": (
            "output_relative_error"
            if problem.state is not None
            else "operator_relative_error"
        ),
        "singular_value_pseudoinverse": "solution_relative_error",
    }[target]


def _workflow_error(result: Any, metric: str) -> float:
    value = getattr(result, metric, None)
    if value is None:
        raise ValueError(f"workflow did not produce required error metric {metric!r}.")
    error = float(value)
    if not np.isfinite(error) or error < 0.0:
        raise ValueError(
            f"workflow metric {metric!r} is not a finite non-negative error."
        )
    return error


def _coefficient_sets(result: Any) -> tuple[tuple[str, np.ndarray], ...]:
    coefficient_sets = []
    for name in ("coeffs", "cos_coeffs", "sin_coeffs", "real_coeffs", "imag_coeffs"):
        value = getattr(result, name, None)
        if value is not None:
            coefficient_sets.append((name, np.asarray(value, dtype=float)))
    if not coefficient_sets:
        raise ValueError("workflow result contains no polynomial coefficient sets.")
    return tuple(coefficient_sets)


def _single_sequence_realizable(coeffs: np.ndarray) -> bool:
    from .synthesis import classify_polynomial_realizability

    return classify_polynomial_realizability(coeffs).single_sequence_realizable


def _synthesize_with_fallback(
    coeffs: np.ndarray,
    config: QSVTExecutionConfig,
) -> PhaseSynthesisResult:
    if not config.angle_solvers:
        raise ValueError("angle_solvers must contain at least one solver name.")
    attempts: list[PhaseSynthesisResult] = []
    adapters = set(available_phase_solver_adapters())
    for solver in config.angle_solvers:
        result = (
            synthesize_phases_with_adapter(
                coeffs,
                adapter=solver,
                reconstruction_num_points=config.reconstruction_num_points,
            )
            if solver in adapters
            else synthesize_phases_cached(
                coeffs,
                angle_solver=solver,
                reconstruction_num_points=config.reconstruction_num_points,
            )
        )
        attempts.append(result)
        if (
            result.succeeded
            and result.reconstruction_max_error is not None
            and result.reconstruction_max_error <= config.phase_reconstruction_tolerance
        ):
            return result
    succeeded = [attempt for attempt in attempts if attempt.succeeded]
    if succeeded:
        return min(
            succeeded,
            key=lambda attempt: (
                np.inf
                if attempt.reconstruction_max_error is None
                else attempt.reconstruction_max_error
            ),
        )
    return attempts[-1]


def _signal_matrix(result: Any) -> np.ndarray | None:
    scaled = getattr(result, "scaled_operator", None)
    if scaled is not None:
        return np.asarray(scaled.matrix)
    matrix = getattr(result, "matrix", None)
    scale = getattr(result, "scale", None)
    if matrix is not None and scale is not None:
        return np.asarray(matrix) / float(scale)
    return None


def _spec_signal_matrix(spec: BlockEncodingSpec) -> np.ndarray | None:
    if spec.kind in {"dense-matrix", "sparse-matrix"}:
        return spec.dense_matrix() / spec.alpha
    if spec.kind == "pennylane-operator":
        source = cast(Any, spec.source)
        return (
            np.asarray(qml.matrix(source, wire_order=list(source.wires))) / spec.alpha
        )
    return None


def _select_execution_spec(
    result: Any,
    provided: BlockEncodingSpec | None,
    config: QSVTExecutionConfig,
) -> tuple[BlockEncodingSpec | None, str, str]:
    signal = _signal_matrix(result)
    if provided is not None and signal is not None:
        provided_signal = _spec_signal_matrix(provided)
        if (
            provided_signal is not None
            and provided_signal.shape == signal.shape
            and np.allclose(
                provided_signal,
                signal,
                atol=1e-9,
                rtol=1e-9,
            )
        ):
            return (
                provided,
                "provided-access-model",
                ("The provided block encoding matches the workflow signal operator."),
            )
    if signal is None:
        return None, "unavailable", "workflow does not expose a finite signal matrix"
    if not config.allow_matrix_fallback:
        return (
            None,
            "normalization-mismatch",
            (
                "The provided access model does not encode the workflow's normalized "
                "signal operator and matrix fallback is disabled."
            ),
        )
    try:
        spec = matrix_block_encoding_spec(
            signal,
            alpha=1.0,
            block_encoding=config.block_encoding,
        )
    except Exception as exc:
        return None, "unavailable", str(exc)
    return (
        spec,
        "matrix-fallback",
        (
            "A finite matrix block encoding of the workflow-normalized signal operator "
            "was constructed for validation."
        ),
    )


def _execution_input(
    problem: QSVTProblemSpec,
    target: PlanningTarget,
) -> np.ndarray | None:
    if target in {"linear_system", "singular_value_pseudoinverse"}:
        return problem.rhs
    if target == "resolvent":
        return problem.source
    return problem.state


def _hamiltonian_coherent_components(
    plan: QSVTPlan,
) -> tuple[CoherentQSVTComponent, CoherentQSVTComponent]:
    result = plan.workflow.result
    coeffs = dict(plan.coefficient_sets)
    if "cos_coeffs" not in coeffs or "sin_coeffs" not in coeffs:
        raise ValueError(
            "Hamiltonian execution requires cosine and sine coefficient sets."
        )
    offset = float(result.scaled_operator.offset)
    time = float(result.time)
    physical_phase = complex(np.exp(-1j * offset * time))
    return (
        CoherentQSVTComponent(
            "cosine",
            coeffs["cos_coeffs"],
            physical_phase,
        ),
        CoherentQSVTComponent(
            "sine",
            coeffs["sin_coeffs"],
            -1j * physical_phase,
        ),
    )


def _coherent_resource_plan(
    target: PlanningTarget,
    coefficient_sets: tuple[tuple[str, np.ndarray], ...],
    synthesis_results: tuple[tuple[str, PhaseSynthesisResult], ...],
    resource_estimates: tuple[tuple[str, EncodingAwareResourceEstimate], ...],
) -> dict[str, object] | None:
    if target != "hamiltonian_simulation":
        return None
    syntheses = dict(synthesis_results)
    estimates = dict(resource_estimates)
    if any(name not in estimates for name, _ in coefficient_sets):
        return None
    component_rows: list[dict[str, object]] = []
    lcu_normalization = 0.0
    for name, coeffs in coefficient_sets:
        extrema_weight = float(certify_polynomial_boundedness(coeffs).max_abs_value)
        if extrema_weight <= 1e-15:
            continue
        weight = extrema_weight / (1.0 - 1e-8)
        lcu_normalization += weight
        synthesis = syntheses[name]
        estimate = estimates[name]
        component_rows.append(
            {
                "name": name,
                "lcu_weight": weight,
                "polynomial_degree": estimate.degree,
                "phase_count_per_sequence": (
                    None if synthesis.angles is None else int(synthesis.angles.size)
                ),
                "forward_signal_operator_calls": estimate.signal_operator_calls
                + estimate.inverse_signal_operator_calls,
                "adjoint_signal_operator_calls": estimate.signal_operator_calls
                + estimate.inverse_signal_operator_calls,
                "total_signal_operator_calls": 2
                * (
                    estimate.signal_operator_calls
                    + estimate.inverse_signal_operator_calls
                ),
                "encoding_aware_total_wires": estimate.total_wires,
                "encoding_aware_total_gates_per_sequence": estimate.total_gates,
            }
        )
    gate_counts = [
        row["encoding_aware_total_gates_per_sequence"]
        for row in component_rows
        if row["encoding_aware_total_gates_per_sequence"] is not None
    ]
    wire_counts = [
        row["encoding_aware_total_wires"]
        for row in component_rows
        if row["encoding_aware_total_wires"] is not None
    ]
    phase_counts = [
        int(cast(Any, row["phase_count_per_sequence"]))
        for row in component_rows
        if row["phase_count_per_sequence"] is not None
    ]
    selector_count = max(1, (2 * len(component_rows) - 1).bit_length())
    return {
        "mode": "coherent-qsvt-resource-plan",
        "implementation_kind": "encoding-aware-coherent-lcu-resource-estimate",
        "component_sequence_count": len(component_rows),
        "selected_unitary_branch_count": 2 * len(component_rows),
        "selection_ancilla_count": selector_count,
        "lcu_normalization": lcu_normalization,
        "component_resource_ledger": component_rows,
        "total_phase_count": 2 * sum(phase_counts),
        "total_signal_operator_calls": sum(
            int(cast(Any, row["total_signal_operator_calls"])) for row in component_rows
        ),
        "estimated_total_wires": (
            None
            if not wire_counts
            else max(int(cast(Any, value)) for value in wire_counts) + selector_count
        ),
        "estimated_total_gates": (
            None
            if len(gate_counts) != len(component_rows)
            else 2 * sum(int(cast(Any, value)) for value in gate_counts) + 3
        ),
        "selector_overhead_model": [
            "selector_state_preparation",
            "branch_phase_diagonal",
            "selector_uncomputation",
        ],
        "success_probability": None,
        "amplitude_amplification_overhead": None,
        "omitted_costs": [
            "application_state_preparation",
            "amplitude_amplification",
            "application_readout_or_tomography",
            "provider_compilation_and_routing",
            "fault_tolerant_synthesis",
        ],
    }


def _projectors_from_synthesis(
    spec: BlockEncodingSpec,
    synthesis: PhaseSynthesisResult,
) -> list[qml.operation.Operator] | None:
    if synthesis.angles is None:
        return None
    if spec.kind == "pennylane-operator":
        source = cast(Any, spec.source)
        wires = list(spec.encoding_wires) + list(source.wires)
        dimensions = [spec.logical_shape[0]] * len(synthesis.angles)
    elif spec.block_encoding == "fable":
        wires = list(spec.encoding_wires)
        dimensions = [spec.logical_shape[0]] * len(synthesis.angles)
    else:
        wires = list(spec.encoding_wires)
        rows, columns = spec.logical_shape
        dimensions = [
            rows if index % 2 == 0 else columns
            for index in range(len(synthesis.angles))
        ]
    return [
        qml.PCPhase(float(angle), dim=int(dimension), wires=wires)
        for angle, dimension in zip(synthesis.angles, dimensions, strict=True)
    ]


def _builtin_solver_name(name: str) -> str:
    if name.startswith("adapter:"):
        return "root-finding"
    return name.replace("pennylane:", "")


def _observable_reports(
    problem: QSVTProblemSpec,
    executions: tuple[
        tuple[
            str,
            BlockEncodingQSVTExecutionResult | CoherentQSVTExecutionResult,
        ],
        ...,
    ],
) -> dict[str, dict[str, float | complex | None]]:
    reports: dict[str, dict[str, float | complex | None]] = {}
    for component, execution in executions:
        if execution.classical_reference_output is None:
            continue
        reference = np.asarray(execution.classical_reference_output, dtype=complex)
        reference_norm = float(np.linalg.norm(reference))
        circuit = (
            None
            if execution.logical_output is None
            else np.real(np.asarray(execution.logical_output, dtype=complex))
        )
        circuit_norm = None if circuit is None else float(np.linalg.norm(circuit))
        for name, observable in problem.observables.items():
            key = f"{component}:{name}"
            reports[key] = {
                "polynomial_reference": (
                    None
                    if reference_norm == 0.0
                    else expectation_value(observable, reference / reference_norm)
                ),
                "circuit": (
                    None
                    if circuit is None or circuit_norm == 0.0
                    else expectation_value(
                        observable, circuit / cast(float, circuit_norm)
                    )
                ),
            }
    return reports


def _error_budget(
    plan: QSVTPlan,
    executions: tuple[
        tuple[
            str,
            BlockEncodingQSVTExecutionResult | CoherentQSVTExecutionResult,
        ],
        ...,
    ],
) -> dict[str, float | None]:
    synthesis_errors = [
        result.reconstruction_max_error
        for _, result in plan.synthesis_results
        if result.reconstruction_max_error is not None
    ]
    execution_errors = [
        result.logical_output_relative_error
        for _, result in executions
        if result.logical_output_relative_error is not None
    ]
    sampling_errors = [
        result.maximum_probability_standard_error
        for _, result in executions
        if result.maximum_probability_standard_error is not None
    ]
    components = {
        "workflow_approximation_error": plan.achieved_error,
        "phase_reconstruction_error": max(synthesis_errors, default=None),
        "circuit_vs_polynomial_error": max(execution_errors, default=None),
        "maximum_sampling_standard_error": max(sampling_errors, default=None),
    }
    finite = [float(value) for value in components.values() if value is not None]
    components["additive_error_proxy"] = sum(finite) if finite else None
    return components


__all__ = [
    "PlanningTarget",
    "QSVTExecutionConfig",
    "QSVTPlan",
    "QSVTPlanRunResult",
    "QSVTProblemSpec",
    "QSVTTransformSpec",
    "plan_qsvt",
    "run_qsvt_plan",
]
