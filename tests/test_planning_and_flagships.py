import json

import numpy as np
import pennylane as qml
import pytest

import qsvt.resources as resource_module
from qsvt.__main__ import main
from qsvt.block_encoding import (
    matrix_block_encoding_spec,
    pennylane_operator_block_encoding_spec,
)
from qsvt.degree import search_design_degree, search_polynomial_degree
from qsvt.flagship import poisson_qsvt_workflow, spectral_filter_qsvt_workflow
from qsvt.planning import (
    QSVTExecutionConfig,
    QSVTProblemSpec,
    QSVTTransformSpec,
    plan_qsvt,
    run_qsvt_plan,
)
from qsvt.resources import estimate_encoding_aware_resources
from qsvt.synthesis import (
    clear_phase_synthesis_cache,
    phase_synthesis_cache_info,
    register_phase_solver_adapter,
    synthesize_phases_cached,
    synthesize_phases_with_adapter,
    unregister_phase_solver_adapter,
)


def test_generic_degree_search_selects_first_passing_candidate_and_keeps_metadata():
    result = search_polynomial_degree(
        lambda degree: np.array([0.0, 1.0 / degree]),
        lambda coeffs, degree: (1.0 / degree, {"scale": coeffs[1]}),
        tolerance=0.3,
        degrees=[1, 2, 4, 8],
        metric="synthetic_error",
    )

    assert result.met_tolerance is True
    assert result.chosen_degree == 4
    assert result.achieved_error == pytest.approx(0.25)
    assert len(result.candidates) == 3
    assert result.candidates[-1].metadata["scale"] == pytest.approx(0.25)


def test_design_degree_search_uses_public_design_diagnostics():
    result = search_design_degree(
        "sign",
        gamma=0.3,
        tolerance=1.0,
        min_degree=3,
        max_degree=3,
        num_points=101,
        bounded_num_points=201,
    )

    assert result.chosen_degree == 3
    assert result.chosen_coeffs is not None
    assert result.metric == "max_error"


def test_phase_synthesis_cache_and_convention_safe_adapter():
    clear_phase_synthesis_cache()
    first = synthesize_phases_cached([0.0, 1.0], reconstruction_num_points=33)
    second = synthesize_phases_cached([0.0, 1.0], reconstruction_num_points=33)

    assert first is second
    assert phase_synthesis_cache_info() == {"size": 1, "hits": 1, "misses": 1}
    with pytest.raises(ValueError, match="converter is required"):
        register_phase_solver_adapter(
            "unsafe-test-adapter",
            lambda coeffs, routine: coeffs,
            convention="foreign-qsp-convention",
        )

    name = "converted-test-adapter"
    unregister_phase_solver_adapter(name)

    def solver(coeffs, routine):
        return qml.poly_to_angles(coeffs, routine, angle_solver="root-finding")

    register_phase_solver_adapter(
        name,
        solver,
        convention="test-foreign-convention",
        converter=lambda angles, routine: angles,
    )
    try:
        adapted = synthesize_phases_with_adapter(
            [0.0, 1.0],
            adapter=name,
            reconstruction_num_points=33,
        )
        adapted_plan = plan_qsvt(
            QSVTProblemSpec(np.diag([1.0, 2.0]), rhs=np.ones(2)),
            QSVTTransformSpec(
                "linear_system",
                tolerance=2.0,
                degree=3,
                parameters={"num_points": 101, "bounded_num_points": 201},
            ),
            QSVTExecutionConfig(execute=False, angle_solvers=(name,)),
        )
    finally:
        unregister_phase_solver_adapter(name)

    assert adapted.succeeded is True
    assert adapted.implementation_kind.endswith(name)
    assert adapted.reconstruction_max_error is not None
    assert adapted.reconstruction_max_error < 1e-8
    assert "explicit adapter conversion" in adapted.convention
    assert adapted_plan.synthesis_results[0][1].angle_solver == f"adapter:{name}"


def test_encoding_aware_resources_cover_matrix_and_pauli_lcu_models():
    matrix_spec = matrix_block_encoding_spec(np.diag([0.2, 0.8]))
    matrix_estimate = estimate_encoding_aware_resources(
        matrix_spec,
        [0.0, 0.0, 1.0],
    )
    operator = qml.dot([0.4, 0.6], [qml.Z(1), qml.X(1)])
    pauli_spec = pennylane_operator_block_encoding_spec(
        operator,
        encoding_wires=[0],
        block_encoding="prepselprep",
    )
    pauli_estimate = estimate_encoding_aware_resources(pauli_spec, [0.0, 1.0])

    assert matrix_estimate.normalization_alpha == pytest.approx(0.8)
    assert matrix_estimate.signal_operator_calls == 1
    assert matrix_estimate.inverse_signal_operator_calls == 1
    assert matrix_estimate.estimator_model == "arbitrary-unitary-block-encoding"
    assert pauli_estimate.normalization_alpha == pytest.approx(1.0)
    assert pauli_estimate.estimator_model == "pauli-lcu-qubitization"
    assert pauli_estimate.total_gates is not None
    assert (
        pauli_estimate.as_report()["truth_contract"]["is_fault_tolerant_estimate"]
        is False
    )


def test_encoding_aware_resources_have_a_version_independent_fallback(monkeypatch):
    def unavailable_estimator():
        raise ModuleNotFoundError("simulated older PennyLane")

    monkeypatch.setattr(
        resource_module,
        "_load_pennylane_estimator",
        unavailable_estimator,
    )
    operator = qml.dot([0.4, 0.6], [qml.Z(1), qml.X(1)])
    spec = pennylane_operator_block_encoding_spec(
        operator,
        encoding_wires=[0],
        block_encoding="prepselprep",
    )

    estimate = estimate_encoding_aware_resources(spec, [0.0, 1.0])
    truth = estimate.as_report()["truth_contract"]

    assert estimate.estimator_available is True
    assert estimate.estimator_kind == "qsvt.logical-primitive-fallback"
    assert estimate.estimator_model == "pauli-lcu-qubitization"
    assert estimate.total_wires is not None
    assert estimate.total_gates is not None
    assert estimate.total_gates > 0
    assert estimate.error_type == "ModuleNotFoundError"
    assert truth["uses_logical_primitive_fallback"] is True
    assert truth["uses_pennylane_logical_estimator"] is False


@pytest.mark.parametrize(
    ("operator", "expected_kind"),
    [
        (np.diag([1.0, 2.0]), "finite-matrix"),
        (
            qml.dot([1.5, 0.5], [qml.I(0), qml.Z(0)]),
            "pennylane-operator",
        ),
        (
            matrix_block_encoding_spec(np.diag([1.0, 2.0])),
            "block-encoding:dense-matrix",
        ),
    ],
)
def test_planner_accepts_matrix_operator_and_block_encoding_spec(
    operator, expected_kind
):
    plan = plan_qsvt(
        QSVTProblemSpec(operator, rhs=np.array([1.0, 1.0])),
        QSVTTransformSpec(
            "linear_system",
            tolerance=2.0,
            degree=3,
            parameters={"num_points": 101, "bounded_num_points": 201},
        ),
        QSVTExecutionConfig(execute=False),
    )

    assert plan.input_kind == expected_kind
    assert plan.selected_degree == 3
    assert plan.achieved_error >= 0.0
    assert plan.block_encoding_spec is not None


def test_accuracy_driven_plan_executes_selected_polynomial():
    plan = plan_qsvt(
        QSVTProblemSpec(
            np.diag([1.0, 2.0]),
            rhs=np.array([1.0, 1.0]),
            observables={"population_0": np.diag([1.0, 0.0])},
        ),
        QSVTTransformSpec(
            "linear_system",
            tolerance=0.4,
            min_degree=3,
            max_degree=9,
            degree_step=2,
            parameters={"num_points": 201, "bounded_num_points": 401},
        ),
        QSVTExecutionConfig(execute=True, reconstruction_num_points=65),
    )
    execution = run_qsvt_plan(plan)

    assert plan.met_tolerance is True
    assert plan.selected_degree <= 9
    assert plan.execution_ready is True
    assert execution.succeeded is True
    assert execution.executions[0][1].logical_output_relative_error < 1e-6
    assert execution.observables["coeffs:population_0"]["circuit"] is not None


def test_hamiltonian_plan_executes_coherent_cosine_sine_combination():
    plan = plan_qsvt(
        QSVTProblemSpec(
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            state=np.array([1.0, 0.0]),
        ),
        QSVTTransformSpec(
            "hamiltonian_simulation",
            tolerance=1e-5,
            degree=8,
            parameters={"time": 0.5, "num_points": 201},
        ),
        QSVTExecutionConfig(execute=True, reconstruction_num_points=65),
    )
    execution = run_qsvt_plan(plan)

    assert plan.execution_ready is True
    assert plan.coherent_resource_estimate is not None
    assert plan.coherent_resource_estimate["component_sequence_count"] == 2
    assert plan.coherent_resource_estimate["selection_ancilla_count"] == 2
    assert plan.coherent_resource_estimate["total_signal_operator_calls"] == 30
    assert execution.succeeded is True
    assert execution.executions[0][0] == "coherent_cosine_sine"
    assert execution.executions[0][1].logical_output_relative_error < 1e-6
    assert execution.error_budget["circuit_vs_polynomial_error"] < 1e-6


def test_spectral_filter_flagship_executes_pauli_lcu_qsvt():
    operator = qml.dot(
        [0.4, 0.3, 0.2],
        [qml.Z(0), qml.Z(1), qml.X(0)],
    )
    result = spectral_filter_qsvt_workflow(
        operator,
        np.ones(4) / 2.0,
        lower=-0.4,
        upper=0.4,
        tolerance=0.16,
        min_degree=2,
        max_degree=4,
        degree_step=2,
        num_points=401,
        execute=True,
    )

    assert result.degree_search.met_tolerance is True
    assert result.polynomial_operator_error <= 0.16
    assert result.synthesis.succeeded is True
    assert result.execution is not None and result.execution.succeeded is True
    assert result.execution.logical_output_relative_error < 1e-8
    assert result.resource_estimate.estimator_model == "pauli-lcu-qubitization"
    assert result.error_budget["polynomial_approximation_error"] <= 0.16
    acceptance = result.as_report()["acceptance"]
    assert acceptance["accepted_for_stated_scope"] is True
    assert acceptance["full_qsvt_acceptance"] is True


def test_poisson_flagship_compares_direct_cg_polynomial_and_circuit_paths():
    result = poisson_qsvt_workflow(
        4,
        tolerance=0.4,
        min_degree=5,
        max_degree=5,
        access_model="prepselprep",
        num_points=401,
        execute=True,
    )

    assert result.conjugate_gradient["converged"] is True
    assert result.degree_search.met_tolerance is True
    assert result.polynomial_relative_error <= 0.4
    assert result.execution is not None and result.execution.succeeded is True
    assert result.circuit_relative_error is not None
    assert result.circuit_relative_error == pytest.approx(
        result.polynomial_relative_error,
        abs=1e-8,
    )
    assert result.error_budget["discretization_error"] is not None
    assert result.resource_estimate.estimator_model == "pauli-lcu-qubitization"
    acceptance = result.as_report()["acceptance"]
    assert acceptance["accepted_for_stated_scope"] is True
    assert acceptance["full_qsvt_acceptance"] is True


def test_new_cli_commands_emit_machine_readable_reports(capsys):
    main(
        [
            "degree-search",
            "--kind",
            "sign",
            "--gamma",
            "0.3",
            "--tolerance",
            "1.0",
            "--min-degree",
            "3",
            "--max-degree",
            "3",
            "--num-points",
            "101",
            "--bounded-num-points",
            "201",
        ]
    )
    degree_report = json.loads(capsys.readouterr().out)
    main(
        [
            "poisson-qsvt",
            "--n-points",
            "4",
            "--tolerance",
            "0.4",
            "--min-degree",
            "5",
            "--max-degree",
            "5",
            "--num-points",
            "201",
            "--no-execute",
        ]
    )
    poisson_report = json.loads(capsys.readouterr().out)

    assert degree_report["mode"] == "polynomial-degree-search"
    assert degree_report["chosen_degree"] == 3
    assert poisson_report["mode"] == "poisson-qsvt-flagship"
    assert poisson_report["execution"] is None
    assert poisson_report["acceptance"]["accepted_for_stated_scope"] is False
