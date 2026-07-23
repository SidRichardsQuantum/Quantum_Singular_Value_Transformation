import numpy as np
import pytest

from qsvt._algorithm_reports import (
    algorithm_truth_contract,
    algorithm_truth_contract_issues,
)
from qsvt.algorithms import (
    fermi_dirac_occupation_workflow,
    fixed_point_amplification_workflow,
    ground_state_filtering_workflow,
    hamiltonian_simulation_workflow,
    linear_system_comparison_summary_table,
    linear_system_comparison_workflow,
    linear_system_workflow,
    matrix_log_entropy_workflow,
    quantum_walk_search_workflow,
    resolvent_workflow,
    singular_value_filtering_workflow,
    singular_value_pseudoinverse_workflow,
    spectral_counting_workflow,
    spectral_density_workflow,
    spectral_thresholding_workflow,
    thermal_gibbs_workflow,
    write_linear_system_comparison_csv,
)
from qsvt.reports import report_to_jsonable, validate_report_schema

HERMitian_2X2 = np.array(
    [
        [0.0, 0.2],
        [0.2, 1.0],
    ],
)
STATE_2 = np.array([1.0, 0.3])


def _validated_algorithm_report(result):
    report = report_to_jsonable(result.as_report())
    compatibility = validate_report_schema(report, require_schema=True)

    assert compatibility.supported is True
    assert report["schema_name"] == "qsvt-algorithm-workflow"
    assert report["schema_version"] == "1.1"
    contract = report["truth_contract"]
    assert algorithm_truth_contract_issues(contract) == ()
    assert contract["execution_tier"] in {"polynomial_core", "qsvt_circuit"}
    assert contract["qnode_executed"] is False
    assert contract["physical_device_executed"] is False
    assert contract["resource_completeness"] == "partial"
    assert contract["polynomial_evidence"]["component_count"] >= 1
    for component in contract["polynomial_evidence"]["components"].values():
        assert {
            "coeffs",
            "design_domain",
            "qsvt_certification_domain",
            "output_prefactor",
            "boundedness_certificate",
            "parity",
            "realizability_kind",
            "single_sequence_realizable",
            "requires_parity_decomposition",
        } <= component.keys()
    return report


def test_truth_contract_derives_tier_from_realizability_and_execution():
    compatible = algorithm_truth_contract(
        "compatible-test",
        target="test",
        qsvt_check="succeeded",
        polynomials={"transform": [0.0, 0.0, 1.0]},
    )
    mixed = algorithm_truth_contract(
        "mixed-test",
        target="test",
        qsvt_check="succeeded",
        polynomials={"transform": [0.5, 0.5]},
    )

    assert compatible["execution_tier"] == "qsvt_circuit"
    assert compatible["truth_status"] == "verified_finite_qsvt_circuit"
    assert algorithm_truth_contract_issues(compatible) == ()
    assert mixed["execution_tier"] == "polynomial_core"
    assert mixed["truth_status"] == (
        "circuit_evaluated_without_qsvt_realizability_certificate"
    )
    assert mixed["polynomial_evidence"]["requires_parity_decomposition"] is True
    assert algorithm_truth_contract_issues(mixed) == ()


def test_truth_contract_audit_rejects_semantic_contradictions():
    contract = algorithm_truth_contract(
        "audit-test",
        target="test",
        polynomials={"transform": [0.0, 0.0, 1.0]},
    )
    contract["execution_tier"] = "hardware_execution"
    component = contract["polynomial_evidence"]["components"]["transform"]
    component["parity"] = "odd"

    issues = algorithm_truth_contract_issues(contract)

    assert "hardware_tier_without_physical_execution" in issues
    assert "polynomial_evidence_mismatch:transform:parity" in issues


def test_singular_value_filtering_workflow_regression():
    matrix = np.array(
        [
            [2.0, 0.0],
            [0.0, 0.5],
            [0.25, 0.0],
        ],
    )
    result = singular_value_filtering_workflow(
        matrix,
        degree=20,
        cutoff=0.4,
        sharpness=8.0,
        input_vector=np.array([1.0, -0.5]),
        num_points=1001,
    )
    report = _validated_algorithm_report(result)

    assert report["mode"] == "singular-value-filtering-workflow"
    assert report["implementation_kind"] == "dense-svd-polynomial-workflow"
    evidence = report["truth_contract"]["polynomial_evidence"]["components"]
    assert evidence["singular_value_filter"]["design_domain"] == [0.0, 1.0]
    assert evidence["singular_value_filter"]["qsvt_certification_domain"] == [
        -1.0,
        1.0,
    ]
    assert result.polynomial_matrix.shape == matrix.shape
    assert result.reference_matrix.shape == matrix.shape
    assert result.operator_relative_error < 3e-4
    assert result.output_relative_error is not None
    assert result.output_relative_error < 3e-4


def test_singular_value_pseudoinverse_workflow_regression():
    matrix = np.array(
        [
            [2.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
    )
    rhs = np.array([2.0, -1.0, 0.2])

    result = singular_value_pseudoinverse_workflow(
        matrix,
        rhs,
        degree=18,
        cutoff=0.45,
        num_points=501,
    )
    report = _validated_algorithm_report(result)

    assert report["mode"] == "singular-value-pseudoinverse-workflow"
    assert result.polynomial_solution.shape == (2,)
    assert result.solution_relative_error < 0.08
    assert result.residual_norm < 0.25
    assert np.isclose(result.reference_residual_norm, 0.2)


def test_fermi_dirac_occupation_workflow_regression():
    matrix = np.diag([-1.0, 0.2, 1.5])

    result = fermi_dirac_occupation_workflow(
        matrix,
        chemical_potential=0.0,
        beta=4.0,
        degree=24,
        state=np.array([1.0, 0.0, 1.0]),
        num_points=1001,
    )
    report = _validated_algorithm_report(result)

    assert report["mode"] == "fermi-dirac-occupation-workflow"
    assert result.operator_relative_error < 1e-5
    assert result.state_occupation_error is not None
    assert result.state_occupation_error < 1e-5
    assert 1.0 < result.reference_particle_number < 2.0


def test_matrix_log_entropy_workflow_regression():
    matrix = np.diag([0.2, 0.8])

    result = matrix_log_entropy_workflow(
        matrix,
        degree=24,
        epsilon=0.05,
        num_points=1201,
    )
    report = _validated_algorithm_report(result)

    assert report["mode"] == "matrix-log-entropy-workflow"
    assert result.log_operator_relative_error < 1e-3
    assert result.entropy_operator_relative_error < 1e-3
    assert result.reference_entropy > 0.0


def test_spectral_counting_workflow_regression():
    matrix = np.diag([-2.0, -0.5, 0.1, 0.8, 2.0])

    result = spectral_counting_workflow(
        matrix,
        lower=-0.7,
        upper=1.0,
        degree=24,
        sharpness=12.0,
        num_points=1201,
        probe_count=16,
        random_seed=123,
    )
    report = _validated_algorithm_report(result)

    assert report["mode"] == "spectral-counting-workflow"
    assert result.exact_count == 3
    assert abs(result.polynomial_count - 3.0) < 0.2
    assert result.stochastic_count is not None
    assert result.probe_count == 16


def test_fixed_point_amplification_workflow_regression():
    score = np.diag([0.1, 0.8])
    state = np.array([1.0, 1.0])

    result = fixed_point_amplification_workflow(
        score,
        state,
        rounds=4,
    )
    report = _validated_algorithm_report(result)

    assert report["mode"] == "fixed-point-amplification-workflow"
    assert result.degree == 4
    assert result.operator_relative_error < 1e-12
    assert result.state_relative_error < 1e-12
    assert result.amplified_score > result.initial_score


def test_linear_system_workflow_regression():
    matrix = np.array(
        [
            [2.0, 0.25],
            [0.25, 1.25],
        ],
    )
    rhs = np.array([1.0, -0.5])

    result = linear_system_workflow(
        matrix,
        rhs,
        degree=8,
        num_points=201,
        bounded_num_points=401,
        attempt_synthesis=False,
        apply_qsvt=False,
    )
    report = _validated_algorithm_report(result)

    assert report["mode"] == "linear-system-workflow"
    assert report["truth_contract"]["truth_status"] == (
        "validated_polynomial_core_requires_parity_combination"
    )
    assert report["truth_contract"]["is_end_to_end_quantum_algorithm"] is False
    assert report["truth_contract"]["pennylane_qsvt_check"] == "not_attempted"
    assert report["implementation_kind"] == "dense-spectral-polynomial-workflow"
    inverse_evidence = report["truth_contract"]["polynomial_evidence"]["components"][
        "inverse"
    ]
    assert np.isclose(
        inverse_evidence["output_prefactor"],
        1.0 / (result.gamma * result.scaled_operator.scale),
    )
    assert np.isclose(result.gamma, 0.5657414540893352)
    assert np.isclose(result.scaled_min_eigenvalue, result.gamma)
    assert np.isclose(result.scaled_max_eigenvalue, 1.0)
    assert np.isclose(result.condition_number_2, 1.7675918792439986)
    assert np.isclose(result.gamma_condition_proxy, 1.0 / result.gamma)
    assert result.polynomial_residual_norm < 0.06
    assert result.polynomial_relative_error < 0.06
    assert result.compatibility["attempted_pennylane_synthesis"] is False
    assert result.qsvt_solution is None
    assert report["resource_proxy"]["degree"] == 8
    assert report["resource_proxy"]["gamma"] == report["gamma"]
    assert report["resource_proxy"]["requires_block_encoding"] is True
    assert report["resource_proxy"]["requires_readout_strategy"] is True
    assert (
        "oracle_or_block_encoding_construction"
        in report["resource_proxy"]["omitted_layers"]
    )


def test_linear_system_workflow_reports_ill_conditioning_metadata():
    matrix = np.diag([0.05, 1.0])
    rhs = np.array([1.0, 0.25])

    result = linear_system_workflow(
        matrix,
        rhs,
        degree=14,
        num_points=301,
        bounded_num_points=601,
        attempt_synthesis=False,
        apply_qsvt=False,
    )

    assert np.isclose(result.scaled_min_eigenvalue, 0.05)
    assert np.isclose(result.condition_number_2, 20.0)
    assert np.isclose(result.gamma_condition_proxy, 20.0)
    assert result.as_report()["resource_proxy"]["condition_number_2"] == 20.0


def test_linear_system_comparison_workflow_rows():
    matrix = np.array(
        [
            [2.0, 0.25],
            [0.25, 1.25],
        ],
    )
    rhs = np.array([1.0, -0.5])

    comparison = linear_system_comparison_workflow(
        matrix,
        rhs,
        degree=8,
        num_points=201,
        bounded_num_points=401,
        attempt_synthesis=False,
        apply_qsvt=False,
        cg_tolerance=1e-12,
    )
    report = _validated_algorithm_report(comparison)
    rows = {row["solver"]: row for row in report["rows"]}

    assert report["mode"] == "linear-system-comparison-workflow"
    assert report["implementation_kind"] == "linear-system-solver-comparison"
    assert set(rows) == {
        "dense_solve",
        "conjugate_gradient",
        "qsvt_style_polynomial_inverse",
    }
    assert rows["dense_solve"]["relative_solution_error"] == 0.0
    assert rows["conjugate_gradient"]["converged"] is True
    assert rows["conjugate_gradient"]["relative_solution_error"] < 1e-12
    assert rows["qsvt_style_polynomial_inverse"]["degree"] == 8
    assert rows["qsvt_style_polynomial_inverse"]["relative_solution_error"] < 0.06
    assert report["resource_proxy"]["degree"] == 8
    assert report["linear_system_workflow"]["mode"] == "linear-system-workflow"
    summary_rows = linear_system_comparison_summary_table(comparison)
    assert summary_rows[0]["solver"] == "dense_solve"
    assert summary_rows[0]["matrix_dimension"] == 2
    assert summary_rows[-1]["solver"] == "qsvt_style_polynomial_inverse"


def test_linear_system_comparison_workflow_can_include_hhl_execution():
    comparison = linear_system_comparison_workflow(
        np.diag([1.0, 2.0]),
        np.array([1.0, 1.0]) / np.sqrt(2.0),
        degree=4,
        gamma=0.5,
        num_points=201,
        bounded_num_points=401,
        attempt_synthesis=False,
        apply_qsvt=False,
        include_hhl_execution=True,
        hhl_num_phase_qubits=2,
        hhl_evolution_time=np.pi / 2.0,
        hhl_rotation_scale_C=1.0,
        hhl_eigenvalue_lower_bound=1.0,
    )
    rows = {row["solver"]: row for row in comparison.as_report()["rows"]}
    hhl = rows["hhl_circuit_execution"]

    assert hhl["implementation_kind"] == "pennylane-qnode-statevector-hhl-execution"
    assert hhl["status"] == "ok"
    assert hhl["is_executable_hhl_circuit"] is True
    assert hhl["uses_dense_time_evolution"] is True
    assert np.isclose(hhl["success_probability"], 0.625, atol=1e-10)
    assert hhl["fidelity"] is not None
    assert hhl["fidelity"] > 1.0 - 1e-10
    assert hhl["state_error"] is not None
    assert hhl["state_error"] < 1e-10

    summary_rows = {
        row["solver"]: row for row in linear_system_comparison_summary_table(comparison)
    }
    assert summary_rows["hhl_circuit_execution"]["phase_qubits"] == 2
    assert summary_rows["hhl_circuit_execution"]["num_gates"] > 0


def test_linear_system_comparison_hhl_row_reports_incompatible_system():
    comparison = linear_system_comparison_workflow(
        np.diag([1.0, 2.0, 3.0]),
        np.array([1.0, 0.0, 0.0]),
        degree=4,
        gamma=1.0 / 3.0,
        num_points=201,
        bounded_num_points=401,
        attempt_synthesis=False,
        apply_qsvt=False,
        include_hhl_execution=True,
    )
    rows = {row["solver"]: row for row in comparison.as_report()["rows"]}

    assert rows["hhl_circuit_execution"]["status"] == "failed"
    assert "power of two" in rows["hhl_circuit_execution"]["error"]


def test_quantum_walk_search_workflow_regression():
    adjacency = np.ones((4, 4)) - np.eye(4)

    result = quantum_walk_search_workflow(
        adjacency,
        marked_vertex=0,
        degree=14,
        num_points=801,
        num_time_points=121,
    )
    report = _validated_algorithm_report(result)

    assert report["mode"] == "quantum-walk-search-workflow"
    assert report["implementation_kind"] == "dense-spectral-polynomial-workflow"
    assert report["truth_contract"]["truth_status"] == (
        "validated_qsvt_compatible_polynomial_core"
    )
    assert report["truth_contract"]["is_end_to_end_quantum_algorithm"] is False
    assert report["marked_vertex"] == 0
    assert np.isclose(result.gamma, 0.25)
    assert np.isclose(result.best_time, np.pi)
    assert result.best_probability > 1.0 - 1e-12
    assert result.probability_error < 1e-8
    assert result.state_relative_error < 1e-8
    assert report["resource_proxy"]["proxy_kind"] == (
        "quantum-walk-search-resource-proxy"
    )
    assert report["resource_proxy"]["requires_marking_oracle"] is True
    assert "QSP_or_QSVT_phase_synthesis" in report["resource_proxy"]["omitted_layers"]


def test_quantum_walk_search_workflow_validates_inputs():
    with pytest.raises(ValueError, match="Hermitian"):
        quantum_walk_search_workflow(
            np.array([[0.0, 1.0], [0.0, 0.0]]),
            marked_vertex=0,
            degree=4,
        )

    with pytest.raises(ValueError, match="valid vertex"):
        quantum_walk_search_workflow(
            np.ones((3, 3)) - np.eye(3),
            marked_vertex=3,
            degree=4,
        )


def test_write_linear_system_comparison_csv(tmp_path):
    comparison = linear_system_comparison_workflow(
        np.array([[2.0, 0.25], [0.25, 1.25]]),
        np.array([1.0, -0.5]),
        degree=8,
        num_points=201,
        bounded_num_points=401,
        attempt_synthesis=False,
        apply_qsvt=False,
    )
    path = tmp_path / "linear-system-comparison.csv"

    written = write_linear_system_comparison_csv(comparison, path)

    assert written == path
    text = path.read_text(encoding="utf-8")
    assert "solver,implementation_kind" in text
    assert "qsvt_style_polynomial_inverse" in text


def test_ground_state_filtering_workflow_regression():
    result = ground_state_filtering_workflow(
        HERMitian_2X2,
        STATE_2,
        degree=8,
        width=0.45,
        num_points=201,
    )
    report = _validated_algorithm_report(result)

    assert report["mode"] == "ground-state-filtering-workflow"
    assert report["implementation_kind"] == "dense-spectral-polynomial-workflow"
    assert report["truth_contract"]["implementation_kind"] == (
        "dense-spectral-polynomial-workflow"
    )
    assert (
        "block_encoding_construction"
        in report["truth_contract"]["assumed_quantum_components"]
    )
    assert result.reference_state_error < 1e-5
    assert result.operator_relative_error < 1e-3
    assert result.ground_state_overlap > 0.999999


def test_hamiltonian_simulation_workflow_regression():
    result = hamiltonian_simulation_workflow(
        HERMitian_2X2,
        STATE_2,
        time=0.4,
        degree=9,
        num_points=201,
    )
    report = _validated_algorithm_report(result)

    assert result.state_relative_error < 1e-12
    assert result.operator_relative_error < 1e-12
    assert result.norm_drift < 1e-12
    acceptance = report["acceptance"]
    assert acceptance["scope"] == "polynomial_core"
    assert acceptance["accepted_for_stated_scope"] is True
    assert acceptance["full_qsvt_acceptance"] is False


def test_resolvent_workflow_regression():
    result = resolvent_workflow(
        HERMitian_2X2,
        omega=0.4,
        eta=0.3,
        degree=10,
        source=STATE_2,
        num_points=301,
    )
    _validated_algorithm_report(result)

    assert result.operator_relative_error < 0.013
    assert result.response_relative_error is not None
    assert result.response_relative_error < 0.013


def test_spectral_density_workflow_regression():
    result = spectral_density_workflow(
        HERMitian_2X2,
        np.array([-0.1, 0.5, 1.1]),
        width=0.35,
        degree=8,
        state=STATE_2,
        num_points=201,
    )
    _validated_algorithm_report(result)

    assert result.trace_density_error < 1e-4
    assert result.state_weight_error is not None
    assert result.state_weight_error < 2e-4


def test_spectral_thresholding_workflow_regression():
    matrix = np.diag([-0.8, -0.15, 0.2, 0.75])
    state = np.array([0.1, 0.8, 0.5, 0.2])

    result = spectral_thresholding_workflow(
        matrix,
        lower=-0.3,
        upper=0.3,
        degree=36,
        sharpness=20.0,
        state=state,
        num_points=401,
        bounded_num_points=801,
    )
    report = _validated_algorithm_report(result)

    assert report["mode"] == "spectral-thresholding-workflow"
    assert report["truth_contract"]["target"] == (
        "smooth spectral interval-projector approximation"
    )
    assert result.exact_rank == 2
    assert abs(result.polynomial_rank_proxy - result.exact_rank) < 0.15
    assert result.operator_relative_error < 0.12
    assert result.leakage_outside_interval < 0.08
    assert result.state_weight_error is not None
    assert result.state_weight_error < 0.08


def test_thermal_gibbs_workflow_regression():
    result = thermal_gibbs_workflow(
        HERMitian_2X2,
        beta=0.7,
        degree=8,
        state=STATE_2,
        num_points=201,
    )
    _validated_algorithm_report(result)

    assert result.operator_relative_error < 1e-10
    assert result.density_matrix_relative_error < 1e-10
    assert result.weighted_state_error is not None
    assert result.weighted_state_error < 1e-10
