import numpy as np
import pytest

from qsvt.reports import report_to_jsonable
from qsvt.synthesis import (
    BoundednessCertificate,
    MixedParitySynthesisResult,
    PhaseSolverBenchmarkResult,
    PhaseSolverStressResult,
    PhaseSynthesisResult,
    benchmark_phase_solver_stress_matrix,
    benchmark_phase_solvers,
    certify_polynomial_boundedness,
    classify_polynomial_realizability,
    parity_components,
    synthesize,
    synthesize_mixed_parity,
    synthesize_phases,
)
from qsvt.workflow import design_workflow


def test_realizability_distinguishes_single_and_multiple_parity_sequences():
    odd = classify_polynomial_realizability([0.0, 1.0])
    mixed = classify_polynomial_realizability([0.5, 0.5])

    assert odd.kind == "single-sequence-qsp-qsvt"
    assert odd.single_sequence_realizable is True
    assert odd.requires_parity_decomposition is False

    assert mixed.kind == "multiple-parity-sequences-or-lcu"
    assert mixed.single_sequence_realizable is False
    assert mixed.requires_parity_decomposition is True
    assert np.allclose(mixed.even_coeffs, [0.5, 0.0])
    assert np.allclose(mixed.odd_coeffs, [0.0, 0.5])


def test_realizability_distinguishes_unbounded_classical_polynomial():
    result = classify_polynomial_realizability([0.0, 2.0])

    assert result.kind == "classical-polynomial-only"
    assert result.bounded is False
    assert "out_of_bounds" in result.reasons


def test_extrema_certificate_detects_peak_missed_by_endpoint_grid():
    coeffs = [0.996, 0.1, -0.5]
    endpoint_values = np.polynomial.polynomial.polyval([-1.0, 1.0], coeffs)

    certificate = certify_polynomial_boundedness(coeffs)

    assert isinstance(certificate, BoundednessCertificate)
    assert np.max(np.abs(endpoint_values)) < 1.0
    assert certificate.maximizing_point == pytest.approx(0.1)
    assert certificate.max_abs_value == pytest.approx(1.001)
    assert certificate.is_bounded is False
    assert certificate.derivative_root_residual < 1e-12


def test_extrema_certificate_ignores_subnormal_trailing_root_noise():
    coeffs = [0.0, 1.0, 0.0, -1.0, np.finfo(np.float32).tiny]

    certificate = certify_polynomial_boundedness(coeffs)

    assert certificate.max_abs_value == pytest.approx(2.0 / (3.0 * np.sqrt(3.0)))
    assert np.any(np.isclose(certificate.critical_points, 1.0 / np.sqrt(3.0)))


def test_compatibility_realizability_uses_extrema_certificate():
    result = classify_polynomial_realizability(
        [0.996, 0.1, -0.5],
        bounded_num_points=2,
    )

    assert result.bounded is False
    assert result.boundedness_certificate is not None
    assert result.max_abs_value == pytest.approx(1.001)


def test_parity_components_preserve_polynomial():
    even, odd = parity_components([1.0, 2.0, 3.0, 4.0])

    assert np.allclose(even, [1.0, 0.0, 3.0, 0.0])
    assert np.allclose(odd, [0.0, 2.0, 0.0, 4.0])
    assert np.allclose(even + odd, [1.0, 2.0, 3.0, 4.0])


def test_qsvt_phase_synthesis_returns_angles_and_reconstruction_error():
    result = synthesize_phases(
        [0.0, 1.0, 0.0, -0.5, 0.0, 1.0 / 3.0],
        reconstruction_num_points=33,
    )
    report = report_to_jsonable(result.as_report())

    assert isinstance(result, PhaseSynthesisResult)
    assert result.succeeded is True
    assert result.angles is not None
    assert result.angles.size == 6
    assert result.reconstruction_max_error is not None
    assert result.reconstruction_max_error < 1e-9
    assert report["routine"] == "QSVT"
    assert report["phase_count"] == 6


def test_synthesis_returns_actionable_mixed_parity_failure():
    result = synthesize([0.5, 0.5])

    assert result.succeeded is False
    assert result.error_type == "PolynomialRealizabilityError"
    assert result.realizability.requires_parity_decomposition is True
    assert "mixed parity" in result.error
    assert result.quality_report()["status"] == "solver_failed"

    with pytest.raises(ValueError, match="mixed parity"):
        synthesize([0.5, 0.5], raise_on_failure=True)


def test_design_workflow_can_synthesize_its_polynomial():
    design = design_workflow(
        "sign",
        degree=5,
        gamma=0.25,
        num_points=101,
        bounded_num_points=201,
        attempt_synthesis=False,
    )
    synthesis = design.synthesize(reconstruction_num_points=17)

    assert isinstance(synthesis, PhaseSynthesisResult)
    assert np.allclose(synthesis.coeffs, design.coeffs)
    assert synthesis.realizability.parity == "odd"


def test_phase_solver_benchmark_reports_convergence_timing_and_conditioning():
    benchmark = benchmark_phase_solvers(
        [0.0, 1.0],
        solvers=["root-finding", "unsupported-solver"],
        repeats=1,
        reconstruction_num_points=9,
    )
    report = benchmark.as_report()

    assert isinstance(benchmark, PhaseSolverBenchmarkResult)
    assert report["mode"] == "phase-solver-benchmark"
    assert report["conditioning_proxies"]["degree"] == 1
    assert report["rows"][0]["converged"] is True
    assert report["rows"][0]["max_reconstruction_error"] < 1e-9
    assert report["rows"][1]["converged"] is False
    assert report["rows"][1]["error_types"] == ["ValueError"]


def test_phase_solver_stress_matrix_compares_conditioning_regimes():
    stress = benchmark_phase_solver_stress_matrix(
        {
            "linear-margin": [0.0, 0.5],
            "quintic-near-boundary": [0.0, 0.0, 0.0, 0.0, 0.0, 0.95],
        },
        solvers=["root-finding"],
        repeats=1,
        reconstruction_num_points=17,
        reconstruction_tolerance=1e-3,
    )
    report = stress.as_report()

    assert isinstance(stress, PhaseSolverStressResult)
    assert report["mode"] == "phase-solver-stress-matrix"
    assert report["summary"] == {
        "case_count": 2,
        "row_count": 2,
        "total_attempts": 2,
        "total_successes": 2,
        "all_converged": True,
    }
    assert [row["degree"] for row in report["rows"]] == [1, 5]
    assert all(row["max_reconstruction_error"] < 1e-9 for row in report["rows"])
    for row in report["rows"]:
        assert row["reconstruction_tolerance"] == 1e-3
        assert row["validated_successes"] == 1
        assert row["all_reconstructions_passed"]
    assert report["truth_contract"]["is_hardware_runtime"] is False


def test_phase_solver_stress_matrix_requires_named_cases():
    with pytest.raises(ValueError, match="at least one named polynomial"):
        benchmark_phase_solver_stress_matrix({})


def test_mixed_parity_synthesis_reports_components_and_lcu_proxy():
    result = synthesize_mixed_parity(
        [0.5, 0.5],
        reconstruction_num_points=17,
    )
    report = result.as_report()

    assert isinstance(result, MixedParitySynthesisResult)
    assert result.succeeded is True
    assert result.even_synthesis is not None
    assert result.even_synthesis.angle_solver == "analytic-constant"
    assert result.even_synthesis.quality_report()["reconstruction_passed"]
    assert report["component_resource_proxy"]["even_phase_count"] == 1
    assert result.odd_synthesis is not None
    assert result.odd_synthesis.succeeded is True
    assert result.lcu_normalization == pytest.approx(1.0)
    assert result.postselection_probability_proxy == pytest.approx(1.0)
    assert result.reconstruction_max_error is not None
    assert result.reconstruction_max_error < 1e-9
    assert report["component_resource_proxy"]["sequence_count"] == 2
    assert report["component_resource_proxy"]["total_signal_operator_calls"] == 1
    assert report["truth_contract"]["lcu_circuit_implemented"] is False


def test_synthesis_quality_does_not_confuse_returned_phases_with_accuracy():
    from dataclasses import replace

    result = synthesize_phases([0.0, 0.5], reconstruction_num_points=33)
    assert result.quality_report(1e-6)["status"] == "passed"
    inaccurate = replace(result, reconstruction_max_error=0.075)
    assert inaccurate.succeeded is True
    quality = inaccurate.quality_report(1e-6)
    assert quality["solver_returned_phases"] is True
    assert quality["reconstruction_passed"] is False
    assert quality["status"] == "reconstruction_failed"
    assert (
        replace(result, reconstruction_max_error=None).quality_report()["status"]
        == "reconstruction_unavailable"
    )
    assert (
        replace(result, reconstruction_max_error=float("nan")).quality_report()[
            "reconstruction_passed"
        ]
        is False
    )
    assert (
        replace(result, angles=np.array([float("nan")])).quality_report()["status"]
        == "solver_failed"
    )
    for tolerance in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            result.quality_report(tolerance)


@pytest.mark.parametrize(
    "kind,degree",
    [
        ("sign", 13),
        ("inverse", 13),
        ("filter", 10),
    ],
)
def test_iterative_synthesis_reconstructs_studio_boundary_polynomials(kind, degree):
    # These unchanged polynomials expose root-finding failures or poor
    # reconstruction on supported PennyLane versions. Do not rescale them.
    result = design_workflow(
        kind,
        degree=degree,
        num_points=401,
        attempt_synthesis=False,
    )
    synthesis = result.synthesize(angle_solver="iterative")
    np.testing.assert_array_equal(synthesis.coeffs, result.coeffs)
    quality = synthesis.quality_report(1e-6)
    assert quality["reconstruction_passed"] is True, quality


@pytest.mark.integration
@pytest.mark.parametrize(
    "kind,degree",
    [("sign", 25), ("inverse", 25), ("filter", 24)],
)
def test_iterative_high_degree_stress_cases_are_accurate_or_structured(kind, degree):
    # PennyLane's optional iterative backend is environment-sensitive at these
    # degrees. Preserve the input and require an auditable result either way.
    result = design_workflow(
        kind,
        degree=degree,
        num_points=401,
        attempt_synthesis=False,
    )
    synthesis = result.synthesize(angle_solver="iterative")
    np.testing.assert_array_equal(synthesis.coeffs, result.coeffs)
    quality = synthesis.quality_report(1e-4)
    if synthesis.succeeded:
        assert quality["reconstruction_passed"] is True, quality
    else:
        assert quality["status"] == "solver_failed"
        assert synthesis.angles is None
        assert synthesis.error_type
        assert synthesis.error


@pytest.mark.parametrize("constant", [-1.0, -0.3, 0.0, 0.7, 1.0])
def test_constant_qsvt_has_one_projector_and_no_signal_queries(constant):
    import pennylane as qml

    result = synthesize_phases([constant, 0.0, 0.0], reconstruction_num_points=5)
    assert result.succeeded
    assert result.angles.size == 1
    assert result.angle_solver == "analytic-constant"
    assert result.implementation_kind == "analytic-constant-projector-phase"
    assert result.quality_report(1e-14)["reconstruction_passed"]
    for x in (-1.0, -0.2, 0.0, 0.8, 1.0):
        operator = qml.QSVT(
            qml.RX(2 * np.arccos(x), wires=0),
            [qml.PCPhase(result.angles[0], dim=1, wires=0)],
        )
        assert qml.matrix(operator)[0, 0].real == pytest.approx(constant)
        assert len(operator.decomposition()) == 1


@pytest.mark.parametrize("angles", [[], [np.nan], [np.inf], [[0.1]], [0.2j]])
def test_invalid_backend_phases_become_structured_failures(monkeypatch, angles):
    monkeypatch.setattr("qsvt.synthesis.qml.poly_to_angles", lambda *a, **k: angles)
    result = synthesize_phases([0.0, 0.5])
    assert not result.succeeded
    assert result.angles is None
    assert result.error_type == "ValueError"
    assert result.reconstruction_max_error is None
    with pytest.raises(ValueError, match="angles"):
        synthesize_phases([0.0, 0.5], raise_on_failure=True)


@pytest.mark.parametrize("points", [True, 1, 2.5, np.nan, np.inf])
def test_reconstruction_grid_validation_precedes_cache_and_adapter(points):
    from qsvt.synthesis import synthesize_phases_cached, synthesize_phases_with_adapter

    for run in (synthesize_phases, synthesize_phases_cached, synthesize_mixed_parity):
        with pytest.raises(ValueError, match="integer"):
            run([0.0, 0.5], reconstruction_num_points=points)
    with pytest.raises(ValueError, match="integer"):
        synthesize_phases_with_adapter(
            [0.0, 0.5], adapter="missing", reconstruction_num_points=points
        )


def test_benchmark_distinguishes_inaccurate_phases_from_solver_completion(monkeypatch):
    monkeypatch.setattr(
        "qsvt.synthesis.qml.poly_to_angles", lambda *a, **k: np.array([0.0, 0.0])
    )
    row = benchmark_phase_solvers([0.0, 0.5], solvers=["root-finding"], repeats=1).rows[
        0
    ]
    assert row["converged"] is True
    assert row["successes"] == 1
    assert row["validated_successes"] == 0
    assert row["all_reconstructions_passed"] is False
    assert row["max_reconstruction_error"] > row["reconstruction_tolerance"]


def test_benchmark_qsp_without_reconstruction_is_unvalidated():
    row = benchmark_phase_solvers(
        [0.0, 0.5], routine="QSP", solvers=["root-finding"], repeats=1
    ).rows[0]
    assert row["converged"] is True
    assert row["validated_successes"] == 0
    assert row["all_reconstructions_passed"] is False


@pytest.mark.parametrize("tolerance", [-1, np.nan, np.inf])
def test_benchmark_rejects_invalid_reconstruction_tolerance(tolerance):
    with pytest.raises(ValueError, match="reconstruction_tolerance"):
        benchmark_phase_solver_stress_matrix(
            {"linear": [0.0, 0.5]}, reconstruction_tolerance=tolerance
        )


@pytest.mark.parametrize(
    "degree",
    [
        12,
        pytest.param(24, marks=pytest.mark.integration),
    ],
)
def test_iterative_hamiltonian_sine_reconstruction_preserves_coefficients(degree):
    from qsvt.matrix_functions import design_real_time_evolution_polynomials

    polynomial = design_real_time_evolution_polynomials(
        1.4, 1.0, degree=degree, num_points=401
    ).sin_coeffs
    result = synthesize_phases(polynomial, angle_solver="iterative")
    np.testing.assert_array_equal(result.coeffs, polynomial)
    assert result.quality_report(1e-6)["reconstruction_passed"]


@pytest.mark.parametrize("angles", [[], [np.nan], [np.inf], [[0.1]], [0.2j]])
def test_adapter_rejects_invalid_raw_and_converted_phases(angles):
    from qsvt.synthesis import (
        register_phase_solver_adapter,
        synthesize_phases_with_adapter,
        unregister_phase_solver_adapter,
    )

    for converted in (False, True):
        register_phase_solver_adapter(
            "invalid-regression",
            lambda *a, converted=converted, **k: [0.1, 0.2] if converted else angles,
            convention="test convention" if converted else "pennylane-qsvt-projector",
            converter=(lambda *a: angles) if converted else None,
        )
        try:
            result = synthesize_phases_with_adapter(
                [0.0, 0.5], adapter="invalid-regression"
            )
            assert not result.succeeded
            assert result.angles is None
            assert result.error_type == "ValueError"
        finally:
            unregister_phase_solver_adapter("invalid-regression")
