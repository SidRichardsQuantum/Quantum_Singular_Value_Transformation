import numpy as np
import pytest

from qsvt.reports import report_to_jsonable
from qsvt.synthesis import (
    BoundednessCertificate,
    MixedParitySynthesisResult,
    PhaseSolverBenchmarkResult,
    PhaseSynthesisResult,
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
    assert result.odd_synthesis is not None
    assert result.odd_synthesis.succeeded is True
    assert result.lcu_normalization == pytest.approx(1.0)
    assert result.postselection_probability_proxy == pytest.approx(1.0)
    assert result.reconstruction_max_error is not None
    assert result.reconstruction_max_error < 1e-9
    assert report["component_resource_proxy"]["sequence_count"] == 2
    assert report["component_resource_proxy"]["total_signal_operator_calls"] == 1
    assert report["truth_contract"]["lcu_circuit_implemented"] is False
