import json

import numpy as np
import pytest

import qsvt
from qsvt.benchmarks import (
    benchmark_environment_report,
    benchmark_summary_table,
    conjugate_gradient_benchmark,
    conjugate_gradient_solve,
    dense_eigendecomposition_benchmark,
    dense_linear_solve_benchmark,
    plot_benchmark_timings,
    plot_qsvt_proxy_resources,
    polynomial_matrix_function_benchmark,
    spectral_matrix_function_benchmark,
    write_benchmark_summary_csv,
)


def _benchmark_report(
    *,
    algorithm="numpy.linalg.solve",
    problem="dense-linear-system",
    polynomial_degree=None,
    signal_operator_calls=1,
    output_frobenius_norm=None,
):
    metrics = {
        "relative_residual_norm": 1e-14,
        "condition_number_2": 1.5,
    }
    if polynomial_degree is not None:
        metrics["polynomial_degree"] = polynomial_degree
    if output_frobenius_norm is not None:
        metrics["output_frobenius_norm"] = output_frobenius_norm

    return {
        "mode": "classical-benchmark",
        "algorithm": algorithm,
        "problem": problem,
        "matrix_dimension": 2,
        "repeats": 1,
        "best_time_seconds": 0.001,
        "mean_time_seconds": 0.0012,
        "metrics": metrics,
        "qsvt_proxy": {
            "resources": {
                "degree": signal_operator_calls,
                "signal_operator_calls": signal_operator_calls,
            },
        },
        "notes": [],
    }


def test_dense_eigendecomposition_benchmark_reports_reconstruction_error():
    matrix = np.array([[2.0, 0.25], [0.25, 3.0]])

    report = dense_eigendecomposition_benchmark(matrix, repeats=1)

    assert report["mode"] == "classical-benchmark"
    assert report["truth_contract"]["truth_status"] == "classical_timing_reference"
    assert report["truth_contract"]["is_quantum_runtime_benchmark"] is False
    assert report["benchmark_environment"]["timing_kind"] == (
        "python_wall_clock_microbenchmark"
    )
    assert report["benchmark_environment"]["numpy_version"]
    assert report["algorithm"] == "numpy.linalg.eigh"
    assert report["matrix_dimension"] == 2
    assert report["repeats"] == 1
    assert report["best_time_seconds"] >= 0.0
    assert report["metrics"]["reconstruction_relative_error"] < 1e-12
    json.dumps(qsvt.report_to_jsonable(report))


def test_dense_linear_solve_benchmark_reports_residual_and_qsvt_proxy():
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.array([1.0, 2.0])

    report = dense_linear_solve_benchmark(
        matrix,
        rhs,
        repeats=1,
        qsvt_coeffs=[0.0, 1.0],
    )

    assert report["problem"] == "dense-linear-system"
    assert report["metrics"]["relative_residual_norm"] < 1e-12
    assert report["metrics"]["condition_number_2"] > 1.0
    assert report["qsvt_proxy"]["resources"]["degree"] == 1
    assert report["qsvt_proxy"]["resources"]["matrix_dimension"] == 2


def test_conjugate_gradient_solve_matches_dense_solution():
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.array([1.0, 2.0])

    result = conjugate_gradient_solve(matrix, rhs, tolerance=1e-12)

    assert result["converged"] is True
    assert result["relative_residual_norm"] < 1e-12
    np.testing.assert_allclose(
        result["solution"],
        np.linalg.solve(matrix, rhs),
        atol=1e-12,
    )


def test_conjugate_gradient_benchmark_reports_iterations():
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.array([1.0, 2.0])

    report = conjugate_gradient_benchmark(
        matrix,
        rhs,
        tolerance=1e-12,
        repeats=1,
        qsvt_coeffs=[0.0, 1.0],
    )

    assert report["problem"] == "positive-definite-linear-system"
    assert report["metrics"]["converged"] is True
    assert report["metrics"]["iterations"] <= 2
    assert report["metrics"]["matvec_count"] == report["metrics"]["iterations"]
    assert report["qsvt_proxy"]["resources"]["signal_operator_calls"] == 1


def test_polynomial_matrix_function_benchmark_reports_degree():
    matrix = np.array([[0.5, 0.0], [0.0, -0.25]])

    report = polynomial_matrix_function_benchmark(
        matrix,
        [0.0, 0.0, 1.0],
        repeats=1,
    )

    assert report["problem"] == "polynomial-matrix-function"
    assert report["metrics"]["polynomial_degree"] == 2
    assert report["metrics"]["output_frobenius_norm"] > 0.0
    assert report["qsvt_proxy"]["resources"]["qsp_phase_count"] == 3


def test_spectral_matrix_function_benchmark_supports_exponential():
    matrix = np.array([[0.0, 0.0], [0.0, 1.0]])

    report = spectral_matrix_function_benchmark(
        matrix,
        "exponential",
        beta=0.5,
        repeats=1,
    )

    assert report["problem"] == "exponential-matrix-function"
    assert report["metrics"]["function"] == "exponential"
    assert report["metrics"]["output_frobenius_norm"] > 1.0


def test_benchmark_summary_table_extracts_common_fields():
    reports = [_benchmark_report()]

    rows = benchmark_summary_table(reports)

    assert rows == [
        {
            "algorithm": "numpy.linalg.solve",
            "problem": "dense-linear-system",
            "matrix_dimension": 2,
            "best_time_seconds": reports[0]["best_time_seconds"],
            "mean_time_seconds": reports[0]["mean_time_seconds"],
            "relative_residual_norm": reports[0]["metrics"]["relative_residual_norm"],
            "relative_error": None,
            "condition_number_2": reports[0]["metrics"]["condition_number_2"],
            "polynomial_degree": 1,
            "qsvt_signal_operator_calls": 1,
        }
    ]


def test_write_benchmark_summary_csv_writes_compact_rows(tmp_path):
    reports = [_benchmark_report()]
    path = tmp_path / "summary.csv"

    written = write_benchmark_summary_csv(reports, path)
    lines = written.read_text(encoding="utf-8").splitlines()

    assert written == path
    assert lines[0] == (
        "algorithm,problem,matrix_dimension,best_time_seconds,mean_time_seconds,"
        "relative_residual_norm,relative_error,condition_number_2,"
        "polynomial_degree,qsvt_signal_operator_calls"
    )
    assert lines[1].startswith("numpy.linalg.solve,dense-linear-system,2,")


def test_benchmark_plot_helpers_return_matplotlib_axes(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports = [_benchmark_report()]

    timing_fig, timing_ax = plot_benchmark_timings(reports)
    proxy_fig, proxy_ax = plot_qsvt_proxy_resources(reports)
    output = tmp_path / "benchmark-timings.png"
    timing_fig.savefig(output)

    assert timing_ax.get_xlabel() == "benchmark case"
    assert timing_ax.get_ylabel() == "best time (ms)"
    assert timing_ax.get_legend() is not None
    assert proxy_ax.get_ylabel() == "signal-operator calls"
    assert proxy_ax.get_legend() is not None
    assert output.exists()
    plt.close(timing_fig)
    plt.close(proxy_fig)


def test_benchmark_plot_colors_remain_stable_across_filtered_proxy_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports = [
        _benchmark_report(
            algorithm="dense-spectral-matrix-function",
            problem="exponential-matrix-function",
            signal_operator_calls=None,
            output_frobenius_norm=1.4,
        ),
        _benchmark_report(
            algorithm="spectral-polynomial-evaluation",
            problem="polynomial-matrix-function",
            polynomial_degree=2,
            signal_operator_calls=2,
            output_frobenius_norm=0.25,
        ),
    ]

    timing_fig, timing_ax = plot_benchmark_timings(reports)
    proxy_fig, proxy_ax = plot_qsvt_proxy_resources(reports)

    timing_poly_color = timing_ax.patches[1].get_facecolor()
    proxy_poly_color = proxy_ax.patches[0].get_facecolor()

    assert proxy_poly_color == timing_poly_color
    plt.close(timing_fig)
    plt.close(proxy_fig)


def test_benchmarks_reject_invalid_inputs():
    with pytest.raises(ValueError, match="Hermitian"):
        dense_eigendecomposition_benchmark([[0.0, 1.0], [0.0, 0.0]])

    with pytest.raises(ValueError, match="matching matrix dimension"):
        dense_linear_solve_benchmark([[1.0, 0.0], [0.0, 1.0]], [1.0])

    with pytest.raises(ValueError, match="tolerance must be positive"):
        conjugate_gradient_solve([[1.0]], [1.0], tolerance=0.0)


def test_top_level_exports_benchmark_helpers():
    assert qsvt.ClassicalBenchmarkResult is not None
    assert qsvt.benchmark_environment_report is benchmark_environment_report
    assert qsvt.dense_linear_solve_benchmark is dense_linear_solve_benchmark
    assert qsvt.conjugate_gradient_solve is conjugate_gradient_solve
    assert qsvt.plot_benchmark_timings is plot_benchmark_timings
    assert qsvt.plot_qsvt_proxy_resources is plot_qsvt_proxy_resources
    assert qsvt.write_benchmark_summary_csv is write_benchmark_summary_csv
    assert "dense_linear_solve_benchmark" in qsvt.__all__
