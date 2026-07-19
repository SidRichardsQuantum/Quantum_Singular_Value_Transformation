import csv
import json
from pathlib import Path

from qsvt.reports import validate_report_schema

BENCHMARK_JSON = [
    Path("results/benchmarks/linear_system_dense_solve.json"),
    Path("results/benchmarks/linear_system_cg_solve.json"),
    Path("results/benchmarks/matrix_function_exponential_spectral.json"),
    Path("results/benchmarks/matrix_function_thermal_polynomial.json"),
    Path("results/benchmarks/matrix_function_filter_polynomial.json"),
    Path("results/benchmarks/scaling_sweep_reports.json"),
    Path("results/benchmarks/quantum_walk_search_scaling.json"),
    Path("results/benchmarks/encoding_aware_resource_sweep.json"),
]

BENCHMARK_CSV = [
    Path("results/tables/linear_system_benchmark_summary.csv"),
    Path("results/tables/matrix_function_benchmark_summary.csv"),
    Path("results/tables/benchmark_scaling_summary.csv"),
    Path("results/tables/quantum_walk_search_scaling_summary.csv"),
    Path("results/tables/encoding_aware_resource_summary.csv"),
]

ALGORITHM_JSON = [
    Path("results/algorithms/linear_system_comparison.json"),
    Path("results/algorithms/linear_system_quantum_classical_comparison.json"),
]

ALGORITHM_CSV = [
    Path("results/tables/linear_system_comparison_summary.csv"),
    Path("results/tables/linear_system_quantum_classical_summary.csv"),
]


def test_committed_benchmark_json_artifacts_are_well_formed():
    for path in BENCHMARK_JSON:
        assert path.exists(), path
        payload = json.loads(path.read_text(encoding="utf-8"))
        if path.name == "encoding_aware_resource_sweep.json":
            assert payload["mode"] == "encoding-aware-resource-sweep"
            assert payload["truth_contract"]["truth_status"] == (
                "logical_resource_comparison"
            )
            assert payload["truth_contract"]["is_quantum_runtime_benchmark"] is False
            assert payload["truth_contract"]["is_fault_tolerant_estimate"] is False
            assert payload["rows"]
            assert payload["reports"]
            assert all(row["total_gates"] > 0 for row in payload["rows"])
            assert all(
                report["mode"] == "encoding-aware-qsvt-resource-report"
                for reports in payload["reports"].values()
                for report in reports
            )
            continue
        if path.name == "quantum_walk_search_scaling.json":
            assert payload["mode"] == "quantum-walk-search-scaling-benchmark"
            assert payload["truth_contract"]["truth_status"] == (
                "algorithm_comparison_benchmark"
            )
            assert payload["truth_contract"]["is_quantum_runtime_benchmark"] is False
            assert payload["rows"]
            assert payload["reports"]
            assert all(
                report["mode"] == "quantum-walk-search-workflow"
                for report in payload["reports"]
            )
            assert all(
                report["resource_proxy"]["proxy_kind"]
                == "quantum-walk-search-resource-proxy"
                for report in payload["reports"]
            )
            assert all(
                validate_report_schema(report, require_schema=True).supported
                for report in payload["reports"]
            )
            continue
        if path.name == "scaling_sweep_reports.json":
            assert payload["mode"] == "benchmark-scaling-sweep"
            assert len(payload["reports"]) >= 2
            assert all(
                report["mode"] == "classical-benchmark" for report in payload["reports"]
            )
            assert all(
                report["benchmark_environment"]["timing_kind"]
                == "python_wall_clock_microbenchmark"
                for report in payload["reports"]
            )
        else:
            assert payload["mode"] == "classical-benchmark"
            assert payload["truth_contract"]["truth_status"] == (
                "classical_timing_reference"
            )
            assert payload["truth_contract"]["is_quantum_runtime_benchmark"] is False
            assert payload["benchmark_environment"]["timing_kind"] == (
                "python_wall_clock_microbenchmark"
            )
            assert payload["problem"]
            assert payload["matrix_dimension"] > 0
            assert payload["best_time_seconds"] >= 0.0
        reports = (
            payload["reports"]
            if path.name == "scaling_sweep_reports.json"
            else [payload]
        )
        for report in reports:
            qsvt_proxy = report.get("qsvt_proxy")
            if qsvt_proxy is not None:
                assert qsvt_proxy["truth_contract"]["truth_status"] == "proxy_only"


def test_committed_benchmark_csv_artifacts_have_expected_columns():
    required = {
        "algorithm",
        "problem",
        "matrix_dimension",
        "best_time_seconds",
        "mean_time_seconds",
        "qsvt_signal_operator_calls",
    }
    for path in BENCHMARK_CSV:
        assert path.exists(), path
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        assert rows, path
        if path.name == "quantum_walk_search_scaling_summary.csv":
            assert {
                "n_vertices",
                "degree",
                "best_probability",
                "probability_error",
                "state_relative_error",
                "signal_call_proxy",
            }.issubset(rows[0])
        elif path.name == "encoding_aware_resource_summary.csv":
            assert {
                "access_model",
                "degree",
                "normalization_alpha",
                "signal_operator_calls",
                "inverse_signal_operator_calls",
                "total_wires",
                "total_gates",
                "estimator_model",
            }.issubset(rows[0])
        else:
            assert required.issubset(rows[0])


def test_committed_algorithm_json_artifacts_are_well_formed():
    for path in ALGORITHM_JSON:
        assert path.exists(), path
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert validate_report_schema(payload, require_schema=True).supported
        assert validate_report_schema(
            payload["linear_system_workflow"], require_schema=True
        ).supported
        assert payload["mode"] == "linear-system-comparison-workflow"
        assert payload["implementation_kind"] == "linear-system-solver-comparison"
        assert payload["resource_proxy"]["proxy_kind"] == (
            "linear-system-qsvt-style-resource-proxy"
        )
        assert payload["rows"]
        rows = {row["solver"]: row for row in payload["rows"]}
        if path.name == "linear_system_quantum_classical_comparison.json":
            assert rows["hhl_circuit_execution"]["status"] == "ok"
            assert rows["hhl_circuit_execution"]["is_executable_hhl_circuit"] is True
            assert rows["hhl_circuit_execution"]["fidelity"] > 1.0 - 1e-10


def test_committed_algorithm_csv_artifacts_have_expected_columns():
    required = {
        "solver",
        "implementation_kind",
        "matrix_dimension",
        "degree",
        "gamma",
        "condition_number_2",
        "residual_norm",
        "relative_solution_error",
    }
    for path in ALGORITHM_CSV:
        assert path.exists(), path
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        assert rows, path
        assert required.issubset(rows[0])
