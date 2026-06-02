import csv
import json
from pathlib import Path

BENCHMARK_JSON = [
    Path("results/benchmarks/linear_system_dense_solve.json"),
    Path("results/benchmarks/linear_system_cg_solve.json"),
    Path("results/benchmarks/matrix_function_exponential_spectral.json"),
    Path("results/benchmarks/matrix_function_thermal_polynomial.json"),
    Path("results/benchmarks/matrix_function_filter_polynomial.json"),
    Path("results/benchmarks/scaling_sweep_reports.json"),
]

BENCHMARK_CSV = [
    Path("results/tables/linear_system_benchmark_summary.csv"),
    Path("results/tables/matrix_function_benchmark_summary.csv"),
    Path("results/tables/benchmark_scaling_summary.csv"),
]


def test_committed_benchmark_json_artifacts_are_well_formed():
    for path in BENCHMARK_JSON:
        assert path.exists(), path
        payload = json.loads(path.read_text(encoding="utf-8"))
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
        assert required.issubset(rows[0])
