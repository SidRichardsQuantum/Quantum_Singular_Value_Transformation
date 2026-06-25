import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_example(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    pythonpath = str(REPO_ROOT / "src")
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath

    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "examples" / script), *args],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )


def test_design_apply_report_example_writes_json(tmp_path):
    output = tmp_path / "design-apply.json"

    completed = run_example("design_apply_report.py", "--output", str(output))
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert str(output) in completed.stdout
    assert payload["example"] == "design-apply-report"
    assert payload["mode"] == "design-workflow"
    assert payload["kind"] == "sign"
    assert payload["compatibility"]["attempted_pennylane_synthesis"] is False
    assert len(payload["transformed_diagonal"]) == 4


@pytest.mark.integration
def test_linear_system_compare_example_writes_json_and_csv(tmp_path):
    output = tmp_path / "linear-system.json"
    rows_output = tmp_path / "linear-system.csv"

    completed = run_example(
        "linear_system_compare.py",
        "--output",
        str(output),
        "--rows-output",
        str(rows_output),
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(rows_output.read_text(encoding="utf-8").splitlines()))
    solvers = {row["solver"] for row in rows}

    assert str(output) in completed.stdout
    assert str(rows_output) in completed.stdout
    assert payload["example"] == "linear-system-compare"
    assert payload["mode"] == "linear-system-comparison-workflow"
    assert "dense_solve" in solvers
    assert "conjugate_gradient" in solvers
    assert "qsvt_style_polynomial_inverse" in solvers


@pytest.mark.integration
def test_threshold_filter_example_writes_json(tmp_path):
    output = tmp_path / "threshold-filter.json"

    completed = run_example("threshold_filter.py", "--output", str(output))
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert str(output) in completed.stdout
    assert payload["example"] == "threshold-filter"
    assert payload["mode"] == "spectral-thresholding-workflow"
    assert payload["exact_rank"] == 2
    assert payload["state_weight_error"] is not None


@pytest.mark.integration
def test_block_encoded_workflow_example_writes_json(tmp_path):
    output = tmp_path / "block-encoded-workflow.json"

    completed = run_example("block_encoded_workflow.py", "--output", str(output))
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert str(output) in completed.stdout
    assert payload["example"] == "block-encoded-workflow"
    assert payload["mode"] == "block-encoded-qsvt-workflow"
    assert payload["verification"]["block_encoding_verified"] is True
    assert payload["operator_relative_error"] is not None


@pytest.mark.integration
def test_circuit_execution_example_writes_json(tmp_path):
    output = tmp_path / "circuit-execution.json"

    completed = run_example("circuit_execution.py", "--output", str(output))
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert str(output) in completed.stdout
    assert payload["example"] == "circuit-execution"
    assert payload["mode"] == "qsvt-circuit-execution-report"
    assert payload["implementation_kind"] == (
        "pennylane-qnode-statevector-qsvt-execution"
    )
    assert payload["is_end_to_end_quantum_algorithm"] is False


@pytest.mark.integration
def test_block_encoding_execution_example_writes_json(tmp_path):
    output = tmp_path / "block-encoding-execution.json"

    completed = run_example(
        "block_encoding_execution.py",
        "--output",
        str(output),
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert str(output) in completed.stdout
    assert payload["example"] == "block-encoding-execution"
    assert payload["mode"] == "block-encoding-qsvt-execution-report"
    assert payload["succeeded"] is True
    assert payload["block_encoding_spec"]["kind"] == "pennylane-operator"
    assert payload["resource_summary"]["block_encoding_method"] == "prepselprep"


@pytest.mark.integration
def test_rectangular_execution_example_writes_json(tmp_path):
    output = tmp_path / "rectangular-execution.json"

    completed = run_example(
        "rectangular_execution.py",
        "--output",
        str(output),
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert str(output) in completed.stdout
    assert payload["example"] == "rectangular-execution"
    assert payload["succeeded"] is True
    assert payload["block_encoding_spec"]["logical_shape"] == [2, 3]
    assert payload["logical_output_relative_error"] < 1e-9


@pytest.mark.integration
def test_compatibility_report_example_writes_json(tmp_path):
    output = tmp_path / "compatibility-report.json"

    completed = run_example("compatibility_report.py", "--output", str(output))
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert str(output) in completed.stdout
    assert payload["example"] == "compatibility-report"
    assert payload["mode"] == "qsvt-compatibility-report"
    assert payload["compatible"] is False
    assert "mixed_parity" in payload["reasons"]


@pytest.mark.integration
def test_benchmark_summary_example_writes_json_and_csv(tmp_path):
    output = tmp_path / "benchmark-summary.json"
    rows_output = tmp_path / "benchmark-summary.csv"

    completed = run_example(
        "benchmark_summary.py",
        "--output",
        str(output),
        "--rows-output",
        str(rows_output),
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(rows_output.read_text(encoding="utf-8").splitlines()))
    algorithms = {row["algorithm"] for row in rows}

    assert str(output) in completed.stdout
    assert str(rows_output) in completed.stdout
    assert payload["example"] == "benchmark-summary"
    assert payload["mode"] == "benchmark-summary-bundle"
    assert "numpy.linalg.solve" in algorithms
    assert "qsvt.benchmarks.conjugate_gradient_solve" in algorithms
