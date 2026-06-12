import csv
import json
import os
import subprocess
import sys
from pathlib import Path

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


def test_threshold_filter_example_writes_json(tmp_path):
    output = tmp_path / "threshold-filter.json"

    completed = run_example("threshold_filter.py", "--output", str(output))
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert str(output) in completed.stdout
    assert payload["example"] == "threshold-filter"
    assert payload["mode"] == "spectral-thresholding-workflow"
    assert payload["exact_rank"] == 2
    assert payload["state_weight_error"] is not None
