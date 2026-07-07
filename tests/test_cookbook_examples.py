import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _example_environment():
    env = os.environ.copy()
    pythonpath = os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "src")])
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    return env


def _run_python(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        env=_example_environment(),
        check=True,
        text=True,
        capture_output=True,
    )


@pytest.mark.integration
def test_cookbook_examples_write_expected_artifacts(tmp_path):
    runner = """
from pathlib import Path
import runpy
import sys

from examples import (
    block_encoded_workflow,
    block_encoding_execution,
    circuit_execution,
    linear_system_compare,
    problem_workflow,
    rectangular_execution,
    threshold_filter,
)

output_dir = Path(sys.argv[1])
script = Path(sys.argv[2])
sys.argv = [
    str(script),
    "--output", str(output_dir / "design-apply.json"),
]
runpy.run_path(script, run_name="__main__")
linear_system_compare.main([
    "--output", str(output_dir / "linear-system.json"),
    "--rows-output", str(output_dir / "linear-system.csv"),
])
threshold_filter.main(["--output", str(output_dir / "threshold-filter.json")])
problem_workflow.main(["--output", str(output_dir / "problem-workflow.json")])
block_encoded_workflow.main([
    "--output", str(output_dir / "block-encoded-workflow.json"),
])
circuit_execution.main(["--output", str(output_dir / "circuit-execution.json")])
block_encoding_execution.main([
    "--output", str(output_dir / "block-encoding-execution.json"),
])
rectangular_execution.main([
    "--output", str(output_dir / "rectangular-execution.json"),
])
"""
    completed = _run_python(
        "-c",
        runner,
        str(tmp_path),
        str(REPO_ROOT / "examples" / "design_apply_report.py"),
    )

    design_apply = json.loads(
        (tmp_path / "design-apply.json").read_text(encoding="utf-8")
    )
    linear_system = json.loads(
        (tmp_path / "linear-system.json").read_text(encoding="utf-8")
    )
    linear_csv = (tmp_path / "linear-system.csv").read_text(encoding="utf-8")
    threshold = json.loads(
        (tmp_path / "threshold-filter.json").read_text(encoding="utf-8")
    )
    problem = json.loads(
        (tmp_path / "problem-workflow.json").read_text(encoding="utf-8")
    )
    block_encoded = json.loads(
        (tmp_path / "block-encoded-workflow.json").read_text(encoding="utf-8")
    )
    circuit = json.loads(
        (tmp_path / "circuit-execution.json").read_text(encoding="utf-8")
    )
    block_execution = json.loads(
        (tmp_path / "block-encoding-execution.json").read_text(encoding="utf-8")
    )
    rectangular = json.loads(
        (tmp_path / "rectangular-execution.json").read_text(encoding="utf-8")
    )
    assert str(tmp_path / "design-apply.json") in completed.stdout
    assert design_apply["example"] == "design-apply-report"
    assert design_apply["mode"] == "design-workflow"
    assert design_apply["kind"] == "sign"
    assert design_apply["compatibility"]["attempted_pennylane_synthesis"] is False
    assert len(design_apply["transformed_diagonal"]) == 4

    assert str(tmp_path / "linear-system.json") in completed.stdout
    assert linear_system["example"] == "linear-system-compare"
    assert linear_system["mode"] == "linear-system-comparison-workflow"
    assert "dense_solve" in linear_csv
    assert "conjugate_gradient" in linear_csv
    assert "qsvt_style_polynomial_inverse" in linear_csv

    assert threshold["example"] == "threshold-filter"
    assert threshold["mode"] == "spectral-thresholding-workflow"
    assert threshold["exact_rank"] == 2
    assert threshold["state_weight_error"] is not None

    assert problem["example"] == "problem-workflow"
    assert problem["mode"] == "problem-workflow-cookbook"
    assert problem["workflows"]["linear_system"]["schema_name"] == (
        "qsvt-problem-workflow"
    )
    assert problem["workflows"]["linear_system"]["target"] == "linear_system"
    assert problem["workflows"]["resolvent"]["target"] == "resolvent"
    resolvent_components = {
        report["component"]
        for report in problem["workflows"]["resolvent"]["resource_reports"]
    }
    assert resolvent_components == {
        "real_coeffs",
        "imag_coeffs",
    }

    assert block_encoded["example"] == "block-encoded-workflow"
    assert block_encoded["mode"] == "block-encoded-qsvt-workflow"
    assert block_encoded["verification"]["block_encoding_verified"] is True
    assert block_encoded["operator_relative_error"] is not None

    assert circuit["example"] == "circuit-execution"
    assert circuit["mode"] == "qsvt-circuit-execution-report"
    assert circuit["implementation_kind"] == (
        "pennylane-qnode-statevector-qsvt-execution"
    )
    assert circuit["is_end_to_end_quantum_algorithm"] is False

    assert block_execution["example"] == "block-encoding-execution"
    assert block_execution["mode"] == "block-encoding-qsvt-execution-report"
    assert block_execution["succeeded"] is True
    assert block_execution["block_encoding_spec"]["kind"] == "pennylane-operator"
    assert block_execution["resource_summary"]["block_encoding_method"] == (
        "prepselprep"
    )

    assert rectangular["example"] == "rectangular-execution"
    assert rectangular["succeeded"] is True
    assert rectangular["block_encoding_spec"]["logical_shape"] == [2, 3]
    assert rectangular["logical_output_relative_error"] < 1e-9
