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
    accuracy_driven_plan,
    block_encoded_workflow,
    block_encoding_execution,
    circuit_execution,
    custom_block_encoding,
    encoding_aware_resources,
    finite_shot_qsvt,
    hamiltonian_simulation,
    linear_system_compare,
    poisson_qsvt,
    problem_workflow,
    rectangular_execution,
    spectral_filter_qsvt,
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
spectral_filter_qsvt.main([
    "--output", str(output_dir / "spectral-filter-qsvt.json"),
])
poisson_qsvt.main(["--output", str(output_dir / "poisson-qsvt.json")])
hamiltonian_simulation.main([
    "--output", str(output_dir / "hamiltonian-simulation.json"),
])
accuracy_driven_plan.main([
    "--output", str(output_dir / "accuracy-driven-plan.json"),
])
custom_block_encoding.main([
    "--output", str(output_dir / "custom-block-encoding.json"),
])
finite_shot_qsvt.main([
    "--output", str(output_dir / "finite-shot-qsvt.json"),
    "--shots", "2000",
    "--seed", "12345",
])
encoding_aware_resources.main([
    "--output", str(output_dir / "encoding-aware-resources.json"),
    "--rows-output", str(output_dir / "encoding-aware-resources.csv"),
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
    spectral_filter = json.loads(
        (tmp_path / "spectral-filter-qsvt.json").read_text(encoding="utf-8")
    )
    poisson = json.loads((tmp_path / "poisson-qsvt.json").read_text(encoding="utf-8"))
    hamiltonian = json.loads(
        (tmp_path / "hamiltonian-simulation.json").read_text(encoding="utf-8")
    )
    accuracy_plan = json.loads(
        (tmp_path / "accuracy-driven-plan.json").read_text(encoding="utf-8")
    )
    custom_encoding = json.loads(
        (tmp_path / "custom-block-encoding.json").read_text(encoding="utf-8")
    )
    finite_shot = json.loads(
        (tmp_path / "finite-shot-qsvt.json").read_text(encoding="utf-8")
    )
    encoding_resources = json.loads(
        (tmp_path / "encoding-aware-resources.json").read_text(encoding="utf-8")
    )
    encoding_csv = (tmp_path / "encoding-aware-resources.csv").read_text(
        encoding="utf-8"
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

    assert spectral_filter["example"] == "spectral-filter-qsvt"
    assert spectral_filter["mode"] == "spectral-filter-qsvt-flagship"
    assert spectral_filter["degree_search"]["met_tolerance"] is True
    assert spectral_filter["execution"]["succeeded"] is True
    assert spectral_filter["resources"]["estimator_model"] == ("pauli-lcu-qubitization")
    assert spectral_filter["acceptance"]["accepted_for_stated_scope"] is True
    assert spectral_filter["acceptance"]["full_qsvt_acceptance"] is True

    assert poisson["example"] == "poisson-qsvt"
    assert poisson["mode"] == "poisson-qsvt-flagship"
    assert poisson["conjugate_gradient"]["converged"] is True
    assert poisson["degree_search"]["met_tolerance"] is True
    assert poisson["execution"]["succeeded"] is True
    assert poisson["acceptance"]["accepted_for_stated_scope"] is True
    assert poisson["acceptance"]["full_qsvt_acceptance"] is True

    assert hamiltonian["example"] == "hamiltonian-simulation"
    assert hamiltonian["mode"] == "hamiltonian-simulation-workflow"
    assert hamiltonian["acceptance"]["scope"] == "finite_qsvt"
    assert hamiltonian["acceptance"]["accepted_for_stated_scope"] is True
    assert hamiltonian["acceptance"]["full_qsvt_acceptance"] is True
    assert hamiltonian["qsvt_execution"]["succeeded"] is True

    assert accuracy_plan["example"] == "accuracy-driven-plan"
    assert accuracy_plan["mode"] == "accuracy-driven-plan-cookbook"
    assert accuracy_plan["summary"]["selected_degree"] == 7
    assert accuracy_plan["summary"]["met_tolerance"] is True
    assert accuracy_plan["summary"]["access_model_status"] == "matrix-fallback"
    assert accuracy_plan["execution"]["succeeded"] is True
    assert len(accuracy_plan["degree_candidates"]) == 3

    assert custom_encoding["example"] == "custom-block-encoding"
    assert custom_encoding["block_encoding_spec"]["kind"] == "custom-circuit"
    assert custom_encoding["projector_source"] == "caller-supplied-projectors"
    assert custom_encoding["succeeded"] is True
    assert custom_encoding["known_block_validation"]["block_absolute_error"] < 1e-12
    assert (
        custom_encoding["known_block_validation"]["logical_output_absolute_error"]
        < 1e-5
    )

    assert finite_shot["mode"] == "finite-shot-qsvt-cookbook"
    assert finite_shot["sampled_execution"]["succeeded"] is True
    assert finite_shot["sampled_execution"]["preflight"]["passed"] is True
    assert finite_shot["sampled_execution"]["shots"] == 2000
    assert finite_shot["ideal_execution"]["succeeded"] is True
    assert finite_shot["comparison"]["logical_probability_l2_error"] < 0.08
    assert finite_shot["truth_contract"]["uses_real_hardware"] is False

    assert encoding_resources["example"] == "encoding-aware-resources"
    assert encoding_resources["mode"] == "encoding-aware-resource-cookbook"
    assert len(encoding_resources["rows"]) == 20
    assert {row["estimator_model"] for row in encoding_resources["rows"]} == {
        "arbitrary-unitary-block-encoding",
        "pauli-lcu-qubitization",
    }
    assert all(row["total_gates"] > 0 for row in encoding_resources["rows"])
    assert "access_model" in encoding_csv
    assert "PrepSelPrep" in encoding_csv
