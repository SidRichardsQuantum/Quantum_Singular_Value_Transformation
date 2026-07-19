import json

import pytest

import qsvt
from qsvt.__main__ import main
from qsvt.reports import validate_report_schema
from qsvt.research import (
    ResearchOperatorSpec,
    ResearchSweepSpec,
    ResearchTargetSpec,
    expand_research_sweep,
    load_research_sweep_spec,
    run_research_sweep,
    save_research_sweep_spec,
)
from qsvt.research_frontier import run_accuracy_resource_frontier


def _small_spec(
    *,
    access_models=("embedding",),
    targets=None,
    degrees=(3,),
    tolerances=(0.5,),
    shots=(None,),
):
    if targets is None:
        targets = (ResearchTargetSpec("inverse", "inverse"),)
    return ResearchSweepSpec(
        name="small-frontier",
        study="accuracy-resource-frontier",
        operators=(ResearchOperatorSpec("poisson-2", "poisson_1d", {"size": 2}),),
        targets=targets,
        access_models=access_models,
        degrees=degrees,
        tolerances=tolerances,
        phase_solvers=("root-finding",),
        shots=shots,
        seeds=(7,),
        noise_models=("ideal",),
    )


def test_research_spec_round_trip_and_deterministic_expansion(tmp_path):
    spec = _small_spec(degrees=(3, 5), tolerances=(0.5, 0.2))
    path = save_research_sweep_spec(spec, tmp_path / "sweep.json")
    loaded = load_research_sweep_spec(path)
    first = expand_research_sweep(loaded)
    second = expand_research_sweep(loaded)

    assert loaded.as_report() == spec.as_report()
    assert validate_report_schema(loaded.as_report(), require_schema=True).supported
    assert loaded.trial_count == 4
    assert [trial.trial_id for trial in first] == [trial.trial_id for trial in second]
    assert len({trial.trial_id for trial in first}) == 4


def test_research_spec_rejects_non_sequence_and_duplicate_axes():
    report = _small_spec().as_report()
    report["degrees"] = "3"
    with pytest.raises(TypeError, match="degrees must be a sequence"):
        ResearchSweepSpec.from_mapping(report)

    with pytest.raises(ValueError, match="duplicate trials"):
        expand_research_sweep(_small_spec(degrees=(3, 3)))


def test_research_spec_yaml_round_trip_when_optional_dependency_is_available(
    tmp_path,
):
    pytest.importorskip("yaml")
    spec = _small_spec()
    yaml_path = save_research_sweep_spec(spec, tmp_path / "sweep.yaml")

    assert load_research_sweep_spec(yaml_path).as_report() == spec.as_report()


def test_research_runner_persists_and_resumes_trials(tmp_path):
    spec = _small_spec(degrees=(3, 5))
    calls = []

    def evaluator(trial):
        calls.append(trial.trial_id)
        return {
            "status": "completed",
            "summary": {"synthetic_metric": trial.degree / 10.0},
        }

    first = run_research_sweep(spec, evaluator, output_dir=tmp_path)

    def should_not_run(trial):  # pragma: no cover - assertion guard
        raise AssertionError(f"resumed trial was recomputed: {trial.trial_id}")

    second = run_research_sweep(spec, should_not_run, output_dir=tmp_path)
    summary = (tmp_path / "summary.csv").read_text(encoding="utf-8")

    assert len(calls) == 2
    assert first.executed_count == 2
    assert first.resumed_count == 0
    assert second.executed_count == 0
    assert second.resumed_count == 2
    assert second.failed_count == 0
    assert len(list((tmp_path / "trials").glob("*.json"))) == 2
    assert (tmp_path / "manifest.json").exists()
    assert "synthetic_metric" in summary
    assert validate_report_schema(first.trial_reports[0], require_schema=True).supported
    assert validate_report_schema(first.as_report(), require_schema=True).supported


def test_research_runner_recomputes_a_tampered_trial(tmp_path):
    spec = _small_spec()
    calls = []

    def evaluator(trial):
        calls.append(trial.trial_id)
        return {"status": "completed", "summary": {"degree": trial.degree}}

    first = run_research_sweep(spec, evaluator, output_dir=tmp_path)
    trial_path = next((tmp_path / "trials").glob("*.json"))
    report = json.loads(trial_path.read_text(encoding="utf-8"))
    report["factors"]["degree"] = 999
    trial_path.write_text(json.dumps(report), encoding="utf-8")

    second = run_research_sweep(spec, evaluator, output_dir=tmp_path)

    assert first.executed_count == 1
    assert second.executed_count == 1
    assert second.resumed_count == 0
    assert len(calls) == 2


def test_accuracy_resource_frontier_compares_four_access_models(tmp_path):
    spec = _small_spec(
        access_models=("embedding", "fable", "prepselprep", "qubitization"),
    )
    result = run_accuracy_resource_frontier(
        spec,
        output_dir=tmp_path,
        resume=False,
        fail_fast=True,
    )

    assert result.sweep.failed_count == 0
    assert len(result.frontier_rows) == 4
    assert len(result.pareto_rows) >= 1
    assert {row["access_model"] for row in result.frontier_rows} == {
        "embedding",
        "fable",
        "prepselprep",
        "qubitization",
    }
    assert all(row["operator_relative_error"] >= 0.0 for row in result.frontier_rows)
    assert all(row["total_gates"] > 0 for row in result.frontier_rows)
    assert all(row["logical_depth"] is None for row in result.frontier_rows)
    assert all(
        row["logical_success_probability"] is None for row in result.frontier_rows
    )
    normalizations = {
        row["access_model"]: row["normalization_alpha"] for row in result.frontier_rows
    }
    assert normalizations["fable"] > normalizations["embedding"]
    assert (tmp_path / "frontier.json").exists()
    assert (tmp_path / "pareto.csv").exists()
    assert (tmp_path / "frontier-manifest.json").exists()
    assert validate_report_schema(result.as_report(), require_schema=True).supported
    frontier_report = json.loads(
        (tmp_path / "frontier.json").read_text(encoding="utf-8")
    )
    assert validate_report_schema(frontier_report, require_schema=True).supported


def test_accuracy_resource_frontier_supports_all_target_families():
    targets = tuple(
        ResearchTargetSpec(kind.replace("_", "-"), kind)
        for kind in ("inverse", "projector", "band_filter", "resolvent")
    )
    result = run_accuracy_resource_frontier(
        _small_spec(targets=targets),
        fail_fast=True,
    )
    reports = result.sweep.trial_reports
    resolvent = next(
        report
        for report in reports
        if report["factors"]["target"]["kind"] == "resolvent"
    )

    assert len(reports) == 4
    assert all(report["status"] == "completed" for report in reports)
    assert resolvent["result"]["summary"]["polynomial_component_count"] == 2


def test_frontier_retains_unsupported_finite_shot_factor_as_skipped():
    result = run_accuracy_resource_frontier(_small_spec(shots=(200,)))
    report = result.sweep.trial_reports[0]

    assert report["status"] == "skipped"
    assert report["result"]["error_type"] == "UnsupportedResearchFactor"


def test_research_sweep_cli_writes_artifacts(tmp_path, capsys):
    config = save_research_sweep_spec(_small_spec(), tmp_path / "config.json")
    output_dir = tmp_path / "study"
    main(
        [
            "research-sweep",
            "--config",
            str(config),
            "--output-dir",
            str(output_dir),
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert report["mode"] == "accuracy-resource-frontier"
    assert report["sweep"]["trial_count"] == 1
    assert report["sweep"]["failed_count"] == 0
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "summary.csv").exists()


def test_research_interfaces_are_exported_as_experimental():
    assert qsvt.ResearchSweepSpec is ResearchSweepSpec
    assert qsvt.api_status("ResearchSweepSpec") == qsvt.API_STATUS_EXPERIMENTAL
    assert (
        qsvt.api_status("run_accuracy_resource_frontier")
        == qsvt.API_STATUS_EXPERIMENTAL
    )
