import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from qsvt.reports import (
    load_report,
    load_report_with_schema,
    plot_approximation_report,
    report_schema_manifest,
    report_to_jsonable,
    save_report,
    save_report_plot,
    supported_report_schemas,
    validate_report_schema,
    write_report_schema_manifest_csv,
)

matplotlib.use("Agg")

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "reports"
MANIFEST_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "report_manifests"
    / "schema_manifest_v1.json"
)


@pytest.fixture
def approximation_report():
    xs = np.linspace(-1.0, 1.0, 11)
    target_values = xs**2
    polynomial_values = target_values + 0.01 * xs
    errors = polynomial_values - target_values
    return {
        "builder": "test_polynomial",
        "kind": "test",
        "degree": 2,
        "fit_domain": (-1.0, 1.0),
        "xs": xs,
        "target_values": target_values,
        "polynomial_values": polynomial_values,
        "errors": errors,
        "max_error": float(np.max(np.abs(errors))),
        "bounded_margin": 0.0,
    }


def test_report_to_jsonable_converts_numpy_values():
    report = {
        "domain": (-1.0, 1.0),
        "values": np.array([0.0, 0.5, 1.0]),
        "metric": np.float64(0.25),
        "nested": {"ok": np.bool_(True)},
    }

    payload = report_to_jsonable(report)

    assert payload == {
        "domain": [-1.0, 1.0],
        "values": [0.0, 0.5, 1.0],
        "metric": 0.25,
        "nested": {"ok": True},
    }
    json.dumps(payload)


def test_report_to_jsonable_converts_complex_values():
    report = {
        "matrix": np.array([[1 + 2j, 0 - 1j], [0 + 1j, 3 + 0j]]),
        "value": np.complex128(0.5 + 0.25j),
    }

    payload = report_to_jsonable(report)

    assert payload == {
        "matrix": {
            "real": [[1.0, 0.0], [0.0, 3.0]],
            "imag": [[2.0, -1.0], [1.0, 0.0]],
        },
        "value": {
            "real": 0.5,
            "imag": 0.25,
        },
    }
    json.dumps(payload)


def test_report_to_jsonable_converts_lists_and_complex_scalars():
    payload = report_to_jsonable(
        {
            "items": [np.int64(1), (np.float64(2.0), 3 + 4j)],
            5: "integer key",
        }
    )

    assert payload == {
        "items": [1, [2.0, {"real": 3.0, "imag": 4.0}]],
        "5": "integer key",
    }
    json.dumps(payload)


def test_save_and_load_report_round_trip(tmp_path, approximation_report):
    path = tmp_path / "report.json"

    written = save_report(approximation_report, path)
    loaded = load_report(path)

    assert written == path
    assert loaded["builder"] == "test_polynomial"
    assert loaded["fit_domain"] == [-1.0, 1.0]
    assert len(loaded["xs"]) == 11
    assert loaded["max_error"] >= 0.0


def test_plot_approximation_report_returns_figure_and_axes(approximation_report):
    fig, axes = plot_approximation_report(approximation_report)

    assert fig is not None
    assert len(axes) == 2
    assert axes[0].get_ylabel() == "value"
    assert axes[1].get_ylabel() == "error"


def test_plot_approximation_report_supports_existing_axis_and_default_label():
    xs = np.linspace(-1.0, 1.0, 5)
    report = {
        "xs": xs,
        "target_values": xs,
        "polynomial_values": xs,
        "errors": np.zeros_like(xs),
    }
    fig, ax = plt.subplots()

    returned_fig, axes = plot_approximation_report(report, ax=ax)

    assert returned_fig is fig
    assert axes[0] is ax
    assert axes[1].get_ylim() == pytest.approx((-1.0, 1.0))
    assert axes[0].get_legend() is not None
    plt.close(fig)


def test_plot_approximation_report_handles_nonfinite_errors():
    xs = np.linspace(-1.0, 1.0, 5)
    report = {
        "xs": xs,
        "target_values": xs,
        "polynomial_values": xs,
        "errors": np.full_like(xs, np.nan),
    }

    fig, axes = plot_approximation_report(report)

    assert axes[1].get_ylim()[0] < axes[1].get_ylim()[1]
    plt.close(fig)


def test_plot_approximation_report_validates_required_arrays():
    with pytest.raises(KeyError, match="missing required key"):
        plot_approximation_report({})

    with pytest.raises(ValueError, match="one-dimensional"):
        plot_approximation_report(
            {
                "xs": [[0.0, 1.0]],
                "target_values": [0.0, 1.0],
                "polynomial_values": [0.0, 1.0],
                "errors": [0.0, 0.0],
            }
        )


def test_save_report_plot_writes_image(tmp_path, approximation_report):
    path = tmp_path / "report.png"

    written = save_report_plot(approximation_report, path)

    assert written == path
    assert path.exists()
    assert path.stat().st_size > 0


def test_report_schema_fixtures_remain_loadable_and_identifiable():
    fixtures = sorted(FIXTURE_DIR.glob("*.json"))

    assert fixtures, "expected at least one report schema fixture"
    for fixture in fixtures:
        report = load_report(fixture)
        assert isinstance(report["schema_name"], str)
        assert isinstance(report["schema_version"], str)
        assert report["schema_version"]
        json.dumps(report)


def test_report_schema_fixtures_pass_explicit_compatibility_checks():
    fixtures = sorted(FIXTURE_DIR.glob("*.json"))

    assert fixtures, "expected at least one report schema fixture"
    for fixture in fixtures:
        report, compatibility = load_report_with_schema(fixture)

        assert compatibility.supported is True
        assert compatibility.migration_required is False
        assert compatibility.schema_name == report["schema_name"]
        assert compatibility.schema_version == report["schema_version"]
        assert compatibility.missing_fields == ()
        assert "schema_name" in compatibility.required_fields
        json.dumps(compatibility.as_report())


def test_supported_report_schema_registry_matches_fixture_families():
    registry = supported_report_schemas()
    fixtures = sorted(FIXTURE_DIR.glob("*.json"))
    fixture_pairs = {
        (
            load_report(fixture)["schema_name"],
            load_report(fixture)["schema_version"],
        )
        for fixture in fixtures
    }

    assert registry["qsvt-problem-workflow"] == ("1.0",)
    assert registry["block-encoding-qsvt-execution"] == ("1.0",)
    assert registry["hardware-qsvt-execution"] == ("1.0",)
    assert registry["hardware-qsvt-circuit"] == ("1.0",)
    assert fixture_pairs <= {
        (schema_name, version)
        for schema_name, versions in registry.items()
        for version in versions
    }


def test_load_report_with_schema_can_enforce_expected_schema(tmp_path):
    path = tmp_path / "hardware.json"
    save_report(
        {
            "schema_name": "hardware-qsvt-execution",
            "schema_version": "1.0",
            "mode": "hardware-qsvt-execution",
            "implementation_kind": "fixture",
            "truth_contract": {},
            "resource_summary": {},
        },
        path,
    )

    report, compatibility = load_report_with_schema(
        path,
        expected_schema_name="hardware-qsvt-execution",
        expected_schema_version="1.0",
    )

    assert report["schema_name"] == "hardware-qsvt-execution"
    assert compatibility.supported is True
    with pytest.raises(ValueError, match="does not match expected schema"):
        load_report_with_schema(
            path,
            expected_schema_name="qsvt-problem-workflow",
        )
    with pytest.raises(ValueError, match="does not match expected version"):
        load_report_with_schema(
            path,
            expected_schema_version="2.0",
        )


def test_validate_report_schema_reports_missing_required_fields(tmp_path):
    report = {
        "schema_name": "qsvt-problem-workflow",
        "schema_version": "1.0",
        "mode": "qsvt-problem-workflow",
    }

    compatibility = validate_report_schema(report, require_schema=True)

    assert compatibility.supported is False
    assert compatibility.migration_required is False
    assert "missing required fields" in compatibility.message
    assert "truth_contract" in compatibility.missing_fields
    assert "target" in compatibility.required_fields

    path = tmp_path / "incomplete.json"
    save_report(report, path)
    with pytest.raises(ValueError, match="missing required fields"):
        load_report_with_schema(path)


def test_validate_report_schema_reports_unknown_fields_without_failing():
    compatibility = validate_report_schema(
        {
            "schema_name": "hardware-qsvt-execution",
            "schema_version": "1.0",
            "mode": "hardware-qsvt-execution",
            "implementation_kind": "fixture",
            "truth_contract": {},
            "resource_summary": {},
            "extra_field": "allowed but reported",
        },
        require_schema=True,
    )

    assert compatibility.supported is True
    assert compatibility.unknown_fields == ("extra_field",)
    assert compatibility.as_report()["unknown_fields"] == ["extra_field"]


def test_validate_report_schema_reports_intentional_migration_message():
    compatibility = validate_report_schema(
        {
            "schema_name": "qsvt-problem-workflow",
            "schema_version": "2.0",
        },
        require_schema=True,
    )

    assert compatibility.supported is False
    assert compatibility.migration_required is True
    assert "unsupported" in compatibility.message
    assert "supported versions: 1.0" in compatibility.message


def test_report_schema_manifest_summarizes_paths(tmp_path):
    valid_path = tmp_path / "valid.json"
    invalid_path = tmp_path / "invalid.json"
    save_report(
        {
            "schema_name": "hardware-qsvt-execution",
            "schema_version": "1.0",
            "mode": "hardware-qsvt-execution",
            "implementation_kind": "fixture",
            "truth_contract": {},
            "resource_summary": {},
        },
        valid_path,
    )
    invalid_path.write_text("{", encoding="utf-8")

    manifest = report_schema_manifest([valid_path, invalid_path])

    assert manifest[0]["path"] == str(valid_path)
    assert manifest[0]["supported"] is True
    assert manifest[0]["missing_fields"] == []
    assert manifest[1]["path"] == str(invalid_path)
    assert manifest[1]["supported"] is False
    assert "invalid JSON" in manifest[1]["message"]


def test_write_report_schema_manifest_csv_writes_compact_rows(tmp_path):
    report_path = tmp_path / "valid.json"
    csv_path = tmp_path / "manifest.csv"
    save_report(
        {
            "schema_name": "hardware-qsvt-execution",
            "schema_version": "1.0",
            "mode": "hardware-qsvt-execution",
            "implementation_kind": "fixture",
            "truth_contract": {},
            "resource_summary": {},
            "extra_field": "reported",
        },
        report_path,
    )
    rows = report_schema_manifest([report_path])

    written = write_report_schema_manifest_csv(rows, csv_path)
    text = csv_path.read_text(encoding="utf-8")

    assert written == csv_path
    assert "path,schema_name,schema_version,supported" in text
    assert "hardware-qsvt-execution" in text
    assert "extra_field" in text


def test_report_schema_manifest_fixture_matches_committed_report_fixtures():
    fixture = load_report(MANIFEST_FIXTURE)
    paths = [
        Path("tests/fixtures/reports/block_encoding_qsvt_execution_v1.json"),
        Path("tests/fixtures/reports/hardware_qsvt_circuit_v1.json"),
        Path("tests/fixtures/reports/hardware_qsvt_execution_v1.json"),
        Path("tests/fixtures/reports/qsvt_problem_workflow_v1.json"),
    ]

    manifest = report_schema_manifest(paths)

    assert fixture["mode"] == "report-schema-manifest-fixture"
    assert fixture["manifest_version"] == "1.0"
    assert fixture["rows"] == manifest
    assert all(row["supported"] for row in fixture["rows"])


def test_load_report_with_schema_rejects_unversioned_reports_when_required(
    tmp_path,
):
    path = tmp_path / "unversioned.json"
    save_report({"mode": "design-report"}, path)

    with pytest.raises(ValueError, match="missing required schema_name"):
        load_report_with_schema(path)

    report, compatibility = load_report_with_schema(path, require_schema=False)
    assert report["mode"] == "design-report"
    assert compatibility.supported is True
    assert compatibility.schema_name is None


def test_qsvt_problem_workflow_v1_fixture_documents_truth_contract():
    report = load_report(FIXTURE_DIR / "qsvt_problem_workflow_v1.json")

    assert report["schema_name"] == "qsvt-problem-workflow"
    assert report["schema_version"] == "1.0"
    assert report["target"] == "linear_system"
    assert report["truth_contract"]["finite_qsvt_execution"] is False
    assert report["truth_contract"]["classical_reference_available"] is True
    assert "omitted_quantum_layers" in report["truth_contract"]
