import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from qsvt.reports import (
    load_report,
    plot_approximation_report,
    report_to_jsonable,
    save_report,
    save_report_plot,
)

matplotlib.use("Agg")

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "reports"


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


def test_qsvt_problem_workflow_v1_fixture_documents_truth_contract():
    report = load_report(FIXTURE_DIR / "qsvt_problem_workflow_v1.json")

    assert report["schema_name"] == "qsvt-problem-workflow"
    assert report["schema_version"] == "1.0"
    assert report["target"] == "linear_system"
    assert report["truth_contract"]["finite_qsvt_execution"] is False
    assert report["truth_contract"]["classical_reference_available"] is True
    assert "omitted_quantum_layers" in report["truth_contract"]
