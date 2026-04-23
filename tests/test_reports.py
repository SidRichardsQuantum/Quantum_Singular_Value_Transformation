import json

import matplotlib
import numpy as np

from qsvt.design import design_sign_diagnostics
from qsvt.reports import (
    load_report,
    plot_approximation_report,
    report_to_jsonable,
    save_report,
    save_report_plot,
)

matplotlib.use("Agg")


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


def test_save_and_load_report_round_trip(tmp_path):
    report = design_sign_diagnostics(
        gamma=0.2,
        degree=7,
        num_points=51,
        bounded_num_points=101,
    )
    path = tmp_path / "report.json"

    written = save_report(report, path)
    loaded = load_report(path)

    assert written == path
    assert loaded["builder"] == "design_sign_polynomial"
    assert loaded["fit_domain"] == [-1.0, 1.0]
    assert len(loaded["xs"]) == 51
    assert loaded["max_error"] >= 0.0


def test_plot_approximation_report_returns_figure_and_axes():
    report = design_sign_diagnostics(
        gamma=0.2,
        degree=7,
        num_points=51,
        bounded_num_points=101,
    )

    fig, axes = plot_approximation_report(report)

    assert fig is not None
    assert len(axes) == 2
    assert axes[0].get_ylabel() == "value"
    assert axes[1].get_ylabel() == "error"


def test_save_report_plot_writes_image(tmp_path):
    report = design_sign_diagnostics(
        gamma=0.2,
        degree=7,
        num_points=51,
        bounded_num_points=101,
    )
    path = tmp_path / "report.png"

    written = save_report_plot(report, path)

    assert written == path
    assert path.exists()
    assert path.stat().st_size > 0
