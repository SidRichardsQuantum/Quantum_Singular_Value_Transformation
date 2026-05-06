import json

import numpy as np
import pytest

from qsvt.__main__ import main
from qsvt.design import (
    design_filter_polynomial,
    design_inverse_polynomial,
    design_power_polynomial,
    design_sign_polynomial,
    design_sqrt_polynomial,
)
from qsvt.polynomials import chebyshev_t
from qsvt.qsvt import qsvt_scalar_output, qsvt_top_left_block, qsvt_unitary
from qsvt.templates import (
    inverse_like_polynomial,
    sign_approximation_polynomial,
    soft_threshold_filter_polynomial,
    sqrt_approximation_polynomial,
)


@pytest.mark.parametrize("n", [-1, -3])
def test_chebyshev_t_rejects_negative_degree(n):
    with pytest.raises(ValueError, match="non-negative"):
        chebyshev_t(n, 0.5)


@pytest.mark.parametrize("gamma", [0.0, 1.0, -0.1, 1.2])
def test_design_inverse_rejects_invalid_gamma(gamma):
    with pytest.raises(ValueError, match="0 < gamma < 1"):
        design_inverse_polynomial(gamma=gamma, degree=5)


@pytest.mark.parametrize("degree", [-1, -5])
def test_design_functions_reject_negative_degree(degree):
    with pytest.raises(ValueError, match="degree must be non-negative"):
        design_sign_polynomial(gamma=0.25, degree=degree)


@pytest.mark.parametrize(
    ("builder", "kwargs", "message"),
    [
        (design_sqrt_polynomial, {"a": -0.1, "degree": 6}, "0 <= a < 1"),
        (design_power_polynomial, {"alpha": -0.5, "degree": 6}, "non-negative"),
        (design_filter_polynomial, {"cutoff": 1.0, "degree": 6}, "0 < cutoff < 1"),
    ],
)
def test_design_builders_reject_invalid_parameters(builder, kwargs, message):
    with pytest.raises(ValueError, match=message):
        builder(**kwargs)


@pytest.mark.parametrize(
    ("builder", "kwargs", "message"),
    [
        (inverse_like_polynomial, {"degree": 7, "mu": 0.0}, "mu must be positive"),
        (
            sign_approximation_polynomial,
            {"degree": 9, "sharpness": 0.0},
            "sharpness must be positive",
        ),
        (
            soft_threshold_filter_polynomial,
            {"degree": 10, "threshold": -0.1},
            "threshold must lie in \\[0, 1\\]",
        ),
        (
            sqrt_approximation_polynomial,
            {"degree": -1},
            "degree must be non-negative",
        ),
    ],
)
def test_template_builders_reject_invalid_parameters(builder, kwargs, message):
    with pytest.raises(ValueError, match=message):
        builder(**kwargs)


@pytest.mark.parametrize(
    "builder, kwargs",
    [
        (design_inverse_polynomial, {"gamma": 0.25, "degree": 5, "num_points": 1}),
        (inverse_like_polynomial, {"degree": 7, "num_points": 1}),
    ],
)
def test_builders_reject_too_few_grid_points(builder, kwargs):
    with pytest.raises(ValueError, match="num_points must be at least 2"):
        builder(**kwargs)


def test_qsvt_unitary_rejects_non_square_matrix():
    with pytest.raises(ValueError, match="square 2D array"):
        qsvt_unitary(np.array([[1.0, 0.0, 0.0]]), [0.0, 1.0])


def test_qsvt_functions_reject_empty_encoding_wires():
    with pytest.raises(
        ValueError, match="encoding_wires must contain at least one wire"
    ):
        qsvt_scalar_output(0.5, [0.0, 0.0, 1.0], encoding_wires=[])


def test_qsvt_top_left_block_rejects_scalar_input():
    with pytest.raises(ValueError, match="matrix input"):
        qsvt_top_left_block(0.5, [0.0, 0.0, 1.0], encoding_wires=[0])


def test_qsvt_transform_report_rejects_values_outside_qsvt_domain():
    from qsvt.qsvt import qsvt_transform_report

    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        qsvt_transform_report([1.2], [0.0, 1.0], encoding_wires=[0])


def test_cli_poly_command_emits_json(capsys):
    main(["poly", "--x", "0.5", "--poly", "0,0,1"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "polynomial"
    assert payload["poly"] == [0.0, 0.0, 1.0]
    assert payload["x"] == 0.5
    assert payload["value"] == 0.25


def test_cli_cheb_command_emits_json(capsys):
    main(["cheb", "--degree", "3", "--x", "0.5"])
    payload = json.loads(capsys.readouterr().out)

    assert payload == {
        "mode": "chebyshev",
        "degree": 3,
        "x": 0.5,
        "value": -1.0,
    }


def test_cli_design_report_emits_json(capsys):
    main(["design-report", "--kind", "sign", "--gamma", "0.2", "--degree", "13"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "design-report"
    assert payload["kind"] == "sign"
    assert payload["builder"] == "design_sign_polynomial"
    assert payload["max_error"] >= 0.0
    assert payload["bounded_margin"] >= -1e-8


def test_cli_template_report_emits_json(capsys):
    main(["template-report", "--kind", "inverse", "--degree", "7", "--mu", "0.3"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "template-report"
    assert payload["kind"] == "inverse"
    assert payload["builder"] == "inverse_like_polynomial"
    assert payload["max_error"] >= 0.0
    assert payload["bounded_margin"] >= -1e-8


def test_cli_design_report_writes_output_and_plot(tmp_path, capsys):
    output_path = tmp_path / "design-report.json"
    plot_path = tmp_path / "design-report.png"

    main(
        [
            "design-report",
            "--kind",
            "sign",
            "--gamma",
            "0.2",
            "--degree",
            "7",
            "--num-points",
            "51",
            "--bounded-num-points",
            "101",
            "--output",
            str(output_path),
            "--plot",
            str(plot_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["mode"] == "design-report"
    assert written["builder"] == "design_sign_polynomial"
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_cli_template_report_writes_output(tmp_path, capsys):
    output_path = tmp_path / "template-report.json"

    main(
        [
            "template-report",
            "--kind",
            "inverse",
            "--degree",
            "7",
            "--mu",
            "0.3",
            "--num-points",
            "51",
            "--bounded-num-points",
            "101",
            "--output",
            str(output_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["mode"] == "template-report"
    assert written["builder"] == "inverse_like_polynomial"
    assert len(written["xs"]) == 51


def test_cli_compare_report_emits_json(capsys):
    main(
        [
            "compare-report",
            "--values",
            "1.0,0.7,0.3,0.1",
            "--poly",
            "0,0,1",
            "--wires",
            "3",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "qsvt-transform-report"
    assert payload["qsvt_succeeded"] is True
    assert payload["polynomial_degree"] == 2
    assert payload["matrix_dimension"] == 4
    assert payload["max_error"] < 1e-10


def test_cli_compare_report_writes_output(tmp_path, capsys):
    output_path = tmp_path / "qsvt-report.json"

    main(
        [
            "compare-report",
            "--values",
            "1.0,0.7,0.3,0.1",
            "--poly",
            "0,0,1",
            "--wires",
            "3",
            "--output",
            str(output_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["mode"] == "qsvt-transform-report"
    assert written["max_error"] < 1e-10
    assert len(written["qsvt"]) == len(written["classical"])


def test_cli_matrix_report_emits_json(capsys):
    main(
        [
            "matrix-report",
            "--matrix",
            "0.31351701,-0.23499807;-0.23499807,0.68648299",
            "--poly",
            "0,0,1",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "qsvt-matrix-transform-report"
    assert payload["qsvt_succeeded"] is True
    assert payload["polynomial_degree"] == 2
    assert payload["matrix_dimension"] == 2
    assert payload["unitary_dimension"] == 4
    assert payload["max_error"] < 1e-10
    assert payload["max_imag_abs"] > 0.0


def test_cli_matrix_report_writes_output(tmp_path, capsys):
    output_path = tmp_path / "matrix-report.json"

    main(
        [
            "matrix-report",
            "--matrix",
            "0.31351701,-0.23499807;-0.23499807,0.68648299",
            "--poly",
            "0,0,1",
            "--output",
            str(output_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["mode"] == "qsvt-matrix-transform-report"
    assert written["max_error"] < 1e-10
    assert len(written["qsvt"]) == len(written["classical"])
    assert len(written["qsvt_imag"]) == len(written["classical"])


def test_cli_compatibility_report_emits_json(capsys):
    main(["compatibility-report", "--poly", "0,0,1"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "qsvt-compatibility-report"
    assert payload["compatible"] is True
    assert payload["parity"] == "even"
    assert payload["pennylane_synthesis_succeeded"] is True


def test_cli_design_compatibility_reports_synthesis_failure(capsys):
    main(
        [
            "design-compatibility",
            "--kind",
            "sign",
            "--degree",
            "5",
            "--gamma",
            "0.2",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "design-compatibility"
    assert payload["kind"] == "sign"
    assert payload["builder"] == "design_sign_polynomial"
    assert payload["compatible"] is False
    assert "synthesis_failed" in payload["reasons"]


def test_cli_apply_design_emits_json(capsys):
    main(
        [
            "apply-design",
            "--kind",
            "sign",
            "--values=-0.8,-0.3,0.3,0.8",
            "--degree",
            "5",
            "--gamma",
            "0.2",
            "--wires",
            "3",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "apply-design"
    assert payload["kind"] == "sign"
    assert payload["builder"] == "design_sign_polynomial"
    assert payload["polynomial_degree"] == 5
    assert payload["compatibility"]["compatible"] is False
    assert "synthesis_failed" in payload["compatibility"]["reasons"]
    assert payload["qsvt_succeeded"] is False
    assert payload["qsvt_error_type"]
