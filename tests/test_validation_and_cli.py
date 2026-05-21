import json
from importlib import resources

import numpy as np
import pytest

import qsvt
from qsvt.__main__ import main
from qsvt.design import (
    design_filter_polynomial,
    design_interval_projector_polynomial,
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


def test_package_exposes_pep561_type_marker():
    marker = resources.files("qsvt").joinpath("py.typed")

    assert marker.is_file()


def test_top_level_public_api_exports_are_resolvable():
    assert qsvt.__api_status__ == "alpha"
    assert "qsvt.__all__" in qsvt.__public_api_policy__

    exported = set(qsvt.__all__)
    for name in exported:
        assert hasattr(qsvt, name), name

    expected_stable_surface = {
        "design_workflow",
        "linear_system_workflow",
        "hamiltonian_simulation_workflow",
        "ground_state_filtering_workflow",
        "resolvent_workflow",
        "spectral_density_workflow",
        "thermal_gibbs_workflow",
        "qsvt_transform_report",
        "qsvt_matrix_transform_report",
        "report_to_jsonable",
        "save_report",
    }
    assert expected_stable_surface <= exported


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
        (
            design_interval_projector_polynomial,
            {"lower": 0.4, "upper": -0.2, "degree": 6},
            "-1 < lower < upper < 1",
        ),
        (
            design_interval_projector_polynomial,
            {"lower": -0.2, "upper": 0.4, "degree": 6, "sharpness": 0.0},
            "sharpness must be positive",
        ),
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


def test_cli_examples_command_emits_discovery_payload(capsys):
    main(["examples"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "examples"
    assert "sign" in payload["design_kinds"]
    assert "exponential" in payload["template_kinds"]
    assert "cg-solve" in payload["benchmark_commands"]
    assert any("resource-report" in example for example in payload["examples"])


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
    main(
        [
            "design-report",
            "--kind",
            "sign",
            "--gamma",
            "0.2",
            "--degree",
            "5",
            "--num-points",
            "51",
            "--bounded-num-points",
            "101",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "design-report"
    assert payload["kind"] == "sign"
    assert payload["builder"] == "design_sign_polynomial"
    assert payload["max_error"] >= 0.0
    assert payload["bounded_margin"] >= -1e-8


def test_cli_design_workflow_emits_json(capsys):
    main(
        [
            "design-workflow",
            "--kind",
            "sign",
            "--gamma",
            "0.2",
            "--degree",
            "5",
            "--num-points",
            "51",
            "--bounded-num-points",
            "101",
            "--no-synthesis",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "design-workflow"
    assert payload["kind"] == "sign"
    assert payload["builder"] == "design_sign_polynomial"
    assert payload["diagnostics"]["builder"] == "design_sign_polynomial"
    assert payload["compatibility"]["attempted_pennylane_synthesis"] is False
    assert len(payload["coeffs"]) == 6


def test_cli_resource_report_emits_json(capsys):
    main(
        [
            "resource-report",
            "--poly",
            "0,0,1",
            "--matrix-dimension",
            "3",
            "--no-synthesis",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "resource-report"
    assert payload["resources"]["degree"] == 2
    assert payload["resources"]["encoding_qubits"] == 2
    assert payload["compatibility"]["attempted_pennylane_synthesis"] is False


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
    assert payload["report_written"] is True
    assert payload["plot_written"] is True
    assert written["builder"] == "design_sign_polynomial"
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_cli_design_workflow_writes_output(tmp_path, capsys):
    output_path = tmp_path / "design-workflow.json"

    main(
        [
            "design-workflow",
            "--kind",
            "filter",
            "--degree",
            "6",
            "--cutoff",
            "0.4",
            "--num-points",
            "51",
            "--bounded-num-points",
            "101",
            "--no-synthesis",
            "--output",
            str(output_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["mode"] == "design-workflow"
    assert payload["report_written"] is True
    assert written["kind"] == "filter"
    assert written["diagnostics"]["builder"] == "design_filter_polynomial"
    assert written["compatibility"]["attempted_pennylane_synthesis"] is False


def test_cli_design_sweep_writes_compact_manifest(tmp_path, capsys):
    output_path = tmp_path / "design-sweep.json"

    main(
        [
            "design-sweep",
            "--kind",
            "sign",
            "--degrees",
            "5,7",
            "--gamma",
            "0.2",
            "--num-points",
            "51",
            "--bounded-num-points",
            "101",
            "--no-synthesis",
            "--output",
            str(output_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["mode"] == "design-sweep"
    assert payload["report_written"] is True
    assert written["kind"] == "sign"
    assert written["degrees"] == [5, 7]
    assert [row["degree"] for row in written["rows"]] == [5, 7]
    assert {row["builder"] for row in written["rows"]} == {
        "design_sign_polynomial",
    }
    assert all(row["max_error"] >= 0.0 for row in written["rows"])
    assert all(row["attempted_pennylane_synthesis"] is False for row in written["rows"])


def test_cli_design_sweep_can_print_full_payload_with_output(tmp_path, capsys):
    output_path = tmp_path / "filter-sweep.json"

    main(
        [
            "design-sweep",
            "--kind",
            "filter",
            "--degrees",
            "6,10",
            "--cutoff",
            "0.4",
            "--num-points",
            "51",
            "--bounded-num-points",
            "101",
            "--no-synthesis",
            "--output",
            str(output_path),
            "--print-report",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["mode"] == "design-sweep"
    assert payload["kind"] == "filter"
    assert payload["degrees"] == [6, 10]
    assert payload == written


def test_cli_design_sweep_rejects_empty_degree_list():
    with pytest.raises(ValueError, match="expected at least one integer"):
        main(
            [
                "design-sweep",
                "--kind",
                "sign",
                "--degrees",
                "",
                "--no-synthesis",
            ]
        )


def test_cli_design_sweep_rejects_malformed_degree_list():
    with pytest.raises(ValueError, match="invalid literal"):
        main(
            [
                "design-sweep",
                "--kind",
                "sign",
                "--degrees",
                "5,nope",
                "--no-synthesis",
            ]
        )


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
    assert payload["report_written"] is True
    assert payload["plot_written"] is False
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
    assert payload["report_written"] is True
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


def test_cli_matrix_report_supports_complex_hermitian_inputs(capsys):
    main(
        [
            "matrix-report",
            "--matrix",
            "0,0.2j;-0.2j,0",
            "--poly",
            "0,1",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "qsvt-matrix-transform-report"
    assert payload["comparison_basis"] == "full_complex"
    assert payload["max_error"] < 1e-6


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
    assert payload["report_written"] is True
    assert written["max_error"] < 1e-10
    assert len(written["qsvt"]) == len(written["classical"])
    assert len(written["qsvt_imag"]) == len(written["classical"])


def test_cli_report_can_still_print_full_payload_with_output(tmp_path, capsys):
    output_path = tmp_path / "compatibility-report.json"

    main(
        [
            "compatibility-report",
            "--poly",
            "0,0,1",
            "--output",
            str(output_path),
            "--print-report",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "qsvt-compatibility-report"
    assert payload["compatible"] is True


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


def test_cli_threshold_workflow_emits_json(capsys):
    main(
        [
            "threshold-workflow",
            "--matrix=-0.8,0,0,0;0,-0.15,0,0;0,0,0.2,0;0,0,0,0.75",
            "--lower",
            "-0.3",
            "--upper",
            "0.3",
            "--degree",
            "24",
            "--sharpness",
            "16",
            "--state",
            "0.1,0.8,0.5,0.2",
            "--num-points",
            "301",
            "--bounded-num-points",
            "601",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "spectral-thresholding-workflow"
    assert payload["exact_rank"] == 2
    assert abs(payload["polynomial_rank_proxy"] - 2.0) < 0.3
    assert payload["state_weight_error"] < 0.2
    assert payload["diagnostics"]["builder"] == "design_interval_projector_polynomial"


def test_cli_threshold_workflow_writes_output(tmp_path, capsys):
    output_path = tmp_path / "threshold-workflow.json"

    main(
        [
            "threshold-workflow",
            "--matrix=-1,0,0;0,0,0;0,0,1",
            "--lower",
            "-0.25",
            "--upper",
            "0.25",
            "--degree",
            "24",
            "--num-points",
            "301",
            "--bounded-num-points",
            "601",
            "--output",
            str(output_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["mode"] == "spectral-thresholding-workflow"
    assert payload["report_written"] is True
    assert written["exact_rank"] == 1
    assert written["lower"] == -0.25
    assert written["upper"] == 0.25


def test_cli_benchmark_dense_solve_emits_json(capsys):
    main(
        [
            "benchmark",
            "dense-solve",
            "--matrix",
            "4,1;1,3",
            "--rhs",
            "1,2",
            "--repeats",
            "1",
            "--qsvt-poly",
            "0,1",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "classical-benchmark"
    assert payload["algorithm"] == "numpy.linalg.solve"
    assert payload["metrics"]["relative_residual_norm"] < 1e-12
    assert payload["qsvt_proxy"]["resources"]["degree"] == 1


def test_cli_benchmark_cg_solve_writes_output(tmp_path, capsys):
    output_path = tmp_path / "cg-benchmark.json"

    main(
        [
            "benchmark",
            "cg-solve",
            "--matrix",
            "4,1;1,3",
            "--rhs",
            "1,2",
            "--tolerance",
            "1e-12",
            "--repeats",
            "1",
            "--output",
            str(output_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["mode"] == "classical-benchmark"
    assert payload["report_written"] is True
    assert written["problem"] == "positive-definite-linear-system"
    assert written["metrics"]["converged"] is True


def test_cli_benchmark_polynomial_emits_qsvt_proxy(capsys):
    main(
        [
            "benchmark",
            "polynomial",
            "--matrix",
            "0.5,0;0,-0.25",
            "--poly",
            "0,0,1",
            "--repeats",
            "1",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["problem"] == "polynomial-matrix-function"
    assert payload["metrics"]["polynomial_degree"] == 2
    assert payload["qsvt_proxy"]["resources"]["signal_operator_calls"] == 2


def test_cli_benchmark_spectral_function_emits_json(capsys):
    main(
        [
            "benchmark",
            "spectral-function",
            "--matrix",
            "1,0;0,4",
            "--function",
            "sqrt",
            "--repeats",
            "1",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["problem"] == "sqrt-matrix-function"
    assert payload["metrics"]["function"] == "sqrt"
    assert payload["metrics"]["output_frobenius_norm"] > 0.0
