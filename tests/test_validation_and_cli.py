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
