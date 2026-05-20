import numpy as np

from qsvt.algorithms import (
    ground_state_filtering_workflow,
    hamiltonian_simulation_workflow,
    linear_system_workflow,
    resolvent_workflow,
    spectral_density_workflow,
    spectral_thresholding_workflow,
    thermal_gibbs_workflow,
)
from qsvt.reports import report_to_jsonable

HERMitian_2X2 = np.array(
    [
        [0.0, 0.2],
        [0.2, 1.0],
    ],
)
STATE_2 = np.array([1.0, 0.3])


def test_linear_system_workflow_regression():
    matrix = np.array(
        [
            [2.0, 0.25],
            [0.25, 1.25],
        ],
    )
    rhs = np.array([1.0, -0.5])

    result = linear_system_workflow(
        matrix,
        rhs,
        degree=8,
        num_points=201,
        bounded_num_points=401,
        attempt_synthesis=False,
        apply_qsvt=False,
    )
    report = report_to_jsonable(result.as_report())

    assert report["mode"] == "linear-system-workflow"
    assert np.isclose(result.gamma, 0.5657414540893352)
    assert result.polynomial_residual_norm < 0.06
    assert result.polynomial_relative_error < 0.06
    assert result.compatibility["attempted_pennylane_synthesis"] is False
    assert result.qsvt_solution is None


def test_ground_state_filtering_workflow_regression():
    result = ground_state_filtering_workflow(
        HERMitian_2X2,
        STATE_2,
        degree=8,
        width=0.45,
        num_points=201,
    )
    report = report_to_jsonable(result.as_report())

    assert report["mode"] == "ground-state-filtering-workflow"
    assert result.reference_state_error < 1e-5
    assert result.operator_relative_error < 1e-3
    assert result.ground_state_overlap > 0.999999


def test_hamiltonian_simulation_workflow_regression():
    result = hamiltonian_simulation_workflow(
        HERMitian_2X2,
        STATE_2,
        time=0.4,
        degree=9,
        num_points=201,
    )

    assert result.state_relative_error < 1e-12
    assert result.operator_relative_error < 1e-12
    assert result.norm_drift < 1e-12


def test_resolvent_workflow_regression():
    result = resolvent_workflow(
        HERMitian_2X2,
        omega=0.4,
        eta=0.3,
        degree=10,
        source=STATE_2,
        num_points=301,
    )

    assert result.operator_relative_error < 0.013
    assert result.response_relative_error is not None
    assert result.response_relative_error < 0.013


def test_spectral_density_workflow_regression():
    result = spectral_density_workflow(
        HERMitian_2X2,
        np.array([-0.1, 0.5, 1.1]),
        width=0.35,
        degree=8,
        state=STATE_2,
        num_points=201,
    )

    assert result.trace_density_error < 1e-4
    assert result.state_weight_error is not None
    assert result.state_weight_error < 2e-4


def test_spectral_thresholding_workflow_regression():
    matrix = np.diag([-0.8, -0.15, 0.2, 0.75])
    state = np.array([0.1, 0.8, 0.5, 0.2])

    result = spectral_thresholding_workflow(
        matrix,
        lower=-0.3,
        upper=0.3,
        degree=36,
        sharpness=20.0,
        state=state,
        num_points=401,
        bounded_num_points=801,
    )
    report = report_to_jsonable(result.as_report())

    assert report["mode"] == "spectral-thresholding-workflow"
    assert result.exact_rank == 2
    assert abs(result.polynomial_rank_proxy - result.exact_rank) < 0.15
    assert result.operator_relative_error < 0.12
    assert result.leakage_outside_interval < 0.08
    assert result.state_weight_error is not None
    assert result.state_weight_error < 0.08


def test_thermal_gibbs_workflow_regression():
    result = thermal_gibbs_workflow(
        HERMitian_2X2,
        beta=0.7,
        degree=8,
        state=STATE_2,
        num_points=201,
    )

    assert result.operator_relative_error < 1e-10
    assert result.density_matrix_relative_error < 1e-10
    assert result.weighted_state_error is not None
    assert result.weighted_state_error < 1e-10
