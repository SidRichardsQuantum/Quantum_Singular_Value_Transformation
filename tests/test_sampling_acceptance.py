from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from qsvt._sampling_acceptance import probability_acceptance
from qsvt.algorithms import hamiltonian_simulation_workflow
from qsvt.flagship import poisson_qsvt_workflow


def evidence(counts, *, reference=(1.0, 1.0), tolerance=0.05, confidence=0.95):
    counts = np.asarray(counts)
    shots = int(counts.sum())
    execution = SimpleNamespace(
        succeeded=True,
        probabilities=counts / shots,
        shots=shots,
        logical_success_probability=float(counts[: len(reference)].sum() / shots),
    )
    return probability_acceptance(
        execution, np.asarray(reference), tolerance=tolerance, confidence=confidence
    )


def test_probability_acceptance_requires_precision_not_just_reference_overlap():
    assert evidence([2500, 2500, 5000])["status"] == "accepted"
    sparse = evidence([1, 1, 98])
    assert sparse["status"] == "insufficient_shots"
    assert sparse["maximum_probability_error"] == 0
    assert sparse["maximum_probability_error_bound"] > sparse["tolerance"]
    assert evidence([0, 0, 1000])["reason"] == "no_postselected_samples"
    assert evidence([4800, 200, 5000])["status"] == "distribution_mismatch"


def test_confidence_and_postselection_are_accounted_for():
    ordinary = evidence([2500, 2500, 5000])
    stronger = evidence([2500, 2500, 5000], confidence=0.999)
    assert (
        stronger["maximum_probability_error_bound"]
        > ordinary["maximum_probability_error_bound"]
    )
    assert ordinary["accepted_shots"] == 5000
    assert ordinary["statevector_validated"] is False
    rare = evidence([2500, 2500, 99995000])
    assert rare["status"] == "insufficient_shots"
    # Zero and one observed frequencies still have nonzero interval widths.
    boundary = evidence([10000, 0, 0], reference=(1.0, 0.0))
    assert boundary["probability_intervals"][1][1] > 0
    assert boundary["probability_intervals"][0][0] < 1


@pytest.mark.parametrize(
    "probabilities,success",
    [
        ([np.nan, 0, 1], 0),
        ([-0.1, 0.1, 1], 0),
        ([0.2, 0.2, 0.2], 0.4),
        ([0.333, 0.333, 0.334], 0.666),
        ([0.25, 0.25, 0.5], 0.9),
    ],
)
def test_invalid_evidence_cannot_accept(probabilities, success):
    execution = SimpleNamespace(
        succeeded=True,
        probabilities=probabilities,
        shots=100,
        logical_success_probability=success,
    )
    assert (
        probability_acceptance(execution, np.ones(2), tolerance=0.05, confidence=0.95)[
            "status"
        ]
        == "invalid_evidence"
    )


@pytest.mark.parametrize(
    "setting,value",
    [
        ("sampling_tolerance", 0),
        ("sampling_tolerance", np.nan),
        ("sampling_confidence", 1),
        ("sampling_confidence", -1),
    ],
)
def test_invalid_sampling_contract_rejected_before_execution(setting, value):
    with pytest.raises(ValueError, match=setting):
        poisson_qsvt_workflow(4, **{setting: value})


def test_real_shot_hamiltonian_acceptance_and_statevector_boundary():
    result = hamiltonian_simulation_workflow(
        np.diag([-0.5, 0.5]),
        np.array([1.0, 0.0]),
        time=0.3,
        degree=5,
        shots=10000,
        acceptance_tolerance=0.001,
    )
    report = result.as_report()["acceptance"]
    assert report["schema_version"] == "1.2"
    assert report["scope"] == "finite_shot_probabilities"
    assert report["accepted_for_stated_scope"] is True
    assert report["full_qsvt_acceptance"] is False
    assert result.qsvt_execution.logical_output is None
    # Insufficient sampling cannot be fixed by successful synthesis or resources.
    execution = replace(
        result.qsvt_execution,
        shots=4,
        probabilities=np.array([0.25, 0.0, 0.75, 0.0]),
        logical_success_probability=0.25,
    )
    low = replace(result, qsvt_execution=execution).as_report()["acceptance"]
    assert low["accepted_for_stated_scope"] is False
    assert low["sampling"]["status"] == "insufficient_shots"


@pytest.mark.parametrize("workflow", ["poisson", "spectral_filter"])
def test_real_shot_eigenstate_clients_can_earn_probability_acceptance(workflow):
    import pennylane as qml

    from qsvt.flagship import spectral_filter_qsvt_workflow

    if workflow == "poisson":
        result = poisson_qsvt_workflow(
            4,
            tolerance=0.4,
            min_degree=5,
            max_degree=5,
            num_points=401,
            shots=10000,
        )
    else:
        result = spectral_filter_qsvt_workflow(
            qml.dot([0.4, 0.3], [qml.Z(0), qml.Z(1)]),
            np.array([0.0, 1.0, 0.0, 0.0]),
            lower=-0.4,
            upper=0.4,
            tolerance=0.16,
            min_degree=2,
            max_degree=4,
            num_points=401,
            shots=10000,
        )
    acceptance = result.as_report()["acceptance"]
    assert acceptance["accepted_for_stated_scope"], acceptance
    assert not acceptance["full_qsvt_acceptance"]
    assert result.execution.logical_output is None


def test_sampling_schema_preserves_historical_reports_and_requires_new_evidence():
    from pathlib import Path

    from qsvt.reports import load_report, validate_report_schema

    report = load_report(
        Path(__file__).parent / "fixtures/reports/flagship_acceptance_v1.json"
    )
    historical = dict(report)
    assert validate_report_schema(report, require_schema=True).supported
    assert report == historical
    report.update(schema_version="1.2", scope="finite_shot_probabilities")
    assert not validate_report_schema(report, require_schema=True).supported
    report["sampling"] = evidence([2500, 2500, 5000])
    assert validate_report_schema(report, require_schema=True).supported
    del report["sampling"]["confidence"]
    assert (
        "sampling.confidence"
        in validate_report_schema(report, require_schema=True).missing_fields
    )
