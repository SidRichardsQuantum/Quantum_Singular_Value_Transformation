from pathlib import Path

import numpy as np

from qsvt.acceptance import (
    FLAGSHIP_ACCEPTANCE_SCHEMA_NAME,
    evaluate_hamiltonian_simulation_acceptance,
    flagship_acceptance_matrix,
)
from qsvt.algorithms import hamiltonian_simulation_workflow
from qsvt.reports import load_report_with_schema, validate_report_schema

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_flagship_acceptance_matrix_has_three_explicit_scopes():
    matrix = flagship_acceptance_matrix()

    assert set(matrix) == {
        "poisson_qsvt",
        "spectral_filter_qsvt",
        "hamiltonian_simulation",
    }
    assert matrix["poisson_qsvt"]["scope"] == "finite_qsvt"
    assert matrix["spectral_filter_qsvt"]["scope"] == "finite_qsvt"
    assert matrix["hamiltonian_simulation"]["scope"] == "finite_qsvt"
    assert {
        criterion["id"]
        for criterion in matrix["hamiltonian_simulation"]["criteria"]
        if criterion["required_for_full_qsvt"]
    } >= {"finite_qsvt_execution", "diagnostics_and_resources"}


def test_committed_flagship_acceptance_fixture_loads_through_schema_registry():
    report, compatibility = load_report_with_schema(
        REPO_ROOT / "tests/fixtures/reports/flagship_acceptance_v1.json",
        expected_schema_name=FLAGSHIP_ACCEPTANCE_SCHEMA_NAME,
        expected_schema_version="1.0",
    )

    assert compatibility.supported is True
    assert report["workflow"] == "hamiltonian_simulation"
    assert report["full_qsvt_acceptance"] is False


def test_hamiltonian_acceptance_is_schema_valid_and_accepts_coherent_execution():
    result = hamiltonian_simulation_workflow(
        np.diag([-1.0, 0.5]),
        np.array([1.0, 1.0]),
        time=0.4,
        degree=9,
        num_points=201,
    )
    acceptance = evaluate_hamiltonian_simulation_acceptance(result)
    compatibility = validate_report_schema(acceptance, require_schema=True)
    checks = {check["id"]: check for check in acceptance["checks"]}

    assert compatibility.supported is True
    assert acceptance["schema_name"] == FLAGSHIP_ACCEPTANCE_SCHEMA_NAME
    assert acceptance["accepted_for_stated_scope"] is True
    assert acceptance["schema_version"] == "1.1"
    assert acceptance["full_qsvt_acceptance"] is True
    assert checks["finite_qsvt_execution"]["required_for_scope"] is True
    assert checks["finite_qsvt_execution"]["passed"] is True
    assert checks["diagnostics_and_resources"]["passed"] is True


def test_hamiltonian_acceptance_rejects_insufficient_degree_for_declared_tolerance():
    result = hamiltonian_simulation_workflow(
        np.diag([-1.0, 1.0]),
        np.array([1.0, 1.0]),
        time=3.0,
        degree=1,
        num_points=101,
        acceptance_tolerance=1e-8,
    )
    acceptance = result.as_report()["acceptance"]

    assert acceptance["accepted_for_stated_scope"] is False
    assert acceptance["status"] == "acceptance_criteria_not_met"
