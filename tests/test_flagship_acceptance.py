from pathlib import Path

import numpy as np
import pennylane as qml
import pytest

from qsvt.acceptance import (
    FLAGSHIP_ACCEPTANCE_SCHEMA_NAME,
    evaluate_hamiltonian_simulation_acceptance,
    flagship_acceptance_matrix,
)
from qsvt.algorithms import hamiltonian_simulation_workflow
from qsvt.flagship import poisson_qsvt_workflow, spectral_filter_qsvt_workflow
from qsvt.reports import load_report_with_schema, validate_report_schema

REPO_ROOT = Path(__file__).resolve().parents[1]

# Each row corresponds to an advertised high-level path in flagship_workflows.md.
FLAGSHIP_ENCODINGS = [
    ("poisson", "dense"),
    ("poisson", "fable"),
    ("poisson", "prepselprep"),
    ("poisson", "qubitization"),
    ("filter", "prepselprep"),
    ("filter", "qubitization"),
    ("hamiltonian", "embedding"),
    ("hamiltonian", "fable"),
]


@pytest.mark.parametrize("workflow,encoding", FLAGSHIP_ENCODINGS)
@pytest.mark.parametrize("shots", [None, 256], ids=["statevector", "finite-shots"])
def test_flagship_encoding_support_matrix(workflow, encoding, shots):
    if workflow == "poisson":
        # FABLE's larger alpha requires a different inverse approximation.
        degree = 13 if encoding == "fable" else 5
        result = poisson_qsvt_workflow(
            4,
            tolerance=0.4,
            min_degree=degree,
            max_degree=degree,
            num_points=401,
            access_model=encoding,
            shots=shots,
        )
        execution = result.execution
    elif workflow == "filter":
        operator = qml.dot([0.4, 0.3, 0.2], [qml.Z("a"), qml.Z("b"), qml.X("a")])
        result = spectral_filter_qsvt_workflow(
            operator,
            np.ones(4) / 2,
            lower=-0.4,
            upper=0.4,
            tolerance=0.16,
            min_degree=2,
            max_degree=4,
            num_points=401,
            block_encoding=encoding,
            encoding_wires=["aux0", "aux1"],
            shots=shots,
        )
        execution = result.execution
    else:
        matrix = np.array([[-0.8, 0.2], [0.2, 0.5]])
        result = hamiltonian_simulation_workflow(
            matrix,
            np.array([1.0, 1.0j]),
            time=0.4,
            degree=9,
            num_points=201,
            block_encoding=encoding,
            shots=shots,
        )
        execution = result.qsvt_execution
        scaled = result.scaled_operator
        np.testing.assert_allclose(
            scaled.scale * scaled.matrix + scaled.offset * np.eye(2), matrix
        )
        if encoding == "fable":
            assert 2 * np.max(np.abs(scaled.matrix)) <= 1 + 1e-12

    assert execution is not None and execution.succeeded, execution
    acceptance = result.as_report()["acceptance"]
    assert validate_report_schema(acceptance, require_schema=True).supported
    if shots is None:
        assert execution.logical_output_relative_error < 1e-6
        assert acceptance["accepted_for_stated_scope"] is True
        assert acceptance["full_qsvt_acceptance"] is True
    else:
        # Shot acceptance is scoped to measured probabilities, never statevectors.
        assert acceptance["scope"] == "finite_shot_probabilities"
        assert acceptance["sampling"]["shots"] == shots
        assert execution.final_state is None
        assert execution.logical_output is None
        assert execution.logical_output_relative_error is None
        assert execution.maximum_probability_standard_error is not None
        assert acceptance["full_qsvt_acceptance"] is False
        checks = {check["id"]: check for check in acceptance["checks"]}
        assert checks["finite_qsvt_execution"]["passed"] is False


@pytest.mark.parametrize("access_model", ["prepselprep", "qubitization"])
def test_poisson_pauli_access_rejects_non_power_of_two_dimensions(access_model):
    with pytest.raises(ValueError, match="power-of-two"):
        poisson_qsvt_workflow(3, access_model=access_model)


def test_flagship_rejects_unsupported_access_and_wire_contracts():
    with pytest.raises(ValueError, match="access_model must"):
        poisson_qsvt_workflow(4, access_model="custom")
    with pytest.raises(ValueError, match="block_encoding must"):
        hamiltonian_simulation_workflow(
            np.diag([-1.0, 1.0]),
            np.ones(2),
            time=0.4,
            degree=9,
            block_encoding="prepselprep",
        )
    with pytest.raises(ValueError, match="distinct"):
        spectral_filter_qsvt_workflow(
            qml.Z("a"),
            [1.0, 0.0],
            lower=-0.4,
            upper=0.4,
            encoding_wires=["aux", "aux"],
        )


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
    assert acceptance["schema_version"] == "1.2"
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


@pytest.mark.parametrize("error", [float("nan"), float("inf"), 0.075])
def test_poisson_never_executes_unvalidated_solver_output(monkeypatch, error):
    from dataclasses import replace

    from qsvt.synthesis import synthesize_phases

    original = synthesize_phases([0.0, 0.5])
    inaccurate = replace(original, reconstruction_max_error=error)
    monkeypatch.setattr(
        "qsvt.flagship.synthesize_phases_cached", lambda *a, **k: inaccurate
    )
    with pytest.raises(ValueError, match="phase_reconstruction_tolerance"):
        poisson_qsvt_workflow(4, min_degree=5, max_degree=5, num_points=401)


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), 0.0])
def test_poisson_rejects_invalid_phase_tolerance(tolerance):
    with pytest.raises(ValueError, match="positive and finite"):
        poisson_qsvt_workflow(
            4,
            min_degree=5,
            max_degree=5,
            num_points=401,
            phase_reconstruction_tolerance=tolerance,
        )


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"block_encoding": "embedding"}, "block_encoding must"),
        ({"encoding_wires": ["a"]}, "disjoint"),
    ],
)
def test_filter_rejects_unsupported_encoding_contract_before_design(kwargs, message):
    with pytest.raises(ValueError, match=message):
        spectral_filter_qsvt_workflow(
            qml.Z("a"), [1.0, 0.0], lower=-0.4, upper=0.4, **kwargs
        )
