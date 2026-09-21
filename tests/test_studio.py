"""The optional studio must preserve package evidence and exact requests."""

import copy
import inspect
import io
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from studio import adapter
from studio.catalogue import FUNCTIONS, catalogue, validate_request
from studio.plots import render
from studio.records import Store, canonical_json, compare, json_safe, metrics
from studio.server import Studio, handler

from qsvt.stable import design_workflow, report_to_jsonable


def request(workflow="sign", **settings):
    return validate_request(
        {
            "schema_version": "1.1",
            "workflow": workflow,
            "settings": (
                {
                    "degree": 5,
                    "num_points": 101,
                    "bounded_num_points": 201,
                    "attempt_synthesis": False,
                    **settings,
                }
                if workflow == "sign"
                else settings
            ),
        }
    )


@pytest.fixture(scope="module")
def real_reports():
    outputs = {}
    for preset in catalogue()["presets"]:
        if not preset["source"].startswith("examples/"):
            continue
        req = validate_request(
            {
                "schema_version": "1.1",
                "workflow": preset["workflow"],
                "settings": preset["settings"],
            }
        )
        outputs[preset["workflow"]] = (req, json_safe(adapter.execute_request(req)))
    return outputs


def test_adapter_maps_exact_settings_to_public_api(monkeypatch):
    req = request()
    result = Mock()
    result.as_report.return_value = {"mode": "design-workflow"}
    result.resource_report.return_value = {"resources": {"estimate_kind": "proxy"}}
    execute = Mock(return_value=result)
    monkeypatch.setattr(adapter, "design_workflow", execute)
    adapter.execute_request(req)
    kwargs = {
        k: v
        for k, v in req["settings"].items()
        if k in inspect.signature(design_workflow).parameters
    }
    execute.assert_called_once_with("sign", **kwargs)
    result.synthesize.assert_not_called()


def test_design_result_matches_direct_package_call():
    req = request()
    actual = adapter.execute_request(req)
    kwargs = {
        k: v
        for k, v in req["settings"].items()
        if k in inspect.signature(design_workflow).parameters
    }
    direct = design_workflow("sign", **kwargs).as_report()
    for key in direct:
        assert canonical_json({key: actual[key]}) == canonical_json({key: direct[key]})


@pytest.mark.parametrize("workflow", ["inverse", "filter", "interval_projector"])
def test_other_catalogue_designs_use_real_reports(workflow):
    req = request(
        workflow, attempt_synthesis=False, num_points=101, bounded_num_points=201
    )
    report = adapter.execute_request(req)
    assert report["kind"] == workflow
    assert len(render([{"id": "test", "report": json_safe(report)}])) > 1000


@pytest.mark.parametrize(
    "raw",
    [
        [],
        {},
        {"schema_version": "2", "workflow": "sign"},
        {"schema_version": "1.1", "workflow": "missing"},
        {"schema_version": "1.1", "workflow": []},
        {"schema_version": "1.1", "workflow": "sign", "unknown": 2},
    ],
)
def test_invalid_request_envelope(raw):
    with pytest.raises(ValueError):
        validate_request(raw)


@pytest.mark.parametrize(
    "settings",
    [
        {"degree": True},
        {"degree": 4},
        {"degree": 3.5},
        {"degree": 9999},
        {"gamma": float("nan")},
        {"gamma": float("inf")},
        {"gamma": 0},
        {"attempt_synthesis": "false"},
        {"shots": 20},
        {"num_points": -1},
    ],
)
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        request(**settings)


def test_relational_and_enum_validation():
    for workflow, settings in [
        ("poisson", {"min_degree": 9, "max_degree": 3}),
        ("poisson", {"n_points": True}),
        ("poisson", {"device_name": "remote"}),
        ("spectral_filter", {"lower": 0.5, "upper": -0.5}),
    ]:
        with pytest.raises(ValueError):
            request(workflow, **settings)


def test_package_and_catalogue_defaults_agree():
    for entry in catalogue()["workflows"].values():
        params = inspect.signature(FUNCTIONS[entry["api"]]).parameters
        resolved = validate_request({"schema_version": "1.1", "workflow": entry["id"]})
        for key, spec in entry["settings"].items():
            assert resolved["settings"][key] == spec["default"]
            if spec["default_source"] == "package signature":
                assert spec["default"] == (
                    list(params[key].default)
                    if isinstance(params[key].default, tuple)
                    else params[key].default
                )


def test_serialization_is_deterministic_strict_json_and_complex_safe():
    payload = {"z": np.array([1 + 2j]), "a": np.float64(0.3), "bad": float("inf")}
    assert canonical_json(payload) == canonical_json(
        dict(reversed(list(payload.items())))
    )
    decoded = json.loads(
        canonical_json(payload),
        parse_constant=lambda _: pytest.fail("Nonstandard JSON"),
    )
    assert decoded["z"] == {"real": [1.0], "imag": [2.0]}
    assert decoded["bad"] == {"__nonfinite__": "inf"}


def save_run(store, req, report):
    run_id = store.create(req, {"test": True})
    store.write(store.directory(run_id) / "report.json", report)
    store.update(run_id, status="completed", metrics=metrics(report))
    return run_id


def test_reuse_and_favorites_survive_reopening(tmp_path):
    store = Store(tmp_path)
    req = request()
    run_id = store.create(req, {})
    store.favorite(run_id, True)
    restored = Store(tmp_path).reuse(run_id)
    assert restored == req
    restored["settings"]["degree"] = 9
    assert store.reuse(run_id) == req
    store.recover()
    assert store.read(run_id)["favorite"] is True
    assert store.read(run_id)["status"] == "failed"
    assert store.reuse(run_id) == req


@pytest.mark.parametrize(
    "filename, damage",
    [
        ("record.json", b"{"),
        ("record.json", b"[]"),
        ("record.json", b"{}"),
        ("record.json", b"\xff"),
        ("record.json", None),
        ("request.json", b"{"),
        ("request.json", b"{}"),
        ("request.json", None),
    ],
)
def test_damaged_run_does_not_block_startup_or_history(
    tmp_path, caplog, filename, damage
):
    store = Store(tmp_path)
    good = store.create(request(), {})
    bad = store.create(request(), {})
    path = store.directory(bad) / filename
    original = path.read_bytes()
    if damage is None:
        path.unlink()
    else:
        path.write_bytes(damage)

    studio = Studio(tmp_path)
    try:
        assert studio.store.read(good)["status"] == "failed"
        status, body = http(studio, "GET", "/api/runs")
        assert status == 200
        assert [run["id"] for run in json.loads(body)["runs"]] == [good]
        assert bad in caplog.text
        assert caplog.text.count("Skipping unreadable saved run") == 1
        assert (path.read_bytes() if path.exists() else None) == damage

        # Repairing the file restores visibility without restarting or rewriting it.
        path.write_bytes(original)
        assert {run["id"] for run in studio.store.history()} == {good, bad}
        assert path.read_bytes() == original
    finally:
        studio.close()


def test_comparison_uses_stored_evidence_and_rejects_different_targets(
    tmp_path, real_reports
):
    store = Store(tmp_path)
    req, report = real_reports["sign"]
    a = save_run(store, req, report)
    other = copy.deepcopy(req)
    other["settings"]["degree"] = 9
    second_report = json_safe(adapter.execute_request(other))
    b = save_run(store, other, second_report)
    runs = compare(store, [a, b])
    assert (
        metrics(runs[1]["report"])["diagnostics.max_error"]
        == second_report["diagnostics"]["max_error"]
    )
    assert render(runs).startswith(b"\x89PNG")
    other["settings"]["gamma"] = 0.4
    c = save_run(store, other, second_report)
    with pytest.raises(ValueError, match="Incompatible"):
        compare(store, [a, c])
    for ids in ([a], [a, a], [a, store.create(req, {})]):
        with pytest.raises(ValueError):
            compare(store, ids)


@pytest.mark.parametrize("workflow", ["poisson", "spectral_filter"])
def test_flagship_evidence_preserved_in_storage_and_plots(
    tmp_path, real_reports, workflow
):
    req, report = real_reports[workflow]
    assert report["execution"]["succeeded"] is True
    assert report["acceptance"]["full_qsvt_acceptance"] is True
    store = Store(tmp_path)
    run_id = save_run(store, req, report)
    stored = store.read(run_id, report=True)
    for key in ("acceptance", "truth_contract", "resources", "execution", "synthesis"):
        assert stored["report"][key] == report_to_jsonable(report)[key]
    assert render([stored]).startswith(b"\x89PNG")


def test_worker_lifecycle_failure_and_retry(tmp_path, monkeypatch):
    studio = Studio(tmp_path)
    try:
        run_id = studio.store.create(request(), {})
        studio.run(run_id)
        run = studio.store.read(run_id, report=True)
        assert [e["status"] for e in run["events"]] == [
            "configured",
            "validating",
            "executing",
            "saving_artifacts",
            "completed",
        ]
        assert "preview.png" in run["artifacts"]
        monkeypatch.setattr(
            "studio.server.execute_request", Mock(side_effect=ValueError("real error"))
        )
        failed = studio.store.create(request(), {})
        studio.run(failed)
        assert studio.store.read(failed)["error"]["message"] == "real error"
        assert studio.store.reuse(failed) == request()
    finally:
        studio.close()


class Socket:
    """Exercise the real HTTP handler without opening network ports."""

    def __init__(self, wire):
        self.wire = io.BytesIO(wire)
        self.output = io.BytesIO()

    def makefile(self, *_args):
        return self.wire

    def sendall(self, data):
        self.output.write(data)


def http(studio, method, path, body=None, *, protected=True):
    payload = b"" if body is None else json.dumps(body).encode()
    headers = (
        f"{method} {path} HTTP/1.0\r\n"
        f"Content-Length: {len(payload)}\r\n"
        "Content-Type: application/json\r\n"
    )
    if protected:
        headers += "X-QSVT-Studio: 1\r\n"
    socket = Socket(headers.encode() + b"\r\n" + payload)
    handler(studio)(socket, ("127.0.0.1", 1234), None)
    head, content = socket.output.getvalue().split(b"\r\n\r\n", 1)
    return int(head.split()[1]), content


def test_http_api_validation_artifacts_and_mutations(tmp_path, real_reports):
    studio = Studio(tmp_path)
    try:
        status, body = http(studio, "GET", "/api/catalogue")
        assert status == 200 and len(json.loads(body)["workflows"]) == 7
        assert http(studio, "GET", "/")[0] == 200
        assert http(studio, "GET", "/../../pyproject.toml")[0] == 404
        assert http(studio, "GET", "/api/runs/not-an-id")[0] == 400
        assert http(studio, "POST", "/api/runs", request(), protected=False)[0] == 403
        assert (
            http(studio, "POST", "/api/validate", {"schema_version": "bad"})[0] == 400
        )
        run_id = save_run(studio.store, *real_reports["sign"])
        status, body = http(studio, "GET", f"/api/runs/{run_id}/reuse")
        assert status == 200 and json.loads(body) == real_reports["sign"][0]
        assert (
            http(studio, "POST", f"/api/runs/{run_id}/favorite", {"favorite": True})[0]
            == 200
        )
        assert studio.store.read(run_id)["favorite"]
        assert http(studio, "GET", f"/api/runs/{run_id}/report.json")[0] == 200
        assert http(studio, "POST", "/api/compare", {"ids": [run_id]})[0] == 400
        assert http(studio, "POST", "/api/runs", request())[0] == 202
    finally:
        studio.close()


def test_studio_is_not_in_distribution_and_has_no_frontend_science():
    pyproject = Path("pyproject.toml").read_text()
    assert 'where = ["src"]' in pyproject
    js = Path("studio/static/app.js").read_text()
    assert "innerHTML" not in js
    assert "Math.pow" not in js


def test_hamiltonian_adapter_uses_published_problem_and_public_api(monkeypatch):
    from qsvt.hamiltonians import tight_binding_chain

    req = request("hamiltonian_simulation", execute_qsvt=False)
    result = Mock()
    result.as_report.return_value = {"mode": "hamiltonian-simulation-workflow"}
    execute = Mock(return_value=result)
    monkeypatch.setattr(adapter, "hamiltonian_simulation_workflow", execute)
    assert adapter.execute_request(req) == result.as_report.return_value
    matrix, state = execute.call_args.args
    np.testing.assert_array_equal(matrix, tight_binding_chain(6))
    np.testing.assert_array_equal(state, [0, 1, 0, 0, 0, 0])
    assert execute.call_args.kwargs == req["settings"]


def test_hamiltonian_cookbook_acceptance_and_report_preservation(
    tmp_path, real_reports
):
    from examples.hamiltonian_simulation import build_report

    req, report = real_reports["hamiltonian_simulation"]
    direct = json_safe(build_report())
    for key in (
        "acceptance",
        "truth_contract",
        "cos_coeffs",
        "sin_coeffs",
        "evolved_state",
        "reference_state",
        "component_error_ledger",
        "circuit_resource_ledger",
    ):
        assert report[key] == direct[key]
    assert report["qsvt_execution"]["succeeded"] is True
    assert report["acceptance"]["full_qsvt_acceptance"] is True
    assert set(report["qsvt_execution"]["components"]) == {"cosine", "sine"}
    store = Store(tmp_path)
    run_id = save_run(store, req, report)
    stored = store.read(run_id, report=True)
    assert stored["report"] == report
    assert store.reuse(run_id) == req
    assert render([stored]).startswith(b"\x89PNG")
    assert (
        metrics(report)["qsvt_execution.logical_success_probability"]
        == report["qsvt_execution"]["logical_success_probability"]
    )


def test_hamiltonian_comparison_preserves_time_and_execution_boundaries(
    tmp_path, real_reports
):
    req, report = real_reports["hamiltonian_simulation"]
    polynomial_request = request(
        "hamiltonian_simulation", degree=2, execute_qsvt=False, num_points=401
    )
    polynomial = json_safe(adapter.execute_request(polynomial_request))
    assert polynomial["qsvt_execution"] is None
    assert polynomial["circuit_resource_ledger"] is None
    assert polynomial["acceptance"]["full_qsvt_acceptance"] is False
    checks = {c["id"]: c for c in polynomial["acceptance"]["checks"]}
    assert checks["polynomial_accuracy"]["passed"] is False
    assert (
        checks["polynomial_accuracy"]["threshold"]
        == polynomial_request["settings"]["acceptance_tolerance"]
    )
    store = Store(tmp_path)
    a = save_run(store, req, report)
    b = save_run(store, polynomial_request, polynomial)
    assert render(compare(store, [a, b])).startswith(b"\x89PNG")
    assert "qsvt_execution.logical_success_probability" not in metrics(polynomial)
    changed_time = copy.deepcopy(polynomial_request)
    changed_time["settings"]["time"] = 0.5
    c = save_run(store, changed_time, polynomial)
    with pytest.raises(ValueError, match="Incompatible"):
        compare(store, [a, c])


@pytest.mark.parametrize(
    "settings",
    [
        {"execute": True},
        {"time": float("nan")},
        {"degree": 0},
        {"execute_qsvt": "false"},
        {"block_encoding": "fable"},
        {"acceptance_tolerance": 0},
    ],
)
def test_hamiltonian_rejects_unsupported_settings(settings):
    with pytest.raises(ValueError):
        request("hamiltonian_simulation", **settings)


@pytest.mark.parametrize("preset", catalogue()["presets"], ids=lambda p: p["id"])
def test_presets_are_complete_and_meet_their_declared_outcome(preset):
    fields = catalogue()["workflows"][preset["workflow"]]["settings"]
    assert set(preset["settings"]) == set(fields)
    raw = {
        "schema_version": "1.1",
        "workflow": preset["workflow"],
        "settings": preset["settings"],
    }
    assert validate_request(raw) == raw
    report = adapter.execute_request(raw)
    expected = preset["expected"]
    if expected == "finite_qsvt":
        assert report["acceptance"]["full_qsvt_acceptance"] is True
    elif expected == "phases":
        assert report["synthesis_quality"]["status"] == "passed"
        assert (
            report["compatibility"]["angle_solver"]
            == preset["settings"]["angle_solver"]
        )
        assert report["synthesis"]["angle_solver"] == preset["settings"]["angle_solver"]
    elif expected == "synthesis_failure":
        assert report["synthesis_quality"]["status"] == "solver_failed"
        assert report["synthesis"]["error_type"] == "PolynomialRealizabilityError"
    elif expected == "acceptance_failure":
        assert report["acceptance"]["accepted_for_stated_scope"] is False
    elif expected == "polynomial":
        assert report["qsvt_execution"] is None
    else:
        assert report["compatibility"]["is_bounded"] is True
        assert "synthesis" not in report


def test_recommendations_do_not_replace_api_defaults():
    data = catalogue()
    for workflow, entry in data["workflows"].items():
        choices = [
            p for p in data["presets"] if p["workflow"] == workflow and p["recommended"]
        ]
        assert len(choices) == 1
        assert choices[0]["id"] == entry["recommended_preset"]
    default = validate_request({"schema_version": "1.1", "workflow": "sign"})
    recommended = next(p for p in data["presets"] if p["id"] == "sign-cookbook")
    assert default["settings"]["attempt_synthesis"] is True
    assert recommended["settings"]["attempt_synthesis"] is False


@pytest.mark.parametrize("workflow", list(catalogue()["workflows"]))
def test_legacy_configuration_upgrade_preserves_saved_values_and_reports(
    tmp_path, workflow
):
    resolved = request(workflow)
    settings = copy.deepcopy(resolved["settings"])
    additions = {"angle_solver", "angle_solvers"}
    if catalogue()["workflows"][workflow]["api"] == "design":
        additions |= {"reconstruction_num_points", "phase_reconstruction_tolerance"}
    for key in additions:
        settings.pop(key, None)
    legacy = {"schema_version": "1.0", "workflow": workflow, "settings": settings}
    original = copy.deepcopy(legacy)
    store = Store(tmp_path)
    run_id = save_run(store, legacy, {"historical": True})
    upgraded = validate_request(store.reuse(run_id))
    assert upgraded["schema_version"] == "1.1"
    assert all(upgraded["settings"][k] == v for k, v in settings.items())
    assert upgraded == resolved
    assert legacy == original
    assert store.reuse(run_id) == original
    assert store.read(run_id, report=True)["report"] == {"historical": True}


@pytest.mark.parametrize(
    "settings",
    [
        {"angle_solver": "iterative-optax"},
        {"reconstruction_num_points": 2},
        {"phase_reconstruction_tolerance": float("nan")},
    ],
)
def test_design_rejects_unsupported_solver_settings(settings):
    with pytest.raises(ValueError):
        request(**settings)


@pytest.mark.parametrize(
    "solvers", [["iterative"], ["root-finding"], ["root-finding", "iterative"]]
)
def test_flagship_solver_choices_reach_the_package(monkeypatch, solvers):
    result = Mock()
    result.as_report.return_value = {}
    result.synthesis.quality_report.return_value = {"status": "passed"}
    execute = Mock(return_value=result)
    monkeypatch.setattr(adapter, "poisson_qsvt_workflow", execute)
    adapter.execute_request(request("poisson", angle_solvers=solvers))
    assert execute.call_args.kwargs["angle_solvers"] == tuple(solvers)


@pytest.mark.parametrize(
    "workflow", ["poisson", "spectral_filter", "hamiltonian_simulation"]
)
def test_iterative_flagship_method_is_validated_on_the_same_problem(workflow):
    preset = next(
        p
        for p in catalogue()["presets"]
        if p["workflow"] == workflow and p["recommended"]
    )
    settings = copy.deepcopy(preset["settings"])
    key = "angle_solver" if workflow == "hamiltonian_simulation" else "angle_solvers"
    settings[key] = "iterative" if key == "angle_solver" else ["iterative"]
    report = adapter.execute_request(
        {"schema_version": "1.1", "workflow": workflow, "settings": settings}
    )
    assert report["acceptance"]["full_qsvt_acceptance"] is True
    qualities = report.get(
        "component_synthesis_quality", {"single": report.get("synthesis_quality")}
    )
    assert qualities
    for quality in qualities.values():
        assert quality["angle_solver"] == "iterative"
        assert quality["status"] == "passed"
