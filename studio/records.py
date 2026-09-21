"""Versioned file records over authoritative package reports; no database."""

from __future__ import annotations

import copy
import json
import logging
import math
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qsvt.stable import report_to_jsonable

from .catalogue import SCHEMA_VERSION, catalogue, validate_request

TERMINAL = {"completed", "failed"}
LOGGER = logging.getLogger(__name__)


def json_safe(value: dict[str, Any]) -> dict[str, Any]:
    """Keep complex encoding from qsvt; explicitly tag non-finite IEEE values."""

    def clean(item):
        if isinstance(item, float) and not math.isfinite(item):
            return {"__nonfinite__": str(item)}
        if isinstance(item, dict):
            return {k: clean(v) for k, v in item.items()}
        if isinstance(item, list):
            return [clean(v) for v in item]
        return item

    return clean(report_to_jsonable(value))


def canonical_json(value: dict[str, Any]) -> str:
    return (
        json.dumps(json_safe(value), sort_keys=True, indent=2, allow_nan=False) + "\n"
    )


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self._unreadable: set[str] = set()

    def directory(self, run_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise ValueError("Invalid run ID.")
        path = self.root / run_id
        if path.is_symlink():
            raise ValueError("Symlink run directories are not supported.")
        return path

    def write(self, path: Path, value: dict[str, Any]):
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(canonical_json(value), encoding="utf-8")
        temporary.replace(path)

    def create(self, request, environment):
        run_id = uuid.uuid4().hex
        with self.lock:
            directory = self.directory(run_id)
            directory.mkdir()
            self.write(directory / "request.json", request)
            record = {
                "schema_version": SCHEMA_VERSION,
                "id": run_id,
                "created_at": now(),
                "status": "configured",
                "favorite": False,
                "events": [{"status": "configured", "at": now()}],
                "environment": environment,
                "artifacts": [],
            }
            self.write(directory / "record.json", record)
        return run_id

    def read(self, run_id: str, *, report=False):
        directory = self.directory(run_id)
        with self.lock:
            record = json.loads((directory / "record.json").read_text())
            if (
                not isinstance(record, dict)
                or record.get("id") != run_id
                or not isinstance(record.get("created_at"), str)
                or not isinstance(record.get("status"), str)
                or record.get("status")
                not in {
                    "configured",
                    "validating",
                    "executing",
                    "saving_artifacts",
                    "completed",
                    "failed",
                }
                or type(record.get("favorite")) is not bool
                or not isinstance(record.get("events"), list)
                or not isinstance(record.get("artifacts"), list)
            ):
                raise ValueError(f"Invalid saved record: {run_id}")
            record["request"] = json.loads((directory / "request.json").read_text())
            # Validate without replacing the authoritative historical request.
            validate_request(record["request"])
            if report and (directory / "report.json").exists():
                record["report"] = json.loads((directory / "report.json").read_text())
            return record

    def update(self, run_id, **changes):
        with self.lock:
            record = self.read(run_id)
            record.pop("request")
            if "status" in changes:
                record["events"].append({"status": changes["status"], "at": now()})
            record.update(changes)
            self.write(self.directory(run_id) / "record.json", record)

    def history(self):
        records = []
        with self.lock:
            unreadable = set()
            for path in self.root.iterdir():
                if not re.fullmatch(r"[0-9a-f]{32}", path.name):
                    continue
                try:
                    records.append(self.read(path.name))
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    unreadable.add(path.name)
                    if path.name not in self._unreadable:
                        LOGGER.warning(
                            "Skipping unreadable saved run %s: %s", path, exc
                        )
            self._unreadable = unreadable
        return sorted(records, key=lambda item: item["created_at"], reverse=True)

    def recover(self):
        for record in self.history():
            if record["status"] not in TERMINAL:
                self.update(
                    record["id"],
                    status="failed",
                    error={
                        "type": "InterruptedRun",
                        "message": (
                            "Server stopped before this run finished. Reuse to retry."
                        ),
                    },
                )

    def reuse(self, run_id):
        return copy.deepcopy(self.read(run_id)["request"])

    def favorite(self, run_id, value):
        if type(value) is not bool:
            raise ValueError("favorite must be a boolean.")
        self.update(run_id, favorite=value)


def metrics(report):
    """Named report paths keep empirical, execution and estimate metrics distinct."""
    paths = [
        "synthesis_quality.status",
        "synthesis_quality.solver_returned_phases",
        "synthesis_quality.reconstruction_passed",
        "synthesis_quality.tolerance",
        "synthesis.angle_solver",
        "component_synthesis_quality.cosine.status",
        "component_synthesis_quality.sine.status",
        "component_synthesis_quality.cosine.angle_solver",
        "component_synthesis_quality.sine.angle_solver",
        "time",
        "degree",
        "state_relative_error",
        "operator_relative_error",
        "norm_drift",
        "component_error_ledger.phase_reconstruction_max_error",
        "qsvt_execution.succeeded",
        "qsvt_execution.logical_success_probability",
        "qsvt_execution.selection_success_probability",
        "qsvt_execution.logical_output_relative_error",
        "qsvt_execution.lcu_normalization",
        "circuit_resource_ledger.depth",
        "circuit_resource_ledger.num_gates",
        "circuit_resource_ledger.gate_types",
        "circuit_resource_ledger.total_wire_count",
        "circuit_resource_ledger.total_phase_count",
        "circuit_resource_ledger.selection_ancilla_count",
        "circuit_resource_ledger.total_signal_operator_calls",
        "qsvt_execution.components.cosine.synthesis.phase_count",
        "qsvt_execution.components.cosine.synthesis.reconstruction_max_error",
        "qsvt_execution.components.sine.synthesis.phase_count",
        "qsvt_execution.components.sine.synthesis.reconstruction_max_error",
        "diagnostics.degree",
        "diagnostics.max_error",
        "diagnostics.rms_error",
        "compatibility.parity",
        "compatibility.realizability_kind",
        "degree_search.chosen_polynomial_degree",
        "degree_search.achieved_error",
        "degree_search.metric",
        "degree_search.met_tolerance",
        "synthesis.phase_count",
        "synthesis.reconstruction_max_error",
        "synthesis.succeeded",
        "polynomial_relative_error",
        "polynomial_operator_error",
        "polynomial_success_probability",
        "execution.succeeded",
        "execution.logical_success_probability",
        "execution.logical_output_relative_error",
        "execution.resource_summary.depth",
        "execution.resource_summary.num_gates",
        "execution.resource_summary.gate_types",
        "execution.resource_summary.total_wire_count",
        "execution.resource_summary.encoding_wire_count",
        "execution.resource_summary.signal_operator_calls",
        "resources.normalization_alpha",
        "resources.signal_operator_calls",
        "resources.inverse_signal_operator_calls",
        "resources.total_wires",
        "resources.total_gates",
        "resources.estimator_kind",
        "resource_report.resources.estimate_kind",
        "resource_report.resources.signal_operator_calls",
        "resource_report.resources.qsp_phase_count",
        "acceptance.status",
        "acceptance.scope",
        "acceptance.full_qsvt_acceptance",
    ]
    values = {}
    for path in paths:
        value = report
        for part in path.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if value is not None:
            values[path] = value
    return values


def compare(store: Store, ids):
    if (
        not isinstance(ids, list)
        or not 2 <= len(ids) <= 6
        or any(not isinstance(i, str) for i in ids)
    ):
        raise ValueError("Select 2–6 completed experiments.")
    if len(set(ids)) != len(ids):
        raise ValueError("Select distinct experiments.")
    runs = [store.read(run_id, report=True) for run_id in ids]
    signatures = []
    for run in runs:
        if run["status"] != "completed" or "report" not in run:
            raise ValueError("Only completed runs with reports can be compared.")
        request = run["request"]
        fields = catalogue()["workflows"][request["workflow"]]["comparison_fields"]
        signatures.append(
            (request["workflow"], {k: request["settings"][k] for k in fields})
        )
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError(
            "Incompatible target/problem: match workflow and "
            "target settings before comparing."
        )
    return runs
