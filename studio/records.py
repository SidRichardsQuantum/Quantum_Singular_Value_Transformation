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

TERMINAL = {"completed", "failed", "cancelled"}
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
                    "cancelling",
                    "cancelled",
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

    def history_page(self, query):
        """Filter before paging so searches include every saved experiment."""
        limit = int(query.get("limit", "24"))
        offset = int(query.get("offset", "0"))
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("History limit must be 1–100; offset must be nonnegative.")
        records = self.history()
        total = len(records)
        active = sum(r["status"] not in TERMINAL for r in records)
        workflows = catalogue()["workflows"]

        def matches(record):
            request = record["request"]
            settings = request["settings"]
            execution_key = workflows[request["workflow"]].get("execution_setting")
            execution = str(bool(settings.get(execution_key))).lower()
            status = query.get("status")
            return (
                (
                    not query.get("search")
                    or query["search"].lower() in json.dumps(record).lower()
                )
                and (
                    not query.get("workflow")
                    or request["workflow"] == query["workflow"]
                )
                and (
                    not status
                    or (
                        record["status"] not in TERMINAL
                        if status == "active"
                        else record["status"] == status
                    )
                )
                and (
                    not query.get("encoding")
                    or (settings.get("access_model") or settings.get("block_encoding"))
                    == query["encoding"]
                )
                and (not query.get("execution") or execution == query["execution"])
                and (query.get("favorites") != "true" or record["favorite"])
            )

        records = [record for record in records if matches(record)]
        if query.get("sort") == "oldest":
            records.reverse()
        # Keep the last page usable if a filter or favorite mutation shrinks it.
        offset = min(offset, ((len(records) - 1) // limit) * limit) if records else 0
        return {
            "runs": records[offset : offset + limit],
            "total": total,
            "matching": len(records),
            "active": active,
            "offset": offset,
            "limit": limit,
        }

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
        "acceptance.sampling.status",
        "acceptance.sampling.confidence",
        "acceptance.sampling.tolerance",
        "acceptance.sampling.accepted_shots",
        "acceptance.sampling.maximum_probability_error",
        "acceptance.sampling.maximum_probability_error_bound",
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
        request = validate_request(run["request"])
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


def failure_diagnosis(run):
    """Expose saved evidence without inferring stages from exception text."""
    report = run.get("report") or (run.get("error") or {}).get("evidence") or {}
    findings = []
    if run.get("error"):
        findings.append(
            {"stage": "package_call", "source": "error", "evidence": run["error"]}
        )
    syntheses = [
        ("synthesis", report.get("synthesis"), report.get("synthesis_quality"))
    ]
    for name, component in (
        (report.get("qsvt_execution") or {}).get("components", {}).items()
    ):
        syntheses.append(
            (
                f"qsvt_execution.components.{name}.synthesis",
                component.get("synthesis"),
                report.get("component_synthesis_quality", {}).get(name),
            )
        )
    for source, synthesis, quality in syntheses:
        if not synthesis:
            continue
        # Keep every saved attempt, including failures before successful fallback.
        attempts = synthesis.get("attempts") or [synthesis]
        for index, attempt in enumerate(attempts):
            assessment = (
                attempt.get("quality") or (quality if len(attempts) == 1 else {}) or {}
            )
            failed = (
                attempt.get("succeeded") is False
                or assessment.get("reconstruction_passed") is False
            )
            findings.append(
                {
                    "stage": assessment.get("failure_stage")
                    or attempt.get("failure_stage")
                    or ("synthesis" if failed else "synthesis_attempt"),
                    "source": source,
                    "attempt": index + 1,
                    "failed": failed,
                    "evidence": {"synthesis": attempt, "quality": assessment},
                }
            )
    for key in ("execution", "qsvt_execution"):
        execution = report.get(key) or {}
        if execution.get("succeeded") is False:
            findings.append(
                {
                    "stage": "execution",
                    "source": key,
                    "failed": True,
                    "evidence": execution,
                }
            )
    acceptance = report.get("acceptance") or {}
    for check in acceptance.get("checks", []):
        if check.get("passed") is False and check.get("required_for_scope"):
            findings.append(
                {
                    "stage": (
                        "sampling"
                        if check["id"] == "finite_shot_probability_accuracy"
                        else "acceptance"
                    ),
                    "source": f"acceptance.checks.{check['id']}",
                    "failed": True,
                    "evidence": check,
                }
            )
    return findings


def comparison_report(runs):
    """Compare original saved fields; missing differs from explicit null."""

    def flatten(value, path=""):
        if isinstance(value, dict) and value:
            return {
                key: item
                for name, child in value.items()
                for key, item in flatten(
                    child, f"{path}.{name}" if path else name
                ).items()
            }
        return {path: value}

    result: dict[str, Any] = {
        "runs": [
            {
                "id": run["id"],
                "request": run["request"],
                "metrics": metrics(run["report"]),
                "package_call_seconds": run.get("package_call_seconds"),
                "diagnosis": failure_diagnosis(run),
            }
            for run in runs
        ]
    }
    differences = {}
    for section in ("request", "report"):
        values = [flatten(run[section]) for run in runs]
        rows = []
        for path in sorted(set().union(*(value.keys() for value in values))):
            cells = [
                {"present": path in value, "value": value.get(path)} for value in values
            ]
            rows.append(
                {
                    "path": path,
                    "different": any(cell != cells[0] for cell in cells[1:]),
                    "cells": cells,
                }
            )
        differences[section] = rows
    result["differences"] = differences
    return result
