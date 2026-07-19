"""Declarative, resumable experiment sweeps for QSVT research studies."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .reports import load_report, report_to_jsonable, save_report

ResearchEvaluator = Callable[["ResearchTrial"], Mapping[str, Any]]


@dataclass(frozen=True)
class ResearchOperatorSpec:
    """Declarative operator-family input for a research sweep."""

    name: str
    kind: str
    parameters: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("operator name must be non-empty.")
        if not self.kind.strip():
            raise ValueError("operator kind must be non-empty.")
        object.__setattr__(self, "parameters", dict(self.parameters))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ResearchOperatorSpec:
        """Construct an operator specification from JSON/YAML data."""
        unknown = set(value) - {"name", "kind", "parameters"}
        if unknown:
            raise ValueError(f"unknown operator fields: {sorted(unknown)}")
        kind = str(value.get("kind", "")).strip()
        name = str(value.get("name", kind)).strip()
        parameters = value.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise TypeError("operator parameters must be a mapping.")
        return cls(name=name, kind=kind, parameters=dict(parameters))

    def as_report(self) -> dict[str, object]:
        """Return a portable operator specification."""
        return {
            "name": self.name,
            "kind": self.kind,
            "parameters": self.parameters,
        }


@dataclass(frozen=True)
class ResearchTargetSpec:
    """Declarative target-transform input for a research sweep."""

    name: str
    kind: str
    parameters: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("target name must be non-empty.")
        if not self.kind.strip():
            raise ValueError("target kind must be non-empty.")
        object.__setattr__(self, "parameters", dict(self.parameters))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ResearchTargetSpec:
        """Construct a target specification from JSON/YAML data."""
        unknown = set(value) - {"name", "kind", "parameters"}
        if unknown:
            raise ValueError(f"unknown target fields: {sorted(unknown)}")
        kind = str(value.get("kind", "")).strip()
        name = str(value.get("name", kind)).strip()
        parameters = value.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise TypeError("target parameters must be a mapping.")
        return cls(name=name, kind=kind, parameters=dict(parameters))

    def as_report(self) -> dict[str, object]:
        """Return a portable target specification."""
        return {
            "name": self.name,
            "kind": self.kind,
            "parameters": self.parameters,
        }


@dataclass(frozen=True)
class ResearchSweepSpec:
    """Cartesian-product experiment definition with reproducibility metadata."""

    name: str
    study: str
    operators: tuple[ResearchOperatorSpec, ...]
    targets: tuple[ResearchTargetSpec, ...]
    access_models: tuple[str, ...]
    degrees: tuple[int, ...]
    tolerances: tuple[float, ...]
    phase_solvers: tuple[str, ...] = ("root-finding",)
    shots: tuple[int | None, ...] = (None,)
    seeds: tuple[int, ...] = (0,)
    noise_models: tuple[str, ...] = ("ideal",)
    attempt_synthesis: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("sweep name must be non-empty.")
        if not self.study.strip():
            raise ValueError("study must be non-empty.")
        operators = tuple(
            (
                value
                if isinstance(value, ResearchOperatorSpec)
                else ResearchOperatorSpec.from_mapping(value)
            )
            for value in self.operators
        )
        targets = tuple(
            (
                value
                if isinstance(value, ResearchTargetSpec)
                else ResearchTargetSpec.from_mapping(value)
            )
            for value in self.targets
        )
        access_models = tuple(str(value).strip() for value in self.access_models)
        degrees = tuple(int(value) for value in self.degrees)
        tolerances = tuple(float(value) for value in self.tolerances)
        phase_solvers = tuple(str(value).strip() for value in self.phase_solvers)
        shots = tuple(None if value is None else int(value) for value in self.shots)
        seeds = tuple(int(value) for value in self.seeds)
        noise_models = tuple(str(value).strip() for value in self.noise_models)

        named_axes: tuple[tuple[str, Sequence[object]], ...] = (
            ("operators", operators),
            ("targets", targets),
            ("access_models", access_models),
            ("degrees", degrees),
            ("tolerances", tolerances),
            ("phase_solvers", phase_solvers),
            ("shots", shots),
            ("seeds", seeds),
            ("noise_models", noise_models),
        )
        for axis_name, values in named_axes:
            if not values:
                raise ValueError(f"{axis_name} must contain at least one value.")
        if any(not value for value in (*access_models, *phase_solvers, *noise_models)):
            raise ValueError("string-valued sweep axes must not contain empty names.")
        if any(value < 0 for value in degrees):
            raise ValueError("degrees must be non-negative.")
        if any(value <= 0.0 for value in tolerances):
            raise ValueError("tolerances must be positive.")
        if any(value is not None and value <= 0 for value in shots):
            raise ValueError("shots must contain positive integers or null.")

        object.__setattr__(self, "operators", operators)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "access_models", access_models)
        object.__setattr__(self, "degrees", degrees)
        object.__setattr__(self, "tolerances", tolerances)
        object.__setattr__(self, "phase_solvers", phase_solvers)
        object.__setattr__(self, "shots", shots)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "noise_models", noise_models)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ResearchSweepSpec:
        """Construct a validated sweep from decoded JSON/YAML data."""
        allowed = {
            "schema_name",
            "schema_version",
            "name",
            "study",
            "operators",
            "targets",
            "access_models",
            "degrees",
            "tolerances",
            "phase_solvers",
            "shots",
            "seeds",
            "noise_models",
            "attempt_synthesis",
            "metadata",
            "trial_count",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown sweep fields: {sorted(unknown)}")
        schema_name = value.get("schema_name", "qsvt-research-sweep-spec")
        schema_version = str(value.get("schema_version", "1.0"))
        if schema_name != "qsvt-research-sweep-spec":
            raise ValueError("schema_name must be 'qsvt-research-sweep-spec'.")
        if schema_version != "1.0":
            raise ValueError("only research sweep schema version '1.0' is supported.")

        operators = value.get("operators")
        targets = value.get("targets")
        if not isinstance(operators, Sequence) or isinstance(operators, (str, bytes)):
            raise TypeError("operators must be a sequence of mappings.")
        if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
            raise TypeError("targets must be a sequence of mappings.")
        normalized_operators = tuple(
            (
                ResearchOperatorSpec.from_mapping(item)
                if isinstance(item, Mapping)
                else _raise_mapping_type("operator")
            )
            for item in operators
        )
        normalized_targets = tuple(
            (
                ResearchTargetSpec.from_mapping(item)
                if isinstance(item, Mapping)
                else _raise_mapping_type("target")
            )
            for item in targets
        )
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        attempt_synthesis = value.get("attempt_synthesis", False)
        if not isinstance(attempt_synthesis, bool):
            raise TypeError("attempt_synthesis must be a boolean.")

        return cls(
            name=str(value.get("name", "research-sweep")),
            study=str(value.get("study", "accuracy-resource-frontier")),
            operators=normalized_operators,
            targets=normalized_targets,
            access_models=_sequence_axis(
                value.get("access_models", ("embedding",)),
                "access_models",
            ),
            degrees=_sequence_axis(value.get("degrees", (3,)), "degrees"),
            tolerances=_sequence_axis(
                value.get("tolerances", (0.1,)),
                "tolerances",
            ),
            phase_solvers=_sequence_axis(
                value.get("phase_solvers", ("root-finding",)),
                "phase_solvers",
            ),
            shots=_sequence_axis(value.get("shots", (None,)), "shots"),
            seeds=_sequence_axis(value.get("seeds", (0,)), "seeds"),
            noise_models=_sequence_axis(
                value.get("noise_models", ("ideal",)),
                "noise_models",
            ),
            attempt_synthesis=attempt_synthesis,
            metadata=dict(metadata),
        )

    @property
    def trial_count(self) -> int:
        """Return the size of the fully expanded Cartesian product."""
        axes = (
            self.operators,
            self.targets,
            self.access_models,
            self.degrees,
            self.tolerances,
            self.phase_solvers,
            self.shots,
            self.seeds,
            self.noise_models,
        )
        count = 1
        for values in axes:
            count *= len(values)
        return count

    def as_report(self) -> dict[str, object]:
        """Return a JSON-safe declarative sweep specification."""
        return {
            "schema_name": "qsvt-research-sweep-spec",
            "schema_version": "1.0",
            "name": self.name,
            "study": self.study,
            "operators": [value.as_report() for value in self.operators],
            "targets": [value.as_report() for value in self.targets],
            "access_models": list(self.access_models),
            "degrees": list(self.degrees),
            "tolerances": list(self.tolerances),
            "phase_solvers": list(self.phase_solvers),
            "shots": list(self.shots),
            "seeds": list(self.seeds),
            "noise_models": list(self.noise_models),
            "attempt_synthesis": self.attempt_synthesis,
            "metadata": self.metadata,
            "trial_count": self.trial_count,
        }


@dataclass(frozen=True)
class ResearchTrial:
    """One deterministic point in an expanded research sweep."""

    sweep_name: str
    study: str
    operator: ResearchOperatorSpec
    target: ResearchTargetSpec
    access_model: str
    degree: int
    tolerance: float
    phase_solver: str
    shots: int | None
    seed: int
    noise_model: str
    attempt_synthesis: bool

    @property
    def trial_id(self) -> str:
        """Return a stable identifier derived from all computational factors."""
        payload = json.dumps(
            report_to_jsonable(self.factor_report()),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:20]

    def factor_report(self) -> dict[str, object]:
        """Return the trial factors used for identity and reporting."""
        return {
            "sweep_name": self.sweep_name,
            "study": self.study,
            "operator": self.operator.as_report(),
            "target": self.target.as_report(),
            "access_model": self.access_model,
            "degree": self.degree,
            "tolerance": self.tolerance,
            "phase_solver": self.phase_solver,
            "shots": self.shots,
            "seed": self.seed,
            "noise_model": self.noise_model,
            "attempt_synthesis": self.attempt_synthesis,
        }


@dataclass(frozen=True)
class ResearchSweepResult:
    """Completed and resumed trial reports plus artifact metadata."""

    spec: ResearchSweepSpec
    trial_reports: tuple[dict[str, Any], ...]
    executed_count: int
    resumed_count: int
    failed_count: int
    output_dir: Path | None = None

    def as_report(self, *, include_trials: bool = False) -> dict[str, object]:
        """Return a compact manifest or a manifest with embedded trials."""
        output_dir = self.output_dir
        report: dict[str, object] = {
            "schema_name": "qsvt-research-sweep-manifest",
            "schema_version": "1.0",
            "mode": "research-sweep-manifest",
            "study": self.spec.study,
            "name": self.spec.name,
            "trial_count": len(self.trial_reports),
            "executed_count": self.executed_count,
            "resumed_count": self.resumed_count,
            "failed_count": self.failed_count,
            "trial_ids": [report["trial_id"] for report in self.trial_reports],
            "artifacts": {
                "output_dir": None if output_dir is None else str(output_dir),
                "spec": None if output_dir is None else str(output_dir / "sweep.json"),
                "summary_csv": (
                    None if output_dir is None else str(output_dir / "summary.csv")
                ),
                "trials_dir": (
                    None if output_dir is None else str(output_dir / "trials")
                ),
            },
            "truth_contract": {
                "individual_trials_are_persisted": output_dir is not None,
                "resume_uses_deterministic_trial_ids": True,
                "timings_are_environment_dependent": True,
                "study_claims_are_defined_by_the_selected_evaluator": True,
            },
        }
        if include_trials:
            report["trials"] = list(self.trial_reports)
        return report


def expand_research_sweep(spec: ResearchSweepSpec) -> tuple[ResearchTrial, ...]:
    """Expand a declarative sweep into a deterministic Cartesian product."""
    if not isinstance(spec, ResearchSweepSpec):
        raise TypeError("spec must be a ResearchSweepSpec.")
    trials = []
    for values in itertools.product(
        spec.operators,
        spec.targets,
        spec.access_models,
        spec.degrees,
        spec.tolerances,
        spec.phase_solvers,
        spec.shots,
        spec.seeds,
        spec.noise_models,
    ):
        operator, target, access, degree, tolerance, solver, shots, seed, noise = values
        trials.append(
            ResearchTrial(
                sweep_name=spec.name,
                study=spec.study,
                operator=operator,
                target=target,
                access_model=access,
                degree=degree,
                tolerance=tolerance,
                phase_solver=solver,
                shots=shots,
                seed=seed,
                noise_model=noise,
                attempt_synthesis=spec.attempt_synthesis,
            )
        )
    trial_ids = [trial.trial_id for trial in trials]
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError(
            "the sweep expands to duplicate trials; remove duplicate axis values."
        )
    return tuple(trials)


def load_research_sweep_spec(path: str | Path) -> ResearchSweepSpec:
    """Load a research sweep from JSON or optionally from YAML."""
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".json":
        decoded = json.loads(input_path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "YAML sweep files require PyYAML; use JSON or install "
                "qsvt-pennylane[research]."
            ) from exc
        decoded = yaml.safe_load(input_path.read_text(encoding="utf-8"))
    else:
        raise ValueError("research sweep files must use .json, .yaml, or .yml.")
    if not isinstance(decoded, Mapping):
        raise TypeError("research sweep files must contain a top-level mapping.")
    return ResearchSweepSpec.from_mapping(decoded)


def save_research_sweep_spec(spec: ResearchSweepSpec, path: str | Path) -> Path:
    """Persist the normalized sweep specification as JSON or optional YAML."""
    output_path = Path(path)
    suffix = output_path.suffix.lower()
    if suffix == ".json":
        return save_report(spec.as_report(), output_path)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "YAML sweep files require PyYAML; use JSON or install "
                "qsvt-pennylane[research]."
            ) from exc
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            yaml.safe_dump(report_to_jsonable(spec.as_report()), sort_keys=False),
            encoding="utf-8",
        )
        return output_path
    raise ValueError("research sweep files must use .json, .yaml, or .yml.")


def run_research_sweep(
    spec: ResearchSweepSpec,
    evaluator: ResearchEvaluator,
    *,
    output_dir: str | Path | None = None,
    resume: bool = True,
    fail_fast: bool = False,
) -> ResearchSweepResult:
    """Execute or resume every trial and optionally persist all artifacts."""
    if not isinstance(spec, ResearchSweepSpec):
        raise TypeError("spec must be a ResearchSweepSpec.")
    if not callable(evaluator):
        raise TypeError("evaluator must be callable.")

    resolved_output = None if output_dir is None else Path(output_dir)
    trials_dir: Path | None = None
    if resolved_output is not None:
        resolved_output.mkdir(parents=True, exist_ok=True)
        trials_dir = resolved_output / "trials"
        trials_dir.mkdir(parents=True, exist_ok=True)
        save_research_sweep_spec(spec, resolved_output / "sweep.json")

    reports: list[dict[str, Any]] = []
    executed_count = 0
    resumed_count = 0
    failed_count = 0
    for trial in expand_research_sweep(spec):
        trial_path = (
            None if trials_dir is None else trials_dir / f"{trial.trial_id}.json"
        )
        existing = (
            _load_resumable_trial(trial_path, trial)
            if resume and trial_path is not None
            else None
        )
        if existing is not None:
            reports.append(existing)
            resumed_count += 1
            failed_count += int(existing.get("status") == "failed")
            continue

        try:
            result = dict(evaluator(trial))
            status = str(result.get("status", "completed"))
            if status not in {"completed", "skipped", "failed"}:
                raise ValueError(
                    "evaluator status must be 'completed', 'skipped', or 'failed'."
                )
            report = {
                "schema_name": "qsvt-research-trial",
                "schema_version": "1.0",
                "mode": "research-sweep-trial",
                "trial_id": trial.trial_id,
                "status": status,
                "factors": trial.factor_report(),
                "result": result,
            }
        except Exception as exc:
            status = "failed"
            report = {
                "schema_name": "qsvt-research-trial",
                "schema_version": "1.0",
                "mode": "research-sweep-trial",
                "trial_id": trial.trial_id,
                "status": status,
                "factors": trial.factor_report(),
                "result": {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            }
            if fail_fast:
                if trial_path is not None:
                    save_report(report, trial_path)
                raise
        reports.append(report)
        executed_count += 1
        failed_count += int(status == "failed")
        if trial_path is not None:
            save_report(report, trial_path)

    sweep_result = ResearchSweepResult(
        spec=spec,
        trial_reports=tuple(reports),
        executed_count=executed_count,
        resumed_count=resumed_count,
        failed_count=failed_count,
        output_dir=resolved_output,
    )
    if resolved_output is not None:
        write_research_summary_csv(reports, resolved_output / "summary.csv")
        save_report(sweep_result.as_report(), resolved_output / "manifest.json")
    return sweep_result


def research_summary_rows(
    trial_reports: Sequence[Mapping[str, Any]],
) -> list[dict[str, object]]:
    """Flatten trial factors and evaluator summaries into compact rows."""
    rows: list[dict[str, object]] = []
    for report in trial_reports:
        factors = report.get("factors", {})
        operator = factors.get("operator", {}) if isinstance(factors, Mapping) else {}
        target = factors.get("target", {}) if isinstance(factors, Mapping) else {}
        result = report.get("result", {})
        summary = result.get("summary", {}) if isinstance(result, Mapping) else {}
        row: dict[str, object] = {
            "trial_id": report.get("trial_id"),
            "status": report.get("status"),
            "operator": operator.get("name") if isinstance(operator, Mapping) else None,
            "operator_kind": (
                operator.get("kind") if isinstance(operator, Mapping) else None
            ),
            "target": target.get("name") if isinstance(target, Mapping) else None,
            "target_kind": target.get("kind") if isinstance(target, Mapping) else None,
        }
        if isinstance(factors, Mapping):
            for name in (
                "access_model",
                "degree",
                "tolerance",
                "phase_solver",
                "shots",
                "seed",
                "noise_model",
                "attempt_synthesis",
            ):
                row[name] = factors.get(name)
        if isinstance(summary, Mapping):
            for key, value in summary.items():
                row[str(key)] = value
        if isinstance(result, Mapping):
            row["error_type"] = result.get("error_type")
            row["error"] = result.get("error")
        rows.append(row)
    return rows


def write_research_summary_csv(
    trial_reports: Sequence[Mapping[str, Any]],
    path: str | Path,
) -> Path:
    """Write aggregate sweep rows with stable leading factor columns."""
    rows = research_summary_rows(trial_reports)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    leading = [
        "trial_id",
        "status",
        "operator",
        "operator_kind",
        "target",
        "target_kind",
        "access_model",
        "degree",
        "tolerance",
        "phase_solver",
        "shots",
        "seed",
        "noise_model",
        "attempt_synthesis",
    ]
    discovered = sorted({key for row in rows for key in row} - set(leading))
    fieldnames = leading + discovered
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    return output_path


def _load_resumable_trial(
    path: Path,
    trial: ResearchTrial,
) -> dict[str, Any] | None:
    try:
        report = load_report(path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if report.get("schema_name") != "qsvt-research-trial":
        return None
    if report.get("schema_version") != "1.0":
        return None
    if report.get("trial_id") != trial.trial_id:
        return None
    if report.get("status") not in {"completed", "skipped", "failed"}:
        return None
    if report.get("factors") != report_to_jsonable(trial.factor_report()):
        return None
    return report


def _csv_value(value: object) -> object:
    converted = report_to_jsonable({"value": value})["value"]
    if isinstance(converted, (dict, list)):
        return json.dumps(converted, sort_keys=True, separators=(",", ":"))
    return converted


def _raise_mapping_type(kind: str) -> Any:
    raise TypeError(f"each {kind} specification must be a mapping.")


def _sequence_axis(value: object, name: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence.")
    return tuple(value)


__all__ = [
    "ResearchEvaluator",
    "ResearchOperatorSpec",
    "ResearchSweepResult",
    "ResearchSweepSpec",
    "ResearchTargetSpec",
    "ResearchTrial",
    "expand_research_sweep",
    "load_research_sweep_spec",
    "research_summary_rows",
    "run_research_sweep",
    "save_research_sweep_spec",
    "write_research_summary_csv",
]
