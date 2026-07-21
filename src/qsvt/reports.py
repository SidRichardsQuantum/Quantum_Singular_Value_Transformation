"""
qsvt.reports
------------

Utilities for serializing and plotting QSVT polynomial diagnostics reports.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_SUPPORTED_REPORT_SCHEMAS: dict[str, frozenset[str]] = {
    "block-encoding-qsvt-execution": frozenset({"1.0"}),
    "hardware-qsvt-circuit": frozenset({"1.0"}),
    "hardware-qsvt-execution": frozenset({"1.0"}),
    "qsvt-accuracy-resource-frontier": frozenset({"1.0"}),
    "qsvt-accuracy-resource-frontier-rows": frozenset({"1.0"}),
    "qsvt-algorithm-workflow": frozenset({"1.0", "1.1"}),
    "qsvt-problem-workflow": frozenset({"1.0"}),
    "qsvt-research-sweep-manifest": frozenset({"1.0"}),
    "qsvt-research-sweep-spec": frozenset({"1.0"}),
    "qsvt-research-trial": frozenset({"1.0"}),
}

_REQUIRED_REPORT_SCHEMA_FIELDS: dict[str, frozenset[str]] = {
    "block-encoding-qsvt-execution": frozenset(
        {
            "schema_name",
            "schema_version",
            "mode",
            "implementation_kind",
            "truth_contract",
            "resource_summary",
        }
    ),
    "hardware-qsvt-circuit": frozenset(
        {
            "schema_name",
            "schema_version",
            "mode",
            "implementation_kind",
            "truth_contract",
            "logical_resource_summary",
            "decomposed_resource_summary",
        }
    ),
    "hardware-qsvt-execution": frozenset(
        {
            "schema_name",
            "schema_version",
            "mode",
            "implementation_kind",
            "truth_contract",
            "resource_summary",
        }
    ),
    "qsvt-algorithm-workflow": frozenset(
        {
            "schema_name",
            "schema_version",
            "mode",
            "implementation_kind",
            "truth_contract",
        }
    ),
    "qsvt-problem-workflow": frozenset(
        {
            "schema_name",
            "schema_version",
            "target",
            "truth_contract",
        }
    ),
    "qsvt-research-sweep-spec": frozenset(
        {
            "schema_name",
            "schema_version",
            "name",
            "study",
            "operators",
            "targets",
        }
    ),
    "qsvt-research-trial": frozenset(
        {
            "schema_name",
            "schema_version",
            "mode",
            "trial_id",
            "status",
            "factors",
            "result",
        }
    ),
    "qsvt-research-sweep-manifest": frozenset(
        {
            "schema_name",
            "schema_version",
            "mode",
            "study",
            "name",
            "trial_count",
            "artifacts",
            "truth_contract",
        }
    ),
    "qsvt-accuracy-resource-frontier": frozenset(
        {
            "schema_name",
            "schema_version",
            "mode",
            "sweep",
            "frontier_row_count",
            "pareto_row_count",
            "truth_contract",
        }
    ),
    "qsvt-accuracy-resource-frontier-rows": frozenset(
        {
            "schema_name",
            "schema_version",
            "mode",
            "rows",
            "truth_contract",
        }
    ),
}

_REQUIRED_ALGORITHM_TRUTH_FIELDS_V1_1 = frozenset(
    {
        "circuit_evaluated",
        "execution_tier",
        "physical_device_executed",
        "polynomial_evidence",
        "qnode_executed",
        "resource_completeness",
        "truth_status",
    }
)

_KNOWN_REPORT_SCHEMA_FIELDS: dict[str, frozenset[str]] = {
    "block-encoding-qsvt-execution": frozenset(
        {
            "angle_solver",
            "block_encoding_spec",
            "classical_reference_output",
            "complex_leakage_norm",
            "device_name",
            "error",
            "error_type",
            "final_state",
            "implementation_kind",
            "input_state",
            "is_end_to_end_quantum_algorithm",
            "logical_output",
            "logical_output_absolute_error",
            "logical_output_relative_error",
            "logical_probabilities",
            "logical_subspace_leakage_probability",
            "logical_success_probability",
            "logical_success_standard_error",
            "maximum_probability_standard_error",
            "mode",
            "poly",
            "probabilities",
            "probability_normalization_error",
            "projector_convention",
            "projector_source",
            "resource_summary",
            "schema_name",
            "schema_version",
            "shots",
            "statevector_normalization_error",
            "succeeded",
            "truth_contract",
            "wire_order",
        }
    ),
    "hardware-qsvt-circuit": frozenset(
        {
            "angle_solver",
            "block_encoding_spec",
            "decomposed_operations",
            "decomposed_resource_summary",
            "decomposition_error",
            "decomposition_error_type",
            "decomposition_status",
            "device_name",
            "executed",
            "implementation_kind",
            "is_end_to_end_quantum_algorithm",
            "logical_operations",
            "logical_resource_summary",
            "measurements",
            "mode",
            "poly",
            "preflight",
            "projector_source",
            "provider_plugin",
            "schema_name",
            "schema_version",
            "shots",
            "truth_contract",
            "unsupported_decomposed_operations",
            "unsupported_logical_operations",
            "wire_order",
        }
    ),
    "hardware-qsvt-execution": frozenset(
        {
            "angle_solver",
            "block_encoding_spec",
            "device_name",
            "error",
            "error_type",
            "implementation_kind",
            "is_end_to_end_quantum_algorithm",
            "logical_probabilities",
            "logical_success_probability",
            "logical_success_standard_error",
            "maximum_probability_standard_error",
            "mode",
            "poly",
            "preflight",
            "probabilities",
            "probability_normalization_error",
            "projector_source",
            "resource_summary",
            "schema_name",
            "schema_version",
            "shots",
            "succeeded",
            "truth_contract",
            "wire_order",
        }
    ),
    "qsvt-problem-workflow": frozenset(
        {
            "classical_reference",
            "implementation_kind",
            "implementation_layer",
            "input_kind",
            "inputs",
            "mode",
            "package_version",
            "polynomial_resource_proxy",
            "resource_reports",
            "result",
            "schema_name",
            "schema_version",
            "target",
            "truth_contract",
            "workflow_status",
        }
    ),
}


@dataclass(frozen=True)
class ReportSchemaCompatibility:
    """
    Compatibility summary for a versioned machine-readable report.
    """

    schema_name: str | None
    schema_version: str | None
    supported: bool
    migration_required: bool
    message: str
    required_fields: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    unknown_fields: tuple[str, ...] = ()

    def as_report(self) -> dict[str, Any]:
        """Return this compatibility summary as a JSON-safe report payload."""
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "supported": self.supported,
            "migration_required": self.migration_required,
            "message": self.message,
            "required_fields": list(self.required_fields),
            "missing_fields": list(self.missing_fields),
            "unknown_fields": list(self.unknown_fields),
        }


def report_to_jsonable(report: Mapping[str, Any]) -> dict[str, Any]:
    """
    Convert a diagnostics report into JSON-serializable Python containers.

    NumPy arrays become lists, NumPy scalars become Python scalars, complex
    values become `{real, imag}` payloads, and nested dictionaries/lists/tuples
    are converted recursively.
    """
    return _to_jsonable(report)


def save_report(
    report: Mapping[str, Any],
    path: str | Path,
    *,
    indent: int = 2,
) -> Path:
    """
    Write a diagnostics report to JSON.

    Parameters
    ----------
    report
        Report dictionary produced by the design/template diagnostics helpers.
    path
        Destination JSON path.
    indent
        JSON indentation level.

    Returns
    -------
    pathlib.Path
        Path written.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report_to_jsonable(report), indent=indent) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_report(path: str | Path) -> dict[str, Any]:
    """
    Load a JSON diagnostics report from disk.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def supported_report_schemas() -> dict[str, tuple[str, ...]]:
    """
    Return supported versioned report schemas and versions.

    The returned dictionary is a copy, so callers can inspect it without
    mutating the package-level compatibility registry.
    """
    return {
        name: tuple(sorted(versions))
        for name, versions in sorted(_SUPPORTED_REPORT_SCHEMAS.items())
    }


def migrate_algorithm_workflow_report(
    report: Mapping[str, Any],
    *,
    target_version: str = "1.1",
) -> dict[str, Any]:
    """Migrate a ``qsvt-algorithm-workflow`` report from 1.0 to 1.1.

    Schema 1.1 derives execution-tier and polynomial-realizability claims from
    coefficients stored in the report. Migration therefore fails explicitly
    when a legacy report does not retain polynomial coefficients; it never
    fabricates truth evidence from a legacy status label.

    Reports already at 1.1 are returned as independent deep copies. The input
    mapping is never mutated.
    """
    schema_name = str(report.get("schema_name", ""))
    source_version = str(report.get("schema_version", ""))
    if schema_name != "qsvt-algorithm-workflow":
        raise ValueError(
            "algorithm workflow migration requires schema_name "
            "'qsvt-algorithm-workflow'"
        )
    if target_version != "1.1":
        raise ValueError("algorithm workflow reports can only migrate to version '1.1'")
    if source_version == target_version:
        return report_to_jsonable(dict(deepcopy(report)))
    if source_version != "1.0":
        raise ValueError(
            "algorithm workflow migration supports source version '1.0'; "
            f"received {source_version!r}"
        )

    polynomials = _legacy_algorithm_polynomials(report)
    if not polynomials:
        raise ValueError(
            "cannot migrate qsvt-algorithm-workflow 1.0 report to 1.1: "
            "no polynomial coefficients were retained"
        )

    legacy_contract = report.get("truth_contract")
    if not isinstance(legacy_contract, Mapping):
        raise ValueError("legacy algorithm report truth_contract must be a mapping")

    # Import lazily so the generic reporting module does not depend on the
    # algorithm workflow implementation during normal report loading.
    from ._algorithm_reports import algorithm_truth_contract

    workflow = str(legacy_contract.get("workflow", report.get("mode", "unknown")))
    target = str(legacy_contract.get("target", "legacy algorithm workflow"))
    qsvt_check = str(legacy_contract.get("pennylane_qsvt_check", "not_attempted"))
    derived_contract = algorithm_truth_contract(
        workflow,
        target=target,
        qsvt_check=qsvt_check,
        polynomials=polynomials,
        qnode_executed=bool(legacy_contract.get("qnode_executed", False)),
        physical_device_executed=bool(
            legacy_contract.get("physical_device_executed", False)
        ),
    )

    migrated = dict(deepcopy(report))
    migrated["schema_version"] = target_version
    migrated["truth_contract"] = {
        **dict(deepcopy(legacy_contract)),
        **derived_contract,
    }
    migrated["schema_migration"] = {
        "from_version": source_version,
        "to_version": target_version,
        "method": "artifact-derived-polynomial-truth-evidence",
    }
    return report_to_jsonable(migrated)


def report_schema_manifest(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """
    Return schema compatibility summaries for multiple report files.

    Each manifest row includes the input path, schema metadata when present,
    support status, migration status, and missing required fields. Invalid JSON
    files are reported as unsupported rows instead of aborting the whole batch.
    """
    rows: list[dict[str, Any]] = []
    for path in paths:
        report_path = Path(path)
        try:
            compatibility = validate_report_schema(
                load_report(report_path),
                require_schema=True,
            )
        except json.JSONDecodeError as exc:
            compatibility = ReportSchemaCompatibility(
                schema_name=None,
                schema_version=None,
                supported=False,
                migration_required=False,
                message=f"invalid JSON report: {exc.msg}",
            )
        rows.append({"path": str(report_path), **compatibility.as_report()})
    return rows


def write_report_schema_manifest_csv(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
) -> Path:
    """
    Write compact report-schema manifest rows to CSV.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "schema_name",
        "schema_version",
        "supported",
        "migration_required",
        "missing_fields",
        "unknown_fields",
        "message",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_cell(row.get(field)) for field in fieldnames})
    return output_path


def validate_report_schema(
    report: Mapping[str, Any],
    *,
    require_schema: bool = False,
) -> ReportSchemaCompatibility:
    """
    Check whether a loaded report uses a supported versioned schema.

    Unversioned diagnostic reports remain valid package artifacts, so missing
    schema metadata is accepted unless ``require_schema`` is true. Reports with
    known schema names but unsupported versions return a migration-required
    result with an intentional message instead of failing indirectly later.
    """
    schema_name = report.get("schema_name")
    schema_version = report.get("schema_version")

    if schema_name is None or schema_version is None:
        supported = not require_schema
        message = (
            "report has no versioned schema metadata; treating it as an "
            "unversioned diagnostics report"
            if supported
            else "report is missing required schema_name or schema_version metadata"
        )
        return ReportSchemaCompatibility(
            schema_name=str(schema_name) if schema_name is not None else None,
            schema_version=(
                str(schema_version) if schema_version is not None else None
            ),
            supported=supported,
            migration_required=False,
            message=message,
        )

    schema_name_text = str(schema_name)
    schema_version_text = str(schema_version)
    supported_versions = _SUPPORTED_REPORT_SCHEMAS.get(schema_name_text)
    if supported_versions is None:
        return ReportSchemaCompatibility(
            schema_name=schema_name_text,
            schema_version=schema_version_text,
            supported=False,
            migration_required=True,
            message=(
                f"unsupported report schema {schema_name_text!r}; add an "
                "explicit migration or compatibility handler before loading "
                "this report as a stable artifact"
            ),
        )

    if schema_version_text not in supported_versions:
        versions = ", ".join(sorted(supported_versions))
        return ReportSchemaCompatibility(
            schema_name=schema_name_text,
            schema_version=schema_version_text,
            supported=False,
            migration_required=True,
            message=(
                f"unsupported {schema_name_text!r} schema version "
                f"{schema_version_text!r}; supported versions: {versions}"
            ),
        )

    required_fields = tuple(
        sorted(_REQUIRED_REPORT_SCHEMA_FIELDS.get(schema_name_text, ()))
    )
    known_fields = _KNOWN_REPORT_SCHEMA_FIELDS.get(schema_name_text)
    unknown_fields = (
        tuple(sorted(str(field) for field in set(report) - known_fields))
        if known_fields is not None
        else ()
    )
    missing_fields = tuple(field for field in required_fields if field not in report)
    if (
        schema_name_text == "qsvt-algorithm-workflow"
        and schema_version_text == "1.1"
        and "truth_contract" in report
    ):
        truth_contract = report["truth_contract"]
        if not isinstance(truth_contract, Mapping):
            missing_fields += ("truth_contract.<mapping>",)
        else:
            missing_fields += tuple(
                f"truth_contract.{field}"
                for field in sorted(_REQUIRED_ALGORITHM_TRUTH_FIELDS_V1_1)
                if field not in truth_contract
            )
    if missing_fields:
        missing = ", ".join(missing_fields)
        return ReportSchemaCompatibility(
            schema_name=schema_name_text,
            schema_version=schema_version_text,
            supported=False,
            migration_required=False,
            message=(
                f"report schema {schema_name_text!r} version "
                f"{schema_version_text!r} is missing required fields: {missing}"
            ),
            required_fields=required_fields,
            missing_fields=missing_fields,
            unknown_fields=unknown_fields,
        )

    return ReportSchemaCompatibility(
        schema_name=schema_name_text,
        schema_version=schema_version_text,
        supported=True,
        migration_required=False,
        message=(
            f"report schema {schema_name_text!r} version "
            f"{schema_version_text!r} is supported"
        ),
        required_fields=required_fields,
        unknown_fields=unknown_fields,
    )


def load_report_with_schema(
    path: str | Path,
    *,
    require_schema: bool = True,
    expected_schema_name: str | None = None,
    expected_schema_version: str | None = None,
) -> tuple[dict[str, Any], ReportSchemaCompatibility]:
    """
    Load a JSON report and return its schema compatibility summary.

    Raises
    ------
    ValueError
        If the report is missing required schema metadata or needs a migration
        that this package version does not provide.
    """
    report = load_report(path)
    compatibility = validate_report_schema(report, require_schema=require_schema)
    if not compatibility.supported:
        raise ValueError(compatibility.message)
    if (
        expected_schema_name is not None
        and compatibility.schema_name != expected_schema_name
    ):
        raise ValueError(
            "loaded report schema "
            f"{compatibility.schema_name!r} does not match expected schema "
            f"{expected_schema_name!r}"
        )
    if (
        expected_schema_version is not None
        and compatibility.schema_version != expected_schema_version
    ):
        raise ValueError(
            "loaded report schema version "
            f"{compatibility.schema_version!r} does not match expected version "
            f"{expected_schema_version!r}"
        )
    return report, compatibility


def plot_approximation_report(
    report: Mapping[str, Any],
    ax=None,
):
    """
    Plot target, polynomial, and error curves from an approximation report.

    Parameters
    ----------
    report
        Report dictionary containing `xs`, `target_values`,
        `polynomial_values`, and `errors`.
    ax
        Optional Matplotlib axes for the target/polynomial plot.

    Returns
    -------
    tuple
        `(fig, axes)` where `axes` contains the approximation and error axes.
    """
    import matplotlib.pyplot as plt

    xs = _require_array(report, "xs")
    target_values = _require_array(report, "target_values")
    polynomial_values = _require_array(report, "polynomial_values")
    errors = _require_array(report, "errors")

    if ax is None:
        fig, axes = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(7.2, 5.2),
            height_ratios=(2.2, 1.0),
            constrained_layout=True,
        )
        fit_ax, error_ax = axes
    else:
        fit_ax = ax
        fig = fit_ax.figure
        error_ax = fit_ax.twinx()
        axes = (fit_ax, error_ax)

    polynomial_label = _polynomial_label(report)
    fit_ax.plot(xs, target_values, label="target", color="tab:blue", linewidth=2.0)
    fit_ax.plot(
        xs,
        polynomial_values,
        "--",
        label=polynomial_label,
        color="tab:orange",
        linewidth=2.0,
    )
    fit_ax.set_ylabel("value")
    fit_ax.grid(True, color="0.88", linewidth=0.8)
    fit_ax.legend(loc="best", frameon=False)

    error_ax.plot(xs, errors, color="tab:red", linewidth=1.5, label="error")
    error_ax.axhline(0.0, color="0.35", linewidth=0.8)
    error_ax.set_xlabel("x")
    error_ax.set_ylabel("error")
    error_ax.grid(True, color="0.9", linewidth=0.8)
    _set_symmetric_error_limits(error_ax, errors)

    title_parts = []
    if "kind" in report:
        title_parts.append(str(report["kind"]))
    if "degree" in report:
        title_parts.append(f"degree={int(report['degree'])}")
    if "max_error" in report:
        title_parts.append(f"max error={float(report['max_error']):.3g}")
    if "bounded_margin" in report:
        title_parts.append(f"margin={float(report['bounded_margin']):.3g}")
    if title_parts:
        fit_ax.set_title(" | ".join(title_parts))

    if ax is not None:
        fig.tight_layout()
    return fig, axes


def save_report_plot(
    report: Mapping[str, Any],
    path: str | Path,
    *,
    dpi: int = 150,
) -> Path:
    """
    Save a target-vs-polynomial diagnostics plot to an image file.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, _ = plot_approximation_report(report)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    import matplotlib.pyplot as plt

    plt.close(fig)
    return output_path


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return {
                "real": obj.real.tolist(),
                "imag": obj.imag.tolist(),
            }
        return obj.tolist()
    if isinstance(obj, np.generic):
        if np.iscomplexobj(obj):
            return {
                "real": float(np.real(obj)),
                "imag": float(np.imag(obj)),
            }
        return obj.item()
    if isinstance(obj, complex):
        return {
            "real": float(obj.real),
            "imag": float(obj.imag),
        }
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    return obj


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _require_array(report: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in report:
        raise KeyError(f"report is missing required key: {key}")
    values = np.asarray(report[key], dtype=float)
    if values.ndim != 1:
        raise ValueError(f"report[{key!r}] must be one-dimensional.")
    return values


def _polynomial_label(report: Mapping[str, Any]) -> str:
    label = str(report.get("builder", "polynomial"))
    return label.removeprefix("design_").removeprefix("template_")


def _legacy_algorithm_polynomials(report: Mapping[str, Any]) -> dict[str, object]:
    """Extract retained polynomial components from a 1.0 workflow report."""
    polynomials: dict[str, object] = {}
    coeffs_by_center = report.get("coeffs_by_center")
    if isinstance(coeffs_by_center, Mapping):
        polynomials.update(
            {f"center_{center}": coeffs for center, coeffs in coeffs_by_center.items()}
        )

    for key, value in report.items():
        key_text = str(key)
        if key_text == "coeffs":
            polynomials.setdefault("transform", value)
        elif key_text.endswith("_coeffs") and key_text != "coeffs_by_center":
            polynomials[key_text.removesuffix("_coeffs")] = value
    return polynomials


def _set_symmetric_error_limits(ax, errors: np.ndarray) -> None:
    finite_errors = errors[np.isfinite(errors)]
    if finite_errors.size == 0:
        return
    limit = float(np.max(np.abs(finite_errors)))
    if limit == 0.0:
        limit = 1.0
    else:
        limit *= 1.08
    ax.set_ylim(-limit, limit)


__all__ = [
    "ReportSchemaCompatibility",
    "report_to_jsonable",
    "report_schema_manifest",
    "save_report",
    "load_report",
    "load_report_with_schema",
    "migrate_algorithm_workflow_report",
    "plot_approximation_report",
    "save_report_plot",
    "supported_report_schemas",
    "validate_report_schema",
    "write_report_schema_manifest_csv",
]
