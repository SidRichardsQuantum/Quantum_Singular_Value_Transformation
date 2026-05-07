"""
qsvt.reports
------------

Utilities for serializing and plotting QSVT polynomial diagnostics reports.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


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
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.0, 5.0))
        fit_ax, error_ax = axes
    else:
        fit_ax = ax
        fig = fit_ax.figure
        error_ax = fit_ax.twinx()
        axes = (fit_ax, error_ax)

    label = report.get("builder", "polynomial")
    fit_ax.plot(xs, target_values, label="target")
    fit_ax.plot(xs, polynomial_values, label=str(label))
    fit_ax.set_ylabel("value")
    fit_ax.legend(loc="best")

    error_ax.plot(xs, errors, color="tab:red", label="error")
    error_ax.axhline(0.0, color="0.5", linewidth=0.8)
    error_ax.set_xlabel("x")
    error_ax.set_ylabel("error")

    title_parts = []
    if "kind" in report:
        title_parts.append(str(report["kind"]))
    if "max_error" in report:
        title_parts.append(f"max error={float(report['max_error']):.3g}")
    if "bounded_margin" in report:
        title_parts.append(f"margin={float(report['bounded_margin']):.3g}")
    if title_parts:
        fit_ax.set_title(" | ".join(title_parts))

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


def _require_array(report: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in report:
        raise KeyError(f"report is missing required key: {key}")
    values = np.asarray(report[key], dtype=float)
    if values.ndim != 1:
        raise ValueError(f"report[{key!r}] must be one-dimensional.")
    return values


__all__ = [
    "report_to_jsonable",
    "save_report",
    "load_report",
    "plot_approximation_report",
    "save_report_plot",
]
