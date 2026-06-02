"""
Classical benchmark baselines for QSVT-oriented workflows.

These helpers provide small, reproducible classical baselines and cost metadata
for the problem classes used throughout the notebooks: dense eigensolvers,
dense linear solves, conjugate-gradient solves for positive-definite systems,
and spectral or polynomial matrix functions. They are intended as transparent
reference baselines for advantage-oriented studies, not as optimized benchmark
suites.
"""

from __future__ import annotations

import csv
import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ._algorithm_reports import benchmark_truth_contract
from .polynomials import polynomial_degree
from .resources import qsvt_resource_report
from .spectral import apply_function_to_hermitian, apply_polynomial_to_hermitian


@dataclass(frozen=True)
class ClassicalBenchmarkResult:
    """
    Structured classical benchmark report.
    """

    algorithm: str
    problem: str
    matrix_dimension: int
    repeats: int
    best_time_seconds: float
    mean_time_seconds: float
    metrics: dict[str, Any]
    qsvt_proxy: dict[str, Any] | None = None
    notes: tuple[str, ...] = ()

    def as_report(self) -> dict[str, Any]:
        """
        Return a JSON-friendly benchmark report.
        """
        return {
            "mode": "classical-benchmark",
            "truth_contract": benchmark_truth_contract(),
            "algorithm": self.algorithm,
            "problem": self.problem,
            "matrix_dimension": self.matrix_dimension,
            "repeats": self.repeats,
            "best_time_seconds": self.best_time_seconds,
            "mean_time_seconds": self.mean_time_seconds,
            "benchmark_environment": benchmark_environment_report(),
            "metrics": self.metrics,
            "qsvt_proxy": self.qsvt_proxy,
            "notes": list(self.notes),
        }


def dense_eigendecomposition_benchmark(
    matrix: np.ndarray | list[list[complex]],
    *,
    repeats: int = 3,
) -> dict[str, Any]:
    """
    Benchmark dense Hermitian eigendecomposition with ``numpy.linalg.eigh``.
    """
    hermitian = _validate_hermitian_matrix(matrix)

    def run() -> tuple[np.ndarray, np.ndarray]:
        return np.linalg.eigh(hermitian)

    (eigenvalues, eigenvectors), timings = _time_repeated(run, repeats)
    reconstruction = (eigenvectors * eigenvalues) @ eigenvectors.conj().T
    reconstruction_error = _relative_norm(reconstruction - hermitian, hermitian)

    return ClassicalBenchmarkResult(
        algorithm="numpy.linalg.eigh",
        problem="dense-hermitian-eigendecomposition",
        matrix_dimension=hermitian.shape[0],
        repeats=repeats,
        best_time_seconds=min(timings),
        mean_time_seconds=float(np.mean(timings)),
        metrics={
            "min_eigenvalue": float(np.min(eigenvalues)),
            "max_eigenvalue": float(np.max(eigenvalues)),
            "spectral_width": float(np.max(eigenvalues) - np.min(eigenvalues)),
            "reconstruction_relative_error": reconstruction_error,
        },
        notes=(
            "Dense eigendecomposition is the classical reference for small "
            "Hermitian matrix-function examples.",
        ),
    ).as_report()


def dense_linear_solve_benchmark(
    matrix: np.ndarray | list[list[complex]],
    rhs: np.ndarray | list[complex],
    *,
    repeats: int = 3,
    qsvt_coeffs: np.ndarray | list[float] | None = None,
) -> dict[str, Any]:
    """
    Benchmark a dense direct linear solve with ``numpy.linalg.solve``.
    """
    operator = _validate_square_matrix(matrix)
    vector = _validate_vector(rhs, operator.shape[0], "rhs")

    def run() -> np.ndarray:
        return np.linalg.solve(operator, vector)

    solution, timings = _time_repeated(run, repeats)
    residual = operator @ solution - vector

    return ClassicalBenchmarkResult(
        algorithm="numpy.linalg.solve",
        problem="dense-linear-system",
        matrix_dimension=operator.shape[0],
        repeats=repeats,
        best_time_seconds=min(timings),
        mean_time_seconds=float(np.mean(timings)),
        metrics={
            "residual_norm": float(np.linalg.norm(residual)),
            "relative_residual_norm": _relative_norm(residual, vector),
            "solution_norm": float(np.linalg.norm(solution)),
            "condition_number_2": float(np.linalg.cond(operator)),
        },
        qsvt_proxy=_optional_qsvt_proxy(qsvt_coeffs, operator.shape[0]),
        notes=(
            "Dense direct solves are strong small-system baselines but scale "
            "cubically and do not exploit sparsity.",
        ),
    ).as_report()


def conjugate_gradient_solve(
    matrix: np.ndarray | list[list[complex]],
    rhs: np.ndarray | list[complex],
    *,
    tolerance: float = 1e-10,
    max_iterations: int | None = None,
) -> dict[str, Any]:
    """
    Solve a Hermitian positive-definite system by conjugate gradients.
    """
    operator = _validate_hermitian_matrix(matrix)
    vector = _validate_vector(rhs, operator.shape[0], "rhs")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    if max_iterations is None:
        max_iterations = 10 * operator.shape[0]
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")

    solution = np.zeros_like(vector)
    residual = vector - operator @ solution
    direction = residual.copy()
    residual_inner = np.vdot(residual, residual)
    initial_norm = float(np.sqrt(max(residual_inner.real, 0.0)))
    target = tolerance * max(initial_norm, 1.0)
    converged = initial_norm <= target
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        if converged:
            break
        matvec = operator @ direction
        curvature = np.vdot(direction, matvec)
        if abs(curvature) == 0.0:
            break
        step = residual_inner / curvature
        solution = solution + step * direction
        residual = residual - step * matvec
        next_inner = np.vdot(residual, residual)
        residual_norm = float(np.sqrt(max(next_inner.real, 0.0)))
        iterations = iteration
        if residual_norm <= target:
            converged = True
            residual_inner = next_inner
            break
        direction = residual + (next_inner / residual_inner) * direction
        residual_inner = next_inner

    final_residual = vector - operator @ solution
    final_norm = float(np.linalg.norm(final_residual))
    return {
        "solution": solution,
        "iterations": int(iterations),
        "converged": bool(converged),
        "residual_norm": final_norm,
        "relative_residual_norm": _relative_norm(final_residual, vector),
        "tolerance": float(tolerance),
        "max_iterations": int(max_iterations),
    }


def conjugate_gradient_benchmark(
    matrix: np.ndarray | list[list[complex]],
    rhs: np.ndarray | list[complex],
    *,
    tolerance: float = 1e-10,
    max_iterations: int | None = None,
    repeats: int = 3,
    qsvt_coeffs: np.ndarray | list[float] | None = None,
) -> dict[str, Any]:
    """
    Benchmark the package's pure-NumPy conjugate-gradient baseline.
    """
    operator = _validate_hermitian_matrix(matrix)
    vector = _validate_vector(rhs, operator.shape[0], "rhs")

    def run() -> dict[str, Any]:
        return conjugate_gradient_solve(
            operator,
            vector,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    result, timings = _time_repeated(run, repeats)
    eigenvalues = np.linalg.eigvalsh(operator)

    return ClassicalBenchmarkResult(
        algorithm="qsvt.benchmarks.conjugate_gradient_solve",
        problem="positive-definite-linear-system",
        matrix_dimension=operator.shape[0],
        repeats=repeats,
        best_time_seconds=min(timings),
        mean_time_seconds=float(np.mean(timings)),
        metrics={
            "iterations": result["iterations"],
            "converged": result["converged"],
            "residual_norm": result["residual_norm"],
            "relative_residual_norm": result["relative_residual_norm"],
            "tolerance": float(tolerance),
            "condition_number_2": _spd_condition_number(eigenvalues),
            "matvec_count": result["iterations"],
        },
        qsvt_proxy=_optional_qsvt_proxy(qsvt_coeffs, operator.shape[0]),
        notes=(
            "CG is a relevant sparse positive-definite baseline; this "
            "implementation uses dense NumPy matvecs for reproducibility.",
        ),
    ).as_report()


def polynomial_matrix_function_benchmark(
    matrix: np.ndarray | list[list[complex]],
    coeffs: np.ndarray | list[float],
    *,
    repeats: int = 3,
    include_qsvt_proxy: bool = True,
) -> dict[str, Any]:
    """
    Benchmark classical application of a polynomial to a Hermitian matrix.
    """
    hermitian = _validate_hermitian_matrix(matrix)
    coeff_arr = _validate_coeffs(coeffs)

    def run() -> np.ndarray:
        return apply_polynomial_to_hermitian(hermitian, coeff_arr)

    transformed, timings = _time_repeated(run, repeats)

    return ClassicalBenchmarkResult(
        algorithm="spectral-polynomial-evaluation",
        problem="polynomial-matrix-function",
        matrix_dimension=hermitian.shape[0],
        repeats=repeats,
        best_time_seconds=min(timings),
        mean_time_seconds=float(np.mean(timings)),
        metrics={
            "polynomial_degree": polynomial_degree(coeff_arr),
            "coefficient_count": int(coeff_arr.size),
            "output_frobenius_norm": float(np.linalg.norm(transformed)),
        },
        qsvt_proxy=(
            _optional_qsvt_proxy(coeff_arr, hermitian.shape[0])
            if include_qsvt_proxy
            else None
        ),
        notes=(
            "The classical reference diagonalizes the Hermitian input and "
            "evaluates the polynomial on its eigenvalues.",
        ),
    ).as_report()


def spectral_matrix_function_benchmark(
    matrix: np.ndarray | list[list[complex]],
    function: str | Callable[[np.ndarray], np.ndarray],
    *,
    repeats: int = 3,
    beta: float = 1.0,
    shift: float = 0.0,
) -> dict[str, Any]:
    """
    Benchmark a dense spectral matrix function via eigendecomposition.
    """
    hermitian = _validate_hermitian_matrix(matrix)
    name, func = _resolve_spectral_function(function, beta=beta, shift=shift)

    def run() -> np.ndarray:
        return apply_function_to_hermitian(hermitian, func)

    transformed, timings = _time_repeated(run, repeats)
    eigenvalues = np.linalg.eigvalsh(hermitian)

    return ClassicalBenchmarkResult(
        algorithm="dense-spectral-matrix-function",
        problem=f"{name}-matrix-function",
        matrix_dimension=hermitian.shape[0],
        repeats=repeats,
        best_time_seconds=min(timings),
        mean_time_seconds=float(np.mean(timings)),
        metrics={
            "function": name,
            "min_eigenvalue": float(np.min(eigenvalues)),
            "max_eigenvalue": float(np.max(eigenvalues)),
            "output_frobenius_norm": float(np.linalg.norm(transformed)),
        },
        notes=(
            "Dense spectral matrix functions are exact small-system references "
            "for QSVT polynomial approximation studies.",
        ),
    ).as_report()


def benchmark_summary_table(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert benchmark reports into compact table rows.
    """
    rows = []
    for report in reports:
        metrics = report.get("metrics", {})
        qsvt_proxy = report.get("qsvt_proxy") or {}
        resources = qsvt_proxy.get("resources", {})
        rows.append(
            {
                "algorithm": report.get("algorithm"),
                "problem": report.get("problem"),
                "matrix_dimension": report.get("matrix_dimension"),
                "best_time_seconds": report.get("best_time_seconds"),
                "mean_time_seconds": report.get("mean_time_seconds"),
                "relative_residual_norm": metrics.get("relative_residual_norm"),
                "relative_error": metrics.get("relative_error"),
                "condition_number_2": metrics.get("condition_number_2"),
                "polynomial_degree": metrics.get("polynomial_degree")
                or resources.get("degree"),
                "qsvt_signal_operator_calls": resources.get("signal_operator_calls"),
            }
        )
    return rows


def benchmark_environment_report() -> dict[str, object]:
    """
    Return version and platform metadata for interpreting timing snapshots.
    """
    return {
        "timing_kind": "python_wall_clock_microbenchmark",
        "timer": "time.perf_counter",
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "processor": platform.processor(),
        "stability_note": (
            "Timing fields are environment-dependent snapshots. Use metrics "
            "and QSVT proxy fields for stable schema checks; regenerate timings "
            "deliberately for benchmark studies."
        ),
    }


def write_benchmark_summary_csv(
    reports: list[dict[str, Any]],
    path: str | Path,
) -> Path:
    """
    Write compact benchmark report rows to a CSV file.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = benchmark_summary_table(reports)
    fieldnames = [
        "algorithm",
        "problem",
        "matrix_dimension",
        "best_time_seconds",
        "mean_time_seconds",
        "relative_residual_norm",
        "relative_error",
        "condition_number_2",
        "polynomial_degree",
        "qsvt_signal_operator_calls",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def plot_benchmark_timings(
    reports: list[dict[str, Any]],
    *,
    ax: Any | None = None,
    title: str = "Classical benchmark timings",
    use_log_scale: bool = True,
) -> tuple[Any, Any]:
    """
    Plot best classical wall-clock timings from benchmark reports.
    """
    rows = _benchmark_plot_rows(_validate_reports_for_plotting(reports))
    fig, axes = _figure_and_axes(ax)
    x_values = list(range(1, len(rows) + 1))
    values = [1000.0 * float(row["best_time_seconds"]) for row in rows]

    colors = _benchmark_plot_color_map(rows)
    for x_value, value, row in zip(x_values, values, rows, strict=True):
        axes.bar(
            x_value,
            value,
            color=colors[_benchmark_plot_key(row)],
            label=_benchmark_plot_label(row),
        )
    axes.set_xticks(x_values, [str(value) for value in x_values])
    axes.set_xlabel("benchmark case")
    axes.set_ylabel("best time (ms)")
    axes.set_title(title)
    axes.grid(axis="y", alpha=0.3)
    _place_benchmark_legend(axes)
    if use_log_scale and _should_use_log_scale(values):
        axes.set_yscale("log")
    fig.tight_layout()
    return fig, axes


def plot_qsvt_proxy_resources(
    reports: list[dict[str, Any]],
    *,
    ax: Any | None = None,
    title: str = "QSVT resource proxy",
) -> tuple[Any, Any]:
    """
    Plot QSVT signal-call proxies from benchmark reports.
    """
    rows = _benchmark_plot_rows(_validate_reports_for_plotting(reports))
    proxy_rows = [
        row for row in rows if row.get("qsvt_signal_operator_calls") is not None
    ]
    if not proxy_rows:
        raise ValueError("reports do not contain QSVT proxy resource fields.")

    fig, axes = _figure_and_axes(ax)
    x_values = list(range(1, len(proxy_rows) + 1))
    values = [int(row["qsvt_signal_operator_calls"]) for row in proxy_rows]

    colors = _benchmark_plot_color_map(rows)
    for x_value, value, row in zip(
        x_values,
        values,
        proxy_rows,
        strict=True,
    ):
        axes.bar(
            x_value,
            value,
            color=colors[_benchmark_plot_key(row)],
            label=_benchmark_plot_label(row),
        )
    axes.set_xticks(x_values, [str(value) for value in x_values])
    axes.set_xlabel("benchmark case")
    axes.set_ylabel("signal-operator calls")
    axes.set_title(title)
    axes.grid(axis="y", alpha=0.3)
    _place_benchmark_legend(axes)
    fig.tight_layout()
    return fig, axes


def _validate_square_matrix(matrix: np.ndarray | list[list[complex]]) -> np.ndarray:
    arr = np.asarray(matrix, dtype=complex if np.iscomplexobj(matrix) else float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("matrix must be a square 2D array.")
    if arr.shape[0] == 0:
        raise ValueError("matrix must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("matrix must contain only finite values.")
    return arr


def _validate_hermitian_matrix(matrix: np.ndarray | list[list[complex]]) -> np.ndarray:
    arr = _validate_square_matrix(matrix)
    if not np.allclose(arr, arr.conj().T):
        raise ValueError("matrix must be Hermitian.")
    return arr


def _validate_vector(
    values: np.ndarray | list[complex],
    dimension: int,
    name: str,
) -> np.ndarray:
    arr = np.asarray(values, dtype=complex if np.iscomplexobj(values) else float)
    if arr.ndim != 1 or arr.shape[0] != dimension:
        raise ValueError(f"{name} must be a vector matching matrix dimension.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _validate_coeffs(coeffs: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(coeffs, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("coeffs must be a non-empty one-dimensional sequence.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("coeffs must contain only finite values.")
    return arr


def _time_repeated(
    func: Callable[[], Any],
    repeats: int,
) -> tuple[Any, list[float]]:
    repeats = int(repeats)
    if repeats < 1:
        raise ValueError("repeats must be positive.")

    result = None
    timings: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        result = func()
        timings.append(perf_counter() - start)
    return result, timings


def _relative_norm(numerator: np.ndarray, denominator: np.ndarray) -> float:
    scale = float(np.linalg.norm(denominator))
    if scale == 0.0:
        return float(np.linalg.norm(numerator))
    return float(np.linalg.norm(numerator) / scale)


def _spd_condition_number(eigenvalues: np.ndarray) -> float:
    min_eval = float(np.min(eigenvalues))
    max_eval = float(np.max(eigenvalues))
    if min_eval <= 0.0:
        return float("inf")
    return max_eval / min_eval


def _optional_qsvt_proxy(
    coeffs: np.ndarray | list[float] | None,
    matrix_dimension: int,
) -> dict[str, Any] | None:
    if coeffs is None:
        return None
    return qsvt_resource_report(
        coeffs,
        matrix_dimension=matrix_dimension,
        attempt_synthesis=False,
    )


def _validate_reports_for_plotting(
    reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not reports:
        raise ValueError("reports must contain at least one benchmark report.")
    for report in reports:
        if "best_time_seconds" not in report or "problem" not in report:
            raise ValueError("each report must be a benchmark report dictionary.")
    return reports


def _benchmark_plot_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = benchmark_summary_table(reports)
    for index, (row, report) in enumerate(zip(rows, reports, strict=True), start=1):
        row["_plot_index"] = index
        row["_output_frobenius_norm"] = report.get("metrics", {}).get(
            "output_frobenius_norm"
        )
    return rows


def _figure_and_axes(ax: Any | None) -> tuple[Any, Any]:
    if ax is not None:
        return ax.figure, ax
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=(7.0, 3.8))


def _benchmark_plot_label(row: dict[str, Any]) -> str:
    short_algorithm = _benchmark_label_abbreviation(row)
    dimension = row.get("matrix_dimension")
    label = f"{short_algorithm}; n={dimension}"
    output_norm = row.get("_output_frobenius_norm")
    if output_norm is not None:
        label = f"{label}; out={output_norm:.3g}"
    return label


def _benchmark_label_abbreviation(row: dict[str, Any]) -> str:
    algorithm = str(row.get("algorithm") or "")
    problem = str(row.get("problem") or "")
    if algorithm == "numpy.linalg.solve" or problem == "dense-linear-system":
        return "DLS"
    if (
        "conjugate_gradient" in algorithm
        or problem == "positive-definite-linear-system"
    ):
        return "CGS"
    if algorithm == "dense-spectral-matrix-function":
        return "DSMF"
    if algorithm == "spectral-polynomial-evaluation":
        return "PME"
    return algorithm.rsplit(".", maxsplit=1)[-1] or "benchmark"


def _benchmark_plot_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("_plot_index"),)


def _place_benchmark_legend(ax: Any) -> None:
    ax.legend(
        title="case",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize="small",
    )


def _benchmark_plot_color_map(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], str]:
    # Okabe-Ito inspired palette: readable on white backgrounds and safer for
    # common color-vision deficiencies than arbitrary saturated hues.
    palette = [
        "#0072b2",
        "#d55e00",
        "#009e73",
        "#cc79a7",
        "#e69f00",
        "#56b4e9",
        "#f0e442",
        "#000000",
    ]
    color_map: dict[tuple[Any, ...], str] = {}
    for row in rows:
        key = _benchmark_plot_key(row)
        if key not in color_map:
            color_map[key] = palette[len(color_map) % len(palette)]
    return color_map


def _should_use_log_scale(values: list[float]) -> bool:
    positive = [value for value in values if value > 0.0]
    if len(positive) < 2:
        return False
    return max(positive) / min(positive) >= 20.0


def _resolve_spectral_function(
    function: str | Callable[[np.ndarray], np.ndarray],
    *,
    beta: float,
    shift: float,
) -> tuple[str, Callable[[np.ndarray], np.ndarray]]:
    if callable(function):
        return getattr(function, "__name__", "custom"), function

    name = function.replace("-", "_")
    if name == "inverse":
        return name, lambda x: 1.0 / (x + shift)
    if name == "sqrt":
        return name, np.sqrt
    if name == "sign":
        return name, np.sign
    if name in {"exponential", "imaginary_time"}:
        return "exponential", lambda x: np.exp(-beta * x)
    if name == "positive_projector":
        return name, lambda x: (x > shift).astype(float)
    raise ValueError(
        "function must be one of inverse, sqrt, sign, exponential, "
        "imaginary_time, or positive_projector."
    )


__all__ = [
    "ClassicalBenchmarkResult",
    "benchmark_summary_table",
    "conjugate_gradient_benchmark",
    "conjugate_gradient_solve",
    "dense_eigendecomposition_benchmark",
    "dense_linear_solve_benchmark",
    "plot_benchmark_timings",
    "plot_qsvt_proxy_resources",
    "polynomial_matrix_function_benchmark",
    "spectral_matrix_function_benchmark",
    "write_benchmark_summary_csv",
]
