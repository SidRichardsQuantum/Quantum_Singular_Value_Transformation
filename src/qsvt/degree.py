"""Tolerance-driven polynomial degree selection helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .polynomials import polynomial_degree


@dataclass(frozen=True)
class DegreeSearchCandidate:
    """One evaluated polynomial-degree candidate."""

    requested_degree: int
    polynomial_degree: int | None
    error: float | None
    met_tolerance: bool
    metadata: dict[str, object]
    error_type: str | None = None
    error_message: str | None = None

    def as_report(self) -> dict[str, object]:
        """Return the candidate as plain report data."""
        return {
            "requested_degree": self.requested_degree,
            "polynomial_degree": self.polynomial_degree,
            "error": self.error,
            "met_tolerance": self.met_tolerance,
            "metadata": self.metadata,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class DegreeSearchResult:
    """Result of searching for the smallest degree meeting an error target."""

    tolerance: float
    metric: str
    candidates: tuple[DegreeSearchCandidate, ...]
    chosen_degree: int | None
    chosen_polynomial_degree: int | None
    chosen_coeffs: np.ndarray | None
    achieved_error: float | None
    met_tolerance: bool

    def as_report(self) -> dict[str, object]:
        """Return a machine-readable degree-search report."""
        return {
            "mode": "polynomial-degree-search",
            "tolerance": self.tolerance,
            "metric": self.metric,
            "chosen_degree": self.chosen_degree,
            "chosen_polynomial_degree": self.chosen_polynomial_degree,
            "chosen_coeffs": self.chosen_coeffs,
            "achieved_error": self.achieved_error,
            "met_tolerance": self.met_tolerance,
            "candidates": [candidate.as_report() for candidate in self.candidates],
        }


def search_polynomial_degree(
    builder: Callable[[int], np.ndarray],
    evaluator: Callable[[np.ndarray, int], float | tuple[float, dict[str, object]]],
    *,
    tolerance: float,
    degrees: Iterable[int],
    metric: str = "error",
    stop_at_first: bool = True,
) -> DegreeSearchResult:
    """Search candidate degrees and retain failures and numerical diagnostics.

    ``builder`` receives the requested degree. ``evaluator`` receives the built
    coefficient array and requested degree, and returns either an error or an
    ``(error, metadata)`` pair. The first passing candidate is selected by
    default; if no candidate passes, the lowest-error successful candidate is
    returned with ``met_tolerance=False``.
    """
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite.")
    requested = tuple(int(degree) for degree in degrees)
    if not requested:
        raise ValueError("degrees must contain at least one candidate.")
    if any(degree < 0 for degree in requested):
        raise ValueError("candidate degrees must be non-negative.")
    if len(set(requested)) != len(requested):
        raise ValueError("candidate degrees must be distinct.")

    candidates: list[DegreeSearchCandidate] = []
    successful: list[tuple[DegreeSearchCandidate, np.ndarray]] = []
    selected: tuple[DegreeSearchCandidate, np.ndarray] | None = None
    for requested_degree in requested:
        try:
            coeffs = np.asarray(builder(requested_degree), dtype=float)
            if coeffs.ndim != 1 or coeffs.size == 0:
                raise ValueError(
                    "degree builder must return a non-empty one-dimensional array."
                )
            if not np.all(np.isfinite(coeffs)):
                raise ValueError("degree builder must return finite coefficients.")
            evaluated = evaluator(coeffs, requested_degree)
            if isinstance(evaluated, tuple):
                error, metadata = evaluated
            else:
                error, metadata = evaluated, {}
            error = float(error)
            if not np.isfinite(error) or error < 0.0:
                raise ValueError(
                    "degree evaluator must return a finite non-negative error."
                )
            candidate = DegreeSearchCandidate(
                requested_degree=requested_degree,
                polynomial_degree=polynomial_degree(coeffs),
                error=error,
                met_tolerance=error <= tolerance,
                metadata=dict(metadata),
            )
            candidates.append(candidate)
            successful.append((candidate, coeffs))
            if candidate.met_tolerance and selected is None:
                selected = (candidate, coeffs)
                if stop_at_first:
                    break
        except Exception as exc:
            candidates.append(
                DegreeSearchCandidate(
                    requested_degree=requested_degree,
                    polynomial_degree=None,
                    error=None,
                    met_tolerance=False,
                    metadata={},
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )

    if selected is None and successful:
        selected = min(
            successful,
            key=lambda item: (
                float(item[0].error) if item[0].error is not None else np.inf,
                item[0].requested_degree,
            ),
        )
    if selected is None:
        return DegreeSearchResult(
            tolerance=tolerance,
            metric=str(metric),
            candidates=tuple(candidates),
            chosen_degree=None,
            chosen_polynomial_degree=None,
            chosen_coeffs=None,
            achieved_error=None,
            met_tolerance=False,
        )

    chosen, coeffs = selected
    return DegreeSearchResult(
        tolerance=tolerance,
        metric=str(metric),
        candidates=tuple(candidates),
        chosen_degree=chosen.requested_degree,
        chosen_polynomial_degree=chosen.polynomial_degree,
        chosen_coeffs=coeffs,
        achieved_error=chosen.error,
        met_tolerance=chosen.met_tolerance,
    )


def search_design_degree(
    kind: str,
    *,
    tolerance: float,
    min_degree: int = 2,
    max_degree: int = 32,
    degree_step: int = 2,
    metric: str = "max_error",
    **design_kwargs: Any,
) -> DegreeSearchResult:
    """Select a degree for a public :func:`qsvt.workflow.design_workflow` target."""
    if degree_step <= 0:
        raise ValueError("degree_step must be positive.")
    if min_degree < 0 or max_degree < min_degree:
        raise ValueError("degree bounds must satisfy 0 <= min_degree <= max_degree.")

    from .workflow import design_workflow

    reports: dict[int, Any] = {}

    def build(degree: int) -> np.ndarray:
        result = design_workflow(
            kind,  # type: ignore[arg-type]
            degree=degree,
            attempt_synthesis=False,
            **design_kwargs,
        )
        reports[degree] = result
        return result.coeffs

    def evaluate(
        coeffs: np.ndarray,
        degree: int,
    ) -> tuple[float, dict[str, object]]:
        del coeffs
        result = reports[degree]
        diagnostics = result.diagnostics
        if metric not in diagnostics:
            raise ValueError(f"design diagnostics do not contain metric {metric!r}.")
        return float(diagnostics[metric]), {
            "bounded": bool(diagnostics.get("bounded", False)),
            "bounded_margin": diagnostics.get("bounded_margin"),
        }

    return search_polynomial_degree(
        build,
        evaluate,
        tolerance=tolerance,
        degrees=range(min_degree, max_degree + 1, degree_step),
        metric=metric,
    )


__all__ = [
    "DegreeSearchCandidate",
    "DegreeSearchResult",
    "search_design_degree",
    "search_polynomial_degree",
]
