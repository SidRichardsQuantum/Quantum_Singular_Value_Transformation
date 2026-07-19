"""Resolvent and spectral-density response workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._algorithm_reports import (
    algorithm_truth_contract,
    algorithm_workflow_schema_fields,
    scaled_operator_report,
)
from ._algorithm_shared import _normalize_state, _relative_error, _validate_state
from .diagnostics import operator_error
from .matrix_functions import (
    design_gaussian_window_polynomial,
    design_resolvent_polynomials,
)
from .rescaling import ScaledOperator, rescale_hermitian_to_unit_interval
from .spectral import apply_function_to_hermitian, apply_polynomial_to_hermitian


@dataclass(frozen=True)
class ResolventWorkflowResult:
    """
    Structured output from a Green's-function / resolvent workflow.
    """

    real_coeffs: np.ndarray
    imag_coeffs: np.ndarray
    scaled_operator: ScaledOperator
    source: np.ndarray | None
    polynomial_operator: np.ndarray
    reference_operator: np.ndarray
    polynomial_response: np.ndarray | None
    reference_response: np.ndarray | None
    omega: float
    eta: float
    degree: int
    operator_relative_error: float
    response_relative_error: float | None

    def as_report(self) -> dict[str, Any]:
        """
        Return a report-style dictionary for JSON conversion or persistence.
        """
        return {
            **algorithm_workflow_schema_fields(),
            "mode": "resolvent-workflow",
            "implementation_kind": "dense-spectral-polynomial-workflow",
            "truth_contract": algorithm_truth_contract(
                "resolvent-workflow",
                target="Green's-function resolvent matrix function",
            ),
            "omega": self.omega,
            "eta": self.eta,
            "degree": self.degree,
            "real_coeffs": self.real_coeffs,
            "imag_coeffs": self.imag_coeffs,
            "scaled_operator": scaled_operator_report(self.scaled_operator),
            "source": self.source,
            "polynomial_operator": self.polynomial_operator,
            "reference_operator": self.reference_operator,
            "polynomial_response": self.polynomial_response,
            "reference_response": self.reference_response,
            "operator_relative_error": self.operator_relative_error,
            "response_relative_error": self.response_relative_error,
        }


@dataclass(frozen=True)
class SpectralDensityWorkflowResult:
    """
    Structured output from a Gaussian spectral-density workflow.
    """

    centers: np.ndarray
    width: float
    degree: int
    coeffs_by_center: list[np.ndarray]
    scaled_operator: ScaledOperator
    polynomial_trace_density: np.ndarray
    reference_trace_density: np.ndarray
    trace_density_error: float
    state: np.ndarray | None = None
    polynomial_state_weights: np.ndarray | None = None
    reference_state_weights: np.ndarray | None = None
    state_weight_error: float | None = None

    def as_report(self) -> dict[str, Any]:
        """
        Return a report-style dictionary for JSON conversion or persistence.
        """
        return {
            **algorithm_workflow_schema_fields(),
            "mode": "spectral-density-workflow",
            "implementation_kind": "dense-spectral-polynomial-workflow",
            "truth_contract": algorithm_truth_contract(
                "spectral-density-workflow",
                target="Gaussian-window spectral-density estimate",
            ),
            "centers": self.centers,
            "width": self.width,
            "degree": self.degree,
            "coeffs_by_center": self.coeffs_by_center,
            "scaled_operator": scaled_operator_report(self.scaled_operator),
            "polynomial_trace_density": self.polynomial_trace_density,
            "reference_trace_density": self.reference_trace_density,
            "trace_density_error": self.trace_density_error,
            "state": self.state,
            "polynomial_state_weights": self.polynomial_state_weights,
            "reference_state_weights": self.reference_state_weights,
            "state_weight_error": self.state_weight_error,
        }


def _state_weight_error(
    state: np.ndarray | None,
    reference: np.ndarray | None,
    approximate: np.ndarray | None,
) -> float | None:
    if state is None:
        return None
    if reference is None or approximate is None:
        raise ValueError("state-weight arrays are required when state is provided.")
    return _relative_error(reference, approximate)


def _gaussian_window_function(
    center: float,
    width: float,
) -> Callable[[np.ndarray], np.ndarray]:
    def gaussian_window(x: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * ((x - center) / width) ** 2)

    return gaussian_window


def resolvent_workflow(
    matrix: np.ndarray,
    *,
    omega: float,
    eta: float,
    degree: int,
    source: np.ndarray | None = None,
    num_points: int = 2001,
) -> ResolventWorkflowResult:
    """
    Approximate the Green's-function resolvent ``(omega+i eta-H)^-1``.
    """
    scaled = rescale_hermitian_to_unit_interval(matrix)
    real_coeffs, imag_coeffs = design_resolvent_polynomials(
        omega=omega,
        eta=eta,
        scale=scaled.scale,
        offset=scaled.offset,
        degree=degree,
        num_points=num_points,
    )
    polynomial_operator = apply_polynomial_to_hermitian(
        scaled.matrix,
        real_coeffs,
    ) + 1j * apply_polynomial_to_hermitian(scaled.matrix, imag_coeffs)
    reference_operator = apply_function_to_hermitian(
        np.asarray(matrix),
        lambda x: 1.0 / (float(omega) + 1j * float(eta) - x),
    )

    source_vec = None
    polynomial_response = None
    reference_response = None
    response_error = None
    if source is not None:
        source_vec = _validate_state(source, scaled.matrix.shape[0], name="source")
        polynomial_response = polynomial_operator @ source_vec
        reference_response = reference_operator @ source_vec
        response_error = _relative_error(reference_response, polynomial_response)

    return ResolventWorkflowResult(
        real_coeffs=real_coeffs,
        imag_coeffs=imag_coeffs,
        scaled_operator=scaled,
        source=source_vec,
        polynomial_operator=polynomial_operator,
        reference_operator=reference_operator,
        polynomial_response=polynomial_response,
        reference_response=reference_response,
        omega=float(omega),
        eta=float(eta),
        degree=int(degree),
        operator_relative_error=operator_error(reference_operator, polynomial_operator),
        response_relative_error=response_error,
    )


def spectral_density_workflow(
    matrix: np.ndarray,
    centers: np.ndarray,
    *,
    width: float,
    degree: int,
    state: np.ndarray | None = None,
    num_points: int = 2001,
) -> SpectralDensityWorkflowResult:
    """
    Estimate Gaussian-window trace density and optional state spectral weights.

    The ``centers`` and ``width`` parameters are expressed in the original
    spectral coordinate of ``matrix``.
    """
    scaled = rescale_hermitian_to_unit_interval(matrix)
    center_values = np.asarray(centers, dtype=float)
    if center_values.ndim != 1:
        raise ValueError("centers must be a one-dimensional array.")
    if width <= 0.0:
        raise ValueError("width must be positive.")

    state_vec = None
    if state is not None:
        state_vec = _normalize_state(_validate_state(state, scaled.matrix.shape[0]))

    coeffs_by_center = []
    polynomial_density = []
    reference_density = []
    polynomial_weights = []
    reference_weights = []

    dimension = scaled.matrix.shape[0]
    for physical_center in center_values:
        scaled_center = (float(physical_center) - scaled.offset) / scaled.scale
        scaled_width = float(width) / scaled.scale
        coeffs = design_gaussian_window_polynomial(
            center=scaled_center,
            width=scaled_width,
            degree=degree,
            num_points=num_points,
        )
        coeffs_by_center.append(coeffs)
        polynomial_operator = apply_polynomial_to_hermitian(scaled.matrix, coeffs)
        reference_operator = apply_function_to_hermitian(
            np.asarray(matrix),
            _gaussian_window_function(float(physical_center), float(width)),
        )
        polynomial_density.append(np.trace(polynomial_operator) / dimension)
        reference_density.append(np.trace(reference_operator) / dimension)
        if state_vec is not None:
            polynomial_weights.append(
                np.vdot(state_vec, polynomial_operator @ state_vec)
            )
            reference_weights.append(np.vdot(state_vec, reference_operator @ state_vec))

    polynomial_density_arr = np.real_if_close(np.asarray(polynomial_density))
    reference_density_arr = np.real_if_close(np.asarray(reference_density))
    polynomial_weights_arr = (
        np.real_if_close(np.asarray(polynomial_weights))
        if state_vec is not None
        else None
    )
    reference_weights_arr = (
        np.real_if_close(np.asarray(reference_weights))
        if state_vec is not None
        else None
    )

    return SpectralDensityWorkflowResult(
        centers=center_values,
        width=float(width),
        degree=int(degree),
        coeffs_by_center=coeffs_by_center,
        scaled_operator=scaled,
        polynomial_trace_density=polynomial_density_arr,
        reference_trace_density=reference_density_arr,
        trace_density_error=_relative_error(
            reference_density_arr,
            polynomial_density_arr,
        ),
        state=state_vec,
        polynomial_state_weights=polynomial_weights_arr,
        reference_state_weights=reference_weights_arr,
        state_weight_error=_state_weight_error(
            state_vec,
            reference_weights_arr,
            polynomial_weights_arr,
        ),
    )
