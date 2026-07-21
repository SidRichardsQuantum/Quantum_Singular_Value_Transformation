"""Ground-state, threshold, counting, and amplification workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._algorithm_reports import (
    algorithm_truth_contract,
    algorithm_workflow_schema_fields,
    scaled_operator_report,
)
from ._algorithm_shared import _normalize_state, _state_error, _validate_state
from .design import (
    design_interval_projector_diagnostics,
    design_interval_projector_polynomial,
)
from .diagnostics import expectation_value, ground_state_overlap, operator_error
from .matrix_functions import design_gaussian_window_polynomial
from .rescaling import ScaledOperator, rescale_hermitian_to_unit_interval
from .spectral import (
    apply_function_to_hermitian,
    apply_polynomial_to_hermitian,
    eigh_hermitian,
)


@dataclass(frozen=True)
class GroundStateFilteringWorkflowResult:
    """
    Structured output from a Gaussian ground-state filtering workflow.
    """

    coeffs: np.ndarray
    scaled_operator: ScaledOperator
    input_state: np.ndarray
    filtered_state: np.ndarray
    unnormalized_filtered_state: np.ndarray
    reference_filtered_state: np.ndarray
    ground_energy: float
    filtered_energy: float | complex
    ground_state_overlap: float
    reference_state_error: float
    center: float
    width: float
    degree: int
    polynomial_operator: np.ndarray
    reference_operator: np.ndarray
    operator_relative_error: float

    def as_report(self) -> dict[str, Any]:
        """
        Return a report-style dictionary for JSON conversion or persistence.
        """
        return {
            **algorithm_workflow_schema_fields(),
            "mode": "ground-state-filtering-workflow",
            "implementation_kind": "dense-spectral-polynomial-workflow",
            "truth_contract": algorithm_truth_contract(
                "ground-state-filtering-workflow",
                target="low-energy Gaussian spectral filtering",
                polynomials={"gaussian_filter": self.coeffs},
            ),
            "degree": self.degree,
            "center": self.center,
            "width": self.width,
            "coeffs": self.coeffs,
            "scaled_operator": scaled_operator_report(self.scaled_operator),
            "input_state": self.input_state,
            "filtered_state": self.filtered_state,
            "unnormalized_filtered_state": self.unnormalized_filtered_state,
            "reference_filtered_state": self.reference_filtered_state,
            "ground_energy": self.ground_energy,
            "filtered_energy": self.filtered_energy,
            "ground_state_overlap": self.ground_state_overlap,
            "reference_state_error": self.reference_state_error,
            "polynomial_operator": self.polynomial_operator,
            "reference_operator": self.reference_operator,
            "operator_relative_error": self.operator_relative_error,
        }


@dataclass(frozen=True)
class SpectralThresholdingWorkflowResult:
    """
    Structured output from a spectral interval-projector workflow.
    """

    coeffs: np.ndarray
    scaled_operator: ScaledOperator
    polynomial_projector: np.ndarray
    reference_projector: np.ndarray
    lower: float
    upper: float
    scaled_lower: float
    scaled_upper: float
    sharpness: float
    degree: int
    exact_rank: int
    polynomial_rank_proxy: float | complex
    operator_relative_error: float
    leakage_outside_interval: float
    diagnostics: dict[str, object]
    state: np.ndarray | None = None
    polynomial_state_weight: float | complex | None = None
    reference_state_weight: float | complex | None = None
    state_weight_error: float | None = None

    def as_report(self) -> dict[str, Any]:
        """
        Return a report-style dictionary for JSON conversion or persistence.
        """
        return {
            **algorithm_workflow_schema_fields(),
            "mode": "spectral-thresholding-workflow",
            "implementation_kind": "dense-spectral-polynomial-workflow",
            "truth_contract": algorithm_truth_contract(
                "spectral-thresholding-workflow",
                target="smooth spectral interval-projector approximation",
                polynomials={"interval_projector": self.coeffs},
            ),
            "degree": self.degree,
            "lower": self.lower,
            "upper": self.upper,
            "scaled_lower": self.scaled_lower,
            "scaled_upper": self.scaled_upper,
            "sharpness": self.sharpness,
            "coeffs": self.coeffs,
            "scaled_operator": scaled_operator_report(self.scaled_operator),
            "polynomial_projector": self.polynomial_projector,
            "reference_projector": self.reference_projector,
            "exact_rank": self.exact_rank,
            "polynomial_rank_proxy": self.polynomial_rank_proxy,
            "operator_relative_error": self.operator_relative_error,
            "leakage_outside_interval": self.leakage_outside_interval,
            "diagnostics": self.diagnostics,
            "state": self.state,
            "polynomial_state_weight": self.polynomial_state_weight,
            "reference_state_weight": self.reference_state_weight,
            "state_weight_error": self.state_weight_error,
        }


@dataclass(frozen=True)
class SpectralCountingWorkflowResult:
    """
    Structured output from a spectral interval-counting workflow.
    """

    coeffs: np.ndarray
    scaled_operator: ScaledOperator
    polynomial_projector: np.ndarray
    reference_projector: np.ndarray
    lower: float
    upper: float
    scaled_lower: float
    scaled_upper: float
    sharpness: float
    degree: int
    exact_count: int
    polynomial_count: float | complex
    count_error: float
    stochastic_count: float | None = None
    stochastic_count_error: float | None = None
    probe_count: int | None = None

    def as_report(self) -> dict[str, Any]:
        return {
            **algorithm_workflow_schema_fields(),
            "mode": "spectral-counting-workflow",
            "implementation_kind": "dense-spectral-polynomial-workflow",
            "truth_contract": algorithm_truth_contract(
                "spectral-counting-workflow",
                target="spectral interval counting through smooth projector traces",
                polynomials={"interval_projector": self.coeffs},
            ),
            "degree": self.degree,
            "lower": self.lower,
            "upper": self.upper,
            "scaled_lower": self.scaled_lower,
            "scaled_upper": self.scaled_upper,
            "sharpness": self.sharpness,
            "coeffs": self.coeffs,
            "scaled_operator": scaled_operator_report(self.scaled_operator),
            "polynomial_projector": self.polynomial_projector,
            "reference_projector": self.reference_projector,
            "exact_count": self.exact_count,
            "polynomial_count": self.polynomial_count,
            "count_error": self.count_error,
            "stochastic_count": self.stochastic_count,
            "stochastic_count_error": self.stochastic_count_error,
            "probe_count": self.probe_count,
        }


@dataclass(frozen=True)
class FixedPointAmplificationWorkflowResult:
    """
    Structured output from a monotone fixed-point spectral amplification workflow.
    """

    coeffs: np.ndarray
    score_operator: np.ndarray
    input_state: np.ndarray
    amplified_state: np.ndarray
    reference_state: np.ndarray
    polynomial_operator: np.ndarray
    reference_operator: np.ndarray
    rounds: int
    degree: int
    initial_score: float | complex
    amplified_score: float | complex
    reference_score: float | complex
    state_relative_error: float
    operator_relative_error: float

    def as_report(self) -> dict[str, Any]:
        return {
            **algorithm_workflow_schema_fields(),
            "mode": "fixed-point-amplification-workflow",
            "implementation_kind": "dense-spectral-polynomial-workflow",
            "truth_contract": algorithm_truth_contract(
                "fixed-point-amplification-workflow",
                target="monotone fixed-point polynomial amplification of scores",
                polynomials={"score_amplification": self.coeffs},
                polynomial_design_domains={"score_amplification": (0.0, 1.0)},
            ),
            "rounds": self.rounds,
            "degree": self.degree,
            "coeffs": self.coeffs,
            "score_operator": self.score_operator,
            "input_state": self.input_state,
            "amplified_state": self.amplified_state,
            "reference_state": self.reference_state,
            "polynomial_operator": self.polynomial_operator,
            "reference_operator": self.reference_operator,
            "initial_score": self.initial_score,
            "amplified_score": self.amplified_score,
            "reference_score": self.reference_score,
            "state_relative_error": self.state_relative_error,
            "operator_relative_error": self.operator_relative_error,
        }


def spectral_counting_workflow(
    matrix: np.ndarray,
    *,
    lower: float,
    upper: float,
    degree: int,
    sharpness: float = 12.0,
    num_points: int = 2001,
    probe_count: int | None = None,
    random_seed: int | None = None,
) -> SpectralCountingWorkflowResult:
    """
    Count eigenvalues in an interval using a smooth polynomial projector trace.
    """
    lower = float(lower)
    upper = float(upper)
    if not lower < upper:
        raise ValueError("lower must be less than upper.")
    evals, _ = eigh_hermitian(matrix)
    if lower <= float(evals[0]) or upper >= float(evals[-1]):
        raise ValueError(
            "lower and upper must lie strictly inside the matrix spectral range."
        )
    scaled = rescale_hermitian_to_unit_interval(matrix)
    scaled_lower = (lower - scaled.offset) / scaled.scale
    scaled_upper = (upper - scaled.offset) / scaled.scale
    coeffs = design_interval_projector_polynomial(
        lower=scaled_lower,
        upper=scaled_upper,
        degree=degree,
        sharpness=sharpness,
        num_points=num_points,
    )
    polynomial_projector = apply_polynomial_to_hermitian(scaled.matrix, coeffs)
    reference_projector = apply_function_to_hermitian(
        np.asarray(matrix),
        lambda x: np.where((lower <= x) & (x <= upper), 1.0, 0.0),
    )
    exact_count = int(np.count_nonzero((evals >= lower) & (evals <= upper)))
    polynomial_count = np.real_if_close(np.trace(polynomial_projector)).item()

    stochastic_count = None
    stochastic_error = None
    if probe_count is not None:
        probes = int(probe_count)
        if probes <= 0:
            raise ValueError("probe_count must be positive when provided.")
        rng = np.random.default_rng(random_seed)
        estimates = []
        dimension = scaled.matrix.shape[0]
        for _ in range(probes):
            signs = rng.choice([-1.0, 1.0], size=dimension)
            estimates.append(np.vdot(signs, polynomial_projector @ signs))
        stochastic_count = float(np.real_if_close(np.mean(estimates)).item())
        stochastic_error = float(abs(stochastic_count - exact_count))

    return SpectralCountingWorkflowResult(
        coeffs=coeffs,
        scaled_operator=scaled,
        polynomial_projector=polynomial_projector,
        reference_projector=reference_projector,
        lower=lower,
        upper=upper,
        scaled_lower=float(scaled_lower),
        scaled_upper=float(scaled_upper),
        sharpness=float(sharpness),
        degree=int(degree),
        exact_count=exact_count,
        polynomial_count=polynomial_count,
        count_error=float(abs(polynomial_count - exact_count)),
        stochastic_count=stochastic_count,
        stochastic_count_error=stochastic_error,
        probe_count=probe_count,
    )


def fixed_point_amplification_workflow(
    score_operator: np.ndarray,
    state: np.ndarray,
    *,
    rounds: int,
) -> FixedPointAmplificationWorkflowResult:
    """
    Apply the monotone fixed-point polynomial ``1 - (1 - x)^rounds``.

    The score operator must be positive semidefinite with spectrum in
    ``[0, 1]``. This is a robust projector/score amplification primitive for
    finite spectral workflows, not a full Grover iterate implementation.
    """
    if rounds <= 0:
        raise ValueError("rounds must be positive.")
    evals, _ = eigh_hermitian(score_operator)
    if evals[0] < -1e-10 or evals[-1] > 1.0 + 1e-10:
        raise ValueError("score_operator spectrum must lie in [0, 1].")
    A = np.asarray(score_operator)
    psi = _normalize_state(_validate_state(state, A.shape[0]))

    coeffs = np.zeros(rounds + 1, dtype=float)
    for k in range(1, rounds + 1):
        coeffs[k] = ((-1.0) ** (k + 1)) * math.comb(rounds, k)

    polynomial_operator = apply_polynomial_to_hermitian(A, coeffs)
    reference_operator = apply_function_to_hermitian(
        A,
        lambda x: 1.0 - (1.0 - x) ** rounds,
    )
    amplified = _normalize_state(polynomial_operator @ psi)
    reference = _normalize_state(reference_operator @ psi)

    return FixedPointAmplificationWorkflowResult(
        coeffs=coeffs,
        score_operator=A,
        input_state=psi,
        amplified_state=amplified,
        reference_state=reference,
        polynomial_operator=polynomial_operator,
        reference_operator=reference_operator,
        rounds=int(rounds),
        degree=int(rounds),
        initial_score=np.real_if_close(np.vdot(psi, A @ psi)).item(),
        amplified_score=np.real_if_close(np.vdot(amplified, A @ amplified)).item(),
        reference_score=np.real_if_close(np.vdot(reference, A @ reference)).item(),
        state_relative_error=_state_error(reference, amplified),
        operator_relative_error=operator_error(reference_operator, polynomial_operator),
    )


def ground_state_filtering_workflow(
    matrix: np.ndarray,
    state: np.ndarray,
    *,
    degree: int,
    width: float = 0.25,
    center: float = -1.0,
    num_points: int = 2001,
) -> GroundStateFilteringWorkflowResult:
    """
    Apply a Gaussian low-energy filter to a trial state.

    The Hamiltonian is affinely mapped to ``[-1, 1]`` and filtered with a
    Gaussian window centered near the low-energy edge by default. The
    ``center`` and ``width`` parameters are expressed in the scaled spectral
    coordinate.
    """
    evals, _ = eigh_hermitian(matrix)
    scaled = rescale_hermitian_to_unit_interval(matrix)
    psi = _validate_state(state, scaled.matrix.shape[0])
    psi = _normalize_state(psi)

    if width <= 0.0:
        raise ValueError("width must be positive.")

    coeffs = design_gaussian_window_polynomial(
        center=center,
        width=width,
        degree=degree,
        num_points=num_points,
    )
    polynomial_operator = apply_polynomial_to_hermitian(scaled.matrix, coeffs)
    reference_operator = apply_function_to_hermitian(
        scaled.matrix,
        lambda x: np.exp(-0.5 * ((x - float(center)) / float(width)) ** 2),
    )

    unnormalized = polynomial_operator @ psi
    filtered = _normalize_state(unnormalized)
    reference_filtered = _normalize_state(reference_operator @ psi)

    return GroundStateFilteringWorkflowResult(
        coeffs=coeffs,
        scaled_operator=scaled,
        input_state=psi,
        filtered_state=filtered,
        unnormalized_filtered_state=unnormalized,
        reference_filtered_state=reference_filtered,
        ground_energy=float(evals[0]),
        filtered_energy=expectation_value(np.asarray(matrix), filtered),
        ground_state_overlap=ground_state_overlap(np.asarray(matrix), filtered),
        reference_state_error=_state_error(reference_filtered, filtered),
        center=float(center),
        width=float(width),
        degree=int(degree),
        polynomial_operator=polynomial_operator,
        reference_operator=reference_operator,
        operator_relative_error=operator_error(reference_operator, polynomial_operator),
    )


def spectral_thresholding_workflow(
    matrix: np.ndarray,
    *,
    lower: float,
    upper: float,
    degree: int,
    sharpness: float = 12.0,
    state: np.ndarray | None = None,
    num_points: int = 2001,
    bounded_num_points: int = 4001,
) -> SpectralThresholdingWorkflowResult:
    """
    Approximate a spectral interval projector with a QSVT-style polynomial.

    The physical interval ``[lower, upper]`` is mapped into the scaled
    ``[-1, 1]`` coordinate before designing a smooth interval-projector
    polynomial. The exact reference is the hard spectral projector onto
    eigenvalues of ``matrix`` that lie inside the physical interval.
    """
    lower = float(lower)
    upper = float(upper)
    if not lower < upper:
        raise ValueError("lower must be less than upper.")

    evals, _ = eigh_hermitian(matrix)
    if lower <= float(evals[0]) or upper >= float(evals[-1]):
        raise ValueError(
            "lower and upper must lie strictly inside the matrix spectral range."
        )

    scaled = rescale_hermitian_to_unit_interval(matrix)
    scaled_lower = (lower - scaled.offset) / scaled.scale
    scaled_upper = (upper - scaled.offset) / scaled.scale
    if not (-1.0 < scaled_lower < scaled_upper < 1.0):
        raise ValueError("scaled interval must lie strictly inside [-1, 1].")

    coeffs = design_interval_projector_polynomial(
        lower=scaled_lower,
        upper=scaled_upper,
        degree=degree,
        sharpness=sharpness,
        num_points=num_points,
    )
    diagnostics = design_interval_projector_diagnostics(
        lower=scaled_lower,
        upper=scaled_upper,
        degree=degree,
        sharpness=sharpness,
        num_points=num_points,
        bounded_num_points=bounded_num_points,
    )
    polynomial_projector = apply_polynomial_to_hermitian(scaled.matrix, coeffs)
    reference_projector = apply_function_to_hermitian(
        np.asarray(matrix),
        lambda x: np.where((lower <= x) & (x <= upper), 1.0, 0.0),
    )

    inside_mask = (evals >= lower) & (evals <= upper)
    exact_rank = int(np.count_nonzero(inside_mask))
    outside_projector = apply_function_to_hermitian(
        np.asarray(matrix),
        lambda x: np.where((x < lower) | (x > upper), 1.0, 0.0),
    )
    leakage = float(np.linalg.norm(outside_projector @ polynomial_projector))

    state_vec = None
    polynomial_weight = None
    reference_weight = None
    state_weight_error = None
    if state is not None:
        state_vec = _normalize_state(_validate_state(state, scaled.matrix.shape[0]))
        polynomial_weight = np.real_if_close(
            np.vdot(state_vec, polynomial_projector @ state_vec)
        ).item()
        reference_weight = np.real_if_close(
            np.vdot(state_vec, reference_projector @ state_vec)
        ).item()
        state_weight_error = float(abs(polynomial_weight - reference_weight))

    return SpectralThresholdingWorkflowResult(
        coeffs=coeffs,
        scaled_operator=scaled,
        polynomial_projector=polynomial_projector,
        reference_projector=reference_projector,
        lower=lower,
        upper=upper,
        scaled_lower=float(scaled_lower),
        scaled_upper=float(scaled_upper),
        sharpness=float(sharpness),
        degree=int(degree),
        exact_rank=exact_rank,
        polynomial_rank_proxy=np.real_if_close(np.trace(polynomial_projector)).item(),
        operator_relative_error=operator_error(
            reference_projector,
            polynomial_projector,
        ),
        leakage_outside_interval=leakage,
        diagnostics=diagnostics,
        state=state_vec,
        polynomial_state_weight=polynomial_weight,
        reference_state_weight=reference_weight,
        state_weight_error=state_weight_error,
    )
