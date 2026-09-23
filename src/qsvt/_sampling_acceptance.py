"""Conservative acceptance of conditional computational-basis probabilities."""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_sampling_contract(tolerance: float, confidence: float) -> None:
    """Validate probability-error units independently of statevector tolerances."""
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("sampling_tolerance must be finite and between zero and one.")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("sampling_confidence must be finite and between zero and one.")


def probability_acceptance(
    execution: Any, reference_state: np.ndarray, *, tolerance: float, confidence: float
) -> dict[str, Any]:
    """Bound each conditional probability error with simultaneous Hoeffding bounds.

    Split the failure budget between the binomial postselection rate and all
    conditional basis outcomes. Conditional on the observed accepted count,
    the retained outcomes are iid draws from the postselected distribution.
    This validates probabilities, not phases, amplitudes, or general observables.
    """
    validate_sampling_contract(tolerance, confidence)
    evidence: dict[str, Any] = {
        "status": "unavailable",
        "passed": False,
        "measurement": "conditional_computational_basis_probabilities",
        "method": "simultaneous_hoeffding",
        "confidence": confidence,
        "tolerance": tolerance,
        "shots": execution.shots,
        "assumption": "independent identically distributed shots",
        "reference": "normalized_exact_workflow_output",
        "statevector_validated": False,
    }
    if not execution.succeeded or execution.probabilities is None:
        evidence["reason"] = "execution_evidence_unavailable"
        return evidence
    shots = execution.shots
    probabilities = np.asarray(execution.probabilities, dtype=float)
    reference = np.asarray(reference_state, dtype=complex)
    if (
        isinstance(shots, (bool, np.bool_))
        or not isinstance(shots, (int, np.integer))
        or shots <= 0
        or probabilities.ndim != 1
        or reference.ndim != 1
        or reference.size == 0
        or probabilities.size < reference.size
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.isclose(probabilities.sum(), 1, atol=1e-10, rtol=0)
        or not np.all(np.isfinite(reference))
    ):
        evidence.update(
            status="invalid_evidence", reason="invalid_probabilities_or_reference"
        )
        return evidence
    counts = np.rint(probabilities * shots)
    if not np.allclose(probabilities * shots, counts, atol=1e-6, rtol=0):
        evidence.update(
            status="invalid_evidence", reason="probabilities_are_not_shot_counts"
        )
        return evidence
    reference_probabilities = np.abs(reference) ** 2
    reference_norm = float(reference_probabilities.sum())
    if not np.isfinite(reference_norm) or reference_norm <= 0:
        evidence.update(
            status="invalid_evidence", reason="zero_or_invalid_reference_norm"
        )
        return evidence
    reference_probabilities /= reference_norm
    logical_counts = counts[: reference.size]
    accepted = int(logical_counts.sum())
    success = accepted / shots
    if execution.logical_success_probability is None or not np.isclose(
        execution.logical_success_probability, success, atol=1e-10, rtol=0
    ):
        evidence.update(
            status="invalid_evidence", reason="inconsistent_postselection_rate"
        )
        return evidence
    alpha = 1 - confidence
    success_radius = float(np.sqrt(np.log(4 / alpha) / (2 * shots)))
    evidence.update(
        accepted_shots=accepted,
        logical_counts=logical_counts.astype(int).tolist(),
        postselection_probability=success,
        postselection_interval=[
            max(0.0, success - success_radius),
            min(1.0, success + success_radius),
        ],
        reference_probabilities=reference_probabilities.tolist(),
    )
    if accepted == 0:
        evidence.update(status="insufficient_shots", reason="no_postselected_samples")
        return evidence
    conditional = logical_counts / accepted
    radius = float(np.sqrt(np.log(4 * reference.size / alpha) / (2 * accepted)))
    lower = np.maximum(0.0, conditional - radius)
    upper = np.minimum(1.0, conditional + radius)
    errors = np.abs(conditional - reference_probabilities)
    error_bound = float(
        np.max(
            np.maximum(
                np.abs(lower - reference_probabilities),
                np.abs(upper - reference_probabilities),
            )
        )
    )
    passed = error_bound <= tolerance and success > success_radius
    # A mismatch is established only when the confidence interval excludes the
    # tolerance band; a large point error alone may still be inconclusive.
    mismatch = bool(
        np.any(
            (lower > reference_probabilities + tolerance)
            | (upper < reference_probabilities - tolerance)
        )
    )
    evidence.update(
        status=(
            "accepted"
            if passed
            else "distribution_mismatch" if mismatch else "insufficient_shots"
        ),
        passed=bool(passed),
        conditional_probabilities=conditional.tolist(),
        probability_intervals=np.column_stack((lower, upper)).tolist(),
        simultaneous_probability_radius=radius,
        maximum_probability_error=float(np.max(errors)),
        maximum_probability_error_bound=error_bound,
        reason=(
            "probability_error_bound_met"
            if passed
            else (
                "reference_outside_tolerance_band"
                if mismatch
                else "uncertainty_exceeds_tolerance_or_postselection_unresolved"
            )
        ),
    )
    return evidence
