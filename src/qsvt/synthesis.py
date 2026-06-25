"""
Polynomial realizability and phase-synthesis workflows.

The helpers in this module distinguish ordinary polynomial functional calculus
from polynomials that one standard QSP/QSVT phase sequence can realize.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np
import pennylane as qml

from .polynomials import eval_polynomial, polynomial_degree, polynomial_parity

RealizabilityKind = Literal[
    "invalid-polynomial",
    "classical-polynomial-only",
    "single-sequence-qsp-qsvt",
    "multiple-parity-sequences-or-lcu",
]
SynthesisRoutine = Literal["QSP", "QSVT"]
_SUPPORTED_ANGLE_SOLVERS = ("root-finding", "iterative", "iterative-optax")


@dataclass(frozen=True)
class BoundednessCertificate:
    """Numerical extrema certificate for a real polynomial on an interval."""

    coeffs: np.ndarray
    domain: tuple[float, float]
    bound: float
    tolerance: float
    critical_points: np.ndarray
    evaluated_points: np.ndarray
    values: np.ndarray
    max_abs_value: float
    maximizing_point: float
    margin: float
    is_bounded: bool
    derivative_root_residual: float
    certification_kind: str = "numerical-polynomial-extrema"

    def as_report(self) -> dict[str, object]:
        """Return the extrema calculation and its numerical tolerance."""
        return {
            "mode": "polynomial-boundedness-certificate",
            "certification_kind": self.certification_kind,
            "coeffs": self.coeffs,
            "domain": self.domain,
            "bound": self.bound,
            "tolerance": self.tolerance,
            "critical_points": self.critical_points,
            "evaluated_points": self.evaluated_points,
            "values": self.values,
            "max_abs_value": self.max_abs_value,
            "maximizing_point": self.maximizing_point,
            "margin": self.margin,
            "is_bounded": self.is_bounded,
            "derivative_root_residual": self.derivative_root_residual,
            "interpretation": (
                "All interval endpoints and numerically real derivative roots "
                "were evaluated. This is a floating-point extrema certificate, "
                "not an interval-arithmetic proof."
            ),
        }


@dataclass(frozen=True)
class PolynomialRealizability:
    """Classification of a polynomial's QSP/QSVT realizability."""

    coeffs: np.ndarray
    degree: int
    parity: str
    bounded: bool
    max_abs_value: float | None
    bounded_domain: tuple[float, float]
    bounded_num_points: int
    kind: RealizabilityKind
    single_sequence_realizable: bool
    requires_parity_decomposition: bool
    even_coeffs: np.ndarray
    odd_coeffs: np.ndarray
    reasons: tuple[str, ...]
    boundedness_certificate: BoundednessCertificate | None = None

    def as_report(self) -> dict[str, object]:
        """Return a machine-readable realizability report."""
        return {
            "mode": "polynomial-realizability-report",
            "coeffs": self.coeffs,
            "degree": self.degree,
            "parity": self.parity,
            "bounded": self.bounded,
            "max_abs_value": self.max_abs_value,
            "bounded_domain": self.bounded_domain,
            "bounded_num_points": self.bounded_num_points,
            "realizability_kind": self.kind,
            "single_sequence_realizable": self.single_sequence_realizable,
            "requires_parity_decomposition": self.requires_parity_decomposition,
            "even_coeffs": self.even_coeffs,
            "odd_coeffs": self.odd_coeffs,
            "reasons": list(self.reasons),
            "boundedness_certificate": (
                None
                if self.boundedness_certificate is None
                else self.boundedness_certificate.as_report()
            ),
            "interpretation": _realizability_interpretation(self.kind),
        }


@dataclass(frozen=True)
class PhaseSynthesisResult:
    """Angles and numerical validation from a QSP/QSVT synthesis attempt."""

    coeffs: np.ndarray
    routine: SynthesisRoutine
    angle_solver: str
    solver_kwargs: dict[str, object]
    realizability: PolynomialRealizability
    succeeded: bool
    angles: np.ndarray | None
    synthesis_time_seconds: float
    reconstruction_max_error: float | None
    reconstruction_rms_error: float | None
    reconstruction_num_points: int
    convention: str
    error_type: str | None = None
    error: str | None = None

    def as_report(self) -> dict[str, object]:
        """Return a machine-readable phase-synthesis report."""
        return {
            "mode": "phase-synthesis-report",
            "implementation_kind": "pennylane-poly-to-angles",
            "coeffs": self.coeffs,
            "routine": self.routine,
            "angle_solver": self.angle_solver,
            "solver_kwargs": self.solver_kwargs,
            "convention": self.convention,
            "succeeded": self.succeeded,
            "angles": self.angles,
            "phase_count": None if self.angles is None else int(self.angles.size),
            "synthesis_time_seconds": self.synthesis_time_seconds,
            "reconstruction_max_error": self.reconstruction_max_error,
            "reconstruction_rms_error": self.reconstruction_rms_error,
            "reconstruction_num_points": self.reconstruction_num_points,
            "realizability": self.realizability.as_report(),
            "error_type": self.error_type,
            "error": self.error,
            "truth_contract": {
                "is_end_to_end_quantum_algorithm": False,
                "implemented_components": [
                    "polynomial_realizability_classification",
                    "pennylane_phase_synthesis",
                    (
                        "scalar_phase_sequence_reconstruction"
                        if self.routine == "QSVT"
                        else "phase_synthesis_without_package_reconstruction"
                    ),
                ],
                "omitted_components": [
                    "problem_specific_block_encoding",
                    "state_preparation",
                    "readout",
                    "hardware_compilation",
                ],
            },
        }


@dataclass(frozen=True)
class PhaseSolverBenchmarkResult:
    """Repeated phase-solver comparison for one polynomial."""

    coeffs: np.ndarray
    routine: SynthesisRoutine
    repeats: int
    rows: tuple[dict[str, object], ...]
    realizability: PolynomialRealizability

    def as_report(self) -> dict[str, object]:
        """Return compact solver timing, convergence, and error rows."""
        return {
            "mode": "phase-solver-benchmark",
            "implementation_kind": "pennylane-phase-solver-microbenchmark",
            "coeffs": self.coeffs,
            "routine": self.routine,
            "repeats": self.repeats,
            "rows": list(self.rows),
            "realizability": self.realizability.as_report(),
            "conditioning_proxies": {
                "degree": self.realizability.degree,
                "coefficient_dynamic_range": _coefficient_dynamic_range(self.coeffs),
                "boundedness_margin": (
                    None
                    if self.realizability.boundedness_certificate is None
                    else self.realizability.boundedness_certificate.margin
                ),
            },
            "truth_contract": {
                "timing_kind": "python_wall_clock_microbenchmark",
                "is_hardware_runtime": False,
                "measured_component": "classical_phase_synthesis",
            },
        }


@dataclass(frozen=True)
class MixedParitySynthesisResult:
    """Separate even/odd synthesis plus an LCU-style combination model."""

    coeffs: np.ndarray
    even_coeffs: np.ndarray
    odd_coeffs: np.ndarray
    even_weight: float
    odd_weight: float
    lcu_normalization: float
    postselection_probability_proxy: float
    even_synthesis: PhaseSynthesisResult | None
    odd_synthesis: PhaseSynthesisResult | None
    succeeded: bool
    reconstruction_max_error: float | None
    assumptions: tuple[str, ...]

    def as_report(self) -> dict[str, object]:
        """Return component phases and conditional LCU cost metadata."""
        even_phase_count = _phase_count(self.even_synthesis)
        odd_phase_count = _phase_count(self.odd_synthesis)
        even_signal_calls = (
            0 if self.even_synthesis is None else polynomial_degree(self.even_coeffs)
        )
        odd_signal_calls = (
            0 if self.odd_synthesis is None else polynomial_degree(self.odd_coeffs)
        )
        return {
            "mode": "mixed-parity-synthesis-report",
            "implementation_kind": "separate-parity-sequences-with-lcu-model",
            "coeffs": self.coeffs,
            "even_coeffs": self.even_coeffs,
            "odd_coeffs": self.odd_coeffs,
            "even_weight": self.even_weight,
            "odd_weight": self.odd_weight,
            "lcu_normalization": self.lcu_normalization,
            "postselection_probability_proxy": self.postselection_probability_proxy,
            "component_resource_proxy": {
                "sequence_count": int(self.even_synthesis is not None)
                + int(self.odd_synthesis is not None),
                "even_phase_count": even_phase_count,
                "odd_phase_count": odd_phase_count,
                "total_phase_count": even_phase_count + odd_phase_count,
                "even_signal_operator_calls": even_signal_calls,
                "odd_signal_operator_calls": odd_signal_calls,
                "total_signal_operator_calls": even_signal_calls + odd_signal_calls,
            },
            "even_synthesis": (
                None if self.even_synthesis is None else self.even_synthesis.as_report()
            ),
            "odd_synthesis": (
                None if self.odd_synthesis is None else self.odd_synthesis.as_report()
            ),
            "succeeded": self.succeeded,
            "reconstruction_max_error": self.reconstruction_max_error,
            "assumptions": list(self.assumptions),
            "truth_contract": {
                "component_phase_sequences_synthesized": self.succeeded,
                "lcu_circuit_implemented": False,
                "postselection_probability_is_proxy": True,
                "omitted_components": [
                    "ancilla_state_preparation_for_lcu_weights",
                    "controlled_selection_between_phase_sequences",
                    "uncomputation",
                    "amplitude_amplification",
                    "application_specific_readout",
                ],
            },
        }


def certify_polynomial_boundedness(
    poly: Any,
    *,
    domain: tuple[float, float] = (-1.0, 1.0),
    bound: float = 1.0,
    tolerance: float = 1e-10,
    root_imag_tolerance: float = 1e-9,
) -> BoundednessCertificate:
    """
    Numerically certify boundedness by evaluating every polynomial extremum.

    The derivative is solved in the monomial basis. All numerically real roots
    in the interval and both endpoints are evaluated, avoiding the missed-peak
    risk of grid-only boundedness checks.
    """
    coeffs = np.asarray(list(poly), dtype=float)
    if coeffs.ndim != 1 or coeffs.size == 0:
        raise ValueError("poly must contain at least one coefficient.")
    if not np.all(np.isfinite(coeffs)):
        raise ValueError("polynomial coefficients must be finite.")
    lower, upper = map(float, domain)
    if upper <= lower:
        raise ValueError("domain must satisfy lower < upper.")
    if not np.isfinite(bound) or bound <= 0.0:
        raise ValueError("bound must be positive and finite.")
    if tolerance < 0.0 or root_imag_tolerance < 0.0:
        raise ValueError("tolerances must be non-negative.")

    derivative = np.polynomial.polynomial.polyder(coeffs)
    critical: list[float] = []
    residual = 0.0
    if derivative.size > 1 or np.any(np.abs(derivative) > 0.0):
        roots = np.polynomial.polynomial.polyroots(derivative)
        for root in roots:
            if abs(float(np.imag(root))) <= root_imag_tolerance:
                point = float(np.real(root))
                if lower - tolerance <= point <= upper + tolerance:
                    point = min(max(point, lower), upper)
                    critical.append(point)
                    residual = max(
                        residual,
                        abs(float(np.polynomial.polynomial.polyval(point, derivative))),
                    )

    critical_points = np.asarray(sorted(set(critical)), dtype=float)
    evaluated_points = np.concatenate(
        ([lower], critical_points, [upper]),
    )
    values = np.asarray(eval_polynomial(coeffs, evaluated_points), dtype=float)
    maximizing_index = int(np.argmax(np.abs(values)))
    max_abs = float(abs(values[maximizing_index]))
    margin = float(bound - max_abs)
    return BoundednessCertificate(
        coeffs=coeffs,
        domain=(lower, upper),
        bound=float(bound),
        tolerance=float(tolerance),
        critical_points=critical_points,
        evaluated_points=evaluated_points,
        values=values,
        max_abs_value=max_abs,
        maximizing_point=float(evaluated_points[maximizing_index]),
        margin=margin,
        is_bounded=bool(max_abs <= float(bound) + tolerance),
        derivative_root_residual=float(residual),
    )


def classify_polynomial_realizability(
    poly: Any,
    *,
    bounded_domain: tuple[float, float] = (-1.0, 1.0),
    bounded_num_points: int = 4001,
    bound: float = 1.0,
    parity_tol: float = 1e-10,
) -> PolynomialRealizability:
    """
    Classify a polynomial for classical use and standard QSP/QSVT synthesis.

    A finite, extrema-bounded polynomial with definite parity is classified as
    realizable by one standard QSP/QSVT sequence. An extrema-bounded mixed-parity
    polynomial is still valid for classical polynomial functional calculus but
    requires separate even/odd sequences and a combination mechanism such as
    linear combination of unitaries.
    """
    coeffs = np.asarray(list(poly), dtype=float)
    if coeffs.ndim != 1 or coeffs.size == 0:
        raise ValueError("poly must contain at least one coefficient.")
    if bounded_num_points < 2:
        raise ValueError("bounded_num_points must be at least 2.")
    lower, upper = map(float, bounded_domain)
    if upper <= lower:
        raise ValueError("bounded_domain must satisfy lower < upper.")
    if not np.isfinite(bound) or bound <= 0.0:
        raise ValueError("bound must be positive and finite.")

    finite = bool(np.all(np.isfinite(coeffs)))
    parity = polynomial_parity(coeffs, tol=parity_tol)
    even_coeffs, odd_coeffs = parity_components(coeffs)
    reasons: list[str] = []

    max_abs_value: float | None
    certificate: BoundednessCertificate | None = None
    bounded = False
    if finite:
        certificate = certify_polynomial_boundedness(
            coeffs,
            domain=(lower, upper),
            bound=bound,
            tolerance=1e-10,
        )
        max_abs_value = certificate.max_abs_value
        bounded = certificate.is_bounded
    else:
        max_abs_value = None
        reasons.append("non_finite_coefficients")

    if not bounded:
        reasons.append("out_of_bounds")

    definite_parity = parity in {"even", "odd", "zero"}
    if finite and bounded and definite_parity:
        kind: RealizabilityKind = "single-sequence-qsp-qsvt"
        single_sequence = True
        requires_decomposition = False
    elif finite and bounded:
        kind = "multiple-parity-sequences-or-lcu"
        single_sequence = False
        requires_decomposition = True
        reasons.append("mixed_parity")
    elif finite:
        kind = "classical-polynomial-only"
        single_sequence = False
        requires_decomposition = parity == "mixed"
    else:
        kind = "invalid-polynomial"
        single_sequence = False
        requires_decomposition = False

    return PolynomialRealizability(
        coeffs=coeffs,
        degree=int(coeffs.size - 1),
        parity=parity,
        bounded=bounded,
        max_abs_value=max_abs_value,
        bounded_domain=(lower, upper),
        bounded_num_points=int(bounded_num_points),
        kind=kind,
        single_sequence_realizable=single_sequence,
        requires_parity_decomposition=requires_decomposition,
        even_coeffs=even_coeffs,
        odd_coeffs=odd_coeffs,
        reasons=tuple(reasons),
        boundedness_certificate=certificate,
    )


def parity_components(poly: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the even and odd coefficient components of a polynomial."""
    coeffs = np.asarray(list(poly), dtype=float)
    if coeffs.ndim != 1 or coeffs.size == 0:
        raise ValueError("poly must contain at least one coefficient.")
    even = coeffs.copy()
    odd = coeffs.copy()
    even[1::2] = 0.0
    odd[0::2] = 0.0
    return even, odd


def synthesize_phases(
    poly: Any,
    *,
    routine: SynthesisRoutine = "QSVT",
    angle_solver: str = "root-finding",
    bounded_num_points: int = 4001,
    reconstruction_num_points: int = 257,
    raise_on_failure: bool = False,
    **solver_kwargs: Any,
) -> PhaseSynthesisResult:
    """
    Synthesize QSP/QSVT phases and retain diagnostics and failure metadata.

    QSVT results are reconstructed on scalar signal values using the explicit
    phase sequence. QSP synthesis is supported, but package-level scalar
    reconstruction is currently reported as unavailable because its phase
    convention differs from the QSVT projector-phase convention.
    """
    normalized_routine = str(routine).upper()
    if normalized_routine not in {"QSP", "QSVT"}:
        raise ValueError("routine must be 'QSP' or 'QSVT'.")
    routine = cast(SynthesisRoutine, normalized_routine)
    if reconstruction_num_points < 2:
        raise ValueError("reconstruction_num_points must be at least 2.")

    realizability = classify_polynomial_realizability(
        poly,
        bounded_num_points=bounded_num_points,
    )
    start = perf_counter()
    angles: np.ndarray | None = None
    error_type: str | None = None
    error: str | None = None

    if not realizability.single_sequence_realizable:
        error_type = "PolynomialRealizabilityError"
        error = _realizability_interpretation(realizability.kind)
    else:
        try:
            if angle_solver not in _SUPPORTED_ANGLE_SOLVERS:
                raise ValueError(
                    f"Invalid angle solver method: {angle_solver!r}. "
                    f"Supported solvers: {list(_SUPPORTED_ANGLE_SOLVERS)}"
                )
            angles = np.asarray(
                qml.poly_to_angles(
                    realizability.coeffs,
                    routine,
                    angle_solver=angle_solver,
                    **solver_kwargs,
                ),
                dtype=float,
            )
        except Exception as exc:  # PennyLane exposes solver-specific failures.
            error_type = type(exc).__name__
            error = str(exc)

    elapsed = perf_counter() - start
    max_error: float | None = None
    rms_error: float | None = None
    if angles is not None and routine == "QSVT":
        xs = np.linspace(-1.0, 1.0, int(reconstruction_num_points))
        reconstructed = np.asarray(
            [_evaluate_qsvt_phase_sequence(float(x), angles) for x in xs],
            dtype=float,
        )
        target = np.asarray(eval_polynomial(realizability.coeffs, xs), dtype=float)
        errors = reconstructed - target
        max_error = float(np.max(np.abs(errors)))
        rms_error = float(np.sqrt(np.mean(np.abs(errors) ** 2)))

    result = PhaseSynthesisResult(
        coeffs=realizability.coeffs,
        routine=routine,
        angle_solver=str(angle_solver),
        solver_kwargs=dict(solver_kwargs),
        realizability=realizability,
        succeeded=angles is not None,
        angles=angles,
        synthesis_time_seconds=float(elapsed),
        reconstruction_max_error=max_error,
        reconstruction_rms_error=rms_error,
        reconstruction_num_points=int(reconstruction_num_points),
        convention=(
            "PennyLane QSVT projector-phase convention; polynomial coefficients "
            "are in ascending monomial order."
            if routine == "QSVT"
            else "PennyLane QSP angle convention; coefficients are ascending."
        ),
        error_type=error_type,
        error=error,
    )
    if raise_on_failure and not result.succeeded:
        raise ValueError(error or "phase synthesis failed")
    return result


def synthesis_workflow(poly: Any, **kwargs: Any) -> PhaseSynthesisResult:
    """Alias for :func:`synthesize_phases` as a workflow-level entry point."""
    return synthesize_phases(poly, **kwargs)


def synthesize(poly: Any, **kwargs: Any) -> PhaseSynthesisResult:
    """Concise alias for :func:`synthesize_phases`."""
    return synthesize_phases(poly, **kwargs)


def benchmark_phase_solvers(
    poly: Any,
    *,
    solvers: tuple[str, ...] | list[str] = ("root-finding", "iterative"),
    routine: SynthesisRoutine = "QSVT",
    repeats: int = 3,
    reconstruction_num_points: int = 65,
    solver_kwargs: dict[str, dict[str, Any]] | None = None,
) -> PhaseSolverBenchmarkResult:
    """Compare phase solvers by convergence, timing, and reconstruction error."""
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    if not solvers:
        raise ValueError("solvers must contain at least one solver name.")
    realizability = classify_polynomial_realizability(poly)
    rows: list[dict[str, object]] = []
    kwargs_by_solver = solver_kwargs or {}
    for solver in solvers:
        attempts = [
            synthesize_phases(
                realizability.coeffs,
                routine=routine,
                angle_solver=solver,
                reconstruction_num_points=reconstruction_num_points,
                **kwargs_by_solver.get(solver, {}),
            )
            for _ in range(int(repeats))
        ]
        succeeded = [attempt for attempt in attempts if attempt.succeeded]
        times = [attempt.synthesis_time_seconds for attempt in attempts]
        errors = [
            attempt.reconstruction_max_error
            for attempt in succeeded
            if attempt.reconstruction_max_error is not None
        ]
        rows.append(
            {
                "angle_solver": solver,
                "attempts": int(repeats),
                "successes": len(succeeded),
                "converged": len(succeeded) == int(repeats),
                "best_time_seconds": min(times),
                "mean_time_seconds": float(np.mean(times)),
                "max_reconstruction_error": max(errors) if errors else None,
                "mean_reconstruction_error": (
                    float(np.mean(errors)) if errors else None
                ),
                "phase_count": (
                    None
                    if not succeeded or succeeded[0].angles is None
                    else int(succeeded[0].angles.size)
                ),
                "error_types": sorted(
                    {
                        attempt.error_type
                        for attempt in attempts
                        if attempt.error_type is not None
                    }
                ),
            }
        )
    return PhaseSolverBenchmarkResult(
        coeffs=realizability.coeffs,
        routine=routine,
        repeats=int(repeats),
        rows=tuple(rows),
        realizability=realizability,
    )


def synthesize_mixed_parity(
    poly: Any,
    *,
    angle_solver: str = "root-finding",
    reconstruction_num_points: int = 257,
    **solver_kwargs: Any,
) -> MixedParitySynthesisResult:
    """
    Synthesize even and odd components and report an LCU combination model.

    Each nonzero component is normalized by its extrema norm before synthesis.
    The component norms become LCU weights. The returned postselection value is
    the idealized ``1 / lambda**2`` proxy for ``lambda = weight_even + weight_odd``.
    """
    coeffs = np.asarray(list(poly), dtype=float)
    classification = classify_polynomial_realizability(coeffs)
    even, odd = classification.even_coeffs, classification.odd_coeffs
    even_weight = certify_polynomial_boundedness(even).max_abs_value
    odd_weight = certify_polynomial_boundedness(odd).max_abs_value

    even_result = _synthesize_normalized_component(
        even,
        even_weight,
        angle_solver=angle_solver,
        reconstruction_num_points=reconstruction_num_points,
        **solver_kwargs,
    )
    odd_result = _synthesize_normalized_component(
        odd,
        odd_weight,
        angle_solver=angle_solver,
        reconstruction_num_points=reconstruction_num_points,
        **solver_kwargs,
    )
    component_results = [
        result for result in (even_result, odd_result) if result is not None
    ]
    succeeded = bool(component_results) and all(
        result.succeeded for result in component_results
    )
    lcu_normalization = float(even_weight + odd_weight)
    postselection = (
        1.0 if lcu_normalization == 0.0 else min(1.0, 1.0 / lcu_normalization**2)
    )
    max_error: float | None = None
    if succeeded:
        errors = [
            weight * (result.reconstruction_max_error or 0.0)
            for weight, result in (
                (even_weight, even_result),
                (odd_weight, odd_result),
            )
            if result is not None
        ]
        max_error = float(sum(errors))
    return MixedParitySynthesisResult(
        coeffs=coeffs,
        even_coeffs=even,
        odd_coeffs=odd,
        even_weight=float(even_weight),
        odd_weight=float(odd_weight),
        lcu_normalization=lcu_normalization,
        postselection_probability_proxy=float(postselection),
        even_synthesis=even_result,
        odd_synthesis=odd_result,
        succeeded=succeeded,
        reconstruction_max_error=max_error,
        assumptions=(
            "The even and odd QSVT sequences can be coherently selected.",
            "LCU ancilla preparation amplitudes encode the component weights.",
            "The postselection proxy omits amplitude-amplification overhead.",
        ),
    )


def _evaluate_qsvt_phase_sequence(x: float, angles: np.ndarray) -> float:
    block_encoding = qml.RX(2.0 * np.arccos(np.clip(x, -1.0, 1.0)), wires=0)
    projectors = [qml.PCPhase(float(phi), dim=1, wires=0) for phi in angles]
    unitary = qml.matrix(
        qml.QSVT(block_encoding, projectors),
        wire_order=[0],
    )
    return float(np.real(np.asarray(unitary)[0, 0]))


def _realizability_interpretation(kind: RealizabilityKind) -> str:
    messages = {
        "invalid-polynomial": "The coefficients are not finite.",
        "classical-polynomial-only": (
            "The polynomial can be evaluated classically but is not bounded by "
            "one on the sampled QSP/QSVT signal domain."
        ),
        "single-sequence-qsp-qsvt": (
            "The polynomial is extrema-bounded and has definite parity, so it is "
            "eligible for one standard QSP/QSVT phase sequence."
        ),
        "multiple-parity-sequences-or-lcu": (
            "The polynomial is extrema-bounded but has mixed parity. Realize its "
            "even and odd components separately and combine them, for example "
            "with an LCU-style construction."
        ),
    }
    return messages[kind]


def _coefficient_dynamic_range(coeffs: np.ndarray) -> float:
    nonzero = np.abs(coeffs[np.abs(coeffs) > 0.0])
    if nonzero.size < 2:
        return 1.0
    return float(np.max(nonzero) / np.min(nonzero))


def _synthesize_normalized_component(
    coeffs: np.ndarray,
    weight: float,
    *,
    angle_solver: str,
    reconstruction_num_points: int,
    **solver_kwargs: Any,
) -> PhaseSynthesisResult | None:
    if weight <= 1e-15:
        return None
    normalized = coeffs / weight
    if np.count_nonzero(np.abs(normalized[1:]) > 1e-14) == 0:
        realizability = classify_polynomial_realizability(normalized)
        return PhaseSynthesisResult(
            coeffs=normalized,
            routine="QSVT",
            angle_solver="analytic-constant",
            solver_kwargs={},
            realizability=realizability,
            succeeded=True,
            angles=np.asarray([], dtype=float),
            synthesis_time_seconds=0.0,
            reconstruction_max_error=0.0,
            reconstruction_rms_error=0.0,
            reconstruction_num_points=reconstruction_num_points,
            convention="Analytic constant branch; no signal queries are required.",
        )
    return synthesize_phases(
        normalized,
        angle_solver=angle_solver,
        reconstruction_num_points=reconstruction_num_points,
        **solver_kwargs,
    )


def _phase_count(result: PhaseSynthesisResult | None) -> int:
    if result is None or result.angles is None:
        return 0
    return int(result.angles.size)


__all__ = [
    "BoundednessCertificate",
    "MixedParitySynthesisResult",
    "PhaseSynthesisResult",
    "PhaseSolverBenchmarkResult",
    "PolynomialRealizability",
    "RealizabilityKind",
    "SynthesisRoutine",
    "benchmark_phase_solvers",
    "certify_polynomial_boundedness",
    "classify_polynomial_realizability",
    "parity_components",
    "synthesize",
    "synthesize_mixed_parity",
    "synthesize_phases",
    "synthesis_workflow",
]
