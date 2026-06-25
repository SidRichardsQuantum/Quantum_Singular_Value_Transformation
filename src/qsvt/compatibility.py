"""
qsvt.compatibility
------------------

QSVT polynomial compatibility checks.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pennylane as qml

from .polynomials import polynomial_parity
from .synthesis import (
    certify_polynomial_boundedness,
    classify_polynomial_realizability,
)


def qsvt_compatibility_report(
    poly: Iterable[float],
    *,
    bounded_domain: tuple[float, float] = (-1.0, 1.0),
    bounded_num_points: int = 4001,
    bound: float = 1.0,
    parity_tol: float = 1e-10,
    attempt_synthesis: bool = True,
    block_encoding: str = "embedding",
    angle_solver: str = "root-finding",
) -> dict[str, object]:
    """
    Check whether polynomial coefficients are suitable for PennyLane QSVT.

    This report separates lightweight structural checks from PennyLane's phase
    synthesis attempt. A bounded polynomial can still fail synthesis if it does
    not meet parity or solver-specific requirements.

    Parameters
    ----------
    poly
        Polynomial coefficients in ascending degree order.
    bounded_domain
        Domain used for extrema-based boundedness checks.
    bounded_num_points
        Retained for report-schema compatibility; extrema checks do not use a grid.
    bound
        Absolute-value bound expected by QSVT.
    parity_tol
        Tolerance used for parity classification.
    attempt_synthesis
        If True, attempt PennyLane QSVT synthesis on a scalar input.
    block_encoding
        PennyLane block-encoding mode used for the synthesis attempt.

    Returns
    -------
    dict[str, object]
        Compatibility report with reasons for any failed checks.
    """
    coeffs = np.asarray(list(poly), dtype=float)

    if coeffs.ndim != 1 or coeffs.size == 0:
        raise ValueError("poly must contain at least one coefficient.")
    if bounded_num_points < 2:
        raise ValueError("bounded_num_points must be at least 2.")
    lower, upper = float(bounded_domain[0]), float(bounded_domain[1])
    if upper <= lower:
        raise ValueError("bounded_domain must satisfy lower < upper.")

    coeffs_finite = bool(np.all(np.isfinite(coeffs)))
    reasons: list[str] = []
    if not coeffs_finite:
        reasons.append("non_finite_coefficients")

    parity = polynomial_parity(coeffs, tol=parity_tol)
    has_definite_parity = parity in {"even", "odd", "zero"}
    if not has_definite_parity:
        reasons.append("mixed_parity")

    if coeffs_finite:
        certificate = certify_polynomial_boundedness(
            coeffs,
            domain=(lower, upper),
            bound=bound,
        )
        max_abs_value = certificate.max_abs_value
        bounded_margin = float(bound - max_abs_value)
        is_bounded = certificate.is_bounded
    else:
        certificate = None
        max_abs_value = None
        bounded_margin = None
        is_bounded = False

    if not is_bounded:
        reasons.append("out_of_bounds")

    report: dict[str, object] = {
        "mode": "qsvt-compatibility-report",
        "poly": coeffs,
        "polynomial_degree": int(coeffs.size - 1),
        "coefficients_finite": coeffs_finite,
        "parity": parity,
        "has_definite_parity": has_definite_parity,
        "bounded_domain": (lower, upper),
        "bounded_num_points": int(bounded_num_points),
        "bound": float(bound),
        "max_abs_value": max_abs_value,
        "bounded_margin": bounded_margin,
        "is_bounded": is_bounded,
        "boundedness_check": (None if certificate is None else certificate.as_report()),
        "attempted_pennylane_synthesis": bool(attempt_synthesis),
        "angle_solver": angle_solver,
    }
    realizability = classify_polynomial_realizability(
        coeffs,
        bounded_domain=(lower, upper),
        bounded_num_points=bounded_num_points,
        bound=bound,
        parity_tol=parity_tol,
    )
    report["realizability"] = realizability.as_report()
    report["realizability_kind"] = realizability.kind
    report["single_sequence_realizable"] = realizability.single_sequence_realizable
    report["requires_parity_decomposition"] = (
        realizability.requires_parity_decomposition
    )

    synthesis_succeeded: bool | None = None
    if attempt_synthesis and coeffs_finite:
        try:
            qml.qsvt(
                0.5,
                coeffs,
                encoding_wires=[0],
                block_encoding=block_encoding,
                angle_solver=angle_solver,
            )
        except Exception as exc:
            synthesis_succeeded = False
            reasons.append("synthesis_failed")
            report.update(
                {
                    "pennylane_synthesis_succeeded": False,
                    "pennylane_error_type": type(exc).__name__,
                    "pennylane_error": str(exc),
                }
            )
        else:
            synthesis_succeeded = True
            report["pennylane_synthesis_succeeded"] = True
    else:
        report["pennylane_synthesis_succeeded"] = synthesis_succeeded

    report["reasons"] = reasons
    report["compatible"] = bool(
        coeffs_finite
        and has_definite_parity
        and is_bounded
        and (synthesis_succeeded is not False)
    )
    return report


__all__ = ["qsvt_compatibility_report"]
