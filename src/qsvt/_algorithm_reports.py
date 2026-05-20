"""
Shared report helpers for algorithm workflow result dataclasses.
"""

from __future__ import annotations

from .rescaling import ScaledOperator


def scaled_operator_report(scaled: ScaledOperator) -> dict[str, object]:
    """
    Return a report dictionary for a scaled dense operator.
    """
    return {
        "matrix": scaled.matrix,
        "offset": scaled.offset,
        "scale": scaled.scale,
        "eigenvalue_bounds": scaled.eigenvalue_bounds,
    }
