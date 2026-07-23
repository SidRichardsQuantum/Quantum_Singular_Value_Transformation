"""Compatibility imports for :mod:`qsvt.comparisons.hhl`.

HHL is retained as a quantum linear-system comparator rather than presented as
a QSVT implementation. New code should use :mod:`qsvt.comparisons`.
"""

from __future__ import annotations

from .comparisons.hhl import (
    HHLCircuitExecutionResult,
    execute_hhl_circuit,
    hhl_circuit_truth_contract,
)

__all__ = [
    "HHLCircuitExecutionResult",
    "execute_hhl_circuit",
    "hhl_circuit_truth_contract",
]
