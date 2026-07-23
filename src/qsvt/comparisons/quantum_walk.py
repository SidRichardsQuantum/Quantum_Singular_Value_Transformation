"""Continuous-time quantum-walk comparison workflow.

The implementation remains shared with the legacy algorithm facade during the
compatibility window. This module is the canonical namespace for new code.
"""

from __future__ import annotations

from .._algorithm_dynamics import (
    QuantumWalkSearchWorkflowResult,
    quantum_walk_search_resource_proxy,
    quantum_walk_search_workflow,
)

__all__ = [
    "QuantumWalkSearchWorkflowResult",
    "quantum_walk_search_resource_proxy",
    "quantum_walk_search_workflow",
]
