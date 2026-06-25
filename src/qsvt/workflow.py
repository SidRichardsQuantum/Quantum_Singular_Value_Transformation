"""
High-level polynomial design workflows.

This module provides a small structured API for callers that want the designed
coefficients, approximation diagnostics, and QSVT compatibility report from a
single operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .compatibility import qsvt_compatibility_report
from .design import (
    design_filter_diagnostics,
    design_filter_polynomial,
    design_interval_projector_diagnostics,
    design_interval_projector_polynomial,
    design_inverse_diagnostics,
    design_inverse_polynomial,
    design_power_diagnostics,
    design_power_polynomial,
    design_projector_diagnostics,
    design_projector_polynomial,
    design_sign_diagnostics,
    design_sign_polynomial,
    design_sqrt_diagnostics,
    design_sqrt_polynomial,
)
from .resources import qsvt_resource_report
from .synthesis import PhaseSynthesisResult, SynthesisRoutine, synthesize_phases

DesignKind = Literal[
    "inverse",
    "sign",
    "projector",
    "sqrt",
    "power",
    "filter",
    "interval_projector",
]


@dataclass(frozen=True)
class DesignWorkflowResult:
    """
    Structured output from a design workflow.
    """

    kind: DesignKind
    builder: str
    coeffs: np.ndarray
    diagnostics: dict[str, object]
    compatibility: dict[str, object]

    def as_report(self) -> dict[str, object]:
        """
        Return a report-style dictionary for JSON conversion or persistence.
        """
        return {
            "mode": "design-workflow",
            "kind": self.kind,
            "builder": self.builder,
            "coeffs": self.coeffs,
            "diagnostics": self.diagnostics,
            "compatibility": self.compatibility,
        }

    def resource_report(
        self,
        *,
        matrix_dimension: int | None = None,
        encoding_qubits: int | None = None,
        block_encoding: str = "dense-block-encoding",
    ) -> dict[str, object]:
        """
        Return a resource proxy report carrying this workflow's diagnostics.
        """
        report = qsvt_resource_report(
            self.coeffs,
            matrix_dimension=matrix_dimension,
            encoding_qubits=encoding_qubits,
            block_encoding=block_encoding,
            attempt_synthesis=False,
            diagnostics=self.diagnostics,
        )
        report.update(
            {
                "kind": self.kind,
                "builder": self.builder,
                "compatibility": self.compatibility,
            }
        )
        return report

    def synthesize(
        self,
        *,
        routine: SynthesisRoutine = "QSVT",
        angle_solver: str = "root-finding",
        reconstruction_num_points: int = 257,
        **solver_kwargs: Any,
    ) -> PhaseSynthesisResult:
        """Synthesize and validate phases for this designed polynomial."""
        return synthesize_phases(
            self.coeffs,
            routine=routine,
            angle_solver=angle_solver,
            reconstruction_num_points=reconstruction_num_points,
            **solver_kwargs,
        )


def design_workflow(
    kind: DesignKind,
    *,
    degree: int,
    gamma: float = 0.25,
    a: float = 0.2,
    alpha: float = 0.5,
    cutoff: float = 0.45,
    lower: float = -0.25,
    upper: float = 0.25,
    sharpness: float = 12.0,
    num_points: int = 2001,
    bounded_num_points: int = 4001,
    attempt_synthesis: bool = True,
) -> DesignWorkflowResult:
    """
    Build a design polynomial with diagnostics and compatibility metadata.
    """
    polynomial_builders = {
        "inverse": lambda: design_inverse_polynomial(
            gamma=gamma,
            degree=degree,
            num_points=num_points,
        ),
        "sign": lambda: design_sign_polynomial(
            gamma=gamma,
            degree=degree,
            num_points=num_points,
        ),
        "projector": lambda: design_projector_polynomial(
            gamma=gamma,
            degree=degree,
            num_points=num_points,
        ),
        "sqrt": lambda: design_sqrt_polynomial(
            a=a,
            degree=degree,
            num_points=num_points,
        ),
        "power": lambda: design_power_polynomial(
            alpha=alpha,
            degree=degree,
            a=a,
            num_points=num_points,
        ),
        "filter": lambda: design_filter_polynomial(
            cutoff=cutoff,
            degree=degree,
            sharpness=sharpness,
            num_points=num_points,
        ),
        "interval_projector": lambda: design_interval_projector_polynomial(
            lower=lower,
            upper=upper,
            degree=degree,
            sharpness=sharpness,
            num_points=num_points,
        ),
    }
    diagnostic_builders = {
        "inverse": lambda: design_inverse_diagnostics(
            gamma=gamma,
            degree=degree,
            num_points=num_points,
            bounded_num_points=bounded_num_points,
        ),
        "sign": lambda: design_sign_diagnostics(
            gamma=gamma,
            degree=degree,
            num_points=num_points,
            bounded_num_points=bounded_num_points,
        ),
        "projector": lambda: design_projector_diagnostics(
            gamma=gamma,
            degree=degree,
            num_points=num_points,
            bounded_num_points=bounded_num_points,
        ),
        "sqrt": lambda: design_sqrt_diagnostics(
            a=a,
            degree=degree,
            num_points=num_points,
            bounded_num_points=bounded_num_points,
        ),
        "power": lambda: design_power_diagnostics(
            alpha=alpha,
            degree=degree,
            a=a,
            num_points=num_points,
            bounded_num_points=bounded_num_points,
        ),
        "filter": lambda: design_filter_diagnostics(
            cutoff=cutoff,
            degree=degree,
            sharpness=sharpness,
            num_points=num_points,
            bounded_num_points=bounded_num_points,
        ),
        "interval_projector": lambda: design_interval_projector_diagnostics(
            lower=lower,
            upper=upper,
            degree=degree,
            sharpness=sharpness,
            num_points=num_points,
            bounded_num_points=bounded_num_points,
        ),
    }

    if kind not in polynomial_builders:
        choices = ", ".join(polynomial_builders)
        raise ValueError(f"kind must be one of: {choices}.")

    coeffs = polynomial_builders[kind]()
    diagnostics = diagnostic_builders[kind]()
    compatibility = qsvt_compatibility_report(
        coeffs,
        bounded_num_points=bounded_num_points,
        attempt_synthesis=attempt_synthesis,
    )

    return DesignWorkflowResult(
        kind=kind,
        builder=f"design_{kind}_polynomial",
        coeffs=coeffs,
        diagnostics=diagnostics,
        compatibility=compatibility,
    )


__all__ = ["DesignKind", "DesignWorkflowResult", "design_workflow"]
