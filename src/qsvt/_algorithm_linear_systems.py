"""Linear-system algorithm workflows and reports."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._algorithm_reports import (
    algorithm_truth_contract,
    algorithm_workflow_schema_fields,
    scaled_operator_report,
)
from ._algorithm_shared import _relative_error
from .compatibility import qsvt_compatibility_report
from .design import (
    design_positive_inverse_diagnostics,
    design_positive_inverse_polynomial,
)
from .matrix import qsvt_matrix_transform
from .rescaling import ScaledOperator, rescale_positive_semidefinite
from .spectral import apply_polynomial_to_hermitian, eigh_hermitian


@dataclass(frozen=True)
class LinearSystemWorkflowResult:
    """
    Structured output from a positive-definite linear-system workflow.
    """

    coeffs: np.ndarray
    scaled_operator: ScaledOperator
    rhs: np.ndarray
    classical_solution: np.ndarray
    polynomial_solution: np.ndarray
    polynomial_residual_norm: float
    polynomial_relative_error: float
    gamma: float
    scaled_min_eigenvalue: float
    scaled_max_eigenvalue: float
    condition_number_2: float
    gamma_condition_proxy: float
    degree: int
    diagnostics: dict[str, object]
    compatibility: dict[str, object]
    qsvt_solution: np.ndarray | None = None
    qsvt_residual_norm: float | None = None
    qsvt_relative_error: float | None = None
    qsvt_error: str | None = None

    def as_report(self) -> dict[str, Any]:
        """
        Return a report-style dictionary for JSON conversion or persistence.
        """
        if self.qsvt_error is not None:
            qsvt_check = "failed"
        elif self.qsvt_solution is not None:
            qsvt_check = "succeeded"
        else:
            qsvt_check = "not_attempted"
        return {
            **algorithm_workflow_schema_fields(),
            "mode": "linear-system-workflow",
            "implementation_kind": "dense-spectral-polynomial-workflow",
            "truth_contract": algorithm_truth_contract(
                "linear-system-workflow",
                target="positive-definite linear-system inverse action",
                qsvt_check=qsvt_check,
            ),
            "gamma": self.gamma,
            "scaled_min_eigenvalue": self.scaled_min_eigenvalue,
            "scaled_max_eigenvalue": self.scaled_max_eigenvalue,
            "condition_number_2": self.condition_number_2,
            "gamma_condition_proxy": self.gamma_condition_proxy,
            "degree": self.degree,
            "coeffs": self.coeffs,
            "scaled_operator": scaled_operator_report(self.scaled_operator),
            "rhs": self.rhs,
            "classical_solution": self.classical_solution,
            "polynomial_solution": self.polynomial_solution,
            "polynomial_residual_norm": self.polynomial_residual_norm,
            "polynomial_relative_error": self.polynomial_relative_error,
            "qsvt_solution": self.qsvt_solution,
            "qsvt_residual_norm": self.qsvt_residual_norm,
            "qsvt_relative_error": self.qsvt_relative_error,
            "qsvt_error": self.qsvt_error,
            "diagnostics": self.diagnostics,
            "compatibility": self.compatibility,
            "resource_proxy": linear_system_resource_proxy(
                degree=self.degree,
                gamma=self.gamma,
                scaled_min_eigenvalue=self.scaled_min_eigenvalue,
                condition_number_2=self.condition_number_2,
                polynomial_residual_norm=self.polynomial_residual_norm,
                polynomial_relative_error=self.polynomial_relative_error,
                qsvt_check=qsvt_check,
                attempted_pennylane_synthesis=bool(
                    self.compatibility.get("attempted_pennylane_synthesis", False)
                ),
            ),
        }


@dataclass(frozen=True)
class LinearSystemComparisonResult:
    """
    Structured comparison for small positive-definite linear-system solvers.
    """

    workflow: LinearSystemWorkflowResult
    rows: tuple[dict[str, Any], ...]
    dense_solution: np.ndarray
    cg_solution: np.ndarray | None
    reference_solution: np.ndarray
    notes: tuple[str, ...] = ()

    def as_report(self) -> dict[str, Any]:
        """
        Return a report-style dictionary for JSON conversion or persistence.
        """
        workflow_report = self.workflow.as_report()
        return {
            **algorithm_workflow_schema_fields(),
            "mode": "linear-system-comparison-workflow",
            "implementation_kind": "linear-system-solver-comparison",
            "truth_contract": algorithm_truth_contract(
                "linear-system-comparison-workflow",
                target="positive-definite linear-system solver comparison",
                qsvt_check=_qsvt_check_from_result(self.workflow),
            ),
            "rows": list(self.rows),
            "reference_solution": self.reference_solution,
            "dense_solution": self.dense_solution,
            "cg_solution": self.cg_solution,
            "linear_system_workflow": workflow_report,
            "resource_proxy": workflow_report["resource_proxy"],
            "notes": list(self.notes),
        }


def _validate_linear_system_inputs(
    matrix: np.ndarray,
    rhs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.asarray(matrix)
    b = np.asarray(rhs, dtype=complex if np.iscomplexobj(rhs) else float)

    evals, _ = eigh_hermitian(A)
    if evals[0] <= 0.0:
        raise ValueError("matrix must be positive definite.")
    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError("rhs must be a vector whose length matches matrix dimension.")

    dtype = complex if np.iscomplexobj(A) or np.iscomplexobj(b) else float
    return A.astype(dtype, copy=False), b.astype(dtype, copy=False), evals


def _qsvt_check_from_result(result: LinearSystemWorkflowResult) -> str:
    if result.qsvt_error is not None:
        return "failed"
    if result.qsvt_solution is not None:
        return "succeeded"
    return "not_attempted"


def linear_system_workflow(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    degree: int,
    gamma: float | None = None,
    num_points: int = 2001,
    bounded_num_points: int = 4001,
    attempt_synthesis: bool = True,
    apply_qsvt: bool = True,
) -> LinearSystemWorkflowResult:
    """
    Solve a positive-definite linear system with a QSVT-style inverse workflow.

    The input matrix is scaled by its largest eigenvalue so the scaled spectrum
    lies in ``[gamma_actual, 1]``. A bounded polynomial approximating
    ``gamma / x`` is designed on that interval, then rescaled back to an
    approximation of ``A^{-1} b``.
    """
    A, b, evals = _validate_linear_system_inputs(matrix, rhs)
    scaled = rescale_positive_semidefinite(A)
    scaled_min = float(evals[0] / scaled.scale)
    scaled_max = float(evals[-1] / scaled.scale)
    condition_number = float(evals[-1] / evals[0])

    if gamma is None:
        gamma = min(scaled_min, 1.0 - 1e-12)
    gamma = float(gamma)
    if not (0.0 < gamma < 1.0 and gamma <= scaled_min + 1e-12):
        raise ValueError(
            "gamma must satisfy 0 < gamma < 1 and be no larger than the "
            "scaled minimum eigenvalue."
        )

    coeffs = design_positive_inverse_polynomial(
        gamma=gamma,
        degree=degree,
        num_points=num_points,
    )
    diagnostics = design_positive_inverse_diagnostics(
        gamma=gamma,
        degree=degree,
        num_points=num_points,
        bounded_num_points=bounded_num_points,
    )
    compatibility = qsvt_compatibility_report(
        coeffs,
        bounded_num_points=bounded_num_points,
        attempt_synthesis=attempt_synthesis,
    )

    classical_solution = np.linalg.solve(A, b)
    polynomial_operator = apply_polynomial_to_hermitian(scaled.matrix, coeffs)
    solution_scale = gamma * scaled.scale
    polynomial_solution = (polynomial_operator @ b) / solution_scale
    polynomial_residual = A @ polynomial_solution - b

    qsvt_solution = None
    qsvt_residual_norm = None
    qsvt_relative_error = None
    qsvt_error = None

    if apply_qsvt:
        try:
            raw_qsvt = qsvt_matrix_transform(scaled.matrix, coeffs) @ b
            qsvt_solution = raw_qsvt / solution_scale
            qsvt_residual = A @ qsvt_solution - b
            qsvt_residual_norm = float(np.linalg.norm(qsvt_residual))
            qsvt_relative_error = _relative_error(classical_solution, qsvt_solution)
        except Exception as exc:  # pragma: no cover - backend-dependent path
            qsvt_error = f"{type(exc).__name__}: {exc}"

    return LinearSystemWorkflowResult(
        coeffs=coeffs,
        scaled_operator=scaled,
        rhs=b,
        classical_solution=classical_solution,
        polynomial_solution=polynomial_solution,
        polynomial_residual_norm=float(np.linalg.norm(polynomial_residual)),
        polynomial_relative_error=_relative_error(
            classical_solution,
            polynomial_solution,
        ),
        gamma=gamma,
        scaled_min_eigenvalue=scaled_min,
        scaled_max_eigenvalue=scaled_max,
        condition_number_2=condition_number,
        gamma_condition_proxy=float(1.0 / gamma),
        degree=int(degree),
        diagnostics=diagnostics,
        compatibility=compatibility,
        qsvt_solution=qsvt_solution,
        qsvt_residual_norm=qsvt_residual_norm,
        qsvt_relative_error=qsvt_relative_error,
        qsvt_error=qsvt_error,
    )


def linear_system_resource_proxy(
    *,
    degree: int,
    gamma: float,
    scaled_min_eigenvalue: float,
    condition_number_2: float,
    polynomial_residual_norm: float,
    polynomial_relative_error: float,
    qsvt_check: str,
    attempted_pennylane_synthesis: bool,
) -> dict[str, object]:
    """
    Return machine-readable proxy metadata for a linear-system workflow.
    """
    return {
        "proxy_kind": "linear-system-qsvt-style-resource-proxy",
        "degree": int(degree),
        "gamma": float(gamma),
        "scaled_min_eigenvalue": float(scaled_min_eigenvalue),
        "gamma_condition_proxy": float(1.0 / gamma),
        "condition_number_2": float(condition_number_2),
        "polynomial_residual_norm": float(polynomial_residual_norm),
        "polynomial_relative_error": float(polynomial_relative_error),
        "pennylane_qsvt_check": qsvt_check,
        "attempted_pennylane_synthesis": bool(attempted_pennylane_synthesis),
        "requires_block_encoding": True,
        "requires_state_preparation": True,
        "requires_success_probability_management": True,
        "requires_readout_strategy": True,
        "omitted_layers": [
            "state_preparation_or_data_loading",
            "oracle_or_block_encoding_construction",
            "controlled_reflection_sequence_cost",
            "success_probability_management",
            "amplitude_amplification",
            "measurement_readout_or_tomography",
            "fault_tolerant_synthesis",
            "hardware_compilation",
        ],
        "limitations": (
            "This is a polynomial-degree and conditioning proxy for a finite "
            "dense instance. It is not a quantum runtime or an end-to-end "
            "linear-system algorithm cost."
        ),
    }


def linear_system_comparison_workflow(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    degree: int,
    gamma: float | None = None,
    num_points: int = 2001,
    bounded_num_points: int = 4001,
    attempt_synthesis: bool = True,
    apply_qsvt: bool = True,
    include_conjugate_gradient: bool = True,
    cg_tolerance: float = 1e-10,
    cg_max_iterations: int | None = None,
    include_hhl_execution: bool = False,
    hhl_num_phase_qubits: int = 2,
    hhl_evolution_time: float | None = None,
    hhl_rotation_scale_C: float | None = None,
    hhl_eigenvalue_lower_bound: float | None = None,
) -> LinearSystemComparisonResult:
    """
    Compare dense, iterative, QSVT-style, and optional HHL linear-system paths.

    This workflow is not a timing benchmark. It returns a compact numerical
    table using the dense solve as the reference solution. When
    ``include_hhl_execution`` is true, the HHL row reports finite PennyLane
    QNode execution diagnostics for power-of-two positive-definite systems.
    """
    result = linear_system_workflow(
        matrix,
        rhs,
        degree=degree,
        gamma=gamma,
        num_points=num_points,
        bounded_num_points=bounded_num_points,
        attempt_synthesis=attempt_synthesis,
        apply_qsvt=apply_qsvt,
    )
    A, b, _ = _validate_linear_system_inputs(matrix, rhs)
    reference = result.classical_solution
    dense_residual = A @ reference - b
    rows: list[dict[str, Any]] = [
        {
            "solver": "dense_solve",
            "implementation_kind": "classical-dense-reference",
            "residual_norm": float(np.linalg.norm(dense_residual)),
            "relative_solution_error": 0.0,
            "condition_number_2": result.condition_number_2,
        },
        {
            "solver": "qsvt_style_polynomial_inverse",
            "implementation_kind": "dense-spectral-polynomial-workflow",
            "degree": result.degree,
            "gamma": result.gamma,
            "residual_norm": result.polynomial_residual_norm,
            "relative_solution_error": result.polynomial_relative_error,
            "condition_number_2": result.condition_number_2,
        },
    ]

    cg_solution = None
    if include_conjugate_gradient:
        from .benchmarks import conjugate_gradient_solve

        cg = conjugate_gradient_solve(
            A,
            b,
            tolerance=cg_tolerance,
            max_iterations=cg_max_iterations,
        )
        cg_solution = np.asarray(cg["solution"])
        rows.insert(
            1,
            {
                "solver": "conjugate_gradient",
                "implementation_kind": "classical-iterative-reference",
                "iterations": cg["iterations"],
                "converged": cg["converged"],
                "residual_norm": cg["residual_norm"],
                "relative_residual_norm": cg["relative_residual_norm"],
                "relative_solution_error": _relative_error(reference, cg_solution),
                "condition_number_2": result.condition_number_2,
            },
        )

    if result.qsvt_solution is not None:
        rows.append(
            {
                "solver": "pennylane_qsvt_matrix_check",
                "implementation_kind": "pennylane-small-qsvt-verification",
                "degree": result.degree,
                "gamma": result.gamma,
                "residual_norm": result.qsvt_residual_norm,
                "relative_solution_error": result.qsvt_relative_error,
                "condition_number_2": result.condition_number_2,
            },
        )
    elif result.qsvt_error is not None:
        rows.append(
            {
                "solver": "pennylane_qsvt_matrix_check",
                "implementation_kind": "pennylane-small-qsvt-verification",
                "status": "failed",
                "error": result.qsvt_error,
            },
        )

    if include_hhl_execution:
        rows.append(
            _hhl_comparison_row(
                A,
                b,
                num_phase_qubits=hhl_num_phase_qubits,
                evolution_time=hhl_evolution_time,
                rotation_scale_C=hhl_rotation_scale_C,
                eigenvalue_lower_bound=hhl_eigenvalue_lower_bound,
            )
        )

    return LinearSystemComparisonResult(
        workflow=result,
        rows=tuple(rows),
        dense_solution=reference,
        cg_solution=cg_solution,
        reference_solution=reference,
        notes=(
            "The dense solve is the numerical reference for this finite instance.",
            (
                "The QSVT-style row measures polynomial inverse accuracy, "
                "not quantum runtime."
            ),
        ),
    )


def _hhl_comparison_row(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    num_phase_qubits: int,
    evolution_time: float | None,
    rotation_scale_C: float | None,
    eigenvalue_lower_bound: float | None,
) -> dict[str, Any]:
    from .hhl import execute_hhl_circuit

    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
        lower_bound = (
            float(np.min(eigenvalues))
            if eigenvalue_lower_bound is None
            else float(eigenvalue_lower_bound)
        )
        t = (
            float(np.pi / np.max(eigenvalues))
            if evolution_time is None
            else float(evolution_time)
        )
        C = lower_bound if rotation_scale_C is None else float(rotation_scale_C)
        hhl = execute_hhl_circuit(
            matrix,
            rhs,
            num_phase_qubits=num_phase_qubits,
            evolution_time=t,
            rotation_scale_C=C,
            eigenvalue_lower_bound=lower_bound,
            normalize_state=True,
        )
    except Exception as exc:
        return {
            "solver": "hhl_circuit_execution",
            "implementation_kind": "pennylane-qnode-hhl-execution",
            "status": "failed",
            "error": str(exc),
        }

    resources = hhl.resource_summary
    return {
        "solver": "hhl_circuit_execution",
        "implementation_kind": hhl.execution_kind,
        "status": "ok",
        "success_probability": hhl.success_probability,
        "fidelity": hhl.fidelity,
        "state_error": hhl.state_error,
        "phase_qubits": hhl.num_phase_qubits,
        "system_qubits": len(hhl.system_wires),
        "total_wires": len(hhl.wire_order),
        "num_gates": resources.get("num_gates"),
        "circuit_depth": resources.get("depth"),
        "uses_dense_time_evolution": hhl.uses_dense_time_evolution,
        "is_executable_hhl_circuit": True,
        "is_scalable_hhl_implementation": not hhl.uses_dense_time_evolution,
        "evolution_time": hhl.evolution_time,
        "rotation_scale_C": hhl.rotation_scale_C,
        "eigenvalue_lower_bound": hhl.eigenvalue_lower_bound,
    }


def linear_system_comparison_summary_table(
    report: LinearSystemComparisonResult | dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Convert a linear-system comparison report into compact table rows.
    """
    payload = (
        report.as_report()
        if isinstance(report, LinearSystemComparisonResult)
        else report
    )
    workflow = payload.get("linear_system_workflow", {})
    resource_proxy = payload.get("resource_proxy", {})
    rhs = workflow.get("rhs", [])
    rows = []
    for row in payload.get("rows", []):
        rows.append(
            {
                "solver": row.get("solver"),
                "implementation_kind": row.get("implementation_kind"),
                "matrix_dimension": len(rhs),
                "degree": row.get("degree", resource_proxy.get("degree")),
                "gamma": row.get("gamma", resource_proxy.get("gamma")),
                "condition_number_2": row.get(
                    "condition_number_2",
                    resource_proxy.get("condition_number_2"),
                ),
                "iterations": row.get("iterations"),
                "converged": row.get("converged"),
                "residual_norm": row.get("residual_norm"),
                "relative_residual_norm": row.get("relative_residual_norm"),
                "relative_solution_error": row.get("relative_solution_error"),
                "success_probability": row.get("success_probability"),
                "fidelity": row.get("fidelity"),
                "state_error": row.get("state_error"),
                "phase_qubits": row.get("phase_qubits"),
                "total_wires": row.get("total_wires"),
                "num_gates": row.get("num_gates"),
                "circuit_depth": row.get("circuit_depth"),
                "uses_dense_time_evolution": row.get("uses_dense_time_evolution"),
                "status": row.get("status", "ok"),
            }
        )
    return rows


def write_linear_system_comparison_csv(
    report: LinearSystemComparisonResult | dict[str, Any],
    path: str | Path,
) -> Path:
    """
    Write compact linear-system comparison rows to a CSV file.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = linear_system_comparison_summary_table(report)
    fieldnames = [
        "solver",
        "implementation_kind",
        "matrix_dimension",
        "degree",
        "gamma",
        "condition_number_2",
        "iterations",
        "converged",
        "residual_norm",
        "relative_residual_norm",
        "relative_solution_error",
        "success_probability",
        "fidelity",
        "state_error",
        "phase_qubits",
        "total_wires",
        "num_gates",
        "circuit_depth",
        "uses_dense_time_evolution",
        "status",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return output_path
