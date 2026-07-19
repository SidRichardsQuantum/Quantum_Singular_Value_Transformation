"""Thermal, occupation, and matrix-entropy workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._algorithm_reports import (
    algorithm_truth_contract,
    algorithm_workflow_schema_fields,
    scaled_operator_report,
)
from ._algorithm_shared import _normalize_state, _relative_error, _validate_state
from .approximation import chebyshev_fit_function
from .diagnostics import operator_error
from .matrix_functions import ScaledPolynomial, design_imaginary_time_polynomial
from .polynomials import chebyshev_to_monomial
from .rescaling import (
    ScaledOperator,
    rescale_hermitian_to_unit_interval,
    rescale_positive_semidefinite,
)
from .spectral import (
    apply_function_to_hermitian,
    apply_polynomial_to_hermitian,
    eigh_hermitian,
)


@dataclass(frozen=True)
class ThermalGibbsWorkflowResult:
    """
    Structured output from an imaginary-time / Gibbs weighting workflow.
    """

    coeffs: np.ndarray
    prefactor: float
    scaled_operator: ScaledOperator
    polynomial_boltzmann_operator: np.ndarray
    reference_boltzmann_operator: np.ndarray
    polynomial_gibbs_state: np.ndarray
    reference_gibbs_state: np.ndarray
    beta: float
    degree: int
    polynomial_partition_function: float | complex
    reference_partition_function: float | complex
    operator_relative_error: float
    density_matrix_relative_error: float
    state: np.ndarray | None = None
    polynomial_weighted_state: np.ndarray | None = None
    reference_weighted_state: np.ndarray | None = None
    weighted_state_error: float | None = None

    def as_report(self) -> dict[str, Any]:
        """
        Return a report-style dictionary for JSON conversion or persistence.
        """
        return {
            **algorithm_workflow_schema_fields(),
            "mode": "thermal-gibbs-workflow",
            "implementation_kind": "dense-spectral-polynomial-workflow",
            "truth_contract": algorithm_truth_contract(
                "thermal-gibbs-workflow",
                target="imaginary-time Boltzmann weighting and Gibbs normalization",
            ),
            "beta": self.beta,
            "degree": self.degree,
            "coeffs": self.coeffs,
            "prefactor": self.prefactor,
            "scaled_operator": scaled_operator_report(self.scaled_operator),
            "polynomial_boltzmann_operator": self.polynomial_boltzmann_operator,
            "reference_boltzmann_operator": self.reference_boltzmann_operator,
            "polynomial_gibbs_state": self.polynomial_gibbs_state,
            "reference_gibbs_state": self.reference_gibbs_state,
            "polynomial_partition_function": self.polynomial_partition_function,
            "reference_partition_function": self.reference_partition_function,
            "operator_relative_error": self.operator_relative_error,
            "density_matrix_relative_error": self.density_matrix_relative_error,
            "state": self.state,
            "polynomial_weighted_state": self.polynomial_weighted_state,
            "reference_weighted_state": self.reference_weighted_state,
            "weighted_state_error": self.weighted_state_error,
        }


@dataclass(frozen=True)
class FermiDiracWorkflowResult:
    """
    Structured output from a Fermi-Dirac occupation workflow.
    """

    coeffs: np.ndarray
    scaled_operator: ScaledOperator
    polynomial_occupation_operator: np.ndarray
    reference_occupation_operator: np.ndarray
    chemical_potential: float
    beta: float
    degree: int
    particle_number: float | complex
    reference_particle_number: float | complex
    operator_relative_error: float
    state: np.ndarray | None = None
    polynomial_state_occupation: float | complex | None = None
    reference_state_occupation: float | complex | None = None
    state_occupation_error: float | None = None

    def as_report(self) -> dict[str, Any]:
        return {
            **algorithm_workflow_schema_fields(),
            "mode": "fermi-dirac-occupation-workflow",
            "implementation_kind": "dense-spectral-polynomial-workflow",
            "truth_contract": algorithm_truth_contract(
                "fermi-dirac-occupation-workflow",
                target="finite-temperature Fermi-Dirac spectral occupation",
            ),
            "chemical_potential": self.chemical_potential,
            "beta": self.beta,
            "degree": self.degree,
            "coeffs": self.coeffs,
            "scaled_operator": scaled_operator_report(self.scaled_operator),
            "polynomial_occupation_operator": self.polynomial_occupation_operator,
            "reference_occupation_operator": self.reference_occupation_operator,
            "particle_number": self.particle_number,
            "reference_particle_number": self.reference_particle_number,
            "operator_relative_error": self.operator_relative_error,
            "state": self.state,
            "polynomial_state_occupation": self.polynomial_state_occupation,
            "reference_state_occupation": self.reference_state_occupation,
            "state_occupation_error": self.state_occupation_error,
        }


@dataclass(frozen=True)
class MatrixLogEntropyWorkflowResult:
    """
    Structured output from a regularized matrix-log and entropy workflow.
    """

    log_coeffs: np.ndarray
    entropy_coeffs: np.ndarray
    scaled_operator: ScaledOperator
    polynomial_log_operator: np.ndarray
    reference_log_operator: np.ndarray
    polynomial_entropy_operator: np.ndarray
    reference_entropy_operator: np.ndarray
    epsilon: float
    degree: int
    polynomial_entropy: float | complex
    reference_entropy: float | complex
    log_operator_relative_error: float
    entropy_operator_relative_error: float

    def as_report(self) -> dict[str, Any]:
        return {
            **algorithm_workflow_schema_fields(),
            "mode": "matrix-log-entropy-workflow",
            "implementation_kind": "dense-spectral-polynomial-workflow",
            "truth_contract": algorithm_truth_contract(
                "matrix-log-entropy-workflow",
                target="regularized matrix logarithm and x log x entropy density",
            ),
            "epsilon": self.epsilon,
            "degree": self.degree,
            "log_coeffs": self.log_coeffs,
            "entropy_coeffs": self.entropy_coeffs,
            "scaled_operator": scaled_operator_report(self.scaled_operator),
            "polynomial_log_operator": self.polynomial_log_operator,
            "reference_log_operator": self.reference_log_operator,
            "polynomial_entropy_operator": self.polynomial_entropy_operator,
            "reference_entropy_operator": self.reference_entropy_operator,
            "polynomial_entropy": self.polynomial_entropy,
            "reference_entropy": self.reference_entropy,
            "log_operator_relative_error": self.log_operator_relative_error,
            "entropy_operator_relative_error": self.entropy_operator_relative_error,
        }


def _stable_fermi_dirac(
    energies: np.ndarray,
    *,
    chemical_potential: float,
    beta: float,
) -> np.ndarray:
    z = np.clip(float(beta) * (energies - float(chemical_potential)), -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(z))


def fermi_dirac_occupation_workflow(
    matrix: np.ndarray,
    *,
    chemical_potential: float,
    beta: float,
    degree: int,
    state: np.ndarray | None = None,
    num_points: int = 2001,
) -> FermiDiracWorkflowResult:
    """
    Approximate finite-temperature Fermi-Dirac occupations for a Hamiltonian.
    """
    if beta < 0.0:
        raise ValueError("beta must be non-negative.")
    scaled = rescale_hermitian_to_unit_interval(matrix)
    coeffs = chebyshev_to_monomial(
        chebyshev_fit_function(
            lambda x: _stable_fermi_dirac(
                scaled.offset + scaled.scale * x,
                chemical_potential=chemical_potential,
                beta=beta,
            ),
            degree=degree,
            num_points=num_points,
        )
    )
    polynomial_operator = apply_polynomial_to_hermitian(scaled.matrix, coeffs)
    reference_operator = apply_function_to_hermitian(
        np.asarray(matrix),
        lambda x: _stable_fermi_dirac(
            x,
            chemical_potential=chemical_potential,
            beta=beta,
        ),
    )

    state_vec = None
    polynomial_state_occupation = None
    reference_state_occupation = None
    state_error = None
    if state is not None:
        state_vec = _normalize_state(_validate_state(state, scaled.matrix.shape[0]))
        polynomial_state_occupation = np.real_if_close(
            np.vdot(state_vec, polynomial_operator @ state_vec)
        ).item()
        reference_state_occupation = np.real_if_close(
            np.vdot(state_vec, reference_operator @ state_vec)
        ).item()
        state_error = float(
            abs(polynomial_state_occupation - reference_state_occupation)
        )

    return FermiDiracWorkflowResult(
        coeffs=coeffs,
        scaled_operator=scaled,
        polynomial_occupation_operator=polynomial_operator,
        reference_occupation_operator=reference_operator,
        chemical_potential=float(chemical_potential),
        beta=float(beta),
        degree=int(degree),
        particle_number=np.real_if_close(np.trace(polynomial_operator)).item(),
        reference_particle_number=np.real_if_close(np.trace(reference_operator)).item(),
        operator_relative_error=operator_error(reference_operator, polynomial_operator),
        state=state_vec,
        polynomial_state_occupation=polynomial_state_occupation,
        reference_state_occupation=reference_state_occupation,
        state_occupation_error=state_error,
    )


def matrix_log_entropy_workflow(
    matrix: np.ndarray,
    *,
    degree: int,
    epsilon: float = 1e-8,
    num_points: int = 2001,
) -> MatrixLogEntropyWorkflowResult:
    """
    Approximate a regularized matrix logarithm and ``-x log(x)`` entropy term.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    evals, _ = eigh_hermitian(matrix)
    if evals[0] < -1e-10:
        raise ValueError("matrix must be positive semidefinite.")
    scaled = rescale_positive_semidefinite(np.asarray(matrix))

    def physical_from_scaled(x: np.ndarray) -> np.ndarray:
        return scaled.scale * x

    log_coeffs = chebyshev_to_monomial(
        chebyshev_fit_function(
            lambda x: np.log(physical_from_scaled(x) + float(epsilon)),
            degree=degree,
            domain=(0.0, 1.0),
            num_points=num_points,
        ),
        domain=(0.0, 1.0),
    )
    entropy_coeffs = chebyshev_to_monomial(
        chebyshev_fit_function(
            lambda x: (
                -physical_from_scaled(x)
                * np.log(physical_from_scaled(x) + float(epsilon))
            ),
            degree=degree,
            domain=(0.0, 1.0),
            num_points=num_points,
        ),
        domain=(0.0, 1.0),
    )
    polynomial_log = apply_polynomial_to_hermitian(scaled.matrix, log_coeffs)
    reference_log = apply_function_to_hermitian(
        np.asarray(matrix),
        lambda x: np.log(x + float(epsilon)),
    )
    polynomial_entropy_op = apply_polynomial_to_hermitian(
        scaled.matrix,
        entropy_coeffs,
    )
    reference_entropy_op = apply_function_to_hermitian(
        np.asarray(matrix),
        lambda x: -x * np.log(x + float(epsilon)),
    )

    return MatrixLogEntropyWorkflowResult(
        log_coeffs=log_coeffs,
        entropy_coeffs=entropy_coeffs,
        scaled_operator=scaled,
        polynomial_log_operator=polynomial_log,
        reference_log_operator=reference_log,
        polynomial_entropy_operator=polynomial_entropy_op,
        reference_entropy_operator=reference_entropy_op,
        epsilon=float(epsilon),
        degree=int(degree),
        polynomial_entropy=np.real_if_close(np.trace(polynomial_entropy_op)).item(),
        reference_entropy=np.real_if_close(np.trace(reference_entropy_op)).item(),
        log_operator_relative_error=operator_error(reference_log, polynomial_log),
        entropy_operator_relative_error=operator_error(
            reference_entropy_op,
            polynomial_entropy_op,
        ),
    )


def thermal_gibbs_workflow(
    matrix: np.ndarray,
    *,
    beta: float,
    degree: int,
    state: np.ndarray | None = None,
    num_points: int = 2001,
) -> ThermalGibbsWorkflowResult:
    """
    Approximate ``exp(-beta H)`` and the normalized Gibbs density matrix.
    """
    if beta < 0.0:
        raise ValueError("beta must be non-negative.")

    scaled = rescale_hermitian_to_unit_interval(matrix)
    design: ScaledPolynomial = design_imaginary_time_polynomial(
        beta=beta,
        scale=scaled.scale,
        offset=scaled.offset,
        degree=degree,
        num_points=num_points,
    )
    polynomial_boltzmann = design.prefactor * apply_polynomial_to_hermitian(
        scaled.matrix,
        design.coeffs,
    )
    reference_boltzmann = apply_function_to_hermitian(
        np.asarray(matrix),
        lambda x: np.exp(-float(beta) * x),
    )
    polynomial_partition = np.trace(polynomial_boltzmann)
    reference_partition = np.trace(reference_boltzmann)
    polynomial_gibbs = polynomial_boltzmann / polynomial_partition
    reference_gibbs = reference_boltzmann / reference_partition

    state_vec = None
    polynomial_weighted = None
    reference_weighted = None
    weighted_error = None
    if state is not None:
        state_vec = _normalize_state(_validate_state(state, scaled.matrix.shape[0]))
        polynomial_weighted = polynomial_boltzmann @ state_vec
        reference_weighted = reference_boltzmann @ state_vec
        weighted_error = _relative_error(reference_weighted, polynomial_weighted)

    return ThermalGibbsWorkflowResult(
        coeffs=design.coeffs,
        prefactor=design.prefactor,
        scaled_operator=scaled,
        polynomial_boltzmann_operator=polynomial_boltzmann,
        reference_boltzmann_operator=reference_boltzmann,
        polynomial_gibbs_state=polynomial_gibbs,
        reference_gibbs_state=reference_gibbs,
        beta=float(beta),
        degree=int(degree),
        polynomial_partition_function=np.real_if_close(polynomial_partition).item(),
        reference_partition_function=np.real_if_close(reference_partition).item(),
        operator_relative_error=operator_error(
            reference_boltzmann,
            polynomial_boltzmann,
        ),
        density_matrix_relative_error=operator_error(reference_gibbs, polynomial_gibbs),
        state=state_vec,
        polynomial_weighted_state=polynomial_weighted,
        reference_weighted_state=reference_weighted,
        weighted_state_error=weighted_error,
    )
