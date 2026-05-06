"""
qsvt
----

Educational PennyLane-based utilities for Quantum Singular Value
Transformation (QSVT) and Quantum Signal Processing (QSP).

This package provides lightweight helpers for:

- Chebyshev and polynomial utilities
- function approximation on bounded intervals
- small Hermitian matrix construction
- spectral matrix-function reference calculations
- explicit QSVT matrix extraction and comparison workflows

The project is designed for small-scale demonstrations, notebooks, and
classical validation of QSVT/QSP ideas.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from .design import (
    design_filter_polynomial,
    design_filter_diagnostics,
    design_inverse_polynomial,
    design_inverse_diagnostics,
    design_power_polynomial,
    design_power_diagnostics,
    design_projector_polynomial,
    design_projector_diagnostics,
    design_sign_polynomial,
    design_sign_diagnostics,
    design_sqrt_polynomial,
    design_sqrt_diagnostics,
)
from .approximation import (
    approximation_quality_report,
    chebyshev_approximant,
    chebyshev_eval,
    chebyshev_fit_function,
    fit_and_build_approximant,
    max_error,
    rms_error,
    sample_approximation,
    scale_from_chebyshev_domain,
    scale_to_chebyshev_domain,
)
from .templates import (
    exponential_approximation_polynomial,
    exponential_approximation_diagnostics,
    inverse_like_polynomial,
    inverse_like_diagnostics,
    sign_approximation_polynomial,
    sign_approximation_diagnostics,
    soft_threshold_filter_polynomial,
    soft_threshold_filter_diagnostics,
    sqrt_approximation_polynomial,
    sqrt_approximation_diagnostics,
)
from .matrices import (
    diagonal_matrix,
    embed_vector,
    hermitian_from_eigendecomposition,
    identity,
    involutory_diagonal,
    normalized_vector,
    pauli_x,
    pauli_z,
    rotated_diagonal,
    rotation,
)
from .polynomials import (
    chebyshev_t,
    chebyshev_t3,
    eval_polynomial,
    is_bounded_on_interval,
    normalize_coefficients,
    polynomial_degree,
    polynomial_parity,
)
from .qsvt import (
    apply_qsvt_to_embedded_vector,
    classical_diagonal_polynomial_transform,
    compare_qsvt_vs_classical_diagonal,
    compare_qsvt_vs_classical_matrix,
    qsvt_diagonal_transform,
    qsvt_matrix_transform,
    qsvt_matrix_transform_report,
    qsvt_operator,
    qsvt_compatibility_report,
    qsvt_scalar_output,
    qsvt_scalar_scan,
    qsvt_top_left_block,
    qsvt_transform_report,
    qsvt_unitary,
)
from .reports import (
    load_report,
    plot_approximation_report,
    report_to_jsonable,
    save_report,
    save_report_plot,
)
from .spectral import (
    apply_function_to_hermitian,
    apply_polynomial_to_hermitian,
    eigh_hermitian,
    matrix_fractional_power,
    matrix_from_eigendecomposition,
    matrix_power_spectral,
    matrix_sign,
    matrix_square_root,
    negative_projector_from_sign,
    positive_projector_from_sign,
    spectral_projector_negative,
    spectral_projector_positive,
    transformed_eigenvalues,
)

# Version ----------------------------------------------------------------------
try:
    __version__ = _pkg_version("qsvt-pennylane")
except PackageNotFoundError:  # pragma: no cover
    # Allows editable installs / local runs without installed dist metadata.
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "approximation_quality_report",
    "chebyshev_approximant",
    "chebyshev_eval",
    "chebyshev_fit_function",
    "fit_and_build_approximant",
    "max_error",
    "rms_error",
    "sample_approximation",
    "scale_from_chebyshev_domain",
    "scale_to_chebyshev_domain",
    "diagonal_matrix",
    "embed_vector",
    "exponential_approximation_polynomial",
    "exponential_approximation_diagnostics",
    "inverse_like_polynomial",
    "inverse_like_diagnostics",
    "sign_approximation_polynomial",
    "sign_approximation_diagnostics",
    "soft_threshold_filter_polynomial",
    "soft_threshold_filter_diagnostics",
    "sqrt_approximation_polynomial",
    "sqrt_approximation_diagnostics",
    "hermitian_from_eigendecomposition",
    "identity",
    "involutory_diagonal",
    "normalized_vector",
    "pauli_x",
    "pauli_z",
    "design_filter_polynomial",
    "design_filter_diagnostics",
    "design_inverse_polynomial",
    "design_inverse_diagnostics",
    "design_power_polynomial",
    "design_power_diagnostics",
    "design_projector_polynomial",
    "design_projector_diagnostics",
    "design_sign_polynomial",
    "design_sign_diagnostics",
    "design_sqrt_polynomial",
    "design_sqrt_diagnostics",
    "rotated_diagonal",
    "rotation",
    "chebyshev_t",
    "chebyshev_t3",
    "eval_polynomial",
    "is_bounded_on_interval",
    "normalize_coefficients",
    "polynomial_degree",
    "polynomial_parity",
    "apply_qsvt_to_embedded_vector",
    "classical_diagonal_polynomial_transform",
    "compare_qsvt_vs_classical_diagonal",
    "compare_qsvt_vs_classical_matrix",
    "qsvt_diagonal_transform",
    "qsvt_matrix_transform",
    "qsvt_matrix_transform_report",
    "qsvt_operator",
    "qsvt_compatibility_report",
    "qsvt_scalar_output",
    "qsvt_scalar_scan",
    "qsvt_top_left_block",
    "qsvt_transform_report",
    "qsvt_unitary",
    "load_report",
    "plot_approximation_report",
    "report_to_jsonable",
    "save_report",
    "save_report_plot",
    "apply_function_to_hermitian",
    "apply_polynomial_to_hermitian",
    "eigh_hermitian",
    "matrix_fractional_power",
    "matrix_from_eigendecomposition",
    "matrix_power_spectral",
    "matrix_sign",
    "matrix_square_root",
    "negative_projector_from_sign",
    "positive_projector_from_sign",
    "spectral_projector_negative",
    "spectral_projector_positive",
    "transformed_eigenvalues",
]
