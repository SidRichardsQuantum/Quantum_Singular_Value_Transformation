import numpy as np
import pytest

from qsvt.approximation import (
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
from qsvt.matrices import (
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
from qsvt.spectral import (
    apply_function_to_hermitian,
    apply_polynomial_to_hermitian,
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


def test_matrix_constructors_and_validation_paths():
    assert np.allclose(diagonal_matrix([1.0, -1.0]), np.diag([1.0, -1.0]))
    assert np.allclose(identity(2), np.eye(2))
    assert np.allclose(pauli_x() @ pauli_x(), np.eye(2))
    assert np.allclose(pauli_z() @ pauli_z(), np.eye(2))

    R = rotation(0.3)
    assert np.allclose(R.T @ R, np.eye(2))
    assert np.allclose(
        np.linalg.eigvalsh(rotated_diagonal([0.2, 0.8], 0.3)), [0.2, 0.8]
    )

    V = np.array([[1.0, 0.0], [0.0, 1.0j]])
    assert np.allclose(
        hermitian_from_eigendecomposition([1.0, 2.0], V),
        np.diag([1.0, 2.0]),
    )
    assert np.allclose(
        involutory_diagonal([1, -1]) @ involutory_diagonal([1, -1]), np.eye(2)
    )
    assert np.allclose(normalized_vector([3.0, 4.0]), [0.6, 0.8])
    assert np.allclose(embed_vector([1.0, 2.0], 4), [1.0, 2.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="positive"):
        identity(0)
    with pytest.raises(ValueError, match="2x2"):
        rotated_diagonal([1.0, 2.0, 3.0], 0.1)
    with pytest.raises(ValueError, match="square"):
        hermitian_from_eigendecomposition([1.0, 2.0], np.ones((2, 3)))
    with pytest.raises(ValueError, match="Mismatch"):
        hermitian_from_eigendecomposition([1.0], np.eye(2))
    with pytest.raises(ValueError, match="Entries"):
        involutory_diagonal([1, 0])
    with pytest.raises(ValueError, match="zero vector"):
        normalized_vector([0.0, 0.0])
    with pytest.raises(ValueError, match="dimension"):
        embed_vector([1.0, 2.0, 3.0], 2)


def test_approximation_helpers_cover_scalars_arrays_and_invalid_inputs():
    assert scale_to_chebyshev_domain(0.5, (0.0, 1.0)) == pytest.approx(0.0)
    assert scale_from_chebyshev_domain(0.0, (0.0, 1.0)) == pytest.approx(0.5)
    assert np.allclose(
        scale_to_chebyshev_domain(np.array([0.0, 1.0]), (0.0, 1.0)),
        [-1.0, 1.0],
    )

    coeffs = chebyshev_fit_function(lambda x: x**2, degree=2, num_points=20)
    assert chebyshev_eval(coeffs, 0.5) == pytest.approx(0.25)
    approx = chebyshev_approximant(coeffs)
    assert np.allclose(approx(np.array([0.0, 0.5])), [0.0, 0.25])
    assert max_error(lambda x: x**2, approx, num_points=25) < 1e-12
    assert rms_error(lambda x: x**2, approx, num_points=25) < 1e-12

    built_coeffs, built = fit_and_build_approximant(lambda x: x + 1.0, 1, num_points=10)
    assert len(built_coeffs) == 2
    assert built(0.25) == pytest.approx(1.25)

    xs, target, values = sample_approximation(
        lambda x: x, lambda x: x + 0.1, num_points=5
    )
    assert xs.shape == target.shape == values.shape == (5,)
    report = approximation_quality_report(
        lambda x: x,
        lambda x: x + 0.1,
        num_points=5,
        bounded_domain=(-0.5, 0.5),
        bounded_num_points=7,
        coeffs=[0.1, 1.0],
    )
    assert report["max_error"] == pytest.approx(0.1)
    assert report["coeffs"].shape == (2,)

    with pytest.raises(ValueError, match="2-tuple"):
        scale_to_chebyshev_domain(0.0, (0.0, 1.0, 2.0))
    with pytest.raises(ValueError, match="finite"):
        scale_to_chebyshev_domain(0.0, (0.0, np.inf))
    with pytest.raises(ValueError, match="lower < upper"):
        scale_to_chebyshev_domain(0.0, (1.0, 1.0))
    with pytest.raises(ValueError, match="non-negative"):
        chebyshev_fit_function(lambda x: x, degree=-1)
    with pytest.raises(ValueError, match="at least 2"):
        max_error(lambda x: x, lambda x: x, num_points=1)


def test_spectral_helpers_and_validation_paths():
    A = np.array([[2.0, 0.0], [0.0, 3.0]])
    assert np.allclose(matrix_from_eigendecomposition([2.0, 3.0], np.eye(2)), A)
    assert np.allclose(
        apply_function_to_hermitian(A, lambda x: x + 1.0), np.diag([3.0, 4.0])
    )
    assert np.allclose(
        apply_polynomial_to_hermitian(A, [1.0, 0.0, 1.0]), np.diag([5.0, 10.0])
    )
    assert np.allclose(matrix_power_spectral(A, 0), np.eye(2))
    assert np.allclose(matrix_power_spectral(A, 2), np.diag([4.0, 9.0]))
    assert np.allclose(matrix_square_root(A), np.diag(np.sqrt([2.0, 3.0])))
    assert np.allclose(matrix_fractional_power(A, 0.5), np.diag(np.sqrt([2.0, 3.0])))

    B = np.diag([-2.0, 0.0, 3.0])
    assert np.allclose(matrix_sign(B), np.diag([-1.0, 0.0, 1.0]))
    assert np.allclose(spectral_projector_positive(B), np.diag([0.0, 0.0, 1.0]))
    assert np.allclose(spectral_projector_negative(B), np.diag([1.0, 0.0, 0.0]))
    assert np.allclose(positive_projector_from_sign(B), np.diag([0.0, 0.5, 1.0]))
    assert np.allclose(negative_projector_from_sign(B), np.diag([1.0, 0.5, 0.0]))
    assert np.allclose(transformed_eigenvalues(B, np.abs), [2.0, 0.0, 3.0])

    with pytest.raises(ValueError, match="square"):
        matrix_from_eigendecomposition([1.0, 2.0], np.ones((2, 3)))
    with pytest.raises(ValueError, match="Mismatch"):
        matrix_from_eigendecomposition([1.0], np.eye(2))
    with pytest.raises(ValueError, match="square"):
        apply_function_to_hermitian(np.ones((2, 3)), lambda x: x)
    with pytest.raises(ValueError, match="Hermitian"):
        apply_function_to_hermitian(np.array([[0.0, 1.0], [0.0, 0.0]]), lambda x: x)
    with pytest.raises(ValueError, match="same shape"):
        apply_function_to_hermitian(A, lambda x: np.array([1.0]))
    with pytest.raises(ValueError, match="non-negative"):
        matrix_power_spectral(A, -1)
    with pytest.raises(ValueError, match="negative eigenvalues"):
        matrix_fractional_power(B, 0.5)
    with pytest.raises(ValueError, match="same shape"):
        transformed_eigenvalues(A, lambda x: np.array([1.0]))
