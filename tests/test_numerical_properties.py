import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qsvt.block_encoding import block_encode_matrix, verify_block_encoding
from qsvt.polynomials import eval_polynomial
from qsvt.spectral import apply_polynomial_to_hermitian
from qsvt.synthesis import (
    certify_polynomial_boundedness,
    classify_polynomial_realizability,
    synthesize_phases,
)

FINITE_COEFFS = st.lists(
    st.floats(
        min_value=-2.0,
        max_value=2.0,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    ),
    min_size=1,
    max_size=9,
)


@settings(max_examples=40, deadline=None)
@given(FINITE_COEFFS)
def test_extrema_certificate_dominates_dense_sampling(coeffs):
    certificate = certify_polynomial_boundedness(coeffs)
    points = np.linspace(-1.0, 1.0, 4001)
    sampled_max = float(np.max(np.abs(eval_polynomial(coeffs, points))))

    assert certificate.max_abs_value >= sampled_max - 1e-8
    assert certificate.is_bounded == (
        certificate.max_abs_value <= certificate.bound + certificate.tolerance
    )
    assert -1.0 <= certificate.maximizing_point <= 1.0


@settings(max_examples=30, deadline=None)
@given(
    st.sampled_from(["even", "odd"]),
    st.lists(
        st.floats(
            min_value=-1.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
        min_size=1,
        max_size=5,
    ).filter(lambda values: any(abs(value) > 1e-5 for value in values)),
)
def test_same_parity_polynomials_are_bounded_and_single_sequence(parity, values):
    offset = 0 if parity == "even" else 1
    coeffs = np.zeros(2 * len(values), dtype=float)
    coeffs[offset::2] = values
    coeffs *= 0.9 / max(float(np.sum(np.abs(coeffs))), 1.0)

    classification = classify_polynomial_realizability(coeffs)

    assert classification.parity == parity
    assert classification.bounded is True
    assert classification.single_sequence_realizable is True
    assert classification.requires_parity_decomposition is False


@settings(max_examples=25, deadline=None)
@given(
    st.floats(
        min_value=1e-4,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
    st.floats(
        min_value=1e-4,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)
def test_mixed_parity_polynomials_require_explicit_combination(even, odd):
    coeffs = np.array([even, odd], dtype=float)
    coeffs *= 0.9 / float(np.sum(np.abs(coeffs)))

    classification = classify_polynomial_realizability(coeffs)

    assert classification.parity == "mixed"
    assert classification.bounded is True
    assert classification.single_sequence_realizable is False
    assert classification.requires_parity_decomposition is True


@settings(max_examples=25, deadline=None)
@given(
    st.lists(
        st.floats(
            min_value=-2.0,
            max_value=2.0,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
        min_size=4,
        max_size=4,
    )
)
def test_dense_block_encoding_preserves_normalization_and_unitarity(entries):
    matrix = np.asarray(entries, dtype=float).reshape(2, 2)
    norm = float(np.linalg.norm(matrix, ord=2))
    alpha = max(1.0, 1.05 * norm)

    encoding = block_encode_matrix(matrix, alpha=alpha)
    verification = verify_block_encoding(encoding, block_atol=1e-9, unitary_atol=1e-9)

    assert np.allclose(encoding.top_left_block(), matrix / alpha, atol=1e-9)
    assert verification["block_encoding_verified"] is True
    assert verification["unitary_verified"] is True


@settings(max_examples=12, deadline=None)
@given(
    st.floats(
        min_value=1e-12,
        max_value=1e-5,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    )
)
def test_polynomial_matrix_action_is_stable_at_spectral_boundaries(epsilon):
    eigenvalues = np.array([-1.0, -1.0 + epsilon, 1.0 - epsilon, 1.0])
    matrix = np.diag(eigenvalues)
    coeffs = np.array([0.0, 0.25, 0.0, -0.5, 0.0, 0.2])

    transformed = apply_polynomial_to_hermitian(matrix, coeffs)

    assert np.all(np.isfinite(transformed))
    assert np.allclose(
        np.diag(transformed),
        eval_polynomial(coeffs, eigenvalues),
        atol=1e-11,
        rtol=1e-11,
    )


@pytest.mark.parametrize("degree", [16, 25, 36])
def test_high_degree_chebyshev_polynomials_have_exact_boundedness_certificates(degree):
    chebyshev_coeffs = np.zeros(degree + 1)
    chebyshev_coeffs[degree] = 0.95
    coeffs = np.polynomial.chebyshev.cheb2poly(chebyshev_coeffs)

    certificate = certify_polynomial_boundedness(coeffs, tolerance=1e-8)
    classification = classify_polynomial_realizability(coeffs)

    assert certificate.is_bounded is True
    assert certificate.max_abs_value == pytest.approx(0.95, abs=2e-3)
    assert classification.parity == ("even" if degree % 2 == 0 else "odd")
    assert classification.single_sequence_realizable is True


@settings(max_examples=7, deadline=None)
@given(st.integers(min_value=1, max_value=7))
def test_monomial_phase_synthesis_reconstructs_across_parities(degree):
    coeffs = np.zeros(degree + 1)
    coeffs[degree] = 0.5

    result = synthesize_phases(coeffs, reconstruction_num_points=65)

    assert result.succeeded is True
    assert result.reconstruction_max_error is not None
    assert result.reconstruction_max_error < 1e-7
