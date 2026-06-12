import numpy as np
import pytest

from qsvt.algorithms import (
    GroundStateFilteringWorkflowResult,
    HamiltonianSimulationWorkflowResult,
    LinearSystemWorkflowResult,
    ResolventWorkflowResult,
    SpectralDensityWorkflowResult,
    ThermalGibbsWorkflowResult,
    ground_state_filtering_workflow,
    hamiltonian_simulation_workflow,
    linear_system_workflow,
    resolvent_workflow,
    spectral_density_workflow,
    thermal_gibbs_workflow,
)
from qsvt.design import (
    design_positive_inverse_diagnostics,
    design_positive_inverse_polynomial,
)
from qsvt.diagnostics import (
    density_matrix_error,
    expectation_value,
    ground_state_overlap,
    operator_error,
    relative_state_error,
    spectral_weights,
)
from qsvt.hamiltonians import (
    heisenberg_chain,
    ising_hamiltonian,
    pauli_string_matrix,
    tight_binding_chain,
)
from qsvt.matrix_functions import (
    design_gaussian_window_polynomial,
    design_imaginary_time_polynomial,
    design_real_time_evolution_polynomials,
    design_resolvent_polynomials,
)
from qsvt.pde import (
    dirichlet_laplacian_1d,
    dirichlet_laplacian_2d,
    periodic_laplacian_1d,
)
from qsvt.polynomials import (
    chebyshev_to_monomial,
    eval_polynomial,
    monomial_to_chebyshev,
)
from qsvt.rescaling import (
    rescale_hermitian_about_cutoff,
    rescale_hermitian_to_unit_interval,
    rescale_positive_semidefinite,
    spectral_bounds,
)
from qsvt.spectral import apply_polynomial_to_hermitian


def test_basis_conversion_round_trips_on_canonical_domain():
    coeffs = np.array([0.25, -0.5, 0.75, 0.1])
    cheb = monomial_to_chebyshev(coeffs)
    monomial = chebyshev_to_monomial(cheb)
    xs = np.linspace(-1.0, 1.0, 31)

    assert np.allclose(eval_polynomial(coeffs, xs), eval_polynomial(monomial, xs))


def test_basis_conversion_respects_physical_domain():
    # T_1 on domain (0, 2) is x - 1 in the physical coordinate.
    assert np.allclose(
        chebyshev_to_monomial([0.0, 1.0], domain=(0.0, 2.0)),
        [-1.0, 1.0],
    )
    assert np.allclose(
        monomial_to_chebyshev([-1.0, 1.0], domain=(0.0, 2.0)),
        [0.0, 1.0],
    )


def test_rescaling_helpers_map_spectra_as_expected():
    A = np.diag([2.0, 4.0, 6.0])
    scaled = rescale_hermitian_to_unit_interval(A)
    assert np.allclose(np.linalg.eigvalsh(scaled.matrix), [-1.0, 0.0, 1.0])
    assert scaled.offset == 4.0
    assert scaled.scale == 2.0
    assert spectral_bounds(A) == (2.0, 6.0)

    cutoff_scaled = rescale_hermitian_about_cutoff(A, cutoff=3.0)
    assert np.max(np.abs(np.linalg.eigvalsh(cutoff_scaled.matrix))) <= 1.0
    assert np.linalg.eigvalsh(cutoff_scaled.matrix)[0] < 0.0

    psd_scaled = rescale_positive_semidefinite(A)
    assert np.allclose(np.linalg.eigvalsh(psd_scaled.matrix), [1 / 3, 2 / 3, 1.0])


def test_hamiltonian_constructors_return_hermitian_matrices():
    assert np.allclose(pauli_string_matrix("XZ").conj().T, pauli_string_matrix("XZ"))

    tb = tight_binding_chain(4, periodic=True)
    ising = ising_hamiltonian(3, coupling=0.8, transverse_field=0.4)
    heisenberg = heisenberg_chain(3)

    for matrix in [tb, ising, heisenberg]:
        assert matrix.shape[0] == matrix.shape[1]
        assert np.allclose(matrix, matrix.conj().T)


def test_hamiltonian_constructors_cover_validation_and_boundary_paths():
    assert np.allclose(
        tight_binding_chain(2, periodic=True), [[0.0, -1.0], [-1.0, 0.0]]
    )
    assert np.allclose(
        tight_binding_chain(2, onsite=[0.25, -0.5]),
        [[0.25, -1.0], [-1.0, -0.5]],
    )
    assert ising_hamiltonian(2, periodic=True).shape == (4, 4)
    assert heisenberg_chain(2, jx=0.5, jy=0.25, jz=0.125).shape == (4, 4)

    with pytest.raises(ValueError, match="nonempty"):
        pauli_string_matrix("")
    with pytest.raises(ValueError, match="I, X, Y, and Z"):
        pauli_string_matrix("XA")
    with pytest.raises(ValueError, match="n_sites must be positive"):
        tight_binding_chain(0)
    with pytest.raises(ValueError, match="onsite must have length"):
        tight_binding_chain(2, onsite=[0.0])
    with pytest.raises(ValueError, match="n_spins must be positive"):
        ising_hamiltonian(0)
    with pytest.raises(ValueError, match="greater than 1"):
        heisenberg_chain(1)


def test_pde_laplacian_constructors_return_expected_shapes():
    x, l1 = dirichlet_laplacian_1d(5)
    xp, lp = periodic_laplacian_1d(5)
    x2, y2, l2 = dirichlet_laplacian_2d(3, 4)

    assert x.shape == (5,)
    assert xp.shape == (5,)
    assert l1.shape == (5, 5)
    assert lp.shape == (5, 5)
    assert x2.shape == (3,)
    assert y2.shape == (4,)
    assert l2.shape == (12, 12)
    assert np.all(np.linalg.eigvalsh(l1) > 0.0)


def test_pde_laplacian_constructors_reject_invalid_domains():
    with pytest.raises(ValueError, match="n_points must be positive"):
        dirichlet_laplacian_1d(0)
    with pytest.raises(ValueError, match="length must be positive"):
        dirichlet_laplacian_1d(3, length=0.0)
    with pytest.raises(ValueError, match="greater than 2"):
        periodic_laplacian_1d(2)
    with pytest.raises(ValueError, match="length must be positive"):
        periodic_laplacian_1d(3, length=-1.0)
    with pytest.raises(ValueError, match="n_points must be positive"):
        dirichlet_laplacian_2d(0, 2)
    with pytest.raises(ValueError, match="length must be positive"):
        dirichlet_laplacian_2d(2, 2, ly=0.0)


def test_diagnostics_error_paths_and_complex_expectation_values():
    reference_state = np.array([1.0, 0.0])
    approximate_state = np.array([0.5, 0.5])
    assert relative_state_error(reference_state, approximate_state) > 0.0

    with pytest.raises(ValueError, match="reference state must be nonzero"):
        relative_state_error(np.zeros(2), approximate_state)

    reference_operator = np.eye(2)
    approximate_operator = np.diag([1.0, 2.0])
    assert operator_error(reference_operator, approximate_operator, relative=False) == (
        pytest.approx(1.0)
    )
    assert density_matrix_error(reference_operator, approximate_operator) > 0.0

    with pytest.raises(ValueError, match="reference operator must be nonzero"):
        operator_error(np.zeros((2, 2)), approximate_operator)

    value = expectation_value(
        np.array([[0.0, 1.0j], [1.0j, 0.0]]),
        np.array([1.0, 1.0]) / np.sqrt(2.0),
    )
    assert value == pytest.approx(1.0j)


def test_rescaling_helpers_cover_error_and_inverse_coordinate_paths():
    identity_matrix = np.eye(2)
    with pytest.raises(ValueError, match="nonzero spectral width"):
        rescale_hermitian_to_unit_interval(identity_matrix)
    with pytest.raises(ValueError, match="cutoff must not equal"):
        rescale_hermitian_about_cutoff(identity_matrix, cutoff=1.0)
    with pytest.raises(ValueError, match="positive semidefinite"):
        rescale_positive_semidefinite(np.diag([-0.1, 1.0]))
    with pytest.raises(ValueError, match="positive maximum"):
        rescale_positive_semidefinite(np.zeros((2, 2)))

    scaled = rescale_hermitian_about_cutoff(
        np.diag([0.0, 2.0]),
        cutoff=1.0,
        low_energy_positive=False,
    )
    assert scaled.original_from_scaled_value(np.array([-1.0, 1.0])).tolist() == [
        0.0,
        2.0,
    ]
    assert scaled.original_from_scaled_value(0.5) == pytest.approx(1.5)


def test_matrix_function_builders_match_simple_targets():
    A = np.diag([-1.0, -0.25, 0.5, 1.0])

    evo = design_real_time_evolution_polynomials(time=0.4, scale=2.0, degree=12)
    cos_A = apply_polynomial_to_hermitian(A, evo.cos_coeffs)
    assert np.allclose(np.diag(cos_A), np.cos(0.8 * np.diag(A)), atol=1e-8)

    imag = design_imaginary_time_polynomial(
        beta=0.7,
        scale=2.0,
        offset=1.5,
        degree=12,
    )
    values = imag.prefactor * eval_polynomial(imag.coeffs, np.diag(A))
    assert np.allclose(values, np.exp(-0.7 * (1.5 + 2.0 * np.diag(A))), atol=1e-8)

    real_coeffs, imag_coeffs = design_resolvent_polynomials(
        omega=0.2,
        eta=0.5,
        scale=1.0,
        degree=18,
    )
    xs = np.array([-0.5, 0.0, 0.5])
    denom = (0.2 - xs) ** 2 + 0.5**2
    assert np.max(np.abs(eval_polynomial(real_coeffs, xs) - (0.2 - xs) / denom)) < 1e-3
    assert np.max(np.abs(eval_polynomial(imag_coeffs, xs) - (-0.5 / denom))) < 1e-3

    window = design_gaussian_window_polynomial(center=0.1, width=0.3, degree=18)
    assert eval_polynomial(window, 0.1) > eval_polynomial(window, 0.9)


def test_positive_inverse_design_and_diagnostics():
    gamma = 0.2
    coeffs = design_positive_inverse_polynomial(gamma=gamma, degree=20)
    xs = np.linspace(gamma, 1.0, 50)
    assert np.max(np.abs(eval_polynomial(coeffs, xs) - gamma / xs)) < 0.08

    report = design_positive_inverse_diagnostics(gamma=gamma, degree=20)
    assert report["builder"] == "design_positive_inverse_polynomial"
    assert report["fit_domain"] == (gamma, 1.0)
    assert report["selected_extension"] in {"even", "flat"}


def test_positive_inverse_auto_is_no_worse_than_available_extensions():
    gamma = 0.055
    xs = np.linspace(gamma, 1.0, 100)
    target = gamma / xs

    auto = design_positive_inverse_polynomial(gamma=gamma, degree=40)
    even = design_positive_inverse_polynomial(
        gamma=gamma,
        degree=40,
        extension="even",
    )
    flat = design_positive_inverse_polynomial(
        gamma=gamma,
        degree=40,
        extension="flat",
    )

    auto_error = np.max(np.abs(eval_polynomial(auto, xs) - target))
    best_explicit_error = min(
        np.max(np.abs(eval_polynomial(even, xs) - target)),
        np.max(np.abs(eval_polynomial(flat, xs) - target)),
    )

    assert auto_error <= best_explicit_error + 1e-12


def test_linear_system_workflow_solves_positive_definite_problem():
    A = np.diag([1.0, 2.0])
    b = np.array([1.0, 1.0])

    result = linear_system_workflow(
        A,
        b,
        degree=20,
        num_points=501,
        bounded_num_points=1001,
        attempt_synthesis=False,
        apply_qsvt=True,
    )

    assert isinstance(result, LinearSystemWorkflowResult)
    assert result.gamma == 0.5
    assert np.allclose(result.classical_solution, [1.0, 0.5])
    assert result.polynomial_residual_norm < 0.05
    assert result.polynomial_relative_error < 0.05
    if result.qsvt_error is None:
        assert result.qsvt_solution is not None
        assert result.qsvt_residual_norm is not None
        assert result.qsvt_residual_norm < 0.12
    else:
        assert result.qsvt_solution is None
    assert result.compatibility["compatible"] is True
    assert result.as_report()["mode"] == "linear-system-workflow"


def test_linear_system_workflow_handles_scaled_identity_default_gamma():
    result = linear_system_workflow(
        2.0 * np.eye(2),
        np.array([1.0, -1.0]),
        degree=4,
        num_points=101,
        bounded_num_points=201,
        attempt_synthesis=False,
        apply_qsvt=False,
    )

    assert result.gamma < 1.0
    assert result.polynomial_residual_norm < 1e-6


def test_ground_state_filtering_workflow_improves_ground_overlap():
    H = np.diag([0.0, 1.0, 2.0])
    state = np.array([0.4, 0.8, 0.4])

    result = ground_state_filtering_workflow(
        H,
        state,
        degree=18,
        width=0.35,
        num_points=501,
    )

    assert isinstance(result, GroundStateFilteringWorkflowResult)
    assert result.ground_energy == 0.0
    assert result.ground_state_overlap > abs(state[0] / np.linalg.norm(state)) ** 2
    assert result.reference_state_error < 1e-3
    assert result.operator_relative_error < 1e-3
    assert result.as_report()["mode"] == "ground-state-filtering-workflow"


def test_hamiltonian_simulation_workflow_matches_exact_evolution():
    H = np.diag([-1.0, 0.25, 1.0])
    state = np.array([1.0, 1.0j, -0.5])

    result = hamiltonian_simulation_workflow(
        H,
        state,
        time=0.7,
        degree=16,
        num_points=501,
    )

    assert isinstance(result, HamiltonianSimulationWorkflowResult)
    assert result.state_relative_error < 1e-8
    assert result.operator_relative_error < 1e-8
    assert result.norm_drift < 1e-8
    assert result.as_report()["mode"] == "hamiltonian-simulation-workflow"


def test_resolvent_workflow_matches_exact_green_function_response():
    H = np.diag([-0.5, 0.2, 0.9])
    source = np.array([1.0, 0.0, -1.0])

    result = resolvent_workflow(
        H,
        omega=0.1,
        eta=0.4,
        degree=22,
        source=source,
        num_points=701,
    )

    assert isinstance(result, ResolventWorkflowResult)
    assert result.operator_relative_error < 5e-4
    assert result.response_relative_error is not None
    assert result.response_relative_error < 5e-4
    assert result.as_report()["mode"] == "resolvent-workflow"


def test_spectral_density_workflow_matches_gaussian_windows():
    H = np.diag([-1.0, -0.2, 0.6, 1.0])
    centers = np.array([-1.0, 0.0, 1.0])
    state = np.array([1.0, 0.5, 0.25, 0.0])

    result = spectral_density_workflow(
        H,
        centers,
        width=0.35,
        degree=22,
        state=state,
        num_points=701,
    )

    assert isinstance(result, SpectralDensityWorkflowResult)
    assert result.trace_density_error < 1e-3
    assert result.state_weight_error is not None
    assert result.state_weight_error < 1e-3
    assert result.polynomial_trace_density.shape == centers.shape
    assert result.as_report()["mode"] == "spectral-density-workflow"


def test_thermal_gibbs_workflow_matches_exact_boltzmann_weights():
    H = np.diag([0.0, 0.8, 1.5])
    state = np.array([0.5, 1.0, 0.25])

    result = thermal_gibbs_workflow(
        H,
        beta=0.9,
        degree=18,
        state=state,
        num_points=501,
    )

    assert isinstance(result, ThermalGibbsWorkflowResult)
    assert result.operator_relative_error < 1e-8
    assert result.density_matrix_relative_error < 1e-8
    assert result.weighted_state_error is not None
    assert result.weighted_state_error < 1e-8
    assert np.isclose(np.trace(result.polynomial_gibbs_state), 1.0)
    assert result.as_report()["mode"] == "thermal-gibbs-workflow"


def test_diagnostics_helpers():
    H = np.diag([0.0, 2.0])
    state = np.array([1.0, 0.0])
    other = np.array([0.8, 0.6])

    assert relative_state_error(state, state) == 0.0
    assert operator_error(H, H) == 0.0
    assert expectation_value(H, state) == 0.0
    assert ground_state_overlap(H, state) == 1.0
    assert np.allclose(spectral_weights(H, other), [0.64, 0.36])
