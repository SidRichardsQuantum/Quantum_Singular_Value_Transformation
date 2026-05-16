# Physics Workflows

The physics-facing APIs provide general building blocks for small, executable
spectral problems. They are intentionally not problem-specific solvers. The
recommended pattern is:

1. build a Hamiltonian or PDE operator,
2. rescale its spectrum to a QSVT-compatible interval,
3. design a bounded polynomial matrix function,
4. apply it classically for validation or pass the coefficients to QSVT tools.

The real-example notebooks in `notebooks/real_examples/` use this pattern for
Hamiltonian simulation, ground-state filtering, quantum chemistry, Green's
functions, spectral density estimation, Gibbs states, PDE linear systems,
transport physics, spin-chain phase diagnostics, electronic occupations,
photonic band gaps, graphene nanoribbon density of states, and tensor-network
hybrid filtering.

## Hamiltonians

`qsvt.hamiltonians` contains reusable small physics operators:

- `tight_binding_chain(n_sites, hopping=1.0, onsite=None, periodic=False)`
- `ising_hamiltonian(n_spins, coupling=1.0, transverse_field=1.0, periodic=False)`
- `heisenberg_chain(n_spins, jx=1.0, jy=1.0, jz=1.0, periodic=False)`
- `pauli_string_matrix(pauli_string)`

```python
from qsvt.hamiltonians import tight_binding_chain
from qsvt.spectral import eigh_hermitian

H = tight_binding_chain(8, hopping=1.0)
energies, modes = eigh_hermitian(H)
```

## PDE Operators

`qsvt.pde` provides finite-difference operators for linear PDE examples:

- `dirichlet_laplacian_1d(n_points, length=1.0)`
- `periodic_laplacian_1d(n_points, length=1.0)`
- `dirichlet_laplacian_2d(nx, ny, lx=1.0, ly=1.0)`

```python
from qsvt.pde import dirichlet_laplacian_1d

x, L = dirichlet_laplacian_1d(24)
```

## Spectral Rescaling

`qsvt.rescaling` centralizes common spectral normalizations:

- `spectral_bounds(matrix)`
- `rescale_hermitian_to_unit_interval(matrix)`
- `rescale_hermitian_about_cutoff(matrix, cutoff, low_energy_positive=True)`
- `rescale_positive_semidefinite(matrix)`

Each rescaling helper returns a `ScaledOperator` with the scaled matrix, affine
metadata, and original eigenvalue bounds.

```python
from qsvt.rescaling import rescale_hermitian_to_unit_interval

scaled = rescale_hermitian_to_unit_interval(H)
A = scaled.matrix
```

## Matrix-Function Polynomials

`qsvt.matrix_functions` provides general polynomial builders for common physics
matrix functions:

- `design_real_time_evolution_polynomials(time, scale, degree=...)`
- `design_imaginary_time_polynomial(beta, scale, offset=0.0, degree=...)`
- `design_resolvent_polynomials(omega, eta, scale, offset=0.0, degree=...)`
- `design_gaussian_window_polynomial(center, width, degree=...)`
- `design_low_energy_projector_polynomial(gap, degree=...)`
- `design_positive_inverse_matrix_polynomial(gamma, degree=...)`

```python
from qsvt.matrix_functions import design_real_time_evolution_polynomials
from qsvt.spectral import apply_polynomial_to_hermitian

polys = design_real_time_evolution_polynomials(
    time=1.0,
    scale=scaled.scale,
    degree=19,
)

cos_Ht = apply_polynomial_to_hermitian(scaled.matrix, polys.cos_coeffs)
sin_Ht = apply_polynomial_to_hermitian(scaled.matrix, polys.sin_coeffs)
```

## Diagnostics

`qsvt.diagnostics` provides small reusable metrics:

- `relative_state_error(reference, approximate)`
- `operator_error(reference, approximate, relative=True)`
- `expectation_value(operator, state)`
- `ground_state_overlap(hamiltonian, state)`
- `spectral_weights(operator, state)`
- `density_matrix_error(reference, approximate)`

## Positive Inverse Design

`qsvt.design.design_positive_inverse_polynomial` targets positive definite
operators rescaled to spectra in `[gamma, 1]`. Its default `extension="auto"`
tries bounded full-interval extensions and selects the lower sampled error on
the positive design interval.

```python
from qsvt.design import design_positive_inverse_polynomial

coeffs = design_positive_inverse_polynomial(gamma=0.1, degree=30)
```

For harder linear systems, small `gamma` means large condition number and a
higher degree is usually required.
