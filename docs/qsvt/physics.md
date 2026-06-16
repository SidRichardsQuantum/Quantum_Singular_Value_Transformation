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
singular-value inverse problems, photonic band gaps, graphene nanoribbon
density of states, and tensor-network hybrid filtering.

For concise theory and diagnostics notes on the high-level workflow functions,
see [Algorithm notes](algorithms.md). For coefficient conventions, rescaling,
boundedness checks, and report serialization choices, see
[Implementation notes](implementation.md).

Focused theory notes for the main physics algorithm families are also
available:

- [Linear systems](linear_systems.md) covers inverse-polynomial workflows,
  solver comparisons, and finite HHL circuit execution.
- [Spectral filters](spectral_filters.md) covers ground-state filtering,
  interval projectors, sign/threshold filters, and spectral density windows.
- [Time evolution and response](time_evolution_and_response.md) covers
  Hamiltonian simulation, resolvents, imaginary-time evolution, and Gibbs
  weighting.

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

## Linear-System Workflow

`qsvt.algorithms.linear_system_workflow` combines the positive-inverse design,
positive-semidefinite rescaling, classical validation, optional PennyLane QSVT
application, residual diagnostics, and compatibility metadata for small
positive-definite systems. Its report includes scaled spectral bounds,
2-norm condition-number metadata, a `gamma` condition proxy, and an explicit
resource-proxy block that lists omitted quantum layers such as state
preparation, block encoding, success-probability management, and readout.

```python
import numpy as np
from qsvt.algorithms import linear_system_workflow

A = np.diag([1.0, 2.0])
b = np.array([1.0, 1.0])

result = linear_system_workflow(
    A,
    b,
    degree=20,
    attempt_synthesis=False,
)

print(result.polynomial_solution)
print(result.polynomial_residual_norm)
```

For solver-oriented comparisons, use
`qsvt.algorithms.linear_system_comparison_workflow`. It returns rows for the
dense reference solve, optional conjugate gradients, the QSVT-style polynomial
inverse, and the optional PennyLane QSVT matrix check. The comparison is a
finite numerical diagnostic, not a quantum runtime benchmark.

The workflow is intended for educational and simulator-scale experiments. It
does not replace production state preparation, block-encoding construction, or
fault-tolerant resource estimation.

See [Linear systems](linear_systems.md) for the inverse-polynomial and HHL
theory behind these workflows.

## Algorithm Workflows

`qsvt.algorithms` also provides end-to-end workflows for common small physics
tasks. These combine rescaling, polynomial design, spectral reference
calculations, and diagnostics:

- `ground_state_filtering_workflow(matrix, state, degree=...)`
- `fermi_dirac_occupation_workflow(matrix, chemical_potential=..., beta=..., degree=...)`
- `hamiltonian_simulation_workflow(matrix, state, time=..., degree=...)`
- `singular_value_pseudoinverse_workflow(matrix, rhs, cutoff=..., degree=...)`
- `resolvent_workflow(matrix, omega=..., eta=..., degree=..., source=None)`
- `spectral_density_workflow(matrix, centers, width=..., degree=..., state=None)`
- `thermal_gibbs_workflow(matrix, beta=..., degree=..., state=None)`

```python
import numpy as np
from qsvt.algorithms import hamiltonian_simulation_workflow

H = np.diag([-1.0, 0.0, 1.0])
psi = np.array([1.0, 1.0j, 0.5])

result = hamiltonian_simulation_workflow(
    H,
    psi,
    time=0.75,
    degree=18,
)

print(result.state_relative_error)
print(result.norm_drift)
```

Each workflow result exposes `as_report()` for JSON-safe conversion through
`qsvt.reports.report_to_jsonable`.

See [Spectral filters](spectral_filters.md) for filtering and projector
workflows, and [Time evolution and response](time_evolution_and_response.md)
for Hamiltonian simulation, resolvents, and Gibbs weighting.

For concrete package-client examples, see
`notebooks/real_examples/25_fermi_dirac_electronic_occupations.ipynb` for
finite-temperature electronic occupations and
`notebooks/real_examples/29_singular_value_pseudoinverse_deblurring.ipynb` for
a regularized inverse problem driven by singular-value pseudoinverse
polynomials, and
`notebooks/real_examples/30_matrix_log_entropy_graph_laplacian.ipynb` for a
regularized graph-spectrum entropy diagnostic.

## Interval Projectors

`qsvt.design.design_interval_projector_polynomial` builds a bounded smooth
band-pass polynomial for a target interval inside `[-1, 1]`.

```python
from qsvt.design import design_interval_projector_polynomial

coeffs = design_interval_projector_polynomial(
    lower=-0.2,
    upper=0.4,
    degree=30,
    sharpness=18.0,
)
```

This is useful for spectral windows, density-of-states demos, band filtering,
and projector-style workflows where both interval edges matter.

See [Spectral filters](spectral_filters.md) for the related filter and
projector theory.
