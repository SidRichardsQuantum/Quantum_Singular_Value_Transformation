# Time Evolution and Response

Time-evolution, response, and thermal workflows all use the same matrix-function
idea: approximate a target scalar function on the spectrum of a rescaled
Hermitian operator, then apply the corresponding polynomial to the matrix.

For a Hermitian operator

$$
H = V \operatorname{diag}(\lambda_i) V^\dagger,
$$

the exact reference is

$$
f(H) = V \operatorname{diag}(f(\lambda_i)) V^\dagger.
$$

QSVT supplies a route to implement compatible polynomial approximations to
these scalar functions once a block encoding of the normalized operator is
available. The package validates the polynomial matrix functions on small dense
instances.

Concise per-workflow pages:

- [Hamiltonian simulation workflow](workflow_hamiltonian_simulation.md)
- [Quantum walk search workflow](workflow_quantum_walk_search.md)
- [Resolvent workflow](workflow_resolvent.md)
- [Thermal Gibbs workflow](workflow_thermal_gibbs.md)
- [Fermi-Dirac occupation workflow](workflow_fermi_dirac.md)

## Real-time evolution

`qsvt.algorithms.hamiltonian_simulation_workflow` targets

$$
e^{-iHt}|\psi\rangle.
$$

After affine rescaling \(H = \mathrm{offset}\,I + \mathrm{scale}\,X\), the
phase splits as

$$
e^{-iHt}
= e^{-i\,\mathrm{offset}\,t}
\left(\cos(\mathrm{scale}\,t\,X)
- i\sin(\mathrm{scale}\,t\,X)\right).
$$

The workflow approximates cosine and sine with separate bounded polynomials,
then reconstructs the complex evolution. Diagnostics include state error,
operator error, scaled time, and norm drift.

This is a polynomial matrix-function validation path. It is not an optimized
Hamiltonian-simulation circuit synthesis routine.

See [Hamiltonian simulation workflow](workflow_hamiltonian_simulation.md).

## Resolvents and Green's functions

`qsvt.algorithms.resolvent_workflow` targets the shifted inverse

$$
G(\omega, \eta) = (\omega + i\eta - H)^{-1}.
$$

The small positive broadening \(\eta\) keeps the response finite and controls
spectral resolution. Smaller \(\eta\) creates sharper peaks near eigenvalues,
which generally requires higher polynomial degree.

The complex response is split into real and imaginary scalar functions in the
scaled coordinate. The workflow reports polynomial and exact resolvent
operators, operator relative error, and optional source-vector response error.

See [Resolvent workflow](workflow_resolvent.md).

## Imaginary-time and Gibbs weighting

`qsvt.algorithms.thermal_gibbs_workflow` approximates Boltzmann weighting:

$$
B = e^{-\beta H}.
$$

After polynomial approximation, the workflow forms the normalized Gibbs density
matrix

$$
\rho = \frac{B}{\operatorname{Tr}(B)}.
$$

The polynomial is scaled so the represented function remains numerically
bounded on the design interval. Diagnostics include the partition function,
operator error, density-matrix error, and optional weighted-state error.

This validates Gibbs-style matrix functions on finite dense systems. It does
not implement a quantum thermal-state preparation algorithm.

See [Thermal Gibbs workflow](workflow_thermal_gibbs.md).

## Spectral response patterns

These workflows share a few practical constraints:

- large evolution time increases oscillation and usually requires higher
  degree,
- small resolvent broadening \(\eta\) increases spectral sharpness,
- large inverse temperature \(\beta\) makes Gibbs weights more selective,
- rescaling metadata is essential because physical parameters must be converted
  into the normalized polynomial coordinate,
- boundedness and parity matter if the polynomial is intended for QSVT
  synthesis.

## Diagnostics

The relevant diagnostics depend on the target:

- time evolution: state relative error, operator relative error, and norm
  drift,
- response functions: operator error and source-vector response error,
- Gibbs weighting: partition function, density-matrix error, and optional
  weighted-state error.

Use these metrics as finite-instance validation of the polynomial transform.
They are not hardware runtime estimates.

## Scope

The package computes dense spectral references and polynomial approximations
for small inspectable systems. A full quantum algorithm would additionally
need a scalable block encoding or Hamiltonian simulation oracle, state
preparation, success-probability management where relevant, readout strategy,
fault-tolerant synthesis, and hardware compilation.
