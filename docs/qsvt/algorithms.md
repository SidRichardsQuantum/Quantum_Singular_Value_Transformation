# Algorithm Notes

The high-level functions in `qsvt.algorithms` are simulator-scale workflows.
They combine spectral rescaling, bounded polynomial design, direct classical
matrix-function references, and diagnostics. They are intended to make the
algorithmic pattern inspectable, not to hide state preparation, block encoding,
or resource-estimation costs.

Each workflow returns a frozen dataclass with numerical fields and an
`as_report()` method. Reports can be passed through
`qsvt.reports.report_to_jsonable` or `qsvt.reports.save_report`.

## Common Pattern

Most Hermitian workflows follow the same structure:

1. validate a small dense Hermitian input,
2. rescale the spectrum to a polynomial design interval,
3. build one or more bounded polynomial approximations,
4. apply the polynomial classically through eigendecomposition,
5. compare against an exact spectral reference,
6. return diagnostics that quantify approximation error.

The QSVT interpretation is that a block encoding of the scaled operator would
allow the same polynomial transform to be implemented as a quantum signal
processing sequence, provided the polynomial satisfies the necessary
boundedness and parity constraints.

For focused theory by algorithm family, see [Linear systems](linear_systems.md),
[Spectral filters](spectral_filters.md), and
[Time evolution and response](time_evolution_and_response.md).

The singular-value workflows are the rectangular/non-Hermitian exception to
this Hermitian pattern. They use dense SVD validation: normalize singular
values, apply a polynomial to those singular values, and compare against a
dense SVD reference.

`block_encoded_qsvt_workflow` is the package's finite block-encoded exception
to the usual dense-polynomial-only pattern. It constructs an explicit dense
unitary block encoding, verifies the top-left block and unitarity numerically,
then applies the PennyLane QSVT matrix transform to the normalized positive
Hermitian signal operator. It is still a finite simulator workflow: scalable
oracle construction, state loading, readout, amplitude amplification, and
hardware costs remain outside the implementation.

## Truth Contract

Every `as_report()` payload from this module includes a `truth_contract` field.
That field is part of the public reporting surface and should be read before
using a workflow result in a benchmark, paper draft, or comparison table.

The contract states that the workflow is a dense spectral-polynomial
implementation, not an end-to-end quantum algorithm. It records:

- the concrete target being approximated,
- the implemented pieces: validation, spectral rescaling, polynomial
  construction/application, dense classical reference, and numerical
  diagnostics,
- whether a PennyLane QSVT block check was attempted for that run,
- the conditional QSVT statement: a compatible polynomial is the transform core
  once a valid block encoding and state-preparation model are supplied,
- omitted quantum layers such as block-encoding cost, state preparation,
  success-probability management, readout, amplitude amplification,
  fault-tolerant synthesis, and hardware compilation.

This makes the workflows useful for real physics/math studies: users can plug
in concrete finite models, measure approximation error and degree requirements,
and compare classical baselines, while keeping the missing quantum data-access
and hardware assumptions explicit.

## Linear Systems

`linear_system_workflow(matrix, rhs, degree=...)`

Purpose:
: Approximate the solution of a small positive-definite linear system
  `A x = b`.

Target function:
: For a positive scaled spectrum in `[gamma, 1]`, the workflow designs a
  bounded approximation to `gamma / x`. After applying the polynomial to the
  scaled matrix, the result is divided by `gamma * scale` to approximate
  `A^{-1} b`.

Notation:
: `A` is the positive-definite input matrix, `b` is the right-hand side,
  `x` is the scalar normalized spectral variable in the polynomial design,
  `gamma` is the lower bound of the scaled positive spectrum, `scale` is the
  positive factor used to normalize `A`, and `P` is the bounded inverse-like
  polynomial.

Rescaling:
: `rescale_positive_semidefinite` divides the matrix by its largest eigenvalue.
  If `gamma` is omitted, the scaled minimum eigenvalue is used.

Diagnostics:
: The result includes the classical solution, polynomial solution, residual
  norm, relative solution error, polynomial diagnostics, compatibility metadata,
  condition-number metadata, scaled spectral bounds, a linear-system
  resource-proxy block, and optional QSVT-applied solution fields.

Limitations:
: This is a dense simulator workflow. It does not include quantum state
  preparation, block-encoding construction, amplitude amplification, readout,
  fault-tolerant synthesis, or hardware costs. The resource-proxy fields expose
  degree, `gamma`, residuals, and conditioning; they are not quantum runtimes.

`linear_system_comparison_workflow(matrix, rhs, degree=...)`

Purpose:
: Compare the dense solve, optional conjugate-gradient baseline, QSVT-style
  polynomial inverse row, and optional PennyLane QSVT matrix check for the same
  small positive-definite instance.

Output:
: The result returns table-like rows with solver name, implementation kind,
  residual norm, relative solution error, conditioning metadata, and degree or
  iteration fields where applicable.

Interpretation:
: This is a numerical comparison helper, not a wall-clock benchmark. Use the
  benchmark module when timing classical baselines is the primary goal.

## Singular-Value Filtering and Pseudoinverses

`singular_value_filtering_workflow(matrix, cutoff=..., degree=...)`

Purpose:
: Apply a smooth filter to the singular values of a rectangular or
  non-Hermitian matrix.

Target function:
: The workflow normalizes singular values by the largest singular value and
  fits a smooth threshold
  `0.5 * (1 + tanh(sharpness * (sigma - cutoff)))` on `[0, 1]`.

Diagnostics:
: The result includes original and normalized singular values, polynomial and
  exact filtered matrices, operator relative error, and optional output-vector
  error when an input vector is supplied.

`singular_value_pseudoinverse_workflow(matrix, rhs, cutoff=..., degree=...)`

Purpose:
: Approximate a truncated SVD pseudoinverse action for inverse problems and
  least-squares systems.

Target function:
: For normalized singular values above `cutoff`, the workflow designs a
  bounded inverse-like polynomial and rescales it to approximate `1 / sigma`.
  Singular values below the cutoff are omitted from the dense reference
  pseudoinverse.

Diagnostics:
: The result includes the polynomial and truncated-SVD reference solutions,
  residual norms, solution relative error, pseudoinverse operator error,
  singular values, cutoff, scale, and coefficients.

Interpretation:
: These are dense SVD workflows for studying singular-value transformations,
  regularization, and noise amplification. A full QSVT algorithm would still
  need a block encoding of the rectangular operator, state preparation,
  success-probability management, and readout.

## Block-Encoded QSVT

`block_encoded_qsvt_workflow(matrix, coeffs, state=None, alpha=None)`

Purpose:
: Validate a finite block-encoded QSVT polynomial transform for a positive
  Hermitian matrix.

Target function:
: The coefficients define a bounded polynomial `P(x)` in the normalized signal
  coordinate. The workflow compares the QSVT logical block with the spectral
  reference `P(matrix / alpha)`.

Notation:
: `matrix` is the finite positive Hermitian logical operator, `alpha` is the
  block-encoding normalization, `x` is the normalized signal variable, and
  `P` is the polynomial represented by `coeffs`.

Block encoding:
: `qsvt.block_encoding.block_encode_matrix` constructs an explicit dense
  unitary dilation whose top-left block is `matrix / alpha`. If `alpha` is not
  supplied, the workflow chooses a conservative normalization so the signal
  spectrum stays away from the QSVT boundary.

Diagnostics:
: The result includes the dense unitary block encoding, verification metadata,
  QSVT operator, exact spectral reference, operator relative error, and optional
  state-vector output/error fields.

Limitations:
: This is a true finite block encoding and a true finite QSVT verification, but
  it is not a scalable oracle construction or an end-to-end quantum algorithm.
  The direct PennyLane comparison is scoped to positive-semidefinite Hermitian
  signal operators where this package's matrix-QSVT wrapper agrees with
  ordinary spectral polynomial functional calculus.

## Ground-State Filtering

`ground_state_filtering_workflow(matrix, state, degree=...)`

Purpose:
: Use a Gaussian low-energy window to filter a trial state toward the ground
  eigenspace.

Target function:
: A Gaussian window
  `exp(-0.5 * ((x - center) / width)^2)` on the scaled spectrum.

Rescaling:
: `rescale_hermitian_to_unit_interval` maps the Hamiltonian spectrum to
  `[-1, 1]`. The default center is near the low-energy edge.

Diagnostics:
: The result reports the normalized filtered state, unnormalized state,
  exact-reference filtered state, filtered energy, ground-state overlap,
  state error, operator error, and polynomial/reference operators.

Interpretation:
: High overlap indicates the filter emphasized the low-energy eigenspace for
  the chosen trial state. It does not prove efficiency for a large quantum
  system.

## Hamiltonian Simulation

`hamiltonian_simulation_workflow(matrix, state, time=..., degree=...)`

Purpose:
: Approximate real-time evolution `exp(-i H t)|psi>`.

Target functions:
: The workflow approximates cosine and sine components separately on the scaled
  spectrum, then reconstructs
  `exp(-i offset t) * (cos(scale * x * t) - i sin(scale * x * t))`.

Rescaling:
: `rescale_hermitian_to_unit_interval` supplies the affine offset and scale.

Diagnostics:
: The result includes cosine and sine coefficients, the polynomial unitary,
  exact unitary, evolved state, exact evolved state, state relative error,
  operator relative error, scaled time, and norm drift.

Limitations:
: The workflow validates polynomial matrix-function accuracy. It does not
  synthesize an optimized Hamiltonian-simulation circuit.

## Quantum Walk Search

`quantum_walk_search_workflow(adjacency, marked_vertex, degree=...)`

Purpose:
: Demonstrate continuous-time quantum walk search and amplitude amplification
  toward a marked vertex.

Search Hamiltonian:
: The workflow uses
  `H = -gamma * A - oracle_strength * |m><m|`, where `A` is the graph
  adjacency matrix and `m` is the marked vertex.

Implementation:
: The workflow samples exact dense spectral evolution over a time grid, selects
  the best marked-vertex probability, then fits polynomial cosine and sine
  phase components for the best sampled time in the scaled spectral coordinate.

Diagnostics:
: `QuantumWalkSearchWorkflowResult` stores the adjacency matrix, search
  Hamiltonian, marked vertex, hopping rate, time grid, marked-vertex
  probabilities, best exact state, polynomial best-time state, probability
  error, state error, operator error, and a resource proxy.

Limitations:
: The workflow demonstrates a finite dense graph instance and the
  polynomial-transform view of the search propagator. It does not implement
  scalable graph or marking oracles, state preparation, phase synthesis,
  amplitude estimation, or hardware execution.

## Fixed-Point Amplification

`fixed_point_amplification_workflow(score_operator, state, rounds=...)`

Purpose:
: Apply the monotone fixed-point polynomial `1 - (1 - x)^rounds` to a finite
  positive score operator with spectrum in `[0, 1]`.

Target function:
: The scalar polynomial boosts larger scores toward one while leaving low
  scores suppressed. This is useful for robust projector or high-score
  amplification demonstrations.

Diagnostics:
: The result reports the polynomial and exact reference operators, normalized
  amplified state, state error, operator error, initial score, amplified score,
  reference score, degree, and coefficients.

Interpretation:
: This is a spectral polynomial amplification primitive, not a full Grover
  iterate or amplitude-amplification circuit. The notebook or caller supplies
  the finite score/projector operator.

## Resolvents

`resolvent_workflow(matrix, omega=..., eta=..., degree=..., source=None)`

Purpose:
: Approximate a Green's-function / resolvent operator
  `(omega + i eta - H)^-1`.

Target function:
: The complex rational response is split into real and imaginary polynomial
  approximations after spectral rescaling.

Rescaling:
: The original physical `omega` and `eta` are translated into the scaled
  coordinate through the stored affine map.

Diagnostics:
: The result reports real and imaginary coefficients, polynomial and exact
  resolvent operators, operator relative error, and optional source-vector
  response/error fields.

Interpretation:
: Smaller `eta` creates sharper spectral features and generally needs higher
  degree. The workflow is useful for compact response-function demonstrations.

## Spectral Density

`spectral_density_workflow(matrix, centers, width=..., degree=..., state=None)`

Purpose:
: Estimate Gaussian-window trace density over selected spectral centers, with
  optional state-resolved spectral weights.

Target function:
: For each physical center `c`, the workflow designs a Gaussian window
  `exp(-0.5 * ((lambda - c) / width)^2)`.

Rescaling:
: Centers and width are converted from physical spectral coordinates to the
  scaled coordinate before polynomial design.

Diagnostics:
: The result includes one coefficient vector per center, polynomial trace
  density, exact trace density, trace-density error, and optional state weight
  arrays/errors.

Limitations:
: This is a small dense trace calculation. It does not implement stochastic
  trace estimation or large-scale density-of-states sampling.

## Spectral Counting

`spectral_counting_workflow(matrix, lower=..., upper=..., degree=...)`

Purpose:
: Estimate how many eigenvalues lie in a physical interval by tracing a smooth
  polynomial interval projector.

Target function:
: The workflow maps `[lower, upper]` into the scaled coordinate and applies
  `design_interval_projector_polynomial`.

Diagnostics:
: The result includes the polynomial projector, exact hard projector, exact
  count, polynomial trace count, count error, and optional Hutchinson-style
  stochastic trace estimate when `probe_count` is supplied.

Interpretation:
: This supports density-of-states, band-counting, Weyl-law, and graph-spectrum
  examples. The stochastic option is a finite dense diagnostic and not a
  scalable trace-estimation implementation.

## Spectral Thresholding

`spectral_thresholding_workflow(matrix, lower=..., upper=..., degree=...)`

Purpose:
: Approximate a hard spectral interval projector with a smooth bounded
  polynomial.

Target function:
: A smooth band-pass indicator for eigenvalues in `[lower, upper]`, expressed
  in the scaled spectral coordinate and implemented with
  `design_interval_projector_polynomial`.

Rescaling:
: `rescale_hermitian_to_unit_interval` maps the physical interval into
  `[-1, 1]` before polynomial design.

Diagnostics:
: The result includes the polynomial projector, exact hard spectral projector,
  exact rank, trace-based polynomial rank proxy, leakage outside the selected
  interval, operator error, design diagnostics, and optional state-retained
  weight comparison.

Interpretation:
: This workflow is useful for low-rank projection, band selection, and
  thresholding studies. Sharp interval edges require higher degree, and the
  reported leakage should be considered alongside any retained-rank proxy.

## Fermi-Dirac Occupations

`fermi_dirac_occupation_workflow(matrix, chemical_potential=..., beta=..., degree=...)`

Purpose:
: Approximate finite-temperature Fermi-Dirac occupation of a Hamiltonian.

Target function:
: `1 / (1 + exp(beta * (H - chemical_potential * I)))`, represented as a
  polynomial on the rescaled Hamiltonian spectrum.

Diagnostics:
: The result includes polynomial and exact occupation operators, particle
  numbers, operator relative error, coefficients, rescaling metadata, and
  optional state occupation/error fields.

Interpretation:
: This workflow is useful for electronic occupations, band filling, and
  chemical-potential sensitivity on small finite Hamiltonians. It does not
  implement quantum thermal state preparation or electronic-structure data
  loading.

## Thermal Gibbs Weighting

`thermal_gibbs_workflow(matrix, beta=..., degree=..., state=None)`

Purpose:
: Approximate imaginary-time / Boltzmann weighting and the normalized Gibbs
  density matrix.

Target function:
: `exp(-beta H)`, implemented as a scaled polynomial with a prefactor so the
  polynomial remains numerically bounded on the scaled interval.

Rescaling:
: `rescale_hermitian_to_unit_interval` gives the affine map used by
  `design_imaginary_time_polynomial`.

Diagnostics:
: The result includes the polynomial and exact Boltzmann operators, normalized
  Gibbs states, partition functions, operator relative error, density-matrix
  relative error, and optional weighted-state error.

Interpretation:
: This workflow is a matrix-function validation path for Gibbs-style weighting.
  It does not implement quantum thermal state preparation.

## Matrix Log and Entropy

`matrix_log_entropy_workflow(matrix, epsilon=..., degree=...)`

Purpose:
: Approximate a regularized matrix logarithm and the entropy-like spectral
  density `-x log(x + epsilon)` for positive semidefinite matrices.

Target function:
: The matrix is normalized to a positive spectrum in `[0, 1]`; the workflow
  fits separate polynomials for `log(x + epsilon)` and
  `-x log(x + epsilon)` in physical units.

Diagnostics:
: The result includes log and entropy coefficients, polynomial and exact log
  operators, polynomial and exact entropy operators, trace entropy values, and
  relative operator errors.

Interpretation:
: This supports covariance-spectrum, graph entropy, free-energy proxy, and
  regularized log-determinant examples. The `epsilon` parameter is part of the
  mathematical problem definition and should be reported with the degree and
  spectral scale.

## Choosing Degree

Degree controls the approximation space. Higher degree usually reduces error,
but the improvement can be non-monotonic for sampled maximum error, especially
near sharp transitions. Use:

- `design_workflow` for one polynomial with diagnostics,
- `design-sweep` for compact degree comparisons,
- notebook plots for visual inspection of transition regions,
- workflow regression tests for fixed small-matrix behavior.

The committed result artefacts under `results/reports/` and
`results/tables/design_sweep_summary.csv` show the release-time sweep pattern.
