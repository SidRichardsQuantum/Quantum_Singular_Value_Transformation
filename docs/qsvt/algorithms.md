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

The workflows follow the same structure:

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

Rescaling:
: `rescale_positive_semidefinite` divides the matrix by its largest eigenvalue.
  If `gamma` is omitted, the scaled minimum eigenvalue is used.

Diagnostics:
: The result includes the classical solution, polynomial solution, residual
  norm, relative solution error, polynomial diagnostics, compatibility metadata,
  and optional QSVT-applied solution fields.

Limitations:
: This is a dense simulator workflow. It does not include quantum state
  preparation, block-encoding construction, amplitude amplification, condition
  number resource estimates, or fault-tolerant costs.

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
