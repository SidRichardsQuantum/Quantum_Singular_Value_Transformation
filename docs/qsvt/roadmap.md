# Roadmap

This project should be useful to users who want to learn, prototype, and audit
QSVT-style workflows without having to read every notebook first.

The package direction is:

- keep algorithms and implementation helpers general,
- keep numerical assumptions and omitted quantum layers explicit,
- make real physics and mathematics examples thin clients of the package,
- keep tutorial notebooks succinct and focused on how to use the package
  algorithms and implementations,
- use real-example notebooks to demonstrate domain workflows, not to hide
  reusable logic,
- benchmark quantum/QSVT algorithms and implementations against the relevant
  classical algorithms and implementations,
- keep artifacts reproducible and clearly separated from release validation.

## Near-Term Priorities

### 1. User-Facing Workflow Polish

The next package work should make the common user paths easier to discover and
reuse:

- tighten the public API around polynomial design, compatibility checks,
  block-encoded workflows, circuit execution, reports, and benchmarks,
- maintain small cookbook examples for "choose a polynomial, apply it, inspect
  the diagnostics, save the report" in `examples/`,
- keep tutorial notebooks short enough that users can copy the package usage
  pattern without reading a paper-length example,
- keep CLI and Python examples aligned so users can move between scripts,
  notebooks, and shell commands without changing concepts.

### 2. Notebook Roles

Notebook families should have distinct jobs:

- tutorials: succinct package-client walkthroughs for using the available
  algorithms, implementations, diagnostics, reports, and CLI commands,
- real examples: near-pure package-client applications to concrete physics and
  mathematics problems,
- benchmarks: comparisons between quantum/QSVT algorithms or implementations
  and the relevant classical algorithms or implementations for the same task.

Benchmarks should use actual finite PennyLane/QSVT execution paths where that
is implemented and clearly marked QSVT resource proxies where full execution is
not the measured object. Classical comparisons should be the appropriate
baseline for the task: dense direct solve, conjugate gradient, dense spectral
matrix functions, polynomial matrix evaluation, or other domain-relevant
classical methods.

### 3. General Algorithms, Specific Client Notebooks

Core algorithms should stay domain-general. A workflow should accept matrices,
operators, spectral bounds, target functions, tolerances, and report options
without knowing whether the caller is studying a PDE, lattice model,
Hamiltonian, Green's function, thermal state, or linear system.

Real-world physics and mathematics notebooks should become near-pure package
clients:

- construct or load the domain operator,
- call package helpers for rescaling, polynomial design, validation, and
  reporting,
- keep only domain explanation, parameter choices, and interpretation in the
  notebook,
- move reusable construction, diagnostics, and plotting logic back into
  `src/qsvt` when a second notebook needs it.

### 4. Package-Client Problem Targets

Good real-world physics and mathematics notebooks should be thin clients of
the package: they should build or load a domain operator, call one or two
general workflow functions, and spend the remaining notebook space on
interpretation. Useful targets include:

- image deblurring, tomography, and inverse scattering with
  `singular_value_pseudoinverse_workflow`, using singular-value cutoffs to
  study regularization, residuals, and noise amplification,
- low-rank denoising, PCA, model reduction, and compressed sensing diagnostics
  with `singular_value_filtering_workflow`, comparing retained singular modes
  against dense SVD references,
- Poisson, Helmholtz, diffusion, and graph-Laplacian linear systems with
  `linear_system_workflow` or `linear_system_comparison_workflow`, where the
  notebook supplies the finite-difference or graph operator,
- quantum chemistry and condensed-matter occupation problems with
  `fermi_dirac_occupation_workflow`, estimating finite-temperature particle
  number, band filling, and chemical-potential sensitivity,
- Gibbs weights, imaginary-time filters, and thermal density matrices with
  `thermal_gibbs_workflow`, including small spin chains, lattice Hamiltonians,
  and molecular toy models,
- Green's functions, spectral response, local density of states, and transport
  response with `resolvent_workflow`, using source vectors to probe spatial or
  orbital response channels,
- density of states, band-gap detection, Weyl-law checks, and graph spectral
  statistics with `spectral_density_workflow` and
  `spectral_counting_workflow`,
- topological band projectors, edge-state selection, photonic band gaps, and
  spectral subspace isolation with `spectral_thresholding_workflow`,
- entropy, free-energy proxies, covariance spectra, graph von Neumann entropy,
  and regularized log-determinant studies with `matrix_log_entropy_workflow`,
- real-time Schrödinger dynamics, wave propagation, coupled oscillator motion,
  and continuous-time quantum walks with `hamiltonian_simulation_workflow` or
  `quantum_walk_search_workflow`,
- robust marked-subspace or high-score amplification demonstrations with
  `fixed_point_amplification_workflow`, where the notebook supplies a finite
  score/projector operator.

Each notebook should report the implemented finite workflow, approximation
error, degree or cutoff choices, and the omitted quantum layers. If a notebook
needs reusable domain constructors, plotting helpers, or diagnostics that would
benefit a second example, that logic should move into `src/qsvt` before the
notebook grows around it.

### 5. Claim-Boundary and Resource Modeling

The package should continue to separate implemented finite workflows from
conditional quantum-algorithm interpretation. Reports should make these layers
machine-readable:

- dense spectral reference,
- finite block encoding,
- PennyLane QNode circuit execution,
- QSVT resource proxy,
- omitted state-preparation, oracle, readout, synthesis, and hardware costs.

Future work should improve resource proxies without turning them into hardware
runtime claims.

### 6. Reproducible Artifacts Without Release Churn

Release checks should verify code, docs, package metadata, and committed
artifact structure without rewriting timing snapshots. Artifact regeneration
belongs to explicit notebook/result workflows such as
`scripts/update_notebook_results.sh`.

Benchmark timing fields should be treated as environment-specific snapshots.
When they are refreshed, the commit should say so directly.

### 7. Packaging and Distribution

PyPI artifacts should stay focused on the importable package and essential
project metadata. The full notebooks, rendered documentation, result snapshots,
and regression tests belong in the GitHub repository and project website.

This keeps installation lightweight while preserving the repository as the
auditable source for examples and reproducible outputs.

## Later Directions

- expand and enforce stable/experimental API labels before a `1.0` release,
- more package-level plotting/report helpers shared by real-example notebooks,
- stronger compatibility tests across supported PennyLane, NumPy, and Python
  versions,
- optional adapters for user-provided sparse matrices or domain operators,
- more examples where one general package workflow solves several concrete
  physics or mathematics instances.
