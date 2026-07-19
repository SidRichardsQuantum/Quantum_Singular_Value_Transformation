# Roadmap

This project aims to support users who are learning, implementing,
experimenting with, or researching Quantum Singular Value Transformation
(QSVT). The package should connect the mathematical description of QSVT to
auditable polynomial design, phase synthesis, block encodings, finite circuit
execution, validation, and resource analysis.

The roadmap is organized by implementation area and describes the capabilities,
research infrastructure, and implementation upgrades the repository should
continue to pursue.

## Project Direction

- keep mathematical assumptions, numerical approximations, and omitted quantum
  layers explicit,
- provide useful entry points at several levels, from polynomial experiments to
  caller-supplied circuit components,
- keep algorithms and implementation helpers domain-general,
- validate quantum-facing results against appropriate finite classical
  references,
- distinguish executable circuits from analytical models and resource proxies,
- make examples thin, reproducible clients of tested package functionality,
- maintain a compact and discoverable public API without hiding lower-level
  research interfaces.

## Implementation Priorities

The sections below are ordered approximately by implementation dependency:
polynomial construction, block encoding, execution, verification, reporting,
and user-facing workflows.

### Completion and Stability Criteria

Before the project should be described as a stable educational and research
package, it should have an explicit definition of done:

- treat "complete" as a stable educational/research package milestone, not as
  a claim that QSVT research or all possible application algorithms are
  exhausted,
- publish a compact stable public API list with expected input shapes, report
  schemas, error semantics, and examples for each supported workflow,
- define the main supported user journeys across polynomial design,
  realizability checks, phase synthesis, block-encoding specifications, finite
  execution or proxy execution, validation, reporting, and resource estimation,
- document a deprecation policy and add migration tests for every public report
  schema or API marked stable,
- choose a small set of flagship workflows and finish them end to end: Python
  API, CLI command, classical reference, finite QSVT execution or explicitly
  labeled proxy path, machine-readable report, docs page, cookbook script, and
  regression tests,
- define acceptance criteria for each flagship workflow, including numerical
  tolerances, required diagnostics, supported access models, and the exact
  quantum layers that are implemented or omitted,
- require the supported CI gates and package preflight to pass from a clean
  checkout, including linting, formatting, typing, tests across supported Python
  environments, dependency compatibility, package build, documentation build,
  notebook checks, distribution metadata checks, and ordered publishing
  workflows,
- verify the package from the built wheel in a fresh environment, including CLI
  smoke tests and import/API-status checks,
- keep generated build outputs, coverage files, caches, virtual environments,
  and rendered documentation out of version control while preserving deliberate
  research artifacts under `results/`,
- add stronger adversarial and property-style tests for boundedness, parity,
  normalization, singular spectra, near-boundary inputs, wire layouts, and
  synthesis failures,
- document provider and hardware execution as either supported, experimental,
  or out of scope for each workflow, with paid/live execution behind explicit
  opt-in tests,
- ensure the README, usage guide, API reference, changelog, package metadata,
  and package-index description agree on stability status and the supported
  Python/dependency range,
- update the package stability classifier and public wording only when the
  stable API, flagship workflow acceptance criteria, package checks, and
  documentation consistency criteria are all satisfied.

### User-Facing Workflow Priorities

Prioritize a clear package path for users who bring a small physics or
mathematics problem and want an auditable QSVT workflow:

1. define the finite problem as a matrix, operator, or block-encoding access
   model,
2. choose a target transform such as inverse, projector, filter, resolvent,
   exponential, or singular-value function,
3. design and validate the bounded polynomial,
4. classify QSVT compatibility and synthesize phases where applicable,
5. execute a finite QSVT path when implemented, or emit an explicitly labeled
   polynomial/resource proxy,
6. compare against an appropriate classical reference,
7. export a machine-readable report with assumptions, errors, resources, and
   omitted quantum layers.

This path should be demonstrated by a small set of flagship workflows
rather than by adding many more notebooks:

- linear systems or Poisson-type PDE solves,
- ground-state, band, or spectral-projector filtering,
- Hamiltonian simulation or wave propagation,
- resolvents, Green's functions, or response functions,
- singular-value pseudoinverse, deblurring, or inverse problems.

Each flagship workflow should be callable from a short script, reusable from a
notebook, and backed by tests that verify the package API rather than only the
notebook narrative.

#### Current foundation

The repository already provides the following foundation:

- `qsvt.planning` accepts finite matrices, PennyLane operators, and
  `BlockEncodingSpec` inputs, searches degree from an error target, synthesizes
  phases with validated fallback, selects an access model, estimates logical
  resources, and can execute the resulting finite plan,
- `qsvt.flagship.spectral_filter_qsvt_workflow` provides the Pauli-LCU
  PrepSelPrep/qubitization spectral-filter path,
- `qsvt.flagship.poisson_qsvt_workflow` compares direct, conjugate-gradient,
  polynomial, and finite-QSVT paths for a Dirichlet Poisson system,
- `qsvt.resources.estimate_encoding_aware_resources` uses a concrete matrix or
  Pauli-LCU logical estimator model while preserving normalization and omitted
  costs,
- `qsvt.degree` and the phase-synthesis cache/adapter APIs provide reusable
  tolerance search and convention-safe optional solver integration.
- the accuracy-driven planning tutorial and encoding-aware resource benchmark
  exercise these APIs as thin package clients, while the Poisson and Ising
  notebooks carry executable circuit validation into concrete physics examples.
- focused cookbook scripts now cover accuracy-driven planning, caller-supplied
  custom block encodings and projectors, credential-free finite-shot FABLE
  validation, and access-model-aware resource comparison.
- `qsvt.research` now provides versioned JSON/YAML sweep definitions,
  deterministic trial identities, resumable per-trial reports, aggregate CSV
  summaries, and JSON manifests,
- `qsvt.research_frontier` provides the initial Poisson, Ising, and
  graph-Laplacian accuracy-resource study across inverse, projector,
  band-filter, and resolvent targets and four access models.

The remaining roadmap work is to broaden numerical regimes, access models,
hardware compilation, and application-specific state preparation without
weakening the explicit finite-simulation and logical-resource claim boundary.

### Polynomial Design, Realizability, and Phase Synthesis

- strengthen bounded polynomial design for inverse, sign, threshold, filter,
  exponential, resolvent, and other matrix-function targets,
- preserve explicit classification of classical-only, single-sequence, and
  mixed-parity constructions,
- improve numerical robustness for high-degree and poorly conditioned phase
  synthesis,
- compare supported phase solvers across degree, coefficient dynamic range,
  boundedness margin, convergence, runtime, and reconstruction error,
- implement adapters for modern stable phase-finding methods, including
  structured-factorization and fast fixed-point approaches, and benchmark them
  against the existing root-finding and iterative solvers,
- quantify how phase rounding, synthesis residuals, and coherent phase
  perturbations propagate into logical states, success probabilities, and
  application observables,
- provide actionable diagnostics when a polynomial is bounded but not
  realizable by the requested construction,
- continue developing certification and reconstruction checks that do not rely
  only on sampled grids,
- document phase conventions and conversion rules alongside every synthesis
  interface.

### Block Encodings and Signal Conventions

- maintain consistent interfaces across dense, sparse-like, rectangular,
  PennyLane-operator, and custom circuit access models,
- verify normalization, encoded blocks, signal subspaces, dimensions, and
  unitarity wherever finite verification is possible,
- add encoding-specific diagnostics and resource reports for embedding, FABLE,
  PrepSelPrep, qubitization, and custom encodings,
- benchmark FABLE compression thresholds against block-encoding error, compiled
  gates, depth, and final observable error for structured operator families,
- investigate Hamiltonian-access transformation paths that reduce or avoid
  conventional block-encoding overhead, while labeling their assumptions and
  execution support separately from established QSVT paths,
- make unsupported backend combinations fail with structured, informative
  reports,
- keep scalable oracle assumptions separate from finite matrix constructions,
- continue examples that compare multiple valid encodings of the same logical
  operator beyond the initial embedding, FABLE, PrepSelPrep, and qubitization
  resource sweep.

### Executable QSVT Interfaces

- maintain and extend lower-level execution from `BlockEncodingSpec` inputs,
  including caller-supplied block encodings and signal projectors,
- broaden backend coverage for the implemented dense, rectangular,
  PrepSelPrep, qubitization, and custom-circuit execution paths,
- prioritize decomposable, auditable QSVT execution paths over additional
  dense spectral proxy demonstrations,
- strengthen explicit wire, projector, normalization, and convention contracts
  for custom circuits,
- expand rectangular singular-value transformation beyond dense classical
  references and small embedding demonstrations,
- expose statevector and finite-shot execution with consistent validation and
  report schemas,
- add configurable coherent phase, depolarizing, amplitude-damping, and readout
  noise models to simulator execution,
- make the planner jointly choose polynomial degree, phase solver, access
  model, and shot budget under an approximation-and-noise error target.

### Hardware-Executable QSVT

- maintain the experimental `qsvt.hardware` layer for finite-shot execution on
  caller-supplied PennyLane devices, including caller-supplied preparation
  circuits, local preflight checks, probability measurements, shot-noise
  uncertainty, logical resource summaries, credential-free provider/fake-backend
  metadata capture, advertised native-operation checks, advertised shot-limit
  checks, non-executing logical/decomposed circuit audit reports, and explicit
  provider-omission metadata,
- document and test provider-plugin setup, beginning with
  `pennylane-qiskit`/`qiskit.remote`, including compatible dependency versions,
  credential configuration, backend selection, and a minimal finite-shot
  connectivity smoke test,
- keep provider credentials outside package configuration and reports, while
  recording non-sensitive provider, backend, and plugin-version metadata,
- extend hardware preflight beyond the current metadata-driven checks to cover
  provider shot-mode metadata, queue/submission constraints, and richer
  plugin-specific capability reports before any live backend is used,
- extend finite-shot hardware measurements beyond full-register probabilities
  to support observables, marginal probabilities, and postselected
  probabilities without depending on statevector access,
- target decomposable block encodings such as FABLE, PrepSelPrep, qubitization,
  and explicitly supplied hardware-compatible custom circuits,
- reject simulator-only constructions such as undecomposed `BlockEncode` when
  a target device cannot execute them,
- extend the current PennyLane decomposition audit into provider-native
  compilation to a requested or device-native gate set, with validation that no
  unsupported operations remain,
- extend logical and decomposed circuit reports with provider-compiled circuit
  exports so researchers can audit phase ordering, block-encoding calls, wire
  mapping, native gates, and compilation changes,
- report pre- and post-compilation depth, total gates, two-qubit gates, wire
  mapping, shots, device metadata, job metadata, and statistical uncertainty,
- support hardware job submission and result retrieval with persistent job
  identifiers, status reporting, bounded retries, timeouts, and cancellation,
  without embedding provider-specific objects in portable result schemas,
- require explicit shot and submission limits, expose provider cost estimates
  when available, and make paid execution an intentional opt-in action,
- preserve raw hardware results and record calibration timestamps, transpiler
  settings, and optional mitigation configuration; report mitigated results
  separately rather than replacing raw measurements,
- compare identical small circuits across an ideal simulator, a noisy
  simulator, and available real hardware,
- keep credential-free provider/fake-backend integration tests in the default
  suite, and add separately marked opt-in tests for live provider connectivity
  and hardware execution,
- add end-to-end hardware demonstrations beginning with one- or two-qubit
  logical Hamiltonians, low-degree polynomials, simple basis-state
  preparation, and narrowly defined measurements,
- define acceptance criteria for each supported hardware path, including
  successful native-gate compilation, finite-shot execution, reproducible
  metadata capture, and comparison against an ideal finite reference,
- keep hardware execution claims separate from scalability, fault tolerance,
  practical advantage, and complete application-algorithm claims.

### Verification and Numerical Reliability

- provide dense spectral and singular-value references for every finite
  executable workflow,
- report approximation, synthesis, block-encoding, state, observable, and
  sampling errors separately,
- add repeated seeded trials, confidence intervals, and bootstrap summaries for
  finite-shot and noisy experiments,
- add property-based and adversarial tests for parity, boundedness,
  normalization, wire layouts, degenerate spectra, and near-boundary inputs,
- strengthen compatibility tests across supported Python, NumPy, PennyLane,
  and optional solver versions,
- record tolerances, dependency versions, solver configuration, random seeds,
  and report-schema versions in generated research artifacts,
- maintain a report-schema registry, compatibility fixtures, and migration
  tests before changing versioned execution or benchmark report formats, so old
  committed reports remain loadable or fail with an intentional migration
  message,
- maintain regression cases for known synthesis and backend failure modes.

### Resource Estimation and Claim Boundaries

- report polynomial degree, phase count, signal-operator calls, encoding width,
  circuit depth, gate counts, shots, and postselection proxies where available,
- identify which costs come from polynomial transformation, block encoding,
  state preparation, amplitude amplification, readout, compilation, and
  hardware assumptions,
- compare encoding-specific resource profiles without presenting simulator
  timings as hardware runtime claims,
- distinguish machine-readable implementation layers:
  dense reference, verified finite block encoding, QNode execution, analytical
  resource proxy, and omitted quantum components,
- add scaling studies that state the access model and classical baseline
  required for a meaningful comparison,
- generate multi-objective Pareto fronts across accuracy, normalization,
  signal-operator calls, wires, compiled gates, depth, shots, and execution
  success probability.

### Public API and Package Architecture

- define a compact stable facade around design, synthesis, execution,
  verification, reporting, and resource estimation,
- keep the stable API intentionally small, with lower-level research
  interfaces available but not prematurely stabilized,
- keep experimental lower-level interfaces clearly labeled and accessible for
  research,
- keep algorithm and CLI implementations split by workflow family as the
  supported surface grows,
- use shared result and report types across Python, CLI, examples, and
  notebooks,
- keep type annotations, API-status metadata, and generated API documentation
  synchronized with exported behavior.

### Learning and Implementation Guides

- maintain a progressive path from scalar polynomial transforms through QSP,
  block encoding, QSVT circuits, and application workflows,
- include derivations and numerical checks for parity, boundedness,
  normalization, projector conventions, and phase conventions,
- add small implementation exercises whose expected outputs can be reproduced
  with package functions,
- show both successful constructions and representative failure cases,
- keep tutorials succinct and focused on reusable package interfaces rather
  than notebook-local implementations,
- link theory, code, tests, examples, and result artifacts so users can audit a
  workflow at the level they need.

### Research Workflows and Domain Examples

Core workflows should accept matrices, operators, spectral bounds, target
functions, tolerances, access models, and report options without depending on a
particular application domain.

Useful package-client examples include:

- linear systems, regularized pseudoinverses, image deblurring, tomography, and
  inverse problems,
- singular-value filtering, low-rank approximation, PCA, denoising, and model
  reduction,
- spectral projectors, ground-state filtering, band selection, and topological
  subspaces,
- Hamiltonian simulation, quantum walks, and wave propagation,
- Gibbs weights, imaginary-time evolution, and finite-temperature occupation
  functions,
- resolvents, Green's functions, response functions, and density-of-states
  estimation,
- PDE, graph-Laplacian, condensed-matter, quantum-chemistry, and
  matrix-function studies.

Each example should state the implemented finite workflow, approximation and
execution errors, parameter choices, access assumptions, resource model, and
omitted quantum layers. Reusable constructors, diagnostics, and plots should
move into `src/qsvt` when they are useful beyond one example.

Real examples should meet a consistent quality bar before new examples are
added: a clear problem statement, reusable package-level construction, a
classical reference, QSVT target definition, approximation and execution or
proxy errors, resource interpretation, and an explicit statement of whether the
workflow used true finite QSVT execution or a polynomial/resource proxy.

- prioritize physics studies based on narrowly defined observables rather than
  full-state reconstruction, beginning with Ising spectral filtering, Poisson
  inversion, and resolvent or Green's-function response,
- compare direct classical, polynomial, ideal-QSVT, finite-shot, and noisy-QSVT
  estimates for the same observable and problem instance,
- study how condition number, spectral gap, normalization, and state overlap
  affect observable accuracy and postselection or sampling cost.

### Benchmarks and Comparative Evaluation

- extend the declarative experiment runner with an evaluator registry,
  controlled parallel execution, environment provenance, aggregate confidence
  intervals, and standardized plot generation,
- broaden the initial accuracy-resource frontier to sparse structured and
  caller-supplied operators, compiled depth, executable success probability,
  finite-shot/noise studies, and application-level observables,
- compare embedding, FABLE, PrepSelPrep, qubitization, and supported custom or
  Hamiltonian-access constructions on the same logical operators,
- compare QSVT workflows with the relevant classical algorithm for the same
  finite task,
- use actual finite circuit execution when implemented and explicitly labeled
  resource proxies otherwise,
- include dense direct solves, iterative solvers, eigendecomposition, SVD,
  polynomial matrix evaluation, or domain-specific methods as appropriate,
- separate approximation quality, synthesis cost, circuit resources, sampling
  cost, and wall-clock timing,
- retain individual seeded trials as well as aggregate confidence intervals so
  sampling and noisy-circuit conclusions can be independently reproduced,
- treat benchmark timings as environment-specific snapshots rather than
  portable performance claims,
- provide machine-readable tables and reports for reproducible comparisons,
- generate standardized crossover, scaling, and Pareto-front plots from saved
  reports rather than notebook-local calculations,
- capture environment provenance, dependency metadata, random seeds, solver
  settings, compilation settings, and numerical tolerances in every research
  benchmark artifact.

## Repository and Documentation Policies

### Notebook Roles

- tutorials explain how QSVT concepts map to package interfaces,
- real examples apply general workflows to concrete physics and mathematics
  problems,
- benchmarks compare implementations and assumptions for a defined task.

Notebooks should construct the problem, call package helpers, and focus on
interpretation. Reusable algorithm, validation, reporting, and plotting logic
belongs in the package.

### Reproducible Artifacts

Committed notebook outputs and generated reports should remain reproducible
through explicit scripts such as `scripts/update_notebook_results.sh`.
Release checks should verify code, documentation, metadata, and artifact
structure without silently refreshing environment-dependent timing snapshots.

Reports should be JSON-safe, schema-versioned where appropriate, and explicit
about the software environment and numerical tolerances that produced them.

### Packaging and Distribution

PyPI artifacts should remain focused on the importable package and essential
project metadata. Full notebooks, rendered documentation, result snapshots,
benchmark artifacts, and regression tests belong in the repository and project
website as the auditable research record.
