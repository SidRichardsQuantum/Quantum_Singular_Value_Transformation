# Roadmap

This project aims to support users who are learning, implementing,
experimenting with, or researching Quantum Singular Value Transformation
(QSVT). The package should connect the mathematical description of QSVT to
auditable polynomial design, phase synthesis, block encodings, finite circuit
execution, validation, and resource analysis.

The roadmap is organized by implementation area rather than release number.
Release versions describe shipped snapshots; this document describes the
capabilities and upgrades the repository should continue to pursue.

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

### Polynomial Design, Realizability, and Phase Synthesis

- strengthen bounded polynomial design for inverse, sign, threshold, filter,
  exponential, resolvent, and other matrix-function targets,
- preserve explicit classification of classical-only, single-sequence, and
  mixed-parity constructions,
- improve numerical robustness for high-degree and poorly conditioned phase
  synthesis,
- compare supported phase solvers across degree, coefficient dynamic range,
  boundedness margin, convergence, runtime, and reconstruction error,
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
- make unsupported backend combinations fail with structured, informative
  reports,
- keep scalable oracle assumptions separate from finite matrix constructions,
- add examples that compare multiple valid encodings of the same logical
  operator.

### Executable QSVT Interfaces

- maintain and extend lower-level execution from `BlockEncodingSpec` inputs,
  including caller-supplied block encodings and signal projectors,
- broaden backend coverage for the implemented dense, rectangular,
  PrepSelPrep, qubitization, and custom-circuit execution paths,
- strengthen explicit wire, projector, normalization, and convention contracts
  for custom circuits,
- expand rectangular singular-value transformation beyond dense classical
  references and small embedding demonstrations,
- expose statevector and finite-shot execution with consistent validation and
  report schemas.

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
- add property-based and adversarial tests for parity, boundedness,
  normalization, wire layouts, degenerate spectra, and near-boundary inputs,
- strengthen compatibility tests across supported Python, NumPy, PennyLane,
  and optional solver versions,
- record tolerances, dependency versions, solver configuration, random seeds,
  and report-schema versions in generated research artifacts,
- add report-schema compatibility fixtures and migration tests before changing
  versioned execution or benchmark report formats,
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
  required for a meaningful comparison.

### Public API and Package Architecture

- define a compact stable facade around design, synthesis, execution,
  verification, reporting, and resource estimation,
- keep experimental lower-level interfaces clearly labeled and accessible for
  research,
- split large algorithm and CLI modules by workflow family,
- use shared result and report types across Python, CLI, examples, and
  notebooks,
- establish a documented deprecation policy and migration tests before a
  stable major API,
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

### Benchmarks and Comparative Evaluation

- compare QSVT workflows with the relevant classical algorithm for the same
  finite task,
- use actual finite circuit execution when implemented and explicitly labeled
  resource proxies otherwise,
- include dense direct solves, iterative solvers, eigendecomposition, SVD,
  polynomial matrix evaluation, or domain-specific methods as appropriate,
- separate approximation quality, synthesis cost, circuit resources, sampling
  cost, and wall-clock timing,
- treat benchmark timings as environment-specific snapshots rather than
  portable performance claims,
- provide machine-readable tables and reports for reproducible comparisons.

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
