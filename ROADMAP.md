# Roadmap

## Package Mission

This project helps users import, implement, validate, and study Quantum
Singular Value Transformation (QSVT), including its use on finite physics and
mathematics problems. The package connects the mathematical description of
QSVT to:

- bounded polynomial design and certification,
- realizability classification and phase synthesis,
- block-encoding specifications and verification,
- finite circuit construction and execution,
- classical validation and error diagnostics,
- QSVT-specific logical resource estimates,
- small, auditable application workflows.

The package should make mathematical assumptions, numerical approximations,
access models, and omitted quantum layers explicit. It should distinguish
executable QSVT circuits from polynomial studies and analytical resource
proxies.

## Scope Boundaries

### Core package

The core package owns reusable, domain-general QSVT primitives and a compact
public facade around design, synthesis, block encoding, execution, validation,
reporting, and resource estimation.

Classical matrix functions, eigendecompositions, singular-value references, and
iterative solvers belong in the package only when they validate or contextualize
a QSVT result. The project is not intended to become a general numerical
linear-algebra library.

### Physics and mathematics applications

PDEs, inverse problems, condensed-matter models, quantum chemistry, graph
problems, imaging, and data-analysis tasks are package-client examples. They
demonstrate how general QSVT interfaces apply to real problems; they do not
define domain libraries that the core package must maintain.

Code should move from an example into `src/qsvt` only when it is a reusable
QSVT primitive, diagnostic, or report component. Generic domain constructors,
classical solvers, plotting code, and application-specific analysis should stay
with the examples or repository documentation.

### Adjacent quantum algorithms

HHL and quantum-walk algorithms are experimental comparisons and tutorials, not
core QSVT implementations or stable-package milestones. They may be maintained
to explain algorithmic tradeoffs and validate comparisons, but they should not
have independent roadmap tracks that compete with QSVT work.

### Research infrastructure

Declarative sweeps, resumable trials, statistical aggregation, standardized
plots, and Pareto-front generation are repository research and benchmark
tooling. Existing experimental helpers may support the repository, but this
infrastructure is not part of the stable QSVT facade and should not grow into a
general experiment-management framework.

### Hardware and providers

Experimental hardware support is limited to finite-shot execution on a
caller-supplied PennyLane device, local preflight and decomposition checks, and
portable result and resource reports.

Provider accounts, credential handling, provider-specific plugin orchestration,
job persistence, submission queues, retries, cancellation, billing, and paid
execution management are outside the package scope. Users and provider plugins
own those responsibilities.

## Stable Educational and Research Milestone

The package may be described as stable for its stated educational and research
scope when:

- the compact stable API documents input shapes, error semantics, report
  schemas, examples, and deprecation guarantees,
- the three flagship workflows below satisfy versioned numerical and execution
  acceptance criteria,
- reports derive execution, realizability, and resource claims from artifacts
  produced by the run,
- adversarial and property-style tests cover boundedness, parity,
  normalization, singular spectra, wire layouts, near-boundary inputs, and
  synthesis failures,
- supported lint, formatting, typing, test, documentation, notebook,
  compatibility, build, and distribution checks pass from a clean checkout,
- the built wheel passes fresh-environment imports, CLI smoke tests, and API
  status checks,
- the README, usage guide, API reference, changelog, package metadata, and
  package-index description agree on the supported scope, stability, Python
  versions, and dependency ranges.

This milestone means that the supported package workflows are stable; it does
not imply that QSVT research or all possible applications are complete.

## Priorities

### Now — General executable QSVT core

Finite coherent mixed-parity execution landed in `0.2.21`. The immediate goal
is to harden that execution contract while improving phase synthesis and block
encoding support across numerical regimes and access models.

#### Mixed-parity execution hardening

- extend coherent component-LCU execution beyond the initial finite
  matrix-encoding path where backend decomposability permits,
- validate normalization, selector postselection, complex component weights,
  and circuit-resource ledgers across additional access models,
- preserve classification of classical-only, single-sequence, and mixed-parity
  constructions,
- reject or clearly diagnose bounded polynomials that are not realizable by the
  requested construction,
- add QSVT-valid full-domain extensions for targets currently designed only on
  positive singular values or spectra.

#### Phase synthesis and certification

- improve robustness for high degree, small boundedness margins, and poorly
  conditioned phase synthesis,
- add adapters for stable structured-factorization and fast fixed-point phase
  methods where they provide a maintained, convention-safe interface,
- compare supported phase solvers across degree, coefficient range,
  convergence, runtime, and reconstruction error,
- quantify how phase rounding, synthesis residuals, and coherent perturbations
  affect output states, success probabilities, and observables,
- continue certification and reconstruction checks that do not rely only on
  sampled grids,
- document phase conventions and conversions beside every synthesis interface.

#### Block encodings and signal conventions

- maintain consistent specifications for dense, sparse-like, rectangular,
  PennyLane-operator, Pauli-LCU, and caller-supplied circuit access models,
- verify normalization, encoded blocks, signal subspaces, dimensions, wire
  contracts, and unitarity wherever finite verification is possible,
- strengthen diagnostics and resource reports for embedding, FABLE,
  PrepSelPrep, qubitization, and custom encodings,
- keep scalable-oracle assumptions separate from finite matrix constructions,
- make unsupported backend and encoding combinations fail with structured,
  informative reports.

#### Evidence and numerical reliability

- derive each workflow's truth contract from the exact polynomial, synthesis
  result, access model, circuit, and execution artifacts returned by that run,
- use execution tiers such as `classical_reference`, `polynomial_core`,
  `qsvt_circuit`, and `hardware_execution` independently of scalability and
  resource completeness,
- record design and certification domains, normalization or prefactors,
  boundedness, parity, realizability, required combination mechanisms,
  synthesis status, QNode execution, and device execution where applicable,
- report approximation, synthesis, block-encoding, state, observable, and
  sampling errors separately,
- validate every finite executable workflow against dense spectral or
  singular-value references,
- maintain regression cases for known synthesis and backend failures,
- keep full statevectors and solution vectors as simulator validation data
  rather than presenting them as efficient quantum outputs.

### Next — Stabilize three complete flagship workflows

The three representative workflows now have finite accepted paths. Stabilize
their contracts and broaden their supported access models:

1. quantum linear systems through a Poisson-type finite problem,
2. spectral or ground-state filtering through a Pauli-LCU problem,
3. Hamiltonian simulation through a finite Hermitian problem.

Each flagship must provide:

- a short Python entry point, CLI command, cookbook script, notebook client, and
  focused documentation page,
- matrix, operator, or block-encoding problem input with explicit spectral and
  normalization assumptions,
- target-polynomial design, boundedness certification, realizability
  classification, and phase synthesis,
- finite QSVT execution, or a precisely labeled incomplete tier when a required
  mechanism is unavailable,
- an appropriate classical reference and application-level observable,
- component error and resource ledgers,
- a JSON-safe, schema-versioned report with acceptance results,
- regression tests with numerical tolerances and explicit supported access
  models.

#### Execution and resource interfaces

- extend lower-level execution from `BlockEncodingSpec` inputs, including
  caller-supplied block encodings and signal projectors,
- support statevector and finite-shot execution with consistent validation and
  report schemas,
- expand rectangular singular-value transformation beyond dense references and
  small embedding demonstrations,
- prioritize decomposable, auditable circuits over additional dense spectral
  proxy demonstrations,
- report polynomial degree, phase count, signal-operator calls, encoding width,
  gates, depth, wires, shots, and postselection or sampling costs where
  available,
- separate polynomial transformation, block encoding, state preparation,
  parity combination, amplitude amplification, readout, and compilation costs,
- mark resource reports as partial whenever a required layer is assumed,
  omitted, or lacks a concrete estimate,
- compare encoding-specific logical resources without presenting simulator
  timings as hardware runtime.

#### Package and release readiness

- keep the stable facade small while leaving lower-level research interfaces
  clearly experimental,
- use shared result and report types across Python, CLI, examples, and
  notebooks,
- keep type annotations, API-status metadata, report schemas, and generated API
  documentation synchronized,
- maintain compatibility fixtures and intentional migrations for stable report
  schemas,
- keep workflow-family implementation and CLI modules separated as the
  supported surface grows.

### Later / Experimental

Later work must not delay the general executable core or the three flagship
workflows.

#### Device execution

- broaden finite-shot measurements on caller-supplied PennyLane devices from
  full-register probabilities to observables, marginals, and postselected
  probabilities,
- audit local decomposition against device-advertised operations and reject
  undecomposed simulator-only constructions,
- export logical and locally decomposed circuit descriptions with wire mapping,
  gates, depth, shots, device metadata, and uncertainty,
- compare small circuits on ideal, caller-configured noisy, and caller-managed
  hardware devices,
- keep live-provider tests explicitly opt-in and outside default package
  validation.

The package will not manage provider credentials, provider-native job
lifecycles, submission costs, queues, retries, cancellation, calibration
records, or provider-specific mitigation.

#### Noise-aware planning

- consume noise behavior exposed by caller-configured PennyLane devices rather
  than implementing a general noise-model library,
- investigate joint selection of polynomial degree, phase solver, access model,
  and shot budget under an approximation-and-noise target,
- report repeated seeded trials and confidence intervals for finite-shot or
  noisy studies.

#### Broader application examples

After the three flagship workflows are complete, examples may cover:

- regularized pseudoinverses, deblurring, denoising, and inverse problems,
- band projectors, topological subspaces, and density-of-states estimation,
- resolvents, Green's functions, and response functions,
- wave propagation, Gibbs weights, imaginary-time transforms, and
  finite-temperature occupations,
- graph-Laplacian, condensed-matter, quantum-chemistry, and other
  matrix-function studies.

These remain thin clients of domain-general QSVT APIs. New examples should be
added only when they exercise a distinct QSVT construction, access model,
validation method, or observable.

## Repository and Documentation Policies

### Examples and notebooks

- tutorials progress from scalar transforms through QSP, block encoding, QSVT
  circuits, and application workflows,
- real examples construct a physics or mathematics problem, call package
  helpers, and focus on parameter choices and interpretation,
- benchmarks compare implementations and assumptions for a defined task,
- examples show both successful constructions and representative failure
  cases,
- reusable QSVT algorithm, validation, and reporting logic belongs in the
  package; domain and presentation logic stays in the client.

Reserve “QSVT implementation” for an executed or explicitly constructible QSVT
path. Use “classical polynomial surrogate” or “QSVT-compatible polynomial core”
for dense spectral and SVD studies.

### Research and benchmarks

- compare QSVT workflows with an appropriate classical algorithm for the same
  finite task,
- compare encodings on the same logical operator and state their access and
  normalization assumptions,
- separate approximation quality, synthesis cost, circuit resources, sampling
  cost, and environment-specific wall-clock timing,
- retain seeded trials and aggregate confidence intervals for finite-shot and
  noisy conclusions,
- generate tables and plots from saved reports rather than notebook-local
  calculations.

Research orchestration remains repository tooling and is not promoted through
`qsvt.stable`.

### Reproducible artifacts

Committed notebook outputs and generated reports should remain reproducible
through explicit scripts such as `scripts/update_notebook_results.sh`. Release
checks should verify code, documentation, metadata, and artifact structure
without silently refreshing environment-dependent timing snapshots.

Reports should be JSON-safe and schema-versioned where appropriate. Research
artifacts should record the software environment, dependencies, random seeds,
solver and compilation settings, and numerical tolerances needed to reproduce
them.

### Packaging and distribution

PyPI artifacts should remain focused on the importable QSVT package and
essential project metadata. Full notebooks, rendered documentation, result
snapshots, benchmark artifacts, research orchestration, and regression tests
belong in the repository and project website as the auditable research record.
