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

Selected applications are also acceptance clients for the package. They should
drive reusable requirements back into the core and prove that the same public
interfaces cover polynomial design, synthesis, block encoding, execution,
validation, and reporting for a complete finite problem.

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

### Experiment Studio

The Studio is a local, single-user repository client of the public package
APIs. It owns configuration, saved-run browsing, comparison, and visualization
of package reports. Numerical algorithms, acceptance criteria, and scientific
claims remain owned by the package. The Studio is not shipped in the package
distributions and does not introduce hosted services or provider management.

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

### Now — Harden reusable QSVT implementations through flagship clients

Finite coherent mixed-parity execution landed in `0.2.21`; `0.2.22` broadened
its tested contract to FABLE, PrepSelPrep, qubitization, and caller-supplied
projector conventions. All three flagship workflows now have finite executable
acceptance paths, and the repository Studio exposes each through public package
APIs. The immediate goal is to harden synthesis, access-model support, and
execution/report contracts across those workflows.

#### Shared implementation and execution interfaces

- make `BlockEncodingSpec` the common lower-level input for the three flagship
  workflows, including caller-supplied block encodings, signal projectors, and
  explicit wire contracts,
- publish a tested support matrix for embedding, FABLE, PrepSelPrep,
  qubitization, and custom circuits, distinguishing representation, logical
  resource estimation, simulator execution, and backend decomposition,
- support statevector and finite-shot execution through consistent validation,
  result, error, and resource schemas,
- extend coherent component-LCU execution to additional signal conventions and
  rectangular singular-value transformations only where the mathematical
  construction and backend decomposition are explicit,
- preserve classification of classical-only, single-sequence, and mixed-parity
  constructions and return structured failures for unsupported combinations,
- prioritize decomposable, auditable circuits over additional dense spectral
  proxy demonstrations.

#### Phase synthesis, interoperability, and certification

- improve robustness for high degree, small boundedness margins, and poorly
  conditioned phase synthesis using failures observed on real workflow
  polynomials,
- provide convention-safe import, conversion, and adapter interfaces for
  maintained phase-synthesis implementations without making optional solvers
  mandatory dependencies,
- use the multi-polynomial stress matrix to compare supported solvers across
  degree, coefficient range, boundedness margin, convergence, classical
  runtime, phase count, and reconstruction error,
- add QSVT-valid full-domain extensions for targets currently designed only on
  positive singular values or spectra,
- quantify how phase rounding, synthesis residuals, and coherent perturbations
  affect output states, success probabilities, and observables,
- continue certification and reconstruction checks that do not rely only on
  sampled grids and document the phase convention beside every synthesis
  interface.

#### Flagship acceptance clients

The following workflows are the primary acceptance clients for the reusable
package interfaces:

1. quantum linear systems through a Poisson-type finite problem,
2. spectral or ground-state filtering through a Pauli-LCU problem,
3. Hamiltonian simulation through a finite Hermitian problem.

Each flagship must use the same package path from problem input through design,
synthesis, block encoding, execution, validation, and reporting. Each must
provide:

- a short Python entry point, CLI command, cookbook script, notebook client, and
  focused documentation page,
- matrix, operator, or `BlockEncodingSpec` input with explicit spectral,
  normalization, access-model, signal-subspace, and wire assumptions,
- target-polynomial design, boundedness certification, realizability
  classification, and phase synthesis,
- finite QSVT execution, or a precisely labeled incomplete tier when a required
  mechanism is unavailable,
- an appropriate classical reference and application-level observable,
- separate approximation, synthesis, block-encoding, state, observable,
  sampling, and resource ledgers,
- a JSON-safe, schema-versioned report with versioned acceptance results,
- regression tests with numerical tolerances and an explicit access-model
  support matrix.

#### Evidence and resource contracts

- derive each claim from the exact polynomial, synthesis result, access model,
  circuit, execution, and measurement artifacts returned by that run,
- keep execution tier, scalability, and resource completeness independent so a
  finite QSVT circuit is not mislabeled as a scalable oracle implementation,
- report polynomial degree, phase count, signal-operator calls, encoding width,
  gates, depth, wires, shots, postselection, and sampling costs where available,
- separate polynomial transformation, block encoding, state preparation,
  parity combination, amplitude amplification, readout, and compilation costs,
- mark resource reports as partial whenever a required layer is assumed,
  omitted, or lacks a concrete estimate,
- validate every finite executable workflow against dense spectral or
  singular-value references while treating full statevectors and solution
  vectors as simulator validation data rather than efficient quantum outputs.

### Next — Stabilize problem-solving workflows and reach beta readiness

After the three flagships share the reusable implementation path, stabilize the
supported problem-solving surface and use additional applications to test
distinct QSVT capabilities.

#### Package and release readiness

- keep the stable facade small while leaving lower-level research interfaces
  clearly experimental,
- use shared result and report types across Python, CLI, examples, and
  notebooks,
- document stable input shapes, error semantics, supported access models,
  report schemas, examples, and deprecation guarantees,
- keep type annotations, API-status metadata, report schemas, compatibility
  fixtures, migrations, and generated API documentation synchronized,
- pass lint, formatting, typing, tests, documentation, notebook, build,
  distribution, and fresh-wheel checks from a clean checkout,
- consider a beta-quality `0.3` line only after the three flagship acceptance
  clients pass their versioned contracts across every advertised access model.

#### Additional validation clients

- promote existing resolvent, regularized pseudoinverse, deblurring, spectral
  density, band-projector, thermal, and graph-matrix-function studies to
  acceptance clients only when they exercise a distinct QSVT construction,
  access model, validation method, or observable,
- keep application clients thin: domain construction, classical solvers,
  plotting, and interpretation remain in examples or notebooks while reusable
  QSVT logic moves into the package,
- compare each QSVT workflow with an appropriate classical method for the same
  finite task and state all normalization, input-loading, postselection,
  readout, and scalability assumptions,
- generate application tables and plots from saved, reproducible reports
  rather than notebook-local calculations.

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

Prefer deeper validation of an existing client over a new survey example
unless the example demonstrates a new QSVT construction, access model,
measurement strategy, or scientific observable.

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
