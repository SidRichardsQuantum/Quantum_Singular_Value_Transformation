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

### Repository clients and experimental tooling

Examples, notebooks, adjacent-algorithm comparisons, declarative research
sweeps, benchmarks, and the local Experiment Studio support the package and
remain repository tooling. They do not expand the stable QSVT facade or define
separate product roadmaps.

#### Experiment Studio UX and workflow improvements

The repository-only Studio should remain a local, single-user scientific
workbench and continue to focus on package-backed workflows rather than a
separate product surface. Targeted improvements in scope for the repo include:

- richer run-history navigation with search, filters, saved queries, and
  tagging for workflow families, failure modes, and compatibility outcomes,
- more expressive run summaries that distinguish queued, executing,
  synthesizing, reconstructing, and failed states without hiding the underlying
  package evidence,
- better management of comparison and reuse workflows, including side-by-side
  diffs for request settings, package reports, and numerical diagnostics,
- clearer progress feedback for long jobs, including lifecycle acknowledgements,
  cancellation confirmation, and a compact execution log for each run,
- stronger export and portability tools for configurations, reports, and plots,
  including JSON/CSV/PNG outputs that match saved workspace artifacts,
- improved onboarding for new users via recommended presets, quick-start
  examples, and explicit guidance on when a run is a design-only study versus an
  executed finite QSVT workflow,
- tighter failure communication: the Studio should make it obvious when a run
  failed because of synthesis, reconstruction, acceptance, sampling, or
  execution limits while preserving the exact package report for inspection,
- local workspace ergonomics such as filtered favorites, run retention policy,
  and better handling of large result collections without introducing a remote
  service or multi-user backend.

These refinements belong in the repository tooling layer: they improve the
scientific workflow and reproducibility story without expanding the package's
stable API or introducing separate product responsibilities.

## Stable Educational Milestone

The package may be described as stable for its stated educational scope when:

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
not imply that all possible applications are complete.

## Priorities

### Now — Harden reusable QSVT implementations through flagship clients

Finite coherent mixed-parity execution landed in `0.2.21`; `0.2.22` broadened
its tested contract to FABLE, PrepSelPrep, qubitization, and caller-supplied
projector conventions. All three flagship workflows now have finite executable
acceptance paths, and the repository Studio exposes each through public package
APIs. The immediate goal is to harden synthesis, access-model support, and
execution/report contracts across those workflows.

The support matrix in `docs/qsvt/flagship_workflows.md` now has
statevector and finite-shot regressions for all eight advertised high-level
workflow/encoding pairs. FABLE Hamiltonian scaling and non-finite synthesis
residual rejection are covered explicitly. Shared caller-supplied `BlockEncodingSpec` inputs and backend decomposition
remain separate work; the support matrix does not claim those are complete.
Finite-shot runs now have a separate conditional-probability acceptance scope
with simultaneous confidence bounds and postselection evidence. General
observable measurements and statevector validation from shots remain outside
that scope. Studio now provides saved request/report differences and structured
failure and synthesis-attempt inspection.
Synthesis stress reports retain per-attempt failure stages and reconstruction
evidence, with focused near-boundary and cosine-fit regressions.

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

#### Additional real-world notebook clients

Add a small number of application notebooks only when they exercise a distinct
package workflow and remain near-pure clients. Phonon density of states and
finite-temperature Heisenberg observables are now maintained clients in
notebooks 09 and 10. Finite disordered transport is covered by notebook 11,
with executed coherent QSVT and transport-observable validation. The remaining
candidates are:

- ground-state preparation and overlap estimation for a finite Heisenberg
  chain, using the ground-state filtering workflow,
- occupied/unoccupied band separation in a disordered material model, using
  spectral thresholding.

Each candidate should include a physical observable, a dense classical
reference, package-generated polynomial results, an application-level error
comparison, and explicit finite-simulator and access-model boundaries. Domain
constructors, plotting, and application-specific analysis should remain in the
notebook unless they become reusable QSVT primitives.

#### Additional validation clients

##### Real-measurement deblurring feasibility study

Use genuinely blurred camera images to test whether a bounded singular-value
filter can recover useful information from real optical measurements, and to
measure the additional error and cost introduced when that filter is
implemented through QSVT. The practical objective is improved text
recognition; the scientific objective is to isolate the accuracy and resource
requirements of the QSVT contribution. This is not, by itself, evidence of a
quantum speedup.

The proposed pilot uses the Helsinki Deblur Challenge dataset, which provides
focused and deliberately defocused photographs, calibration targets for
estimating blur, and text transcriptions for evaluating readability. Begin
with a modest subset spanning mild, moderate, and strong blur. Define the
evaluation before tuning, keep crops from one photograph in the same split,
separate calibration and parameter selection from final testing, and freeze
the OCR system. Character-recognition error is the primary outcome; image
reconstruction error and forward-model residuals are supporting diagnostics.
Report failures and uncertainty across images rather than selecting attractive
reconstructions.

The study should proceed in the following order:

1. **Validate the physical model.** Represent the measurement as
   \(y = Bx + \eta\), where \(x\) is the sharp image, \(B\) is optical blur,
   and \(\eta\) is measurement noise. Estimate \(B\) from calibration targets
   and validate alignment, intensity scaling, boundary treatment, and
   predictions on independent sharp references. Do not proceed to more
   accurate QSVT execution if the forward model is not trustworthy.
2. **Establish classical references.** Compare the blurred input with
   truncated SVD on small problems and with Tikhonov or Wiener reconstruction.
   Select regularization on development data and evaluate only after the
   settings are frozen.
3. **Evaluate the package polynomial.** Use
   `docs/qsvt/workflow_singular_value_pseudoinverse.md` as the starting point,
   while labeling its dense-SVD path as a polynomial validation study rather
   than a complete deblurring circuit. Investigate a smooth filter such as
   \(f_\lambda(\sigma) = \sigma/(\sigma^2+\lambda)\), with explicit checks for
   normalization, parity, approximation error, and behavior over the full
   signal domain. Separate inverse-problem and regularization error from
   polynomial-approximation error.
4. **Audit reduced quantum instances.** Derive small representative instances
   from the measured data with explicit boundary conditions. Verify block
   encodings, synthesize phases, execute ideal and finite-shot circuits with
   controlled noise, and compare against the identical classical
   transformation. Distinguish these experiments from larger classical image
   reconstructions.
5. **Define an honest readout boundary.** Basis probabilities are not
   reconstructed image amplitudes: finite-shot checks do not establish pixel
   intensities, signs, or global normalization. Image reconstruction therefore
   requires an explicit readout procedure, or the quantum experiment must use a
   narrower directly measurable output. Audit state preparation, decomposition,
   depth, postselection, measurement, and resource costs before considering
   physical hardware. Simulator-oriented encodings and
   hardware-decomposable alternatives must remain clearly distinguished.

The repository work should include a dataset manifest, calibration and
preprocessing, a fixed evaluation protocol, reproducible parameter sweeps,
classical baselines, validated regularization-filter design, an audited circuit
path for the selected matrix and adjoint conventions, readout and uncertainty
validation, and a thin Studio client for package reports, comparisons, and
failures. Imaging-specific constructors, OCR, plotting, and analysis stay in
repository tooling; only reusable QSVT primitives and diagnostics belong in
the package.

Use structured or matrix-free convolution for larger classical studies. A
32–64 GB workstation is an initial planning assumption rather than a measured
requirement: a dense float64 blur matrix for a million-pixel image would
require roughly 8 TB. More compute does not resolve phase-synthesis
difficulties, quantum data-loading costs, deep circuits, or image-readout
costs. The first deliverable is a small reproducible feasibility study with
real blurred images, independently calibrated blur, held-out readability
results, polynomial-versus-classical comparisons, and a resource-audited
reduced-instance circuit demonstration. Use its evidence to decide whether a
larger experiment is justified.

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
