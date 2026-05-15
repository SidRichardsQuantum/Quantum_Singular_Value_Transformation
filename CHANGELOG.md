# Changelog

---

## [0.1.12] – 15th May 2026

### Added

- added committed CLI-generated release artefacts under `results/`, including
  design, diagonal QSVT, and Hermitian matrix transform reports
- added an extracted introductory notebook plot gallery to `RESULTS.md`
- added `scripts/extract_notebook_plots.py` to regenerate notebook plot
  artefacts from embedded notebook PNG outputs

### Changed

- configured pytest with `pythonpath = ["src"]` so fresh checkouts can run
  tests without requiring an editable install first
- expanded notebook execution coverage to include both introductory notebooks
  and real physics example notebooks
- updated the 2x2 linear-solver notebook plot to use real-valued amplitudes
  explicitly, removing complex-value matplotlib warnings from the test suite

---

## [0.1.11] – 15th May 2026

### Documentation

- refreshed the GitHub Pages landing page and Sphinx/Furo styling to better
  match the main Sid Richards portfolio site
- reorganised the documentation sidebar into Start Here, Guides, and Reference
  sections
- added first-class Sphinx pages for theory, usage, notebooks, results, and the
  changelog
- added `RESULTS.md` as a notebook result index and convention for future
  reports, plots, and tables
- replaced fragile author links in rendered documentation with the main
  portfolio link

### Changed

- normalised real physics example notebooks with explicit cell IDs for future
  `nbformat` compatibility
- verified all introductory and real physics notebooks execute with
  `nbconvert`

---

## [0.1.10] – 15th May 2026

### Added

- added real physics example notebooks under `notebooks/real_examples/`
  covering Hamiltonian simulation, ground-state filtering, quantum chemistry,
  Green's functions, spectral density estimation, Gibbs states, PDE linear
  systems, transport physics, and tensor-network hybrid filtering
- added `qsvt.hamiltonians` with reusable small Hamiltonian constructors for
  tight-binding chains, transverse-field Ising models, Heisenberg chains, and
  Pauli strings
- added `qsvt.pde` with finite-difference Laplacian builders for 1D and 2D
  PDE examples
- added `qsvt.rescaling` for Hermitian spectral normalization and cutoff-based
  rescaling workflows
- added `qsvt.matrix_functions` with polynomial builders for real-time
  evolution, imaginary-time evolution, resolvents, Gaussian spectral windows,
  low-energy projectors, and positive inverse workflows
- added `qsvt.diagnostics` with reusable state, operator, density-matrix,
  expectation-value, ground-state-overlap, and spectral-weight metrics
- added public Chebyshev/monomial basis conversion helpers in
  `qsvt.polynomials`
- added `design_positive_inverse_polynomial` and
  `design_positive_inverse_diagnostics` for positive definite inverse-style
  workflows
- added notebook smoke-test coverage for all real physics example notebooks

### Changed

- updated real physics notebooks to use general package APIs instead of local
  copies of reusable constructors and polynomial builders
- improved positive inverse design with `extension="auto"`, which compares
  bounded full-interval extensions and selects the lower sampled error on the
  positive design interval
- updated the README, usage guide, API reference, package overview, and docs
  landing page to document the new physics workflow surface area

### Documentation

- added a dedicated physics workflow guide at `docs/qsvt/physics.md`
- added a `notebooks/real_examples/README.md` mapping major physics areas to
  executable notebooks

---

## [0.1.9] – 14th May 2026

### Changed

- split the QSVT implementation into focused operator, diagonal, matrix, and
  compatibility modules while keeping `qsvt.qsvt` as a compatibility re-export
- moved CLI command implementation into `qsvt.cli`, leaving `qsvt.__main__` as
  a small entry point
- extracted shared Chebyshev fitting, parity projection, and boundedness
  utilities for design and template builders
- expanded Ruff checks to include import sorting and bug-prone rule families

### Added

- added `qsvt.workflow.design_workflow` for returning coefficients,
  diagnostics, and compatibility metadata from one structured API
- added a `design-workflow` CLI command for writing complete design workflow
  reports from the command line
- added regression tests for split-module imports, non-power-of-two matrix
  defaults, complex report serialization, non-finite compatibility inputs, and
  design workflow results

### Documentation

- documented the new workflow API and split implementation modules in the
  README and API reference

---

## [0.1.8] – 6th May 2026

### Changed

- fixed Hermitian matrix handling across `qsvt.spectral`, `qsvt.matrices`, and
  `qsvt.qsvt` so complex Hermitian inputs are preserved instead of being
  silently cast to real arrays
- updated non-diagonal QSVT matrix comparison reports to distinguish between
  real-symmetric comparisons (`comparison_basis = "real_part"`) and full
  complex Hermitian comparisons (`comparison_basis = "full_complex"`)
- extended report serialization so complex scalars and arrays are written as
  JSON-safe `{real, imag}` payloads
- updated report-oriented CLI commands so `--output` and `--plot` write compact
  stdout summaries by default, with `--print-report` available to force full
  JSON output
- fixed the package metadata `Documentation` URL to point at the live GitHub
  Pages site
- pruned `docs/_build` from source distributions and repository export archives
  for cleaner release artefacts

### Documentation

- updated the README, usage guide, API reference, and report guides to reflect
  the new CLI output behaviour and complex Hermitian matrix support
- refreshed the docs landing page styling so the generated site matches the
  linked portfolio site more closely

---

## [0.1.7] – 23rd April 2026

### Changed

- added non-diagonal Hermitian QSVT matrix transform helpers and reports
- added a `matrix-report` CLI command for comparing extracted QSVT blocks with
  classical spectral polynomial references
- fixed default matrix QSVT wire selection for embedding block encodings

---

## [0.1.6] – 23rd April 2026

### Changed

- added `qsvt.reports` helpers for JSON-safe diagnostics payloads, report
  save/load, and target-vs-polynomial plotting
- added `--output` and `--plot` options to `design-report` and
  `template-report`
- added `qsvt_transform_report` for comparing QSVT diagonal transforms with
  direct classical polynomial evaluation
- added `qsvt_compatibility_report` for parity, boundedness, coefficient
  finiteness, and PennyLane synthesis checks
- added `compare-report` and `apply-design` CLI commands for QSVT transform
  reports
- added `compatibility-report` and `design-compatibility` CLI commands
- documented diagnostics report workflows in the README, API reference, and
  dedicated reports guide
- cleaned generated local build, test, and lint artifacts after the `v0.1.5`
  release
- added repository archive attributes for consistent line endings and cleaner
  generated source archives
- added source-distribution manifest rules so docs, notebooks, tests, and
  release notes are included predictably
- added a package-build CI workflow that runs `python -m build` and
  `twine check` on pushes and pull requests

---

## [0.1.5] – 23rd April 2026

### Changed

- added diagnostics helpers for `qsvt.design` and `qsvt.templates`
- added `design-report` and `template-report` CLI commands for JSON
  approximation-quality reports
- expanded the markdown docs to cover the new diagnostics surface area

### Documentation

- updated the README to show the report commands and current release
- documented the report field structure in the API reference
- added short diagnostics examples to the design and template guides

---

## [0.1.4] – 23rd April 2026

### Changed

- refreshed the release metadata and rebuilt the package for `v0.1.4`
- published the `0.1.4` release to PyPI
- updated the documentation set to cover the newer `qsvt.design` and
  `qsvt.templates` modules more clearly

### Documentation

- expanded the package index to include the full module set
- cleaned up the API reference module overview and cross-links
- removed stale citation placeholders from the templates docs

---

## [0.1.3] – 23rd April 2026

### Changed

- intermediate release tag used to validate the automated build and publishing
  workflow

---

## [0.1.2] – 10th April 2026

### Added

New module: `qsvt.design`

Provides higher-level bounded polynomial construction helpers for common
QSVT/QSP design tasks, with outputs returned in standard ascending-degree
monomial coefficient form.

Included design helpers:

- `design_inverse_polynomial(gamma, degree)`
  - bounded odd inverse-like polynomial
  - approximates the normalized inverse profile `gamma / x` away from zero
  - intended for linear-solver-style or regularized inverse experiments

- `design_sign_polynomial(gamma, degree)`
  - bounded odd sign surrogate
  - useful for spectral sign and threshold-style workflows

- `design_projector_polynomial(gamma, degree)`
  - bounded projector-style polynomial based on `(1 + sign(x)) / 2`
  - useful for positive/negative spectral subspace experiments

- `design_sqrt_polynomial(a, degree)`
  - bounded square-root surrogate on a positive interval
  - useful for matrix-function and amplitude-scaling experiments

- `design_power_polynomial(alpha, degree, a=0.0)`
  - bounded positive-power surrogate on a positive interval
  - useful for simple spectral shaping workflows

- `design_filter_polynomial(cutoff, degree)`
  - bounded even soft-threshold filter
  - useful for singular-value filtering and smooth pass/reject experiments

### Implementation notes

- reuses the existing lightweight Chebyshev fitting helper in
  `qsvt.approximation`
- converts fitted Chebyshev polynomials back to ascending monomial coefficients
- numerically enforces boundedness on `[-1, 1]` by rescaling when necessary
- enforces odd/even parity where structurally appropriate
- keeps implementations NumPy-only and notebook-friendly

### Tests

Added minimal smoke tests covering:

- construction of all design helpers
- boundedness on `[-1, 1]`
- expected parity for inverse/sign/filter builders
- basic qualitative behaviour checks for inverse/sign/projector/sqrt/power

### Public API

Updated `qsvt.__init__` to export the new design helpers.

---

## [0.1.1] – 10th April 2026

### Added

New module: `qsvt.templates`

Provides ready-to-use bounded polynomial templates for QSVT/QSP-style
experiments, returned in standard ascending-degree coefficient form.

Included template families:

- `inverse_like_polynomial`
  - smooth bounded odd inverse-like surrogate on `[-1, 1]`
  - useful for regularised small-scale linear-solver style experiments

- `sign_approximation_polynomial`
  - smooth odd sign surrogate based on a steep bounded transition
  - useful for spectral sign and projector-style workflows

- `soft_threshold_filter_polynomial`
  - even soft pass/reject filter based on `|x|`
  - useful for singular-value filtering intuition and threshold experiments

- `sqrt_approximation_polynomial`
  - bounded square-root-like template on the canonical interval
  - useful for matrix-function and amplitude-scaling experiments

- `exponential_approximation_polynomial`
  - bounded exponential-like weighting polynomial
  - useful for smooth spectral damping and low-degree filter examples

### Implementation notes

- templates are constructed via lightweight Chebyshev fitting on `[-1, 1]`
- outputs are converted back to standard monomial coefficients
- boundedness on `[-1, 1]` is enforced numerically by rescaling if needed
- odd/even parity is enforced where appropriate for sign/inverse/filter families

### Tests

Added minimal smoke tests covering:

- template construction
- boundedness on `[-1, 1]`
- expected parity for odd/even templates
- basic qualitative behaviour checks

### Public API

Updated `qsvt.__init__` to export the new template builders and bumped the
package version to `0.1.1`.

---

## [0.1.0] – 10th April 2026

### Added

Core package structure under `src/qsvt/`:

- `polynomials.py`
  - Chebyshev polynomial utilities
  - polynomial evaluation helpers
  - parity detection and coefficient utilities

- `approximation.py`
  - Chebyshev-based bounded polynomial approximation
  - domain scaling helpers for mapping intervals to [-1,1]
  - approximation error utilities (max error, RMS error)
  - callable approximant construction

- `matrices.py`
  - small Hermitian matrix constructors
  - rotation matrices and rotated diagonal matrices
  - involutory diagonal matrices (±1 spectra)
  - helper utilities for vector normalisation and embedding

- `spectral.py`
  - classical spectral matrix-function utilities
  - eigendecomposition helpers
  - matrix powers and fractional powers
  - matrix sign function
  - spectral projectors onto positive/negative eigenspaces
  - polynomial transforms applied via eigenspectrum

- `qsvt.py`
  - thin PennyLane QSVT wrapper layer
  - scalar QSVT helpers (QSP-equivalent behaviour)
  - explicit unitary extraction via `qml.matrix`
  - logical top-left block extraction utilities
  - diagonal singular value transform helpers
  - classical vs QSVT comparison utilities
  - embedded-vector QSVT workflows

- `__init__.py`
  - curated public API surface
  - version exposure via `__version__`

- `__main__.py`
  - lightweight CLI entry point
  - scalar QSVT evaluation
  - diagonal singular-value transformation
  - Chebyshev polynomial evaluation
  - JSON-formatted output
  - numpy-safe serialization

### CLI

Added console script:

```

qsvt

```

Example usage:

```

qsvt scalar --x 0.5 --poly "0,0,1"

qsvt diag 
--values "1.0,0.7,0.3,0.1" 
--poly "0,0,1" 
--wires 3

qsvt cheb --degree 3 --x 0.5

```

### Packaging

- Added `pyproject.toml`
- configured setuptools backend
- package layout uses `src/` structure
- editable installs supported (`pip install -e .`)
- console script entry point configured
- Python ≥3.10 supported

### Tests

- added smoke tests for core functionality:
  - Chebyshev polynomial evaluation
  - polynomial parity detection
  - scalar QSVT correctness
  - diagonal QSVT correctness

### Documentation

- updated README.md to include:
  - PyPI installation instructions
  - CLI usage examples
  - package module overview
  - relationship between notebooks and reusable code

### Scope

Initial release focuses on:

- educational clarity
- explicit small-scale QSVT demonstrations
- classical reference implementations
- minimal abstraction over PennyLane primitives

No circuit optimisation, amplitude amplification, or hardware resource estimation included.
