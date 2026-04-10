# Changelog

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
