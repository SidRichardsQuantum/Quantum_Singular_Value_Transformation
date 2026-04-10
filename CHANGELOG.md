# Changelog

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
