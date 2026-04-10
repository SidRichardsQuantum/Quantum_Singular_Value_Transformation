# Quantum Singular Value Transformation (QSVT)

[![PyPI version](https://img.shields.io/pypi/v/qsvt-pennylane.svg)](https://pypi.org/project/qsvt-pennylane/)
[![Python versions](https://img.shields.io/pypi/pyversions/qsvt-pennylane.svg)](https://pypi.org/project/qsvt-pennylane/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

Lightweight tools for experimenting with **Quantum Singular Value Transformation (QSVT)** using PennyLane.

This repository combines:

• a **notebook-first introduction** to QSVT
• a reusable **Python package** for bounded polynomial transforms

The focus is on **spectral intuition**:

how bounded polynomials transform singular values or eigenvalues via block encodings.

---

# Installation

Install from PyPI:

```bash
pip install qsvt-pennylane
```

Install from source:

```bash
git clone https://github.com/SidRichardsQuantum/Quantum_Singular_Value_Transformation.git
cd Quantum_Singular_Value_Transformation

pip install -e .
```

Requirements:

- Python ≥ 3.10
- PennyLane ≥ 0.36
- NumPy ≥ 1.23
- Matplotlib ≥ 3.7

---

# Quick example

Scalar polynomial transform:

```python
from qsvt.qsvt import qsvt_scalar_output

qsvt_scalar_output(
    x=0.5,
    poly=[0,0,1],  # x²
    encoding_wires=[0],
)
```

Diagonal transform:

```python
from qsvt.qsvt import qsvt_diagonal_transform

qsvt_diagonal_transform(
    values=[1.0, 0.7, 0.3, 0.1],
    poly=[0,0,1],
    encoding_wires=[0,1,2],
)
```

Design a bounded sign polynomial:

```python
from qsvt.design import design_sign_polynomial

coeffs = design_sign_polynomial(
    gamma=0.25,
    degree=13,
)
```

---

# Package overview

The package provides small, composable utilities for constructing and applying bounded polynomial transforms.

## Polynomial utilities

`qsvt.polynomials`

- Chebyshev polynomials
- polynomial degree and parity
- boundedness checks
- coefficient normalisation

---

## Polynomial approximation

`qsvt.approximation`

- Chebyshev fitting
- approximation error metrics
- polynomial evaluation helpers

---

## Polynomial templates

`qsvt.templates`

Ready-to-use bounded polynomial families:

- inverse-like polynomials
- sign approximations
- soft threshold filters
- sqrt approximations
- exponential weighting functions

Useful for quick experiments.

---

## Polynomial design

`qsvt.design`

Task-oriented polynomial builders:

- inverse-like transforms
- sign polynomials
- projector polynomials
- sqrt approximations
- power-law transforms
- smooth spectral filters

Designed for reusable QSVT workflows.

---

## Matrix helpers

`qsvt.matrices`

Small Hermitian test matrices:

- diagonal matrices
- rotated diagonal matrices
- involutory matrices

---

## Classical spectral reference

`qsvt.spectral`

Reference matrix-function utilities:

- matrix powers
- matrix square roots
- matrix sign
- spectral projectors

Useful for validating polynomial transforms.

---

## QSVT simulation utilities

`qsvt.qsvt`

Thin wrappers around PennyLane QSVT:

- scalar QSVT transforms
- diagonal transforms
- block extraction
- classical vs QSVT comparisons

---

# Documentation

Full documentation:

- Theory: [THEORY.md](THEORY.md)
- Usage guide: [USAGE.md](USAGE.md)
- API reference: [docs/qsvt/api_reference.md](docs/qsvt/api_reference.md)
- Package index: [docs/qsvt/index.md](docs/qsvt/index.md)

---

# Notebooks

The notebooks provide a guided introduction to QSVT as polynomial functional calculus.

1. scalar intuition
2. singular value filtering
3. QSP polynomials
4. exact linear solvers
5. approximate inverse behaviour
6. polynomial design and approximation
7. matrix powers and roots
8. sign function and projectors
9. reusable polynomial workflows

The examples emphasise:

- bounded polynomial structure
- spectral interpretation
- simple matrices
- reproducible results

---

# CLI

After installation:

```bash
qsvt scalar --x 0.5 --poly "0,0,1"

qsvt diag \
  --values "1.0,0.7,0.3,0.1" \
  --poly "0,0,1" \
  --wires 3

qsvt cheb --degree 3 --x 0.5
```

---

# Scope and philosophy

This repository is intentionally:

- educational
- explicit
- simulator-friendly
- polynomial-focused

The goal is to make QSVT easier to experiment with and understand.

Topics intentionally outside scope:

- circuit optimisation
- resource estimation
- fault tolerance
- amplitude amplification
- state preparation methods

The emphasis is understanding how polynomial transforms act on spectra.

---

# Author

Sid Richards

GitHub:
[https://github.com/SidRichardsQuantum](https://github.com/SidRichardsQuantum)

LinkedIn:
[https://www.linkedin.com/in/sid-richards-21374b30b/](https://www.linkedin.com/in/sid-richards-21374b30b/)

---

# License

MIT License — see LICENSE
