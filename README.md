# Quantum Singular Value Transformation (QSVT)

<p align="center">

<a href="https://pypi.org/project/qsvt-pennylane/">
<img src="https://img.shields.io/pypi/v/qsvt-pennylane?style=flat-square" alt="PyPI Version">
</a>

<a href="https://pypi.org/project/qsvt-pennylane/">
<img src="https://img.shields.io/pypi/pyversions/qsvt-pennylane?style=flat-square" alt="Python Versions">
</a>

<a href="LICENSE">
<img src="https://img.shields.io/github/license/SidRichardsQuantum/Quantum_Singular_Value_Transformation?style=flat-square" alt="License">
</a>

<a href="tests/">
<img src="https://img.shields.io/badge/tests-passing-brightgreen?style=flat-square" alt="Tests">
</a>

<a href="https://github.com/sponsors/SidRichardsQuantum">
<img src="https://img.shields.io/badge/sponsor-GitHub-ea4aaa?style=flat-square&logo=githubsponsors" alt="Sponsor">
</a>

</p>

**PyPI:** [https://pypi.org/project/qsvt-pennylane/](https://pypi.org/project/qsvt-pennylane/)

**Website:** [https://SidRichardsQuantum.github.io/Quantum_Singular_Value_Transformation/](https://SidRichardsQuantum.github.io/Quantum_Singular_Value_Transformation/)

Lightweight tools for experimenting with **Quantum Singular Value Transformation (QSVT)** using PennyLane.

This repository combines:

- a **notebook-first introduction** to QSVT
- a reusable **Python package** for bounded polynomial transforms

The focus is on **spectral intuition**:

how bounded polynomials transform singular values or eigenvalues via block encodings.

---

## Table of Contents

- [Installation](#installation)

- [Quick example](#quick-example)

- [Package overview](#package-overview)

  - [Polynomial utilities](#polynomial-utilities)
  - [Polynomial approximation](#polynomial-approximation)
  - [Polynomial templates](#polynomial-templates)
  - [Polynomial design](#polynomial-design)
  - [Design workflows](#design-workflows)
  - [Reports](#reports)
  - [Matrix helpers](#matrix-helpers)
  - [Classical spectral reference](#classical-spectral-reference)
  - [QSVT simulation utilities](#qsvt-simulation-utilities)

- [Documentation](#documentation)

- [Notebooks](#notebooks)

- [CLI](#cli)

- [Scope and philosophy](#scope-and-philosophy)

- [Support development](#support-development)

- [Author](#author)

- [License](#license)

---

## Installation

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

## Quick example

Scalar polynomial transform:

```python
from qsvt.qsvt import qsvt_scalar_output

qsvt_scalar_output(
    x=0.5,
    poly=[0,0,1],  ## x²
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

Collect coefficients, diagnostics, and compatibility in one workflow:

```python
from qsvt.workflow import design_workflow

result = design_workflow(
    kind="sign",
    gamma=0.25,
    degree=13,
)

coeffs = result.coeffs
report = result.as_report()
```

---

## Package overview

The package provides small, composable utilities for constructing and applying bounded polynomial transforms.

### Polynomial utilities

`qsvt.polynomials`

- Chebyshev polynomials
- polynomial degree and parity
- boundedness checks
- coefficient normalisation

---

### Polynomial approximation

`qsvt.approximation`

- Chebyshev fitting
- approximation error metrics
- polynomial evaluation helpers

---

### Polynomial templates

`qsvt.templates`

Ready-to-use bounded polynomial families:

- inverse-like polynomials
- sign approximations
- soft threshold filters
- sqrt approximations
- exponential weighting functions
- approximation-quality reports

Useful for quick experiments.

---

### Polynomial design

`qsvt.design`

Task-oriented polynomial builders:

- inverse-like transforms
- sign polynomials
- projector polynomials
- sqrt approximations
- power-law transforms
- smooth spectral filters
- approximation-quality reports

Designed for reusable QSVT workflows.

---

### Design workflows

`qsvt.workflow`

- structured design results
- coefficients plus diagnostics
- QSVT compatibility report
- report-style export via `DesignWorkflowResult.as_report()`

Useful when a script or notebook needs a complete design artefact instead of
separate calls into `qsvt.design` and `qsvt.qsvt`.

---

### Reports

`qsvt.reports`

- convert diagnostics reports to JSON-safe containers
- save and load report JSON files
- plot target, polynomial, and error curves

Useful for recording approximation quality and making report output reusable
outside notebooks.

---

### Matrix helpers

`qsvt.matrices`

Small Hermitian test matrices:

- diagonal matrices
- rotated diagonal matrices
- involutory matrices

---

### Physics workflow helpers

Reusable general-purpose modules for real physics examples:

- `qsvt.hamiltonians`: tight-binding, Ising, Heisenberg, and Pauli-string matrices
- `qsvt.pde`: finite-difference Laplacian operators
- `qsvt.rescaling`: Hermitian spectral normalization helpers
- `qsvt.matrix_functions`: real-time, imaginary-time, resolvent, window, projector, and inverse polynomial builders
- `qsvt.diagnostics`: state, operator, overlap, and spectral-weight metrics

These are designed to keep notebooks problem-focused while the package supplies
the reusable spectral machinery.

---

### Classical spectral reference

`qsvt.spectral`

Reference matrix-function utilities:

- matrix powers
- matrix square roots
- matrix sign
- spectral projectors

Useful for validating polynomial transforms.

---

### QSVT simulation utilities

`qsvt.qsvt`

Thin wrappers around PennyLane QSVT:

- scalar QSVT transforms
- diagonal transforms
- non-diagonal Hermitian matrix transforms
- block extraction
- classical vs QSVT comparisons
- QSVT transform reports

---

## Documentation

Full documentation:

- Theory: [THEORY.md](THEORY.md)
- Usage guide: [USAGE.md](USAGE.md)
- API reference: [docs/qsvt/api_reference.md](docs/qsvt/api_reference.md)
- Package index: [docs/qsvt/index.md](docs/qsvt/index.md)
- Physics workflows: [docs/qsvt/physics.md](docs/qsvt/physics.md)
- Changelog: [CHANGELOG.md](CHANGELOG.md)

Current release: `0.1.10`

---

## Notebooks

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

Real physics examples live in `notebooks/real_examples/` and cover Hamiltonian
simulation, ground-state filtering, quantum chemistry, Green's functions,
spectral density estimation, Gibbs states, PDE linear systems, transport
physics, and tensor-network hybrid filtering.

---

## CLI

After installation:

```bash
qsvt scalar --x 0.5 --poly "0,0,1"

qsvt diag \
  --values "1.0,0.7,0.3,0.1" \
  --poly "0,0,1" \
  --wires 3

qsvt cheb --degree 3 --x 0.5

qsvt design-report --kind sign --gamma 0.2 --degree 13 \
  --output sign-report.json \
  --plot sign-report.png

qsvt design-workflow --kind sign --gamma 0.2 --degree 13 \
  --output sign-workflow.json

qsvt template-report --kind inverse --degree 7 --mu 0.3 \
  --output inverse-report.json

qsvt compatibility-report --poly "0,0,1"

qsvt design-compatibility \
  --kind sign \
  --degree 13 \
  --gamma 0.2

qsvt compare-report \
  --values "1.0,0.7,0.3,0.1" \
  --poly "0,0,1" \
  --wires 3 \
  --output qsvt-report.json

qsvt matrix-report \
  --matrix "0.31351701,-0.23499807;-0.23499807,0.68648299" \
  --poly "0,0,1" \
  --output matrix-report.json

qsvt apply-design \
  --kind sign \
  --values="-0.8,-0.3,0.3,0.8" \
  --degree 13 \
  --gamma 0.2 \
  --wires 3
```

The report commands print the same JSON diagnostics used by the Python
helpers, including fit error and boundedness information. `design-workflow`
combines coefficients, diagnostics, and QSVT compatibility metadata in one
JSON payload. `--output` writes the report to JSON, and `--plot` writes a
target-vs-polynomial plot for approximation reports. When either flag is used,
stdout switches to a compact write summary; add `--print-report` if you also
want the full JSON report on stdout.

Compatibility reports distinguish bounded polynomial approximation from
PennyLane QSVT synthesis compatibility.

---

## Scope and philosophy

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

## Support development

If this repository is useful for research, learning, or experimentation, you can support continued development via GitHub Sponsors:

[https://github.com/sponsors/SidRichardsQuantum](https://github.com/sponsors/SidRichardsQuantum)

Sponsorship supports continued work on open-source implementations of quantum algorithms, including polynomial-based quantum signal processing, spectral transforms, and reproducible research tooling.

Support helps maintain accessible reference implementations for experimenting with QSVT, QSP, and matrix functional calculus workflows.

---

## Author

Sid Richards

GitHub: [https://github.com/SidRichardsQuantum](https://github.com/SidRichardsQuantum)

LinkedIn: [https://www.linkedin.com/in/sid-richards-21374b30b/](https://www.linkedin.com/in/sid-richards-21374b30b/)

---

## License

MIT License — see [LICENSE](LICENSE)
