# qsvt-pennylane documentation

`qsvt-pennylane` is a lightweight Python package for exploring **Quantum Singular Value Transformation (QSVT)** and **Quantum Signal Processing (QSP)** using PennyLane.

The package provides small, explicit utilities for:

- constructing bounded polynomials suitable for QSVT
- approximating functions on bounded spectral intervals
- building simple Hermitian test matrices
- applying classical spectral matrix functions
- extracting explicit QSVT transforms using `qml.qsvt`
- comparing classical polynomial transforms with QSVT outputs
- reporting QSVT-vs-classical transform error
- building task-oriented polynomial designs
- reusing ready-made polynomial templates
- reporting fit error and boundedness for polynomial builders
- saving, loading, and plotting diagnostics reports

The repository also includes a sequence of notebooks that introduce QSVT concepts step-by-step.

---

## Documentation overview

### Theory

Mathematical background and conceptual overview:

- polynomial constraints for QSVT
- block-encoding intuition
- spectral transformations
- inverse-like behaviour via bounded polynomials
- projectors and sign functions

See: [Theory](../../THEORY.md)

---

### API reference

Detailed reference for the Python package:

- module structure
- function descriptions
- input and output conventions
- minimal usage examples

See: [API reference](api_reference.md)

---

## Package structure

```

qsvt
├── polynomials.py
├── approximation.py
├── matrices.py
├── spectral.py
├── design.py
├── templates.py
├── reports.py
├── qsvt.py
└── __main__.py

```

Each module is intentionally small and focused:

| module | purpose |
|--------|--------|
| `polynomials` | Chebyshev utilities and polynomial helpers |
| `approximation` | bounded polynomial approximation tools |
| `matrices` | small Hermitian test matrices |
| `spectral` | classical spectral matrix functions |
| `design` | task-oriented bounded polynomial builders |
| `templates` | ready-made bounded polynomial families |
| `reports` | diagnostics serialization and plotting helpers |
| `qsvt` | PennyLane QSVT wrappers |
| `__main__` | command line interface |

---

## Relationship between notebooks and package

The notebooks provide conceptual explanations and worked examples.

The package extracts reusable components from these notebooks into a stable importable interface.

Typical workflow:

1. read notebook explanation
2. experiment interactively
3. reuse helpers from `qsvt`
4. build new examples or approximations

---

## Minimal example

```python
from qsvt.qsvt import qsvt_scalar_output
from qsvt.polynomials import chebyshev_t

# polynomial transform via QSVT
result = qsvt_scalar_output(
    0.5,
    [0, 0, 1],
    encoding_wires=[0],
)

print(result)
```

---

## Command line interface

The package also provides a small CLI:

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

qsvt compatibility-report --poly "0,0,1"

qsvt design-compatibility \
  --kind sign \
  --degree 13 \
  --gamma 0.2

qsvt compare-report \
  --values "1.0,0.7,0.3,0.1" \
  --poly "0,0,1" \
  --wires 3

qsvt apply-design \
  --kind sign \
  --values="-0.8,-0.3,0.3,0.8" \
  --degree 13 \
  --gamma 0.2 \
  --wires 3
```

---

## Scope

This project focuses on:

- clarity of spectral intuition
- explicit small-dimensional examples
- polynomial mechanism understanding
- reproducible notebook workflows

It does not aim to implement production-scale quantum linear solvers or hardware-optimised circuits.

---

## Links

GitHub repository: [https://github.com/SidRichardsQuantum/Quantum_Singular_Value_Transformation](https://github.com/SidRichardsQuantum/Quantum_Singular_Value_Transformation)

PyPI package: [https://pypi.org/project/qsvt-pennylane/](https://pypi.org/project/qsvt-pennylane/)
