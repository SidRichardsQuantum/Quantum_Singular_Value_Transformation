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
- building small Hamiltonians and finite-difference PDE operators
- designing physics matrix-function polynomials
- rescaling Hermitian spectra for QSVT-compatible workflows
- comparing classical baselines with QSVT-oriented resource proxies

The repository also includes a sequence of notebooks that introduce QSVT concepts step-by-step.

---

## Quick navigation

| goal | start here |
| --- | --- |
| Install the package and run a transform | [Usage guide](usage.md) |
| Understand the mathematical setup | [Theory](theory.md) |
| Explore executable examples | [Notebooks](notebooks.md) |
| Inspect notebook-generated outputs | [Results](results.md) |
| Compare classical baselines and QSVT proxies | [Classical benchmarks](benchmarks.md) |
| Understand baseline assumptions | [Classical baseline details](classical_baselines.md) |
| Interpret QSVT resource proxies | [QSVT resource model](qsvt_resource_model.md) |
| Compare tutorial outputs directly | [Tutorial notebook outputs](tutorial_results.md) |
| Browse real-example outputs | [Real-example notebook outputs](real_example_results.md) |
| Inspect benchmark outputs | [Benchmark notebook outputs](benchmark_results.md) |
| Use package APIs | [API reference](api_reference.md) |

---

## Documentation overview

### Theory

Mathematical background and conceptual overview:

- polynomial constraints for QSVT
- block-encoding intuition
- spectral transformations
- inverse-like behaviour via bounded polynomials
- projectors and sign functions

See: [Theory](theory.md)

---

### API reference

Detailed reference for the Python package:

- module structure
- function descriptions
- input and output conventions
- minimal usage examples

See: [API reference](api_reference.md)

### Usage guide

Practical workflows and command line examples:

- installing `qsvt-pennylane`
- choosing or designing bounded polynomials
- applying transforms to scalars, diagonal matrices, and Hermitian matrices
- comparing classical and QSVT outputs
- using the package CLI

See: [Usage guide](usage.md)

### Classical benchmarks

Classical benchmark reports cover dense eigensolvers, dense linear solves,
conjugate gradients, spectral matrix functions, and polynomial matrix-function
references. They can attach QSVT resource proxies for degree and signal-call
comparison.

See: [Classical benchmarks](benchmarks.md)

For the assumptions behind each classical timing path, see
[Classical baseline details](classical_baselines.md). For the proxy quantities
attached to QSVT-style comparisons, see [QSVT resource model](qsvt_resource_model.md).

### Notebooks

Notebook-first examples cover the core QSVT path and real physics workflows,
including matrix functions, spectral filters, PDE operators, Hamiltonian
simulation, and transport examples.

See: [Notebooks](notebooks.md)

### Results

Notebook-derived outcomes and reproducible artefact conventions:

- which notebooks currently include embedded plots and text results
- which real physics workflows are clean execution sources
- where future JSON reports, plots, and tables should live
- command line examples for regenerating report artefacts

See: [Results](results.md)

Rendered notebook artefacts are kept separately from the root result index.

- Tutorial outputs: [Tutorial notebook outputs](tutorial_results.md)
- Real-example outputs: [Real-example notebook outputs](real_example_results.md)
- Benchmark outputs: [Benchmark notebook outputs](benchmark_results.md)

### Algorithm and implementation notes

Workflow-level theory and implementation conventions:

- [Algorithm notes](algorithms.md): concise descriptions of the high-level
  algorithm workflows, their mathematical targets, diagnostics, and limits
- [Implementation notes](implementation.md): coefficient conventions,
  rescaling choices, boundedness/compatibility checks, report serialization,
  and public API status

### Changelog

Release notes document package, notebook, documentation, and generated artefact
changes.

See: [Changelog](changelog.md)

---

## Package structure

```

qsvt
├── polynomials.py
├── approximation.py
├── design.py
├── templates.py
├── workflow.py
├── algorithms.py
├── reports.py
├── benchmarks.py
├── compatibility.py
├── matrices.py
├── hamiltonians.py
├── pde.py
├── rescaling.py
├── matrix_functions.py
├── diagnostics.py
├── spectral.py
├── operators.py
├── diagonal.py
├── matrix.py
├── qsvt.py
└── __main__.py

```

Each module is intentionally small and focused:

| module | purpose |
|--------|--------|
| `polynomials` | Chebyshev utilities and polynomial helpers |
| `approximation` | bounded polynomial approximation tools |
| `design` | task-oriented bounded polynomial builders |
| `templates` | ready-made bounded polynomial families |
| `workflow` | combined coefficient, diagnostic, and compatibility workflows |
| `algorithms` | end-to-end simulator-scale algorithm workflows |
| `reports` | diagnostics serialization and plotting helpers |
| `benchmarks` | classical baselines and QSVT-oriented benchmark summaries |
| `compatibility` | sampled QSVT compatibility and synthesis checks |
| `matrices` | small Hermitian test matrices |
| `hamiltonians` | reusable small physics Hamiltonians |
| `pde` | finite-difference PDE operators |
| `rescaling` | spectral normalization helpers |
| `matrix_functions` | polynomial builders for physics matrix functions |
| `diagnostics` | state, operator, and spectral diagnostics |
| `spectral` | classical spectral matrix functions |
| `operators`, `diagonal`, `matrix` | focused QSVT construction and comparison helpers |
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

qsvt design-workflow --kind sign --gamma 0.2 --degree 13 \
  --output sign-workflow.json

qsvt compatibility-report --poly "0,0,1"

qsvt design-compatibility \
  --kind sign \
  --degree 13 \
  --gamma 0.2

qsvt compare-report \
  --values "1.0,0.7,0.3,0.1" \
  --poly "0,0,1" \
  --wires 3

qsvt matrix-report \
  --matrix "0.31351701,-0.23499807;-0.23499807,0.68648299" \
  --poly "0,0,1"

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
