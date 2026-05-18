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
<a href="https://github.com/sponsors/SidRichardsQuantum">
<img src="https://img.shields.io/badge/sponsor-GitHub-ea4aaa?style=flat-square&logo=githubsponsors" alt="Sponsor">
</a>
</p>

Lightweight tools for experimenting with **Quantum Singular Value
Transformation (QSVT)** and bounded polynomial transforms using PennyLane.

This repository combines:

- a notebook-first introduction to QSVT and QSP
- a reusable Python package for polynomial design, spectral transforms, and
  small PennyLane QSVT checks
- reproducible examples for scalar, matrix, PDE, and small physics workflows

The focus is spectral intuition: how bounded polynomials transform singular
values or eigenvalues through block encodings.

## Links

- PyPI: [qsvt-pennylane](https://pypi.org/project/qsvt-pennylane/)
- Website: [project documentation](https://SidRichardsQuantum.github.io/Quantum_Singular_Value_Transformation/)
- Usage guide: [USAGE.md](USAGE.md)
- Theory notes: [THEORY.md](THEORY.md)
- Results index: [RESULTS.md](RESULTS.md)
- API reference: [docs/qsvt/api_reference.md](docs/qsvt/api_reference.md)

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

- Python >= 3.10
- PennyLane >= 0.36
- NumPy >= 1.23
- Matplotlib >= 3.7

## Quick Example

Apply a scalar polynomial transform:

```python
from qsvt.qsvt import qsvt_scalar_output

result = qsvt_scalar_output(
    x=0.5,
    poly=[0, 0, 1],  # x^2
    encoding_wires=[0],
)
```

Design a bounded sign polynomial and keep the diagnostics:

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

Use the command line interface:

```bash
qsvt scalar --x 0.5 --poly "0,0,1"
qsvt design-workflow --kind sign --gamma 0.2 --degree 13
```

See [USAGE.md](USAGE.md) for full Python and CLI workflows.

## Package Map

The public package lives under `src/qsvt`.

| module | purpose |
| --- | --- |
| `qsvt.polynomials` | Chebyshev utilities, parity checks, boundedness checks |
| `qsvt.approximation` | polynomial fitting and approximation error helpers |
| `qsvt.design` | task-oriented polynomial builders |
| `qsvt.templates` | ready-made bounded polynomial families |
| `qsvt.workflow` | combined coefficient, diagnostic, and compatibility workflows |
| `qsvt.reports` | JSON-safe reports and plot helpers |
| `qsvt.matrices` | small Hermitian test matrices |
| `qsvt.spectral` | classical spectral matrix-function references |
| `qsvt.qsvt` | PennyLane QSVT wrappers and comparisons |
| `qsvt.hamiltonians`, `qsvt.pde`, `qsvt.rescaling` | reusable physics and PDE helpers |
| `qsvt.matrix_functions`, `qsvt.diagnostics` | matrix-function designs and validation metrics |

For detailed function-level documentation, use
[docs/qsvt/api_reference.md](docs/qsvt/api_reference.md).

## Documentation

- [USAGE.md](USAGE.md): practical package and CLI workflows
- [THEORY.md](THEORY.md): QSVT, QSP, polynomial constraints, and spectral
  interpretation
- [RESULTS.md](RESULTS.md): result-producing notebooks and reproducible
  artefact conventions
- [docs/qsvt/result_gallery.md](docs/qsvt/result_gallery.md): tutorial plot
  gallery
- [docs/qsvt/real_example_results.md](docs/qsvt/real_example_results.md):
  real-example plot and table gallery
- [docs/qsvt/design.md](docs/qsvt/design.md): polynomial design helpers
- [docs/qsvt/templates.md](docs/qsvt/templates.md): template polynomial
  families
- [docs/qsvt/physics.md](docs/qsvt/physics.md): Hamiltonian, PDE, rescaling,
  and matrix-function workflows
- [docs/qsvt/notebooks.md](docs/qsvt/notebooks.md): tutorial and real-example
  notebook index

Current release: `0.1.13`

## Notebooks

Tutorial notebooks live in `notebooks/tutorials/` and introduce QSVT as
polynomial functional calculus, from scalar transforms through sign functions,
projectors, matrix functions, and reusable design workflows.

Real physics examples live in `notebooks/real_examples/` and cover Hamiltonian
simulation, ground-state filtering, quantum chemistry, Green's functions,
spectral density estimation, Gibbs states, PDE systems, transport physics,
spin-chain diagnostics, electronic occupations, photonic band gaps, graphene
density of states, and tensor-network hybrid filtering.

See [docs/qsvt/notebooks.md](docs/qsvt/notebooks.md) for the full notebook map.

## Scope

This project is intentionally educational, explicit, simulator-friendly, and
polynomial-focused.

It does not aim to provide production-scale circuit optimization, resource
estimation, fault-tolerant constructions, amplitude amplification, or state
preparation methods. The emphasis is understanding how polynomial transforms
act on spectra.

## Support

If this repository is useful for research, learning, or experimentation, you
can support continued development through
[GitHub Sponsors](https://github.com/sponsors/SidRichardsQuantum).

## Author

Sid Richards

- GitHub: [SidRichardsQuantum](https://github.com/SidRichardsQuantum)
- LinkedIn: [Sid Richards](https://www.linkedin.com/in/sid-richards-21374b30b/)

## License

MIT License. See [LICENSE](LICENSE).
