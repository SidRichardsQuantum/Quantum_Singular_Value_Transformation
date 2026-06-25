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
  small PennyLane QSVT checks where the backend can synthesize the transform
- explicit polynomial realizability classification and phase-synthesis reports
- block-encoding specifications for matrices, PennyLane operators, and
  user-provided circuit factories
- reproducible examples for scalar, matrix, PDE, and small physics workflows

The focus is spectral intuition and reproducible validation: how bounded
polynomials transform singular values or eigenvalues, what approximation error
they incur on concrete finite instances, and which extra quantum assumptions
would be needed to turn the polynomial core into a complete algorithm.

## Links

- PyPI: [qsvt-pennylane](https://pypi.org/project/qsvt-pennylane/)
- Website: [project documentation](https://SidRichardsQuantum.github.io/Quantum_Singular_Value_Transformation/)
- Usage guide: [USAGE.md](USAGE.md)
- Cookbook examples: [examples/](examples/)
- Theory notes: [THEORY.md](THEORY.md)
- Results index: [RESULTS.md](RESULTS.md)
- Roadmap: [ROADMAP.md](ROADMAP.md)
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
qsvt phase-synthesis --poly "0,1,0,-0.5,0,0.333333"
qsvt boundedness-certificate --poly "0.996,0.1,-0.5"
qsvt phase-solver-benchmark --poly "0,1" --solvers root-finding --repeats 3
qsvt mixed-parity-synthesis --poly "0.5,0.5"
qsvt design-workflow --kind sign --gamma 0.2 --degree 13
qsvt design-sweep --kind sign --degrees "5,9,13,17" --gamma 0.2 \
  --no-synthesis --output sign-degree-sweep.json
qsvt resource-report --poly "0,0,1" --matrix-dimension 4 --no-synthesis
qsvt execute-spec --kind matrix --matrix "0.2,0;0,0.8" \
  --poly "0,0,1" --state "1,0"
qsvt benchmark cg-solve --matrix "4,1;1,3" --rhs "1,2" --qsvt-poly "0,1"
qsvt examples
```

Run copy-pasteable cookbook scripts from the repository root:

```bash
python examples/design_apply_report.py --output /tmp/qsvt-design-apply.json
python examples/linear_system_compare.py \
  --output /tmp/qsvt-linear-system.json \
  --rows-output /tmp/qsvt-linear-system.csv
python examples/threshold_filter.py --output /tmp/qsvt-threshold-filter.json
python examples/block_encoded_workflow.py \
  --output /tmp/qsvt-block-encoded-workflow.json
python examples/circuit_execution.py --output /tmp/qsvt-circuit-execution.json
python examples/block_encoding_execution.py \
  --output /tmp/qsvt-block-encoding-execution.json
python examples/rectangular_execution.py \
  --output /tmp/qsvt-rectangular-execution.json
python examples/compatibility_report.py --output /tmp/qsvt-compatibility.json
python examples/benchmark_summary.py \
  --output /tmp/qsvt-benchmark-summary.json \
  --rows-output /tmp/qsvt-benchmark-summary.csv
```

See [USAGE.md](USAGE.md) for full Python and CLI workflows.

## Package Map

The public package lives under `src/qsvt`.

| module | purpose |
| --- | --- |
| `qsvt.polynomials` | Chebyshev utilities, parity checks, boundedness checks |
| `qsvt.approximation` | polynomial fitting and approximation error helpers |
| `qsvt.design` | task-oriented polynomial builders |
| `qsvt.algorithms` | end-to-end simulator-scale algorithm workflows |
| `qsvt.block_encoding` | finite dense block-encoding construction and verification |
| `qsvt.execution` | QNode execution for matrices and block-encoding specifications |
| `qsvt.synthesis` | realizability classification, parity decomposition, and phase synthesis |
| `qsvt.templates` | ready-made bounded polynomial families |
| `qsvt.workflow` | combined coefficient, diagnostic, and compatibility workflows |
| `qsvt.reports` | JSON-safe reports and plot helpers |
| `qsvt.resources` | degree, phase-count, width, and compatibility proxy reports |
| `qsvt.benchmarks` | classical baselines and QSVT-oriented benchmark summaries |
| `qsvt.notebook` | experimental notebook presentation and path helpers used by committed notebooks |
| `qsvt.matrices` | small Hermitian test matrices |
| `qsvt.spectral` | classical spectral matrix-function references |
| `qsvt.qsvt` | PennyLane QSVT wrappers and comparisons |
| `qsvt.hamiltonians`, `qsvt.pde`, `qsvt.rescaling` | reusable physics and PDE helpers |
| `qsvt.matrix_functions`, `qsvt.diagnostics` | matrix-function designs and validation metrics |

For detailed function-level documentation, use
[docs/qsvt/api_reference.md](docs/qsvt/api_reference.md).

The package includes a `py.typed` marker so type checkers can consume the
inline type annotations shipped with the public modules.

During the `0.x` series, `qsvt.api_status(name)` labels exported names as
`stable` or `experimental`. Workflow-level helpers and report/export utilities
are the most stable user-facing surface; lower-level circuit execution and
backend-adapter helpers remain experimental while the package approaches a
`1.0` API.

## Roadmap

The project is moving toward general package workflows that users can apply to
specific physics and mathematics problems from thin client notebooks. Core
helpers should stay reusable across domains; notebooks should focus on problem
setup, parameter choices, and interpretation.

See [ROADMAP.md](ROADMAP.md) for the current development direction.

## Documentation

- [USAGE.md](USAGE.md): practical package and CLI workflows
- [examples/](examples/): short cookbook scripts for common package workflows
- [THEORY.md](THEORY.md): QSVT, QSP, polynomial constraints, and spectral
  interpretation
- [RESULTS.md](RESULTS.md): result-producing notebooks and reproducible
  artefact conventions
- [docs/qsvt/tutorial_results.md](docs/qsvt/tutorial_results.md): generated
  tutorial notebook outputs
- [docs/qsvt/real_example_results.md](docs/qsvt/real_example_results.md):
  generated real-example notebook outputs
- [docs/qsvt/benchmark_results.md](docs/qsvt/benchmark_results.md): generated
  benchmark notebook outputs
- [docs/qsvt/classical_baselines.md](docs/qsvt/classical_baselines.md):
  classical benchmark assumptions and baseline details
- [docs/qsvt/qsvt_resource_model.md](docs/qsvt/qsvt_resource_model.md):
  QSVT proxy-resource interpretation and omitted costs
- [docs/qsvt/design.md](docs/qsvt/design.md): polynomial design helpers
- [docs/qsvt/algorithms.md](docs/qsvt/algorithms.md): workflow-level
  algorithm notes, diagnostics, and limitations
- [docs/qsvt/block_encoding.md](docs/qsvt/block_encoding.md): finite dense
  block encodings, normalization, verification, and omitted oracle costs
- [docs/qsvt/compatibility.md](docs/qsvt/compatibility.md): QSVT boundedness,
  parity, synthesis checks, and common failure modes
- [docs/qsvt/templates.md](docs/qsvt/templates.md): template polynomial
  families
- [docs/qsvt/physics.md](docs/qsvt/physics.md): Hamiltonian, PDE, rescaling,
  and matrix-function workflows
- [docs/qsvt/implementation.md](docs/qsvt/implementation.md): implementation
  conventions, report serialization, and API status
- [docs/qsvt/notebooks.md](docs/qsvt/notebooks.md): tutorial, benchmark, and
  real-example notebook index

Current release: `0.2.9`

## Notebooks

Tutorial notebooks live in `notebooks/tutorials/` and introduce QSVT as
polynomial functional calculus, from scalar transforms through sign functions,
projectors, matrix functions, reusable design workflows, end-to-end algorithm
workflows, reproducible reports, degree/error tradeoff studies, and
resource-proxy limitations.

Real physics examples live in `notebooks/real_examples/` and cover Hamiltonian
simulation, ground-state filtering, quantum chemistry, Green's functions,
spectral density estimation, Gibbs states, PDE systems, transport physics,
spin-chain diagnostics, electronic occupations, singular-value inverse
problems, matrix-log graph entropy, photonic band gaps, graphene density of
states, topological band projectors, and tensor-network hybrid filtering. Each
real-example notebook includes a near-top orientation block for the system,
QSVT implementation strategy, and quantum relevance.

Benchmark notebooks live in `notebooks/benchmarks/` and compare classical
linear-system, spectral, and polynomial matrix-function baselines against
QSVT-oriented resource proxies and their underlying assumptions.

See [docs/qsvt/notebooks.md](docs/qsvt/notebooks.md) for the full notebook map.

## Notebook Result Workflow

Committed notebook outputs and generated result artefacts are the source of
truth for the published documentation. GitHub Pages builds from committed
`docs/qsvt/*_results.md`, `results/plots/`, and `results/tables/` files; it
does not execute notebooks during deployment.

Before committing notebook or result changes, run:

```bash
scripts/update_notebook_results.sh
```

Commit the updated notebooks, extracted plots, manifests, and generated result
pages together. CI checks that the committed result pages and manifests can be
regenerated from the committed notebook outputs without re-executing notebooks.

## Packaging

PyPI distributions are package-focused: they include the importable `qsvt`
package plus essential project metadata and root documentation. Full notebooks,
rendered documentation, committed result snapshots, and regression tests remain
in the GitHub repository and project website, where they can be audited and
regenerated without making installs or source distributions unnecessarily
large.

## Truth Contract

The package is designed to be useful for education, research prototyping, and
small real physics/math case studies, but its claims are deliberately scoped.

Implemented and tested:

- dense spectral polynomial references for finite matrices,
- bounded polynomial design and sampled diagnostics,
- simulator-scale workflows for linear systems, filters, matrix functions,
  resolvents, Gibbs weights, spectral density, and projectors,
- PennyLane QSVT block checks for supported small compatible polynomials,
- PennyLane QNode execution for finite QSVT circuits with state preparation,
  queued `qml.qsvt`, statevector/probability measurement, and circuit resource
  metadata,
- lower-level QSVT execution from dense, rectangular, PennyLane-operator, and
  custom-circuit block-encoding specifications, including caller-supplied
  signal projectors and structured backend failures,
- classical benchmark baselines plus QSVT-oriented proxy metadata.

Reported as assumptions or proxies:

- block-encoding availability and query cost,
- input-state preparation and data loading,
- measurement/readout and amplitude amplification,
- fault-tolerant synthesis, error correction, and hardware compilation,
- end-to-end runtime or quantum advantage claims.

Hardware execution is not yet a package implementation layer. The roadmap
includes small hardware-executable QSVT demonstrations using decomposable
FABLE, PrepSelPrep, qubitization, or custom block encodings; caller-supplied
preparation circuits; finite-shot measurements; and device-native compilation.
Such demonstrations would establish genuine execution of the selected finite
QSVT circuit, not scalability, fault tolerance, or quantum advantage.

Every high-level algorithm, direct QSVT comparison, resource, and benchmark
report includes a `truth_contract` field. Circuit execution reports separately
state when a QNode was actually executed. The fields state the implemented
dense-polynomial, small-backend check, or QNode path, the conditional QSVT
interpretation, and the omitted quantum components. Resource reports are proxy
summaries, not hardware estimates; benchmark reports time only the classical
baseline path and include `benchmark_environment` metadata for interpreting
timing snapshots.

## Scope

This project is intentionally educational, explicit, research-oriented, and
polynomial-focused. Its current execution layer is simulator-first, while
small auditable hardware experiments are a planned extension.

It does not aim to provide production-scale circuit optimization,
fault-tolerant constructions, amplitude amplification, or problem-specific
scalable state preparation methods. The emphasis is understanding how
polynomial transforms act on spectra and how finite QSVT circuits behave under
explicit simulator execution.

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
