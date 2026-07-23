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
- Release checklist: [RELEASING.md](RELEASING.md)
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
- PennyLane >= 0.42, < 0.46
- NumPy >= 1.23, < 3

Plotting helpers are optional:

```bash
pip install "qsvt-pennylane[plot]"
```

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
from qsvt.stable import design_workflow

result = design_workflow(
    kind="sign",
    gamma=0.25,
    degree=13,
)

coeffs = result.coeffs
report = result.as_report()
```

Plan from a finite problem and a requested error instead of choosing a degree
manually:

```python
from qsvt.stable import QSVTProblemSpec, QSVTTransformSpec, plan_qsvt

plan = plan_qsvt(
    QSVTProblemSpec(np.diag([1.0, 2.0]), rhs=np.ones(2)),
    QSVTTransformSpec(
        "linear_system", tolerance=0.4, min_degree=3, max_degree=9
    ),
)
```

Run a finite problem workflow with a uniform report:

```python
import numpy as np
from qsvt import qsvt_problem_workflow

result = qsvt_problem_workflow(
    "linear_system",
    np.diag([1.0, 2.0]),
    rhs=np.array([1.0, 1.0]),
    degree=12,
)

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
qsvt problem-workflow --target linear_system --matrix "2,0;0,1" \
  --rhs "1,1" --degree 8 --no-synthesis --no-qsvt
qsvt plan-workflow --target linear_system --matrix "2,0;0,1" \
  --rhs "1,1" --tolerance 0.2 --no-execute
qsvt degree-search --kind sign --gamma 0.2 --tolerance 0.05
qsvt spectral-filter-qsvt --pauli-terms "0.4:ZI,0.3:IZ,0.2:XI" \
  --state "0.5,0.5,0.5,0.5" --lower -0.4 --upper 0.4 --tolerance 0.16
qsvt poisson-qsvt --n-points 4 --tolerance 0.4
qsvt hamiltonian-simulation --matrix "0,1;1,0" \
  --state "1,0" --time 0.5 --degree 8
qsvt research-sweep --config examples/accuracy_resource_frontier.json \
  --output-dir /tmp/qsvt-accuracy-resource-frontier
qsvt accuracy-resource-frontier --degrees 3,5,7 --tolerances 0.2 \
  --output-dir /tmp/qsvt-accuracy-resource-frontier
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
python examples/problem_workflow.py --output /tmp/qsvt-problem-workflow.json
python examples/threshold_filter.py --output /tmp/qsvt-threshold-filter.json
python examples/block_encoded_workflow.py \
  --output /tmp/qsvt-block-encoded-workflow.json
python examples/circuit_execution.py --output /tmp/qsvt-circuit-execution.json
python examples/block_encoding_execution.py \
  --output /tmp/qsvt-block-encoding-execution.json
python examples/rectangular_execution.py \
  --output /tmp/qsvt-rectangular-execution.json
python examples/spectral_filter_qsvt.py \
  --output /tmp/qsvt-spectral-filter.json
python examples/poisson_qsvt.py --output /tmp/qsvt-poisson.json
python examples/hamiltonian_simulation.py \
  --output /tmp/qsvt-hamiltonian-simulation.json
python examples/accuracy_driven_plan.py \
  --output /tmp/qsvt-accuracy-driven-plan.json
python examples/custom_block_encoding.py \
  --output /tmp/qsvt-custom-block-encoding.json
python examples/finite_shot_qsvt.py \
  --output /tmp/qsvt-finite-shot.json --shots 2000 --seed 12345
python examples/encoding_aware_resources.py \
  --output /tmp/qsvt-encoding-aware-resources.json \
  --rows-output /tmp/qsvt-encoding-aware-resources.csv
```

See [USAGE.md](USAGE.md) for full Python and CLI workflows.

## Package Map

The public package lives under `src/qsvt`.

| module | purpose |
| --- | --- |
| `qsvt.polynomials` | Chebyshev utilities, parity checks, boundedness checks |
| `qsvt.approximation` | polynomial fitting and approximation error helpers |
| `qsvt.design` | task-oriented and physics matrix-function polynomial builders |
| `qsvt.algorithms` | end-to-end simulator-scale algorithm workflows |
| `qsvt.block_encoding` | finite dense block-encoding construction and verification |
| `qsvt.execution` | single-sequence and coherent component-LCU QNode execution for block-encoding specifications |
| `qsvt.hardware` | experimental finite-shot execution on caller-supplied PennyLane devices with local preflight and decomposition reports |
| `qsvt.synthesis` | realizability classification, parity decomposition, and phase synthesis |
| `qsvt.presets` | ready-made named bounded polynomial families |
| `qsvt.workflow` | combined coefficient, diagnostic, compatibility, and high-level problem workflows |
| `qsvt.planning`, `qsvt.degree` | typed problem planning and target-error degree selection |
| `qsvt.flagship` | executable Pauli spectral-filter and Poisson workflows |
| `qsvt.acceptance` | versioned acceptance contracts for the three flagship workflows |
| `qsvt.stable` | frozen compact facade for the remainder of the 0.x series |
| `qsvt.research` | experimental repository tooling for deterministic and resumable research sweeps |
| `qsvt.research_frontier` | experimental repository accuracy/resource frontier tooling |
| `qsvt.reports` | JSON-safe reports, schema checks, and plot helpers |
| `qsvt.resources` | polynomial proxies and encoding-aware logical resource estimates |
| `qsvt.benchmarks` | classical baselines and QSVT-oriented benchmark summaries |
| `qsvt.comparisons` | experimental HHL and quantum-walk comparisons and tutorials |
| `qsvt.matrices` | small Hermitian test matrices |
| `qsvt.spectral` | classical spectral matrix-function references |
| `qsvt.qsvt` | PennyLane QSVT wrappers and comparisons |
| `qsvt.hamiltonians`, `qsvt.pde`, `qsvt.rescaling` | small helpers supporting QSVT examples and validation |
| `qsvt.diagnostics` | application-level validation metrics |

For detailed function-level documentation, use
[docs/qsvt/api_reference.md](docs/qsvt/api_reference.md).

The package includes a `py.typed` marker so type checkers can consume the
inline type annotations shipped with the public modules.

During the `0.x` series, new applications should import from `qsvt.stable`.
That module contains the frozen 20-name facade. `qsvt.api_status(name)` labels
other root exports as `compatibility` or `experimental`; compatibility names
remain importable and receive at least two minor releases of deprecation
warning before removal. See
[API stability](docs/qsvt/stability.md) for the exact list and policy.

## Roadmap

Finite coherent mixed-parity execution and all three flagship acceptance paths
are now implemented. The immediate priorities are phase-synthesis robustness,
block-encoding verification across access models, and hardening the coherent
execution/report contracts. Hardware execution, noise-aware planning, and
broader applications remain experimental or later work.

Physics and mathematics applications are thin clients of domain-general QSVT
interfaces, not domain libraries maintained by the core package. HHL and
quantum-walk implementations remain adjacent experimental comparisons.
Research sweeps and aggregate plotting support repository benchmarks and are
not part of `qsvt.stable`.

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
- [docs/qsvt/planning.md](docs/qsvt/planning.md): typed accuracy-driven
  planning, degree selection, and finite execution
- [docs/qsvt/flagship_workflows.md](docs/qsvt/flagship_workflows.md): executable
  Pauli spectral-filter and Poisson workflows plus the three-workflow
  acceptance matrix
- [docs/qsvt/stability.md](docs/qsvt/stability.md): compact stable facade,
  compatibility tier, and deprecation policy
- [docs/qsvt/research.md](docs/qsvt/research.md): experimental repository
  sweep tooling, deterministic artifacts, and the accuracy-resource frontier
- [docs/qsvt/block_encoding.md](docs/qsvt/block_encoding.md): finite dense
  block encodings, normalization, verification, and omitted oracle costs
- [docs/qsvt/compatibility.md](docs/qsvt/compatibility.md): QSVT boundedness,
  parity, synthesis checks, and common failure modes
- [docs/qsvt/presets.md](docs/qsvt/presets.md): named polynomial preset
  families
- [docs/qsvt/physics.md](docs/qsvt/physics.md): Hamiltonian, PDE, rescaling,
  and matrix-function workflows
- [docs/qsvt/implementation.md](docs/qsvt/implementation.md): implementation
  conventions, report serialization, and API status
- [docs/qsvt/notebooks.md](docs/qsvt/notebooks.md): tutorial, benchmark, and
  real-example notebook index

Current release: `0.2.21`

## Notebooks

Tutorial notebooks live in `notebooks/tutorials/` and introduce QSVT as
polynomial functional calculus, from scalar transforms through sign functions,
projectors, matrix functions, reusable design workflows, end-to-end algorithm
workflows, reproducible reports, degree/error tradeoff studies, and
resource-proxy limitations. The accuracy-driven planning tutorial connects a
requested error to degree search, phase synthesis, access-model selection,
logical resources, and finite circuit execution. The workflow tutorial and
three corresponding real examples also show the versioned flagship acceptance
checks and distinguish stated-scope acceptance from full-QSVT acceptance.

Real physics examples live in `notebooks/real_examples/`. The curated
eight-notebook gallery covers Poisson inversion, Hamiltonian simulation,
Green's functions, Ising filtering, electronic occupations, topological band
projectors, singular-value deblurring, and matrix-log graph entropy. Each
notebook identifies the physical system, QSVT implementation strategy, and
classical validation boundary.

Benchmark notebooks live in `notebooks/benchmarks/` and compare classical
linear-system, spectral, and polynomial matrix-function baselines against
QSVT-oriented resource proxies and their underlying assumptions. The
encoding-aware resource benchmark compares embedding, FABLE, PrepSelPrep, and
qubitization for the same logical operator.

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
- tolerance-driven planning from matrices, PennyLane operators, and
  block-encoding specifications,
- executable Pauli-LCU spectral filtering and finite-difference Poisson
  direct/CG/QSVT comparisons with component error ledgers,
- PennyLane QSVT block checks for supported small compatible polynomials,
- PennyLane QNode execution for finite QSVT circuits with state preparation,
  queued `qml.qsvt`, statevector/probability measurement, and circuit resource
  metadata,
- lower-level QSVT execution from dense, rectangular, PennyLane-operator, and
  custom-circuit block-encoding specifications, including caller-supplied
  signal projectors and structured backend failures,
- coherent even/odd and complex component-LCU execution using forward and
  adjoint QSVT sequences, selector ancillas, measured postselection
  probabilities, and component circuit-resource ledgers,
- finite coherent-QSVT Hamiltonian simulation with cosine/sine sequence
  combination and full stated-scope flagship acceptance,
- finite-shot QSVT execution on caller-supplied PennyLane devices with
  caller-supplied preparation circuits, local preflight checks, probability
  measurements, shot-noise uncertainty fields, logical resource summaries, and
  credential-free provider/fake-backend metadata capture,
- non-executing hardware circuit audit reports that expose logical and
  decomposed operation sequences plus unsupported-operation checks before
  spending shots,
- classical benchmark baselines plus QSVT-oriented proxy metadata.

Reported as assumptions or proxies:

- scalable block-encoding availability beyond the reported finite or logical
  access model,
- input-state preparation and data loading,
- measurement/readout and amplitude amplification,
- fault-tolerant synthesis, error correction, provider-native hardware
  compilation, and provider job management,
- end-to-end runtime or quantum advantage claims.

Hardware-oriented execution is now an experimental package layer for small
finite-shot circuits on caller-created PennyLane devices. It performs local
preflight checks before execution, records provider/fake-backend metadata when
devices expose it, checks advertised native operations and shot limits, and
records compilation fields explicitly. Provider credential management, paid
submission, native compilation, job persistence, calibration capture, and
mitigation remain outside the portable report schema.

Every high-level algorithm, direct QSVT comparison, resource, and benchmark
report includes a `truth_contract` field. Circuit execution reports separately
state when a QNode was actually executed. The fields state the implemented
dense-polynomial, small-backend check, or QNode path, the conditional QSVT
interpretation, and the omitted quantum components. Resource reports are proxy
summaries, not hardware estimates; benchmark reports time only the classical
baseline path and include `benchmark_environment` metadata for interpreting
timing snapshots.

Stable algorithm reports derive an `execution_tier` and per-polynomial
`polynomial_evidence` from the artifacts returned by that run. The evidence
records the design and QSVT certification domains, output prefactor,
extrema-based boundedness certificate, parity, single-sequence realizability,
and any parity-decomposition requirement. A successful backend call alone does
not promote a polynomial to the `qsvt_circuit` tier when the polynomial lacks a
QSVT realizability certificate.

New algorithm reports emit `qsvt-algorithm-workflow` schema `1.1`. Legacy
schema `1.0` reports remain loadable and can be migrated when their saved
polynomial coefficients are sufficient to derive the 1.1 evidence.

## Scope

This project is intentionally educational, explicit, research-oriented, and
polynomial-focused. Its hardware-oriented execution layer is for small
auditable finite-shot experiments, not production hardware optimization.

It does not aim to provide production-scale circuit optimization,
fault-tolerant constructions, amplitude amplification, or problem-specific
scalable state preparation methods. The emphasis is understanding how
polynomial transforms act on spectra and how finite QSVT circuits behave under
explicit simulator or caller-supplied device execution.

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
