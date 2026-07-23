# Experimental Repository Research Sweeps

The experimental `qsvt.research` layer turns a declarative experiment into a
deterministic Cartesian sweep. It is intended for studies where every trial,
including failures and skipped configurations, must remain independently
auditable.

This layer supports the repository's benchmarks and generated research
artifacts. It is not exported by `qsvt.stable`, is not a general
experiment-management framework, and may evolve with repository needs.
Statistical aggregation, standardized plots, and Pareto-front generation
remain research tooling rather than core QSVT API commitments.

The built-in `qsvt.research_frontier` study compares finite spectral accuracy
with encoding-aware logical resource estimates. It covers Poisson, transverse-
field Ising, and graph-Laplacian operators; inverse, projector, band-filter, and
resolvent targets; and embedding, FABLE, PrepSelPrep, and qubitization access
models.

## Run the built-in study

From a checkout, run the committed JSON definition:

```bash
qsvt research-sweep \
  --config examples/accuracy_resource_frontier.json \
  --output-dir results/research/accuracy_resource_frontier
```

Or use the built-in configuration directly:

```bash
qsvt accuracy-resource-frontier \
  --degrees 3,5,7 \
  --tolerances 0.2 \
  --output-dir results/research/accuracy_resource_frontier
```

Both commands resume by default. An existing completed, skipped, or failed
trial report is reused when its deterministic identifier is present in the
output directory. Pass
`--no-resume` to recompute all points or `--fail-fast` to raise the first
evaluator error instead of preserving it as a failed trial.

The output layout is:

```text
accuracy_resource_frontier/
├── sweep.json
├── manifest.json
├── summary.csv
├── frontier.json
├── frontier-manifest.json
├── pareto.csv
└── trials/
    └── <deterministic-trial-id>.json
```

`summary.csv` contains all trial factors and scalar evaluator metrics.
`pareto.csv` contains configurations not dominated simultaneously on operator
relative error, logical gates, and logical wires. The JSON files retain the
full operator, target, normalization, polynomial, resource, and truth-contract
details.

## Declarative format

A sweep definition has the versioned schema `qsvt-research-sweep-spec`:

```json
{
  "schema_name": "qsvt-research-sweep-spec",
  "schema_version": "1.0",
  "name": "small-frontier",
  "study": "accuracy-resource-frontier",
  "operators": [
    {
      "name": "poisson-1d-4",
      "kind": "poisson_1d",
      "parameters": {"size": 4}
    }
  ],
  "targets": [{"name": "inverse", "kind": "inverse"}],
  "access_models": ["embedding", "fable"],
  "degrees": [3, 5, 7],
  "tolerances": [0.2],
  "phase_solvers": ["root-finding"],
  "shots": [null],
  "seeds": [0],
  "noise_models": ["ideal"],
  "attempt_synthesis": false,
  "metadata": {"purpose": "small reproducible comparison"}
}
```

Every axis participates in the Cartesian product and in the deterministic
trial identity. Duplicate configurations are rejected. Operator and target
`parameters` are evaluator-specific; unknown top-level fields are rejected so
misspelled axes do not silently change a study.

JSON works with the base installation. YAML input and output require the
optional research extra:

```bash
pip install "qsvt-pennylane[research]"
```

## Python interface

Use `ResearchSweepSpec` for typed definitions and `run_research_sweep` with any
callable evaluator that returns a mapping:

```python
from qsvt import (
    ResearchOperatorSpec,
    ResearchSweepSpec,
    ResearchTargetSpec,
    run_research_sweep,
)

spec = ResearchSweepSpec(
    name="degree-study",
    study="custom-study",
    operators=(ResearchOperatorSpec("A", "matrix", {"matrix": [[1, 0], [0, 2]]}),),
    targets=(ResearchTargetSpec("inverse", "inverse"),),
    access_models=("embedding",),
    degrees=(3, 5, 7),
    tolerances=(0.1,),
)

def evaluate(trial):
    return {
        "status": "completed",
        "summary": {"requested_degree": trial.degree},
    }

result = run_research_sweep(spec, evaluate, output_dir="study-output")
```

The evaluator may return `completed`, `skipped`, or `failed`. Unexpected
exceptions become structured failed-trial reports unless `fail_fast=True`.
Use `load_research_sweep_spec`, `save_research_sweep_spec`,
`expand_research_sweep`, and `research_summary_rows` when a study needs custom
orchestration.

For the built-in frontier:

```python
from qsvt import accuracy_resource_frontier_spec, run_accuracy_resource_frontier

spec = accuracy_resource_frontier_spec(
    degrees=(3, 5, 7),
    tolerances=(0.2,),
)
result = run_accuracy_resource_frontier(spec, output_dir="study-output")
print(len(result.pareto_rows))
```

## Frontier semantics

The study diagonalizes each finite Hermitian operator for its classical
reference and designs the target polynomial on the signal domain induced by
the selected access model's normalization `alpha`. Changing an access model
can therefore change both approximation accuracy and logical resources; the
study does not pretend that normalization is a free implementation detail.

The reported logical resource fields come from
`estimate_encoding_aware_resources` and include signal-operator calls, inverse
signal-operator calls, wires, gates, and estimator model. They are estimates,
not provider-compiled circuits. Logical depth and success probability are
reported as `null` because the current estimator does not derive them and the
study does not execute application state preparation, postselection, or
amplitude amplification.

Complex resolvents are represented by separate real and imaginary polynomial
sequences. Their gate and signal-call figures are summed and their wire figure
is the maximum, while coherent combination overhead remains explicitly
omitted. Finite-shot or non-ideal noise axes are retained by the generic runner
but are marked `skipped` by this ideal frontier evaluator.

These boundaries make the study useful for hypothesis generation and
implementation comparison without turning logical proxies into runtime,
hardware, scalability, or quantum-advantage claims.
