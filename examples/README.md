# Cookbook Examples

These scripts are short package-client examples for common QSVT-style
workflows. They are intended to be copied into experiments or run from a local
checkout.

Run them from the repository root:

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

Each script writes a JSON report with machine-readable diagnostics. The
linear-system and encoding-aware resource examples also write compact CSV
summary tables.

The examples cover design/report basics, high-level finite problem workflows,
linear systems, spectral filtering, finite block encodings, PennyLane circuit
execution, specification-based block-encoding execution, and rectangular
singular-value execution. The three flagship scripts use the frozen
`qsvt.stable` facade and persist their versioned acceptance reports. Poisson
inversion and spectral filtering additionally perform
tolerance-driven degree search, phase synthesis, encoding-aware logical
resource estimation, finite QNode execution, and classical-reference checks.
Hamiltonian simulation coherently combines cosine and sine QSVT sequences,
executes the finite selector-LCU circuit, and reports full stated-scope
finite-QSVT acceptance with a concrete circuit-resource ledger.
The additional cookbook scripts show the same APIs in isolation: planning from
an accuracy target, supplying a custom circuit and signal projectors, checking
a credential-free finite-shot FABLE run against an ideal reference, and
comparing four access models for one logical operator. The finite-shot example
uses a seeded local simulator and does not claim real-hardware execution.

Compatibility and benchmark reporting already have direct CLI commands:

```bash
qsvt compatibility-report --poly "0.25,0.25"
qsvt benchmark cg-solve --matrix "4,1;1,3" --rhs "1,2"
qsvt research-sweep \
  --config examples/accuracy_resource_frontier.json \
  --output-dir /tmp/qsvt-accuracy-resource-frontier
```

`accuracy_resource_frontier.json` is a declarative research configuration, not
a Python cookbook script. It expands Poisson, Ising, and graph-Laplacian
operators across four target transforms, four access models, and three degrees.
The runner persists every deterministic trial plus aggregate summary, frontier,
and Pareto artifacts, and resumes matching trials by default.
