# Usage Guide

This guide shows how to use `qsvt-pennylane` for practical Quantum Singular
Value Transformation (QSVT) experiments.

The package is organized around one workflow:

1. choose or design a bounded polynomial,
2. inspect its scalar behavior,
3. apply it to a matrix spectrum,
4. compare the classical spectral transform with the QSVT output.

## Installation

```bash
pip install qsvt-pennylane
```

For local development:

```bash
git clone https://github.com/SidRichardsQuantum/Quantum_Singular_Value_Transformation.git
cd Quantum_Singular_Value_Transformation
pip install -e .
```

## Core Idea

QSVT implements a polynomial transformation

$$
A \rightarrow P(A)
$$

where `A` is a block-encoded operator with spectrum normalized to the QSVT
domain and `P(x)` is a bounded polynomial. In the Hermitian examples in this
repository, eigenvectors are preserved and eigenvalues are transformed.

## Typical Workflow

### 1. Choose Or Design A Polynomial

Use a ready-made template:

```python
from qsvt.templates import sign_approximation_polynomial

coeffs = sign_approximation_polynomial(
    degree=11,
    sharpness=8.0,
)
```

Construct a task-oriented polynomial:

```python
from qsvt.design import design_inverse_polynomial

coeffs = design_inverse_polynomial(
    gamma=0.25,
    degree=13,
)
```

Fit a polynomial approximation directly:

```python
from qsvt.approximation import chebyshev_fit_function

coeffs = chebyshev_fit_function(
    lambda x: x**2,
    degree=6,
)
```

### 2. Inspect The Scalar Transform

```python
import numpy as np

xs = np.linspace(-1, 1, 200)
values = np.polynomial.polynomial.polyval(xs, coeffs)
```

Useful checks:

```python
from qsvt.polynomials import is_bounded_on_interval, polynomial_parity

parity = polynomial_parity(coeffs)
bounded = is_bounded_on_interval(coeffs)
```

### 3. Apply The Polynomial To A Matrix

```python
from qsvt.matrices import diagonal_matrix
from qsvt.spectral import apply_function_to_hermitian

A = diagonal_matrix([0.9, 0.6, 0.3, 0.1])

P_A = apply_function_to_hermitian(
    A,
    lambda x: np.polynomial.polynomial.polyval(x, coeffs),
)
```

### 4. Compare Classical And QSVT Output

```python
from qsvt.qsvt import qsvt_diagonal_transform

vals_qsvt = qsvt_diagonal_transform(
    [0.9, 0.6, 0.3, 0.1],
    coeffs,
    encoding_wires=[0, 1, 2],
)
```

For a complete coefficient, diagnostic, and compatibility payload:

```python
from qsvt.workflow import design_workflow

result = design_workflow(
    kind="sign",
    gamma=0.2,
    degree=13,
)

coeffs = result.coeffs
report = result.as_report()
```

### Classify Realizability And Synthesize Phases

Classical polynomial evaluation, structural QSP/QSVT compatibility, and
numerical phase synthesis are separate layers:

```python
from qsvt import classify_polynomial_realizability, synthesize

classification = classify_polynomial_realizability([0.5, 0.5])
assert classification.requires_parity_decomposition

result = synthesize(
    [0.0, 1.0, 0.0, -0.5, 0.0, 1.0 / 3.0],
    routine="QSVT",
    angle_solver="root-finding",
)
```

Mixed-parity bounded polynomials are reported as requiring separate even and
odd sequences plus a combination mechanism such as LCU. Synthesis results keep
the generated phases, convention, solver, timing, reconstruction error, and
structured failure metadata.

```bash
qsvt phase-synthesis --poly "0,1,0,-0.5,0,0.333333"
```

See [docs/qsvt/synthesis.md](docs/qsvt/synthesis.md) for the full contract.

Use extrema-based boundedness checks when a sampled grid is not sufficient:

```python
from qsvt import certify_polynomial_boundedness

certificate = certify_polynomial_boundedness([0.996, 0.1, -0.5])
```

Compare angle solvers and synthesize mixed-parity components:

```python
from qsvt import benchmark_phase_solvers, synthesize_mixed_parity

benchmark = benchmark_phase_solvers(
    [0.0, 1.0],
    solvers=["root-finding", "iterative"],
)
mixed = synthesize_mixed_parity([0.5, 0.5])
```

```bash
qsvt boundedness-certificate --poly "0.996,0.1,-0.5"
qsvt phase-solver-benchmark --poly "0,1" --solvers root-finding --repeats 3
qsvt mixed-parity-synthesis --poly "0.5,0.5"
```

### Describe A Block-Encoding Access Model

```python
import numpy as np
from qsvt import matrix_block_encoding_spec

spec = matrix_block_encoding_spec(
    np.array([[0.2, 0.1, 0.0], [0.0, 0.3, 0.1]]),
    alpha=0.5,
)

print(spec.logical_shape)
print(spec.execution_supported)
print(spec.as_report()["lower_level_qsvt_supported"])
```

Rectangular matrices, sparse-like objects exposing `toarray()`, PennyLane
operators, and custom operation factories can be represented. The specification
reports whether the package can pass that source through PennyLane's high-level
QSVT helper. The report separately records lower-level execution support;
representation still does not imply compatibility with every device.

### Execute From A Block-Encoding Specification

```python
import pennylane as qml
from qsvt import (
    execute_qsvt_from_spec,
    pennylane_operator_block_encoding_spec,
)

operator = qml.dot([0.3, 0.7], [qml.Z(1), qml.X(1)])
spec = pennylane_operator_block_encoding_spec(
    operator,
    encoding_wires=[0],
    block_encoding="prepselprep",
)
result = execute_qsvt_from_spec(spec, [0.0, 1.0], [1.0, 0.0])

print(result.succeeded)
print(result.logical_output_relative_error)
print(result.resource_summary)
```

This path uses PennyLane's lower-level `qml.QSVT` operation. It supports
matrix, PrepSelPrep, qubitization, and custom-circuit specifications.
Rectangular matrix specifications are checked against a finite SVD reference.
Pass explicit `projectors` when a custom encoding needs a caller-defined signal
subspace convention. Backend failures are returned in `error_type` and `error`;
use `raise_on_failure=True` when exception behavior is preferred.

The equivalent matrix-specification CLI path is:

```bash
qsvt execute-spec --kind matrix \
  --matrix "0.2,0;0,0.8" \
  --poly "0,0,1" \
  --state "1,0" \
  --output matrix-execution.json
```

Execution reports use schema name `block-encoding-qsvt-execution` and schema
version `1.0`. Diagnostics separate real-output absolute/relative error,
complex leakage, logical-subspace leakage, probability and statevector
normalization error, and finite-shot standard errors.

## Cookbook Scripts

The repository includes short package-client scripts in
[`examples/`](examples/). They keep common workflows visible without requiring
a notebook:

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

The scripts cover polynomial design, matrix application, saved diagnostics,
linear-system comparison, threshold filtering, block-encoded QSVT checks,
PennyLane circuit execution, specification-based block-encoding execution,
compatibility reports, and benchmark summary export.

## Common Tasks

### Sign Transform

```python
from qsvt.design import design_sign_polynomial

coeffs = design_sign_polynomial(
    gamma=0.25,
    degree=13,
)
```

This produces a bounded odd polynomial with

$$
P(x) \approx \mathrm{sign}(x)
$$

away from the transition region around zero. It is useful for spectral
separation, thresholding, and projector construction.

### Spectral Projector

```python
from qsvt.design import design_projector_polynomial

coeffs = design_projector_polynomial(
    gamma=0.25,
    degree=13,
)
```

This approximates

$$
\frac{1 + \mathrm{sign}(x)}{2}.
$$

### Inverse-Like Transform

```python
from qsvt.design import design_inverse_polynomial

coeffs = design_inverse_polynomial(
    gamma=0.25,
    degree=13,
)
```

This approximates

$$
\frac{\gamma}{x}
\quad \text{for } |x| \ge \gamma.
$$

The scaling keeps the target bounded on the design interval.

### Spectral Filtering

```python
from qsvt.design import design_filter_polynomial

coeffs = design_filter_polynomial(
    cutoff=0.4,
    degree=12,
)
```

This suppresses small singular values while preserving larger ones.

### Matrix Functions

```python
from qsvt.design import design_power_polynomial

coeffs = design_power_polynomial(
    alpha=0.5,
    degree=12,
)
```

Examples:

| function | `alpha` |
| --- | ---: |
| square root | `0.5` |
| square | `2` |
| cube | `3` |

## Physics-Style Workflow

The physics helpers use the same bounded-polynomial pattern:

1. build a Hamiltonian or PDE operator,
2. rescale its spectrum,
3. design a bounded matrix-function polynomial,
4. validate against a classical spectral transform.

```python
from qsvt.hamiltonians import tight_binding_chain
from qsvt.matrix_functions import design_real_time_evolution_polynomials
from qsvt.rescaling import rescale_hermitian_to_unit_interval
from qsvt.spectral import apply_polynomial_to_hermitian

H = tight_binding_chain(8)
scaled = rescale_hermitian_to_unit_interval(H)

polys = design_real_time_evolution_polynomials(
    time=1.0,
    scale=scaled.scale,
    degree=19,
)

cos_Ht = apply_polynomial_to_hermitian(scaled.matrix, polys.cos_coeffs)
sin_Ht = apply_polynomial_to_hermitian(scaled.matrix, polys.sin_coeffs)
```

See [docs/qsvt/physics.md](docs/qsvt/physics.md) for the detailed physics API
map.

## Real Problem Workflow Example

For a small concrete physics or mathematics model, start with the dense
operator you want to study and use a workflow report as the reproducible
record. This example filters a tight-binding Hamiltonian toward its low-energy
subspace and keeps both the numerical diagnostics and the claim boundary.

```python
import numpy as np

from qsvt.algorithms import ground_state_filtering_workflow
from qsvt.hamiltonians import tight_binding_chain
from qsvt.reports import report_to_jsonable, save_report

H = tight_binding_chain(8)
trial_state = np.ones(H.shape[0])

result = ground_state_filtering_workflow(
    H,
    trial_state,
    degree=18,
    width=0.3,
    num_points=801,
)

report = report_to_jsonable(result.as_report())

print(report["ground_energy"])
print(report["ground_state_overlap"])
print(report["operator_relative_error"])
print(report["truth_contract"]["truth_status"])
print(report["truth_contract"]["is_end_to_end_quantum_algorithm"])

save_report(report, "tight-binding-filter-report.json")
```

Read the fields as follows:

- `operator_relative_error` and `reference_state_error` quantify the dense
  polynomial approximation for this finite instance.
- `ground_state_overlap` is the physics diagnostic for whether the filter
  emphasized the low-energy eigenspace.
- `truth_contract` states that the report validates the spectral-polynomial
  core and does not include block-encoding construction, state preparation,
  readout, amplitude amplification, or hardware costs.

For research benchmarking, pair this workflow with a resource proxy:

```python
from qsvt.resources import qsvt_resource_report

proxy = qsvt_resource_report(
    result.coeffs,
    matrix_dimension=H.shape[0],
    attempt_synthesis=False,
    diagnostics={
        "operator_relative_error": result.operator_relative_error,
        "ground_state_overlap": result.ground_state_overlap,
    },
)

print(proxy["resources"]["degree"])
print(proxy["resources"]["signal_operator_calls"])
print(proxy["truth_contract"]["truth_status"])
```

The proxy helps compare polynomial degree and signal-call requirements across
models or tolerances. It should be cited as a polynomial-resource proxy, not as
a full quantum runtime estimate.

## CLI Usage

Scalar transform:

```bash
qsvt scalar --x 0.5 --poly "0,0,1"
```

Diagonal transform:

```bash
qsvt diag \
  --values "1.0,0.7,0.3,0.1" \
  --poly "0,0,1" \
  --wires 3
```

Chebyshev evaluation:

```bash
qsvt cheb --degree 3 --x 0.5
```

Complete design workflow report:

```bash
qsvt design-workflow \
  --kind sign \
  --gamma 0.2 \
  --degree 13 \
  --output sign-workflow.json
```

Degree/error/boundedness sweep:

```bash
qsvt design-sweep \
  --kind sign \
  --degrees "5,9,13,17" \
  --gamma 0.2 \
  --no-synthesis \
  --output sign-degree-sweep.json
```

The sweep command writes a compact manifest with one row per degree, including
sampled maximum error, RMS error, boundedness margin, and compatibility-check
metadata. Use it when comparing approximation quality across candidate degrees
without opening a notebook.

Resource proxy report:

```bash
qsvt resource-report \
  --poly "0,0,1" \
  --matrix-dimension 4 \
  --no-synthesis
```

The resource report combines polynomial degree, coefficient count, QSP
phase-count proxy, signal-call proxy, optional encoding width, and compatibility
metadata. It is for comparing small workflows; it is not a hardware runtime or
fault-tolerant resource estimate.

Linear-system comparison report:

```bash
qsvt linear-system-compare \
  --matrix "2,0.25;0.25,1.25" \
  --rhs "1,-0.5" \
  --degree 8 \
  --no-synthesis \
  --no-qsvt \
  --output results/algorithms/linear_system_comparison.json \
  --rows-output results/tables/linear_system_comparison_summary.csv
```

The comparison command reports dense solve, optional conjugate gradients,
QSVT-style polynomial inverse, and optional PennyLane QSVT matrix-check rows.
The CSV path is a compact artifact table, not a timing benchmark.

Use `qsvt examples` to list workflow families, benchmark subcommands, and
compact command-line examples.

Classical benchmark report:

```bash
qsvt benchmark cg-solve \
  --matrix "4,1;1,3" \
  --rhs "1,2" \
  --tolerance 1e-10 \
  --qsvt-poly "0,1"
```

Benchmark subcommands include `eigh`, `dense-solve`, `cg-solve`,
`polynomial`, and `spectral-function`. They report classical baseline timings,
residuals or matrix-function metadata, and optional QSVT resource proxies.

Non-diagonal Hermitian matrix report:

```bash
qsvt matrix-report \
  --matrix "0.31351701,-0.23499807;-0.23499807,0.68648299" \
  --poly "0,0,1" \
  --output matrix-report.json
```

When a report command is given `--output` or `--plot`, it writes the full
artifact to disk and prints a compact summary to stdout. Add `--print-report`
to also emit the full JSON payload on stdout.

Compatibility reports distinguish bounded polynomial approximation from
PennyLane QSVT synthesis compatibility. Use
`python examples/compatibility_report.py --output /tmp/qsvt-compatibility.json`
for the shortest Python workflow, or use `qsvt design-workflow` and inspect the
compatibility payload in the generated report.

For release validation from a local checkout, run
`python scripts/release_check.py` to execute lint, formatting, type, fast test,
documentation, build, and distribution metadata checks.

## Where To Go Next

- [THEORY.md](THEORY.md): conceptual background
- [docs/qsvt/api_reference.md](docs/qsvt/api_reference.md): public API details
- [docs/qsvt/design.md](docs/qsvt/design.md): design helper reference
- [docs/qsvt/templates.md](docs/qsvt/templates.md): reusable template families
- [docs/qsvt/notebooks.md](docs/qsvt/notebooks.md): tutorial and real-example notebooks
- [RESULTS.md](RESULTS.md): reproducible report and plot conventions
