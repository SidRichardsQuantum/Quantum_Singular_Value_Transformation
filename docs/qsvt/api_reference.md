# API Reference

This page documents the public Python API for the `qsvt-pennylane` package.

The package provides lightweight utilities for:

- polynomial and Chebyshev helpers
- bounded polynomial approximation
- small Hermitian matrix construction
- classical spectral matrix-function calculations
- explicit PennyLane QSVT wrappers
- QNode-based QSVT execution for finite instances

Install with:

```bash
pip install qsvt-pennylane
```

---

## Package overview

The package is organised into the following modules:

- `qsvt.polynomials`
- `qsvt.approximation`
- `qsvt.algorithms`
- `qsvt.block_encoding`
- `qsvt.synthesis`
- `qsvt.matrices`
- `qsvt.hamiltonians`
- `qsvt.pde`
- `qsvt.rescaling`
- `qsvt.matrix_functions`
- `qsvt.diagnostics`
- `qsvt.spectral`
- `qsvt.design`
- `qsvt.templates`
- `qsvt.workflow`
- `qsvt.reports`
- `qsvt.resources`
- `qsvt.benchmarks`
- `qsvt.api`
- `qsvt.operators`
- `qsvt.execution`
- `qsvt.hardware`
- `qsvt.diagonal`
- `qsvt.matrix`
- `qsvt.compatibility`
- `qsvt.qsvt` compatibility re-exports
- `qsvt.__main__`

You can either import from submodules directly:

```python
from qsvt.qsvt import qsvt_scalar_output
from qsvt.polynomials import chebyshev_t
```

or import selected names from the package root:

```python
from qsvt import qsvt_scalar_output, chebyshev_t
```

For the higher-level polynomial builders and ready-made templates, see:

- [Polynomial design helpers](design.md)
- [Polynomial templates](templates.md)
- [Algorithm notes](algorithms.md)
- [Physics workflows](physics.md)
- [Implementation notes](implementation.md)
- [Diagnostics reports](reports.md)
- [QSVT transform reports](qsvt_reports.md)

### `qsvt.api`

During the `0.x` series, exported names carry coarse stability labels:

```python
import qsvt

qsvt.api_status("design_workflow")
qsvt.api_status("execute_qsvt_circuit")
qsvt.api_status("execute_qsvt_from_spec")
qsvt.api_status("execute_qsvt_on_device")
qsvt.api_status("qsvt_hardware_circuit_report")
qsvt.api_status("qsvt_provider_plugin_report")
```

Workflow-level helpers, report/export utilities, and benchmark baselines are
the most stable user-facing surface. Lower-level circuit execution and backend
adapter helpers are marked experimental until the package approaches `1.0`.

### `qsvt.workflow`

Use `design_workflow` when you want coefficients, approximation diagnostics,
and a QSVT compatibility report from one call:

```python
from qsvt.workflow import design_workflow

result = design_workflow("sign", gamma=0.25, degree=13)
coeffs = result.coeffs
payload = result.as_report()
resource_payload = result.resource_report(matrix_dimension=4)
```

The same combined workflow report is available from the CLI:

```bash
qsvt design-workflow --kind sign --gamma 0.25 --degree 13 \
  --output sign-workflow.json
```

### `qsvt.synthesis`

Use `classify_polynomial_realizability` to distinguish classical-only,
single-sequence, and mixed-parity multi-sequence cases. Use `synthesize` or
`synthesize_phases` to obtain PennyLane phase angles and reconstruction
diagnostics:

```python
from qsvt import classify_polynomial_realizability, synthesize

classification = classify_polynomial_realizability([0.5, 0.5])
result = synthesize([0.0, 1.0])
```

Related workflow-level helpers are:

- `certify_polynomial_boundedness`
- `benchmark_phase_solvers`
- `synthesize_mixed_parity`

See [Phase synthesis](synthesis.md).

### `qsvt.algorithms`

Use `linear_system_workflow` for small positive-definite linear-system
experiments that combine inverse-polynomial design, rescaling, solution
estimates, residual diagnostics, and QSVT compatibility metadata:

```python
import numpy as np
from qsvt.algorithms import linear_system_workflow

result = linear_system_workflow(
    np.diag([1.0, 2.0]),
    np.array([1.0, 1.0]),
    degree=20,
    attempt_synthesis=False,
)

print(result.polynomial_solution)
print(result.polynomial_residual_norm)
```

Use `linear_system_comparison_workflow` when you want the same finite instance
reported as rows for dense solve, conjugate gradients, QSVT-style polynomial
inverse, and the optional PennyLane QSVT matrix check:

```python
from qsvt.algorithms import linear_system_comparison_workflow

comparison = linear_system_comparison_workflow(
    np.diag([1.0, 2.0]),
    np.array([1.0, 1.0]),
    degree=20,
    attempt_synthesis=False,
    apply_qsvt=False,
)

for row in comparison.as_report()["rows"]:
    print(row["solver"], row["residual_norm"])
```

Compact comparison rows can be written as CSV:

```python
from qsvt.algorithms import write_linear_system_comparison_csv

write_linear_system_comparison_csv(
    comparison,
    "results/tables/linear_system_comparison_summary.csv",
)
```

The module also includes simulator-scale physics workflows that wrap existing
matrix-function polynomial builders with exact spectral references:

- `block_encoded_qsvt_workflow`
- `ground_state_filtering_workflow`
- `hamiltonian_simulation_workflow`
- `quantum_walk_search_workflow`
- `resolvent_workflow`
- `spectral_density_workflow`
- `spectral_thresholding_workflow`
- `thermal_gibbs_workflow`

Each returns a frozen dataclass with numerical outputs, diagnostics, and an
`as_report()` helper.

Use `quantum_walk_search_workflow` for a finite continuous-time graph-search
example with exact marked-vertex probability and a polynomial phase
approximation at the best sampled time:

```python
import numpy as np
from qsvt.algorithms import quantum_walk_search_workflow

n_vertices = 4
adjacency = np.ones((n_vertices, n_vertices)) - np.eye(n_vertices)

result = quantum_walk_search_workflow(
    adjacency,
    marked_vertex=0,
    degree=14,
)

print(result.best_probability)
print(result.probability_error)
```

Use `spectral_thresholding_workflow` when you want a smooth QSVT-style
interval projector and an exact hard-projector reference:

```python
import numpy as np
from qsvt.algorithms import spectral_thresholding_workflow

matrix = np.diag([-0.8, -0.15, 0.2, 0.75])
state = np.array([0.1, 0.8, 0.5, 0.2])

result = spectral_thresholding_workflow(
    matrix,
    lower=-0.3,
    upper=0.3,
    degree=32,
    sharpness=18.0,
    state=state,
)

print(result.exact_rank)
print(result.polynomial_rank_proxy)
print(result.leakage_outside_interval)
```

The same workflow is available from the CLI:

```bash
qsvt threshold-workflow \
  --matrix="-0.8,0,0,0;0,-0.15,0,0;0,0,0.2,0;0,0,0,0.75" \
  --lower -0.3 \
  --upper 0.3 \
  --degree 32 \
  --state "0.1,0.8,0.5,0.2"
```

For workflow-level targets, rescaling conventions, diagnostics, and limitations,
see [Algorithm notes](algorithms.md).

---

### `qsvt.block_encoding`

Use `block_encode_matrix` when you need an explicit dense finite block encoding
whose top-left block is `A / alpha`:

```python
import numpy as np
from qsvt.block_encoding import block_encode_matrix, verify_block_encoding

encoding = block_encode_matrix(np.array([[2.0, 0.5], [0.5, 1.0]]))
verification = verify_block_encoding(encoding)

print(verification["block_encoding_verified"])
print(verification["unitary_verified"])
```

The higher-level `block_encoded_qsvt_workflow` combines this encoding check
with a small PennyLane QSVT transform for positive Hermitian signal operators.

Research-facing block-encoding specifications cover more access models:

- `matrix_block_encoding_spec`
- `pennylane_operator_block_encoding_spec`
- `circuit_block_encoding_spec`
- `build_block_encoding_operator`
- `qsvt_operator_from_block_encoding`

Rectangular matrix and custom-circuit specifications are retained even when
the package's high-level PennyLane QSVT adapter cannot execute them directly.

---

### `qsvt.resources`

Use `qsvt_resource_report` to compare small QSVT-style workflows by polynomial
degree, coefficient count, QSP phase-count proxy, signal-call proxy, optional
matrix-register width, and compatibility metadata:

```python
from qsvt.resources import qsvt_resource_report

report = qsvt_resource_report(
    [0.0, 0.0, 1.0],
    matrix_dimension=4,
    attempt_synthesis=False,
)
```

The same report is available from the CLI:

```bash
qsvt resource-report --poly "0,0,1" --matrix-dimension 4 --no-synthesis
```

These are proxy reports for simulator-scale comparison. They do not include
block-encoding construction, state preparation, error correction, compilation,
or hardware runtime costs.

For interpretation limits and omitted costs, see
[QSVT resource model](qsvt_resource_model.md).

---

### `qsvt.benchmarks`

Use `qsvt.benchmarks` to build classical baseline reports that can be compared
with QSVT resource proxies:

```python
import numpy as np
from qsvt.benchmarks import (
    conjugate_gradient_benchmark,
    dense_linear_solve_benchmark,
)

matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
rhs = np.array([1.0, 2.0])
coeffs = [0.0, 1.0]

dense = dense_linear_solve_benchmark(matrix, rhs, qsvt_coeffs=coeffs)
cg = conjugate_gradient_benchmark(matrix, rhs, qsvt_coeffs=coeffs)
```

Available baseline helpers include:

- `dense_eigendecomposition_benchmark`
- `dense_linear_solve_benchmark`
- `conjugate_gradient_benchmark`
- `polynomial_matrix_function_benchmark`
- `spectral_matrix_function_benchmark`
- `benchmark_summary_table`
- `write_benchmark_summary_csv`
- `plot_benchmark_timings`
- `plot_qsvt_proxy_resources`
- `linear_system_comparison_summary_table`
- `write_linear_system_comparison_csv`

The same baseline reports are available from the CLI:

```bash
qsvt benchmark cg-solve \
  --matrix "4,1;1,3" \
  --rhs "1,2" \
  --qsvt-poly "0,1"
```

The linear-system comparison workflow is also available from the CLI:

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

Benchmark reports are classical references and QSVT cost proxies. They are
intended to support advantage-oriented comparisons, not to claim end-to-end
quantum speedups.

For per-baseline assumptions and cost-model context, see
[Classical baseline details](classical_baselines.md).

---

## `qsvt.polynomials`

Utilities for working with Chebyshev polynomials and standard coefficient-form polynomials.

### `chebyshev_t(n, x)`

Evaluate the Chebyshev polynomial of the first kind:

$$
T_n(x) = \cos(n \arccos x).
$$

**Parameters**

- `n` (`int`): polynomial degree
- `x` (`float | np.ndarray`): evaluation point(s)

**Returns**

- scalar or NumPy array with the same shape as `x`

**Example**

```python
from qsvt.polynomials import chebyshev_t

value = chebyshev_t(3, 0.5)
print(value)  # -1.0
```

---

### `chebyshev_t3(x)`

Evaluate the cubic Chebyshev polynomial:

$$
T_3(x) = 4x^3 - 3x.
$$

**Example**

```python
from qsvt.polynomials import chebyshev_t3

print(chebyshev_t3(0.5))   # -1.0
print(chebyshev_t3(-0.5))  # 1.0
```

---

### `eval_polynomial(coeffs, x)`

Evaluate a polynomial with coefficients in ascending order:

$$
P(x) = c_0 + c_1 x + c_2 x^2 + \dots
$$

**Parameters**

- `coeffs`: iterable of coefficients
- `x`: scalar or array of inputs

**Example**

```python
from qsvt.polynomials import eval_polynomial

print(eval_polynomial([0, 0, 1], 0.5))  # 0.25
```

---

### `polynomial_degree(coeffs)`

Return the effective polynomial degree, ignoring trailing zeros.

---

### `polynomial_parity(coeffs)`

Classify a polynomial as:

- `"even"`
- `"odd"`
- `"mixed"`
- `"zero"`

**Example**

```python
from qsvt.polynomials import polynomial_parity

print(polynomial_parity([0, 0, 1]))  # even
print(polynomial_parity([0, 1]))     # odd
```

---

### `is_bounded_on_interval(coeffs, lower=-1.0, upper=1.0, bound=1.0, ...)`

Numerically check whether:

$$
|P(x)| \le \text{bound}
$$

on a sampled grid over an interval.

This is useful for quick QSVT-style boundedness checks.

---

### `normalize_coefficients(coeffs)`

Clean a coefficient list by zeroing tiny values and removing trailing zeros.

---

### `chebyshev_to_monomial(coeffs, domain=(-1.0, 1.0))`

Convert Chebyshev-basis coefficients to ascending monomial coefficients.

### `monomial_to_chebyshev(coeffs, domain=(-1.0, 1.0))`

Convert ascending monomial coefficients to Chebyshev-basis coefficients.

---

## `qsvt.approximation`

Helpers for constructing and evaluating Chebyshev approximations on bounded intervals.

### `scale_to_chebyshev_domain(x, domain)`

Map a physical interval `[a, b]` to `[-1, 1]`.

### `scale_from_chebyshev_domain(t, domain)`

Map from `[-1, 1]` back to `[a, b]`.

---

### `chebyshev_fit_function(func, degree, domain=(-1.0, 1.0), num_points=500)`

Fit a function with a Chebyshev polynomial over a bounded interval.

**Returns**

- Chebyshev-basis coefficients

**Example**

```python
import numpy as np
from qsvt.approximation import chebyshev_fit_function

coeffs = chebyshev_fit_function(np.sqrt, degree=6, domain=(0.2, 1.0))
```

---

### `chebyshev_eval(coeffs, x, domain=(-1.0, 1.0))`

Evaluate a Chebyshev approximation on its physical interval.

---

### `chebyshev_approximant(coeffs, domain=(-1.0, 1.0))`

Construct a callable approximation function from Chebyshev coefficients.

**Example**

```python
import numpy as np
from qsvt.approximation import chebyshev_fit_function, chebyshev_approximant

coeffs = chebyshev_fit_function(np.sqrt, degree=6, domain=(0.2, 1.0))
P = chebyshev_approximant(coeffs, domain=(0.2, 1.0))

print(P(0.5))
```

---

### `max_error(func, approx, domain=(-1.0, 1.0), num_points=1000)`

Compute the maximum sampled absolute error on a grid.

---

### `rms_error(func, approx, domain=(-1.0, 1.0), num_points=1000)`

Compute the RMS approximation error on a grid.

---

### `fit_and_build_approximant(func, degree, domain=(-1.0, 1.0), num_points=500)`

Convenience function returning both:

- approximation coefficients
- callable approximation

---

### `sample_approximation(func, approx, domain=(-1.0, 1.0), num_points=400)`

Sample the true function and approximation on a shared grid.

Useful for plotting.

---

### `approximation_quality_report(func, approx, domain=(-1.0, 1.0), num_points=1000, bound=1.0, ...)`

Build a compact sampled report with:

- fit error metrics
- boundedness margin
- target vs polynomial sample arrays

This is the shared reporting helper used by the `qsvt.design` and
`qsvt.templates` diagnostics functions.

For JSON serialization, saving, loading, and plotting, see `qsvt.reports`.

### Report fields

| field | meaning |
| --- | --- |
| `mode` | CLI wrapper label for the selected report command |
| `kind` | The chosen report kind, such as `sign` or `inverse` |
| `builder` | Underlying builder function name |
| `fit_domain` | Interval used for fit-error sampling |
| `bounded_domain` | Interval used for boundedness sampling |
| `max_error` | Maximum sampled absolute error on the fit domain |
| `rms_error` | Root-mean-square sampled error on the fit domain |
| `bounded_margin` | `bound - max_abs_value` on the boundedness domain |
| `is_bounded` | Whether the sampled values stayed within the bound |
| `xs`, `target_values`, `polynomial_values`, `errors` | Fit-domain sample arrays |
| `bounded_xs`, `bounded_polynomial_values` | Bounded-domain sample arrays |
| `coeffs` | Generated coefficient array |

---

## `qsvt.reports`

Helpers for converting diagnostics reports into reusable artifacts.

### `report_to_jsonable(report)`

Convert a diagnostics report containing NumPy arrays/scalars into plain Python
containers that can be passed to `json.dumps`.

---

### `save_report(report, path, indent=2)`

Write a diagnostics report to JSON.

---

### `load_report(path)`

Load a JSON diagnostics report from disk.

---

### `plot_approximation_report(report, ax=None)`

Plot the sampled target values, polynomial values, and error curve from a
diagnostics report.

Returns a Matplotlib `(fig, axes)` pair.

---

### `save_report_plot(report, path, dpi=150)`

Save a target-vs-polynomial diagnostics plot to an image file.

---

## `qsvt.matrices`

Small matrix constructors for explicit QSVT and spectral demos.

### `diagonal_matrix(values)`

Construct a diagonal matrix from a list of entries.

---

### `identity(n)`

Construct the `n x n` identity matrix.

---

### `pauli_x()`

Return the Pauli-$X$ matrix.

---

### `pauli_z()`

Return the Pauli-$Z$ matrix.

---

### `rotation(theta)`

Construct the real 2D rotation matrix:

$$
R(\theta) =
\begin{pmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{pmatrix}.
$$

Here $\theta$ is the rotation angle in radians.

**Example**

```python
from qsvt.matrices import rotation

R = rotation(0.6)
```

---

### `rotated_diagonal(eigenvalues, theta)`

Construct a symmetric matrix with known eigenvalues:

$$
A = R(\theta),\mathrm{diag}(\lambda),R(\theta)^T.
$$

This is useful for generating small Hermitian matrices with non-trivial eigenvectors.

---

### `hermitian_from_eigendecomposition(eigenvalues, eigenvectors)`

Reconstruct a Hermitian matrix from its spectral data.

---

### `involutory_diagonal(sign_pattern)`

Construct a diagonal involutory matrix with entries `±1`.

These satisfy:

$$
A^2 = I.
$$

---

### `normalized_vector(values)`

Return a unit-norm version of a vector.

---

### `embed_vector(vec, dimension)`

Embed a vector into a larger Hilbert space by padding with zeros.

---

## `qsvt.spectral`

Classical matrix-function helpers based on eigendecomposition.

### `eigh_hermitian(matrix)`

Compute the eigendecomposition of a Hermitian matrix.

Returns:

- eigenvalues
- eigenvectors

---

### `matrix_from_eigendecomposition(eigenvalues, eigenvectors)`

Reconstruct a Hermitian matrix from its eigendecomposition.

---

### `apply_function_to_hermitian(matrix, func)`

Apply a scalar function spectrally:

$$
A = V \operatorname{diag}(\lambda) V^\dagger
\quad\Rightarrow\quad
f(A) = V \operatorname{diag}(f(\lambda)) V^\dagger.
$$

**Example**

```python
import numpy as np
from qsvt.spectral import apply_function_to_hermitian

A = np.diag([0.2, 0.8])
A2 = apply_function_to_hermitian(A, lambda x: x**2)
```

---

### `apply_polynomial_to_hermitian(matrix, coeffs)`

Apply a standard coefficient-form polynomial to a Hermitian matrix.

---

### `matrix_power_spectral(matrix, power)`

Compute an integer power spectrally.

---

### `matrix_fractional_power(matrix, power, require_nonnegative_spectrum=True)`

Compute a fractional matrix power via the eigenspectrum.

---

### `matrix_square_root(matrix)`

Compute the principal square root of a positive semidefinite Hermitian matrix.

---

### `matrix_sign(matrix, zero_tol=1e-12)`

Compute the matrix sign function.

---

### `spectral_projector_positive(matrix, zero_tol=1e-12)`

Construct the projector onto the positive-eigenvalue subspace.

---

### `spectral_projector_negative(matrix, zero_tol=1e-12)`

Construct the projector onto the negative-eigenvalue subspace.

---

### `positive_projector_from_sign(matrix, zero_tol=1e-12)`

Construct the positive projector using:

$$
\Pi_+ = \frac{I + \mathrm{sgn}(A)}{2}.
$$

---

### `negative_projector_from_sign(matrix, zero_tol=1e-12)`

Construct the negative projector using:

$$
\Pi_- = \frac{I - \mathrm{sgn}(A)}{2}.
$$

---

### `transformed_eigenvalues(matrix, func)`

Apply a scalar function directly to the eigenvalues of a Hermitian matrix.

---

## `qsvt.qsvt`

Thin PennyLane-facing wrappers for explicit QSVT calculations.

### `qsvt_operator(operator, poly, encoding_wires=None, block_encoding="embedding")`

Construct the PennyLane `qml.qsvt(...)` operator.

---

### `qsvt_unitary(operator, poly, encoding_wires=None, wire_order=None, block_encoding="embedding")`

Extract the explicit matrix representation of a QSVT transform using `qml.matrix(...)`.

**Example**

```python
from qsvt.qsvt import qsvt_unitary

U = qsvt_unitary(0.5, [0, 0, 1], encoding_wires=[0])
print(U)
```

---

### `qsvt_top_left_block(operator, poly, ...)`

For a matrix input, extract the logical top-left block of the full QSVT unitary.

This is the key helper for explicit small-scale examples.

---

### `qsvt_scalar_output(x, poly, ...)`

Apply QSVT to a scalar and return the top-left matrix element.

This is the scalar-QSP-style helper used throughout the introductory notebooks.

**Example**

```python
from qsvt.qsvt import qsvt_scalar_output

out = qsvt_scalar_output(0.5, [0, 0, 1], encoding_wires=[0])
print(out)  # ~0.25
```

---

### `qsvt_scalar_scan(xs, poly, ...)`

Evaluate scalar QSVT outputs over many scalar inputs.

Useful for plotting QSVT curves against classical polynomial curves.

---

### `qsvt_diagonal_transform(diagonal, poly, ...)`

Apply QSVT to a diagonal matrix and return the transformed diagonal entries.

**Example**

```python
from qsvt.qsvt import qsvt_diagonal_transform

vals = qsvt_diagonal_transform(
    [1.0, 0.7, 0.3, 0.1],
    [0, 0, 1],
    encoding_wires=[0, 1, 2],
)
print(vals)
```

---

### `qsvt_matrix_transform(operator, poly, ...)`

Apply QSVT to a small Hermitian matrix and return the logical matrix block.

When `real_output=True`, this returns the real part of the extracted block.
This is the quantity compared against the classical reference in the standard
real-symmetric report path.

**Example**

```python
from qsvt.matrices import rotated_diagonal
from qsvt.qsvt import qsvt_matrix_transform

A = rotated_diagonal([0.2, 0.8], theta=0.45)
block = qsvt_matrix_transform(A, [0, 0, 1])
print(block)
```

---

### `apply_qsvt_to_embedded_vector(operator, vector, poly, ...)`

Embed a logical vector into the enlarged Hilbert space, apply the full QSVT unitary, and extract the logical output.

This is useful for explicit linear-solver-style demonstrations.

---

### `execute_qsvt_circuit(operator, poly, state, ...)`

Execute a finite QSVT circuit through a PennyLane QNode.

This path prepares the supplied logical state in the first amplitudes of the
QNode register, queues `qml.qsvt`, and measures either:

- a statevector plus exact probabilities when `shots=None`,
- sampled probabilities when `shots` is finite.

The result includes `execution_kind`, `resource_summary`,
`logical_success_probability`, and a dense classical polynomial output used
only as a validation reference. The helper does not call `qml.matrix`
internally.

Example:

```python
import numpy as np
from qsvt.qsvt import execute_qsvt_circuit

A = np.diag([0.2, 0.8])
result = execute_qsvt_circuit(
    A,
    [0, 0, 1],
    [1.0, 0.0],
    encoding_wires=[0, 1],
)

print(result.execution_kind)
print(result.logical_success_probability)
print(result.resource_summary["gate_types"])
```

Use `result.as_report()` when you need a machine-readable truth contract for
the circuit execution layer.

---

### `execute_qsvt_from_spec(spec, poly, state, ...)`

Execute QSVT from a `BlockEncodingSpec` using PennyLane's lower-level
`qml.QSVT` operation.

The helper supports matrix, rectangular matrix, PrepSelPrep, qubitization, and
custom-circuit specifications. It synthesizes projector phases from `poly` by
default or accepts explicit `projectors` for caller-defined signal subspaces.
The returned `BlockEncodingQSVTExecutionResult` contains:

- structured success or backend-failure data,
- statevector or finite-shot probabilities,
- logical output and success probability,
- dense spectral or SVD validation where available,
- normalization, wire, phase, signal-call, gate, depth, and shot resources.

Reports use schema name `block-encoding-qsvt-execution`, schema version `1.0`,
and separate real-output error, complex leakage, logical-subspace leakage,
normalization error, and finite-shot uncertainty fields.

```python
import pennylane as qml
from qsvt import (
    execute_qsvt_from_spec,
    pennylane_operator_block_encoding_spec,
)

H = qml.dot([0.3, 0.7], [qml.Z(1), qml.X(1)])
spec = pennylane_operator_block_encoding_spec(H, encoding_wires=[0])
result = execute_qsvt_from_spec(spec, [0.0, 1.0], [1.0, 0.0])

print(result.succeeded)
print(result.resource_summary["block_encoding_method"])
```

The matrix-specification CLI equivalent is:

```bash
qsvt execute-spec --kind matrix --matrix "0.2,0;0,0.8" \
  --poly "0,0,1" --state "1,0"
```

---

### `execute_qsvt_on_device(spec, poly, preparation, device, ...)`

Execute finite-shot QSVT on a caller-supplied PennyLane device.

This experimental hardware-oriented path accepts an existing device and a
caller-supplied preparation function. It runs `qsvt_hardware_preflight` before
execution, requires a positive finite shot count, queues lower-level `qml.QSVT`,
and returns probabilities only. It never requests device statevectors.

By default, preflight rejects `StatePrep`-style preparation so hardware examples
use explicit preparation circuits. Reports use schema name
`hardware-qsvt-execution`, schema version `1.0`, and include logical resource
fields, finite-shot uncertainty, preflight details, and explicit placeholders
for provider-native compilation metadata.

```python
import numpy as np
import pennylane as qml
from qsvt import execute_qsvt_on_device, matrix_block_encoding_spec

spec = matrix_block_encoding_spec(np.diag([0.2, 0.8]), alpha=1.0)
device = qml.device("default.qubit", wires=spec.encoding_wires)

def prepare_zero():
    return None

result = execute_qsvt_on_device(
    spec,
    [0.0, 0.0, 1.0],
    prepare_zero,
    device,
    shots=200,
)

print(result.preflight.passed)
print(result.logical_success_probability)
print(result.resource_summary["compilation_status"])
```

Provider credentials, paid submission limits, job persistence, calibration
capture, mitigation, and native provider compilation remain outside this
portable helper.

---

### `qsvt_hardware_circuit_report(spec, poly, preparation, device, ...)`

Build a non-executing hardware circuit audit report.

The report constructs the same logical QSVT tape used by
`execute_qsvt_on_device`, attempts PennyLane decomposition, and compares both
logical and decomposed operation sequences with the device's advertised native
operations. It does not execute a QNode and does not submit a provider job.

```python
from qsvt import qsvt_hardware_circuit_report

circuit = qsvt_hardware_circuit_report(
    spec,
    [0.0, 1.0],
    prepare_zero,
    device,
    shots=200,
)

print(circuit.logical_operations)
print(circuit.decomposed_operations)
print(circuit.unsupported_logical_operations)
print(circuit.unsupported_decomposed_operations)
```

Reports use schema name `hardware-qsvt-circuit`, schema version `1.0`, and
include provider/fake-backend metadata, preflight data, wire order, operation
sequences, resource summaries, and decomposition status.

---

### `qsvt_provider_plugin_report(device, ...)`

Collect credential-free provider, backend, plugin, native-gate, fake-backend,
and shot-limit metadata from a PennyLane device.

The helper is duck-typed: it reads device attributes and `capabilities()`
mappings when available, and it records optional package versions only if those
packages are installed. This lets users validate provider-shaped fake backends
in local tests and CI without adding live credentials.

```python
import pennylane as qml
from qsvt import qsvt_provider_plugin_report

device = qml.device("default.qubit", wires=[0, 1])
device.provider_name = "ExampleProvider"
device.backend_name = "fake_two_qubit_backend"
device.native_gate_set = ("QSVT",)
device.max_shots = 500
device.is_fake_backend = True

report = qsvt_provider_plugin_report(device)
print(report.as_report())
```

The same provider report is embedded in hardware preflight and execution
reports. If native operations or shot limits are advertised, preflight uses
them to reject incompatible circuits before execution.

---

### `classical_diagonal_polynomial_transform(diagonal, poly)`

Apply a polynomial classically to a list of diagonal entries.

---

### `compare_qsvt_vs_classical_diagonal(diagonal, poly, ...)`

Return a comparison dictionary containing:

- input values
- QSVT outputs
- classical outputs
- absolute error

This is useful for smoke tests and validation.

---

### `compare_qsvt_vs_classical_matrix(operator, poly, ...)`

Return a comparison dictionary for a Hermitian matrix containing:

- input matrix
- QSVT real block
- QSVT imaginary block
- classical spectral polynomial matrix
- absolute error

This is useful for validating non-diagonal block-encoded examples.

---

### `qsvt_compatibility_report(poly, ...)`

Check whether polynomial coefficients are structurally suitable for PennyLane
QSVT synthesis.

The report includes:

- coefficient finiteness
- parity classification
- extrema-based boundedness on `[-1, 1]`
- optional PennyLane synthesis status
- structured failure reasons such as `mixed_parity`, `out_of_bounds`, and
  `synthesis_failed`

Example:

```python
from qsvt.qsvt import qsvt_compatibility_report

report = qsvt_compatibility_report([0, 0, 1])
print(report["compatible"], report["reasons"])
```

---

### `qsvt_transform_report(diagonal, poly, ...)`

Build a QSVT-vs-classical report for a diagonal transform.

The report includes:

- input values and polynomial coefficients
- QSVT output and direct classical output
- absolute error, max error, and RMS error
- `qsvt_succeeded` plus synthesis error details when requested by callers
- encoding wires, wire order, block-encoding mode, and dimension metadata

Example:

```python
from qsvt.qsvt import qsvt_transform_report

report = qsvt_transform_report(
    [1.0, 0.7, 0.3, 0.1],
    [0, 0, 1],
    encoding_wires=[0, 1, 2],
)

print(report["max_error"])
```

---

### `qsvt_matrix_transform_report(operator, poly, ...)`

Build a QSVT-vs-classical report for a non-diagonal Hermitian matrix.

The report includes:

- input matrix, eigenvalues, and polynomial coefficients
- real and imaginary parts of the extracted QSVT logical block
- classical spectral polynomial matrix `P(A)`
- absolute error, max error, RMS error, and Frobenius error
- maximum absolute imaginary block entry
- `qsvt_succeeded` plus synthesis error details when requested by callers
- encoding wires, wire order, block-encoding mode, and dimension metadata

Example:

```python
from qsvt.matrices import rotated_diagonal
from qsvt.qsvt import qsvt_matrix_transform_report

A = rotated_diagonal([0.2, 0.8], theta=0.45)
report = qsvt_matrix_transform_report(A, [0, 0, 1])

print(report["max_error"], report["max_imag_abs"])
```

---

## Minimal example

```python
import numpy as np
from qsvt.qsvt import qsvt_scalar_output, qsvt_diagonal_transform
from qsvt.polynomials import chebyshev_t

print(qsvt_scalar_output(0.5, [0, 0, 1], encoding_wires=[0]))

vals = qsvt_diagonal_transform(
    [1.0, 0.7, 0.3, 0.1],
    [0, 0, 1],
    encoding_wires=[0, 1, 2],
)
print(vals)

print(chebyshev_t(3, 0.5))
```

---

## Notes

- The package is designed for **educational and small-scale explicit use**.
- Most QSVT examples use `block_encoding="embedding"`.
- The API is intentionally lightweight and close to the corresponding notebook logic.
- For conceptual background, see the notebooks and [THEORY.md](../../THEORY.md).
