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
PennyLane QSVT synthesis compatibility.

## Where To Go Next

- [THEORY.md](THEORY.md): conceptual background
- [docs/qsvt/api_reference.md](docs/qsvt/api_reference.md): public API details
- [docs/qsvt/design.md](docs/qsvt/design.md): design helper reference
- [docs/qsvt/templates.md](docs/qsvt/templates.md): reusable template families
- [docs/qsvt/notebooks.md](docs/qsvt/notebooks.md): tutorial and real-example notebooks
- [RESULTS.md](RESULTS.md): reproducible report and plot conventions
