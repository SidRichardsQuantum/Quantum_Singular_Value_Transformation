# Usage Guide

This guide shows how to use the `qsvt-pennylane` package for practical
Quantum Singular Value Transformation (QSVT) experiments.

The focus is on **bounded polynomial transforms** applied to:

- scalars
- diagonal matrices
- small Hermitian matrices

using PennyLane's QSVT implementation.

---

## Table of Contents

- [Core idea](#core-idea)

- [Installation](#installation)

- [Typical workflow](#typical-workflow)

  - [Step 1 — design or choose a polynomial](#step-1--design-or-choose-a-polynomial)
  - [Step 2 — inspect the scalar transform](#step-2--inspect-the-scalar-transform)
  - [Step 3 — apply the polynomial to a matrix](#step-3--apply-the-polynomial-to-a-matrix)
  - [Step 4 — compare classical vs QSVT output](#step-4--compare-classical-vs-qsvt-output)

- [Common tasks](#common-tasks)

  - [Sign transform](#sign-transform)
  - [Spectral projector](#spectral-projector)
  - [Inverse-like transform](#inverse-like-transform)
  - [Spectral filtering](#spectral-filtering)
  - [Matrix functions](#matrix-functions)

- [Module overview](#module-overview)

  - [qsvt.polynomials](#qsvtpolynomials)
  - [qsvt.approximation](#qsvtapproximation)
  - [qsvt.templates](#qsvttemplates)
  - [qsvt.design](#qsvtdesign)
  - [qsvt.matrices](#qsvtmatrices)
  - [qsvt.spectral](#qsvtspectral)
  - [qsvt.qsvt](#qsvtqsvt)

- [CLI usage](#cli-usage)

- [Recommended notebook path](#recommended-notebook-path)

- [Summary](#summary)

- [Author](#author)

- [License](#license)

---

## Core idea

QSVT implements a polynomial transformation

$$
A \;\rightarrow\; P(A)
$$

where:

- $A$ is a block-encoded operator with spectrum in $[-1,1]$
- $P(x)$ is a bounded polynomial
- eigenvectors are preserved
- eigenvalues are transformed

Most workflows therefore consist of:

1. designing a bounded polynomial
2. inspecting its scalar behaviour
3. applying it to a matrix
4. comparing classical vs QSVT output

---

## Installation

```bash
pip install qsvt-pennylane
```

---

## Typical workflow

### Step 1 — design or choose a polynomial

#### Option A — use a ready-made template

```python
from qsvt.templates import sign_approximation_polynomial

coeffs = sign_approximation_polynomial(
    degree=11,
    sharpness=8.0,
)
```

#### Option B — construct a polynomial for a task

```python
from qsvt.design import design_inverse_polynomial

coeffs = design_inverse_polynomial(
    gamma=0.25,
    degree=13,
)
```

#### Option C — fit a polynomial approximation directly

```python
from qsvt.approximation import chebyshev_fit_function

coeffs = chebyshev_fit_function(
    lambda x: x**2,
    degree=6,
)
```

---

### Step 2 — inspect the scalar transform

```python
import numpy as np

xs = np.linspace(-1,1,200)

values = np.polynomial.polynomial.polyval(xs, coeffs)
```

Useful checks:

```python
from qsvt.polynomials import polynomial_parity
from qsvt.polynomials import is_bounded_on_interval

polynomial_parity(coeffs)

is_bounded_on_interval(coeffs)
```

---

### Step 3 — apply the polynomial to a matrix

#### diagonal matrix

```python
from qsvt.matrices import diagonal_matrix
from qsvt.spectral import apply_function_to_hermitian

A = diagonal_matrix([0.9, 0.6, 0.3, 0.1])

P_A = apply_function_to_hermitian(
    A,
    lambda x: np.polynomial.polynomial.polyval(x, coeffs),
)
```

---

### Step 4 — compare classical vs QSVT output

```python
from qsvt.qsvt import qsvt_diagonal_transform

vals_qsvt = qsvt_diagonal_transform(
    [0.9,0.6,0.3,0.1],
    coeffs,
    encoding_wires=[0,1,2],
)
```

---

## Common tasks

### Sign transform

```python
from qsvt.design import design_sign_polynomial

coeffs = design_sign_polynomial(
    gamma=0.25,
    degree=13,
)
```

Produces:

$$
P(x) \approx \mathrm{sign}(x)
$$

useful for:

- spectral separation
- projectors
- thresholding

---

### Spectral projector

```python
from qsvt.design import design_projector_polynomial

coeffs = design_projector_polynomial(
    gamma=0.25,
    degree=13,
)
```

Approximates:

$$
\frac{1 + \mathrm{sign}(x)}{2}
$$

---

### Inverse-like transform

```python
from qsvt.design import design_inverse_polynomial

coeffs = design_inverse_polynomial(
    gamma=0.25,
    degree=13,
)
```

Approximates:

$$
\frac{\gamma}{x}
\quad \text{for } |x|\ge\gamma
$$

useful for:

- linear solver intuition
- pseudo-inverse behaviour

---

### Spectral filtering

```python
from qsvt.design import design_filter_polynomial

coeffs = design_filter_polynomial(
    cutoff=0.4,
    degree=12,
)
```

Suppresses small singular values while preserving larger ones.

---

### Matrix functions

```python
from qsvt.design import design_power_polynomial

coeffs = design_power_polynomial(
    alpha=0.5,
    degree=12,
)
```

Examples:

| function | alpha |
| -------- | ----- |
| sqrt     | 0.5   |
| square   | 2     |
| cube     | 3     |

---

## Module overview

### qsvt.polynomials

Basic polynomial utilities:

- Chebyshev polynomials
- degree
- parity
- boundedness checks

---

### qsvt.approximation

Chebyshev approximation helpers:

- function fitting
- approximation error metrics

---

### qsvt.templates

Ready-made bounded polynomial families:

- inverse-like
- sign approximation
- soft threshold filters
- sqrt approximations
- exponential weighting

Best for quick experiments.

---

### qsvt.design

Higher-level polynomial builders:

- inverse-like transforms
- sign polynomials
- projector polynomials
- sqrt approximations
- power transforms
- smooth filters

Best for structured workflows.

---

### qsvt.matrices

Small Hermitian test matrices:

- diagonal matrices
- rotated matrices
- involutory matrices

---

### qsvt.hamiltonians / qsvt.pde / qsvt.rescaling

Physics workflow helpers:

- small Hamiltonian constructors
- finite-difference PDE operators
- Hermitian spectral rescaling to QSVT-compatible intervals

Example:

```python
from qsvt.hamiltonians import tight_binding_chain
from qsvt.rescaling import rescale_hermitian_to_unit_interval

H = tight_binding_chain(8)
scaled = rescale_hermitian_to_unit_interval(H)
```

---

### qsvt.matrix_functions / qsvt.diagnostics

General builders and metrics for physics matrix-function workflows:

- real-time and imaginary-time evolution
- resolvents and Green's-function response
- Gaussian spectral windows
- state and operator errors
- ground-state overlap and spectral weights

---

### qsvt.spectral

Classical reference implementations:

- matrix powers
- matrix sign
- spectral transforms

Useful for validating QSVT behaviour.

---

### qsvt.qsvt

Thin wrappers around PennyLane QSVT:

- scalar transforms
- diagonal transforms
- classical vs QSVT comparisons

---

## CLI usage

Scalar example:

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

Non-diagonal Hermitian matrix report:

```bash
qsvt matrix-report \
  --matrix "0.31351701,-0.23499807;-0.23499807,0.68648299" \
  --poly "0,0,1"
```

Complex Hermitian matrix report:

```bash
qsvt matrix-report \
  --matrix "0,0.2j;-0.2j,0" \
  --poly "0,1"
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

When a report command is given `--output` or `--plot`, it writes the full
artifact to disk and prints a compact summary to stdout. Add `--print-report`
to also emit the full JSON payload on stdout.

---

## Recommended notebook path

1. scalar intuition
2. filtering
3. QSP polynomials
4. linear solvers
5. polynomial approximation
6. matrix functions
7. sign and projectors
8. reusable polynomial workflows

---

## Summary

Typical QSVT workflow:

1. choose a bounded polynomial
2. inspect scalar behaviour
3. apply to matrix spectrum
4. compare classical vs QSVT output

The package is designed to make each step explicit and easy to experiment with.

---

## Author

Sid Richards

GitHub: [https://github.com/SidRichardsQuantum](https://github.com/SidRichardsQuantum)

LinkedIn: [https://www.linkedin.com/in/sid-richards-21374b30b/](https://www.linkedin.com/in/sid-richards-21374b30b/)

---

## License

MIT License — see [LICENSE](LICENSE)
