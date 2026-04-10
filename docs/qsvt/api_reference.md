# API Reference

This page documents the public Python API for the `qsvt-pennylane` package.

The package provides lightweight utilities for:

- polynomial and Chebyshev helpers
- bounded polynomial approximation
- small Hermitian matrix construction
- classical spectral matrix-function calculations
- explicit PennyLane QSVT wrappers

Install with:

```bash
pip install qsvt-pennylane
```

---

## Package overview

The package is organised into the following modules:

- `qsvt.polynomials`
- `qsvt.approximation`
- `qsvt.matrices`
- `qsvt.spectral`
- `qsvt.qsvt`

You can either import from submodules directly:

```python
from qsvt.qsvt import qsvt_scalar_output
from qsvt.polynomials import chebyshev_t
```

or import selected names from the package root:

```python
from qsvt import qsvt_scalar_output, chebyshev_t
```

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
\cos\theta & -\sin\theta \
\sin\theta & \cos\theta
\end{pmatrix}.
$$

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
A = V \operatorname{diag}(\lambda) V^T
\quad\Rightarrow\quad
f(A) = V \operatorname{diag}(f(\lambda)) V^T.
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

### `apply_qsvt_to_embedded_vector(operator, vector, poly, ...)`

Embed a logical vector into the enlarged Hilbert space, apply the full QSVT unitary, and extract the logical output.

This is useful for explicit linear-solver-style demonstrations.

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
