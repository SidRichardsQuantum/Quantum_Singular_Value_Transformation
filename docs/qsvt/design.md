# Polynomial Design Helpers

The `qsvt.design` module provides higher-level utilities for constructing bounded polynomials for common Quantum Singular Value Transformation (QSVT) workflows.

These helpers sit one level above the low-level approximation utilities in `qsvt.approximation`. They are intended for users who want practical polynomial builders for tasks such as inverse-like transforms, sign approximations, spectral filters, and simple matrix-function experiments without having to manually derive coefficient sets each time.

## Overview

QSVT requires bounded polynomial transforms on the interval $[-1,1]$. In practice, many useful spectral transformations are naturally expressed as functions such as:

- $\mathrm{sign}(x)$
- $1/x$
- $\sqrt{x}$
- $x^\alpha$
- threshold-like filters

Notation used on this page:

- $x$ is the normalized scalar spectral variable, usually in `[-1, 1]`.
- $p(x)$ or $f(x)$ is the scalar target or polynomial response.
- `coeffs` is the returned coefficient array in ascending monomial order.
- `degree` is the highest requested polynomial degree.
- $\gamma$ is a positive gap or lower spectral cutoff.
- $a$ is a lower endpoint for positive-domain fits.
- $\alpha$ is an exponent for power-law targets on this page; it is distinct
  from block-encoding normalization, which is documented separately.

The role of `qsvt.design` is to provide small, readable constructors that return polynomial coefficients in the package’s standard format:

- NumPy arrays
- ascending monomial degree order
- bounded on $[-1,1]$

These polynomials can then be passed into other utilities in the package, such as scalar QSVT experiments, diagonal transforms, or embedded-vector simulations.

## Design philosophy

The module follows the same principles as the rest of the package:

- educational clarity first
- explicit polynomial outputs
- NumPy-only implementation
- no heavy optimisation framework
- practical bounded surrogates rather than overly abstract interfaces
- stable, reusable helpers for notebook and package use

These routines are not intended to produce provably optimal minimax approximants. Instead, they provide robust and readable bounded polynomial constructions suitable for experimentation, prototyping, and educational workflows.

## Available functions

## `design_inverse_polynomial`

```python
design_inverse_polynomial(gamma, degree)
```

Constructs a bounded odd inverse-like polynomial for use away from zero.

This is designed for the domain

$$
[-1,-\gamma] \cup [\gamma,1]
$$

where $0 < \gamma < 1$.

Here $\gamma$ is the minimum magnitude where the inverse-like approximation is
expected to be accurate; values closer to zero are deliberately not treated as
part of the inverse target.

Because QSVT-compatible polynomials must remain bounded on $[-1,1]$, this helper approximates the **normalised inverse profile**

$$
f(x) \approx \frac{\gamma}{x}
\quad \text{for } |x| \ge \gamma.
$$

Near the origin, the target is clipped to remain bounded.

### Why this normalisation?

The raw inverse $1/x$ is not bounded near $x=0$, so it is not directly suitable as a QSVT polynomial target on $[-1,1]$. The scaled form $\gamma/x$ is bounded by $1$ on the inverse-design domain, making it compatible with bounded polynomial construction.

If you need an approximation to $1/x$ itself, you can evaluate the resulting polynomial and divide by $\gamma$.

### Example

```python
import numpy as np
from qsvt.design import design_inverse_polynomial

gamma = 0.25
coeffs = design_inverse_polynomial(gamma=gamma, degree=11)

x = 0.5
value = np.polynomial.polynomial.polyval(x, coeffs)
approx_inverse = value / gamma
```

## `design_sign_polynomial`

```python
design_sign_polynomial(gamma, degree)
```

Constructs a bounded odd polynomial approximating the sign function away from zero.

The sign function is a central object in spectral transformations and can be used for tasks such as:

- spectral separation
- sign-based matrix functions
- projector construction
- threshold-like transforms

This helper uses a smooth bounded surrogate and fits a polynomial that remains bounded on $[-1,1]$.

### Target behaviour

The intended behaviour is approximately

$$
p(x) \approx \mathrm{sign}(x)
\quad \text{for } |x| \ge \gamma.
$$

The parameter $\gamma$ controls the width of the transition region around zero.

Here $\gamma$ is the excluded transition half-width: accuracy is judged away
from zero on $[-1,-\gamma] \cup [\gamma,1]$.

### Example

```python
import numpy as np
from qsvt.design import design_sign_polynomial

coeffs = design_sign_polynomial(gamma=0.2, degree=13)

xs = np.linspace(-1, 1, 200)
vals = np.polynomial.polynomial.polyval(xs, coeffs)
```

## `design_projector_polynomial`

```python
design_projector_polynomial(gamma, degree)
```

Constructs a projector-style bounded polynomial based on

$$
\frac{1 + \mathrm{sign}(x)}{2}.
$$

This is useful because, at the spectral level, the expression

$$
\frac{I + \mathrm{sign}(A)}{2}
$$

acts like a projector onto the positive spectral subspace of a Hermitian matrix $A$ when the spectrum is separated from zero.

Here $I$ is the identity matrix, $A$ is Hermitian, and $\mathrm{sign}(A)$
denotes the spectral sign function applied to the eigenvalues of $A$.

### Interpretation

If a sign-approximating polynomial $s(x)$ is available, then the corresponding projector-style polynomial is

$$
p(x) = \frac{1 + s(x)}{2}.
$$

This maps:

- negative values toward $0$
- positive values toward $1$

with a smooth transition around zero.

### Example

```python
import numpy as np
from qsvt.design import design_projector_polynomial

coeffs = design_projector_polynomial(gamma=0.2, degree=13)

xs = np.linspace(-1, 1, 200)
vals = np.polynomial.polynomial.polyval(xs, coeffs)
```

## `design_sqrt_polynomial`

```python
design_sqrt_polynomial(a, degree)
```

Constructs a bounded polynomial approximating

$$
\sqrt{x}
$$

on the interval

$$
[a,1], \qquad 0 \le a < 1.
$$

This is useful for matrix-function experiments involving positive semidefinite spectra or singular values bounded away from zero.

Here $a$ is the smallest normalized spectral value on which the square-root
approximation is intended to be accurate.

### Extension outside the positive interval

To preserve stable bounded behaviour on the full QSVT interval $[-1,1]$, the target is extended in a simple bounded way for non-positive inputs. This keeps the resulting polynomial usable in QSVT-style settings rather than only fitting well on a restricted interval and behaving poorly elsewhere.

### Example

```python
import numpy as np
from qsvt.design import design_sqrt_polynomial

coeffs = design_sqrt_polynomial(a=0.2, degree=12)

xs = np.linspace(0.2, 1.0, 100)
vals = np.polynomial.polynomial.polyval(xs, coeffs)
```

## `design_power_polynomial`

```python
design_power_polynomial(alpha, degree, a=0.0)
```

Constructs a bounded polynomial approximating

$$
x^\alpha
$$

on the interval

$$
[a,1].
$$

This provides a general positive-power builder for smooth spectral shaping tasks.

Here $\alpha$ is the real exponent in the target function $x^\alpha$, and $a$
is the lower endpoint of the positive fitting interval.

### Typical use cases

- square-root style transforms with $\alpha = 1/2$
- linear transforms with $\alpha = 1$
- compressive transforms with $0 < \alpha < 1$
- higher-order positive powers with $\alpha > 1$

Negative powers are intentionally not handled here; inverse-like behaviour should instead use `design_inverse_polynomial`.

### Example

```python
import numpy as np
from qsvt.design import design_power_polynomial

coeffs = design_power_polynomial(alpha=0.5, degree=12, a=0.2)

xs = np.linspace(0.2, 1.0, 100)
vals = np.polynomial.polynomial.polyval(xs, coeffs)
```

## `design_filter_polynomial`

```python
design_filter_polynomial(cutoff, degree)
```

Constructs a bounded even threshold-like filter polynomial.

This is intended for smooth spectral filtering behaviour, where one wants small-magnitude values suppressed and larger-magnitude values retained.

### Intended behaviour

The resulting polynomial behaves qualitatively like a smooth version of

$$
f(x) \approx
\begin{cases}
0, & |x| < \text{cutoff}, \\
1, & |x| > \text{cutoff}.
\end{cases}
$$

Because the construction depends on $|x|$, the designed polynomial is even.

Here `cutoff` is the normalized magnitude threshold separating suppressed and
retained spectral components.

### Use cases

- singular-value filtering
- soft thresholding
- smooth pass/reject windows
- denoising-style experiments
- building intuition for spectral filters before more specialised designs

### Example

```python
import numpy as np
from qsvt.design import design_filter_polynomial

coeffs = design_filter_polynomial(cutoff=0.45, degree=10)

xs = np.linspace(-1, 1, 200)
vals = np.polynomial.polynomial.polyval(xs, coeffs)
```

## `design_positive_inverse_polynomial`

```python
design_positive_inverse_polynomial(gamma, degree, extension="auto")
```

Constructs a bounded polynomial approximating

$$
f(x) \approx \frac{\gamma}{x}
\quad \text{for } x \in [\gamma, 1].
$$

This helper is intended for positive definite operators that have already been
rescaled so their spectra lie in `[gamma, 1]`. It is useful for finite
dimensional linear-system and PDE workflows, where the relevant operator has no
negative spectrum.

Here $\gamma$ is the scaled minimum eigenvalue or user-chosen positive lower
bound for the operator spectrum.

### Extension strategies

QSVT-compatible polynomial targets must be bounded on the full interval
`[-1, 1]`, even when the physical operator only uses the positive part of that
interval. The `extension` parameter controls how the positive-domain inverse
target is extended before fitting:

- `"even"` fits `gamma / max(|x|, gamma)`
- `"flat"` fits `1` for `x < gamma` and `gamma / x` for `x >= gamma`
- `"auto"` tries both and selects the lower sampled error on `[gamma, 1]`

The default is `"auto"`.

### Example

```python
import numpy as np

from qsvt.design import design_positive_inverse_polynomial

gamma = 0.1
coeffs = design_positive_inverse_polynomial(gamma=gamma, degree=30)

xs = np.linspace(gamma, 1.0, 100)
vals = np.polynomial.polynomial.polyval(xs, coeffs)
```

## Return format

All design helpers return:

- `numpy.ndarray`
- one-dimensional coefficient array
- ascending monomial degree order

So if

```python
coeffs = np.array([c0, c1, c2, c3])
```

then the polynomial is interpreted as

$$
p(x) = c_0 + c_1 x + c_2 x^2 + c_3 x^3.
$$

This matches the coefficient convention used throughout the package.

## Boundedness

A central requirement for QSVT-compatible scalar transforms is boundedness on $[-1,1]$:

$$
|p(x)| \le 1
\qquad \text{for all } x \in [-1,1].
$$

The functions in `qsvt.design` are built with this requirement in mind. Where necessary, fitted polynomials are rescaled so that they remain numerically bounded on the canonical interval.

This does not mean every polynomial is an optimal approximation to its target under a minimax criterion. It means the output is practical, stable, and suitable for QSVT-style bounded-transform experimentation.

## Parity structure

Some transformations naturally impose parity constraints.

### Odd designs

The following helpers return odd polynomials:

- `design_inverse_polynomial`
- `design_sign_polynomial`

These correspond to odd target functions.

### Even designs

The following helper returns an even polynomial:

- `design_filter_polynomial`
- `design_positive_inverse_polynomial` when `extension="even"` is selected

because the target depends on $|x|$.

### Mixed parity

The following helpers generally do not enforce a fixed parity:

- `design_projector_polynomial`
- `design_sqrt_polynomial`
- `design_power_polynomial`
- `design_positive_inverse_polynomial` when `extension="flat"` is selected

## Typical workflow

A common usage pattern is:

1. design a bounded polynomial
2. inspect or plot the scalar response
3. use the coefficients in a QSVT simulation utility

For example:

```python
import numpy as np

from qsvt.design import design_sign_polynomial
from qsvt.qsvt import qsvt_scalar_scan

coeffs = design_sign_polynomial(gamma=0.2, degree=13)

xs = np.linspace(-1, 1, 101)
# Then use coeffs in a downstream QSVT workflow
```

For diagonal test problems, these polynomials also work naturally with:

- `qsvt.classical_diagonal_polynomial_transform`
- `qsvt.compare_qsvt_vs_classical_diagonal`
- `qsvt.qsvt_diagonal_transform`

## Relationship to other modules

### `qsvt.approximation`

Use `qsvt.approximation` when you want low-level fitting utilities and direct control over approximation workflows.

Use `qsvt.design` when you want task-oriented builders for common spectral transforms.

### `qsvt.presets`

Use `qsvt.presets` when you want predefined coefficient families or notebook-friendly canned examples.

Use `qsvt.design` when you want a more direct construction API for practical polynomial generation from high-level intent.

### `qsvt.polynomials`

Use `qsvt.polynomials` for low-level polynomial manipulation, evaluation, parity checks, and boundedness checks.

The `qsvt.design` functions are intended to produce coefficient arrays that can be passed directly into those utilities.

## Diagnostics

The module also provides report helpers that return sampled approximation
quality data for each builder:

- `design_inverse_diagnostics`
- `design_positive_inverse_diagnostics`
- `design_sign_diagnostics`
- `design_projector_diagnostics`
- `design_sqrt_diagnostics`
- `design_power_diagnostics`
- `design_filter_diagnostics`
- `design_interval_projector_diagnostics`

Each report includes:

- fit error metrics
- boundedness margin
- sampled target and polynomial values
- the generated coefficient array

Example:

```python
from qsvt.design import design_sign_diagnostics
from qsvt.reports import save_report, save_report_plot

report = design_sign_diagnostics(gamma=0.2, degree=13)
print(report["max_error"], report["bounded_margin"])

save_report(report, "sign-report.json")
save_report_plot(report, "sign-report.png")
```

## Notes and limitations

These helpers are intentionally simple and readable.

They do **not** currently provide:

- minimax-optimal synthesis
- constrained optimisation over approximation error
- phase-angle synthesis for QSP/QSVT circuits
- automatic degree selection from an error tolerance
- rigorous approximation guarantees

Instead, they provide practical bounded polynomial surrogates that are useful for:

- educational exploration
- notebook experiments
- diagonal toy models
- small matrix-function studies
- quick iteration before more specialised constructions

## Summary

The `qsvt.design` module provides simple bounded polynomial builders for common QSVT tasks:

- inverse-like transforms
- sign approximations
- projector-style transforms
- square-root approximations
- positive-power transforms
- smooth spectral filters
- band-pass / interval projectors
- positive definite inverse-style transforms

The emphasis is on clarity, boundedness, and immediate usability in the package’s existing QSVT simulation and matrix-function workflows.
