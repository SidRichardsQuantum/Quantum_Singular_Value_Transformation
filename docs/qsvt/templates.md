# Polynomial Templates

The `qsvt.templates` module provides ready-to-use bounded polynomial families for common Quantum Singular Value Transformation (QSVT) and Quantum Signal Processing (QSP) style experiments. It is intended as a practical starting point when you want a useful polynomial quickly, without setting up a full approximation workflow by hand. :contentReference[oaicite:0]{index=0}

These templates are especially useful for:

- singular-value filtering
- inverse-like transforms
- sign-style spectral separation
- square-root style matrix-function surrogates
- smooth exponential weighting experiments

They are designed to remain lightweight, readable, and notebook-friendly, while still producing bounded coefficient arrays that work naturally with the rest of the package. :contentReference[oaicite:1]{index=1}

## Overview

In many QSVT workflows, the goal is not to derive a best-possible polynomial from scratch, but to obtain a small bounded polynomial with the right qualitative behaviour. The `qsvt.templates` module provides exactly that: a collection of simple polynomial families that are easy to inspect, evaluate, and reuse. :contentReference[oaicite:2]{index=2}

All template builders return:

- `numpy.ndarray`
- one-dimensional coefficient arrays
- coefficients in ascending monomial degree order
- numerically bounded outputs on $[-1,1]$ :contentReference[oaicite:3]{index=3}

This makes them immediately compatible with the rest of the package’s polynomial and QSVT simulation utilities.

## Design philosophy

The templates module follows the same overall design principles as the package:

- educational clarity first
- explicit coefficient-form outputs
- ascending degree ordering
- boundedness on $[-1,1]$
- lightweight fitting rather than heavy optimisation
- implementations simple enough to read directly in notebooks or source form :contentReference[oaicite:4]{index=4}

These templates are not intended to be minimax-optimal constructions. They are practical, readable starting points for prototyping and intuition building before moving to more specialised polynomial design workflows. :contentReference[oaicite:5]{index=5}

## Available functions

## `inverse_like_polynomial`

```python
inverse_like_polynomial(degree, mu=0.25, num_points=2001)
```

Builds a bounded odd inverse-like polynomial on $[-1,1]$. The target profile used internally is

$$
f(x) = \frac{2 \mu x}{x^2 + \mu^2}.
$$

This function is odd and bounded in magnitude by $1$, while behaving like a regularised inverse away from the origin. That makes it useful as a small, smooth inverse-like template in QSVT experiments. 

### Interpretation

For $|x| \gg \mu$, the function behaves qualitatively like a scaled version of $1/x$, while avoiding the singularity at $x=0$. Smaller values of `mu` make the transition sharper and more inverse-like near the origin. 

### Example

```python
import numpy as np
from qsvt.templates import inverse_like_polynomial

coeffs = inverse_like_polynomial(degree=7, mu=0.3)

xs = np.linspace(-1, 1, 200)
vals = np.polynomial.polynomial.polyval(xs, coeffs)
```

## `sign_approximation_polynomial`

```python
sign_approximation_polynomial(degree, sharpness=6.0, num_points=2001)
```

Builds an odd bounded polynomial approximating the sign function. The target used is

$$
f(x) = \tanh(\text{sharpness} \cdot x).
$$

This gives a smooth sign surrogate that remains bounded by $1$ and becomes steeper near the origin as `sharpness` increases. 

### Use cases

This template is useful for:

- sign-style spectral transforms
- matrix sign intuition
- projector construction via $(1 + \mathrm{sign}(x))/2$
- threshold-like experiments with odd symmetry 

### Example

```python
import numpy as np
from qsvt.templates import sign_approximation_polynomial

coeffs = sign_approximation_polynomial(degree=9, sharpness=8.0)

xs = np.linspace(-1, 1, 200)
vals = np.polynomial.polynomial.polyval(xs, coeffs)
```

## `soft_threshold_filter_polynomial`

```python
soft_threshold_filter_polynomial(
    degree,
    threshold=0.5,
    sharpness=12.0,
    num_points=2001,
)
```

Builds an even bounded soft-threshold filter polynomial. The target used is

$$
f(x) = \frac{1}{2}\left(1 + \tanh\bigl(\text{sharpness} \cdot (|x| - \text{threshold})\bigr)\right).
$$

This is close to $0$ for small $|x|$ and close to $1$ for larger $|x|$, with a smooth transition around the threshold. 

### Interpretation

Because the target depends on $|x|$, the resulting polynomial is even. This makes it a natural singular-value filter template when you want pass/reject behaviour without a discontinuous hard cutoff. 

### Example

```python
import numpy as np
from qsvt.templates import soft_threshold_filter_polynomial

coeffs = soft_threshold_filter_polynomial(degree=10, threshold=0.4)

xs = np.linspace(-1, 1, 200)
vals = np.polynomial.polynomial.polyval(xs, coeffs)
```

## `sqrt_approximation_polynomial`

```python
sqrt_approximation_polynomial(degree, num_points=2001)
```

Builds a bounded polynomial approximating the shifted square-root profile

$$
f(x) = \sqrt{\frac{x + 1}{2}}.
$$

This maps the canonical interval $[-1,1]$ into $[0,1]$, giving a square-root-like shape that remains bounded everywhere on the full QSVT interval. 

### Why this shifted form?

Approximating $\sqrt{x}$ directly on all of $[-1,1]$ is not appropriate, since $\sqrt{x}$ is only real for $x \ge 0$. The shifted form avoids that issue while still giving a useful square-root style template for experiments. 

### Example

```python
import numpy as np
from qsvt.templates import sqrt_approximation_polynomial

coeffs = sqrt_approximation_polynomial(degree=8)

xs = np.linspace(-1, 1, 200)
vals = np.polynomial.polynomial.polyval(xs, coeffs)
```

## `exponential_approximation_polynomial`

```python
exponential_approximation_polynomial(degree, beta=1.0, num_points=2001)
```

Builds a bounded exponential-like polynomial on $[-1,1]$. The target used is

$$
f(x) = e^{\beta x - |\beta|}.
$$

This function is strictly positive and bounded by $1$ on $[-1,1]$, making it a useful smooth weighting template for spectral damping or amplification-style experiments. 

### Interpretation

The parameter `beta` controls the tilt of the weighting profile. Positive `beta` favours larger values of $x$, while negative `beta` favours smaller values. The subtraction of $|\beta|$ keeps the target bounded by $1$ on the full interval. 

### Example

```python
import numpy as np
from qsvt.templates import exponential_approximation_polynomial

coeffs = exponential_approximation_polynomial(degree=5, beta=1.5)

xs = np.linspace(-1, 1, 200)
vals = np.polynomial.polynomial.polyval(xs, coeffs)
```

## Internal construction approach

The template builders use a lightweight common fitting path:

1. define a bounded target function on $[-1,1]$
2. sample it on a dense grid
3. fit a Chebyshev approximation
4. optionally enforce even or odd parity
5. convert back to standard monomial coefficients
6. rescale if necessary to keep the polynomial numerically bounded on $[-1,1]$ 

This construction keeps the implementation small and readable while ensuring the outputs match the package’s standard coefficient conventions. 

## Coefficient convention

All outputs use ascending monomial order:

```python
coeffs = [c0, c1, c2, ...]
```

meaning

$$
P(x) = c_0 + c_1 x + c_2 x^2 + \cdots.
$$

This is the same convention used across the rest of the package. 

## Parity structure

Some templates have built-in symmetry:

- `inverse_like_polynomial` returns an odd polynomial
- `sign_approximation_polynomial` returns an odd polynomial
- `soft_threshold_filter_polynomial` returns an even polynomial
- `sqrt_approximation_polynomial` does not impose a fixed parity
- `exponential_approximation_polynomial` does not impose a fixed parity 

This matters when using the templates for QSP/QSVT intuition, since many transformations naturally inherit even or odd structure.

## Typical workflow

A common pattern is:

1. choose a ready-made template
2. inspect its scalar response
3. compare it against a classical polynomial transform
4. use it in a QSVT simulation utility

For example:

```python
import numpy as np

from qsvt.templates import sign_approximation_polynomial
from qsvt.qsvt import qsvt_scalar_scan

coeffs = sign_approximation_polynomial(degree=9, sharpness=8.0)

xs = np.linspace(-1, 1, 101)
vals = np.polynomial.polynomial.polyval(xs, coeffs)
```

These coefficients can then be used in downstream package utilities such as scalar scans, diagonal experiments, or classical-vs-QSVT comparisons.

## Relationship to `qsvt.design`

The `qsvt.templates` and `qsvt.design` modules are closely related, but they serve slightly different purposes.

Use `qsvt.templates` when you want:

- ready-made polynomial families
- simple notebook examples
- quick prototyping
- low-friction starting points

Use `qsvt.design` when you want:

- more task-oriented construction helpers
- direct builders for workflows like inverse, sign, projector, sqrt, power, or filter design
- a higher-level polynomial construction API

In practice, `qsvt.templates` is the simpler “grab a useful family and try it” layer, while `qsvt.design` is the more explicit “construct a polynomial for a particular task” layer.

## Notes and limitations

These templates are intentionally lightweight. They do not currently provide:

- minimax-optimal approximation
- rigorous error guarantees
- automatic degree selection from tolerance targets
- QSP phase synthesis
- constrained optimisation-based design

Instead, they provide bounded, readable, and immediately usable polynomial families for educational and practical experimentation. 

## Summary

The `qsvt.templates` module provides compact bounded polynomial families for common QSVT/QSP-style tasks:

- inverse-like transforms
- sign approximations
- soft-threshold filters
- square-root style surrogates
- exponential weighting profiles

It is best viewed as a practical template layer: easy to use, easy to inspect, and well suited to small experiments, notebook workflows, and intuition building before moving to more specialised polynomial design methods. 
