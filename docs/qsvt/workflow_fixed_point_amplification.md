# Fixed-Point Amplification Workflow

## Target

`fixed_point_amplification_workflow(score_operator, state, rounds=...)`
applies a monotone amplification polynomial to a finite positive score
operator with spectrum in \([0, 1]\).

## QSVT Idea

The scalar polynomial

\[
1 - (1 - x)^r
\]

boosts larger scores toward one while leaving smaller scores relatively
suppressed. This is a simple spectral-polynomial model of robust amplification.

## Implementation

The workflow validates the finite score operator, applies the polynomial by
dense spectral functional calculus, normalizes the amplified state, and
compares with the exact polynomial reference.

## Diagnostics

The result reports the polynomial and reference operators, normalized
amplified state, state error, operator error, initial score, amplified score,
reference score, degree, and coefficients.

## Scope

This is a spectral amplification primitive, not a full Grover iterate or
amplitude-amplification circuit. The caller supplies the finite score or
projector operator.

## API

```python
from qsvt.algorithms import fixed_point_amplification_workflow

result = fixed_point_amplification_workflow(score_operator, psi, rounds=4)
report = result.as_report()
```

See also [Spectral filters](spectral_filters.md) and [Algorithm notes](algorithms.md).
