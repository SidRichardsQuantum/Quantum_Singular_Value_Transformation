# Singular-Value Pseudoinverse Workflow

## Target

`singular_value_pseudoinverse_workflow(matrix, rhs, cutoff=..., degree=...)`
approximates a truncated pseudoinverse action for inverse problems and
least-squares systems.

## QSVT Idea

QSVT can implement bounded transformations of singular values. The target here
is inverse-like behavior above a cutoff, while small singular values are
suppressed to avoid unbounded amplification.

## Implementation

The workflow computes a dense SVD, normalizes the singular values, designs a
bounded inverse-like polynomial on the retained interval, applies it to the
singular values, and compares the resulting solution with a truncated-SVD
reference.

## Diagnostics

The result reports the polynomial solution, truncated-SVD reference solution,
residual norms, solution relative error, pseudoinverse operator error, singular
values, cutoff, scale, and coefficients.

## Scope

This is a finite dense regularization study. It does not supply a rectangular
block encoding, input-state preparation, postselection management, or
large-scale tomography.

## API

```python
from qsvt.algorithms import singular_value_pseudoinverse_workflow

result = singular_value_pseudoinverse_workflow(A, b, cutoff=0.15, degree=24)
report = result.as_report()
```

See also [Spectral filters](spectral_filters.md), [Linear systems](linear_systems.md),
and [Algorithm notes](algorithms.md).
