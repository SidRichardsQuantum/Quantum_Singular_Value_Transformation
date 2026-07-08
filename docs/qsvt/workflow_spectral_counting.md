# Spectral Counting Workflow

## Target

`spectral_counting_workflow(matrix, lower=..., upper=..., degree=...)`
estimates how many eigenvalues lie in a physical interval.

## QSVT Idea

An interval projector can be approximated by a bounded polynomial. The trace of
that polynomial projector estimates the number of eigenvalues in the selected
band.

## Implementation

The workflow rescales the Hermitian matrix, maps the physical interval into
the scaled coordinate, designs a smooth interval-projector polynomial, applies
it through dense eigendecomposition, and compares with the exact hard spectral
projector.

## Diagnostics

The result includes the polynomial projector, exact hard projector, exact
count, polynomial trace count, count error, design diagnostics, and optional
Hutchinson-style stochastic trace estimates when probes are requested.

## Scope

This is a dense finite diagnostic. The stochastic option is still local and
finite; it is not a scalable quantum trace-estimation implementation.

## API

```python
from qsvt.algorithms import spectral_counting_workflow

result = spectral_counting_workflow(H, lower=-0.2, upper=0.3, degree=22)
report = result.as_report()
```

See also [Spectral filters](spectral_filters.md) and [Algorithm notes](algorithms.md).
