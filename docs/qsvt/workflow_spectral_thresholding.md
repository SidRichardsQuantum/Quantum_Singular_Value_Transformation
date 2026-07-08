# Spectral Thresholding Workflow

## Target

`spectral_thresholding_workflow(matrix, lower=..., upper=..., degree=...)`
approximates a hard interval projector with a smooth bounded polynomial.

## QSVT Idea

QSVT can apply polynomial filters to a block-encoded Hermitian operator. Here
the polynomial target is a band-pass indicator for an interval of the spectrum.

## Implementation

The matrix is rescaled to the QSVT design interval, the physical interval is
mapped into the scaled coordinate, an interval-projector polynomial is fitted,
and the dense polynomial projector is compared with the exact hard projector.

## Diagnostics

The result includes the polynomial projector, exact projector, exact rank,
trace-based polynomial rank proxy, leakage outside the selected interval,
operator error, design diagnostics, and optional retained-state weight
comparison.

## Scope

This is a finite spectral filtering study. Sharp interval edges require higher
degree, and rank proxies should be interpreted with the reported leakage and
operator error.

## API

```python
from qsvt.algorithms import spectral_thresholding_workflow

result = spectral_thresholding_workflow(H, lower=-0.4, upper=0.2, degree=24)
report = result.as_report()
```

See also [Spectral filters](spectral_filters.md) and [Algorithm notes](algorithms.md).
