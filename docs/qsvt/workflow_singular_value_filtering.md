# Singular-Value Filtering Workflow

## Target

`singular_value_filtering_workflow(matrix, cutoff=..., degree=...)` applies a
smooth filter to the singular values of a rectangular or non-Hermitian matrix.

## QSVT Idea

QSVT can transform singular values of a block-encoded operator. The workflow
uses this idea at the dense reference level: normalize singular values, design
a bounded threshold-like polynomial, and apply it to the SVD.

## Implementation

The matrix is decomposed with a dense SVD. Singular values are normalized by
the largest singular value, transformed by a polynomial approximation to a
smooth cutoff function, and reconstructed into a filtered matrix.

## Diagnostics

The result includes original and normalized singular values, polynomial and
exact filtered matrices, operator relative error, coefficients, cutoff
metadata, and optional output-vector error when an input vector is supplied.

## Scope

This is a dense SVD validation workflow. A full QSVT algorithm would also need
a block encoding of the rectangular operator, state preparation,
success-probability handling, and readout.

## API

```python
from qsvt.algorithms import singular_value_filtering_workflow

result = singular_value_filtering_workflow(A, cutoff=0.25, degree=18)
report = result.as_report()
```

See also [Spectral filters](spectral_filters.md) and
[Algorithm notes](algorithms.md).
