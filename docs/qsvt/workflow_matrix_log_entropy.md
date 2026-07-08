# Matrix Log and Entropy Workflow

## Target

`matrix_log_entropy_workflow(matrix, epsilon=..., degree=...)` approximates a
regularized matrix logarithm and entropy-like spectral density for a positive
semidefinite matrix.

## QSVT Idea

Logarithms are singular near zero, so the workflow uses a regularized target
\(\log(x + \epsilon)\). Polynomial approximations to the regularized log and
to \(-x\log(x + \epsilon)\) are QSVT-style spectral transformation targets.

## Implementation

The matrix is normalized to a positive spectrum in \([0, 1]\). The workflow
fits separate polynomials for the regularized log and entropy density, applies
them by dense spectral calculus, and compares with exact dense references.

## Diagnostics

The result includes log and entropy coefficients, polynomial and exact log
operators, polynomial and exact entropy operators, trace entropy values,
relative operator errors, epsilon, and spectral scale metadata.

## Scope

This supports covariance-spectrum, graph entropy, free-energy proxy, and
regularized log-determinant examples on finite matrices. The regularization
parameter is part of the mathematical problem and should be reported.

## API

```python
from qsvt.algorithms import matrix_log_entropy_workflow

result = matrix_log_entropy_workflow(A, epsilon=1e-3, degree=24)
report = result.as_report()
```

See also [Spectral filters](spectral_filters.md) and [Algorithm notes](algorithms.md).
