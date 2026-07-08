# Linear System Comparison Workflow

## Target

`linear_system_comparison_workflow(matrix, rhs, degree=...)` compares several
finite solution paths for the same small positive-definite linear system.

## QSVT Idea

The QSVT-style row uses the same bounded inverse-polynomial target as
`linear_system_workflow`: approximate \(\gamma / x\) on the normalized positive
spectrum, then undo the stored scaling to estimate \(A^{-1}b\).

## Implementation

The workflow builds table-like rows for a dense direct solve, optional
conjugate-gradient solve, QSVT-style polynomial inverse action, optional
PennyLane QSVT matrix check, and optional finite HHL circuit execution.

## Diagnostics

Rows report solver name, implementation kind, residual norm, relative solution
error, condition-number metadata, polynomial degree, iteration count, and
optional QSVT or HHL state/circuit diagnostics.

## Scope

This is a finite numerical comparison helper. It is not a wall-clock benchmark
and does not include quantum data loading, block-encoding costs, amplitude
amplification, tomography, hardware compilation, or fault-tolerant resources.

## API

```python
from qsvt.algorithms import linear_system_comparison_workflow

result = linear_system_comparison_workflow(A, b, degree=20)
rows = result.as_report()["rows"]
```

See also [Linear systems](linear_systems.md),
[Linear system workflow](workflow_linear_system.md), and
[Algorithm notes](algorithms.md).
