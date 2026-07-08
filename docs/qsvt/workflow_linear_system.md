# Linear System Workflow

## Target

`linear_system_workflow(matrix, rhs, degree=...)` studies a small positive
definite linear system

\[
A x = b.
\]

The QSVT-style target is an inverse-like scalar function on the positive
spectrum of a normalized matrix.

## QSVT Idea

After rescaling \(A\) so its eigenvalues lie in \([\gamma, 1]\), the workflow
designs a bounded polynomial approximating \(\gamma / x\). Applying this
polynomial to the scaled operator gives an approximation to \(A^{-1}b\) after
the stored normalization is undone.

In an end-to-end quantum algorithm, this polynomial would be applied to a block
encoding of the scaled matrix. This package keeps the polynomial design and
finite validation explicit.

## Implementation

The workflow validates a dense Hermitian positive-definite matrix, rescales the
spectrum, designs a positive inverse polynomial, applies it through a dense
spectral decomposition, and compares against `numpy.linalg.solve`.

It can also attach compatibility metadata and optional small PennyLane QSVT
checks when requested.

## Diagnostics

The result reports the classical solution, polynomial solution, residual norm,
relative solution error, polynomial diagnostics, compatibility metadata,
condition-number metadata, scaled spectral bounds, and a resource-proxy block.

## Scope

This is a finite dense validation workflow. It does not implement scalable
state preparation, sparse or oracle block encoding, amplitude amplification,
readout, fault-tolerant synthesis, or hardware execution.

## API

```python
from qsvt.algorithms import linear_system_workflow

result = linear_system_workflow(A, b, degree=20)
report = result.as_report()
```

See also [Linear systems](linear_systems.md) and [Algorithm notes](algorithms.md).
