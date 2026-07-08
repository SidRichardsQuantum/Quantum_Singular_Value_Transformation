# Block-Encoded QSVT Workflow

## Target

`block_encoded_qsvt_workflow(matrix, coeffs, state=None, alpha=None)` validates
a finite QSVT polynomial transform on an explicitly constructed block encoding.

## QSVT Idea

A unitary block encoding exposes \(A / \alpha\) as a logical block. QSVT then
uses phase-modulated signal processing to implement a polynomial \(P(A /
\alpha)\) on the logical subspace.

## Implementation

The workflow constructs a dense unitary dilation whose top-left block is the
normalized matrix, verifies the block encoding and unitarity numerically, runs
the package's PennyLane matrix-QSVT path, and compares the logical block with a
dense spectral polynomial reference.

## Diagnostics

The result includes the block-encoding verification report, normalization
metadata, QSVT operator, exact spectral reference, operator relative error, and
optional state-vector output/error fields.

## Scope

This is an experimental finite simulator workflow. It verifies a real dense
block encoding and finite QSVT transform, but it is not a scalable oracle
construction and does not include hardware compilation or runtime claims.

## API

```python
from qsvt.algorithms import block_encoded_qsvt_workflow

result = block_encoded_qsvt_workflow(A, [0.0, 1.0])
report = result.as_report()
```

See also [Block encodings](block_encoding.md), [Implementation notes](implementation.md),
and [Algorithm notes](algorithms.md).
