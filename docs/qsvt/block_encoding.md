# Block Encodings

Block encodings are the interface between a matrix problem and a QSVT
polynomial transform. This page describes the finite block-encoding helpers in
`qsvt-pennylane` and how to interpret their reports.

## Definition

A unitary `U` is a block encoding of a matrix `A` when its logical top-left
block is

```text
A / alpha
```

for a positive normalization `alpha`. Equivalently, the enlarged unitary has the
form

```text
U = [[A / alpha,  *],
     [*        ,  *]]
```

where the top-left block acts on the logical matrix register.

The normalization must make the encoded signal operator a contraction:

```text
||A / alpha||_2 <= 1.
```

For Hermitian workflows, this means the normalized eigenvalues lie in the QSVT
signal domain.

## Dense Finite Construction

`qsvt.block_encoding.block_encode_matrix` constructs an explicit dense unitary
dilation for a finite matrix. For

```text
B = A / alpha,
```

the helper uses the standard contraction dilation

```text
[[B,                 sqrt(I - B B*)],
 [sqrt(I - B* B),   -B*           ]].
```

This is a mathematically valid finite block encoding for the supplied matrix.
It is useful for simulator-scale verification because the unitary, top-left
block, reconstruction error, and unitarity error are all directly inspectable.

```python
import numpy as np
from qsvt.block_encoding import block_encode_matrix, verify_block_encoding

A = np.array([[2.0, 0.5], [0.5, 1.0]])
encoding = block_encode_matrix(A)
verification = verify_block_encoding(encoding)

print(encoding.alpha)
print(verification["block_encoding_verified"])
print(verification["unitary_verified"])
```

## What Is Verified

The finite helper verifies:

- the logical top-left block equals `A / alpha`,
- multiplying that block by `alpha` reconstructs `A`,
- the dense dilation is unitary to numerical tolerance,
- the logical dimension and enlarged unitary dimension are explicit.

These checks are stronger than a pure polynomial proxy: they validate an actual
finite unitary encoding for the supplied matrix.

## What Is Not Claimed

The dense construction does not provide:

- scalable sparse-oracle access,
- state preparation or right-hand-side loading,
- measurement or tomography,
- amplitude amplification or amplitude estimation,
- fault-tolerant synthesis,
- hardware compilation,
- quantum runtime or quantum-advantage evidence.

Those components depend on the problem family and access model. They should be
documented separately whenever a notebook or paper draft interprets a QSVT
polynomial as part of a larger quantum algorithm.

## Block-Encoded QSVT Workflow

`qsvt.algorithms.block_encoded_qsvt_workflow` combines finite block-encoding
verification with a PennyLane QSVT transform. It currently targets positive
Hermitian signal operators for which the package's matrix-QSVT comparison
agrees with ordinary spectral polynomial functional calculus.

The workflow:

1. validates a positive Hermitian matrix,
2. chooses or accepts `alpha`,
3. constructs a dense unitary block encoding,
4. verifies the block and unitarity errors,
5. applies the QSVT polynomial to `A / alpha`,
6. compares the QSVT logical block against the exact spectral reference,
7. optionally compares transformed state vectors.

```python
import numpy as np
from qsvt.algorithms import block_encoded_qsvt_workflow

A = np.diag([0.2, 0.6, 0.9])
coeffs = np.array([1.0, 0.0, -1.0])

result = block_encoded_qsvt_workflow(A, coeffs)
print(result.operator_relative_error)
```

## Relation To Resource Proxies

Resource reports track polynomial degree, signal-call proxies, compatibility,
and matrix-register width. A finite block-encoding report tracks a concrete
dense unitary for one supplied matrix. These are complementary:

- resource proxies help compare polynomial designs,
- finite block encodings verify an actual small signal model,
- neither alone supplies a scalable end-to-end quantum runtime.

For broader cost interpretation, see [QSVT resource model](qsvt_resource_model.md).
For workflow-level targets and diagnostics, see [Algorithm notes](algorithms.md).
