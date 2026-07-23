# Block Encodings

Block encodings are the interface between a matrix problem and a QSVT
polynomial transform. This page describes the finite block-encoding helpers in
`qsvt-pennylane` and how to interpret their reports.

For the concise algorithm page that combines finite block-encoding
verification with a QSVT transform, see
[Block-encoded QSVT workflow](workflow_block_encoded_qsvt.md).

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

Notation:

- `A` is the logical matrix to be transformed.
- `U` is the enlarged unitary acting on ancilla plus logical registers.
- `alpha` is the positive normalization; the QSVT signal operator is
  `A / alpha`.
- `B` below is shorthand for the normalized contraction `A / alpha`.
- `B*` denotes the conjugate transpose of `B`.
- `I` is the identity matrix with the same logical dimension as `A`.

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

## Access-Model Specifications

`BlockEncodingSpec` describes how a caller intends to supply a signal operator
without claiming that every source is directly executable by one backend.
Its `execution_supported` and `high_level_qsvt_supported` fields describe
PennyLane's high-level `qml.qsvt` adapter. The separate
`lower_level_qsvt_supported` field identifies specifications accepted by
`execute_qsvt_from_spec`; actual device compatibility is determined during
execution and reported as structured success or failure data.

Supported specification constructors are:

- `matrix_block_encoding_spec` for dense, rectangular, or sparse-like matrices,
- `pennylane_operator_block_encoding_spec` for PrepSelPrep or qubitization,
- `circuit_block_encoding_spec` for user-provided PennyLane operation factories.

`build_block_encoding_operator` constructs the corresponding PennyLane
block-encoding operation. `qsvt_operator_from_block_encoding` supports sources
accepted by PennyLane's high-level QSVT path.

`execute_qsvt_from_spec` uses PennyLane's lower-level `qml.QSVT` operation to
execute these specifications in a QNode. It supports:

- dense and sparse-like matrix specifications,
- rectangular matrix specifications with alternating output/input projector
  dimensions,
- PennyLane operators through PrepSelPrep or qubitization,
- custom operation factories with synthesized or caller-supplied projectors.

For dense `embedding` specifications, the executor supplies the exact finite
unitary dilation to `qml.QSVT`. This preserves the normalization recorded as
`BlockEncodingSpec.alpha` for non-diagonal contractions; it avoids an
additional backend normalization that `qml.BlockEncode` may otherwise apply.

```python
import pennylane as qml
from qsvt import (
    execute_qsvt_from_spec,
    pennylane_operator_block_encoding_spec,
)

H = qml.dot([0.3, 0.7], [qml.Z(1), qml.X(1)])
spec = pennylane_operator_block_encoding_spec(
    H,
    encoding_wires=[0],
    block_encoding="prepselprep",
)
result = execute_qsvt_from_spec(spec, [0.0, 1.0], [1.0, 0.0])
```

The execution report includes the access model, normalization, projector
source, logical output, finite reference output where available, relative
error, and encoding-specific circuit resources. Backend failures are retained
as structured report data unless `raise_on_failure=True`.

Reports identify themselves with schema name
`block-encoding-qsvt-execution` and schema version `1.0`. Numerical diagnostics
separate real-output error, complex leakage, logical-subspace leakage,
normalization error, and finite-shot uncertainty.

Mixed-parity or complex multi-component transforms use
`execute_mixed_parity_qsvt_from_spec` or
`execute_qsvt_component_lcu_from_spec`. These experimental helpers implement
real-part extraction with forward and adjoint QSVT sequences, coherently select
the weighted components, uncompute the selector, and emit
`coherent-qsvt-execution` `1.0` reports with measured postselection and circuit
resource ledgers. The initial coherent-combination contract requires a square
Hermitian logical transform. Rectangular single-sequence execution remains
available through `execute_qsvt_from_spec`.

For a complete rectangular example, run:

```bash
python examples/rectangular_execution.py \
  --output /tmp/qsvt-rectangular-execution.json
```

## Hardware-Oriented Execution

The experimental `execute_qsvt_on_device` API accepts a caller-provided
PennyLane device, preparation circuit, and positive finite-shot measurement
contract. It performs local compatibility preflight, executes full-register
probabilities, reports sampling uncertainty, and keeps provider submission and
credential management outside the portable result.

Hardware-facing paths should use block encodings with gate decompositions:
FABLE for suitable matrix inputs, PrepSelPrep or qubitization for compatible
LCU operators, or explicitly supplied decomposable custom circuits.
`BlockEncode` remains useful for finite simulator validation but should not be
presented as hardware-executable when it cannot be decomposed for the selected
device.

The current report includes logical circuit resources, wire mapping, shots,
device/provider metadata when advertised, and statistical uncertainty. Native
provider compilation, two-qubit gate counts, persistent jobs, calibration
capture, and mitigation remain roadmap items.

Two focused cookbook examples make the signal and device contracts explicit:

```bash
python examples/custom_block_encoding.py \
  --output /tmp/qsvt-custom-block-encoding.json
python examples/finite_shot_qsvt.py \
  --output /tmp/qsvt-finite-shot.json --shots 2000 --seed 12345
```

The finite-shot example uses a seeded local `default.qubit` device and a
decomposable FABLE access model. It validates preflight, uncertainty, and
agreement with an ideal finite reference; it does not submit to real hardware
or require provider credentials.

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

See [Block-encoded QSVT workflow](workflow_block_encoded_qsvt.md) for the
target, QSVT idea, implementation summary, diagnostics, scope, and minimal API.

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
