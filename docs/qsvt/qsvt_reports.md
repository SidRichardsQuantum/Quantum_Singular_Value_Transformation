# QSVT Transform Reports

QSVT transform reports compare explicit QSVT transforms against direct
classical polynomial references. The package supports both diagonal-value
reports and small non-diagonal Hermitian matrix reports.

Use these reports after choosing or designing a polynomial to answer:

- what did the QSVT wrapper return?
- what should the classical polynomial transform return?
- how large is the sampled discrepancy?
- which wires and block-encoding metadata were used?

## Compatibility reports

Before running a transform, you can check whether coefficients look compatible
with the PennyLane QSVT path:

```python
from qsvt.qsvt import qsvt_compatibility_report

report = qsvt_compatibility_report([0, 0, 1])

print(report["compatible"], report["reasons"])
```

Compatibility reports check:

- coefficient finiteness
- definite parity (`even`, `odd`, or `zero`)
- sampled boundedness on `[-1, 1]`
- optional PennyLane QSVT synthesis on a scalar input

A polynomial can be bounded and still fail synthesis. This commonly happens
when the coefficient array has mixed parity or when PennyLane's phase-angle
solver cannot construct a complementary polynomial for that coefficient set.
The report records structured reasons such as `mixed_parity`,
`out_of_bounds`, and `synthesis_failed`.

## Diagonal Python workflow

```python
from qsvt.qsvt import qsvt_transform_report
from qsvt.reports import save_report

report = qsvt_transform_report(
    [1.0, 0.7, 0.3, 0.1],
    [0, 0, 1],
    encoding_wires=[0, 1, 2],
)

print(report["max_error"])
save_report(report, "qsvt-report.json")
```

The report contains:

| field | meaning |
| --- | --- |
| `input` | diagonal input values |
| `poly` | polynomial coefficients in ascending degree order |
| `qsvt` | diagonal values extracted from the QSVT transform |
| `classical` | direct classical polynomial transform |
| `abs_error` | absolute difference between QSVT and classical outputs |
| `max_error` | maximum absolute error |
| `rms_error` | root-mean-square error |
| `qsvt_succeeded` | whether PennyLane synthesized and evaluated the transform |
| `qsvt_error_type`, `qsvt_error` | synthesis error details when available |
| `encoding_wires` | wires passed to the QSVT block encoding |
| `wire_order` | wire order used for matrix extraction |
| `block_encoding` | PennyLane block-encoding mode |
| `matrix_dimension` | dimension of the diagonal input matrix |
| `unitary_dimension` | dimension implied by the wire order |
| `polynomial_degree` | effective degree from the coefficient array length |

## Non-diagonal matrix workflow

For small Hermitian matrices, use `qsvt_matrix_transform_report`. The report
extracts the logical QSVT block and compares its real part with the classical
spectral polynomial reference `P(A)`.

```python
from qsvt.matrices import rotated_diagonal
from qsvt.qsvt import qsvt_matrix_transform_report
from qsvt.reports import save_report

A = rotated_diagonal([0.2, 0.8], theta=0.45)

report = qsvt_matrix_transform_report(
    A,
    [0, 0, 1],
)

print(report["max_error"], report["max_imag_abs"])
save_report(report, "matrix-report.json")
```

The matrix report includes the diagonal report fields that still apply, plus:

| field | meaning |
| --- | --- |
| `input` | Hermitian input matrix |
| `eigenvalues` | eigenvalues used to validate the QSVT domain |
| `qsvt` | real part of the extracted logical QSVT block |
| `qsvt_imag` | imaginary part of the extracted logical QSVT block |
| `classical` | classical spectral polynomial matrix `P(A)` |
| `frobenius_error` | Frobenius norm of `qsvt - classical` |
| `max_imag_abs` | maximum absolute imaginary entry in the extracted QSVT block |

`max_imag_abs` is reported because PennyLane's QSVT convention can leave
complex phases in the extracted block even when the real part matches the
classical Hermitian polynomial reference.

## CLI workflow

For explicit coefficients:

```bash
qsvt compatibility-report --poly "0,0,1"

qsvt compare-report \
  --values "1.0,0.7,0.3,0.1" \
  --poly "0,0,1" \
  --wires 3 \
  --output qsvt-report.json

qsvt matrix-report \
  --matrix "0.31351701,-0.23499807;-0.23499807,0.68648299" \
  --poly "0,0,1" \
  --output matrix-report.json
```

For a designed polynomial:

```bash
qsvt design-compatibility \
  --kind sign \
  --degree 13 \
  --gamma 0.2

qsvt apply-design \
  --kind sign \
  --values="-0.8,-0.3,0.3,0.8" \
  --degree 13 \
  --gamma 0.2 \
  --wires 3 \
  --output sign-qsvt-report.json
```

`compare-report` and `matrix-report` are useful when you already have
coefficients and should fail if PennyLane cannot synthesize the requested QSVT
transform. `apply-design` is useful when you want to build a polynomial from
`qsvt.design` and immediately inspect its QSVT compatibility on a small
diagonal example; when PennyLane rejects the generated coefficients, the
command returns `qsvt_succeeded: false` with error details instead of a Python
traceback.

## Input constraints

Diagonal values and Hermitian matrix eigenvalues must lie in `[-1, 1]`,
matching the QSVT polynomial domain used throughout this package. The report
helpers validate finite input values, finite coefficients, and non-empty arrays
before running the QSVT comparison.
