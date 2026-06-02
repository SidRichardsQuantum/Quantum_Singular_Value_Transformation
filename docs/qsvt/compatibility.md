# QSVT Compatibility

QSVT can implement only polynomials that satisfy structural constraints. This
page summarizes the compatibility checks used by `qsvt-pennylane` and explains
how to read failures without overclaiming.

## Core Conditions

A real polynomial intended for direct QSVT synthesis should satisfy:

- **finite coefficients**: all coefficients are numerical and finite,
- **boundedness**: `|P(x)| <= 1` on the signal interval, usually `[-1, 1]`,
- **definite parity**: the polynomial is either even or odd,
- **backend synthesis support**: the selected QSVT backend can synthesize the
  polynomial for the requested signal model.

Boundedness keeps the transformed block compatible with unitarity. Parity is a
QSP/QSVT structural condition: even and odd polynomial sequences correspond to
different phase-sequence forms and signal conventions.

## Compatibility Reports

`qsvt.compatibility.qsvt_compatibility_report` samples the polynomial on the
bounded domain, checks parity, and can optionally attempt PennyLane synthesis.

```python
from qsvt.compatibility import qsvt_compatibility_report

report = qsvt_compatibility_report([0.0, 0.0, 1.0])

print(report["compatible"])
print(report["parity"])
print(report["reasons"])
```

Typical report fields include:

| field | meaning |
| --- | --- |
| `polynomial_degree` | highest nonzero coefficient degree |
| `parity` | `even`, `odd`, or `mixed` |
| `is_bounded` | sampled boundedness result |
| `max_abs_value` | sampled maximum absolute value |
| `attempted_pennylane_synthesis` | whether backend synthesis was tried |
| `pennylane_synthesis_succeeded` | synthesis result when attempted |
| `reasons` | failure reasons such as `mixed_parity` or `out_of_bounds` |
| `compatible` | combined compatibility result |

## Common Failure Modes

### Mixed Parity

`1 + x` is mixed parity. It may be useful as an ordinary polynomial, but it is
not directly compatible with a single standard QSVT phase sequence.

```python
qsvt_compatibility_report([1.0, 1.0], attempt_synthesis=False)
```

Expected reasons include `mixed_parity`, and for this particular example also
`out_of_bounds`.

### Out Of Bounds

`2x` has odd parity but exceeds the unit bound on `[-1, 1]`. It fails because a
QSVT-transformed block must remain compatible with an embedding into a unitary.

```python
qsvt_compatibility_report([0.0, 2.0], attempt_synthesis=False)
```

Expected reasons include `out_of_bounds`.

### Bounded But Mixed

A polynomial such as `0.25 + 0.25x` is bounded on `[-1, 1]`, but mixed parity.
This is a useful example because it separates boundedness from parity:

```python
qsvt_compatibility_report([0.25, 0.25], attempt_synthesis=False)
```

Expected reasons include `mixed_parity`, while `is_bounded` remains true.

### Backend Synthesis Failure

Some polynomials satisfy sampled boundedness and parity checks but still fail a
specific backend synthesis path. In that case the polynomial may be
mathematically admissible, while the current implementation path cannot produce
or verify the corresponding QSVT operator.

Reports distinguish this with synthesis fields instead of treating all failures
as mathematical failures.

## QSVT-Style Versus Direct QSVT

Many notebooks use dense spectral polynomial transforms to study QSVT-style
matrix functions. That is useful even when a polynomial is not directly
synthesizable by the package's QSVT wrapper.

Use precise language:

- **dense spectral polynomial workflow**: applies `P(A)` classically for a
  finite matrix,
- **QSVT-style polynomial core**: studies a bounded polynomial target relevant
  to a future QSVT algorithm,
- **direct QSVT verification**: constructs or extracts a QSVT block and compares
  it against a reference,
- **end-to-end quantum algorithm**: also supplies scalable block encoding, state
  preparation, readout, success-probability management, and hardware assumptions.

The repository's truth contracts use this distinction throughout reports and
benchmark outputs.

## Where To Learn More

- For broad mathematical context, see [Theory](theory.md).
- For finite block encodings, see [Block encodings](block_encoding.md).
- For polynomial builders, see [Polynomial design helpers](design.md) and
  [Polynomial templates](templates.md).
- For resource interpretation, see [QSVT resource model](qsvt_resource_model.md).
