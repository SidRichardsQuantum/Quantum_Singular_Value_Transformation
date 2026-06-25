# Polynomial Realizability and Phase Synthesis

`qsvt.synthesis` separates three questions that should not be conflated:

1. Can a polynomial be evaluated as ordinary classical functional calculus?
2. Is it bounded and definite-parity, so one standard QSP/QSVT sequence can
   realize it?
3. Can PennyLane's selected numerical angle solver synthesize stable phases for
   the supplied coefficients?

## Realizability Classification

```python
from qsvt import classify_polynomial_realizability

classification = classify_polynomial_realizability([0.5, 0.5])

print(classification.kind)
# multiple-parity-sequences-or-lcu
```

The reported categories are:

- `single-sequence-qsp-qsvt`: extrema-bounded with even, odd, or zero parity,
- `multiple-parity-sequences-or-lcu`: extrema-bounded but mixed parity,
- `classical-polynomial-only`: finite but outside the sampled QSP/QSVT bound,
- `invalid-polynomial`: non-finite coefficients.

Mixed-parity reports include separate even and odd coefficient arrays. They can
be studied classically, but one standard QSP/QSVT phase sequence is not enough.
A quantum realization needs separate parity sequences and a combination
mechanism such as an LCU-style construction.

## Extrema-Based Boundedness Certificates

Grid sampling can miss a narrow polynomial peak. The package therefore checks
the interval endpoints and every numerically real root of the derivative:

```python
from qsvt import certify_polynomial_boundedness

certificate = certify_polynomial_boundedness(
    [0.996, 0.1, -0.5],
    domain=(-1.0, 1.0),
)

print(certificate.maximizing_point)
print(certificate.max_abs_value)
print(certificate.is_bounded)
```

The report records all evaluated extrema, the maximizing point, margin,
tolerance, and derivative-root residual. This is a floating-point polynomial
extrema certificate rather than an interval-arithmetic proof.

The same check is available from the CLI:

```bash
qsvt boundedness-certificate --poly "0.996,0.1,-0.5"
```

## Phase Synthesis

```python
from qsvt import synthesize

result = synthesize(
    [0.0, 1.0, 0.0, -0.5, 0.0, 1.0 / 3.0],
    routine="QSVT",
    angle_solver="root-finding",
)

print(result.angles)
print(result.reconstruction_max_error)
```

`PhaseSynthesisResult` retains:

- the polynomial and realizability classification,
- routine and angle solver,
- synthesized angles,
- synthesis duration,
- scalar QSVT reconstruction error,
- solver exceptions as structured failure metadata,
- the phase and coefficient convention.

The available solver names are passed to PennyLane:

- `root-finding`,
- `iterative`,
- `iterative-optax`.

The iterative solvers may require additional dependencies supplied by
PennyLane. A polynomial can pass structural realizability checks and still fail
numerical synthesis; the result preserves that distinction.

The CLI exposes the same workflow:

```bash
qsvt phase-synthesis \
  --poly "0,1,0,-0.5,0,0.333333333333" \
  --routine QSVT \
  --angle-solver root-finding
```

## Designed Polynomials

Every `DesignWorkflowResult` can invoke the synthesis layer directly:

```python
from qsvt import design_workflow

design = design_workflow("sign", gamma=0.25, degree=13)
synthesis = design.synthesize()
```

Design boundedness and parity are prerequisites, not guarantees that a
particular numerical solver will converge.

## Solver Benchmarks

`benchmark_phase_solvers` compares convergence, synthesis time, phase count,
and reconstruction error while retaining degree, coefficient dynamic range,
and boundedness margin as conditioning proxies:

```python
from qsvt import benchmark_phase_solvers

benchmark = benchmark_phase_solvers(
    [0.0, 1.0, 0.0, -0.5, 0.0, 1.0 / 3.0],
    solvers=["root-finding", "iterative"],
    repeats=3,
)
```

```bash
qsvt phase-solver-benchmark \
  --poly "0,1,0,-0.5,0,0.333333333333" \
  --solvers "root-finding,iterative" \
  --repeats 3
```

Timings cover classical angle synthesis only. They are not quantum-circuit or
hardware runtime measurements.

## Mixed-Parity Synthesis

`synthesize_mixed_parity` separates a bounded mixed-parity polynomial into even
and odd components, normalizes and synthesizes each component independently,
and reports an LCU-style combination model:

```python
from qsvt import synthesize_mixed_parity

result = synthesize_mixed_parity([0.5, 0.5])
```

The component extrema norms become LCU weights. The report includes the
normalization sum and idealized postselection probability proxy
`1 / lambda**2`, together with explicit assumptions and omitted amplitude
amplification costs. The package synthesizes the component sequences but does
not claim to implement the full LCU circuit.

```bash
qsvt mixed-parity-synthesis --poly "0.5,0.5"
```
