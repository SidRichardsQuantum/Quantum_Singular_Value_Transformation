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

### Constant polynomials and solver output validation

QSVT synthesis supports exact constant polynomials in `[-1, 1]`, including
zero and arrays padded with trailing zeros. The result uses one projector phase
`arccos(c)` with no signal queries and records `angle_solver="analytic-constant"`.
Its real signal block is `c`; this does not claim that the full complex block
is the real constant. QSP constant synthesis retains the backend's limitations.
Mixed-parity synthesis uses the same constant sequence and reports its one
projector phase and zero signal calls, replacing the previous empty phase array.

Built-in solvers and external adapters must return a nonempty, one-dimensional
array of finite real phases. Invalid outputs produce structured failures with
`angles=None`; complex phases are rejected rather than silently cast to real.
Reconstruction grids require an integer of at least two points, including when
using the cache or an adapter.

### Solver completion and reconstruction quality

`succeeded` records whether synthesis returned phases. It does not assert that
their sampled response matches the polynomial to any particular tolerance:

```python
quality = result.quality_report(tolerance=1e-6)
print(quality["solver_returned_phases"])
print(quality["reconstruction_passed"])
print(quality["status"])
```

The statuses are `solver_failed`, `reconstruction_unavailable`,
`reconstruction_failed`, and `passed`. Non-finite phases cannot pass; absent or
non-finite reconstruction errors remain unvalidated. QSP results without
package reconstruction therefore cannot pass this assessment. The method
does not change the synthesis result, polynomial, or application acceptance
thresholds. Its tolerance must be finite and non-negative. A passing sampled
reconstruction is not a uniform error certificate or an approximation bound
against the original target function.

The Studio investigation reproduced root-finding failures for the degree-13
sign and normalized-reciprocal designs in PennyLane 0.45.1. The exception arose
when assembling the polynomial and complementary-polynomial arrays with
different lengths; these designs also have small boundedness margins. The
degree-10 soft filter returned phases but had about `0.075` reconstruction
error. These observations are backend/version-dependent, not guarantees that
root-finding always fails on those inputs.

Regression tests verify that the iterative solver reconstructs the same,
unmodified polynomials within `1e-6`. The Studio exposes this method explicitly
and preserves root-finding failures and reconstruction residuals. It does not
patch PennyLane internals, rescale coefficients to hide failures, or change the
stable facade's default solver. Mixed-parity interval designs still need a
multi-sequence construction; switching solvers does not remove that condition.

The CLI exposes the same workflow:

```bash
qsvt phase-synthesis \
  --poly "0,1,0,-0.5,0,0.333333333333" \
  --routine QSVT \
  --angle-solver root-finding
```

Run the compact cookbook client to compare a successful definite-parity
synthesis with bounded mixed-parity and interior-extrema failure cases:

```bash
python examples/synthesis_diagnostics.py \
  --output /tmp/qsvt-synthesis-diagnostics.json
```

The saved report keeps boundedness, realizability, phase convention,
reconstruction error, and structured failure metadata separate for every
case. It measures classical phase synthesis and does not claim circuit or
hardware execution.

The companion compatibility tutorial visualizes the extrema case and executes
the same success and structured-failure paths:

```text
notebooks/tutorials/15_QSVT_Compatibility_Failure_Cases.ipynb
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

Benchmark rows retain `successes` and `converged` as solver-completion fields
for compatibility. `validated_successes` and `all_reconstructions_passed`
separately assess sampled reconstruction against `reconstruction_tolerance`
(default `1e-6`). QSP results without reconstruction remain unvalidated.

Timings cover classical angle synthesis only. They are not quantum-circuit or
hardware runtime measurements.

Use `benchmark_phase_solver_stress_matrix` when solver behavior must be
compared across several degrees or numerical regimes in one report:

```python
from qsvt import benchmark_phase_solver_stress_matrix

stress = benchmark_phase_solver_stress_matrix(
    {
        "linear-margin": [0.0, 0.5],
        "quintic-near-boundary": [0.0, 0.0, 0.0, 0.0, 0.0, 0.95],
    },
    solvers=["root-finding", "iterative"],
    repeats=3,
)
```

```bash
qsvt phase-solver-stress \
  --case "linear-margin=0,0.5" \
  --case "quintic-near-boundary=0,0,0,0,0,0.95" \
  --solvers "root-finding,iterative" \
  --repeats 3
```

The flat stress rows retain the case name, conditioning proxies, convergence,
classical synthesis timing, phase count, and reconstruction errors. This is a
phase-synthesis diagnostic, not a device benchmark.

The committed stress-matrix benchmark fixes the base-install `root-finding`
solver across increasing degree, shrinking boundedness margin, and a scaled
Chebyshev polynomial with a larger monomial-coefficient range:

```text
notebooks/benchmarks/07_phase_synthesis_stress_matrix.ipynb
```

Its JSON/CSV artifacts keep timing environment-qualified and use convergence,
phase count, and reconstruction error as the portable regression checks.

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
amplification costs. This synthesis-only report does not claim circuit
execution.

Use `execute_mixed_parity_qsvt_from_spec` to construct and execute the full
finite selector-LCU circuit. It selects the forward and adjoint form of each
component sequence to extract the real polynomial, uncomputes the selector,
and reports measured postselection probabilities and circuit resources:

```python
from qsvt import (
    execute_mixed_parity_qsvt_from_spec,
    matrix_block_encoding_spec,
)

spec = matrix_block_encoding_spec([[0.2, 0.0], [0.0, 0.8]])
execution = execute_mixed_parity_qsvt_from_spec(
    spec,
    [0.2, 0.3, 0.1],
    [1.0, 0.0],
)
```

The executor remains experimental and is not exported by `qsvt.stable`. It
supports square Hermitian embedding, compatible FABLE, PrepSelPrep,
qubitization, and caller-supplied circuit specifications where backend
decomposition permits. The lower-level component executor accepts a
`projector_factory` for component-specific custom signal conventions.

```bash
qsvt mixed-parity-synthesis --poly "0.5,0.5"
```

## Workflow synthesis stress regression

Run the repository benchmark from an installed source checkout:

```bash
python scripts/benchmark_synthesis_workflows.py \
  --output /tmp/qsvt-synthesis-workflows.json --repeats 1
```

The 16 cases compare root-finding and iterative synthesis on sign, inverse,
and filter designs through degree 25, Hamiltonian cosine/sine designs through
degree 24, degree-16/32 Chebyshev polynomials at two boundedness margins, and
zero/boundary constants. Reports preserve the original coefficients, conditioning
proxies, solver completion, reconstruction quality, timings, and dependency
versions. Constants exercise the analytic path for either requested solver.
Failures are diagnostic results, not a reason to change coefficients or relax
accuracy thresholds. High-degree monomial conditioning can affect both
boundedness assessment and synthesis. This command writes a new report and
does not refresh the committed notebook snapshots.

In the PennyLane 0.45.1 investigation, iterative reconstruction passed `1e-6`
for all six sign/inverse/filter cases and both Hamiltonian sine cases. The
same run retained the degree-10 root-finding filter residual near `0.075`.
The degree-24 cosine fit exceeded one by roughly `1e-15` and both backends
rejected it; degree-32 Chebyshev cases exposed monomial conditioning and
reconstruction failures. These are environment-specific observations, not
solver guarantees. Regression tests cover the higher-degree iterative successes
without changing their coefficients.
