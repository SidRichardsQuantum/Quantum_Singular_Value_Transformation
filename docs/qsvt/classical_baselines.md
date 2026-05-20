# Classical Baseline Details

The benchmark helpers in `qsvt.benchmarks` are intentionally small classical
references. They make the comparison target explicit before attaching QSVT
resource proxies such as polynomial degree, signal-operator calls, and matrix
register width.

These baselines are not tuned benchmark suites. They are reproducible
small-matrix measurements for asking whether a QSVT-style polynomial transform
is being compared against the right classical operation.

## Baseline Summary

| helper | classical operation | typical cost model | QSVT comparison role |
| --- | --- | --- | --- |
| `dense_eigendecomposition_benchmark` | `numpy.linalg.eigh` on a dense Hermitian matrix | cubic dense linear algebra | exact spectral reference for matrix functions |
| `dense_linear_solve_benchmark` | `numpy.linalg.solve` on a dense system | cubic dense linear algebra | strong small-system baseline for inverse-polynomial workflows |
| `conjugate_gradient_benchmark` | pure-NumPy conjugate gradients | iteration count times matrix-vector cost | sparse positive-definite baseline for linear systems |
| `spectral_matrix_function_benchmark` | eigendecompose then apply a scalar function to eigenvalues | eigendecomposition dominated | exact small-system reference for non-polynomial functions |
| `polynomial_matrix_function_benchmark` | apply a polynomial through spectral functional calculus | eigendecomposition dominated in this implementation | classical reference for the same polynomial used by QSVT |

## Dense Eigensolvers

Dense eigendecomposition is the cleanest exact reference for the examples in
this repository. It exposes the spectrum, eigenvectors, reconstruction error,
and spectral width without requiring sparse matrix infrastructure.

What it measures:
: Wall-clock time for dense Hermitian eigendecomposition and reconstruction
  diagnostics.

What it assumes:
: The full matrix is already available classically in dense memory.

Why it matters:
: Many QSVT demonstrations reduce to applying a function to eigenvalues. Dense
  eigendecomposition is the most direct small-scale reference for that action.

Where it is unfair to QSVT:
: It assumes dense classical access to all matrix entries and does not model
  large sparse or oracle-access settings where block encodings could be
  natural.

## Dense Linear Solves

Dense direct solves are strong baselines for the toy linear systems in the
notebooks. They avoid iterative convergence questions and provide residual
norms and condition-number metadata.

What it measures:
: Wall-clock time for `numpy.linalg.solve`, residual norm, relative residual,
  solution norm, and 2-norm condition number.

What it assumes:
: A dense matrix and right-hand side are already available in memory.

QSVT link:
: The associated proxy usually comes from an inverse-like polynomial whose
  degree is related to the spectral gap or condition number after rescaling.

## Conjugate Gradients

Conjugate gradients are the relevant classical baseline for positive-definite
linear systems when matrix-vector products are cheaper than dense
factorization.

This repository's implementation is pure NumPy and uses dense matrix-vector
products for transparency. The reported `matvec_count` is still useful because
it separates iteration behavior from the dense test harness.

What it measures:
: Iterations, convergence status, residuals, condition number, and repeated-run
  timing.

What it assumes:
: Hermitian positive-definite input and a tolerance target.

QSVT link:
: Both CG and QSVT-style inverse transforms are sensitive to spectral gaps and
  condition number. The benchmark table keeps those quantities visible.

## Dense Spectral Matrix Functions

The spectral matrix-function baseline diagonalizes the Hermitian input and
evaluates a scalar function on its eigenvalues. It is used for functions such
as exponentials before comparing with polynomial approximations.

What it measures:
: Dense spectral functional calculus for a named scalar function.

What it assumes:
: Full dense access and an exactly evaluable scalar function.

QSVT link:
: QSVT applies bounded polynomials. This baseline supplies the exact target
  matrix function that polynomial designs approximate.

## Polynomial Matrix Evaluation

The polynomial baseline applies the same polynomial transform that a QSVT
sequence would implement, but classically through eigendecomposition.

What it measures:
: Output norm and timing for applying a supplied polynomial to a Hermitian
  matrix.

What it assumes:
: Dense eigendecomposition rather than a quantum block encoding.

QSVT link:
: This is the closest apples-to-apples functional comparison for a fixed
  polynomial. The remaining difference is the implementation model, not the
  mathematical transform.

## Interpreting Timings

Microbenchmarks in this repository are intentionally small and should not be
read as hardware-performance claims. The useful release artefacts are:

- problem type and matrix dimension,
- residual or approximation diagnostics,
- condition-number and degree metadata,
- QSVT proxy fields when a polynomial is supplied,
- rough timing scale for local reproducibility.

The benchmark notebooks should be used to identify regimes and missing costs,
not to claim end-to-end quantum speedups.

## Related Pages

- [Classical benchmarks](benchmarks.md)
- [Benchmark notebook outputs](benchmark_results.md)
- [QSVT resource model](qsvt_resource_model.md)
