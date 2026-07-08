# Resolvent Workflow

## Target

`resolvent_workflow(matrix, omega=..., eta=..., degree=..., source=None)`
approximates a Green's-function or resolvent operator

\[
(\omega + i\eta - H)^{-1}.
\]

## QSVT Idea

The complex response is represented by real and imaginary scalar functions on
the Hamiltonian spectrum. Each part can be approximated by a bounded
polynomial and interpreted as a QSVT-compatible matrix-function target.

## Implementation

The workflow rescales the Hermitian matrix, translates \(\omega\) and \(\eta\)
into the scaled coordinate, fits real and imaginary response polynomials, and
compares the polynomial operator with the exact dense resolvent.

## Diagnostics

The result reports real and imaginary coefficients, polynomial and exact
resolvent operators, operator relative error, rescaling metadata, and optional
source-vector response/error fields.

## Scope

This is a finite response-function validation path. Small \(\eta\) produces
sharp spectral features and usually requires higher degree. The workflow does
not provide a scalable linear-system or response-estimation algorithm.

## API

```python
from qsvt.algorithms import resolvent_workflow

result = resolvent_workflow(H, omega=0.2, eta=0.1, degree=24)
report = result.as_report()
```

See also [Time evolution and response](time_evolution_and_response.md) and
[Algorithm notes](algorithms.md).
