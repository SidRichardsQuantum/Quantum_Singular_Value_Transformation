# Hamiltonian Simulation Workflow

## Target

`hamiltonian_simulation_workflow(matrix, state, time=..., degree=...)`
approximates real-time evolution

\[
e^{-iHt}|\psi\rangle.
\]

## QSVT Idea

The complex exponential is split into cosine and sine components on the
rescaled Hamiltonian spectrum. QSVT can implement the corresponding bounded
polynomial transformations when a compatible block encoding is available.

## Implementation

The workflow rescales the Hamiltonian, designs cosine and sine polynomial
approximations, reconstructs the complex evolution operator including the
affine phase offset, and compares against an exact dense matrix exponential
reference.

## Diagnostics

The result includes cosine and sine coefficients, polynomial and exact
unitaries, evolved and exact states, state relative error, operator relative
error, scaled time, norm drift, and rescaling metadata.

## Scope

This validates polynomial matrix-function accuracy for small systems. It does
not synthesize an optimized Hamiltonian simulation circuit or account for
Hamiltonian access costs.

## API

```python
from qsvt.algorithms import hamiltonian_simulation_workflow

result = hamiltonian_simulation_workflow(H, psi, time=0.75, degree=18)
report = result.as_report()
```

See also [Time evolution and response](time_evolution_and_response.md) and
[Algorithm notes](algorithms.md).
