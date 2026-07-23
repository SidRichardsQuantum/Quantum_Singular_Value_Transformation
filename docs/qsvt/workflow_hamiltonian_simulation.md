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
error, scaled time, norm drift, rescaling metadata, and a versioned flagship
acceptance report.

## Scope

This validates polynomial matrix-function accuracy for small systems and may
be `accepted_for_stated_scope` when the declared numerical tolerance is met.
Its stated scope is `polynomial_core`. `full_qsvt_acceptance` remains false
because the workflow does not coherently combine the even cosine and odd sine
QSVT sequences and does not report a concrete encoding-aware circuit resource
ledger.

## API

```python
from qsvt.stable import hamiltonian_simulation_workflow

result = hamiltonian_simulation_workflow(H, psi, time=0.75, degree=18)
report = result.as_report()
acceptance = report["acceptance"]
```

## CLI

```bash
qsvt hamiltonian-simulation \
  --matrix "0,1;1,0" --state "1,0" \
  --time 0.5 --degree 8 \
  --output hamiltonian-simulation.json
```

The saved report uses the same schema and acceptance contract as
`HamiltonianSimulationWorkflowResult.as_report()`.

See also [Time evolution and response](time_evolution_and_response.md) and
[Algorithm notes](algorithms.md).
