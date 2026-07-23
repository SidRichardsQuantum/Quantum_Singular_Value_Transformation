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

The workflow rescales the Hamiltonian, designs and synthesizes cosine and sine
polynomial approximations, and coherently combines their QSVT sequences with
the complex coefficients required by the affine phase offset. A selector LCU
uses each forward and adjoint sequence to extract the real polynomial, then
uncomputes and postselects the selector. The recovered finite-circuit output is
compared against both the polynomial evolution and an exact dense matrix
exponential reference.

## Diagnostics

The result includes cosine and sine coefficients and synthesis reports,
polynomial and exact unitaries, evolved and exact states, approximation and
circuit errors, selector and logical success probabilities, scaled time, norm
drift, rescaling metadata, component resources, and a versioned flagship
acceptance report.

## Scope

This is a finite-QSVT validation path for small systems and may reach
`full_qsvt_acceptance` when its polynomial, synthesis, coherent-circuit, norm,
and resource criteria pass. The finite matrix block encoding is not a scalable
Hamiltonian-access claim. Application state preparation, amplitude
amplification, large-scale readout, fault-tolerant synthesis, and hardware
compilation remain omitted.

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
