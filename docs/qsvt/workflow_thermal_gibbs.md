# Thermal Gibbs Workflow

## Target

`thermal_gibbs_workflow(matrix, beta=..., degree=..., state=None)`
approximates imaginary-time or Boltzmann weighting and the normalized Gibbs
density matrix.

## QSVT Idea

The Boltzmann factor \(\exp(-\beta H)\) is a spectral matrix function. On a
rescaled interval it can be approximated by a bounded polynomial and treated as
a QSVT-style transformation target.

## Implementation

The workflow rescales the Hamiltonian, designs a bounded imaginary-time
polynomial with a stored prefactor, applies it by dense spectral calculus, and
normalizes the resulting Boltzmann operator to a Gibbs density matrix.

## Diagnostics

The result includes polynomial and exact Boltzmann operators, normalized Gibbs
states, partition functions, operator relative error, density-matrix relative
error, optional weighted-state error, coefficients, beta, and rescaling
metadata.

## Scope

This is a matrix-function validation path for small finite systems. It does
not implement quantum thermal-state preparation or sampling from the Gibbs
state.

## API

```python
from qsvt.algorithms import thermal_gibbs_workflow

result = thermal_gibbs_workflow(H, beta=2.0, degree=20)
report = result.as_report()
```

See also [Time evolution and response](time_evolution_and_response.md) and
[Algorithm notes](algorithms.md).
