# Fermi-Dirac Occupation Workflow

## Target

`fermi_dirac_occupation_workflow(matrix, chemical_potential=..., beta=..., degree=...)`
approximates finite-temperature Fermi-Dirac occupation of a finite Hamiltonian.

## QSVT Idea

The occupation function

\[
f(E) = \frac{1}{1 + \exp(\beta(E - \mu))}
\]

is a bounded spectral function. A polynomial approximation to this function is
a QSVT-compatible target once a block encoding of the Hamiltonian is supplied.

## Implementation

The workflow rescales the Hamiltonian spectrum, fits a bounded polynomial to
the occupation function, applies it through dense spectral calculus, and
compares the polynomial occupation operator with the exact dense reference.

## Diagnostics

The result includes polynomial and exact occupation operators, particle-number
estimates, operator relative error, coefficients, chemical potential, beta,
rescaling metadata, and optional state occupation/error fields.

## Scope

This is a finite matrix-function validation workflow. It does not implement
electronic-structure data loading, thermal-state preparation, or hardware
sampling.

## API

```python
from qsvt.algorithms import fermi_dirac_occupation_workflow

result = fermi_dirac_occupation_workflow(H, chemical_potential=0.0, beta=4.0)
report = result.as_report()
```

See also [Spectral filters](spectral_filters.md),
[Time evolution and response](time_evolution_and_response.md), and
[Algorithm notes](algorithms.md).
