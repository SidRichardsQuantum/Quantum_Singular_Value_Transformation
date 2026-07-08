# Ground-State Filtering Workflow

## Target

`ground_state_filtering_workflow(matrix, state, degree=...)` filters a trial
state toward the low-energy eigenspace of a finite Hermitian operator.

## QSVT Idea

A bounded low-energy window polynomial can suppress high-energy eigencomponents
while retaining low-energy components. With a block encoding, QSVT would apply
the same polynomial to the encoded Hamiltonian.

## Implementation

The Hamiltonian is rescaled to the QSVT design interval, a Gaussian low-energy
window is fitted, and the polynomial matrix function is applied through dense
eigendecomposition. The result is compared with the exact Gaussian-filtered
reference.

## Diagnostics

The result reports the normalized filtered state, unnormalized state,
exact-reference filtered state, filtered energy, ground-state overlap, state
error, operator error, coefficients, and rescaling metadata.

## Scope

This is a finite spectral filtering workflow. It does not prove efficient
ground-state preparation for large systems and does not include state
preparation, block encoding, postselection, or hardware execution.

## API

```python
from qsvt.algorithms import ground_state_filtering_workflow

result = ground_state_filtering_workflow(H, psi, degree=24)
report = result.as_report()
```

See also [Spectral filters](spectral_filters.md) and [Algorithm notes](algorithms.md).
