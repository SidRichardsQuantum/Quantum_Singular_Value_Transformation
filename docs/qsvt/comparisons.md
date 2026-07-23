# Algorithm Comparisons

`qsvt.comparisons` contains finite implementations of adjacent quantum
algorithms used to put QSVT workflows in context. They are intentionally
separate from the core polynomial-design, block-encoding, synthesis, and QSVT
execution layers.

## HHL

`execute_hhl_circuit` implements the HHL circuit pattern for small
simulator-scale linear systems. Its report records postselection probability,
solution-state agreement, phase-register settings, and resource metadata.

```python
from qsvt.comparisons import execute_hhl_circuit
```

See [Linear systems](linear_systems.md) for the mathematical target, circuit
stages, diagnostics, and omitted scalable-oracle costs.

## Continuous-time quantum-walk search

`quantum_walk_search_workflow` compares exact dense graph-search dynamics with
a polynomial approximation to the best-time propagator.

```python
from qsvt.comparisons import quantum_walk_search_workflow
```

See [Quantum walk search workflow](workflow_quantum_walk_search.md) for the
target, report fields, and scope.

## Compatibility imports

The previous imports remain available during the compatibility window:

- `qsvt.hhl` re-exports the HHL helpers.
- `qsvt.algorithms` continues to re-export the quantum-walk workflow.
- governed package-root names continue to resolve lazily.

New code should import these comparison algorithms from `qsvt.comparisons`.
