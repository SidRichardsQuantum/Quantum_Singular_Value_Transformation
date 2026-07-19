# Accuracy-Driven Workflow Planning

`qsvt.planning` provides a typed path from a finite problem and target error to
an executable QSVT plan. It accepts three operator forms:

- a finite NumPy matrix,
- a PennyLane operator,
- a `BlockEncodingSpec` for a matrix, PennyLane operator, or custom circuit.

The planner searches candidate degrees, runs the existing classical workflow
for each candidate, retains failures and error diagnostics, synthesizes each
single-parity polynomial with solver fallback, selects an access model, and
adds encoding-aware resources. Execution is a separate explicit step.

## Minimal linear-system plan

```python
import numpy as np

from qsvt import (
    QSVTExecutionConfig,
    QSVTProblemSpec,
    QSVTTransformSpec,
    plan_qsvt,
    run_qsvt_plan,
)

problem = QSVTProblemSpec(
    np.diag([1.0, 2.0]),
    rhs=np.array([1.0, 1.0]),
    observables={"population_0": np.diag([1.0, 0.0])},
)
target = QSVTTransformSpec(
    "linear_system",
    tolerance=0.4,
    min_degree=3,
    max_degree=9,
    degree_step=2,
)
config = QSVTExecutionConfig(execute=True)

plan = plan_qsvt(problem, target, config)
run = run_qsvt_plan(plan)
```

The plan report includes every tried degree, the selected numerical error,
coefficient and synthesis reports, the selected block encoding, logical
resources, warnings, and an `execution_ready` flag. The run report adds QNode
results, observables, and separate approximation, phase-reconstruction,
circuit, and sampling error entries.

## Access-model behavior

If a supplied `BlockEncodingSpec` encodes the exact normalized signal operator
used by the selected polynomial workflow, the planner preserves it. If the
normalizations differ, the default policy constructs a finite matrix encoding
for validation and labels it `matrix-fallback`. Set
`allow_matrix_fallback=False` to require the supplied access model instead.

Custom circuit specifications are opaque to classical planning. Supply
`reference_matrix` in `QSVTProblemSpec` so degree selection and validation have
a finite reference. The planner never labels a finite matrix fallback as a
scalable oracle.

## Degree and phase policy

`search_polynomial_degree` is the generic builder/evaluator interface.
`search_design_degree` applies it to public polynomial design targets. During
planning, a degree must meet the workflow error target and, when execution is
requested, every coefficient component must be realizable by one QSVT phase
sequence.

Phase results are cached by coefficients, routine, solver, reconstruction grid,
and solver keyword arguments. Solver fallback continues until a result meets
`phase_reconstruction_tolerance`; if none does, the lowest-reconstruction-error
successful result is retained and its error remains visible.

## CLI

```bash
qsvt plan-workflow --target linear_system --matrix "2,0;0,1" \
  --rhs "1,1" --tolerance 0.2 --no-execute

qsvt degree-search --kind sign --gamma 0.2 --tolerance 0.05 \
  --min-degree 5 --max-degree 25 --degree-step 2
```

The available planner targets are linear systems, spectral projectors,
ground-state filters, Hamiltonian simulation, resolvents, singular-value
filters, and singular-value pseudoinverses.

## Scope

Planning and finite QNode execution do not supply scalable application-state
preparation, amplitude amplification, full-vector tomography, provider-native
compilation, or error correction. Those omissions are recorded in the report
rather than folded into the polynomial or logical gate estimate.
