# API Stability

## Frozen facade

`qsvt.stable` is the compact public facade whose signatures and documented
behavior are frozen for the remainder of the `0.x` series. New applications
should prefer:

```python
from qsvt.stable import (
    QSVTExecutionConfig,
    QSVTProblemSpec,
    QSVTTransformSpec,
    design_workflow,
    plan_qsvt,
    run_qsvt_plan,
)
```

The facade contains exactly 20 names:

| area | stable names |
| --- | --- |
| problem and access specifications | `BlockEncodingSpec`, `QSVTExecutionConfig`, `QSVTProblemSpec`, `QSVTTransformSpec` |
| design and planning | `design_workflow`, `qsvt_problem_workflow`, `plan_qsvt`, `run_qsvt_plan` |
| flagship workflows | `poisson_qsvt_workflow`, `spectral_filter_qsvt_workflow`, `hamiltonian_simulation_workflow` |
| realizability and synthesis | `certify_polynomial_boundedness`, `classify_polynomial_realizability`, `synthesize_phases` |
| resources | `estimate_encoding_aware_resources` |
| reporting | `report_to_jsonable`, `save_report`, `load_report_with_schema`, `validate_report_schema`, `supported_report_schemas` |

The result objects returned by these functions remain importable from their
documented modules. Their versioned report schemas, stated truth contracts,
and documented fields are part of the corresponding workflow contract.

## Status tiers

`qsvt.api_status(name)` returns one of three labels:

- `stable`: exported by `qsvt.stable` and frozen through the `0.x` series,
- `compatibility`: a previously stable root or submodule import that remains
  supported but is outside the compact facade,
- `experimental`: a research or lower-level interface that may change between
  minor releases.

The complete manifests are available as `qsvt.STABLE_API_NAMES` and
`qsvt.COMPATIBILITY_API_NAMES`.

## Deprecation policy

A compatibility name may be removed only after:

1. its replacement or removal is announced in the changelog,
2. accessing or calling it emits `DeprecationWarning`,
3. at least two minor releases containing that warning have been published.

Experimental interfaces do not receive that fixed window, but incompatible
changes must still be recorded in the changelog. Versioned report schemas
remain governed by their schema registry and migration policy independently of
the Python import tier.

## Flagship acceptance

API stability does not imply that every workflow has the same quantum
execution scope. See [Executable flagship workflows](flagship_workflows.md)
for the versioned acceptance matrix. Poisson inversion, spectral filtering,
and Hamiltonian simulation now each have a finite-QSVT acceptance path. Their
reports still distinguish finite simulator validation from scalable access,
state preparation, amplitude amplification, readout, and hardware execution.
