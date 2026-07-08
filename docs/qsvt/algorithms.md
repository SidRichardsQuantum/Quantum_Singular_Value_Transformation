# Algorithm Notes

The high-level functions in `qsvt.algorithms` are simulator-scale workflows.
They combine spectral rescaling, bounded polynomial design, direct classical
matrix-function references, and diagnostics. They are intended to make the
algorithmic pattern inspectable, not to hide state preparation, block encoding,
or resource-estimation costs.

Each workflow returns a frozen dataclass with numerical fields and an
`as_report()` method. Reports can be passed through
`qsvt.reports.report_to_jsonable` or `qsvt.reports.save_report`.

```{toctree}
:hidden:
:maxdepth: 1
:caption: Workflow Theory

Linear System Workflow <workflow_linear_system>
Linear System Comparison Workflow <workflow_linear_system_comparison>
Singular-Value Filtering Workflow <workflow_singular_value_filtering>
Singular-Value Pseudoinverse Workflow <workflow_singular_value_pseudoinverse>
Block-Encoded QSVT Workflow <workflow_block_encoded_qsvt>
Ground-State Filtering Workflow <workflow_ground_state_filtering>
Hamiltonian Simulation Workflow <workflow_hamiltonian_simulation>
Quantum Walk Search Workflow <workflow_quantum_walk_search>
Fixed-Point Amplification Workflow <workflow_fixed_point_amplification>
Resolvent Workflow <workflow_resolvent>
Spectral Density Workflow <workflow_spectral_density>
Spectral Counting Workflow <workflow_spectral_counting>
Spectral Thresholding Workflow <workflow_spectral_thresholding>
Fermi-Dirac Occupation Workflow <workflow_fermi_dirac>
Thermal Gibbs Workflow <workflow_thermal_gibbs>
Matrix Log and Entropy Workflow <workflow_matrix_log_entropy>
```

## Workflow Pages

The workflow-specific pages below give each major algorithm path a succinct
target, QSVT idea, implementation summary, diagnostics guide, scope statement,
and minimal API example.

| workflow | page | family |
| --- | --- | --- |
| `linear_system_workflow` | [Linear system workflow](workflow_linear_system.md) | linear systems |
| `linear_system_comparison_workflow` | [Linear system comparison workflow](workflow_linear_system_comparison.md) | linear systems |
| `singular_value_filtering_workflow` | [Singular-value filtering workflow](workflow_singular_value_filtering.md) | singular-value transforms |
| `singular_value_pseudoinverse_workflow` | [Singular-value pseudoinverse workflow](workflow_singular_value_pseudoinverse.md) | singular-value transforms |
| `block_encoded_qsvt_workflow` | [Block-encoded QSVT workflow](workflow_block_encoded_qsvt.md) | finite QSVT execution |
| `ground_state_filtering_workflow` | [Ground-state filtering workflow](workflow_ground_state_filtering.md) | spectral filters |
| `hamiltonian_simulation_workflow` | [Hamiltonian simulation workflow](workflow_hamiltonian_simulation.md) | time evolution |
| `quantum_walk_search_workflow` | [Quantum walk search workflow](workflow_quantum_walk_search.md) | time evolution/search |
| `fixed_point_amplification_workflow` | [Fixed-point amplification workflow](workflow_fixed_point_amplification.md) | amplification/filtering |
| `resolvent_workflow` | [Resolvent workflow](workflow_resolvent.md) | response functions |
| `spectral_density_workflow` | [Spectral density workflow](workflow_spectral_density.md) | spectral filters |
| `spectral_counting_workflow` | [Spectral counting workflow](workflow_spectral_counting.md) | spectral filters |
| `spectral_thresholding_workflow` | [Spectral thresholding workflow](workflow_spectral_thresholding.md) | spectral filters |
| `fermi_dirac_occupation_workflow` | [Fermi-Dirac occupation workflow](workflow_fermi_dirac.md) | thermal/electronic |
| `thermal_gibbs_workflow` | [Thermal Gibbs workflow](workflow_thermal_gibbs.md) | thermal weighting |
| `matrix_log_entropy_workflow` | [Matrix log and entropy workflow](workflow_matrix_log_entropy.md) | matrix functions |

## Common Pattern

Most Hermitian workflows follow the same structure:

1. validate a small dense Hermitian input,
2. rescale the spectrum to a polynomial design interval,
3. build one or more bounded polynomial approximations,
4. apply the polynomial classically through eigendecomposition,
5. compare against an exact spectral reference,
6. return diagnostics that quantify approximation error.

The QSVT interpretation is that a block encoding of the scaled operator would
allow the same polynomial transform to be implemented as a quantum signal
processing sequence, provided the polynomial satisfies the necessary
boundedness and parity constraints.

For focused theory by algorithm family, see [Linear systems](linear_systems.md),
[Spectral filters](spectral_filters.md), and
[Time evolution and response](time_evolution_and_response.md).

The singular-value workflows are the rectangular/non-Hermitian exception to
this Hermitian pattern. They use dense SVD validation: normalize singular
values, apply a polynomial to those singular values, and compare against a
dense SVD reference.

`block_encoded_qsvt_workflow` is the package's finite block-encoded exception
to the usual dense-polynomial-only pattern. It constructs an explicit dense
unitary block encoding, verifies the top-left block and unitarity numerically,
then applies the PennyLane QSVT matrix transform to the normalized positive
Hermitian signal operator. It is still a finite simulator workflow: scalable
oracle construction, state loading, readout, amplitude amplification, and
hardware costs remain outside the implementation.

## Truth Contract

Every `as_report()` payload from this module includes a `truth_contract` field.
That field is part of the public reporting surface and should be read before
using a workflow result in a benchmark, paper draft, or comparison table.

The contract states that the workflow is a dense spectral-polynomial
implementation, not an end-to-end quantum algorithm. It records:

- the concrete target being approximated,
- the implemented pieces: validation, spectral rescaling, polynomial
  construction/application, dense classical reference, and numerical
  diagnostics,
- whether a PennyLane QSVT block check was attempted for that run,
- the conditional QSVT statement: a compatible polynomial is the transform core
  once a valid block encoding and state-preparation model are supplied,
- omitted quantum layers such as block-encoding cost, state preparation,
  success-probability management, readout, amplitude amplification,
  fault-tolerant synthesis, and hardware compilation.

This makes the workflows useful for real physics/math studies: users can plug
in concrete finite models, measure approximation error and degree requirements,
and compare classical baselines, while keeping the missing quantum data-access
and hardware assumptions explicit.

## Workflow Families

Use the workflow pages above for per-algorithm target functions,
implementation details, diagnostics, and scope boundaries. The family pages
provide broader mathematical context:

- [Linear systems](linear_systems.md) covers inverse-polynomial workflows,
  solver comparisons, and finite HHL circuit execution.
- [Spectral filters](spectral_filters.md) covers ground-state filtering,
  interval projectors, sign/threshold filters, spectral density windows,
  singular-value filters, and related projector-style workflows.
- [Time evolution and response](time_evolution_and_response.md) covers
  Hamiltonian simulation, resolvents, quantum walk search, imaginary-time
  evolution, thermal weighting, and Fermi-Dirac occupations.
- [Block encodings](block_encoding.md) covers finite dense block encodings,
  access-model specifications, and block-encoded QSVT execution.

## Choosing Degree

Degree controls the approximation space. Higher degree usually reduces error,
but the improvement can be non-monotonic for sampled maximum error, especially
near sharp transitions. Use:

- `design_workflow` for one polynomial with diagnostics,
- `design-sweep` for compact degree comparisons,
- notebook plots for visual inspection of transition regions,
- workflow regression tests for fixed small-matrix behavior.

The committed result artefacts under `results/reports/` and
`results/tables/design_sweep_summary.csv` show the release-time sweep pattern.
