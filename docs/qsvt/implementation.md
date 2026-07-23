# Implementation Notes

This package separates dense references, small matrix verification, and QNode
execution. The implementation is organized so notebooks, tests, and CLI reports
can share reusable helpers without blurring which layer actually ran.

## Coefficient Conventions

Most public polynomial helpers use monomial coefficients in ascending order:

```text
[c0, c1, c2] -> c0 + c1*x + c2*x^2
```

Chebyshev fitting is used internally for stable approximation on bounded
intervals, then converted to monomial coefficients when a public polynomial is
returned. This keeps downstream evaluation simple and matches NumPy's
`numpy.polynomial.polynomial.polyval` convention.

## Boundedness And Parity

QSVT-compatible scalar polynomials must satisfy structural constraints,
including boundedness on `[-1, 1]` and parity constraints tied to degree.

The package separates three ideas:

- **approximation quality**: sampled error against a target function,
- **boundedness**: sampled maximum absolute value on a design or compatibility
  interval,
- **synthesis compatibility**: whether PennyLane accepts the coefficients for
  QSVT phase synthesis.

`qsvt.compatibility.qsvt_compatibility_report` collects these checks without
pretending they are formal proofs.

## Spectral Rescaling

QSVT polynomials act on normalized spectra. The rescaling helpers store the
affine map explicitly:

```text
scaled = (matrix - offset * I) / scale
```

The main conventions are:

- Hermitian spectra are mapped to `[-1, 1]` for general matrix functions.
- Positive-semidefinite spectra are divided by the largest eigenvalue for
  positive inverse workflows.
- Cutoff-centered rescaling is available for low-energy projector examples.

Each helper returns a `ScaledOperator` containing the scaled matrix, offset,
scale, and original eigenvalue bounds.

## Classical References

The dense reference path uses eigendecomposition:

```text
A = V diag(lambda_i) V^*
f(A) = V diag(f(lambda_i)) V^*
```

This gives exact small-matrix references for polynomial transforms, matrix
functions, state errors, operator errors, and density-matrix comparisons. These
references are intentionally classical; they make the notebook and regression
tests transparent.

## PennyLane QSVT Paths

The QSVT wrappers use PennyLane where practical to build explicit small
operators and compare the top-left block or transformed output against the
classical polynomial reference.

The implementation treats PennyLane synthesis as a validation path, not as the
only source of truth. Report commands can allow QSVT failure and still record
classical polynomial behavior, which is useful for studying polynomials that
are educational but not accepted by a synthesis backend.

## Circuit Execution Path

`qsvt.execution.execute_qsvt_circuit` is the circuit-level path. It prepares a
logical input state inside a PennyLane QNode, queues `qml.qsvt`, and measures
either the final statevector (`shots=None`) or output probabilities (finite
shots).

This path is still simulator-scale when the input operator is an explicit dense
matrix, but it is a true circuit execution model for the implemented finite
instance:

- it uses QNode state preparation,
- it queues a PennyLane QSVT operation,
- it measures through the QNode,
- it records circuit resource metadata from PennyLane specs,
- it keeps the dense polynomial output as a validation reference only.

The execution helper deliberately does not call `qml.matrix` internally. Tests
guard that boundary so the circuit path cannot silently regress into explicit
unitary extraction.

The circuit report still marks `is_end_to_end_quantum_algorithm = false`
because scalable block-encoding construction, problem-specific state
preparation cost, postselection/amplitude amplification, readout/tomography,
and hardware compilation are separate layers.

## Hardware-Oriented Device Execution

`qsvt.hardware.execute_qsvt_on_device` is the finite-shot device-facing path.
It accepts a caller-created PennyLane device and a caller-supplied preparation
function, performs `qsvt_hardware_preflight`, queues lower-level `qml.QSVT`,
and returns probabilities only.

The helper requires a positive finite shot count and does not request
statevectors. Preflight checks verify wire coverage, advertised operation and
measurement support when a device exposes those capabilities, and reject
`StatePrep` by default so hardware examples use explicit preparation circuits.

Hardware execution payloads use schema name `hardware-qsvt-execution`, schema
version `1.0`, and record:

- preflight pass/fail details,
- provider, backend, plugin, fake-backend, native-gate, and shot-limit metadata
  when exposed by device attributes or `capabilities()`,
- logical gate, depth, phase, signal-call, wire, and shot metadata,
- finite-shot success-probability standard errors,
- provider-native compilation fields as explicit `None`/`not_requested`
  placeholders.

`qsvt_provider_plugin_report` is the credential-free metadata extractor used by
preflight and execution reports. It can inspect local fake backends in tests and
CI without importing provider plugins as required dependencies. If a fake or
provider-backed device advertises native operations or shot limits, preflight
uses that information to reject incompatible circuits before execution.

`qsvt_hardware_circuit_report` is the non-executing audit path. It constructs
the logical hardware QSVT tape, attempts PennyLane decomposition, records
logical and decomposed operation sequences, compares both against advertised
native operations, and reports schema name `hardware-qsvt-circuit`. This is the
recommended pre-live-backend workflow for checking whether unsupported logical
operations such as `QSVT` decompose into operations accepted by a fake or
provider-shaped backend.

Provider credentials, paid job submission, native compilation, job persistence,
calibration capture, mitigation, and provider-specific result objects remain
outside the portable package report.

## Reports And JSON

Diagnostics often contain NumPy arrays, complex numbers, and NumPy scalars.
`qsvt.reports.report_to_jsonable` recursively converts those values to plain
JSON-safe containers.

Report-oriented APIs follow this pattern:

1. compute a dictionary or dataclass report,
2. convert arrays/scalars only at the serialization boundary,
3. keep numerical arrays available for notebooks and tests.

The CLI mirrors this behavior: report commands print full JSON by default, and
write compact stdout summaries when `--output` or `--plot` is used.

## Algorithm Truth Contracts

High-level workflow, direct QSVT comparison, resource, and benchmark reports
include a `truth_contract` field generated by shared report helpers. The field
is intentionally machine-readable so downstream notebooks, benchmark tables,
and papers can preserve the same claim boundary as the code:

- `implementation_kind` is `dense-spectral-polynomial-workflow`.
- `execution_tier` distinguishes a dense `polynomial_core` from a verified
  finite `qsvt_circuit`; QNode and physical-device execution are recorded
  separately.
- `truth_status` is derived from the exact polynomial evidence rather than
  being a single generic value.
- `polynomial_evidence` records each component's coefficients, design and QSVT
  certification domains, output prefactor, boundedness certificate, parity,
  realizability class, and parity-decomposition requirement.
- `resource_completeness` remains `partial` while required quantum layers are
  assumed or omitted.
- `is_end_to_end_quantum_algorithm` is `false`.
- `implemented_components` list the dense numerical operations actually run.
- `assumed_quantum_components` and `omitted_quantum_costs` list the layers that
  must be supplied separately before making a full quantum-algorithm claim.
- `pennylane_qsvt_check` records whether that run attempted and succeeded at a
  small backend QSVT block check.

Truth-contract semantic audits independently recompute the polynomial
classification and reject circuit or hardware tiers that contradict the
reported execution and realizability artifacts.

Stable `qsvt.algorithms` result reports also share the
`qsvt-algorithm-workflow` schema. New reports use version `1.1`, whose truth
contract guarantees artifact-derived execution-tier and polynomial evidence;
version `1.0` remains supported for reading. The `mode` field identifies the
concrete workflow while the common schema envelope guarantees
`implementation_kind` and `truth_contract` metadata across workflow families.

Direct `qsvt-transform-report` and `qsvt-matrix-transform-report` payloads use
`implementation_kind = "pennylane-small-qsvt-verification"` when they compare
an explicit small QSVT block against a classical polynomial reference.
Circuit execution payloads use an implementation kind beginning with
`pennylane-qnode-...` and mean that a QNode was executed, not that a scalable
problem oracle or hardware deployment was supplied.
Hardware execution payloads use
`pennylane-device-finite-shot-qsvt-execution` and mean that a finite-shot
probability circuit was submitted to the caller-supplied PennyLane device after
local preflight checks.
Classical benchmark payloads use
`implementation_kind = "classical-baseline-with-optional-qsvt-proxy"` and mark
`is_quantum_runtime_benchmark = false`.

This is stricter than prose-only documentation: if a report is serialized, the
assumptions travel with the numbers.

## Resource Proxy Reports

`qsvt.resources.qsvt_resource_report` combines polynomial degree, coefficient
count, QSP phase-count proxy, signal-call proxy, optional matrix-register
width, QSVT compatibility metadata, and optional diagnostics into one
JSON-friendly report.

Design workflow results expose the same summary through
`DesignWorkflowResult.resource_report(...)`, which carries the workflow's
diagnostics into the resource report.

The CLI exposes the same path:

```bash
qsvt resource-report --poly "0,0,1" --matrix-dimension 4 --no-synthesis
```

These reports are intentionally comparative and simulator-scale. They do not
include block-encoding construction, state preparation, amplitude
amplification, error correction, hardware compilation, or runtime estimates.

## Notebooks And Artefacts

Notebooks remain executable examples. Committed plots and JSON files are release
artefacts extracted from those workflows or generated by the CLI.

The intended split is:

- notebooks explain and visualize,
- package helpers implement reusable operations,
- tests protect small deterministic behavior,
- `results/` stores stable release artefacts,
- docs summarize and link to the artefacts.

## Error Metrics

The package uses simple diagnostics:

- maximum sampled scalar error,
- RMS sampled scalar error,
- relative operator Frobenius error,
- phase-aligned relative state error,
- residual norms for linear systems,
- overlap and expectation diagnostics for state filtering,
- trace-density and density-matrix relative errors.

These metrics are for small-matrix validation. They should not be read as
asymptotic resource estimates.

## Public API Status

The package root exposes `__api_status__ = "alpha"`, a
`__public_api_policy__` string, and `qsvt.api_status(name)`. The exact frozen
surface lives in `qsvt.stable`; existing names in `qsvt.__all__` remain
available for compatibility or experimental use.

`qsvt.api_status(name)` returns:

- `stable` for the 20 names exported by `qsvt.stable`,
- `compatibility` for previously stable imports that remain supported but are
  outside the compact frozen facade,
- `experimental` for lower-level circuit execution, backend-adapter helpers,
  hardware-oriented device helpers,
  and any exported name that has not yet been explicitly promoted.

Notebook presentation helpers live in the `qsvt.notebook` submodule. They are
shipped with the package so committed notebooks can import them, but they are
not re-exported from the package root in `0.2.8`. Treat them as experimental
notebook support rather than stable algorithm APIs.

Compatibility names cannot be removed until the changelog announces the
deprecation and the package has emitted `DeprecationWarning` for at least two
minor releases. See [API stability](stability.md) for the complete policy and
stable-name list.

The local release preflight checks code, types, tests, cookbook integration,
docs, package build metadata, and a built-wheel smoke test. It can also include
all executable notebooks:

```bash
python scripts/release_check.py --no-build-isolation --include-notebooks
```

Notebook subprocess examples use `sys.executable` so CLI checks run with the
same environment as the notebook kernel.

The built-wheel smoke step installs the wheel in a temporary virtual
environment, checks import metadata and `py.typed`, validates representative
API-status labels, runs `qsvt --help`, and executes a minimal scalar CLI
command. See [Release checklist](releasing.md) for the full release process.

## Scope Boundaries

The implementation intentionally does not provide:

- production-scale block encoding construction,
- problem-specific scalable state preparation methods,
- amplitude amplification,
- fault-tolerant synthesis,
- production-scale hardware optimization.

Those are separate engineering layers. The package now provides circuit-level
QNode execution for finite instances plus the polynomial and spectral mechanics
that larger systems would use. The hardware-facing layer supports small
finite-shot circuits on caller-provided devices with caller-provided
preparation circuits, local preflight checks, and provider-shaped fake-backend
metadata capture. Native provider compilation, live hardware job metadata,
scalability, fault tolerance, and quantum advantage claims remain outside the
implemented scope.
