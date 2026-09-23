# Executable Flagship Workflows

The package includes three end-to-end finite workflows that connect a physical
problem to degree selection, block encoding, phase synthesis, QNode execution,
classical validation, observables, resources, and an error ledger.

Every result report includes a versioned `acceptance` section.

## Tested encoding and execution support

The high-level flagship support matrix below is exercised by
`tests/test_flagship_acceptance.py::test_flagship_encoding_support_matrix`.
Each row runs both statevector and finite-shot execution on `default.qubit`.
This is a finite simulator contract; backend decomposition and physical-device
support require a separate device audit.

| flagship | encoding argument | input and normalization | statevector acceptance | finite shots |
| --- | --- | --- | --- | --- |
| Poisson | `access_model="dense"` | finite Dirichlet matrix; spectral-norm normalization | tested | probabilities and uncertainty |
| Poisson | `access_model="fable"` | real matrix; FABLE entry/dimension normalization | tested | probabilities and uncertainty |
| Poisson | `access_model="prepselprep"` | Pauli decomposition; LCU one-norm; power-of-two point count | tested | probabilities and uncertainty |
| Poisson | `access_model="qubitization"` | Pauli decomposition; LCU one-norm; power-of-two point count | tested | probabilities and uncertainty |
| Spectral filtering | `block_encoding="prepselprep"` | Hermitian Pauli operator; LCU one-norm | tested | probabilities and uncertainty |
| Spectral filtering | `block_encoding="qubitization"` | Hermitian Pauli operator; LCU one-norm | tested | probabilities and uncertainty |
| Hamiltonian simulation | `block_encoding="embedding"` | finite Hermitian matrix; affine spectral scaling | tested | probabilities and uncertainty |
| Hamiltonian simulation | `block_encoding="fable"` | real symmetric matrix; affine scaling widened for FABLE | tested | probabilities and uncertainty |

The matrix tests use a four-point Poisson system, a two-qubit Pauli filter,
and a non-diagonal two-level Hamiltonian with a complex input state. These
are representative acceptance regressions, not guarantees for arbitrary
degree, tolerance, dimension, or conditioning. FABLE can require a higher
polynomial degree because its normalization changes the signal domain. The
Poisson regression uses degree 13 for FABLE and degree 5 for the other encodings.

### Finite-shot scientific acceptance

Acceptance schema `1.2` adds the `finite_shot_probabilities` scope for executed
shot-based runs. `accepted_for_stated_scope` can pass when conditional
computational-basis probabilities agree with the normalized exact workflow
output, including uncertainty. `full_qsvt_acceptance` remains false: measuring
basis probabilities does not validate phases, amplitudes, or statevector error.
Statevector runs retain the existing `finite_qsvt` criteria. Historical `1.0`
and `1.1` reports remain readable and retain their original verdicts.

All three flagship APIs accept `sampling_tolerance=0.05` (maximum absolute
basis-probability error) and `sampling_confidence=0.95`. These are independent
of polynomial, statevector, and phase-reconstruction tolerances. The CLI exposes
`--sampling-tolerance` and `--sampling-confidence`; Studio exposes the same
package-backed controls.

The package recovers integer counts from the measured full-register
probabilities and records total shots, accepted shots, logical postselection
rate, conditional probabilities, exact reference probabilities, simultaneous
confidence intervals, and the maximum probability-error bound. For failure
budget `alpha = 1 - confidence`, the postselection radius is
`sqrt(log(4 / alpha) / (2 * shots))`; each conditional probability radius is
`sqrt(log(4 * dimension / alpha) / (2 * accepted_shots))`. Clipped intervals
use a union bound over all basis outcomes and the postselection rate. This
uses [Hoeffding's inequality](https://doi.org/10.1080/01621459.1963.10500830)
and assumes independent identically distributed shots.

Acceptance requires the entire probability interval to lie within the caller's
error tolerance and a positive lower confidence bound on postselection.
Polynomial accuracy, validated phase synthesis, references, and resources
remain required. Sparse samples cannot pass merely because a wide interval
contains the reference. The evidence distinguishes `accepted`,
`insufficient_shots`, `distribution_mismatch` (a confidence interval excludes
the allowed reference band), `invalid_evidence`, and `unavailable`.

This scope validates the probabilities of basis-state projectors. It does not
certify arbitrary observables, a solution norm, the ideal postselection rate,
or tomography. Poisson and filtering single-sequence circuits can retain a
complex response: successful real-polynomial reconstruction does not guarantee
that their raw probabilities match the normalized physical reference. Such
runs must earn measurement acceptance independently and may fail it even when
the statevector real-part check passes. No imaginary component is discarded
from shot data. Hamiltonian coherent component-LCU measurements include logical
and selector postselection directly.

Example:

```bash
qsvt hamiltonian-simulation --matrix "-0.5,0;0,0.5" --state "1,0" \
  --time 0.3 --degree 5 --acceptance-tolerance 0.001 --shots 10000 \
  --sampling-tolerance 0.05 --sampling-confidence 0.95
```

### Wire, projector, and unsupported-input boundaries

- Poisson and Hamiltonian simulation construct their own encoding wires and
  projectors. The high-level APIs do not accept custom `BlockEncodingSpec`
  objects or caller-supplied projectors.
- Spectral filtering preserves the operator's wire labels and accepts explicit
  encoding wires. The support test uses string labels; encoding wires must be
  distinct and disjoint from system wires. Projectors are inferred from the
  encoding and synthesized PennyLane QSVT phases.
- Custom circuits, explicit projector factories, alternative wire orders,
  and rectangular singular-value transforms belong to the lower-level
  `execute_qsvt_from_spec` / `execute_qsvt_component_lcu_from_spec` contracts.
  Their tests in `tests/test_execution.py` do not imply high-level flagship
  support. Coherent rectangular transforms remain outside the advertised path.
- Unsupported encoding names, overlapping or duplicate filter wires, and
  non-power-of-two Poisson Pauli-LCU dimensions are rejected explicitly.
  An executable encoding does not guarantee that a chosen polynomial can meet
  its approximation or phase-reconstruction tolerance.

## Acceptance matrix

| workflow | stated scope | required acceptance evidence | current full-QSVT boundary |
| --- | --- | --- | --- |
| Poisson inversion | `finite_qsvt` | direct and CG references, tolerance-selected inverse polynomial, validated phases, successful finite QNode, observables, error ledger, encoding-aware resources | scalable right-hand-side preparation, amplification, norm estimation, and readout remain omitted |
| Pauli-LCU spectral filtering | `finite_qsvt` | exact projector, tolerance-selected filter, validated phases, successful finite QNode, success probability, observables, error ledger, encoding-aware resources | application state preparation, amplification, and large-scale measurement remain omitted |
| Hamiltonian simulation | `finite_qsvt` | exact dense exponential, accurate cosine/sine polynomial pair, validated component phases, coherent selector-LCU QNode, bounded norm drift, component error and circuit-resource ledgers | scalable Hamiltonian access, application state preparation, amplification, and large-scale readout remain omitted |

The machine-readable source is
`qsvt.acceptance.flagship_acceptance_matrix()`. Acceptance reports use schema
`qsvt-flagship-acceptance` version `1.2`; historical `1.0` and `1.1` reports remain
readable. `accepted_for_stated_scope` evaluates
only criteria required by the declared scope; `full_qsvt_acceptance` evaluates
all criteria needed for the finite QSVT circuit claim.

## Hamiltonian simulation

`hamiltonian_simulation_workflow` approximates
`exp(-i H t)|psi>` with separate cosine and sine Chebyshev polynomials and
checks the result against an exact dense matrix exponential. It normalizes and
synthesizes both components, extracts each real polynomial through
`(U + U_adjoint) / 2`, coherently applies the complex cosine/sine weights,
uncomputes the selector, and measures the finite result. The CLI emits the same
schema-versioned report and acceptance summary:

```bash
qsvt hamiltonian-simulation \
  --matrix "0,1;1,0" --state "1,0" \
  --time 0.5 --degree 8
```

The report records selector and logical postselection probabilities, phase and
circuit errors, LCU normalization, selector ancillas, forward/adjoint signal
calls, and actual finite-circuit resources. It can reach
`full_qsvt_acceptance = true` for its finite matrix-encoding scope; scalable
Hamiltonian access, application state preparation, amplitude amplification,
and readout remain omitted.

For FABLE, the workflow widens the affine signal scale to satisfy its
entry/dimension normalization requirement before designing cosine and sine.
The physical Hamiltonian, evolution time, and dense reference remain the same;
the reported `scaled_operator.scale` and `scaled_time` reflect the encoding.

## Pauli-Hamiltonian spectral filter

`spectral_filter_qsvt_workflow` accepts a PennyLane Pauli Hamiltonian, an input
state, and a physical energy interval. The Hamiltonian's LCU one-norm is the
block-encoding normalization `alpha`, so the polynomial is designed on
`[lower / alpha, upper / alpha]`. The workflow can use `PrepSelPrep` or
qubitization.

```python
import numpy as np
import pennylane as qml

from qsvt.stable import spectral_filter_qsvt_workflow

hamiltonian = qml.dot(
    [0.4, 0.3, 0.2],
    [qml.Z(0), qml.Z(1), qml.X(0)],
)
result = spectral_filter_qsvt_workflow(
    hamiltonian,
    np.ones(4) / 2,
    lower=-0.4,
    upper=0.4,
    tolerance=0.16,
    min_degree=2,
    max_degree=4,
)
```

The tolerance is evaluated against the hard spectral projector in relative
operator norm, not only against the smooth fitting target. The result also
reports the postselected reference and polynomial states, success
probabilities, requested observables, phase reconstruction error, finite-QNode
agreement, and logical Pauli-LCU resource costs.

For finite shots, diagonal observables can be recovered from conditional
logical probabilities. General observable values require the statevector path
or a separate measurement strategy.

CLI equivalent:

```bash
qsvt spectral-filter-qsvt \
  --pauli-terms "0.4:ZI,0.3:IZ,0.2:XI" \
  --state "0.5,0.5,0.5,0.5" \
  --lower -0.4 --upper 0.4 --tolerance 0.16
```

## Poisson linear system

`poisson_qsvt_workflow` discretizes the one-dimensional Dirichlet problem
`-u'' = f`, then compares four layers:

1. a dense direct solve,
2. conjugate gradients,
3. a bounded positive-inverse polynomial,
4. a finite block-encoded QSVT circuit when requested.

```python
from qsvt.stable import poisson_qsvt_workflow

result = poisson_qsvt_workflow(
    4,
    tolerance=0.4,
    min_degree=5,
    max_degree=5,
    access_model="prepselprep",
)
```

Supported access models are dense unitary embedding, FABLE, PrepSelPrep, and
qubitization. Pauli-LCU access uses a Pauli decomposition and currently
requires the number of interior points to be a power of two. Dense and FABLE
paths remain finite matrix constructions; they are not sparse-oracle claims.

The default sine source has an analytic continuum solution, allowing the report
to separate discretization error from polynomial, phase, circuit, and sampling
errors. It also reports the residual, condition number, solution integral, and
source-solution energy. The circuit's full solution vector is simulator
validation data; scalable state preparation, amplitude amplification,
solution-norm estimation, and tomography are omitted.

CLI equivalent:

```bash
qsvt poisson-qsvt --n-points 4 --tolerance 0.4 \
  --access-model prepselprep
```

## Encoding-aware resources

The Poisson and spectral-filter workflows call
`estimate_encoding_aware_resources`. When PennyLane's
logical estimator is available, Pauli Hamiltonians are costed with a
Pauli-LCU/qubitization model and matrix/custom sources with an explicit generic
unitary model. Reports include normalization, forward and adjoint query counts,
wire and logical gate totals, gate types, model assumptions, and omitted costs.

These are logical algorithm estimates. They are neither executed measurement
counts nor fault-tolerant resource estimates, and they exclude application
state preparation, postselection or amplitude amplification, readout,
routing, and error correction.

## Cookbook scripts

```bash
python examples/spectral_filter_qsvt.py \
  --output /tmp/qsvt-spectral-filter.json
python examples/poisson_qsvt.py --output /tmp/qsvt-poisson.json
python examples/hamiltonian_simulation.py \
  --output /tmp/qsvt-hamiltonian-simulation.json
```

Each script prints the persisted report path and a compact acceptance summary.
The Hamiltonian script reports
`accepted_for_stated_scope (scope=finite_qsvt, full_qsvt=True)` when its
finite circuit and numerical criteria pass.
