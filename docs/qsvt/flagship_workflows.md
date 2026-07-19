# Executable Flagship Workflows

The package includes two end-to-end finite workflows that connect a physical
problem to degree selection, block encoding, phase synthesis, QNode execution,
classical validation, observables, resources, and an error ledger.

## Pauli-Hamiltonian spectral filter

`spectral_filter_qsvt_workflow` accepts a PennyLane Pauli Hamiltonian, an input
state, and a physical energy interval. The Hamiltonian's LCU one-norm is the
block-encoding normalization `alpha`, so the polynomial is designed on
`[lower / alpha, upper / alpha]`. The workflow can use `PrepSelPrep` or
qubitization.

```python
import numpy as np
import pennylane as qml

from qsvt import spectral_filter_qsvt_workflow

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
from qsvt import poisson_qsvt_workflow

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

Both workflows call `estimate_encoding_aware_resources`. When PennyLane's
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
```
