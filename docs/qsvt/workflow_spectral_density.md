# Spectral Density Workflow

## Target

`spectral_density_workflow(matrix, centers, width=..., degree=..., state=None)`
estimates Gaussian-window spectral density over selected energy centers.

## QSVT Idea

Smooth window polynomials can approximate localized spectral projectors. With
a block encoding, QSVT would implement those windows as spectral filters.

## Implementation

The workflow rescales the Hermitian matrix, converts each physical center and
width into the scaled coordinate, designs a Gaussian window polynomial for
each center, and evaluates dense trace and optional state-resolved weights.

## Diagnostics

The result includes one coefficient vector per center, polynomial trace
density, exact trace density, trace-density error, optional state weight arrays
and errors, width, centers, and rescaling metadata.

## Scope

This is a small dense trace calculation. It does not implement scalable
density-of-states estimation, quantum trace estimation, or hardware sampling.

## API

```python
from qsvt.algorithms import spectral_density_workflow

result = spectral_density_workflow(H, centers=[-0.5, 0.0, 0.5], width=0.2)
report = result.as_report()
```

See also [Spectral filters](spectral_filters.md),
[Time evolution and response](time_evolution_and_response.md), and
[Algorithm notes](algorithms.md).
