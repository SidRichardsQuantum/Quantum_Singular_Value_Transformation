# Spectral Filters

Spectral filters are polynomial matrix functions that emphasize selected
eigenvalue regions and suppress others. In QSVT language, a compatible
polynomial filter can be applied to a block encoding of a normalized operator.
In this package, the same transform is validated on small dense matrices by
classical eigendecomposition and, where practical, optional PennyLane QSVT
checks.

Concise per-workflow pages:

- [Ground-state filtering workflow](workflow_ground_state_filtering.md)
- [Singular-value filtering workflow](workflow_singular_value_filtering.md)
- [Singular-value pseudoinverse workflow](workflow_singular_value_pseudoinverse.md)
- [Spectral density workflow](workflow_spectral_density.md)
- [Spectral counting workflow](workflow_spectral_counting.md)
- [Spectral thresholding workflow](workflow_spectral_thresholding.md)
- [Fixed-point amplification workflow](workflow_fixed_point_amplification.md)
- [Fermi-Dirac occupation workflow](workflow_fermi_dirac.md)
- [Matrix log and entropy workflow](workflow_matrix_log_entropy.md)

## Filter model

For a Hermitian operator \(H\), write

$$
H = V \operatorname{diag}(\lambda_i) V^\dagger.
$$

A scalar filter \(f\) defines the matrix function

$$
f(H) = V \operatorname{diag}(f(\lambda_i)) V^\dagger.
$$

The package rescales spectra to a QSVT-compatible interval, usually
\([-1, 1]\), then builds a bounded polynomial \(P(x)\) that approximates the
desired filter in the scaled coordinate.

## Ground-state filtering

`qsvt.algorithms.ground_state_filtering_workflow` uses a Gaussian low-energy
window to filter a trial state toward the ground eigenspace:

$$
P(x) \approx \exp\left(-\frac{1}{2}
\left(\frac{x-c}{w}\right)^2\right).
$$

The center \(c\) is chosen near the low-energy edge after spectral rescaling,
and \(w\) controls the width. A narrower window is more selective but typically
requires higher polynomial degree.

The workflow reports the filtered state, filtered energy, ground-state overlap,
state error against the exact dense filter, and operator error.

See [Ground-state filtering workflow](workflow_ground_state_filtering.md).

## Interval projectors

`qsvt.algorithms.spectral_thresholding_workflow` and
`qsvt.design.design_interval_projector_polynomial` approximate a hard spectral
projector onto an interval \([a, b]\). The exact object is

$$
\Pi_{[a,b]} =
\sum_{\lambda_i \in [a,b]} |v_i\rangle\langle v_i|,
$$

but a bounded polynomial can only approximate the discontinuous edges. The
implemented design uses a smooth band-pass profile with tunable sharpness.

Diagnostics include exact rank, trace-based polynomial rank proxy, leakage
outside the target interval, retained state weight when a state is supplied,
and operator error against the hard projector.

See [Spectral thresholding workflow](workflow_spectral_thresholding.md) and
[Spectral counting workflow](workflow_spectral_counting.md).

## Sign and threshold filters

Sign-like polynomials separate positive and negative spectral regions. They are
useful for thresholding, low-energy/high-energy separation, and projector
construction through expressions like

$$
\frac{1 - \operatorname{sign}(x-c)}{2}.
$$

Near the threshold \(c\), the sign function changes abruptly. A finite-degree
bounded polynomial therefore has a transition band. Degree, transition width,
and boundedness should be read together.

## Spectral density windows

`qsvt.algorithms.spectral_density_workflow` evaluates Gaussian windows over
many centers. For centers \(c_k\), the workflow constructs filters

$$
P_k(\lambda) \approx
\exp\left(-\frac{1}{2}\left(\frac{\lambda-c_k}{w}\right)^2\right)
$$

and compares polynomial trace density with the exact dense trace density. When
a state is supplied, it also reports state-resolved spectral weights.

This supports small density-of-states demonstrations and band-selection
examples, but it does not implement stochastic trace estimation for large
systems.

See [Spectral density workflow](workflow_spectral_density.md).

## Design tradeoffs

Filter quality depends mainly on:

- spectral rescaling and where the target window lands in \([-1, 1]\),
- polynomial degree,
- transition width or Gaussian width,
- boundedness margin,
- parity and synthesis constraints when a direct QSVT check is attempted.

Sharp edges, narrow windows, and small spectral gaps are harder. In practice,
the best diagnostic is not a single scalar: inspect sampled approximation
error, leakage, operator error, and the problem-specific state or trace metric.

## Scope

The filter workflows implement dense spectral polynomial validation. A QSVT
interpretation requires a valid block encoding of the scaled operator and a
compatible bounded polynomial. The package does not include scalable oracle
construction, data loading, amplitude amplification, readout, or hardware
execution for these filter workflows.
