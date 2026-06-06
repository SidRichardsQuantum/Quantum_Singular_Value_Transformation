# QSVT Resource Model

The resource reports in this repository are proxy reports for small
QSVT-style studies. They track the pieces that are visible from a polynomial
and a matrix dimension, while deliberately excluding costs that require a
specific quantum data-access and hardware model.

## Reported Quantities

| field | meaning |
| --- | --- |
| degree | highest nonzero polynomial degree |
| coefficient count | number of supplied polynomial coefficients |
| phase-count proxy | approximate number of QSP/QSVT phase parameters |
| signal-operator calls | proxy for uses of the block-encoded signal operator |
| matrix-register width | `ceil(log2(matrix_dimension))` when a dimension is supplied |
| compatibility fields | sampled boundedness, parity, and optional PennyLane synthesis status |

These quantities are useful because degree usually dominates the number of
signal uses in an idealized QSVT sequence. They are not enough to determine
runtime by themselves.

Every resource report includes a `truth_contract` field with
`implementation_kind = "polynomial-resource-proxy"` and
`is_end_to_end_quantum_resource_estimate = false`. The report is truthful only
as a polynomial-level proxy: it can compare degrees, phase-count proxies,
signal-call proxies, widths, and sampled compatibility checks. It cannot by
itself justify a runtime, hardware, or quantum-advantage claim.

Notation used in resource discussions:

- `degree` is the polynomial degree `d`.
- `matrix_dimension` is the logical dimension `N` of the encoded matrix.
- `encoding_qubits` is usually `ceil(log2(N))` when inferred from `N`.
- `signal_operator_calls` is a proxy for calls to the block-encoded signal
  operator, not a measured runtime.
- For linear systems, `gamma` is the scaled positive lower spectral bound and
  `1 / gamma` is a condition-number-style proxy when the scaled maximum
  eigenvalue is one.

## Omitted Costs

The proxy model does not include:

- block-encoding construction,
- state preparation or right-hand-side loading,
- measurement and amplitude estimation,
- amplitude amplification,
- sparse-oracle query implementation,
- Hamiltonian simulation subroutines used to build a block encoding,
- fault-tolerant synthesis and error correction,
- hardware compilation,
- memory movement or classical pre-processing.

For many realistic workflows, these omitted costs dominate the polynomial
degree. The reports are therefore best read as polynomial-resource summaries,
not full algorithmic complexity estimates.

For a finite dense construction that verifies an actual top-left block for one
small matrix, see [Block encodings](block_encoding.md). That page covers what
the package can validate directly and what still requires a scalable oracle or
problem-specific circuit.

## How To Use The Proxy

Use the proxy when comparing candidate polynomial designs:

1. hold the problem normalization fixed,
2. compare degrees needed to reach similar approximation error,
3. inspect boundedness and parity compatibility,
4. compare signal-call proxies across designs,
5. separately document how the block encoding and input state would be
   prepared.

This separation keeps the notebook examples honest: a low-degree polynomial is
promising only if the surrounding access model is also favorable.

## Linear Systems

For inverse-like linear-system workflows, polynomial degree is tied to the
scaled minimum eigenvalue and condition number. Classical dense solves and
conjugate gradients remain important baselines because they make residuals,
conditioning, and iteration behavior visible.

The proxy does not include quantum state preparation for `b`, nor the cost of
extracting a classical solution vector from a quantum state.

## Spectral Projectors And Filters

For thresholding, band-pass filters, and low-rank projection, degree depends on
transition width. Sharper spectral edges need higher degree, and leakage
outside the selected interval should be reported alongside any rank or retained
weight proxy.

The proxy does not include how the selected state or projected subspace would
be read out.

## Matrix Functions

For exponentials, square roots, Gibbs weights, and resolvents, degree depends
on the function smoothness over the scaled spectral domain. Singularities,
small broadening parameters, and sharp windows usually increase degree.

Dense spectral matrix functions remain the exact small-system reference.

## Related Pages

- [Classical baseline details](classical_baselines.md)
- [Classical benchmarks](benchmarks.md)
- [Algorithm notes](algorithms.md)
- [Block encodings](block_encoding.md)
- [QSVT compatibility](compatibility.md)
- [Implementation notes](implementation.md)
