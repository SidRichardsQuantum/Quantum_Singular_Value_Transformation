# Benchmark Notebooks

These notebooks use the installed `qsvt` package as a client library. They
benchmark quantum/QSVT algorithms or implementations against the relevant
classical algorithms or implementations for the same task, and write compact
artifacts under `results/benchmarks/` and `results/tables/`.

Use implemented finite PennyLane/QSVT execution paths where available. When the
quantum path is represented by a QSVT resource proxy rather than timed circuit
execution, label that boundary explicitly in the notebook and artifact.

Plot legends use compact labels: dense linear solve (DLS), conjugate gradient
solve (CGS), dense spectral matrix function (DSMF), and polynomial matrix
evaluation (PME).

| Topic | Notebook |
| --- | --- |
| Linear systems, QSVT proxy, and finite HHL execution | `01_linear_system_classical_vs_qsvt_proxy.ipynb` |
| Spectral and polynomial matrix functions | `02_matrix_functions_spectral_baselines.ipynb` |
| Dimension, conditioning, and degree sweeps | `03_scaling_sweeps.ipynb` |
| Classical baseline assumptions | `04_classical_baseline_assumptions.ipynb` |
| Quantum walk search scaling | `05_quantum_walk_search_scaling.ipynb` |
| Encoding-aware logical resources | `06_encoding_aware_resources.ipynb` |
