# Benchmark Notebooks

These notebooks use the installed `qsvt` package as a client library. They
compare small classical physics baselines against QSVT-oriented resource
proxies and write compact artifacts under `results/benchmarks/` and
`results/tables/`.

Plot legends use compact labels: dense linear solve (DLS), conjugate gradient
solve (CGS), dense spectral matrix function (DSMF), and polynomial matrix
evaluation (PME).

| Topic | Notebook |
| --- | --- |
| Linear systems and PDE baselines | `01_linear_system_classical_vs_qsvt_proxy.ipynb` |
| Spectral and polynomial matrix functions | `02_matrix_functions_spectral_baselines.ipynb` |
| Dimension, conditioning, and degree sweeps | `03_scaling_sweeps.ipynb` |
