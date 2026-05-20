# Classical Benchmarks

The benchmark helpers provide small classical baselines for QSVT-oriented
workflows. They are designed for reproducible notebook studies and advantage
screening, not for claiming end-to-end quantum speedups.

## Available Baselines

The public API is in `qsvt.benchmarks`:

- `dense_eigendecomposition_benchmark`
- `dense_linear_solve_benchmark`
- `conjugate_gradient_benchmark`
- `polynomial_matrix_function_benchmark`
- `spectral_matrix_function_benchmark`
- `benchmark_summary_table`
- `write_benchmark_summary_csv`
- `plot_benchmark_timings`
- `plot_qsvt_proxy_resources`

Each benchmark report includes:

- classical baseline algorithm and problem type
- matrix dimension
- best and mean wall-clock time across repeated runs
- residual, condition-number, or matrix-function metadata where applicable
- optional QSVT resource proxy metadata for polynomial workflows

## Plot Abbreviations

Benchmark plots use compact legend labels:

| abbreviation | meaning |
| --- | --- |
| DLS | dense linear solve |
| CGS | conjugate gradient solve |
| DSMF | dense spectral matrix function |
| PME | polynomial matrix evaluation |

## CLI Workflow

```bash
qsvt benchmark dense-solve \
  --matrix "4,1;1,3" \
  --rhs "1,2" \
  --qsvt-poly "0,1" \
  --output dense-solve-benchmark.json
```

```bash
qsvt benchmark cg-solve \
  --matrix "4,1;1,3" \
  --rhs "1,2" \
  --tolerance 1e-10 \
  --qsvt-poly "0,1"
```

```bash
qsvt benchmark polynomial \
  --matrix "0.5,0;0,-0.25" \
  --poly "0,0,1"
```

## Benchmark Notebooks

Benchmark notebooks live in `notebooks/benchmarks/`:

| notebook | output |
| --- | --- |
| `01_linear_system_classical_vs_qsvt_proxy.ipynb` | dense and CG Poisson-system baselines |
| `02_matrix_functions_spectral_baselines.ipynb` | spectral and polynomial matrix-function baselines |
| `03_scaling_sweeps.ipynb` | dimension and inverse-degree benchmark sweeps |

The generated notebook outputs are published on the
[Benchmark notebook outputs](benchmark_results.md) page.

## Committed Artifacts

The current benchmark notebooks write JSON reports under `results/benchmarks/`
and compact CSV tables under `results/tables/`.

| artifact | contents |
| --- | --- |
| [`linear_system_dense_solve.json`](../../results/benchmarks/linear_system_dense_solve.json) | dense direct solve baseline with QSVT proxy |
| [`linear_system_cg_solve.json`](../../results/benchmarks/linear_system_cg_solve.json) | conjugate-gradient baseline with QSVT proxy |
| [`matrix_function_exponential_spectral.json`](../../results/benchmarks/matrix_function_exponential_spectral.json) | dense spectral exponential baseline |
| [`matrix_function_thermal_polynomial.json`](../../results/benchmarks/matrix_function_thermal_polynomial.json) | polynomial thermal matrix-function baseline |
| [`matrix_function_filter_polynomial.json`](../../results/benchmarks/matrix_function_filter_polynomial.json) | polynomial filter matrix-function baseline |
| [`scaling_sweep_reports.json`](../../results/benchmarks/scaling_sweep_reports.json) | combined dense/CG scaling sweep reports |
| [`linear_system_benchmark_summary.csv`](../../results/tables/linear_system_benchmark_summary.csv) | compact linear-system benchmark table |
| [`matrix_function_benchmark_summary.csv`](../../results/tables/matrix_function_benchmark_summary.csv) | compact matrix-function benchmark table |
| [`benchmark_scaling_summary.csv`](../../results/tables/benchmark_scaling_summary.csv) | compact scaling-sweep benchmark table |
| [`benchmark_plot_manifest.csv`](../../results/tables/benchmark_plot_manifest.csv) | generated plot manifest for benchmark notebooks |

## Interpretation

The QSVT proxy fields summarize polynomial degree, phase-count proxy,
signal-call proxy, and encoding width. They do not include block-encoding
construction, state preparation, amplitude amplification, error correction,
hardware compilation, or data-loading costs.

Use these reports to compare regimes and identify where a quantum implementation
would need favorable block encoding, state preparation, and scaling assumptions.
