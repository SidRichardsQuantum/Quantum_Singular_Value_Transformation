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

- a `truth_contract` marking the report as a classical baseline, not a quantum
  runtime benchmark,
- classical baseline algorithm and problem type
- matrix dimension
- best and mean wall-clock time across repeated runs
- `benchmark_environment` metadata including Python, NumPy, platform, timer,
  and a stability note for timing snapshots
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
| `01_linear_system_classical_vs_qsvt_proxy.ipynb` | dense/CG Poisson timings plus finite QSVT/HHL/classical comparison |
| `02_matrix_functions_spectral_baselines.ipynb` | spectral and polynomial matrix-function baselines |
| `03_scaling_sweeps.ipynb` | dimension and inverse-degree benchmark sweeps |
| `04_classical_baseline_assumptions.ipynb` | timed baseline assumptions versus QSVT proxy fields |
| `05_quantum_walk_search_scaling.ipynb` | quantum walk search success, polynomial error, and QSVT signal-call proxies |

The generated notebook outputs are published on the
[Benchmark notebook outputs](benchmark_results.md) page.

## Committed Artifacts

The current benchmark notebooks write JSON reports under `results/benchmarks/`
and compact CSV tables under `results/tables/`. Algorithm comparison artifacts
that are not wall-clock benchmarks live under `results/algorithms/` and share
the same compact table directory.

| artifact | contents |
| --- | --- |
| [`linear_system_dense_solve.json`](../../results/benchmarks/linear_system_dense_solve.json) | dense direct solve baseline with QSVT proxy |
| [`linear_system_cg_solve.json`](../../results/benchmarks/linear_system_cg_solve.json) | conjugate-gradient baseline with QSVT proxy |
| [`matrix_function_exponential_spectral.json`](../../results/benchmarks/matrix_function_exponential_spectral.json) | dense spectral exponential baseline |
| [`matrix_function_thermal_polynomial.json`](../../results/benchmarks/matrix_function_thermal_polynomial.json) | polynomial thermal matrix-function baseline |
| [`matrix_function_filter_polynomial.json`](../../results/benchmarks/matrix_function_filter_polynomial.json) | polynomial filter matrix-function baseline |
| [`scaling_sweep_reports.json`](../../results/benchmarks/scaling_sweep_reports.json) | combined dense/CG scaling sweep reports |
| [`quantum_walk_search_scaling.json`](../../results/benchmarks/quantum_walk_search_scaling.json) | quantum walk search scaling and polynomial approximation report |
| [`linear_system_benchmark_summary.csv`](../../results/tables/linear_system_benchmark_summary.csv) | compact linear-system benchmark table |
| [`matrix_function_benchmark_summary.csv`](../../results/tables/matrix_function_benchmark_summary.csv) | compact matrix-function benchmark table |
| [`benchmark_scaling_summary.csv`](../../results/tables/benchmark_scaling_summary.csv) | compact scaling-sweep benchmark table |
| [`quantum_walk_search_scaling_summary.csv`](../../results/tables/quantum_walk_search_scaling_summary.csv) | compact quantum walk search scaling table |
| [`benchmark_plot_manifest.csv`](../../results/tables/benchmark_plot_manifest.csv) | generated plot manifest for benchmark notebooks |
| [`linear_system_comparison.json`](../../results/algorithms/linear_system_comparison.json) | dense, CG, and QSVT-style linear-system comparison report |
| [`linear_system_comparison_summary.csv`](../../results/tables/linear_system_comparison_summary.csv) | compact linear-system comparison rows |
| [`linear_system_quantum_classical_comparison.json`](../../results/algorithms/linear_system_quantum_classical_comparison.json) | dense, CG, QSVT-style inverse, and executable finite HHL comparison report |
| [`linear_system_quantum_classical_summary.csv`](../../results/tables/linear_system_quantum_classical_summary.csv) | compact finite HHL/QSVT/classical comparison rows |

## Interpretation

The QSVT proxy fields summarize polynomial degree, phase-count proxy,
signal-call proxy, and encoding width. HHL rows marked as executable are finite
PennyLane QNode executions for simulator-scale systems. Neither path includes
all scalable block-encoding construction, state preparation, amplitude
amplification, error correction, hardware compilation, or data-loading costs.

Use these reports to compare regimes and identify where a quantum implementation
would need favorable block encoding, state preparation, and scaling assumptions.

For per-baseline assumptions, see [Classical baseline details](classical_baselines.md).
For the proxy quantities attached to QSVT comparisons, see
[QSVT resource model](qsvt_resource_model.md).
