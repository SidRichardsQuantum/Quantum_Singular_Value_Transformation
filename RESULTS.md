# Results

This repository is notebook-first: executable notebooks are the source of truth
for demonstrations, plots, and numerical checks. Use this page as the root
index for result-producing workflows and committed reproducibility artefacts.

Rendered result pages:

- Tutorial notebook outputs: [docs/qsvt/tutorial_results.md](docs/qsvt/tutorial_results.md)
- Real-example plots and tables: [docs/qsvt/real_example_results.md](docs/qsvt/real_example_results.md)
- Benchmark notebook outputs: [docs/qsvt/benchmark_results.md](docs/qsvt/benchmark_results.md)

## Tutorial Notebook Results

| notebook | current output status | result focus |
| --- | --- | --- |
| `notebooks/tutorials/01_QSVT_Scalar_and_Diagonal_Matrix.ipynb` | embedded plots | scalar and diagonal polynomial transforms |
| `notebooks/tutorials/02_QSVT_Singular_Value_Filter.ipynb` | embedded plots | soft singular-value filtering |
| `notebooks/tutorials/03_QSP_Polynomial_Demo.ipynb` | embedded plots | QSP polynomial behaviour from two perspectives |
| `notebooks/tutorials/04_QSVT_Exact_Linear_Solver_Toy_Cases.ipynb` | embedded plots | exact 2x2 and 4x4 toy linear-solver cases |
| `notebooks/tutorials/05_QSVT_Polynomial_Design_and_Approximation.ipynb` | embedded plots | polynomial approximation and boundedness |
| `notebooks/tutorials/06_QSVT_Matrix_Functions_Powers_and_Roots.ipynb` | embedded plots | matrix powers and square-root-style transforms |
| `notebooks/tutorials/07_QSVT_Sign_Function_and_Projectors.ipynb` | embedded plots | sign functions and spectral projectors |
| `notebooks/tutorials/08_QSVT_Design_and_Presets.ipynb` | embedded plots | reusable design and preset families |
| `notebooks/tutorials/09_QSVT_Algorithm_Workflows.ipynb` | embedded plot | QSVT algorithm workflow diagnostics |
| `notebooks/tutorials/10_QSVT_Reports_CLI_and_Artifacts.ipynb` | embedded plot | QSVT report, CLI, and artifact workflows |
| `notebooks/tutorials/11_QSVT_Design_Tradeoffs.ipynb` | embedded plots | QSVT design degree/error/boundedness tradeoffs |
| `notebooks/tutorials/12_QSVT_Resource_Proxy_Limits.ipynb` | embedded plots | block-encoding assumptions and resource-proxy limits |
| `notebooks/tutorials/13_Block_Encoded_QSVT_Workflow.ipynb` | embedded plots | finite dense block-encoded QSVT workflow |
| `notebooks/tutorials/14_Sparse_Oracle_Assumptions.ipynb` | embedded plot | sparse-oracle and access-model assumptions |
| `notebooks/tutorials/15_QSVT_Compatibility_Failure_Cases.ipynb` | embedded plot | boundedness, parity, and synthesis failure cases |
| `notebooks/tutorials/16_QSVT_Linear_System_Comparisons.ipynb` | embedded plots | dense, CG, and QSVT-style linear-system comparison rows |
| `notebooks/tutorials/17_HHL_Linear_System_Solver.ipynb` | embedded plots | finite simulator-scale HHL linear-system solver |
| `notebooks/tutorials/18_Quantum_Walk_Search_Workflow.ipynb` | embedded plots | continuous-time quantum-walk search workflow |
| `notebooks/tutorials/19_Accuracy_Driven_QSVT_Planning.ipynb` | embedded plots | accuracy-driven degree selection, access models, logical resources, and finite execution |

## Real-Example Notebook Results

Real physics examples live in `notebooks/real_examples/`. The notebooks remain
the executable source of truth, and selected embedded PNG outputs are committed
under `results/plots/real_examples/` for stable documentation rendering.
[docs/qsvt/real_example_results.md](docs/qsvt/real_example_results.md) renders
the real-example plot ledger.

| notebook | current output status | result focus |
| --- | --- | --- |
| `notebooks/real_examples/01_poisson_equation_pde.ipynb` | committed plots | Poisson domain schematics, inverse workflow, and executable block-encoded QSVT comparison |
| `notebooks/real_examples/02_hamiltonian_simulation_schrodinger_dynamics.ipynb` | committed plot | real-time Hamiltonian dynamics |
| `notebooks/real_examples/03_greens_function_response.ipynb` | committed plot | resolvent and Green's-function response |
| `notebooks/real_examples/04_ising_phase_transition_filtering.ipynb` | committed plots | spin-chain diagnostics and executable Pauli-LCU ground-band filtering |
| `notebooks/real_examples/05_fermi_dirac_electronic_occupations.ipynb` | committed plot | Fermi-Dirac electronic occupations |
| `notebooks/real_examples/06_topological_band_projector_chern_marker.ipynb` | committed plots | QWZ lattice schematic and Chern-marker diagnostics |
| `notebooks/real_examples/07_singular_value_pseudoinverse_deblurring.ipynb` | committed plot | singular-value pseudoinverse deblurring |
| `notebooks/real_examples/08_matrix_log_entropy_graph_laplacian.ipynb` | committed plot | matrix-log graph entropy |

## Committed Release Artefacts

The following artefacts provide small, reproducible release snapshots generated
from the package CLI.

| artefact | workflow | max error | RMS error | notes |
| --- | --- | ---: | ---: | --- |
| `results/reports/sign-report.json` | sign polynomial design | `2.091981425741754e-01` | `1.1105097545055853e-01` | degree-13 sign approximation with `gamma=0.2`; bounded on `[-1, 1]` |
| `results/plots/sign-report.png` | sign polynomial design plot | n/a | n/a | target-vs-polynomial plot for `sign-report.json` |
| `results/reports/sign-degree-sweep.json` | sign polynomial degree sweep | n/a | n/a | compact degree/error/boundedness manifest for degrees `5,9,13,17` with `gamma=0.2` |
| `results/reports/filter-degree-sweep.json` | filter polynomial degree sweep | n/a | n/a | compact degree/error/boundedness manifest for degrees `6,10,14,18` with `cutoff=0.4` |
| `results/reports/qsvt-report.json` | diagonal QSVT transform | `9.999778782798785e-13` | `5.585577546102077e-13` | compares QSVT output with direct evaluation of `x^2` |
| `results/reports/matrix-report.json` | Hermitian matrix QSVT transform | `5.264677582772492e-13` | `3.2060825311797223e-13` | real-part comparison against the classical spectral polynomial |
| `results/tables/design_sweep_summary.csv` | design sweep summary table | n/a | n/a | tabular summary of committed design-sweep JSON reports |
| `results/tables/qsvt-error-summary.csv` | release summary table | n/a | n/a | compact index over the generated JSON reports |

These snapshots were refreshed for package version `0.2.20`.

## Benchmark Artefacts

Benchmark notebooks live in `notebooks/benchmarks/` and compare classical
baselines with QSVT-oriented resource proxies. The generated output page is
[docs/qsvt/benchmark_results.md](docs/qsvt/benchmark_results.md).

| artefact | contents |
| --- | --- |
| `results/benchmarks/linear_system_dense_solve.json` | dense linear solve (DLS) baseline with QSVT proxy metadata |
| `results/benchmarks/linear_system_cg_solve.json` | conjugate gradient solve (CGS) baseline with QSVT proxy metadata |
| `results/benchmarks/matrix_function_exponential_spectral.json` | dense spectral matrix function (DSMF) baseline |
| `results/benchmarks/matrix_function_thermal_polynomial.json` | polynomial matrix evaluation (PME) thermal baseline |
| `results/benchmarks/matrix_function_filter_polynomial.json` | PME filter baseline |
| `results/benchmarks/scaling_sweep_reports.json` | combined DLS/CGS scaling-sweep reports |
| `results/benchmarks/quantum_walk_search_scaling.json` | quantum-walk search success and QSVT proxy scaling |
| `results/benchmarks/encoding_aware_resource_sweep.json` | embedding, FABLE, PrepSelPrep, and qubitization logical-resource sweep |
| `results/tables/linear_system_benchmark_summary.csv` | compact linear-system benchmark table |
| `results/tables/matrix_function_benchmark_summary.csv` | compact matrix-function benchmark table |
| `results/tables/benchmark_scaling_summary.csv` | compact scaling-sweep benchmark table |
| `results/tables/quantum_walk_search_scaling_summary.csv` | compact quantum-walk search scaling table |
| `results/tables/encoding_aware_resource_summary.csv` | compact encoding-aware logical-resource comparison |
| `results/tables/benchmark_plot_manifest.csv` | manifest for extracted benchmark notebook plots |
| `results/plots/benchmarks/` | extracted benchmark timing and QSVT-proxy PNG plots |

The benchmark notebook set also includes
`notebooks/benchmarks/04_classical_baseline_assumptions.ipynb`, which is an
executable assumption-check notebook rather than a committed timing-artifact
generator.

Benchmark artefacts were refreshed for package version `0.2.20`.

## Real-Example Artefacts

Real-example plot artefacts are committed under `results/plots/real_examples/`.
The complete machine-readable manifest is
`results/tables/real_examples_plot_manifest.csv`, and the rendered ledger is
[docs/qsvt/real_example_results.md](docs/qsvt/real_example_results.md).

| artefact | notebook | result type | notes |
| --- | --- | --- | --- |
| `results/tables/real_examples_plot_manifest.csv` | curated real-example notebooks | table | manifest for 14 extracted plots |
| `results/plots/real_examples/01_poisson_equation_pde-plot-01.png` | `01_poisson_equation_pde.ipynb` | plot | representative PDE setup output |
| `results/plots/real_examples/04_ising_phase_transition_filtering-plot-01.png` | `04_ising_phase_transition_filtering.ipynb` | plot | representative spin-chain output |
| `results/plots/real_examples/06_topological_band_projector_chern_marker-plot-01.png` | `06_topological_band_projector_chern_marker.ipynb` | plot | representative QWZ lattice setup schematic |

## Regeneration Commands

Committed notebook outputs and generated artefacts are the source of truth for
the published result pages. GitHub Pages builds from committed files and does
not execute notebooks during deployment.

Before committing notebook or result changes, run the local helper:

```bash
scripts/update_notebook_results.sh
```

Then commit the updated notebooks, extracted plots, manifests, and generated
result pages together.

The helper executes notebooks, extracts their embedded outputs, and regenerates
the rendered result pages. The underlying command is:

```bash
python scripts/extract_notebook_plots.py --preset all --execute --write-docs
```

Refresh the pages from already-saved notebook outputs without re-executing:

```bash
python scripts/extract_notebook_plots.py --preset all --write-docs
```

Generate the committed report examples:

```bash
mkdir -p results/reports results/plots results/tables
```

```bash
qsvt design-report --kind sign --gamma 0.2 --degree 13 \
  --output results/reports/sign-report.json \
  --plot results/plots/sign-report.png
```

```bash
qsvt design-sweep --kind sign --degrees "5,9,13,17" --gamma 0.2 \
  --num-points 401 \
  --bounded-num-points 801 \
  --no-synthesis \
  --output results/reports/sign-degree-sweep.json
```

```bash
qsvt design-sweep --kind filter --degrees "6,10,14,18" --cutoff 0.4 \
  --sharpness 12 \
  --num-points 401 \
  --bounded-num-points 801 \
  --no-synthesis \
  --output results/reports/filter-degree-sweep.json
```

```bash
qsvt compare-report \
  --values "1.0,0.7,0.3,0.1" \
  --poly "0,0,1" \
  --wires 3 \
  --output results/reports/qsvt-report.json
```

```bash
qsvt matrix-report \
  --matrix "0.31351701,-0.23499807;-0.23499807,0.68648299" \
  --poly "0,0,1" \
  --output results/reports/matrix-report.json
```

## Artefact Convention

Generated artefacts should be saved in predictable locations:

| artefact type | suggested path | examples |
| --- | --- | --- |
| JSON reports | `results/reports/` | `sign-report.json`, `matrix-report.json` |
| static plots | `results/plots/` | `sign-report.png`, `filter-response.png` |
| extracted notebook plots | `results/plots/notebooks/` | `01_QSVT_Scalar_and_Diagonal_Matrix-plot-01.png` |
| real-example plots | `results/plots/real_examples/` | `01_ground_state_filtering-overlap.png` |
| real-example tables | `results/tables/real_examples/` | `01_ground_state_filtering-summary.csv` |
| tabular summaries | `results/tables/` | `qsvt-error-summary.csv` |

`RESULTS.md` should summarize stable outcomes and point to reproducible
artefacts. It should not duplicate notebook explanations or paste large raw
outputs. Good additions include short benchmark tables, links to committed JSON
reports, package versions, seeds, backend details, and the command used to
generate each artefact.

Large notebook-derived outputs belong in generated docs pages:

- tutorial notebook outputs: `docs/qsvt/tutorial_results.md`
- real-example outputs: `docs/qsvt/real_example_results.md`
