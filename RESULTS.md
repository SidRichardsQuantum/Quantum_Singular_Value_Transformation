# Results

This repository is notebook-first: executable notebooks are the source of truth
for demonstrations, plots, and numerical checks. Use this page as the root
index for result-producing workflows and committed reproducibility artefacts.

Rendered result pages:

- Tutorial plot gallery: [docs/qsvt/result_gallery.md](docs/qsvt/result_gallery.md)
- Real-example plots and tables: [docs/qsvt/real_example_results.md](docs/qsvt/real_example_results.md)

## Tutorial Notebook Results

| notebook | current output status | result focus |
| --- | --- | --- |
| `notebooks/tutorials/01_QSVT_Scalar_and_Diagonal_Matrix.ipynb` | embedded plots | scalar and diagonal polynomial transforms |
| `notebooks/tutorials/02_QSVT_Singular_Value_Filter.ipynb` | embedded plots | soft singular-value filtering |
| `notebooks/tutorials/03_QSP_Polynomial_Demo.ipynb` | embedded plots | QSP polynomial behaviour from two perspectives |
| `notebooks/tutorials/04_QSVT_Linear_Solver_2x2.ipynb` | embedded plots | small exact linear-solver-style check |
| `notebooks/tutorials/05_QSVT_Linear_Solver_4x4.ipynb` | embedded plots | larger diagonal linear-solver-style check |
| `notebooks/tutorials/06_QSVT_Linear_Solver_Approximate.ipynb` | embedded plots | Chebyshev inverse-like approximation |
| `notebooks/tutorials/07_QSVT_Polynomial_Design_and_Approximation.ipynb` | embedded plots | polynomial approximation and boundedness |
| `notebooks/tutorials/08_QSVT_Matrix_Functions_Powers_and_Roots.ipynb` | embedded plots | matrix powers and square-root-style transforms |
| `notebooks/tutorials/09_QSVT_Sign_Function_and_Projectors.ipynb` | embedded plots | sign functions and spectral projectors |
| `notebooks/tutorials/10_QSVT_Design_and_Templates.ipynb` | embedded plots | reusable design and template families |
| `notebooks/tutorials/11_End_to_End_Algorithm_Workflows.ipynb` | embedded plot | high-level algorithm workflow diagnostics |
| `notebooks/tutorials/12_Reports_CLI_and_Reproducible_Artifacts.ipynb` | embedded plot | report, CLI, and artifact workflows |
| `notebooks/tutorials/13_Degree_Error_and_Boundedness_Tradeoffs.ipynb` | embedded plots | degree, error, and boundedness tradeoffs |

## Real-Example Notebook Results

Real physics examples live in `notebooks/real_examples/`. The notebooks remain
the executable source of truth, and selected embedded PNG outputs are committed
under `results/plots/real_examples/` for stable documentation rendering.
[docs/qsvt/real_example_results.md](docs/qsvt/real_example_results.md) renders
the real-example plot ledger and representative gallery.

| notebook | current output status | result focus |
| --- | --- | --- |
| `notebooks/real_examples/01_ground_state_filtering.ipynb` | committed plot | ground-state filtering and overlap checks |
| `notebooks/real_examples/02_tight_binding_band_filter.ipynb` | committed plot | band-pass and band-rejection filters |
| `notebooks/real_examples/03_imaginary_time_filtering.ipynb` | committed plot | imaginary-time exponential filtering |
| `notebooks/real_examples/04_heat_equation_pde.ipynb` | committed plots | heat-equation diffusion workflow |
| `notebooks/real_examples/05_poisson_equation_pde.ipynb` | committed plots | Poisson-equation inverse workflow |
| `notebooks/real_examples/06_hamiltonian_simulation_schrodinger_dynamics.ipynb` | committed plot | real-time Hamiltonian dynamics |
| `notebooks/real_examples/07_quantum_chemistry_h2_toy_solver.ipynb` | committed plot | toy H2 Hamiltonian solver |
| `notebooks/real_examples/08_greens_function_response.ipynb` | committed plot | resolvent and Green's-function response |
| `notebooks/real_examples/09_spectral_density_estimation.ipynb` | committed plot | spectral density estimation |
| `notebooks/real_examples/10_gibbs_state_thermal_weights.ipynb` | committed plot | Gibbs-state thermal weights |
| `notebooks/real_examples/11_transport_physics_landauer_chain.ipynb` | committed plot | Landauer-style chain transmission |
| `notebooks/real_examples/12_tensor_network_hybrid_filtering.ipynb` | committed plot | product-state energy filtering |
| `notebooks/real_examples/13_heat_equation_2d_pde.ipynb` | committed plots | 2D heat-equation diffusion workflow |
| `notebooks/real_examples/14_advection_diffusion_pde.ipynb` | committed plot | advection-diffusion PDE workflow |
| `notebooks/real_examples/15_wave_equation_dynamics.ipynb` | committed plot | wave-equation dynamics |
| `notebooks/real_examples/16_helmholtz_equation_pde.ipynb` | committed plot | Helmholtz-equation PDE workflow |
| `notebooks/real_examples/17_quantum_walk_search_toy.ipynb` | committed plot | quantum-walk search toy model |
| `notebooks/real_examples/18_ssh_chain_edge_state_filtering.ipynb` | committed plot | SSH-chain edge-state filtering |
| `notebooks/real_examples/19_anderson_localization.ipynb` | committed plot | Anderson localization |
| `notebooks/real_examples/20_schrodinger_bound_states.ipynb` | committed plot | Schrodinger bound states |
| `notebooks/real_examples/21_quantum_harmonic_oscillator_grid.ipynb` | committed plot | harmonic-oscillator grid spectrum |
| `notebooks/real_examples/22_electrostatic_green_function_poisson.ipynb` | committed plot | electrostatic Green's function |
| `notebooks/real_examples/23_coupled_oscillator_normal_modes.ipynb` | committed plot | coupled-oscillator normal modes |
| `notebooks/real_examples/24_ising_phase_transition_filtering.ipynb` | committed plot | Ising phase-transition filtering |
| `notebooks/real_examples/25_diffusion_heat_treatment_slab.ipynb` | committed plot | diffusion-limited heat treatment |
| `notebooks/real_examples/26_graphene_nanoribbon_density_of_states.ipynb` | committed plot | graphene nanoribbon density of states |
| `notebooks/real_examples/27_fermi_dirac_electronic_occupations.ipynb` | committed plot | Fermi-Dirac electronic occupations |
| `notebooks/real_examples/28_photonic_crystal_band_gap_filtering.ipynb` | committed plots | photonic-crystal band-gap filtering |
| `notebooks/real_examples/29_topological_band_projector_chern_marker.ipynb` | committed plot | topological band projectors and Chern markers |

## Committed Release Artefacts

The following artefacts provide small, reproducible release snapshots generated
from the package CLI.

| artefact | workflow | max error | RMS error | notes |
| --- | --- | ---: | ---: | --- |
| `results/reports/sign-report.json` | sign polynomial design | `2.091981425741754e-01` | `1.1105097545055853e-01` | degree-13 sign approximation with `gamma=0.2`; bounded on `[-1, 1]` |
| `results/plots/sign-report.png` | sign polynomial design plot | n/a | n/a | target-vs-polynomial plot for `sign-report.json` |
| `results/reports/qsvt-report.json` | diagonal QSVT transform | `9.999778782798785e-13` | `5.585577546102077e-13` | compares QSVT output with direct evaluation of `x^2` |
| `results/reports/matrix-report.json` | Hermitian matrix QSVT transform | `5.264677582772492e-13` | `3.2060825311797223e-13` | real-part comparison against the classical spectral polynomial |
| `results/tables/qsvt-error-summary.csv` | release summary table | n/a | n/a | compact index over the generated JSON reports |

These snapshots were refreshed for package version `0.1.14`.

## Real-Example Artefacts

Real-example plot artefacts are committed under `results/plots/real_examples/`.
The complete machine-readable manifest is
`results/tables/real_examples_plot_manifest.csv`, and the rendered ledger is
[docs/qsvt/real_example_results.md](docs/qsvt/real_example_results.md).

| artefact | notebook | result type | notes |
| --- | --- | --- | --- |
| `results/tables/real_examples_plot_manifest.csv` | all notebooks with committed PNG outputs | table | manifest for 34 extracted real-example plots |
| `results/plots/real_examples/01_ground_state_filtering-plot-01.png` | `01_ground_state_filtering.ipynb` | plot | representative ground-state filtering output |
| `results/plots/real_examples/13_heat_equation_2d_pde-plot-01.png` | `13_heat_equation_2d_pde.ipynb` | plot | representative 2D PDE output |
| `results/plots/real_examples/28_photonic_crystal_band_gap_filtering-plot-01.png` | `28_photonic_crystal_band_gap_filtering.ipynb` | plot | representative photonic band-gap output |
| `results/plots/real_examples/29_topological_band_projector_chern_marker-plot-01.png` | `29_topological_band_projector_chern_marker.ipynb` | plot | representative topological band-projector output |

## Regeneration Commands

Extract embedded notebook plots:

```bash
python scripts/extract_notebook_plots.py
```

Extract embedded real-example notebook plots:

```bash
python scripts/extract_notebook_plots.py \
  --notebook-glob "notebooks/real_examples/*.ipynb" \
  --output-dir results/plots/real_examples
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

Large visual galleries belong in docs pages:

- tutorial notebook plots: `docs/qsvt/result_gallery.md`
- real-example plots and tables: `docs/qsvt/real_example_results.md`
