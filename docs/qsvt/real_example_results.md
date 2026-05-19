# Real-Example Results

This page curates the notebook-derived outputs for the physics, PDE, transport,
and spectral-analysis examples in `notebooks/real_examples`.

[`RESULTS.md`](../../RESULTS.md) remains the compact root index for all
result-producing workflows. [Results](results.md) provides the cross-project
summary of validation, CLI reports, and benchmark artefacts.

## Current Status

Real-example plot artefacts are committed for 29 of the 29 real-example
notebooks. The source notebooks remain executable workflows, and the committed
PNG files are extracted from embedded notebook outputs.

The machine-readable plot manifest is committed at
[`results/tables/real_examples_plot_manifest.csv`](../../results/tables/real_examples_plot_manifest.csv).

## Representative Results

```{figure} ../../results/plots/real_examples/01_ground_state_filtering-plot-01.png
:alt: Ground-state filtering plot
:width: 520px

Ground-state filtering demonstrates low-energy spectral selection and overlap
diagnostics on a small Hamiltonian.
```

```{figure} ../../results/plots/real_examples/13_heat_equation_2d_pde-plot-01.png
:alt: 2D heat equation PDE plot
:width: 520px

The 2D heat-equation example shows how matrix-function workflows connect QSVT
polynomials to finite-difference PDE operators.
```

```{figure} ../../results/plots/real_examples/26_graphene_nanoribbon_density_of_states-plot-01.png
:alt: Graphene nanoribbon density of states plot
:width: 520px

The graphene nanoribbon example uses spectral-density estimation to expose band
structure features in a compact lattice model.
```

```{figure} ../../results/plots/real_examples/29_topological_band_projector_chern_marker-plot-01.png
:alt: Topological band projector Chern marker plot
:width: 520px

The topological band-projector example applies interval filtering and local
Chern-marker diagnostics to a small Qi-Wu-Zhang lattice model.
```

## Regeneration

Refresh the real-example plot artefacts with:

```bash
python scripts/extract_notebook_plots.py \
  --notebook-glob "notebooks/real_examples/*.ipynb" \
  --output-dir results/plots/real_examples
```

Refresh the manifest with:

```bash
python - <<'PY'
from pathlib import Path
import csv
plots = sorted(Path("results/plots/real_examples").glob("*.png"))
with Path("results/tables/real_examples_plot_manifest.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["artefact", "notebook", "result_type", "notes"])
    for plot in plots:
        stem = plot.stem.rsplit("-plot-", 1)[0]
        writer.writerow([plot.as_posix(), f"notebooks/real_examples/{stem}.ipynb", "plot", "embedded notebook PNG output"])
PY
```

## Complete Plot Ledger

| notebook | committed artefacts |
| --- | --- |
| `01_ground_state_filtering.ipynb` | [plot 1](../../results/plots/real_examples/01_ground_state_filtering-plot-01.png) |
| `02_tight_binding_band_filter.ipynb` | [plot 1](../../results/plots/real_examples/02_tight_binding_band_filter-plot-01.png) |
| `03_imaginary_time_filtering.ipynb` | [plot 1](../../results/plots/real_examples/03_imaginary_time_filtering-plot-01.png) |
| `04_heat_equation_pde.ipynb` | [plot 1](../../results/plots/real_examples/04_heat_equation_pde-plot-01.png), [plot 2](../../results/plots/real_examples/04_heat_equation_pde-plot-02.png) |
| `05_poisson_equation_pde.ipynb` | [plot 1](../../results/plots/real_examples/05_poisson_equation_pde-plot-01.png), [plot 2](../../results/plots/real_examples/05_poisson_equation_pde-plot-02.png) |
| `06_hamiltonian_simulation_schrodinger_dynamics.ipynb` | [plot 1](../../results/plots/real_examples/06_hamiltonian_simulation_schrodinger_dynamics-plot-01.png) |
| `07_quantum_chemistry_h2_toy_solver.ipynb` | [plot 1](../../results/plots/real_examples/07_quantum_chemistry_h2_toy_solver-plot-01.png) |
| `08_greens_function_response.ipynb` | [plot 1](../../results/plots/real_examples/08_greens_function_response-plot-01.png) |
| `09_spectral_density_estimation.ipynb` | [plot 1](../../results/plots/real_examples/09_spectral_density_estimation-plot-01.png) |
| `10_gibbs_state_thermal_weights.ipynb` | [plot 1](../../results/plots/real_examples/10_gibbs_state_thermal_weights-plot-01.png) |
| `11_transport_physics_landauer_chain.ipynb` | [plot 1](../../results/plots/real_examples/11_transport_physics_landauer_chain-plot-01.png) |
| `12_tensor_network_hybrid_filtering.ipynb` | [plot 1](../../results/plots/real_examples/12_tensor_network_hybrid_filtering-plot-01.png) |
| `13_heat_equation_2d_pde.ipynb` | [plot 1](../../results/plots/real_examples/13_heat_equation_2d_pde-plot-01.png), [plot 2](../../results/plots/real_examples/13_heat_equation_2d_pde-plot-02.png), [plot 3](../../results/plots/real_examples/13_heat_equation_2d_pde-plot-03.png) |
| `14_advection_diffusion_pde.ipynb` | [plot 1](../../results/plots/real_examples/14_advection_diffusion_pde-plot-01.png) |
| `15_wave_equation_dynamics.ipynb` | [plot 1](../../results/plots/real_examples/15_wave_equation_dynamics-plot-01.png) |
| `16_helmholtz_equation_pde.ipynb` | [plot 1](../../results/plots/real_examples/16_helmholtz_equation_pde-plot-01.png) |
| `17_quantum_walk_search_toy.ipynb` | [plot 1](../../results/plots/real_examples/17_quantum_walk_search_toy-plot-01.png) |
| `18_ssh_chain_edge_state_filtering.ipynb` | [plot 1](../../results/plots/real_examples/18_ssh_chain_edge_state_filtering-plot-01.png) |
| `19_anderson_localization.ipynb` | [plot 1](../../results/plots/real_examples/19_anderson_localization-plot-01.png) |
| `20_schrodinger_bound_states.ipynb` | [plot 1](../../results/plots/real_examples/20_schrodinger_bound_states-plot-01.png) |
| `21_quantum_harmonic_oscillator_grid.ipynb` | [plot 1](../../results/plots/real_examples/21_quantum_harmonic_oscillator_grid-plot-01.png) |
| `22_electrostatic_green_function_poisson.ipynb` | [plot 1](../../results/plots/real_examples/22_electrostatic_green_function_poisson-plot-01.png) |
| `23_coupled_oscillator_normal_modes.ipynb` | [plot 1](../../results/plots/real_examples/23_coupled_oscillator_normal_modes-plot-01.png) |
| `24_ising_phase_transition_filtering.ipynb` | [plot 1](../../results/plots/real_examples/24_ising_phase_transition_filtering-plot-01.png) |
| `25_diffusion_heat_treatment_slab.ipynb` | [plot 1](../../results/plots/real_examples/25_diffusion_heat_treatment_slab-plot-01.png) |
| `26_graphene_nanoribbon_density_of_states.ipynb` | [plot 1](../../results/plots/real_examples/26_graphene_nanoribbon_density_of_states-plot-01.png) |
| `27_fermi_dirac_electronic_occupations.ipynb` | [plot 1](../../results/plots/real_examples/27_fermi_dirac_electronic_occupations-plot-01.png) |
| `28_photonic_crystal_band_gap_filtering.ipynb` | [plot 1](../../results/plots/real_examples/28_photonic_crystal_band_gap_filtering-plot-01.png), [plot 2](../../results/plots/real_examples/28_photonic_crystal_band_gap_filtering-plot-02.png) |
| `29_topological_band_projector_chern_marker.ipynb` | [plot 1](../../results/plots/real_examples/29_topological_band_projector_chern_marker-plot-01.png) |

## Additional Gallery

```{image} ../../results/plots/real_examples/01_ground_state_filtering-plot-01.png
:alt: Ground-state filtering plot
:width: 300px
```

```{image} ../../results/plots/real_examples/04_heat_equation_pde-plot-01.png
:alt: Heat equation PDE plot
:width: 300px
```

```{image} ../../results/plots/real_examples/07_quantum_chemistry_h2_toy_solver-plot-01.png
:alt: Quantum chemistry H2 toy solver plot
:width: 300px
```

```{image} ../../results/plots/real_examples/13_heat_equation_2d_pde-plot-01.png
:alt: 2D heat equation PDE plot
:width: 300px
```

```{image} ../../results/plots/real_examples/24_ising_phase_transition_filtering-plot-01.png
:alt: Ising phase transition filtering plot
:width: 300px
```

```{image} ../../results/plots/real_examples/28_photonic_crystal_band_gap_filtering-plot-01.png
:alt: Photonic crystal band gap filtering plot
:width: 300px
```

```{image} ../../results/plots/real_examples/29_topological_band_projector_chern_marker-plot-01.png
:alt: Topological band projector Chern marker plot
:width: 300px
```
