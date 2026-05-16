# Results

This repository is primarily notebook-first: the executable notebooks are the
source of truth for demonstrations, plots, and numerical checks. The
introductory notebooks contain embedded output figures, while the real physics
notebooks are kept as clean, executable workflows.

Use this page as an index of the current result-producing workflows and as the
convention for future saved artefacts.

---

## Current notebook results

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

---

## Introductory notebook plot gallery

These PNG files are extracted from the embedded outputs in the introductory
notebooks so the notebook figures are visible from the results index.

Regenerate them after updating notebook outputs with:

```bash
python scripts/extract_notebook_plots.py
```

### `01_QSVT_Scalar_and_Diagonal_Matrix.ipynb`

<a href="results/plots/notebooks/01_QSVT_Scalar_and_Diagonal_Matrix-plot-01.png"><img src="results/plots/notebooks/01_QSVT_Scalar_and_Diagonal_Matrix-plot-01.png" alt="QSVT scalar and diagonal matrix plot 1" width="320"></a>
<a href="results/plots/notebooks/01_QSVT_Scalar_and_Diagonal_Matrix-plot-02.png"><img src="results/plots/notebooks/01_QSVT_Scalar_and_Diagonal_Matrix-plot-02.png" alt="QSVT scalar and diagonal matrix plot 2" width="320"></a>

### `02_QSVT_Singular_Value_Filter.ipynb`

<a href="results/plots/notebooks/02_QSVT_Singular_Value_Filter-plot-01.png"><img src="results/plots/notebooks/02_QSVT_Singular_Value_Filter-plot-01.png" alt="QSVT singular value filter plot 1" width="320"></a>
<a href="results/plots/notebooks/02_QSVT_Singular_Value_Filter-plot-02.png"><img src="results/plots/notebooks/02_QSVT_Singular_Value_Filter-plot-02.png" alt="QSVT singular value filter plot 2" width="320"></a>

### `03_QSP_Polynomial_Demo.ipynb`

<a href="results/plots/notebooks/03_QSP_Polynomial_Demo-plot-01.png"><img src="results/plots/notebooks/03_QSP_Polynomial_Demo-plot-01.png" alt="QSP polynomial demo plot 1" width="320"></a>
<a href="results/plots/notebooks/03_QSP_Polynomial_Demo-plot-02.png"><img src="results/plots/notebooks/03_QSP_Polynomial_Demo-plot-02.png" alt="QSP polynomial demo plot 2" width="320"></a>

### `04_QSVT_Linear_Solver_2x2.ipynb`

<a href="results/plots/notebooks/04_QSVT_Linear_Solver_2x2-plot-01.png"><img src="results/plots/notebooks/04_QSVT_Linear_Solver_2x2-plot-01.png" alt="QSVT 2 by 2 linear solver plot 1" width="320"></a>
<a href="results/plots/notebooks/04_QSVT_Linear_Solver_2x2-plot-02.png"><img src="results/plots/notebooks/04_QSVT_Linear_Solver_2x2-plot-02.png" alt="QSVT 2 by 2 linear solver plot 2" width="320"></a>

### `04_QSVT_Linear_Solver_4x4.ipynb`

<a href="results/plots/notebooks/04_QSVT_Linear_Solver_4x4-plot-01.png"><img src="results/plots/notebooks/04_QSVT_Linear_Solver_4x4-plot-01.png" alt="QSVT 4 by 4 linear solver plot 1" width="320"></a>
<a href="results/plots/notebooks/04_QSVT_Linear_Solver_4x4-plot-02.png"><img src="results/plots/notebooks/04_QSVT_Linear_Solver_4x4-plot-02.png" alt="QSVT 4 by 4 linear solver plot 2" width="320"></a>

### `05_QSVT_Linear_Solver_Approximate.ipynb`

<a href="results/plots/notebooks/05_QSVT_Linear_Solver_Approximate-plot-01.png"><img src="results/plots/notebooks/05_QSVT_Linear_Solver_Approximate-plot-01.png" alt="QSVT approximate linear solver plot 1" width="320"></a>
<a href="results/plots/notebooks/05_QSVT_Linear_Solver_Approximate-plot-02.png"><img src="results/plots/notebooks/05_QSVT_Linear_Solver_Approximate-plot-02.png" alt="QSVT approximate linear solver plot 2" width="320"></a>

### `06_QSVT_Polynomial_Design_and_Approximation.ipynb`

<a href="results/plots/notebooks/06_QSVT_Polynomial_Design_and_Approximation-plot-01.png"><img src="results/plots/notebooks/06_QSVT_Polynomial_Design_and_Approximation-plot-01.png" alt="QSVT polynomial design and approximation plot 1" width="320"></a>
<a href="results/plots/notebooks/06_QSVT_Polynomial_Design_and_Approximation-plot-02.png"><img src="results/plots/notebooks/06_QSVT_Polynomial_Design_and_Approximation-plot-02.png" alt="QSVT polynomial design and approximation plot 2" width="320"></a>
<a href="results/plots/notebooks/06_QSVT_Polynomial_Design_and_Approximation-plot-03.png"><img src="results/plots/notebooks/06_QSVT_Polynomial_Design_and_Approximation-plot-03.png" alt="QSVT polynomial design and approximation plot 3" width="320"></a>

### `07_QSVT_Matrix_Functions_Powers_and_Roots.ipynb`

<a href="results/plots/notebooks/07_QSVT_Matrix_Functions_Powers_and_Roots-plot-01.png"><img src="results/plots/notebooks/07_QSVT_Matrix_Functions_Powers_and_Roots-plot-01.png" alt="QSVT matrix functions powers and roots plot 1" width="320"></a>
<a href="results/plots/notebooks/07_QSVT_Matrix_Functions_Powers_and_Roots-plot-02.png"><img src="results/plots/notebooks/07_QSVT_Matrix_Functions_Powers_and_Roots-plot-02.png" alt="QSVT matrix functions powers and roots plot 2" width="320"></a>
<a href="results/plots/notebooks/07_QSVT_Matrix_Functions_Powers_and_Roots-plot-03.png"><img src="results/plots/notebooks/07_QSVT_Matrix_Functions_Powers_and_Roots-plot-03.png" alt="QSVT matrix functions powers and roots plot 3" width="320"></a>

### `08_QSVT_Sign_Function_and_Projectors.ipynb`

<a href="results/plots/notebooks/08_QSVT_Sign_Function_and_Projectors-plot-01.png"><img src="results/plots/notebooks/08_QSVT_Sign_Function_and_Projectors-plot-01.png" alt="QSVT sign function and projectors plot 1" width="320"></a>
<a href="results/plots/notebooks/08_QSVT_Sign_Function_and_Projectors-plot-02.png"><img src="results/plots/notebooks/08_QSVT_Sign_Function_and_Projectors-plot-02.png" alt="QSVT sign function and projectors plot 2" width="320"></a>

### `09_QSVT_Design_and_Templates.ipynb`

<a href="results/plots/notebooks/09_QSVT_Design_and_Templates-plot-01.png"><img src="results/plots/notebooks/09_QSVT_Design_and_Templates-plot-01.png" alt="QSVT design and templates plot 1" width="320"></a>
<a href="results/plots/notebooks/09_QSVT_Design_and_Templates-plot-02.png"><img src="results/plots/notebooks/09_QSVT_Design_and_Templates-plot-02.png" alt="QSVT design and templates plot 2" width="320"></a>
<a href="results/plots/notebooks/09_QSVT_Design_and_Templates-plot-03.png"><img src="results/plots/notebooks/09_QSVT_Design_and_Templates-plot-03.png" alt="QSVT design and templates plot 3" width="320"></a>
<a href="results/plots/notebooks/09_QSVT_Design_and_Templates-plot-04.png"><img src="results/plots/notebooks/09_QSVT_Design_and_Templates-plot-04.png" alt="QSVT design and templates plot 4" width="320"></a>
<a href="results/plots/notebooks/09_QSVT_Design_and_Templates-plot-05.png"><img src="results/plots/notebooks/09_QSVT_Design_and_Templates-plot-05.png" alt="QSVT design and templates plot 5" width="320"></a>

---

## Real physics workflows

The notebooks under `notebooks/real_examples/` are designed to produce
comparison results when executed locally. They use package APIs for
Hamiltonians, PDE operators, spectral rescaling, diagnostics, and matrix
function polynomials so the same result can be reproduced from scripts or from
the command line.

| notebook | result focus |
| --- | --- |
| `01_ground_state_filtering.ipynb` | ground-state filtering and overlap checks |
| `02_tight_binding_band_filter.ipynb` | band-pass and band-rejection filters |
| `03_imaginary_time_filtering.ipynb` | imaginary-time exponential filtering |
| `04_heat_equation_pde.ipynb` | heat-equation diffusion workflow |
| `05_poisson_equation_pde.ipynb` | Poisson-equation inverse workflow |
| `06_hamiltonian_simulation_schrodinger_dynamics.ipynb` | real-time Hamiltonian dynamics |
| `07_quantum_chemistry_h2_toy_solver.ipynb` | toy H2 Hamiltonian solver |
| `08_greens_function_response.ipynb` | resolvent and Green's-function response |
| `09_spectral_density_estimation.ipynb` | spectral density estimation |
| `10_gibbs_state_thermal_weights.ipynb` | Gibbs-state thermal weights |
| `11_transport_physics_landauer_chain.ipynb` | Landauer-style chain transmission |
| `12_tensor_network_hybrid_filtering.ipynb` | product-state energy filtering |
| `13_heat_equation_2d_pde.ipynb` | 2D heat-equation diffusion workflow |
| `14_advection_diffusion_pde.ipynb` | advection-diffusion PDE workflow |
| `15_wave_equation_dynamics.ipynb` | wave-equation dynamics |
| `16_helmholtz_equation_pde.ipynb` | Helmholtz-equation PDE workflow |
| `17_quantum_walk_search_toy.ipynb` | quantum-walk search toy model |
| `18_ssh_chain_edge_state_filtering.ipynb` | SSH-chain edge-state filtering |
| `19_anderson_localization.ipynb` | Anderson localization |
| `20_schrodinger_bound_states.ipynb` | Schrödinger bound states |
| `21_quantum_harmonic_oscillator_grid.ipynb` | harmonic-oscillator grid spectrum |
| `22_electrostatic_green_function_poisson.ipynb` | electrostatic Green's function |
| `23_coupled_oscillator_normal_modes.ipynb` | coupled-oscillator normal modes |
| `24_ising_phase_transition_filtering.ipynb` | Ising phase-transition filtering |
| `25_diffusion_heat_treatment_slab.ipynb` | diffusion-limited heat treatment |
| `26_graphene_nanoribbon_density_of_states.ipynb` | graphene nanoribbon density of states |
| `27_fermi_dirac_electronic_occupations.ipynb` | Fermi-Dirac electronic occupations |
| `28_photonic_crystal_band_gap_filtering.ipynb` | photonic-crystal band-gap filtering |

---

## Committed release artefacts

The following artefacts provide small, reproducible release snapshots generated
from the package CLI.

| artefact | workflow | max error | RMS error | notes |
| --- | --- | ---: | ---: | --- |
| `results/reports/sign-report.json` | sign polynomial design | `2.091981425741754e-01` | `1.1105097545055853e-01` | degree-13 sign approximation with `gamma=0.2`; bounded on `[-1, 1]` |
| `results/plots/sign-report.png` | sign polynomial design plot | n/a | n/a | target-vs-polynomial plot for `sign-report.json` |
| `results/reports/qsvt-report.json` | diagonal QSVT transform | `9.999778782798785e-13` | `5.585577546102077e-13` | compares QSVT output with direct evaluation of `x^2` |
| `results/reports/matrix-report.json` | Hermitian matrix QSVT transform | `5.264677582772492e-13` | `3.2060825311797223e-13` | real-part comparison against the classical spectral polynomial |
| `results/tables/qsvt-error-summary.csv` | release summary table | n/a | n/a | compact index over the generated JSON reports |

Generated for package version `0.1.12` as preparation for the next patch
release.

---

## Report and plot artefact convention

Generated artefacts should be saved in predictable locations:

| artefact type | suggested path | examples |
| --- | --- | --- |
| JSON reports | `results/reports/` | `sign-report.json`, `matrix-report.json` |
| static plots | `results/plots/` | `sign-report.png`, `filter-response.png` |
| tabular summaries | `results/tables/` | `qsvt-error-summary.csv` |

These paths are not required for using the package. They are the repository
convention for committed result snapshots when a notebook or release needs
stable figures and tables.

---

## Reproducible CLI examples

The package can already write JSON reports and plots from the command line:

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

Create the output directories first when saving artefacts:

```bash
mkdir -p results/reports results/plots results/tables
```

---

## What belongs here

`RESULTS.md` should summarize stable outcomes and point to reproducible
artefacts. It should not duplicate notebook explanations or paste large raw
outputs. Good additions include:

- a short table of benchmark or approximation errors
- a small set of committed figures used in the documentation
- links to JSON reports generated by release notebooks
- notes on the package version and command used to create each artefact

If a result depends on random sampling or an external backend, record the seed,
backend, package version, and command in the table.
