# Curated Real-World Examples

These eight notebooks are the maintained application gallery for
`qsvt-pennylane`. Together they cover the package's main end-to-end paths
without repeating closely related spectral-filter, lattice, and PDE surveys.

| notebook | application | primary package path |
| --- | --- | --- |
| `01_poisson_equation_pde.ipynb` | Poisson PDE inversion | planning, block encoding, finite QSVT, classical comparison |
| `02_hamiltonian_simulation_schrodinger_dynamics.ipynb` | real-time dynamics | paired polynomial design and Hamiltonian workflow acceptance |
| `03_greens_function_response.ipynb` | resolvents and response functions | matrix-function design and classical spectral validation |
| `04_ising_phase_transition_filtering.ipynb` | many-body spectral filtering | Pauli-LCU filtering and finite-QSVT acceptance |
| `05_fermi_dirac_electronic_occupations.ipynb` | thermal electronic occupations | bounded Fermi–Dirac polynomial workflow |
| `06_topological_band_projector_chern_marker.ipynb` | topological band projectors | projector design and observable diagnostics |
| `07_singular_value_pseudoinverse_deblurring.ipynb` | inverse imaging | singular-value pseudoinverse workflow |
| `08_matrix_log_entropy_graph_laplacian.ipynb` | graph entropy | matrix-log workflow and spectral reference |

All notebooks are thin clients of tested package functionality. Shared
presentation and output-path helpers live in `notebooks._support`; they are not
part of the installed package.

Execute the curated set with:

```bash
pytest -m notebook tests/test_real_example_notebooks.py
```

Refresh extracted plots and the generated result ledger with:

```bash
python scripts/extract_notebook_plots.py \
  --preset real-examples --execute --write-docs
```
