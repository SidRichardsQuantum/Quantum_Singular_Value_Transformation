# Notebooks

The repository includes notebook-first examples that introduce QSVT and QSP
concepts step by step, followed by real physics workflows and classical
benchmark notebooks that reuse the package APIs.

## Notebook outputs

The rendered results pages are generated from the embedded outputs of these
notebooks:

| notebook set | generated output page | source directory |
| --- | --- | --- |
| tutorials | [Tutorial notebook outputs](tutorial_results.md) | `notebooks/tutorials/` |
| real physics examples | [Real-example notebook outputs](real_example_results.md) | `notebooks/real_examples/` |
| benchmarks | [Benchmark notebook outputs](benchmark_results.md) | `notebooks/benchmarks/` |

Benchmark notebooks are package-client workflows that write JSON/CSV artifacts
under `results/benchmarks/` and `results/tables/`; their plots are extracted
to `results/plots/benchmarks/`.

For the compact result summary and regeneration commands, see
[Results](results.md).

## Tutorial notebooks

Tutorial notebooks live in `notebooks/tutorials/`.

| notebook | focus |
|---|---|
| `tutorials/01_QSVT_Scalar_and_Diagonal_Matrix.ipynb` | scalar and diagonal QSVT transforms |
| `tutorials/02_QSVT_Singular_Value_Filter.ipynb` | singular-value filtering |
| `tutorials/03_QSP_Polynomial_Demo.ipynb` | QSP polynomial behaviour |
| `tutorials/04_QSVT_Linear_Solver_2x2.ipynb` | small linear-solver intuition |
| `tutorials/05_QSVT_Linear_Solver_4x4.ipynb` | larger diagonal linear-solver experiment |
| `tutorials/06_QSVT_Linear_Solver_Approximate.ipynb` | approximate inverse-like transforms |
| `tutorials/07_QSVT_Polynomial_Design_and_Approximation.ipynb` | polynomial design and approximation |
| `tutorials/08_QSVT_Matrix_Functions_Powers_and_Roots.ipynb` | matrix functions, powers, and roots |
| `tutorials/09_QSVT_Sign_Function_and_Projectors.ipynb` | sign functions and projectors |
| `tutorials/10_QSVT_Design_and_Templates.ipynb` | design helpers and templates |
| `tutorials/11_QSVT_Algorithm_Workflows.ipynb` | QSVT algorithm workflow diagnostics |
| `tutorials/12_QSVT_Reports_CLI_and_Artifacts.ipynb` | QSVT reports, CLI output, and reproducible artifacts |
| `tutorials/13_QSVT_Design_Tradeoffs.ipynb` | QSVT design degree/error/boundedness tradeoffs |
| `tutorials/14_QSVT_Resource_Proxy_Limits.ipynb` | block-encoding assumptions and QSVT resource-proxy limits |
| `tutorials/15_Block_Encoded_QSVT_Workflow.ipynb` | finite block-encoded QSVT workflow |
| `tutorials/16_Sparse_Oracle_Assumptions.ipynb` | sparse operators, oracle assumptions, and omitted costs |
| `tutorials/17_QSVT_Compatibility_Failure_Cases.ipynb` | boundedness, parity, and QSVT compatibility failures |

## Benchmark notebooks

Benchmark notebooks live in `notebooks/benchmarks/`.

| notebook | focus |
|---|---|
| `benchmarks/01_linear_system_classical_vs_qsvt_proxy.ipynb` | dense and CG linear-system baselines with QSVT inverse-polynomial resource proxies |
| `benchmarks/02_matrix_functions_spectral_baselines.ipynb` | dense spectral and polynomial matrix-function baselines for thermal/filter workflows |
| `benchmarks/03_scaling_sweeps.ipynb` | dimension, conditioning, and inverse-degree sweeps for compact benchmark tables |
| `benchmarks/04_classical_baseline_assumptions.ipynb` | what classical benchmark helpers time versus what QSVT proxy fields estimate |

## Real physics examples

Real physics examples live in `notebooks/real_examples/`.
Each notebook now starts with a short orientation block that identifies the
physical system, the QSVT-style implementation used, and the classical reference
or quantum-relevance context for the toy-scale example.

| notebook | focus |
|---|---|
| `01_ground_state_filtering.ipynb` | ground-state filtering |
| `02_tight_binding_band_filter.ipynb` | tight-binding band filters |
| `03_imaginary_time_filtering.ipynb` | imaginary-time filtering |
| `04_heat_equation_pde.ipynb` | heat-equation PDE workflow |
| `05_poisson_equation_pde.ipynb` | Poisson-equation PDE workflow |
| `06_hamiltonian_simulation_schrodinger_dynamics.ipynb` | Hamiltonian simulation |
| `07_quantum_chemistry_h2_toy_solver.ipynb` | H2 toy chemistry solver |
| `08_greens_function_response.ipynb` | Green's-function response |
| `09_spectral_density_estimation.ipynb` | spectral density estimation |
| `10_gibbs_state_thermal_weights.ipynb` | Gibbs-state thermal weights |
| `11_transport_physics_landauer_chain.ipynb` | transport physics and Landauer chains |
| `12_tensor_network_hybrid_filtering.ipynb` | tensor-network hybrid filtering |
| `13_heat_equation_2d_pde.ipynb` | 2D heat-equation PDE workflow |
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
| `29_topological_band_projector_chern_marker.ipynb` | topological band projectors and Chern markers |
| `30_block_encoded_laplacian_smoothing.ipynb` | block-encoded QSVT Laplacian smoothing |

See the repository notebook directory for executable files:
[notebooks](https://github.com/SidRichardsQuantum/Quantum_Singular_Value_Transformation/tree/main/notebooks).
