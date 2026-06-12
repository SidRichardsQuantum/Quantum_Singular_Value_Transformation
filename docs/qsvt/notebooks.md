# Notebooks

The repository has three notebook roles.

- Tutorial notebooks are succinct package-client walkthroughs for how to use
  the available algorithms, implementations, diagnostics, reports, and CLI
  commands.
- Real-example notebooks apply the general package workflows to concrete
  physics and mathematics problems with only domain setup and interpretation in
  the notebook.
- Benchmark notebooks compare quantum/QSVT algorithms or implementations
  against the relevant classical algorithms or implementations for the same
  task.

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
to `results/plots/benchmarks/`. They should use implemented finite
PennyLane/QSVT execution paths where available and clearly labeled QSVT
resource proxies where a full quantum implementation is not the measured path.

For the compact result summary and regeneration commands, see
[Results](results.md).

## Tutorial notebooks

Tutorial notebooks live in `notebooks/tutorials/`.
They should stay compact and teach package usage patterns rather than carrying
domain-specific implementation logic.

| notebook | focus |
|---|---|
| `tutorials/01_QSVT_Scalar_and_Diagonal_Matrix.ipynb` | scalar and diagonal QSVT transforms |
| `tutorials/02_QSVT_Singular_Value_Filter.ipynb` | singular-value filtering |
| `tutorials/03_QSP_Polynomial_Demo.ipynb` | QSP polynomial behaviour |
| `tutorials/04_QSVT_Exact_Linear_Solver_Toy_Cases.ipynb` | exact 2x2 and 4x4 toy linear-solver cases |
| `tutorials/05_QSVT_Approximate_Linear_Solver.ipynb` | approximate inverse-like transforms |
| `tutorials/06_QSVT_Polynomial_Design_and_Approximation.ipynb` | polynomial design and approximation |
| `tutorials/07_QSVT_Matrix_Functions_Powers_and_Roots.ipynb` | matrix functions, powers, and roots |
| `tutorials/08_QSVT_Sign_Function_and_Projectors.ipynb` | sign functions and projectors |
| `tutorials/09_QSVT_Design_and_Templates.ipynb` | design helpers and templates |
| `tutorials/10_QSVT_Algorithm_Workflows.ipynb` | QSVT algorithm workflow diagnostics |
| `tutorials/11_QSVT_Reports_CLI_and_Artifacts.ipynb` | QSVT reports, CLI output, and reproducible artifacts |
| `tutorials/12_QSVT_Design_Tradeoffs.ipynb` | QSVT design degree/error/boundedness tradeoffs |
| `tutorials/13_QSVT_Resource_Proxy_Limits.ipynb` | block-encoding assumptions and QSVT resource-proxy limits |
| `tutorials/14_Block_Encoded_QSVT_Workflow.ipynb` | finite block-encoded QSVT workflow |
| `tutorials/15_Sparse_Oracle_Assumptions.ipynb` | sparse operators, oracle assumptions, and omitted costs |
| `tutorials/16_QSVT_Compatibility_Failure_Cases.ipynb` | boundedness, parity, and QSVT compatibility failures |
| `tutorials/17_QSVT_Linear_System_Comparisons.ipynb` | dense, CG, and QSVT-style linear-system comparison rows |
| `tutorials/18_HHL_Linear_System_Solver.ipynb` | finite simulator-scale HHL linear-system solver |
| `tutorials/19_Quantum_Walk_Search_Workflow.ipynb` | continuous-time quantum walk search and polynomial phase approximation |

## Benchmark notebooks

Benchmark notebooks live in `notebooks/benchmarks/`.
They should benchmark the quantum/QSVT algorithm or implementation being
studied against the relevant classical algorithm or implementation, such as a
dense direct solve, conjugate-gradient solve, dense spectral matrix function,
or polynomial matrix evaluation.

| notebook | focus |
|---|---|
| `benchmarks/01_linear_system_classical_vs_qsvt_proxy.ipynb` | dense/CG linear-system baselines, QSVT inverse-polynomial resource proxies, and finite HHL execution diagnostics |
| `benchmarks/02_matrix_functions_spectral_baselines.ipynb` | dense spectral and polynomial matrix-function baselines for thermal/filter workflows |
| `benchmarks/03_scaling_sweeps.ipynb` | dimension, conditioning, and inverse-degree sweeps for compact benchmark tables |
| `benchmarks/04_classical_baseline_assumptions.ipynb` | what classical benchmark helpers time versus what QSVT proxy fields estimate |
| `benchmarks/05_quantum_walk_search_scaling.ipynb` | quantum walk search success, polynomial error, and QSVT signal-call proxies |

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
| `17_ssh_chain_edge_state_filtering.ipynb` | SSH-chain edge-state filtering |
| `18_anderson_localization.ipynb` | Anderson localization |
| `19_schrodinger_bound_states.ipynb` | Schrödinger bound states |
| `20_quantum_harmonic_oscillator_grid.ipynb` | harmonic-oscillator grid spectrum |
| `21_electrostatic_green_function_poisson.ipynb` | electrostatic Green's function |
| `22_coupled_oscillator_normal_modes.ipynb` | coupled-oscillator normal modes |
| `23_ising_phase_transition_filtering.ipynb` | Ising phase-transition filtering |
| `24_diffusion_heat_treatment_slab.ipynb` | diffusion-limited heat treatment |
| `25_graphene_nanoribbon_density_of_states.ipynb` | graphene nanoribbon density of states |
| `26_fermi_dirac_electronic_occupations.ipynb` | Fermi-Dirac electronic occupations |
| `27_photonic_crystal_band_gap_filtering.ipynb` | photonic-crystal band-gap filtering |
| `28_topological_band_projector_chern_marker.ipynb` | topological band projectors and Chern markers |
| `29_block_encoded_laplacian_smoothing.ipynb` | block-encoded QSVT Laplacian smoothing |

See the repository notebook directory for executable files:
[notebooks](https://github.com/SidRichardsQuantum/Quantum_Singular_Value_Transformation/tree/main/notebooks).
