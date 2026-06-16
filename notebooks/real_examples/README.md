# Real Physics Examples

These notebooks are small, executable real-world physics workflows for
`qsvt-pennylane`. They use finite-dimensional Hamiltonians and PDE
discretizations so each QSVT-style polynomial transform can be compared against
an exact classical spectral reference. Most examples are dense spectral
workflows; the block-encoded Laplacian example additionally verifies an
explicit finite block encoding and runs a finite PennyLane QNode execution
check.

| Area | Notebook |
| --- | --- |
| Ground-state filtering | `01_ground_state_filtering.ipynb` |
| Tight-binding band filters | `02_tight_binding_band_filter.ipynb` |
| Imaginary-time filtering | `03_imaginary_time_filtering.ipynb` |
| PDE linear systems | `04_poisson_equation_pde.ipynb` |
| Hamiltonian simulation | `05_hamiltonian_simulation_schrodinger_dynamics.ipynb` |
| Quantum chemistry | `06_quantum_chemistry_h2_toy_solver.ipynb` |
| Green's functions | `07_greens_function_response.ipynb` |
| Spectral density estimation | `08_spectral_density_estimation.ipynb` |
| Gibbs states | `09_gibbs_state_thermal_weights.ipynb` |
| Transport physics | `10_transport_physics_landauer_chain.ipynb` |
| Tensor-network hybrids | `11_tensor_network_hybrid_filtering.ipynb` |
| PDE heat flow | `12_heat_equation_2d_pde.ipynb` |
| Advection-diffusion PDE | `13_advection_diffusion_pde.ipynb` |
| Wave equation dynamics | `14_wave_equation_dynamics.ipynb` |
| Helmholtz PDE | `15_helmholtz_equation_pde.ipynb` |
| SSH edge states | `16_ssh_chain_edge_state_filtering.ipynb` |
| Anderson localization | `17_anderson_localization.ipynb` |
| Schrödinger bound states | `18_schrodinger_bound_states.ipynb` |
| Harmonic oscillator | `19_quantum_harmonic_oscillator_grid.ipynb` |
| Electrostatic Green's function | `20_electrostatic_green_function_poisson.ipynb` |
| Coupled oscillator modes | `21_coupled_oscillator_normal_modes.ipynb` |
| Ising phase transition | `22_ising_phase_transition_filtering.ipynb` |
| Diffusion heat treatment | `23_diffusion_heat_treatment_slab.ipynb` |
| Graphene nanoribbon DOS | `24_graphene_nanoribbon_density_of_states.ipynb` |
| Fermi-Dirac occupations | `25_fermi_dirac_electronic_occupations.ipynb` |
| Photonic crystal band gap | `26_photonic_crystal_band_gap_filtering.ipynb` |
| Topological band projector | `27_topological_band_projector_chern_marker.ipynb` |
| Block-encoded Laplacian smoothing | `28_block_encoded_laplacian_smoothing.ipynb` |
| Singular-value pseudoinverse deblurring | `29_singular_value_pseudoinverse_deblurring.ipynb` |
| Matrix-log graph entropy | `30_matrix_log_entropy_graph_laplacian.ipynb` |
