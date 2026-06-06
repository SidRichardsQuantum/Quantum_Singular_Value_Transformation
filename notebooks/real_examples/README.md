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
| Hamiltonian simulation | `06_hamiltonian_simulation_schrodinger_dynamics.ipynb` |
| Ground-state filtering | `01_ground_state_filtering.ipynb` |
| Quantum chemistry | `07_quantum_chemistry_h2_toy_solver.ipynb` |
| Green's functions | `08_greens_function_response.ipynb` |
| Spectral density estimation | `09_spectral_density_estimation.ipynb` |
| Gibbs states | `10_gibbs_state_thermal_weights.ipynb` |
| PDE linear systems | `05_poisson_equation_pde.ipynb` |
| PDE heat flow | `13_heat_equation_2d_pde.ipynb` |
| Advection-diffusion PDE | `14_advection_diffusion_pde.ipynb` |
| Wave equation dynamics | `15_wave_equation_dynamics.ipynb` |
| Helmholtz PDE | `16_helmholtz_equation_pde.ipynb` |
| Quantum walk search | `17_quantum_walk_search_toy.ipynb` |
| SSH edge states | `18_ssh_chain_edge_state_filtering.ipynb` |
| Anderson localization | `19_anderson_localization.ipynb` |
| Schrödinger bound states | `20_schrodinger_bound_states.ipynb` |
| Harmonic oscillator | `21_quantum_harmonic_oscillator_grid.ipynb` |
| Electrostatic Green's function | `22_electrostatic_green_function_poisson.ipynb` |
| Coupled oscillator modes | `23_coupled_oscillator_normal_modes.ipynb` |
| Ising phase transition | `24_ising_phase_transition_filtering.ipynb` |
| Diffusion heat treatment | `25_diffusion_heat_treatment_slab.ipynb` |
| Graphene nanoribbon DOS | `26_graphene_nanoribbon_density_of_states.ipynb` |
| Fermi-Dirac occupations | `27_fermi_dirac_electronic_occupations.ipynb` |
| Photonic crystal band gap | `28_photonic_crystal_band_gap_filtering.ipynb` |
| Topological band projector | `29_topological_band_projector_chern_marker.ipynb` |
| Block-encoded Laplacian smoothing | `30_block_encoded_laplacian_smoothing.ipynb` |
| Transport physics | `11_transport_physics_landauer_chain.ipynb` |
| Tensor-network hybrids | `12_tensor_network_hybrid_filtering.ipynb` |
