"""
Reusable small Hamiltonian constructors for physics examples.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

_PAULI = {
    "I": np.eye(2, dtype=float),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float),
}


def pauli_string_matrix(pauli_string: str) -> np.ndarray:
    """
    Construct the matrix for a Pauli string such as "XZI".
    """
    if not pauli_string:
        raise ValueError("pauli_string must be nonempty.")

    result = np.array([[1.0]], dtype=complex)
    for char in pauli_string.upper():
        if char not in _PAULI:
            raise ValueError("pauli_string may only contain I, X, Y, and Z.")
        result = np.kron(result, _PAULI[char])
    return np.real_if_close(result)


def tight_binding_chain(
    n_sites: int,
    *,
    hopping: float = 1.0,
    onsite: Iterable[float] | None = None,
    periodic: bool = False,
) -> np.ndarray:
    """
    Build a 1D nearest-neighbor tight-binding Hamiltonian.
    """
    if n_sites <= 0:
        raise ValueError("n_sites must be positive.")

    H = np.zeros((n_sites, n_sites), dtype=float)
    for i in range(n_sites - 1):
        H[i, i + 1] = -hopping
        H[i + 1, i] = -hopping
    if periodic and n_sites > 2:
        H[0, -1] = -hopping
        H[-1, 0] = -hopping
    if onsite is not None:
        onsite_arr = np.asarray(list(onsite), dtype=float)
        if onsite_arr.shape != (n_sites,):
            raise ValueError("onsite must have length n_sites.")
        H = H + np.diag(onsite_arr)
    return H


def ising_hamiltonian(
    n_spins: int,
    *,
    coupling: float = 1.0,
    transverse_field: float = 1.0,
    periodic: bool = False,
) -> np.ndarray:
    """
    Build a transverse-field Ising Hamiltonian.
    """
    if n_spins <= 0:
        raise ValueError("n_spins must be positive.")

    dim = 2**n_spins
    H = np.zeros((dim, dim), dtype=float)
    pairs = [(i, i + 1) for i in range(n_spins - 1)]
    if periodic and n_spins > 2:
        pairs.append((n_spins - 1, 0))

    for i, j in pairs:
        ops = ["I"] * n_spins
        ops[i] = "Z"
        ops[j] = "Z"
        H = H - coupling * pauli_string_matrix("".join(ops)).real

    for i in range(n_spins):
        ops = ["I"] * n_spins
        ops[i] = "X"
        H = H - transverse_field * pauli_string_matrix("".join(ops)).real
    return H


def heisenberg_chain(
    n_spins: int,
    *,
    jx: float = 1.0,
    jy: float = 1.0,
    jz: float = 1.0,
    periodic: bool = False,
) -> np.ndarray:
    """
    Build a small spin-1/2 Heisenberg-chain Hamiltonian.
    """
    if n_spins <= 1:
        raise ValueError("n_spins must be greater than 1.")

    dim = 2**n_spins
    H = np.zeros((dim, dim), dtype=complex)
    pairs = [(i, i + 1) for i in range(n_spins - 1)]
    if periodic and n_spins > 2:
        pairs.append((n_spins - 1, 0))

    for i, j in pairs:
        for label, strength in [("X", jx), ("Y", jy), ("Z", jz)]:
            ops = ["I"] * n_spins
            ops[i] = label
            ops[j] = label
            H = H + strength * pauli_string_matrix("".join(ops))
    return np.real_if_close(H)


__all__ = [
    "heisenberg_chain",
    "ising_hamiltonian",
    "pauli_string_matrix",
    "tight_binding_chain",
]
