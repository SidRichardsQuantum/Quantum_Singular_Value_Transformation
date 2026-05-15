"""
Finite-difference PDE operators used by physics examples.
"""

from __future__ import annotations

import numpy as np


def dirichlet_laplacian_1d(
    n_points: int,
    *,
    length: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return grid points and the 1D negative Laplacian with Dirichlet boundaries.
    """
    if n_points <= 0:
        raise ValueError("n_points must be positive.")
    if length <= 0.0:
        raise ValueError("length must be positive.")

    dx = length / (n_points + 1)
    main = 2.0 * np.ones(n_points) / dx**2
    off = -1.0 * np.ones(n_points - 1) / dx**2
    matrix = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    grid = np.linspace(dx, length - dx, n_points)
    return grid, matrix


def periodic_laplacian_1d(
    n_points: int,
    *,
    length: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return grid points and the 1D negative Laplacian with periodic boundaries.
    """
    if n_points <= 2:
        raise ValueError("n_points must be greater than 2.")
    if length <= 0.0:
        raise ValueError("length must be positive.")

    dx = length / n_points
    main = 2.0 * np.ones(n_points) / dx**2
    off = -1.0 * np.ones(n_points - 1) / dx**2
    matrix = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    matrix[0, -1] = -1.0 / dx**2
    matrix[-1, 0] = -1.0 / dx**2
    grid = np.linspace(0.0, length - dx, n_points)
    return grid, matrix


def dirichlet_laplacian_2d(
    nx: int,
    ny: int,
    *,
    lx: float = 1.0,
    ly: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return x-grid, y-grid, and the 2D Dirichlet negative Laplacian.
    """
    x, lx_op = dirichlet_laplacian_1d(nx, length=lx)
    y, ly_op = dirichlet_laplacian_1d(ny, length=ly)
    matrix = np.kron(lx_op, np.eye(ny)) + np.kron(np.eye(nx), ly_op)
    return x, y, matrix


__all__ = [
    "dirichlet_laplacian_1d",
    "dirichlet_laplacian_2d",
    "periodic_laplacian_1d",
]
