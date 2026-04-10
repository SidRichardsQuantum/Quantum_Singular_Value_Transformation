# %% [markdown]
# # QSVT as Matrix Functions: Powers and Roots
#
# In this notebook we demonstrate a key conceptual idea:
#
# > **QSVT is a general method for implementing matrix functions f(A)**,
# > not just inversion or filtering.
#
# All QSVT constructions ultimately implement **polynomial transformations**
# of singular values / eigenvalues. By approximating a target function f(x)
# with a bounded polynomial P(x), QSVT implements P(A) ≈ f(A).
#
# Here we focus on:
# - matrix powers,
# - matrix square roots,
# - fractional powers,
#
# using small matrices and explicit spectra to build intuition.

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Matrix functions as spectral maps
#
# If a matrix is diagonalizable:
#
# $$ A = U \Lambda U^\dagger, $$
#
# then for any function f:
#
# $$ f(A) = U f(\Lambda) U^\dagger. $$
#
# That is:
# - eigenvectors are preserved,
# - eigenvalues are transformed individually.
#
# QSVT implements exactly this mechanism, but using polynomial
# approximations and quantum circuits.

# %% [markdown]
# ## 2. A simple test matrix
#
# We use a 2×2 Hermitian matrix with eigenvalues in (0,1]:
#
# $$ A = U \, \mathrm{diag}(\lambda_1,\lambda_2) \, U^\dagger, $$
#
# so that all functions we consider are well-defined.


# %%
def rotation(theta):
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=float,
    )


theta = 0.6
U = rotation(theta)

lambdas = np.array([0.2, 0.8])
A = U @ np.diag(lambdas) @ U.T

eigvals, eigvecs = np.linalg.eigh(A)

print("A =\n", np.round(A, 6))
print("Eigenvalues =", eigvals)

# %% [markdown]
# ## 3. Warm-up: integer powers of A
#
# Matrix powers act on eigenvalues as:
#
# $$ \lambda_i \mapsto \lambda_i^k. $$
#
# For 0 < λ < 1, increasing k:
# - suppresses small eigenvalues,
# - sharpens spectral separation.
#
# This already hints at filtering behaviour.


# %%
def matrix_power(A, k):
    return np.linalg.matrix_power(A, k)


powers = [1, 2, 3, 4]

plt.figure(figsize=(7, 5))
for k in powers:
    plt.plot(
        eigvals,
        eigvals**k,
        "o-",
        label=f"$\\lambda^{{{k}}}$",
    )

plt.xlabel("original eigenvalue")
plt.ylabel("transformed eigenvalue")
plt.title("Spectral Effect of Matrix Powers")
plt.grid(True)
plt.legend()
plt.show()

# %% [markdown]
# Note:
#
# - Even powers lose sign information (for general matrices),
# - Odd powers preserve sign,
# - Powers >1 increasingly suppress small eigenvalues.
#
# These simple functions already behave like **spectral filters**.

# %% [markdown]
# ## 4. Non-polynomial functions: square roots
#
# The square root function:
#
# $$ f(x) = \sqrt{x} $$
#
# is not a polynomial, but it is smooth and bounded on [a,1] for a>0.
#
# We can approximate it with a low-degree polynomial on this interval.

# %% [markdown]
# ### Polynomial approximation of √x on [a,1]
#
# For demonstration, we use a Chebyshev interpolation on [a,1].
# This is not the only method, but it illustrates the idea cleanly.


# %%
def chebyshev_fit(func, a, degree):
    """Chebyshev polynomial approximation on [a,1]."""
    xs = np.linspace(a, 1.0, 500)
    ys = func(xs)
    coeffs = np.polynomial.chebyshev.chebfit(xs, ys, degree)
    return lambda x: np.polynomial.chebyshev.chebval(x, coeffs)


a = 0.2
deg = 6

P_sqrt = chebyshev_fit(np.sqrt, a, deg)

x = np.linspace(a, 1.0, 400)

plt.figure(figsize=(7, 5))
plt.plot(x, np.sqrt(x), "--", label="√x (exact)")
plt.plot(x, P_sqrt(x), label=f"Polynomial approx (deg={deg})")
plt.xlabel("x")
plt.ylabel("value")
plt.title("Polynomial Approximation to √x on [a,1]")
plt.grid(True)
plt.legend()
plt.show()

# %% [markdown]
# The approximation is not perfect, but improves with degree.
# In QSVT, the polynomial degree directly controls circuit depth.

# %% [markdown]
# ## 5. Applying √A spectrally
#
# We now apply:
#
# - exact √A (via eigen-decomposition),
# - approximate P(A),
#
# and compare the results.


# %%
def apply_function(A, f):
    evals, evecs = np.linalg.eigh(A)
    return evecs @ np.diag(f(evals)) @ evecs.T


sqrtA_exact = apply_function(A, np.sqrt)
sqrtA_poly = apply_function(A, P_sqrt)

print("√A (exact) =\n", np.round(sqrtA_exact, 6))
print("\n√A (poly approx) =\n", np.round(sqrtA_poly, 6))

# %% [markdown]
# The two matrices are close, even with a modest polynomial degree.
#
# This is exactly the object that QSVT would implement if we used
# P(x) as the QSVT polynomial.

# %% [markdown]
# ## 6. Fractional powers x^α
#
# More generally, we may want:
#
# $$ f(x) = x^\alpha, \quad 0<\alpha<1. $$
#
# These functions interpolate between identity and projection-like behaviour.
#
# Below we compare several α values.

# %%
alphas = [0.25, 0.5, 0.75]

plt.figure(figsize=(7, 5))
for alpha in alphas:
    plt.plot(eigvals, eigvals**alpha, "o-", label=f"$x^{alpha}$")

plt.xlabel("original eigenvalue")
plt.ylabel("transformed eigenvalue")
plt.title("Fractional Matrix Powers (Spectral View)")
plt.grid(True)
plt.legend()
plt.show()

# %% [markdown]
# As α decreases:
# - small eigenvalues are amplified relative to large ones,
# - the transformation becomes more “flattening.”
#
# All such functions can be approximated by bounded polynomials
# on restricted domains and implemented with QSVT.

# %% [markdown]
# ## 7. Why matrix functions matter
#
# Many quantum algorithms are naturally expressed as matrix functions:
#
# - Inversion: f(x)=1/x
# - Square root: f(x)=√x
# - Projectors: f(x)=step(x)
# - Hamiltonian evolution: f(x)=e^{-ixt}
# - Resolvents: f(x)=(x-z)^{-1}
#
# QSVT unifies these under **one mechanism**:
#
# > polynomial approximation + spectral transformation.

# %% [markdown]
# ## Summary
#
# - QSVT implements polynomial approximations to matrix functions.
# - Matrix functions act on eigenvalues, preserving eigenvectors.
# - Powers and roots are simple but powerful examples.
# - Domain restriction makes non-polynomial functions admissible.
# - This viewpoint unifies filtering, inversion, and simulation.
#
# The next notebook builds on this to show how **projectors and subspace selection**
# arise from sign-function approximations.
