# %% [markdown]
# # QSVT Sign Function and Spectral Projectors
#
# In this notebook we demonstrate one of the most important “matrix-function”
# use-cases of **Quantum Singular Value Transformation (QSVT)**:
#
# - approximating the **sign function**,
# - and using it to build **spectral projectors** (subspace filters).
#
# Why this matters:
#
# - The sign function is closely related to *step functions* and *projectors*.
# - Projectors enable subspace selection and spectral filtering:
#   - keep “positive-eigenvalue” components, suppress “negative-eigenvalue” components,
#   - or more generally keep eigenvalues in a desired band.
# - This is a conceptual bridge toward:
#   - sharper filters,
#   - eigenvalue gap amplification,
#   - and physics-first applications (e.g., Hamiltonian ground-state filtering).
#
# We keep everything **small and explicit**:
# - diagonal and rotated 2×2 examples,
# - bounded odd polynomials on [-1,1],
# - visualizing how eigencomponents are transformed.

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. The sign function and its constraints
#
# The sign function is:
#
# $$
# \mathrm{sgn}(x)=
# \begin{cases}
# +1 & x>0, \\
# -1 & x<0.
# \end{cases}
# $$
#
# It is **discontinuous** at 0, which means:
# - it cannot be represented exactly by a low-degree polynomial,
# - and polynomial approximations must trade off degree vs accuracy.
#
# QSVT requires an admissible polynomial $P(x)$ satisfying:
#
# $$
# |P(x)| \le 1 \quad \text{for all } x\in[-1,1].
# $$
#
# For a sign-like approximation, $P(x)$ should also be **odd**:
#
# $$
# P(-x) = -P(x),
# $$
#
# because sign is odd.

# %% [markdown]
# ## 2. A simple bounded odd polynomial: Chebyshev T₃
#
# A natural first “sign-like” polynomial is the Chebyshev polynomial:
#
# $$
# T_3(x) = 4x^3 - 3x.
# $$
#
# Properties:
# - odd
# - bounded: $|T_3(x)| \le 1$ for $x\in[-1,1]$
#
# It does *not* approximate sgn(x) sharply, but it already pushes values
# away from 0 in a sign-consistent way.

# %%
def T3(x):
    return 4 * x**3 - 3 * x


x = np.linspace(-1, 1, 600)

plt.figure(figsize=(7, 5))
plt.plot(x, np.sign(x), "--", label="sgn(x) (ideal)")
plt.plot(x, T3(x), label=r"$T_3(x)=4x^3-3x$")
plt.axvline(0, color="grey", linestyle=":", linewidth=1)
plt.axhline(1, color="grey", linestyle="--", linewidth=0.5)
plt.axhline(-1, color="grey", linestyle="--", linewidth=0.5)
plt.title("A First Sign-Like Polynomial on [-1, 1]")
plt.xlabel("x")
plt.ylabel("value")
plt.grid(True)
plt.legend()
plt.show()

# %% [markdown]
# ## 3. A family of stronger sign-like polynomials: Tₙ for odd n
#
# Higher-order odd Chebyshev polynomials:
#
# $$
# T_n(x)=\cos(n\arccos x), \quad n\ \text{odd},
# $$
#
# become progressively more step-like.
#
# They remain bounded on [-1,1], so they are QSVT-admissible.
#
# Below we plot several odd degrees and compare them against sgn(x).

# %%
def Tn(n, x):
    return np.cos(n * np.arccos(x))


plt.figure(figsize=(7, 5))
plt.plot(x, np.sign(x), "--", label="sgn(x) (ideal)")

for n in [1, 3, 5, 7, 9]:
    plt.plot(x, Tn(n, x), label=f"T_{n}(x)")

plt.axvline(0, color="grey", linestyle=":", linewidth=1)
plt.title("Odd Chebyshev Polynomials Become More Sign-Like")
plt.xlabel("x")
plt.ylabel("value")
plt.grid(True)
plt.legend()
plt.show()

# %% [markdown]
# Qualitative takeaway:
#
# - Increasing degree pushes the curve closer to ±1 for |x| away from 0.
# - Near x=0, approximation remains limited (discontinuity).
#
# In practice, sign/projector approximations are designed on a restricted domain:
#
# $$
# x \in [-1,-\gamma] \cup [\gamma,1]
# $$
#
# where γ>0 is a spectral gap around zero.

# %% [markdown]
# ## 4. From sign to projectors
#
# The sign function gives a clean way to construct projectors:
#
# - Projector onto the **positive-eigenvalue** subspace:
#
# $$
# \Pi_{+} = \frac{I + \mathrm{sgn}(A)}{2}
# $$
#
# - Projector onto the **negative-eigenvalue** subspace:
#
# $$
# \Pi_{-} = \frac{I - \mathrm{sgn}(A)}{2}
# $$
#
# If we approximate sgn(A) using a polynomial P(A), we get an *approximate*
# projector:
#
# $$
# \tilde{\Pi}_{+} = \frac{I + P(A)}{2}.
# $$
#
# This is the core “spectral filtering” idea:
# - eigencomponents with positive eigenvalues are mapped toward 1,
# - negative eigencomponents are mapped toward 0,
# - quality improves with degree and with a larger spectral gap.

# %% [markdown]
# ## 5. Example 1: Diagonal A, explicit eigencomponent filtering
#
# Consider:
#
# $$
# A=\mathrm{diag}(-a, +a)
# $$
#
# for some a in (0,1].
#
# For a vector b = (b0, b1), applying a sign-like polynomial P(A) results in:
#
# $$
# (b_0, b_1) \mapsto (P(-a) b_0, \, P(+a) b_1).
# $$
#
# Then the projector:
#
# $$
# \tilde{\Pi}_+ b = \frac{1}{2} \left(b + P(A)b\right)
# $$
#
# suppresses the negative eigencomponent (b0) and keeps the positive one (b1).

# %%
def apply_diag(P, a, b):
    """Apply P(A) to b for A = diag(-a, +a)."""
    A_eigs = np.array([-a, a])
    return np.array([P(A_eigs[0]) * b[0], P(A_eigs[1]) * b[1]], dtype=float)


def projector_plus_from_P(P, a, b):
    """Approximate positive projector: (I + P(A))/2 applied to b."""
    Pb = apply_diag(P, a, b)
    return 0.5 * (b + Pb)


a = 0.5
b = np.array([1.0, 1.0])  # equal components

for n in [1, 3, 5, 7, 9]:
    P = lambda x, n=n: Tn(n, x)
    out = projector_plus_from_P(P, a, b)
    out_norm = out / np.linalg.norm(out)
    print(f"n={n:2d}  Pi_+ b (normalized) = {np.round(out_norm, 6)}")

# %% [markdown]
# Interpreting the output:
#
# - As degree increases, the negative eigencomponent is suppressed more strongly.
# - The vector approaches the “pure positive-eigenvalue basis vector” direction.
#
# This is the core mechanism of spectral projectors: they separate subspaces
# based on the sign (or band) of eigenvalues.

# %% [markdown]
# ## 6. Visualizing projector quality vs degree
#
# For the simple 2×2 diagonal case, we can quantify “how much” of the output
# lies in the positive-eigenvalue component (second component).
#
# We compute:
# - the normalized output,
# - and track the squared magnitude of the positive component.

# %%
degrees = [1, 3, 5, 7, 9, 11, 13]
pos_weight = []

for n in degrees:
    P = lambda x, n=n: Tn(n, x)
    out = projector_plus_from_P(P, a, b)
    out_norm = out / np.linalg.norm(out)
    pos_weight.append(out_norm[1] ** 2)

plt.figure(figsize=(6, 4))
plt.plot(degrees, pos_weight, "o-")
plt.ylim(0, 1.05)
plt.title("Approximate Positive-Subspace Weight vs Polynomial Degree")
plt.xlabel("Odd degree n (Chebyshev T_n)")
plt.ylabel("Weight on positive component (squared)")
plt.grid(True)
plt.show()

# %% [markdown]
# The trend illustrates:
#
# - Higher-degree polynomials can provide sharper projectors,
# - assuming the eigenvalues are not too close to 0 (spectral gap).
#
# In larger problems, the same principle applies:
# - projectors are stronger when the spectrum has a clear separation.

# %% [markdown]
# ## 7. Example 2: Non-diagonal A (basis-independence)
#
# QSVT transformations depend on eigenvalues/singular values, not on the basis.
#
# To demonstrate this, we take:
#
# $$
# A = U \,\mathrm{diag}(-a, +a)\, U^\dagger,
# $$
#
# i.e. a rotated version of the diagonal matrix with the same eigenvalues.
#
# The projector still selects the +a eigenspace, but now that eigenspace
# is a rotated direction in the computational basis.

# %%
def rotation_U(theta):
    """A real orthogonal rotation matrix (also unitary)."""
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=float
    )


theta = 0.7
U = rotation_U(theta)

A_diag = np.diag([-a, a])
A = U @ A_diag @ U.T  # real symmetric (Hermitian)

eigvals, eigvecs = np.linalg.eigh(A)

print("A =\n", np.round(A, 6))
print("Eigenvalues =", np.round(eigvals, 6))
print("Eigenvectors (columns) =\n", np.round(eigvecs, 6))

# %% [markdown]
# Now we apply the approximate projector in the eigenbasis directly:
#
# - compute b in eigenbasis,
# - apply (I + P(Λ))/2,
# - rotate back.
#
# This is exactly what QSVT does conceptually (spectral transformation).

# %%
def projector_plus_general(P, A, b):
    """Apply (I + P(A))/2 to b using eigendecomposition (classical spectral calc)."""
    evals, evecs = np.linalg.eigh(A)
    # b in eigenbasis
    coeffs = evecs.T @ b
    # apply projector in eigenbasis
    P_evals = np.array([P(lam) for lam in evals], dtype=float)
    proj_factors = 0.5 * (1.0 + P_evals)
    coeffs_out = proj_factors * coeffs
    # rotate back
    return evecs @ coeffs_out


b2 = np.array([1.0, 1.0], dtype=float)

plt.figure(figsize=(7, 5))
plt.title("Approximate Projector Output for Rotated A (normalized vectors)")
plt.xlabel("component 0")
plt.ylabel("component 1")
plt.grid(True)

# plot original b direction
b2n = b2 / np.linalg.norm(b2)
plt.arrow(0, 0, b2n[0], b2n[1], width=0.01, length_includes_head=True, label="b")

# plot projected directions for different degrees
for n in [1, 3, 5, 7, 9]:
    P = lambda x, n=n: Tn(n, x)
    out = projector_plus_general(P, A, b2)
    outn = out / np.linalg.norm(out)
    plt.arrow(0, 0, outn[0], outn[1], width=0.005, length_includes_head=True)

plt.xlim(-1.05, 1.05)
plt.ylim(-1.05, 1.05)
plt.axhline(0, color="grey", linestyle="--", linewidth=0.5)
plt.axvline(0, color="grey", linestyle="--", linewidth=0.5)
plt.show()

# %% [markdown]
# Interpretation:
#
# - The output direction changes with degree, approaching the true +eigenspace direction.
# - The effect is basis-independent: the same spectral filtering occurs even though A is not diagonal.
#
# This is the essential message:
#
# > QSVT (and its polynomial approximations) act on eigenvalues/singular values,
# > and therefore generalize naturally beyond diagonal matrices.

# %% [markdown]
# ## 8. Connection back to QSVT
#
# In a true QSVT implementation:
#
# - we would provide a block encoding of A,
# - choose an admissible odd polynomial P approximating sgn(x) away from 0,
# - and QSVT would implement P(A) within the encoded subspace.
#
# The “projector trick”:
#
# $$
# \tilde{\Pi}_+ = \frac{I + P(A)}{2}
# $$
#
# is then implemented by combining:
# - P(A) from QSVT,
# - plus simple linear combination with the identity.
#
# This notebook focused on the spectral mechanism and polynomial behaviour.
# Circuit-level construction is handled (in earlier notebooks) by PennyLane’s
# `qml.qsvt` interface for small matrices.

# %% [markdown]
# ## Summary
#
# - The sign function is a central spectral transformation.
# - QSVT can approximate sign via bounded odd polynomials on [-1,1].
# - From sign, we can build spectral projectors:
#   - \(\Pi_+ = (I + \mathrm{sgn}(A))/2\),
#   - \(\Pi_- = (I - \mathrm{sgn}(A))/2\).
# - Higher-degree polynomials give sharper filtering away from x=0.
# - The filtering effect is basis-independent and generalizes beyond diagonal matrices.
#
# This completes an important educational layer of QSVT:
# **spectral projectors and subspace selection**.