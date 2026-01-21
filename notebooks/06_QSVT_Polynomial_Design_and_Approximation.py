# %% [markdown]
# # QSVT Polynomial Design and Approximation
#
# In this notebook we address a natural next question:
#
# > *How do we design the polynomial used in QSVT?*
#
# So far, we have used hand-picked polynomials:
#
# - $f(x) = x^2$
# - $f(x) = x$
# - $f(x) = x^3$
# - $T_3(x) = 4x^3 - 3x$
#
# These are pedagogically useful, but QSVT becomes powerful when we
# **systematically approximate target functions** using bounded polynomials.
#
# In this notebook we:
#
# 1. Review why QSVT requires polynomial approximations.
# 2. Explain why Chebyshev polynomials are the natural basis.
# 3. Construct low-degree polynomial approximations to inverse-like functions.
# 4. Connect approximation quality to QSVT-based linear solvers.
#
# This notebook is *theoretical and visual*: it focuses on understanding
# polynomial behaviour rather than circuit construction.

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Why QSVT uses polynomials
#
# QSVT implements **polynomial transformations** of singular values because:
#
# - Quantum circuits are unitary and therefore bounded.
# - Any function implemented via QSVT must satisfy:
#
# $$ |f(x)| \le 1 \quad \text{for } x \in [-1,1]. $$
#
# This rules out many functions of direct interest (e.g. $1/x$),
# but **polynomial approximations** allow us to work within this constraint.
#
# In practice:
#
# - We approximate a target function $g(x)$ by a bounded polynomial $P(x)$.
# - QSVT implements $P(A)$ instead of $g(A)$.
# - After normalization, this can still recover the desired solution *direction*.

# %% [markdown]
# ## 2. Chebyshev polynomials
#
# The Chebyshev polynomials of the first kind are defined as:
#
# $$ T_n(x) = \cos(n \arccos x). $$
#
# Key properties:
#
# - $|T_n(x)| \le 1$ on $[-1,1]$
# - $T_n$ has definite parity (even or odd)
# - They are optimal (minimax) polynomial approximators on bounded intervals
#
# These properties make Chebyshev polynomials **ideal building blocks** for QSVT.

# %%
def T(n, x):
    """Chebyshev polynomial of the first kind T_n(x)."""
    return np.cos(n * np.arccos(x))

# Plot a few Chebyshev polynomials
x = np.linspace(-1, 1, 400)

plt.figure(figsize=(7, 5))
for n in [1, 2, 3, 4, 5]:
    plt.plot(x, T(n, x), label=f"T_{n}(x)")
plt.axhline(1, color="grey", linestyle="--", linewidth=0.5)
plt.axhline(-1, color="grey", linestyle="--", linewidth=0.5)
plt.title("Chebyshev Polynomials on [-1, 1]")
plt.xlabel("x")
plt.ylabel("T_n(x)")
plt.grid(True)
plt.legend()
plt.show()

# %% [markdown]
# ## 3. Approximating inverse-like behaviour
#
# The inverse function $g(x) = 1/x$ is:
#
# - unbounded near $x = 0$,
# - therefore *not directly admissible* for QSVT.
#
# However, if the spectrum of $A$ satisfies:
#
# $$ |\lambda_i| \ge a > 0, $$
#
# then we only need to approximate $1/x$ on:
#
# $$ [-1, -a] \cup [a, 1]. $$
#
# On such restricted domains, **low-degree Chebyshev polynomials can provide
# effective inverse-like behaviour**.

# %% [markdown]
# ## 4. Example: odd Chebyshev approximation to 1/x
#
# As a simple demonstration, we compare:
#
# - the true inverse $1/x$,
# - the degree-3 Chebyshev polynomial $T_3(x)$,
#
# on the interval $[a, 1]$ with $a = 0.3$.
#
# Note: we are *not* matching magnitudes exactly—only **relative scaling**
# matters after normalization.

# %%
def inv(x):
    return 1 / x

x_pos = np.linspace(0.3, 1.0, 400)

plt.figure(figsize=(7, 5))
plt.plot(x_pos, inv(x_pos), "--", label="1/x")
plt.plot(x_pos, T(3, x_pos), label="T₃(x)")
plt.title("Inverse vs Chebyshev Polynomial (Positive Spectrum)")
plt.xlabel("x")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.show()

# %% [markdown]
# While $T_3(x)$ does *not* approximate $1/x$ in absolute value,
# it often preserves the **ordering and relative ratios** of eigenvalues.
#
# This is exactly the behaviour exploited in the QSVT linear-solver examples:
#
# - Apply $P(A)$ instead of $A^{-1}$,
# - Normalize the output state,
# - Recover the correct solution *direction*.

# %% [markdown]
# ## 5. Error vs polynomial degree
#
# Increasing the polynomial degree improves approximation quality,
# at the cost of deeper QSVT circuits.
#
# As a simple diagnostic, we compute the maximum pointwise error
# between $1/x$ and $T_n(x)$ on $[a,1]$ for odd $n$.

# %%
a = 0.3
x_test = np.linspace(a, 1.0, 1000)

degrees = [1, 3, 5, 7, 9]
errors = []

for n in degrees:
    approx = T(n, x_test)
    err = np.max(np.abs(inv(x_test) - approx))
    errors.append(err)

plt.figure(figsize=(6, 4))
plt.plot(degrees, errors, "o-")
plt.xlabel("Polynomial degree")
plt.ylabel("Max |1/x - P(x)| on [a,1]")
plt.title("Approximation Error vs Polynomial Degree")
plt.grid(True)
plt.show()

# %% [markdown]
# This illustrates a central tradeoff in QSVT:
#
# - Higher-degree polynomials give better approximations,
# - but require deeper circuits and more controlled operations.
#
# In practical algorithms, the polynomial degree scales with:
#
# - the desired precision,
# - the spectral gap $a$,
# - and the condition number of the problem.

# %% [markdown]
# ## 6. Connection to QSVT linear solvers
#
# We can now reinterpret the earlier linear-solver notebooks:
#
# - **Exact inverse examples** used polynomials that *coincided* with $1/x$
#   on the spectrum.
# - **Approximate examples** used bounded Chebyshev polynomials that
#   reproduced inverse-like behaviour *up to normalization*.
#
# This notebook explains *why* those choices worked.

# %% [markdown]
# ## Summary
#
# - QSVT implements bounded polynomial transformations of spectra.
# - Polynomial design is therefore central to QSVT-based algorithms.
# - Chebyshev polynomials provide a natural, optimal basis.
# - Inverse-like behaviour can be achieved without implementing $1/x$ exactly.
# - Polynomial degree controls the accuracy–depth tradeoff.
#
# With this, the conceptual pipeline of this repository is complete:
#
# > QSP → QSVT → polynomial design → filtering → linear-system behaviour