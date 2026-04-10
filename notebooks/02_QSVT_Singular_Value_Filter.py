# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # QSVT as a Threshold Filter
#
# In this notebook we demonstrate how **Quantum Singular Value Transformation (QSVT)** can be used to implement *filters* on singular values.
#
# Instead of a discontinuous step, we use an even, bounded, degree-2 polynomial $x^2$ which produces smooth singular-value suppression.
#

# %%
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Toy Matrix
#
# We define a diagonal matrix whose singular values are:
#
# $$
# \sigma = [1.0,\, 0.7,\, 0.3,\, 0.1].
# $$
#
# Our goal is to apply a QSVT polynomial that keeps the large singular
# values (≥ 0.5) and suppresses the small ones (< 0.5).
#

# %%
singular_values = np.array([1.0, 0.7, 0.3, 0.1])
A = np.diag(singular_values)
print("Matrix A:\n", A)


# %% [markdown]
# ## Polynomial Filter (QSVT-Compatible)
#
# QSVT requires that the polynomial satisfy:
#
# $$ |f(x)| \le 1 \quad \text{for all } x \in [-1, 1]. $$
#
# This means arbitrary step-like polynomials are *not* allowed unless they
# are carefully normalised.
#
# For a simple and QSVT-valid singular-value filter, we use:
#
# $$ f(x) = x^2. $$
#
# This acts as a **soft threshold filter**:
#
# - large singular values stay near 1,
# - small singular values shrink rapidly.
#
# Even low-degree even polynomials give smooth filtering behaviour.
#


# %%
def classical_filter(x):
    return x**2


filter_poly = [0, 0, 1]

x = np.linspace(0, 1, 200)
y = classical_filter(x)

plt.figure(figsize=(6, 4))
plt.plot(x, y, label="f(x) = x²", color="orange")
plt.axvline(0.5, linestyle="--", label="Threshold = 0.5")
plt.ylim(-0.1, 1.1)
plt.title("Soft Singular Value Filter: f(x) = x²")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.show()

# %% [markdown]
# ## Applying QSVT to Matrix A
#
# We now apply:
#
# $$ f(\sigma_i) \approx \text{threshold}(\sigma_i - 0.5). $$
#
# PennyLane can automatically block-encode a diagonal matrix using the
# `"embedding"` method. Then `qml.qsvt()` applies the polynomial
# transformation to the singular values.
#

# %%
wire_order = [0, 1, 2]  # 3 wires is enough for 4 singular values

U_A = qml.matrix(qml.qsvt, wire_order=wire_order)(
    A, filter_poly, encoding_wires=wire_order, block_encoding="embedding"
)

# Extract top-left block diagonal = transformed singular values
# The block-encoding places f(A) in the top-left block.
top_block = U_A[:4, :4]
transformed = np.real(np.diagonal(top_block))

print("Transformed singular values:\n", transformed)

# %% [markdown]
# We compare:
#
# - original singular values σᵢ,
# - transformed values f(σᵢ) = σᵢ².
#
# This is a *soft filter*:
#
# - 1.0   → 1.0
# - 0.7   → 0.49
# - 0.3   → 0.09
# - 0.1   → 0.01
#
# This is not a sharp threshold, but it is a valid QSVT polynomial and
# demonstrates singular-value suppression.
#

# %%
# Plot the results
plt.figure(figsize=(6, 4))
plt.plot(singular_values, transformed, "o", label="QSVT Filtered Values")
plt.plot(
    singular_values,
    classical_filter(singular_values),
    "x",
    label="Classical Polynomial",
    color="orange",
)
plt.axhline(0.0, color="grey", linestyle="--")
plt.axhline(1.0, color="grey", linestyle="--")
plt.xlabel("Original Singular Values")
plt.ylabel("Transformed Values")
plt.axvline(0.5, color="red", linestyle="--", label="Threshold at 0.5")
plt.title("Singular Value Filtering Using QSVT (f(x) ≈ threshold)")
plt.legend()
plt.grid(True)
plt.show()
