# Theory: Quantum Singular Value Transformation (QSVT)

This document provides the theoretical background underlying the notebooks in this repository. It is intentionally concise and focused on the concepts that are *directly exercised* in the examples, rather than a full survey of the QSVT literature.

The emphasis is on:
- how **polynomials act on singular values and eigenvalues**,
- why **boundedness and parity constraints** appear,
- how **QSP emerges as a special case**,
- how **matrix functions are implemented spectrally**,
- how **projectors arise from sign-function approximations**, and
- how **linear-system–style transformations** follow from inverse-like polynomials.

---

## 1. Block encodings

QSVT operates on matrices that are provided via a **block encoding**.

### Definition (block encoding)

A unitary $U$ acting on an enlarged Hilbert space is said to be a block encoding of a matrix $A$ if

$$
U =
\begin{pmatrix}
A / \alpha & * \\
* & *
\end{pmatrix},
$$

for some normalization constant $\alpha \ge 1$, where the top-left block acts on a designated “logical” subspace.

Operationally, this means that when the ancilla qubits are prepared in $|0\rangle$, the action of $U$ on the logical register reproduces the action of $A$ (up to normalization).

### In this repository

- PennyLane’s `block_encoding="embedding"` is used.
- This automatically constructs a valid block encoding for small, explicitly specified matrices.
- The notebooks frequently **extract the top-left block** of the resulting unitary to verify that it matches the expected transformed operator.

This is a simulator-friendly convenience; the theoretical statements of QSVT apply to *any* valid block encoding.

---

## 2. Singular values, eigenvalues, and normalization

QSVT acts on **singular values** $\sigma_i \in [0,1]$. For Hermitian matrices, singular values coincide with absolute eigenvalues.

To apply QSVT:
- the spectrum must lie in $[-1,1]$ (or $[0,1]$ depending on parity),
- or be rescaled so that this holds.

All examples in this repository are chosen so that eigenvalues already lie in $[-1,1]$, avoiding additional normalization steps.

---

## 3. Quantum Singular Value Transformation (QSVT)

### Informal statement

Given:
- a block encoding of a matrix $A$,
- and a real polynomial $f$ satisfying certain constraints,

QSVT constructs a quantum circuit whose effective action applies

$$
\sigma_i \;\longmapsto\; f(\sigma_i)
$$

to each singular value $\sigma_i$ of $A$.

In other words, QSVT implements **matrix functions via polynomial transformations**.

---

### Admissibility constraints

The polynomial $f(x)$ must satisfy:

1. **Boundedness**
   $$
   |f(x)| \le 1 \quad \text{for all } x \in [-1,1].
   $$

2. **Parity structure**
   - Even polynomials act on singular values.
   - Odd polynomials act on signed eigenvalues (Hermitian case).

These constraints ensure that the transformed operator remains compatible with a unitary embedding.

All polynomials used in this repository satisfy these conditions.

---

## 4. Scalar case and Quantum Signal Processing (QSP)

When the “matrix” $A$ is just a **scalar** $x \in [-1,1]$, QSVT reduces to **Quantum Signal Processing (QSP)**.

In QSP:
- a single “signal” unitary encodes $x$ in its eigenphases,
- fixed phase rotations are interleaved with this signal unitary,
- the resulting circuit implements a polynomial function of $x$.

Thus:

$$
\text{QSVT on a scalar} \;\equiv\; \text{QSP}.
$$

This equivalence is demonstrated in Notebook 03 using both PennyLane’s `qml.qsvt` interface and a manual single-qubit construction.

---

## 5. Chebyshev polynomials and boundedness

Chebyshev polynomials of the first kind,

$$
T_n(x) = \cos(n \arccos x),
$$

play a central role in QSP/QSVT because:

- they are bounded on $[-1,1]$,
- they have definite parity,
- they are optimal (minimax) polynomial approximators on bounded intervals.

For example:

$$
T_3(x) = 4x^3 - 3x
$$

is odd, bounded, and directly admissible for QSVT.

This explains why Chebyshev polynomials appear repeatedly in filtering, inversion, projector construction, and Hamiltonian simulation.

---

## 6. Polynomial design and approximation

QSVT does not require exact target functions — only **polynomial approximations**.

Given a target function $g(x)$, one constructs a polynomial $P(x)$ such that:

$$
P(x) \approx g(x) \quad \text{on the relevant spectral interval}.
$$

The polynomial degree controls:
- approximation error,
- circuit depth,
- and ultimately algorithmic cost.

Notebook 06 demonstrates this process explicitly and shows how Chebyshev approximation provides a natural basis for QSVT-compatible polynomials.

---

## 7. QSVT as matrix functional calculus

For a diagonalizable matrix:

$$
A = U \Lambda U^\dagger,
$$

any function $f$ acts spectrally as:

$$
f(A) = U f(\Lambda) U^\dagger.
$$

QSVT implements this *functional calculus* using polynomial approximations.

Notebook 07 demonstrates this viewpoint using:
- matrix powers,
- square roots,
- and fractional powers,

showing explicitly that:
- eigenvectors are preserved,
- eigenvalues are transformed,
- and different functions correspond to different spectral maps.

This perspective unifies filtering, inversion, and simulation under a single mechanism.

---

## 8. Sign function and spectral projectors

The **sign function**:

$$
\mathrm{sgn}(x) =
\begin{cases}
+1 & x > 0 \\
-1 & x < 0
\end{cases}
$$

is central to spectral filtering and subspace selection.

Although discontinuous, it can be approximated by bounded odd polynomials away from $x=0$.

From the sign function, we construct projectors:

$$
\Pi_\pm = \frac{I \pm \mathrm{sgn}(A)}{2}.
$$

Approximating $\mathrm{sgn}(A)$ via QSVT yields **approximate spectral projectors** that:

- suppress unwanted eigencomponents,
- isolate subspaces,
- and are basis-independent.

This mechanism is demonstrated in Notebook 08 and forms the foundation of:
- ground-state filtering,
- gap amplification,
- and Hamiltonian algorithms.

---

## 9. Linear systems via inverse-like polynomials

Solving a linear system

$$
A x = b
$$

formally requires applying $A^{-1}$.

Since $1/x$ is unbounded, QSVT instead implements a polynomial $P(x)$ such that:

$$
P(\lambda_i) \propto \frac{1}{\lambda_i}
$$

on the spectrum of interest.

After normalization, only the **relative scaling of eigencomponents** matters.

Notebooks 04 and 05 demonstrate:
- exact inversion on involutory spectra,
- approximate inversion via Chebyshev polynomials.

These examples isolate the polynomial mechanism without introducing full algorithmic machinery.

---

## Summary

QSVT provides a unifying framework in which:

- quantum circuits implement bounded polynomials,
- polynomials act spectrally on matrices,
- QSP appears as the scalar limit,
- Chebyshev polynomials provide optimal approximations,
- matrix functions are implemented via functional calculus,
- projectors arise from sign-function approximations,
- and linear-system–like behaviour follows naturally.

The notebooks in this repository are concrete realizations of these ideas in their simplest possible forms.

---

## References and further reading

1. Gilyén, A., Su, Y., Low, G. H., & Wiebe, N.  
   *Quantum singular value transformation and beyond: exponential improvements for quantum matrix arithmetics.*  
   arXiv:1806.01838

2. Low, G. H., & Chuang, I. L.  
   *Optimal Hamiltonian simulation by quantum signal processing.*  
   Physical Review Letters 118, 010501 (2017).

3. Low, G. H., & Chuang, I. L.  
   *Hamiltonian simulation by qubitization.*  
   Quantum 3, 163 (2019).

4. PennyLane documentation:  
   *Introduction to Quantum Singular Value Transformation (QSVT).*  
   https://pennylane.ai/qml/demos/tutorial_intro_qsvt

---

Author: Sid Richards (SidRichardsQuantum)  
LinkedIn: https://www.linkedin.com/in/sid-richards-21374b30b/

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
