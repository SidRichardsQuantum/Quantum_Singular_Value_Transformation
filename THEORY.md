# Theory: Quantum Singular Value Transformation (QSVT)

This document provides the theoretical background underlying the notebooks in this repository. It is intentionally concise and focused on the concepts that are *directly exercised* in the examples, rather than a full survey of the QSVT literature.

The emphasis is on:
- how **polynomials act on singular values and eigenvalues**,
- why **boundedness and parity constraints** appear,
- how **QSP emerges as a special case**,
- how **matrix functions are implemented spectrally**,
- how **projectors arise from sign-function approximations**, and
- how **linear-system–style transformations** follow from inverse-like polynomials.

General QSVT/QSP background belongs in this file. Implementation-specific
notes are kept in focused documentation pages:

- `docs/qsvt/block_encoding.md` for finite dense block encodings,
- `docs/qsvt/compatibility.md` for boundedness, parity, and synthesis checks,
- `docs/qsvt/qsvt_resource_model.md` for proxy-resource interpretation,
- `docs/qsvt/algorithms.md` for workflow targets, diagnostics, and limits.

---

## Table of Contents

- [Block encodings](#1-block-encodings)

- [Singular values, eigenvalues, and normalization](#2-singular-values-eigenvalues-and-normalization)

- [Quantum Singular Value Transformation (QSVT)](#3-quantum-singular-value-transformation-qsvt)

  - [Admissibility constraints](#admissibility-constraints)
  - [Hermitian eigenvalue transforms](#hermitian-eigenvalue-transforms)

- [Scalar case and Quantum Signal Processing (QSP)](#4-scalar-case-and-quantum-signal-processing-qsp)

- [Chebyshev polynomials and boundedness](#5-chebyshev-polynomials-and-boundedness)

- [Polynomial design and approximation](#6-polynomial-design-and-approximation)

- [Practical polynomial construction](#7-practical-polynomial-construction)

  - [Template polynomials](#template-polynomials)
  - [Task-oriented polynomial design](#task-oriented-polynomial-design)
  - [Role of Chebyshev approximation](#role-of-chebyshev-approximation)

- [QSVT as matrix functional calculus](#8-qsvt-as-matrix-functional-calculus)

- [Sign function and spectral projectors](#9-sign-function-and-spectral-projectors)

- [Linear systems via inverse-like polynomials](#10-linear-systems-via-inverse-like-polynomials)

- [Access models, success probability, and readout](#11-access-models-success-probability-and-readout)

- [Summary](#summary)

- [References and further reading](#references-and-further-reading)

- [Author](#author)

- [License](#license)

---

## 1. Block encodings

QSVT operates on matrices that are provided via a **block encoding**.

### Definition (block encoding)

A unitary $U$ acting on an enlarged Hilbert space is said to be a block encoding of a matrix $A$ if

$$
U =
\begin{pmatrix}
A / \alpha & - \\
- & *
\end{pmatrix},
$$

for some normalization constant $\alpha \ge 1$, where the top-left block acts on a designated “logical” subspace.

Operationally, this means that when the ancilla qubits are prepared in $|0\rangle$, the action of $U$ on the logical register reproduces the action of $A$ (up to normalization).

The value of $\alpha$ is part of the algorithm. If the original matrix has
large norm, QSVT acts on the normalized signal $A/\alpha$, and any physical
matrix-function interpretation must translate between $A$ and $A/\alpha$.
Choosing $\alpha$ too large compresses the spectrum and can make polynomial
features harder to resolve; choosing it too small invalidates the block
encoding.

For a finite matrix with $\|A/\alpha\|_2 \le 1$, one can always write down a
dense unitary dilation. For large structured matrices, the central algorithmic
question is different: can a block encoding be implemented using efficient
oracles, sparse access, Hamiltonian simulation, or problem-specific circuits?
The QSVT theorem assumes such an encoding has been supplied.

### In this repository

- PennyLane’s `block_encoding="embedding"` is used.
- This automatically constructs a valid block encoding for small, explicitly specified matrices.
- The notebooks frequently **extract the top-left block** of the resulting unitary to verify that it matches the expected transformed operator.
- The package also includes an explicit finite dense block-encoding helper for
  small matrices.

This is a simulator-friendly convenience; the theoretical statements of QSVT apply to *any* valid block encoding.

---

## 2. Singular values, eigenvalues, and normalization

QSVT acts on **singular values** $\sigma_i \in [0,1]$. For Hermitian matrices, singular values coincide with absolute eigenvalues.

To apply QSVT:
- the spectrum must lie in $[-1,1]$ (or $[0,1]$ depending on parity),
- or be rescaled so that this holds.

Many examples in this repository explicitly rescale spectra. Rescaling is not
cosmetic: it changes the variable in which the polynomial is designed. If

$$
\widetilde A = \frac{A - \beta I}{s},
$$

then a polynomial $P(\widetilde A)$ corresponds to the physical function

$$
f(A) = P\!\left(\frac{A-\beta I}{s}\right).
$$

This is why reports track offsets, scales, and spectral bounds. A polynomial
degree that is adequate after one normalization may be inadequate after another
normalization.

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

More precisely, QSVT transforms the singular values of a block-encoded matrix
while preserving the corresponding singular-vector structure. For a singular
value decomposition

$$
A = \sum_i \sigma_i |u_i\rangle\langle v_i|,
$$

the transformed block has the form

$$
P(A)_{\mathrm{QSVT}} \sim
\sum_i P(\sigma_i) |u_i\rangle\langle v_i|,
$$

up to the precise parity and signal-convention details of the construction.

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

Not every polynomial used for dense spectral intuition is directly compatible
with a QSVT synthesis path. The repository distinguishes:

- dense spectral polynomial application,
- QSVT-style polynomial cores,
- direct PennyLane QSVT verification,
- and end-to-end quantum algorithms.

Compatibility reports make this distinction explicit through boundedness,
parity, and optional synthesis checks.

### Hermitian eigenvalue transforms

Many physics examples use Hermitian matrices, where users naturally think in
terms of eigenvalues rather than singular values. A Hermitian matrix has the
spectral decomposition

$$
H = \sum_i \lambda_i |\psi_i\rangle\langle\psi_i|.
$$

After rescaling so that $\lambda_i \in [-1,1]$, polynomial functional calculus
uses

$$
P(H) =
\sum_i P(\lambda_i)|\psi_i\rangle\langle\psi_i|.
$$

Odd polynomials can preserve sign information, while even polynomials depend
only on magnitude. This distinction is why sign functions, projectors, and
positive-spectrum inverse approximations are treated carefully in the notebooks.

---

## 4. Scalar case and Quantum Signal Processing (QSP)

When the “matrix” $A$ is just a **scalar** $x \in [-1,1]$, QSVT reduces to **Quantum Signal Processing (QSP)**.

In QSP:
- a single “signal” unitary encodes $x$ in its eigenphases,
- fixed phase rotations are interleaved with this signal unitary,
- the resulting circuit implements a polynomial function of $x$.

The fixed rotations are often called **QSP phases**. Finding phases for a
target polynomial is a synthesis problem: it is separate from fitting or
designing the polynomial itself. A polynomial may look reasonable as a sampled
approximation but still fail a particular synthesis implementation because of
parity, boundedness, numerical conditioning, or backend limitations.

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

## 7. Practical polynomial construction

In practice, QSVT workflows typically begin by constructing a bounded polynomial with the desired qualitative behaviour on the spectral interval.

Common examples include:

- sign-like polynomials
- inverse-like polynomials
- soft threshold filters
- square-root–type transforms
- power-law transforms

The key requirement is always boundedness:

$$
|P(x)| \le 1
\quad \text{for } x \in [-1,1].
$$

Different design approaches emphasise different trade-offs:

### Template polynomials

Certain bounded functional forms are repeatedly useful in QSVT experiments.

Examples include:

- smooth sign surrogates using $\tanh(\kappa x)$
- regularised inverse-like functions
- soft threshold filters based on $|x|$
- shifted square-root profiles

These functions can be approximated by Chebyshev polynomials to produce admissible QSVT transforms.

Such constructions prioritise:

- simplicity
- stability
- ease of interpretation

over optimal approximation guarantees.

### Task-oriented polynomial design

More structured workflows begin from a desired spectral transformation:

$$
g(x)
$$

and construct a bounded polynomial approximation

$$
P(x) \approx g(x)
$$

on a restricted interval such as

$$
[-1,-\gamma] \cup [\gamma,1].
$$

Examples include:

#### sign approximation

$$
P(x) \approx \mathrm{sgn}(x)
$$

away from zero.

#### inverse-like transforms

$$
P(x) \approx \frac{\gamma}{x}
$$

for $|x| \ge \gamma$.

This produces relative scaling behaviour similar to matrix inversion while preserving boundedness.

#### spectral filters

$$
P(x) \approx
\begin{cases}
0 & |x| < \tau \
1 & |x| > \tau
\end{cases}
$$

using smooth bounded approximations.

### Role of Chebyshev approximation

Chebyshev polynomials provide a convenient basis for constructing bounded approximations because they:

- minimise worst-case approximation error
- preserve parity structure
- remain bounded on $[-1,1]$

In practice:

1. choose a smooth bounded surrogate for the target function
2. approximate using Chebyshev polynomials
3. verify boundedness
4. apply via QSVT

Notebook 09 demonstrates these workflows explicitly.

---

## 8. QSVT as matrix functional calculus

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

## 9. Sign function and spectral projectors

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

## 10. Linear systems via inverse-like polynomials

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

## 11. Access models, success probability, and readout

QSVT is often described as applying a matrix function, but a complete algorithm
also needs to specify how the input and output are accessed.

### Input state preparation

For a vector problem, one usually wants to start from a quantum state

$$
|b\rangle = \frac{b}{\|b\|}.
$$

Preparing this state may be easy for structured data, or it may dominate the
cost for unstructured classical data. The polynomial transform alone does not
solve data loading.

### Success probability

The desired transformed vector often appears in a particular ancilla branch.
For example, after applying a block-encoded transform, the useful component may
be proportional to

$$
P(A)|b\rangle
$$

conditioned on measuring the ancilla in a success state. The norm of this
component determines the success probability. Amplitude amplification or
estimation can improve or estimate success probabilities, but those costs are
additional to the polynomial degree.

### Readout

If the final goal is a scalar expectation value, a norm, an overlap, or a
sample, readout may be efficient. If the goal is the full classical vector
$P(A)b$, tomography or repeated sampling may be expensive. This is why the
notebooks report transformed states and residuals for validation, while the
truth contracts avoid claiming full quantum output costs.

### Classical baselines

Classical dense solvers, iterative solvers, and spectral decompositions remain
important references. A QSVT polynomial with low degree is promising only after
the block encoding, state preparation, success probability, precision, and
readout model are also favorable.

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
- a complete quantum algorithm still needs an access model, success-probability
  analysis, and readout strategy.

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

## Author

**Sid Richards**

Portfolio: [https://sidrichardsquantum.github.io/](https://sidrichardsquantum.github.io/)

GitHub: [https://github.com/SidRichardsQuantum](https://github.com/SidRichardsQuantum)

---

## License

MIT License.
