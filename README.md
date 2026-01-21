# Quantum Singular Value Transformation (QSVT)

This repository is a notebook-first, PennyLane-based introduction to **Quantum Singular Value Transformation (QSVT)** and its close relationship to **Quantum Signal Processing (QSP)**. The focus is on small, explicit examples that show—end to end—how **bounded polynomials** can be applied to the **singular values / eigenvalues** of an operator via a (simulator-friendly) block-encoding.

The notebooks use PennyLane’s high-level `qml.qsvt` interface (with `block_encoding="embedding"`) to keep the emphasis on the mathematical mechanism, spectral intuition, and polynomial design rather than circuit engineering.

---

## What is QSVT

QSVT is a framework for implementing **polynomial transformations** of an operator’s singular values using a quantum circuit, provided the operator is available via a **block-encoding**.

Concretely: given a block-encoding of a matrix $A$, QSVT enables a circuit whose action corresponds (in the encoded subspace) to applying a polynomial $f$ so that singular values $\sigma_i$ are mapped to $f(\sigma_i)$, subject to standard QSVT admissibility constraints such as boundedness on $[-1,1]$ and parity structure.

From this single mechanism follow:
- filtering and suppression,
- matrix functions,
- spectral projectors,
- inverse-like transformations,
- and linear-system–style behaviour.

For a more extensive theoretical discussion, see [THEORY.md](THEORY.md).

---

## Repository structure

```
Quantum_Singular_Value_Transformation
├── LICENSE
├── README.md
├── THEORY.md
└── notebooks
    ├── 01_QSVT_Scalar_and_Diagonal_Matrix.ipynb
    ├── 02_QSVT_Singular_Value_Filter.ipynb
    ├── 03_QSP_Polynomial_Demo.ipynb
    ├── 04_QSVT_Linear_Solver_2x2.ipynb
    ├── 04_QSVT_Linear_Solver_4x4.ipynb
    ├── 05_QSVT_Linear_Solver_Approximate.ipynb
    ├── 06_QSVT_Polynomial_Design_and_Approximation.ipynb
    ├── 07_QSVT_Matrix_Functions_Powers_and_Roots.ipynb
    └── 08_QSVT_Sign_Function_and_Projectors.ipynb
```

---

### 01 — QSVT on a scalar and a diagonal matrix  
**`01_QSVT_Scalar_and_Diagonal_Matrix.ipynb`**

- Minimal “single singular value” intuition: treat a scalar $a \in [-1,1]$ as a block-encoded operator.
- Applies the polynomial $f(x)=x^2$ and verifies the QSVT output matches the classical curve.
- Extends the same idea to diagonal matrices, showing that QSVT transforms eigenvalues independently.

---

### 02 — QSVT as a (soft) singular value filter  
**`02_QSVT_Singular_Value_Filter.ipynb`**

- Introduces filtering and suppression using bounded even polynomials.
- Shows how small singular values are attenuated more strongly than large ones.
- Highlights the key admissibility constraint $|f(x)| \le 1$ on $[-1,1]$.

---

### 03 — QSP demo (two perspectives)  
**`03_QSP_Polynomial_Demo.ipynb`**

- **Part A:** QSP via QSVT-on-a-scalar (QSVT reduces to QSP in the scalar case).
- **Part B:** manual single-qubit construction of a Chebyshev polynomial:
  $$
  T_3(x) = 4x^3 - 3x.
  $$
- Builds intuition for why Chebyshev polynomials are natural in QSP/QSVT.

---

### 04 — QSVT linear solver (exact inverse on a special spectrum)  
**`04_QSVT_Linear_Solver_2x2.ipynb`** and **`04_QSVT_Linear_Solver_4x4.ipynb`**

- Demonstrates QSVT-based inversion for involutory matrices with eigenvalues $\pm1$.
- Uses the polynomial $P(x)=x$ to exactly reproduce the inverse.
- These examples are intentionally spectrally trivial but mathematically exact.

---

### 05 — QSVT linear solver (approximate / inverse-like polynomial)  
**`05_QSVT_Linear_Solver_Approximate.ipynb`**

- A non-trivial example where $A \ne A^{-1}$.
- Uses the bounded odd Chebyshev polynomial $T_3(x)$ to reproduce inverse-like behaviour.
- Shows that matching **relative eigencomponent scaling** is sufficient after normalization.

---

### 06 — Polynomial design and approximation  
**`06_QSVT_Polynomial_Design_and_Approximation.ipynb`**

- Explains *how* QSVT polynomials are designed.
- Introduces Chebyshev approximation as the natural basis for bounded polynomials.
- Shows how polynomial degree controls accuracy and circuit depth.
- Connects approximation quality to linear-system behaviour.

---

### 07 — QSVT as matrix functions (powers and roots)  
**`07_QSVT_Matrix_Functions_Powers_and_Roots.ipynb`**

- Presents QSVT as **functional calculus for matrices**.
- Demonstrates matrix powers, square roots, and fractional powers.
- Shows how eigenvalues are transformed while eigenvectors are preserved.
- Unifies inversion, filtering, and simulation under a single spectral viewpoint.

---

### 08 — Sign function and spectral projectors  
**`08_QSVT_Sign_Function_and_Projectors.ipynb`**

- Approximates the sign function using bounded odd polynomials.
- Builds approximate spectral projectors:
  $$
  \Pi_\pm = \frac{I \pm \mathrm{sgn}(A)}{2}.
  $$
- Demonstrates subspace selection and basis-independence.
- Bridges QSVT toward ground-state filtering and Hamiltonian algorithms.

---

## Scope and limitations (important)

This repository is **educational by design**.

- All examples are small, explicit, and simulator-friendly.
- Block encodings are constructed using PennyLane’s `"embedding"` mode.
- Linear solvers are shown as **polynomial mechanisms**, not full algorithms.
- Topics such as state preparation, success probability, amplitude amplification,
  condition number scaling, and fault-tolerant resource estimates are intentionally excluded.

The goal is conceptual understanding, not algorithmic optimization.

---

## Requirements


- Python 3.10+ (recommended)
- PennyLane
- NumPy
- Matplotlib
- Jupyter (or JupyterLab)

---

## Quickstart

1. Clone the repository:

```bash
git clone https://github.com/SidRichardsQuantum/Quantum_Singular_Value_Transformation.git
cd Quantum_Singular_Value_Transformation
```

2.	Create and activate an environment:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows PowerShell
```

3.	Install dependencies:

```bash
pip install pennylane numpy matplotlib jupyter
```

4.	Launch Jupyter:

```bash
jupyter lab
```

⸻

Author

Sid Richards (SidRichardsQuantum)
LinkedIn: https://www.linkedin.com/in/sid-richards-21374b30b/

This project is licensed under the MIT License – see the LICENSE￼ file for details.
