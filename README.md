# Quantum Singular Value Transformation (QSVT)

This repository provides both:

• a **notebook-first introduction** to Quantum Singular Value Transformation (QSVT)  
• a lightweight **PyPI package** implementing reusable utilities for QSVT experiments in PennyLane  

The focus is on small, explicit examples showing how **bounded polynomials**
can be applied to the **singular values / eigenvalues** of an operator via
(simulator-friendly) block-encodings.

The included Python package:

```bash
pip install qsvt-pennylane
```

provides helpers for:

- Chebyshev and polynomial utilities
- bounded polynomial approximation
- small Hermitian matrix construction
- spectral matrix-function reference calculations
- explicit QSVT matrix extraction
- comparison between classical and QSVT polynomial transforms

The notebooks use PennyLane’s high-level `qml.qsvt` interface
(with `block_encoding="embedding"`) to emphasise mathematical structure,
spectral intuition, and polynomial design rather than circuit engineering.

For theoretical background see [THEORY.md](THEORY.md).

---

## Repository structure

```
Quantum_Singular_Value_Transformation
├── LICENSE
├── README.md
├── THEORY.md
├── pyproject.toml
├── src/qsvt
│   ├── polynomials.py
│   ├── approximation.py
│   ├── matrices.py
│   ├── spectral.py
│   ├── qsvt.py
│   └── __main__.py
├── tests
│   └── test_qsvt_smoke.py
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

- Explains *how- QSVT polynomials are designed.
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

## Python package

The repository now includes a reusable Python package:

```bash
pip install qsvt-pennylane
```

### Example usage

```python
from qsvt.qsvt import qsvt_scalar_output
from qsvt.polynomials import chebyshev_t3

# apply polynomial x^2 via QSVT
qsvt_scalar_output(0.5, [0,0,1], encoding_wires=[0])
```

```python
from qsvt.qsvt import qsvt_diagonal_transform

vals = qsvt_diagonal_transform(
    [1.0, 0.7, 0.3, 0.1],
    [0,0,1],
    encoding_wires=[0,1,2],
)
```

### CLI

After installation:

```bash
qsvt scalar --x 0.5 --poly "0,0,1"

qsvt diag \
  --values "1.0,0.7,0.3,0.1" \
  --poly "0,0,1" \
  --wires 3

qsvt cheb --degree 3 --x 0.5
```

### Modules

| module               | purpose                                    |
| -------------------- | ------------------------------------------ |
| `qsvt.polynomials`   | Chebyshev utilities and polynomial helpers |
| `qsvt.approximation` | bounded polynomial approximation           |
| `qsvt.matrices`      | small Hermitian test matrices              |
| `qsvt.spectral`      | classical spectral matrix functions        |
| `qsvt.qsvt`          | PennyLane QSVT wrappers                    |

---

## Installation

### Install from PyPI

```
pip install qsvt-pennylane
```

### Install from source

```bash
git clone https://github.com/SidRichardsQuantum/Quantum_Singular_Value_Transformation.git
cd Quantum_Singular_Value_Transformation

pip install -e .
```

### Dependencies

- Python $\geq$ 3.10
- PennyLane $\geq$ 0.36
- NumPy $\geq$ 1.23
- Matplotlib $\geq$ 3.7

---

## Quickstart (notebooks)

```bash
git clone https://github.com/SidRichardsQuantum/Quantum_Singular_Value_Transformation.git
cd Quantum_Singular_Value_Transformation

python -m venv .venv
source .venv/bin/activate

pip install -e .

jupyter lab
```

Open the notebooks in order:
1. scalar QSVT intuition
2. singular value filtering
3. QSP polynomials
4. linear solvers
5. polynomial approximation
6. matrix functions
7. spectral projectors

---

## Author

**Sid Richards**

LinkedIn:
[https://www.linkedin.com/in/sid-richards-21374b30b/](https://www.linkedin.com/in/sid-richards-21374b30b/)

GitHub:
[https://github.com/SidRichardsQuantum](https://github.com/SidRichardsQuantum)

---

## License

MIT License — see [LICENSE](LICENSE)