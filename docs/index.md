# qsvt-pennylane

Lightweight utilities and examples for **Quantum Singular Value Transformation
(QSVT)** using PennyLane.

[![PyPI Version](https://img.shields.io/pypi/v/qsvt-pennylane?style=flat-square)](https://pypi.org/project/qsvt-pennylane/)
[![Python Versions](https://img.shields.io/pypi/pyversions/qsvt-pennylane?style=flat-square)](https://pypi.org/project/qsvt-pennylane/)
[![License](https://img.shields.io/github/license/SidRichardsQuantum/Quantum_Singular_Value_Transformation?style=flat-square)](https://github.com/SidRichardsQuantum/Quantum_Singular_Value_Transformation/blob/main/LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/SidRichardsQuantum/Quantum_Singular_Value_Transformation/tests.yml?label=tests&style=flat-square)](https://github.com/SidRichardsQuantum/Quantum_Singular_Value_Transformation/actions/workflows/tests.yml)

The package is structured around small, explicit helpers for bounded
polynomials, QSVT-compatible design patterns, spectral matrix functions, and
PennyLane-based experiments.

- GitHub: <https://github.com/SidRichardsQuantum/Quantum_Singular_Value_Transformation>
- PyPI: <https://pypi.org/project/qsvt-pennylane/>
- Notebooks: <https://github.com/SidRichardsQuantum/Quantum_Singular_Value_Transformation/tree/main/notebooks>
- Repository theory notes: <https://github.com/SidRichardsQuantum/Quantum_Singular_Value_Transformation/blob/main/THEORY.md>
- CLI and usage guide: <https://github.com/SidRichardsQuantum/Quantum_Singular_Value_Transformation/blob/main/USAGE.md>

```{raw} html
<div class="doc-card-grid">
  <article class="doc-card">
    <div>
      <h3>Package Overview</h3>
      <p>Start with the project scope, module map, CLI examples, and links into the theory notes.</p>
    </div>
    <a href="qsvt/index.html">Open overview</a>
  </article>

  <article class="doc-card">
    <div>
      <h3>API Reference</h3>
      <p>Read the public Python API grouped by module, with examples for core helpers.</p>
    </div>
    <a href="qsvt/api_reference.html">Browse API</a>
  </article>

  <article class="doc-card">
    <div>
      <h3>Polynomial Design</h3>
      <p>See the higher-level bounded-polynomial builders for inverse, sign, projector, and filter workflows.</p>
    </div>
    <a href="qsvt/design.html">Open design guide</a>
  </article>

  <article class="doc-card">
    <div>
      <h3>Template Families</h3>
      <p>Use ready-made bounded templates for inverse-like, sign-like, square-root, and exponential transforms.</p>
    </div>
    <a href="qsvt/templates.html">Open templates</a>
  </article>

  <article class="doc-card">
    <div>
      <h3>Diagnostics Reports</h3>
      <p>Reuse JSON-safe diagnostics, plotting helpers, and CLI report outputs outside notebooks.</p>
    </div>
    <a href="qsvt/reports.html">Open reports guide</a>
  </article>

  <article class="doc-card">
    <div>
      <h3>QSVT Reports</h3>
      <p>Compare classical transforms with QSVT outputs and inspect report-oriented experiment helpers.</p>
    </div>
    <a href="qsvt/qsvt_reports.html">Open QSVT reports</a>
  </article>
</div>
```

```{toctree}
:hidden:
:maxdepth: 1
:titlesonly:
:caption: Documentation

qsvt/index
qsvt/api_reference
qsvt/design
qsvt/templates
qsvt/reports
qsvt/qsvt_reports
```
