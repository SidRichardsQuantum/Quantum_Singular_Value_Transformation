# qsvt-pennylane

```{raw} html
<div class="doc-home">
  <section class="hero section">
    <div class="hero-copy">
      <p class="eyebrow">Quantum singular value transformation and PennyLane tooling</p>
      <h1>qsvt-pennylane</h1>
      <p class="hero-text">
        Lightweight utilities and examples for bounded polynomials, QSVT-compatible
        design patterns, spectral matrix functions, and PennyLane-based experiments.
      </p>
      <div class="hero-badges" aria-label="Package badges">
        <img src="https://img.shields.io/pypi/v/qsvt-pennylane?style=flat-square" alt="PyPI version">
        <img src="https://img.shields.io/pypi/pyversions/qsvt-pennylane?style=flat-square" alt="Supported Python versions">
        <img src="https://img.shields.io/github/license/SidRichardsQuantum/Quantum_Singular_Value_Transformation?style=flat-square" alt="Repository license">
        <img src="https://img.shields.io/github/actions/workflow/status/SidRichardsQuantum/Quantum_Singular_Value_Transformation/tests.yml?label=tests&amp;style=flat-square" alt="Test workflow status">
      </div>
      <div class="hero-actions" aria-label="Primary links">
        <a class="button primary" href="qsvt/index.html">Documentation overview</a>
        <a class="button" href="qsvt/api_reference.html">API reference</a>
        <a class="button" href="https://github.com/SidRichardsQuantum/Quantum_Singular_Value_Transformation" target="_blank" rel="noopener noreferrer">GitHub</a>
        <a class="button" href="https://pypi.org/project/qsvt-pennylane/" target="_blank" rel="noopener noreferrer">PyPI</a>
      </div>
    </div>

    <div class="hero-side">
      <div class="hero-visual" aria-hidden="true">
        <svg viewBox="0 0 520 360" role="presentation" focusable="false">
          <defs>
            <pattern id="hero-grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M40 0H0V40" />
            </pattern>
          </defs>
          <rect class="visual-bg" width="520" height="360" rx="8" />
          <rect class="visual-grid" width="520" height="360" rx="8" fill="url(#hero-grid)" />
          <g class="visual-circuit">
            <path d="M72 86h344M72 146h344M72 206h344M72 266h344" />
            <path d="M150 86v180M258 146v120M366 86v180" />
            <circle cx="150" cy="86" r="13" />
            <circle cx="150" cy="266" r="13" />
            <circle cx="258" cy="146" r="13" />
            <circle cx="258" cy="266" r="13" />
            <circle cx="366" cy="86" r="13" />
            <circle cx="366" cy="266" r="13" />
            <path d="M103 68v36M85 86h36M211 128l36 36M247 128l-36 36M339 188l54 36M393 188l-54 36" />
            <rect x="200" y="66" width="52" height="40" rx="8" />
            <rect x="398" y="126" width="52" height="40" rx="8" />
          </g>
          <g class="visual-labels">
            <text x="60" y="322">QSVT</text>
            <text x="148" y="322">QSP</text>
            <text x="224" y="322">POLY</text>
            <text x="314" y="322">DIAG</text>
            <text x="404" y="322">QML</text>
          </g>
        </svg>
      </div>

      <aside class="focus-panel" aria-label="Project focus">
        <h2>Focus Areas</h2>
        <ul>
          <li>Bounded polynomial design for QSVT and QSP workflows</li>
          <li>Classical spectral transforms and matrix-function experiments</li>
          <li>PennyLane wrappers for explicit scalar and matrix checks</li>
          <li>Diagnostics, templates, and report generation for notebook reuse</li>
        </ul>
      </aside>
    </div>
  </section>

  <section class="section">
    <div class="section-heading">
      <p class="eyebrow">Documentation</p>
      <h2>Read the package the same way it is built</h2>
      <p>
        The documentation is organised around small, explicit helpers: API surface,
        polynomial design, reusable template families, and diagnostics workflows.
      </p>
    </div>

    <div class="project-grid">
      <article class="project-card">
        <div>
          <h3>Package Overview</h3>
          <p>Start with the project scope, module map, CLI examples, and links into the theory notes.</p>
        </div>
        <div class="tags">
          <span>Overview</span>
          <span>CLI</span>
          <span>Theory</span>
        </div>
        <div class="card-links">
          <a href="qsvt/index.html">Open overview</a>
        </div>
      </article>

      <article class="project-card">
        <div>
          <h3>API Reference</h3>
          <p>Read the public Python API grouped by module, with examples for the core helpers.</p>
        </div>
        <div class="tags">
          <span>API</span>
          <span>Python</span>
          <span>Examples</span>
        </div>
        <div class="card-links">
          <a href="qsvt/api_reference.html">Browse API</a>
        </div>
      </article>

      <article class="project-card">
        <div>
          <h3>Polynomial Design</h3>
          <p>See the higher-level bounded polynomial builders for inverse, sign, projector, and filter workflows.</p>
        </div>
        <div class="tags">
          <span>Design</span>
          <span>QSVT</span>
          <span>Filters</span>
        </div>
        <div class="card-links">
          <a href="qsvt/design.html">Open design guide</a>
        </div>
      </article>

      <article class="project-card">
        <div>
          <h3>Template Families</h3>
          <p>Use ready-made bounded templates for inverse-like, sign-like, square-root, and exponential transforms.</p>
        </div>
        <div class="tags">
          <span>Templates</span>
          <span>QSP</span>
          <span>Reusable</span>
        </div>
        <div class="card-links">
          <a href="qsvt/templates.html">Open templates</a>
        </div>
      </article>

      <article class="project-card">
        <div>
          <h3>Physics Workflows</h3>
          <p>Build Hamiltonians, PDE operators, spectral rescalings, and matrix-function polynomials for concrete physics examples.</p>
        </div>
        <div class="tags">
          <span>Physics</span>
          <span>PDEs</span>
          <span>Hamiltonians</span>
        </div>
        <div class="card-links">
          <a href="qsvt/physics.html">Open physics guide</a>
        </div>
      </article>

      <article class="project-card">
        <div>
          <h3>Diagnostics Reports</h3>
          <p>Reuse JSON-safe diagnostics, plotting helpers, and CLI report outputs outside notebooks.</p>
        </div>
        <div class="tags">
          <span>Reports</span>
          <span>Plots</span>
          <span>JSON</span>
        </div>
        <div class="card-links">
          <a href="qsvt/reports.html">Open reports guide</a>
        </div>
      </article>

      <article class="project-card">
        <div>
          <h3>QSVT Reports</h3>
          <p>Compare classical transforms with QSVT outputs and inspect report-oriented experiment helpers.</p>
        </div>
        <div class="tags">
          <span>Validation</span>
          <span>Transforms</span>
          <span>PennyLane</span>
        </div>
        <div class="card-links">
          <a href="qsvt/qsvt_reports.html">Open QSVT reports</a>
        </div>
      </article>
    </div>
  </section>

  <section class="section">
    <div class="section-heading">
      <p class="eyebrow">Package</p>
      <h2>Install and link out</h2>
      <p>
        Use the published package directly or jump out to the repository notebooks,
        theory notes, and usage guide.
      </p>
    </div>

    <div class="package-list">
      <article class="package-row">
        <div>
          <h3>qsvt-pennylane</h3>
          <div class="badges" aria-label="qsvt-pennylane package badges">
            <img src="https://img.shields.io/pypi/v/qsvt-pennylane?label=PyPI" alt="qsvt-pennylane PyPI version">
            <img src="https://img.shields.io/pypi/pyversions/qsvt-pennylane" alt="qsvt-pennylane supported Python versions">
            <img src="https://img.shields.io/pypi/l/qsvt-pennylane" alt="qsvt-pennylane license">
          </div>
        </div>
        <code>pip install qsvt-pennylane</code>
        <a href="https://pypi.org/project/qsvt-pennylane/" target="_blank" rel="noopener noreferrer">PyPI</a>
      </article>
    </div>
  </section>

  <section class="section split-section">
    <div class="section-heading">
      <p class="eyebrow">Project links</p>
      <h2>Documentation with the rest of the project nearby</h2>
    </div>
    <div class="about-copy">
      <p>The repo also includes a sequence of notebooks that introduce QSVT concepts step by step, from scalar transforms to linear-solver style experiments and polynomial design workflows.</p>
      <p><a href="https://github.com/SidRichardsQuantum/Quantum_Singular_Value_Transformation/tree/main/notebooks" target="_blank" rel="noopener noreferrer">Browse notebooks</a></p>
      <p><a href="https://github.com/SidRichardsQuantum/Quantum_Singular_Value_Transformation/blob/main/THEORY.md" target="_blank" rel="noopener noreferrer">Read theory notes</a></p>
      <p><a href="https://github.com/SidRichardsQuantum/Quantum_Singular_Value_Transformation/blob/main/USAGE.md" target="_blank" rel="noopener noreferrer">Open CLI and usage guide</a></p>
    </div>
  </section>
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
qsvt/physics
qsvt/templates
qsvt/reports
qsvt/qsvt_reports
```
