# qsvt-pennylane

```{raw} html
<div class="doc-home">
  <header class="site-header" aria-label="Project navigation">
    <a class="brand" href="#" aria-label="qsvt-pennylane home">
      <span class="brand-mark">SR</span>
      <span>qsvt-pennylane</span>
    </a>
    <nav class="nav-links" aria-label="Primary navigation">
      <a href="#documentation">Docs</a>
      <a href="#package">Package</a>
      <a href="#project-links">Project</a>
      <a href="search.html">Search</a>
      <a href="https://sidrichardsquantum.github.io/">Portfolio</a>
    </nav>
  </header>

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
            <path d="M74 96h372M74 174h372M74 252h372" />
            <path d="M122 96v156M398 96v156" />
            <circle cx="122" cy="96" r="11" />
            <circle cx="122" cy="252" r="11" />
            <circle cx="398" cy="96" r="11" />
            <circle cx="398" cy="252" r="11" />
            <rect x="170" y="68" width="70" height="56" rx="8" />
            <rect x="280" y="68" width="70" height="56" rx="8" />
            <rect x="162" y="146" width="86" height="56" rx="8" />
            <rect x="272" y="146" width="86" height="56" rx="8" />
            <rect x="206" y="224" width="108" height="56" rx="8" />
          </g>
          <g class="visual-labels">
            <text x="190" y="101">&Pi;&phi;0</text>
            <text x="306" y="101">U_A</text>
            <text x="184" y="180">&Pi;&phi;1</text>
            <text x="298" y="180">U_A&dagger;</text>
            <text x="227" y="258">&Pi;&phi;2</text>
            <text x="78" y="322">QSVT phase sequence</text>
            <text x="332" y="322">p(A)</text>
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

  <section id="documentation" class="section">
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
          <h3>Theory Notes</h3>
          <p>Read the QSVT background on block encodings, bounded polynomials, QSP, projectors, and inverse-like transforms.</p>
        </div>
        <div class="tags">
          <span>Theory</span>
          <span>QSVT</span>
          <span>QSP</span>
        </div>
        <div class="card-links">
          <a href="qsvt/theory.html">Open theory</a>
        </div>
      </article>

      <article class="project-card">
        <div>
          <h3>Usage Guide</h3>
          <p>Follow practical workflows for choosing polynomials, applying matrix transforms, and using the command line interface.</p>
        </div>
        <div class="tags">
          <span>Usage</span>
          <span>CLI</span>
          <span>Workflow</span>
        </div>
        <div class="card-links">
          <a href="qsvt/usage.html">Open usage guide</a>
        </div>
      </article>

      <article class="project-card">
        <div>
          <h3>Notebooks</h3>
          <p>Browse the introductory and real physics notebook sequence, including PDEs, Hamiltonians, spectral density, and transport examples.</p>
        </div>
        <div class="tags">
          <span>Notebooks</span>
          <span>Examples</span>
          <span>Physics</span>
        </div>
        <div class="card-links">
          <a href="qsvt/notebooks.html">Open notebooks</a>
        </div>
      </article>

      <article class="project-card">
        <div>
          <h3>Results</h3>
          <p>Track notebook output status, reproducible report commands, and conventions for future plots, tables, and JSON artefacts.</p>
        </div>
        <div class="tags">
          <span>Results</span>
          <span>Plots</span>
          <span>Reports</span>
        </div>
        <div class="card-links">
          <a href="qsvt/results.html">Open results</a>
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
          <h3>Roadmap</h3>
          <p>See the direction for general package workflows, thin client notebooks, artifact hygiene, and packaging.</p>
        </div>
        <div class="tags">
          <span>Roadmap</span>
          <span>Users</span>
          <span>Package</span>
        </div>
        <div class="card-links">
          <a href="qsvt/roadmap.html">Open roadmap</a>
        </div>
      </article>

      <article class="project-card">
        <div>
          <h3>Algorithm Notes</h3>
          <p>Read concise workflow-level theory for linear systems, filtering, simulation, resolvents, spectral density, and Gibbs weighting.</p>
        </div>
        <div class="tags">
          <span>Algorithms</span>
          <span>Diagnostics</span>
          <span>Theory</span>
        </div>
        <div class="card-links">
          <a href="qsvt/algorithms.html">Open algorithm notes</a>
        </div>
      </article>

      <article class="project-card">
        <div>
          <h3>Linear Systems</h3>
          <p>Understand QSVT inverse-polynomial workflows, solver comparisons, and finite HHL circuit execution.</p>
        </div>
        <div class="tags">
          <span>Linear Systems</span>
          <span>HHL</span>
          <span>QSVT</span>
        </div>
        <div class="card-links">
          <a href="qsvt/linear_systems.html">Open linear systems</a>
        </div>
      </article>

      <article class="project-card">
        <div>
          <h3>Spectral Filters</h3>
          <p>Review the theory behind ground-state filters, interval projectors, sign thresholds, and spectral density windows.</p>
        </div>
        <div class="tags">
          <span>Filters</span>
          <span>Projectors</span>
          <span>Density</span>
        </div>
        <div class="card-links">
          <a href="qsvt/spectral_filters.html">Open spectral filters</a>
        </div>
      </article>

      <article class="project-card">
        <div>
          <h3>Time and Response</h3>
          <p>Connect Hamiltonian simulation, Green's functions, imaginary-time evolution, and Gibbs weighting to polynomial matrix functions.</p>
        </div>
        <div class="tags">
          <span>Dynamics</span>
          <span>Response</span>
          <span>Thermal</span>
        </div>
        <div class="card-links">
          <a href="qsvt/time_evolution_and_response.html">Open time and response</a>
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
          <h3>Implementation Notes</h3>
          <p>Review coefficient conventions, rescaling, boundedness checks, report serialization, and public API policy.</p>
        </div>
        <div class="tags">
          <span>Implementation</span>
          <span>API</span>
          <span>Reports</span>
        </div>
        <div class="card-links">
          <a href="qsvt/implementation.html">Open implementation notes</a>
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

      <article class="project-card">
        <div>
          <h3>Changelog</h3>
          <p>Track release notes for package APIs, documentation, notebooks, physics workflows, and generated artefacts.</p>
        </div>
        <div class="tags">
          <span>Release Notes</span>
          <span>Package</span>
          <span>Docs</span>
        </div>
        <div class="card-links">
          <a href="qsvt/changelog.html">Open changelog</a>
        </div>
      </article>
    </div>
  </section>

  <section id="package" class="section">
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

  <section id="project-links" class="section split-section">
    <div class="section-heading">
      <p class="eyebrow">Project links</p>
      <h2>Documentation with the rest of the project nearby</h2>
    </div>
    <div class="about-copy">
      <p>The repo also includes a sequence of notebooks that introduce QSVT concepts step by step, from scalar transforms to linear-solver style experiments and polynomial design workflows.</p>
      <p><a href="qsvt/notebooks.html">Browse notebooks</a></p>
      <p><a href="qsvt/theory.html">Read theory notes</a></p>
      <p><a href="qsvt/usage.html">Open CLI and usage guide</a></p>
    </div>
  </section>

  <footer class="site-footer">
    <span>qsvt-pennylane documentation</span>
    <a href="https://sidrichardsquantum.github.io/">Back to Sid Richards portfolio</a>
  </footer>
</div>
```

```{toctree}
:hidden:
:maxdepth: 1
:titlesonly:
:caption: Start Here

Overview <qsvt/index>
Usage Guide <qsvt/usage>
Theory <qsvt/theory>
Notebooks <qsvt/notebooks>
Roadmap <qsvt/roadmap>
Release Checklist <qsvt/releasing>
```

```{toctree}
:hidden:
:maxdepth: 1
:titlesonly:
:caption: Guides

Polynomial Design <qsvt/design>
Phase Synthesis <qsvt/synthesis>
Algorithm Notes <qsvt/algorithms>
Linear Systems <qsvt/linear_systems>
Spectral Filters <qsvt/spectral_filters>
Time Evolution and Response <qsvt/time_evolution_and_response>
Block Encodings <qsvt/block_encoding>
QSVT Compatibility <qsvt/compatibility>
Physics Workflows <qsvt/physics>
Classical Benchmarks <qsvt/benchmarks>
Classical Baseline Details <qsvt/classical_baselines>
QSVT Resource Model <qsvt/qsvt_resource_model>
Polynomial Templates <qsvt/templates>
Implementation Notes <qsvt/implementation>
API Reference <qsvt/api_reference>
```

```{toctree}
:hidden:
:maxdepth: 1
:titlesonly:
:caption: Results

Results Summary <qsvt/results>
Tutorial Notebook Outputs <qsvt/tutorial_results>
Real-Example Notebook Outputs <qsvt/real_example_results>
Benchmark Notebook Outputs <qsvt/benchmark_results>
Diagnostics Reports <qsvt/reports>
QSVT Transform Reports <qsvt/qsvt_reports>
```

```{toctree}
:hidden:
:maxdepth: 1
:titlesonly:
:caption: Reference

Changelog <qsvt/changelog>
```
