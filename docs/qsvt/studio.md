# QSVT Experiment Studio

The optional, repository-only studio provides a local experiment composer,
scientific run gallery, detailed viewer, reuse, favorites, and comparison.
The existing Python package computes every numerical result. There is no
AI model, generated scientific content, or frontend numerical implementation.

## Launch in a fresh GitHub Codespace

From the repository root:

```bash
python -m pip install -e ".[plot]"
python -m studio --port 8765
```

Open port **8765** in the Codespaces **Ports** panel and choose **Open in
Browser**. Keep the forwarded port private. Alternatively open
`http://127.0.0.1:8765` when running locally. The server binds to loopback and
is intended for a single local user; it does not provide multi-user accounts.
The launcher uses a Linux/macOS file lock, suitable for Codespaces.

No JavaScript build, Node installation, API key, or extra Python web framework
is needed. `pip install qsvt-pennylane` remains independent of the studio.
The studio directory is outside the distribution's `src/` package discovery.

Choose **Sign · cookbook** for a quick design run, or **Poisson · four-point
finite QNode** / **Pauli band filter · finite QNode** for executable examples.
The **Hamiltonian · six-site coherent QNode** preset reproduces the
published real-time evolution example. Their sources are
`examples/design_apply_report.py`, `examples/poisson_qsvt.py`,
`examples/spectral_filter_qsvt.py`, and `examples/hamiltonian_simulation.py`;
the composer displays the source path.
Each workflow opens with a named recommended starting configuration. **Use
package defaults** restores the public API defaults, with catalogue choices
for required arguments. It does not select the recommended preset. The stable
package signatures and defaults are unchanged.

## Starting configurations and advanced controls

The catalogue includes 16 presets in three groups:

- **Quick demonstration**: inexpensive design-only studies and the original
  small-degree Poisson/Pauli cookbook circuits. The Poisson cookbook fixes
  degree 5 and allows 40% solution error; the Pauli cookbook allows 16%
  hard-projector error. These are educational starting points.
- **Accuracy study**: Poisson degree search at 20% solution error, Pauli degree
  search at 2% hard-projector error, coherent Hamiltonian evolution at tolerance
  `1e-6`, and design-phase reconstruction at tolerance `1e-6`. Accurate phases
  implement the fitted polynomial; they do not reduce its target-fitting error.
- **Expected failure**: an asymmetric interval that cannot be synthesized as
  one parity sequence, and under-resolved degree-2 Hamiltonian evolution. These
  runs return reports while failing their stated scientific checks.

All exposed settings are pinned in `studio/presets.py`, including the target,
encoding, execution flag, fitting grids, solver choice, and reconstruction
tolerance. Presets have stable IDs and revisions; their source and purpose
appear beside the composer. Changes to presets must update the revision and
their numerical regression expectations. Saved configurations contain the
resolved settings, so later preset edits do not change their reuse behavior.

**Advanced settings** contains the fitting grid, reconstruction controls, and
phase solver. Design and Hamiltonian workflows offer `root-finding` and
`iterative`; Poisson and Pauli workflows additionally offer the package's
ordered root-finding → iterative fallback. The requested method is saved in
the configuration; the actual selected solver is recorded in the synthesis
report (per component for Hamiltonian evolution). Fallback does not guarantee
success: all attempted methods can miss the reconstruction tolerance.

Sign/inverse degree controls advance by two through odd degrees, and the soft
filter advances through even degrees. Interval and flagship search bounds
remain flexible because their construction can involve different parity
components. Fixed simulator and Hamiltonian-encoding choices are displayed as
information rather than dropdowns. Field help separates target approximation
tolerances from sampled phase reconstruction and application acceptance.

## Scientific scope

| Catalogue entry | Package entrypoint | Evidence |
| --- | --- | --- |
| Sign approximation | `qsvt.stable.design_workflow("sign", ...)` | Smooth fitting target, bounded polynomial, sampled error, compatibility |
| Normalized reciprocal | `design_workflow("inverse", ...)` | Bounded inverse-like target; not an unscaled `1/x` solver |
| Soft spectral filter | `design_workflow("filter", ...)` | Smooth even filter design |
| Interval projector design | `design_workflow("interval_projector", ...)` | Smooth band design; not a hard-projector guarantee |
| Poisson inversion | `qsvt.stable.poisson_qsvt_workflow(...)` | Default sine-source Dirichlet problem, dense/CG references, optional finite QNode, acceptance |
| Pauli spectral filtering | `qsvt.stable.spectral_filter_qsvt_workflow(...)` | Published two-qubit Hamiltonian and uniform state, hard-projector reference, optional finite QNode, acceptance |
| Hamiltonian simulation | `qsvt.stable.hamiltonian_simulation_workflow(...)` | Six-site tight-binding evolution, cosine/sine components, dense exponential reference, optional coherent QNode, acceptance |

Designs use the package's normalized domain and target conventions. They do
not execute QNodes. `attempt_synthesis` requests the package compatibility
attempt and a separate public `DesignWorkflowResult.synthesize()` report,
using the chosen solver for both. The adapter uses the package's compatibility
API for non-default solvers without changing the frozen design facade.
`PhaseSynthesisResult.quality_report()` distinguishes returned finite phases
from reconstruction that meets the requested tolerance. Resources from
`DesignWorkflowResult.resource_report()` are polynomial proxies.

The Poisson and Pauli flagship workflows synthesize phases even with
`execute=False`.
`execute=True` requests the package's analytic statevector QNode path on
`default.qubit`; it does not mean physical hardware. Finite shots, arbitrary
matrices, custom sources, external devices, and external phase-solver plugins are not
exposed in this MVP. Pauli filtering fixes the problem to
`0.4 Z₀ + 0.3 Z₁ + 0.2 X₀` with uniform input. Poisson exposes 2, 4, or 8
interior points and its supported access models.

Hamiltonian simulation uses `tight_binding_chain(6)` with the initial state
localized at site 1 (zero-based), exactly as in the cookbook. It exposes time,
degree, fitting grid size, package acceptance and phase-reconstruction tolerances,
and `execute_qsvt`. Required time/degree defaults come from the published
example (1.4 and 12); other defaults come from the package signature. The
preset uses 401 fitting points. The local service supports time in [-5, 5]
and degree 1–24, with dense embedding and analytic `default.qubit` execution.
FABLE is not offered for this preset because the package uses fixed alpha=1
and the normalized chain does not satisfy the FABLE normalization condition.

With `execute_qsvt=False`, only the polynomial core and dense reference run;
phases, circuit resources, and success probability are unavailable. With
execution enabled, the package coherently combines cosine/sine sequences and
reports their separate phases, reconstruction errors, LCU normalization,
postselection probabilities, and circuit resource ledger. The viewer retains
both `component_error_ledger` and `circuit_resource_ledger`, along with the
complete `qsvt_execution` report. Requested execution is not a claim that the
circuit or acceptance checks succeeded.

The Hamiltonian plots show real and imaginary parts of stored amplitudes.
The package's recovered QNode output includes its LCU rescaling; it is labelled
separately from polynomial evolution and the dense exponential reference.
The studio does not normalize these arrays or recompute evolution. Low-degree
runs can complete while failing polynomial accuracy or finite-QSVT acceptance;
those package checks and thresholds remain visible.

Studio limits (degree, grid size, queue length) bound local workloads. They
are UI service limits, not scientific thresholds. Package acceptance
thresholds are never adjusted by the studio. Parameter validation is strict:
unknown settings, non-finite numbers, wrong types, invalid intervals, and
inconsistent degree ranges are rejected before work is queued.

## Architecture and request contract

```text
Catalogue → resolved request → one serial worker → public qsvt API
                                                 ↓
Browser gallery ← run metadata + plots ← complete package report
```

- `studio/catalogue.py`: workflow capabilities, field groups, types, bounds,
  API defaults, recommendations, and comparison compatibility fields.
- `studio/presets.py`: explicit settings, purposes, sources, and revisions for
  tested starting configurations, including expected scientific failures.
- `studio/adapter.py`: the sole request-to-package mapping. No numerical engine.
- `studio/records.py`: atomic JSON writes, history, favorites, reuse,
  comparison validation, and extraction of explicitly named report fields.
- `studio/plots.py`: the package's `plot_approximation_report` for design
  previews; otherwise Matplotlib renders stored solution/state arrays and
  degree-search measurements. Comparison overlays read stored arrays and do
  not rerun scientific code.
- `studio/server.py`: standard-library HTTP server, one background worker,
  bounded queue, lifecycle, exports, and environment provenance.
- `studio/static/`: responsive dark workbench. Form controls are generated
  from `/api/catalogue`. The browser contains only presentation and UI state.

A canonical request has schema version `1.1`:

```json
{
  "schema_version": "1.1",
  "workflow": "sign",
  "settings": {
    "degree": 13,
    "gamma": 0.25,
    "num_points": 401,
    "bounded_num_points": 801,
    "attempt_synthesis": false,
    "angle_solver": "root-finding",
    "phase_reconstruction_tolerance": 0.000001,
    "reconstruction_num_points": 257
  }
}
```

`validate_request` resolves every exposed default before saving the request.
Fields use package argument names; design reconstruction tolerance is passed
to `quality_report(tolerance=...)`, and design solver/sample settings are
passed to the result's synthesis method. Unexposed arguments retain their
package defaults. The run records package versions, Python version, Git
commit, dirty-tree status, and a hash of the scientific module sources.
Reproduction across different package versions is not promised; use the saved
provenance and the same checkout/environment. Exports can be imported through
**Import configuration**. Unsupported schema versions are rejected explicitly.

Schema `1.0` configurations remain importable and reusable. Validation upgrades
them to `1.1` by retaining every saved setting and making the old implicit
solver choices explicit: root-finding for designs/Hamiltonian evolution, and
root-finding → iterative for Poisson/Pauli. Design reconstruction retains 257
samples and adds a separate quality-assessment tolerance of `1e-6`. Newly
exposed fields require schema `1.1`. The UI announces an upgrade; old request
and report files are never rewritten. Historical reports without a quality
assessment retain their original evidence rather than receiving a new verdict.

## Storage, lifecycle, and errors

Default storage is `.qsvt-studio/runs/`, ignored by Git. Override it with:

```bash
python -m studio --data-dir /tmp/my-qsvt-experiments --port 8765
```

Each run has a UUID directory:

```text
<run-id>/
  request.json    # authoritative, fully resolved exposed configuration
  report.json     # complete scientific package report, when available
  record.json     # versioned lifecycle, favorite, provenance, derived summary
  preview.png     # derived scientific visualization, when available
```

There is no database. History scans records on disk. `record.json` holds UI
metadata and a derived metric summary; comparison always reads `report.json`
as its numerical source. Request/report files are never changed by favorites
or reuse. No automatic history cleanup is performed, so favorites cannot age
out. Back up the entire data directory to preserve history.

Serialization uses `qsvt.stable.report_to_jsonable`, including the package's
`{"real": ..., "imag": ...}` encoding. Keys are sorted for deterministic
serialization. Non-finite IEEE floats, if emitted by a failed diagnostic,
are explicitly represented as `{"__nonfinite__": "inf"}` (or `-inf`/`nan`),
not invalid JSON or fabricated finite values. Timestamps and measured runtimes
naturally vary between runs; deterministic serialization is not a promise of
bit-identical runtime reports.

Lifecycle states are **configured → validating → executing → saving_artifacts
→ completed**, or **failed** with the actual exception type and message.
Configured runs wait in the serial queue. The `executing` stage includes the
entire public package call because the package does not expose progress
callbacks. No percentages or fictional internal stages are shown.

A report that fails scientific acceptance still completes as an experiment;
the acceptance failure remains visible. Failed executions retain their
requests and can be reused. Preview failures retain successful scientific
reports and show a warning. On restart, unfinished runs are marked failed
with an interruption message; they are not silently resumed. One server can
own a data directory. Ctrl-C waits for queued scientific work to finish.

Unreadable or invalid saved records and configurations are skipped during
history loading and startup recovery. The server logs the affected run path
once while it remains unreadable; other experiments remain accessible. The
original files are preserved for repair or backup, and repaired runs reappear
on the next history refresh. Scientific report files are loaded only when
requested, so a damaged report does not prevent browsing other runs.

The displayed **package wall time** measures the full adapter call, including
classical references, phase synthesis, simulation and package diagnostics.
It is not quantum runtime. Package-native synthesis timings remain in the
scientific report. Studio elapsed time additionally includes artifact work.

## Gallery, viewer, reuse, and comparison

The gallery supports text search, workflow, run state, requested execution,
encoding, favorite filtering, and chronological sorting. Preview images are
actual scientific plots. Cards show package error metrics with their report
field names, configuration, wall time, and lifecycle state.

**View** exposes resolved configuration, numerical metrics, phase values,
compatibility, degree search, execution outputs, resource model assumptions,
acceptance checks, truth contracts, complete report, provenance, and lifecycle.
Unavailable report fields are omitted. Artifact links and configuration/result
JSON downloads are available in the viewer.

**Reuse** restores all controls from `request.json`, never display strings or
rounded metrics. It creates no new run until **Run experiment** is pressed.
Changing a degree or tolerance produces a new record and leaves the original
report untouched.

Select **2–6 completed runs**, then **Compare selected**. Compatibility requires
an identical workflow and target/problem settings. Sign/inverse must have the
same gamma; smooth filters must have the same cutoff/sharpness or interval;
Poisson must have the same grid size and length; the fixed Pauli problem must
have the same physical interval; the fixed Hamiltonian chain must have the
same evolution time. Degree, tolerance, access model, synthesis,
and execution choices may vary. Comparing different physical problems is
rejected with a clear error. There is no overall score.

The comparison includes stored polynomial/error overlays, Poisson solution
and degree-search overlays, Pauli state and degree-search overlays, or
Hamiltonian real/imaginary amplitude overlays, plus a numerical table of
request settings and available stored metrics. Missing
values appear as em dashes. Logical resource estimates retain their model
names; no conversion to physical gate cost is made. Comparison JSON is
exportable.

## Interpreting acceptance and truth metadata

The package owns the acceptance and truth contracts. They are retained without
rewriting thresholds or deriving new claims. In particular:

- `accepted_for_stated_scope` evaluates the declared scope's required checks.
- `full_qsvt_acceptance` means the package's **finite** QSVT acceptance boundary;
  it is not a scalable, hardware, fault-tolerant, or quantum-advantage claim.
- `execution.succeeded` and execution truth fields describe actual finite
  execution evidence; the composer records only what was requested.
- `synthesis.reconstruction_max_error` is distinct from approximation error.
- `synthesis_quality` distinguishes solver failure, unavailable reconstruction,
  failed reconstruction, and passed sampled reconstruction. Hamiltonian runs
  retain one such assessment per synthesized component under
  `component_synthesis_quality`. Passing this assessment does not imply
  application accuracy or finite-circuit acceptance.
- Design errors are sampled against the package fitting target; they are not
  formal approximation guarantees. Signed error plots follow the package
  reporting convention.
- Poisson and Pauli resources are encoding-aware logical estimates with
  assumptions and omitted costs. Hamiltonian resources are the package
  coherent-circuit ledger when execution was requested. Design resources are
  polynomial proxies.
- Full statevectors and solution vectors are simulator validation outputs.

Read the expanded acceptance checks, truth contracts, and resource assumptions
before interpreting a result. See [flagship workflows](flagship_workflows.md),
[implementation notes](implementation.md), and [resource model](qsvt_resource_model.md).

## Add another workflow

1. Select a mature public package workflow with a tested result/report API.
2. Add catalogue metadata: description, capabilities, settings with signature
   defaults, allowed values, limits, diagnostics, and target compatibility keys.
3. Add a branch in the single adapter calling that API; retain its report.
4. Add a published preset with its source, if available.
5. Add plot rendering only for existing result arrays, and metric paths only
   for real report fields. Do not fit or recompute science in the UI.
6. Test direct API equivalence, request validation, defaults, report/truth
   preservation, reuse, and compatibility. Document the scope and omissions.

The frontend will render the new controls automatically. New plot/report
families may need viewer support. Schema changes require an explicit version
and migration policy; do not reinterpret old requests silently.

## Checks

```bash
python -m pip install -e ".[test,lint,type,docs]"
MPLCONFIGDIR=/tmp/qsvt-matplotlib python -m pytest tests/test_studio.py
MPLCONFIGDIR=/tmp/qsvt-matplotlib python -m pytest
python -m ruff check .
python -m black --check .
python -m mypy src/qsvt studio
python -m sphinx -W -b html docs docs/_build/html
```

Optional browser checks require Node for syntax validation only:
`node --check studio/static/app.js`. There is no frontend bundle to build.

For the optional end-to-end browser smoke check, use a separate terminal to
launch a server with an empty temporary data directory, then run:

```bash
python -m pip install playwright
python -m playwright install --with-deps chromium
python -m studio.check_browser --url http://127.0.0.1:8765
```

The check creates seven real runs and verifies the viewer, export, reuse,
comparison, incompatible-target rejection, favorites after reload, search,
Hamiltonian execution filtering and reuse, recommended versus package defaults,
legacy import, solver selection, quality/failure panels, and mobile layout. It saves
screenshots under `/tmp`. Browser tooling is not a studio runtime dependency.

The reusable `.github/workflows/studio.yml` runs this browser check on pull
requests and in the ordered main-branch release gate. It starts a real server
with disposable storage, waits for readiness, and retains screenshots and the
server log as CI artifacts. The regular Python test matrix includes
`tests/test_studio.py`; lint and local release checks also type-check `studio`.
Both source and wheel distributions exclude the Studio and its local records.

## Design provenance and current limits

The catalogue/composer/history/viewer separation is inspired by
[OpenHiggsfield](https://github.com/wide-trace/open-higgsfield), inspected at
commit `b16a0efe4d7e2707b56f8ccb02387fd2a9d2eddf`. Its declarative settings,
structured reuse records, lifecycle cards, and persistent favorites informed
the architecture. No branding, generative workflows, or source assets are
copied. See its `src/generation/catalog` and `src/openhiggsfield` directories.

This is a single-user local MVP with serial execution, a bounded queue, and no
job cancellation, live internal stage instrumentation, or multi-user access.
It does not import legacy repository reports lacking canonical requests.
There is no automatic artifact deletion. Finite-shot sampling, additional
problem presets, spectrum-response plots, phase plots,
and resource sweep charts are follow-up work; all relevant existing raw
report data remain available in the viewer and JSON exports.
