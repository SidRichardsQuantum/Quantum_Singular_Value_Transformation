# Results

The results pages collect the current numerical evidence for the package:
notebook execution, QSVT-vs-classical agreement, polynomial design diagnostics,
and committed artefacts that can be regenerated from the command line.

The repository root [`RESULTS.md`](../../RESULTS.md) remains the compact
source-of-truth ledger. This page is the portfolio-style summary for rendered
documentation.

## Result Navigation

| page | purpose |
| --- | --- |
| [Tutorial notebook outputs](tutorial_results.md) | generated plots and text outputs from `notebooks/tutorials/` |
| [Real-example notebook outputs](real_example_results.md) | generated plots and text outputs from `notebooks/real_examples/` |
| [Diagnostics reports](reports.md) | design-report JSON fields, plotting helpers, and CLI report commands |
| [QSVT transform reports](qsvt_reports.md) | QSVT-vs-classical comparison reports for diagonal and matrix inputs |

## Current Status

<div class="metric-grid">
  <div class="metric-card">
    <span class="metric-value">42</span>
    <span class="metric-label">validated notebooks</span>
  </div>
  <div class="metric-card">
    <span class="metric-value">29</span>
    <span class="metric-label">real-example workflows</span>
  </div>
  <div class="metric-card">
    <span class="metric-value">1e-12</span>
    <span class="metric-label">diagonal QSVT max-error scale</span>
  </div>
  <div class="metric-card">
    <span class="metric-value">0.1.19</span>
    <span class="metric-label">current release marker</span>
  </div>
</div>

| validation target | current result | source |
| --- | --- | --- |
| tutorial notebooks | all execute in notebook validation | `notebooks/tutorials/` |
| real-example notebooks | all execute in notebook validation | `notebooks/real_examples/` |
| fast unit/regression suite | algorithm workflows and CLI report paths covered | `tests/` |
| package artefacts | source distribution and wheel build cleanly | `pyproject.toml` |
| rendered plots | tutorial and real-example PNG artefacts committed | `results/plots/` |

## Key Outcomes

### Polynomial design improves predictably with degree

The committed sign-design sweep uses `gamma=0.2`, evaluates degrees
`5, 9, 13, 17`, and skips PennyLane synthesis so the table isolates polynomial
design diagnostics. A companion filter sweep uses `cutoff=0.4` and degrees
`6, 10, 14, 18`.

| degree | max error | RMS error | QSVT-compatible checks |
| ---: | ---: | ---: | --- |
| 5 | `5.182981115359634e-01` | `2.695148817806530e-01` | passed sampled boundedness/parity checks |
| 9 | `3.289044195543612e-01` | `1.633101449091314e-01` | passed sampled boundedness/parity checks |
| 13 | `2.002792111439544e-01` | `9.744948051610955e-02` | passed sampled boundedness/parity checks |
| 17 | `1.176310891345335e-01` | `5.570304051373436e-02` | passed sampled boundedness/parity checks |

Artefact:
[`results/reports/sign-degree-sweep.json`](../../results/reports/sign-degree-sweep.json)

### Filter design also benefits from degree tuning

The filter sweep shows that different design families can have non-monotonic
maximum error at low degree while still improving in RMS error and boundedness
margin as the polynomial space grows.

| degree | max error | RMS error | QSVT-compatible checks |
| ---: | ---: | ---: | --- |
| 6 | `1.911439443363911e-01` | `1.003684556902707e-01` | passed sampled boundedness/parity checks |
| 10 | `2.003389491881879e-01` | `8.135058832372982e-02` | passed sampled boundedness/parity checks |
| 14 | `7.397752073483832e-02` | `3.469153756628215e-02` | passed sampled boundedness/parity checks |
| 18 | `6.783434474077632e-02` | `3.391969914072474e-02` | passed sampled boundedness/parity checks |

Artefacts:

- [`results/reports/filter-degree-sweep.json`](../../results/reports/filter-degree-sweep.json)
- [`results/tables/design_sweep_summary.csv`](../../results/tables/design_sweep_summary.csv)

Regenerate:

```bash
qsvt design-sweep \
  --kind sign \
  --degrees "5,9,13,17" \
  --gamma 0.2 \
  --num-points 401 \
  --bounded-num-points 801 \
  --no-synthesis \
  --output results/reports/sign-degree-sweep.json
```

```bash
qsvt design-sweep \
  --kind filter \
  --degrees "6,10,14,18" \
  --cutoff 0.4 \
  --sharpness 12 \
  --num-points 401 \
  --bounded-num-points 801 \
  --no-synthesis \
  --output results/reports/filter-degree-sweep.json
```

### QSVT transforms agree with direct spectral references

The committed scalar/diagonal and Hermitian matrix reports compare explicit
QSVT transforms with direct classical polynomial evaluation.

| artefact | transform | max error | RMS error |
| --- | --- | ---: | ---: |
| [`qsvt-report.json`](../../results/reports/qsvt-report.json) | diagonal `x^2` transform | `9.999778782798785e-13` | `5.585577546102077e-13` |
| [`matrix-report.json`](../../results/reports/matrix-report.json) | Hermitian matrix `x^2` transform | `5.264677582772492e-13` | `3.2060825311797223e-13` |

### Algorithm workflows are regression-tested

The fast test suite covers deterministic small-matrix regressions for:

- positive-definite linear-system approximation
- Gaussian ground-state filtering
- real-time Hamiltonian simulation
- resolvent / Green's-function response
- spectral-density estimation
- thermal Gibbs weighting

These tests protect the package-level workflow APIs separately from the longer
notebook execution checks.

## Representative Figures

```{figure} ../../results/plots/sign-report.png
:alt: Sign polynomial design report plot
:width: 520px

Degree-13 sign design report showing the target, polynomial fit, and residual
structure.
```

```{figure} ../../results/plots/notebooks/11_End_to_End_Algorithm_Workflows-plot-01.png
:alt: End-to-end QSVT algorithm workflow plot
:width: 520px

End-to-end algorithm workflow diagnostics from the tutorial sequence.
```

```{figure} ../../results/plots/notebooks/13_Degree_Error_and_Boundedness_Tradeoffs-plot-01.png
:alt: Degree error tradeoff plot
:width: 520px

Degree/error tradeoff results for bounded QSVT-compatible polynomial design.
```

```{figure} ../../results/plots/real_examples/29_topological_band_projector_chern_marker-plot-01.png
:alt: Topological band projector Chern marker plot
:width: 520px

Topological band-projector example using a small Qi-Wu-Zhang lattice model.
```

Generated notebook output pages:

- [Tutorial notebook outputs](tutorial_results.md)
- [Real-example notebook outputs](real_example_results.md)

## Artefact Ledger

| artefact | workflow | notes |
| --- | --- | --- |
| [`results/reports/sign-report.json`](../../results/reports/sign-report.json) | sign design report | degree-13 sign approximation with `gamma=0.2` |
| [`results/plots/sign-report.png`](../../results/plots/sign-report.png) | sign design plot | target-vs-polynomial diagnostic plot |
| [`results/reports/sign-degree-sweep.json`](../../results/reports/sign-degree-sweep.json) | design sweep | degree/error/boundedness manifest |
| [`results/reports/filter-degree-sweep.json`](../../results/reports/filter-degree-sweep.json) | design sweep | filter degree/error/boundedness manifest |
| [`results/tables/design_sweep_summary.csv`](../../results/tables/design_sweep_summary.csv) | sweep summary | tabular summary of committed design-sweep reports |
| [`results/reports/qsvt-report.json`](../../results/reports/qsvt-report.json) | diagonal QSVT comparison | direct `x^2` reference comparison |
| [`results/reports/matrix-report.json`](../../results/reports/matrix-report.json) | Hermitian matrix QSVT comparison | spectral polynomial reference comparison |
| [`results/tables/qsvt-error-summary.csv`](../../results/tables/qsvt-error-summary.csv) | release summary table | compact index over generated JSON reports |
| [`results/tables/real_examples_plot_manifest.csv`](../../results/tables/real_examples_plot_manifest.csv) | real-example manifest | machine-readable ledger for real-example PNG outputs |

## Regeneration

Execute notebooks, extract their embedded outputs, and regenerate the rendered
result pages:

```bash
python scripts/extract_notebook_plots.py --preset all --execute --write-docs
```

Refresh the pages from already-saved notebook outputs without re-executing:

```bash
python scripts/extract_notebook_plots.py --preset all --write-docs
```

Generate the committed report examples:

```bash
qsvt design-report --kind sign --gamma 0.2 --degree 13 \
  --output results/reports/sign-report.json \
  --plot results/plots/sign-report.png
```

```bash
qsvt compare-report \
  --values "1.0,0.7,0.3,0.1" \
  --poly "0,0,1" \
  --wires 3 \
  --output results/reports/qsvt-report.json
```

```bash
qsvt matrix-report \
  --matrix "0.31351701,-0.23499807;-0.23499807,0.68648299" \
  --poly "0,0,1" \
  --output results/reports/matrix-report.json
```
