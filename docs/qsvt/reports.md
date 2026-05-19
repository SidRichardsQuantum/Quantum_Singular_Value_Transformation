# Diagnostics Reports

The `qsvt.reports` module provides small helpers for reusing diagnostics
reports outside an interactive Python session.

Design and template diagnostics functions return dictionaries that include
NumPy arrays. Those arrays are convenient for plotting and further numerical
work, but they are not directly JSON serializable. The reports module converts
those payloads into plain Python containers, writes JSON files, loads saved
reports, and plots the sampled approximation curves.

## Python workflow

```python
from qsvt.design import design_sign_diagnostics
from qsvt.reports import (
    plot_approximation_report,
    save_report,
    save_report_plot,
)

report = design_sign_diagnostics(gamma=0.2, degree=13)

save_report(report, "sign-report.json")
save_report_plot(report, "sign-report.png")

fig, axes = plot_approximation_report(report)
```

## CLI workflow

```bash
qsvt design-report --kind sign --gamma 0.2 --degree 13 \
  --output sign-report.json \
  --plot sign-report.png

qsvt design-workflow --kind sign --gamma 0.2 --degree 13 \
  --output sign-workflow.json

qsvt design-sweep --kind sign --degrees "5,9,13,17" --gamma 0.2 \
  --no-synthesis \
  --output sign-degree-sweep.json

qsvt template-report --kind inverse --degree 7 --mu 0.3 \
  --output inverse-report.json
```

The report commands print full JSON to standard output by default.
`design-workflow` combines coefficients, diagnostics, and QSVT compatibility
metadata in one JSON report. `design-sweep` runs the same design workflow over
multiple degrees and writes a compact manifest suitable for release summaries
and result tables. When `--output` or `--plot` is supplied, the CLI writes the
full artifact to disk and switches stdout to a compact write summary; add
`--print-report` if you also want the full JSON payload on stdout.

Related rendered result pages:

- [Results summary](results.md)
- [Tutorial notebook outputs](tutorial_results.md)
- [Real-example notebook outputs](real_example_results.md)
- [QSVT transform reports](qsvt_reports.md)

## Report fields

| field | meaning |
| --- | --- |
| `mode` | CLI wrapper label for the selected report command |
| `kind` | Chosen report family, such as `sign` or `inverse` |
| `builder` | Underlying polynomial builder function |
| `fit_domain` | Interval used to sample target-vs-polynomial error |
| `bounded_domain` | Interval used to sample QSVT-style boundedness |
| `max_error` | Maximum sampled absolute fit error |
| `rms_error` | Root-mean-square sampled fit error |
| `max_abs_value` | Maximum sampled absolute polynomial value |
| `bounded_margin` | `bound - max_abs_value` |
| `is_bounded` | Whether the sampled polynomial values stayed within `bound` |
| `coeffs` | Generated polynomial coefficients |
| `xs`, `target_values`, `polynomial_values`, `errors` | Fit-domain samples |
| `bounded_xs`, `bounded_polynomial_values` | Bounded-domain samples |

## Public helpers

- `report_to_jsonable(report)`
- `save_report(report, path)`
- `load_report(path)`
- `plot_approximation_report(report, ax=None)`
- `save_report_plot(report, path)`
