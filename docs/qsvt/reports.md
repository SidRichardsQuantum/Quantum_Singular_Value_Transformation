# Diagnostics Reports

The `qsvt.reports` module provides small helpers for reusing diagnostics
reports outside an interactive Python session.

Design and template diagnostics functions return dictionaries that include
NumPy arrays. Those arrays are convenient for plotting and further numerical
work, but they are not directly JSON serializable. The reports module converts
those payloads into plain Python containers, writes JSON files, loads saved
reports, checks versioned report schemas, and plots the sampled approximation
curves.

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

## Schema compatibility

Versioned workflow and execution reports can be loaded with an explicit schema
compatibility check:

```python
from qsvt.reports import load_report_with_schema

report, compatibility = load_report_with_schema(
    "problem-workflow.json",
    expected_schema_name="qsvt-problem-workflow",
)
assert compatibility.supported
```

Use `report_schema_manifest(paths)` to audit several saved reports at once,
including unsupported versions, invalid JSON files, and missing required
top-level fields. Known-schema reports can also include extra top-level fields;
those are reported as `unknown_fields` but do not make the report unsupported.

The same audit path is available from the CLI:

```bash
qsvt report-schema-manifest \
  --path problem-workflow.json \
  --path hardware-report.json \
  --csv-output schema-manifest.csv \
  --fail-on-unsupported \
  --output schema-manifest.json
```

Use `--fail-on-unsupported` in CI when unsupported versions, malformed JSON, or
missing required fields should fail the command. Use `--csv-output` when you
want compact rows that are easy to diff in release artifacts.

## Schema policy

- Additive optional top-level fields are allowed for a supported schema version;
  they are surfaced as `unknown_fields` until documented or added to the known
  field registry.
- Removing or renaming required fields requires a schema-version bump.
- Changing the meaning, type, or units of an existing field requires a
  schema-version bump unless the old field remains available with its original
  semantics.
- Unsupported schema names or versions must fail with an intentional
  migration-required message rather than with incidental key errors.
- Historical fixtures for supported versions should remain loadable, even when
  newer reports add optional fields.

## Versioned schemas

| schema | versions | required top-level fields |
| --- | --- | --- |
| `qsvt-algorithm-workflow` | `1.0`, `1.1` | `schema_name`, `schema_version`, `mode`, `implementation_kind`, `truth_contract` |
| `qsvt-problem-workflow` | `1.0` | `schema_name`, `schema_version`, `target`, `truth_contract` |
| `block-encoding-qsvt-execution` | `1.0` | `schema_name`, `schema_version`, `mode`, `implementation_kind`, `truth_contract`, `resource_summary` |
| `hardware-qsvt-execution` | `1.0` | `schema_name`, `schema_version`, `mode`, `implementation_kind`, `truth_contract`, `resource_summary` |
| `hardware-qsvt-circuit` | `1.0` | `schema_name`, `schema_version`, `mode`, `implementation_kind`, `truth_contract`, `logical_resource_summary`, `decomposed_resource_summary` |

For algorithm schema `1.1`, `truth_contract` additionally requires
`execution_tier`, `truth_status`, QNode/device/circuit execution flags,
`resource_completeness`, and `polynomial_evidence`. Merely relabeling a `1.0`
payload as `1.1` therefore fails validation; use the migration helper.

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
- `migrate_algorithm_workflow_report(report, target_version="1.1")`
- `supported_report_schemas()`
- `report_schema_manifest(paths)`
- `write_report_schema_manifest_csv(rows, path)`
- `validate_report_schema(report, require_schema=False)`
- `load_report_with_schema(path, require_schema=True, expected_schema_name=None,
  expected_schema_version=None)`
- `plot_approximation_report(report, ax=None)`
- `save_report_plot(report, path)`

Versioned machine-readable reports currently include
`qsvt-algorithm-workflow` at schema versions `1.0` and `1.1`, plus
`qsvt-problem-workflow`, `block-encoding-qsvt-execution`,
`hardware-qsvt-execution`, and `hardware-qsvt-circuit` at schema version `1.0`.
Algorithm schema `1.1` guarantees artifact-derived execution-tier and
polynomial truth evidence. The migration helper upgrades `1.0` reports when
they retain polynomial coefficients and fails explicitly otherwise.
Unsupported schema names or versions return an intentional
migration-required compatibility message before callers rely on stale report
fields. Unknown fields are reported as warnings in compatibility payloads and
CSV manifests, but they do not make an otherwise supported report fail.
