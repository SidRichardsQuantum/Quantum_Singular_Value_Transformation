# Cookbook Examples

These scripts are short package-client examples for common QSVT-style
workflows. They are intended to be copied into experiments or run from a local
checkout.

Run them from the repository root:

```bash
python examples/design_apply_report.py --output /tmp/qsvt-design-apply.json
python examples/linear_system_compare.py \
  --output /tmp/qsvt-linear-system.json \
  --rows-output /tmp/qsvt-linear-system.csv
python examples/threshold_filter.py --output /tmp/qsvt-threshold-filter.json
```

Each script writes a JSON report with machine-readable diagnostics. The
linear-system example also writes a compact CSV summary table.
