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
python examples/block_encoded_workflow.py \
  --output /tmp/qsvt-block-encoded-workflow.json
python examples/circuit_execution.py --output /tmp/qsvt-circuit-execution.json
python examples/block_encoding_execution.py \
  --output /tmp/qsvt-block-encoding-execution.json
python examples/rectangular_execution.py \
  --output /tmp/qsvt-rectangular-execution.json
python examples/compatibility_report.py --output /tmp/qsvt-compatibility.json
python examples/benchmark_summary.py \
  --output /tmp/qsvt-benchmark-summary.json \
  --rows-output /tmp/qsvt-benchmark-summary.csv
```

Each script writes a JSON report with machine-readable diagnostics. The
linear-system and benchmark-summary examples also write compact CSV summary
tables.

The examples cover design/report basics, linear systems, spectral filtering,
finite block encodings, PennyLane circuit execution, specification-based
block-encoding execution, rectangular singular-value execution, compatibility
checks, and benchmark table export.
