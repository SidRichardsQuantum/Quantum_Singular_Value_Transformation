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
```

Each script writes a JSON report with machine-readable diagnostics. The
linear-system example also writes a compact CSV summary table.

The examples cover design/report basics, linear systems, spectral filtering,
finite block encodings, PennyLane circuit execution, specification-based
block-encoding execution, and rectangular singular-value execution.

Compatibility and benchmark reporting already have direct CLI commands:

```bash
qsvt compatibility-report --poly "0.25,0.25"
qsvt benchmark cg-solve --matrix "4,1;1,3" --rhs "1,2"
```
