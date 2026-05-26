#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-qsvt}"

python scripts/extract_notebook_plots.py --preset all --execute --write-docs

git diff --stat -- \
  notebooks \
  docs/qsvt/tutorial_results.md \
  docs/qsvt/real_example_results.md \
  docs/qsvt/benchmark_results.md \
  results/plots \
  results/tables
