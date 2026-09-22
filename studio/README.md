# QSVT Experiment Studio

From the repository root:

```bash
python -m pip install -e ".[plot]"
python -m studio --port 8765
```

Open forwarded port 8765 in Codespaces, or `http://127.0.0.1:8765` locally.
No Node build or AI services are required. The studio is a repository-only
client of the existing Python package and is not shipped in its wheel.

Each workflow opens with a recommended preset. Choose quick demonstrations,
accuracy studies, or expected failures, or select **Use package defaults**.
Advanced controls expose tested phase solvers and reconstruction tolerances;
results distinguish returned phases from validated reconstruction. Saved 1.0
and 1.1 configurations remain reusable through an explicit upgrade to schema
1.2. Queued and executing runs can be cancelled; applicable reports gain phase,
operator-response, and resource plots.

The Studio remembers per-workflow drafts and history filters in this browser.
Run cards separate execution, reconstruction, and scientific acceptance;
linked report sections retain complete JSON and reproducibility details.
History is filtered on the server and displayed in pages of 24 experiments.

See [the complete guide](../docs/qsvt/studio.md) for scientific scope,
architecture, storage, reproducibility, comparison, acceptance interpretation,
and catalogue extension instructions.
