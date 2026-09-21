# Releasing

Use this checklist before publishing a package release.

1. Update the version in `pyproject.toml`.
2. Add a top changelog entry with the release date and user-visible changes.
3. Update release markers in `README.md`, `RESULTS.md`, and
   `docs/qsvt/results.md`. Preserve artifact refresh versions unless the
   corresponding snapshots are regenerated.
4. Check notebook normalization and refresh deliberate notebook result changes:

   ```bash
   scripts/update_notebook_results.sh
   .venv/bin/python scripts/normalize_notebooks.py --check
   ```

   The update helper's final normalization pass removes transient kernel
   metadata while preserving freshly executed outputs.
5. Run the full local preflight from a clean checkout:

   ```bash
   .venv/bin/python scripts/release_check.py --no-build-isolation --include-notebooks
   ```

   The default output is a concise progress and artifact summary. Add
   `--verbose` when complete successful command logs are useful; failed checks
   always replay their captured diagnostics.

6. Confirm the wheel smoke step installs the built wheel in a fresh virtual
   environment, runs `pip check`, imports `qsvt`, checks `py.typed`, validates
   API-status labels, runs `qsvt --help`, and executes minimal scalar and report
   schema CLI commands.
7. Confirm branch coverage remains above the configured project floor and the
   standalone README quick examples execute under the test suite.
8. Confirm generated outputs remain untracked except for deliberate research
   artifacts under `results/`.
9. Publish only after the tagged commit's Ordered Actions run passes lint,
   tests, dependency compatibility, integration, Studio browser checks, docs,
   notebook execution,
   package build, metadata validation, and the exact-wheel smoke test.

PR release checks and Ordered Actions both call the reusable package workflow;
keep build, metadata, wheel-smoke, and artifact-upload behavior centralized
there so the published artifact is validated by the same path.

Studio browser checks use the same reusable workflow on pull requests and in
Ordered Actions. See the checks section in `docs/qsvt/studio.md` to run the
browser check locally against disposable storage. Browser tooling is separate
from package runtime and local release-preflight dependencies.

Live provider or paid hardware execution is not part of the default release
gate. Keep those checks behind explicit opt-in workflows and document the
provider, backend, shot limits, cost assumptions, and credentials used outside
portable reports.
