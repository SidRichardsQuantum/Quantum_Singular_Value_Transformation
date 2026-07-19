# Releasing

Use this checklist before publishing a package release.

1. Update the version in `pyproject.toml`.
2. Add a top changelog entry with the release date and user-visible changes.
3. Update release markers in `README.md`, `RESULTS.md`, and
   `docs/qsvt/results.md`.
4. Run the full local preflight from a clean checkout:

   ```bash
   .venv/bin/python scripts/release_check.py --no-build-isolation --include-notebooks
   ```

5. Confirm the wheel smoke step installs the built wheel in a fresh virtual
   environment, runs `pip check`, imports `qsvt`, checks `py.typed`, validates
   API-status labels, runs `qsvt --help`, and executes minimal scalar and report
   schema CLI commands.
6. Confirm generated outputs remain untracked except for deliberate research
   artifacts under `results/`.
7. Publish only after the tagged commit's Ordered Actions run passes lint,
   tests, dependency compatibility, integration, docs, notebook execution,
   package build, metadata validation, and the exact-wheel smoke test.

Live provider or paid hardware execution is not part of the default release
gate. Keep those checks behind explicit opt-in workflows and document the
provider, backend, shot limits, cost assumptions, and credentials used outside
portable reports.
