"""
Local release preflight checks for qsvt-pennylane.

This script mirrors the main CI release gates without mutating release
metadata. It is intended to be run before the final version/changelog commit.
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(command: Sequence[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _python_module(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


def _require_module(module: str, extra: str) -> None:
    if importlib.util.find_spec(module) is None:
        raise SystemExit(f"{module} is not installed; install .[{extra}].")


def _dist_artifacts() -> list[str]:
    version = _project_version()
    artifacts = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "dist").glob(f"*{version}*")
    )
    if not artifacts:
        raise SystemExit(f"No distribution artifacts found in dist/ for {version}.")
    return artifacts


def _project_version() -> str:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, flags=re.MULTILINE)
    if match is None:
        raise SystemExit("Could not find project version in pyproject.toml.")
    return match.group(1)


def _build_command(*, no_isolation: bool) -> list[str]:
    args = ["--no-isolation"] if no_isolation else []
    return _python_module("build", *args)


def _check_git_hygiene() -> None:
    tracked_paths = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    disallowed_patterns = [
        ".coverage",
        "coverage.xml",
        "dist/**",
        "docs/_build/**",
        "build/**",
        "htmlcov/**",
        ".pytest_cache/**",
        ".ruff_cache/**",
        ".mypy_cache/**",
        ".venv/**",
        "venv/**",
        "env/**",
        "**/__pycache__/**",
        "**/*.pyc",
        "**/*.pyo",
        "**/*.pyd",
    ]
    tracked = [
        path
        for path in tracked_paths
        if any(
            _matches_generated_artifact(path, pattern)
            for pattern in disallowed_patterns
        )
    ]
    if tracked:
        paths = "\n".join(f"  - {path}" for path in tracked)
        raise SystemExit(f"Generated release artifacts are tracked:\n{paths}")


def _check_report_schema_fixtures() -> None:
    fixture_dir = REPO_ROOT / "tests" / "fixtures" / "reports"
    fixtures = sorted(fixture_dir.glob("*.json"))
    if not fixtures:
        raise SystemExit("No report schema fixtures found in tests/fixtures/reports.")

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from qsvt.reports import report_schema_manifest

    rows = report_schema_manifest(fixtures)
    failures = [row for row in rows if not row["supported"]]
    if failures:
        details = "\n".join(f"  - {row['path']}: {row['message']}" for row in failures)
        raise SystemExit(f"Report schema fixtures failed validation:\n{details}")
    command = [
        sys.executable,
        "-m",
        "qsvt",
        "report-schema-manifest",
        "--fail-on-unsupported",
        *[arg for fixture in fixtures for arg in ("--path", str(fixture))],
    ]
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src")
        + os.pathsep
        + os.environ.get("PYTHONPATH", ""),
    }
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True, env=env)


def _check_algorithm_truth_contract_semantics() -> None:
    """Reject release metadata that can overstate QSVT execution."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from qsvt._algorithm_reports import (
        algorithm_truth_contract,
        algorithm_truth_contract_issues,
    )
    from qsvt.reports import load_report, migrate_algorithm_workflow_report

    compatible = algorithm_truth_contract(
        "release-check-compatible",
        target="release preflight",
        qsvt_check="succeeded",
        polynomials={"transform": [0.0, 0.0, 1.0]},
    )
    mixed = algorithm_truth_contract(
        "release-check-mixed",
        target="release preflight",
        qsvt_check="succeeded",
        polynomials={"transform": [0.5, 0.5]},
    )
    failures = {
        "compatible": algorithm_truth_contract_issues(compatible),
        "mixed": algorithm_truth_contract_issues(mixed),
    }
    failures = {name: issues for name, issues in failures.items() if issues}
    if failures:
        raise SystemExit(f"Algorithm truth-contract semantic audit failed: {failures}")
    if compatible["execution_tier"] != "qsvt_circuit":
        raise SystemExit("Compatible executed polynomial was not labeled qsvt_circuit.")
    if mixed["execution_tier"] != "polynomial_core":
        raise SystemExit("Mixed-parity polynomial was mislabeled as qsvt_circuit.")

    legacy_path = (
        REPO_ROOT / "tests" / "fixtures" / "reports" / "algorithm_workflow_v1.json"
    )
    migrated = migrate_algorithm_workflow_report(load_report(legacy_path))
    migrated_issues = algorithm_truth_contract_issues(migrated["truth_contract"])
    if migrated["schema_version"] != "1.1" or migrated_issues:
        raise SystemExit(
            "Algorithm workflow 1.0-to-1.1 migration audit failed: "
            f"schema={migrated['schema_version']!r}, issues={migrated_issues!r}"
        )


def _matches_generated_artifact(path: str, pattern: str) -> bool:
    if pattern.startswith("**/"):
        return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(
            path,
            f"*{pattern[3:]}",
        )
    if pattern.endswith("/**"):
        return path.startswith(pattern[:-3])
    return fnmatch.fnmatch(path, pattern)


def _wheel_artifact() -> Path:
    version = _project_version()
    wheels = sorted((REPO_ROOT / "dist").glob(f"*{version}*.whl"))
    if not wheels:
        raise SystemExit(f"No wheel artifact found in dist/ for {version}.")
    if len(wheels) > 1:
        names = ", ".join(path.name for path in wheels)
        raise SystemExit(
            f"Expected one wheel artifact in dist/ for {version}, found: {names}"
        )
    return wheels[0]


def _run_wheel_smoke() -> None:
    wheel = _wheel_artifact()
    smoke_code = (
        "from importlib import resources; "
        "import qsvt; "
        "assert qsvt.__version__ != '0.0.0'; "
        "assert resources.files('qsvt').joinpath('py.typed').is_file(); "
        "assert qsvt.api_status('design_workflow') == "
        "qsvt.API_STATUS_STABLE; "
        "assert qsvt.api_status('execute_qsvt_circuit') == "
        "qsvt.API_STATUS_EXPERIMENTAL"
    )
    with tempfile.TemporaryDirectory(prefix="qsvt-wheel-smoke-") as tmp:
        venv_dir = Path(tmp) / "venv"
        _run([sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)])
        python = venv_dir / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--force-reinstall",
                str(wheel),
            ]
        )
        _run([str(python), "-m", "pip", "check"])
        _run([str(python), "-c", smoke_code])
        _run([str(python), "-m", "qsvt", "--help"])
        _run([str(python), "-m", "qsvt", "scalar", "--x", "0.5", "--poly", "0,0,1"])
        report_path = Path(tmp) / "schema-report.json"
        report_path.write_text(
            json.dumps(
                {
                    "schema_name": "qsvt-problem-workflow",
                    "schema_version": "1.0",
                    "target": "linear_system",
                    "truth_contract": {},
                }
            ),
            encoding="utf-8",
        )
        _run(
            [
                str(python),
                "-m",
                "qsvt",
                "report-schema-manifest",
                "--path",
                str(report_path),
            ]
        )
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run local release preflight checks.")
    parser.add_argument(
        "--skip-docs",
        action="store_true",
        help="Skip the Sphinx warning-as-error documentation build.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip package build and twine metadata checks.",
    )
    parser.add_argument(
        "--no-build-isolation",
        action="store_true",
        help=(
            "Build the package with the current environment instead of creating "
            "an isolated build environment. Useful for offline local preflights "
            "when build-system requirements are already installed."
        ),
    )
    parser.add_argument(
        "--skip-type",
        action="store_true",
        help="Skip mypy type checking.",
    )
    parser.add_argument(
        "--include-notebooks",
        action="store_true",
        help="Execute the full tutorial, real-example, and benchmark notebook suite.",
    )
    parser.add_argument(
        "--skip-wheel-smoke",
        action="store_true",
        help="Skip installing and smoke-testing the built wheel in a fresh venv.",
    )
    parser.add_argument(
        "--wheel-smoke-only",
        action="store_true",
        help="Smoke-test the current-version wheel already present in dist/.",
    )
    args = parser.parse_args(argv)

    if args.wheel_smoke_only:
        _run_wheel_smoke()
        return

    _check_git_hygiene()
    _check_report_schema_fixtures()
    _check_algorithm_truth_contract_semantics()
    _require_module("ruff", "lint")
    _run(_python_module("ruff", "check", "."))
    _require_module("black", "lint")
    _run(_python_module("black", "--check", "."))
    if not args.skip_type:
        _require_module("mypy", "type")
        _run(_python_module("mypy", "src/qsvt"))
    _run(
        _python_module(
            "pytest",
            "-m",
            "not notebook and not integration",
            "--cov=qsvt",
            "--cov-report=term-missing",
            "--cov-report=xml",
        )
    )
    _run(
        _python_module(
            "pytest",
            "-m",
            "integration",
            "tests/test_cookbook_examples.py",
        )
    )
    if args.include_notebooks:
        _run(
            _python_module(
                "pytest",
                "-m",
                "notebook",
                "tests/test_real_example_notebooks.py",
            )
        )
    if not args.skip_docs:
        _run(_python_module("sphinx", "-W", "-b", "html", "docs", "docs/_build/html"))
    if not args.skip_build:
        _run(_build_command(no_isolation=args.no_build_isolation))
        _run(_python_module("twine", "check", *_dist_artifacts()))
        if not args.skip_wheel_smoke:
            _run_wheel_smoke()


if __name__ == "__main__":
    main()
