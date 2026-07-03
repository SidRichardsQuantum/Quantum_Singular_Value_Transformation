"""
Local release preflight checks for qsvt-pennylane.

This script mirrors the main CI release gates without mutating release
metadata. It is intended to be run before the final version/changelog commit.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
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
    artifacts = sorted(
        str(path.relative_to(REPO_ROOT)) for path in (REPO_ROOT / "dist").glob("*")
    )
    if not artifacts:
        raise SystemExit("No distribution artifacts found in dist/.")
    return artifacts


def _build_command(*, no_isolation: bool) -> list[str]:
    args = ["--no-isolation"] if no_isolation else []
    return _python_module("build", *args)


def _check_git_hygiene() -> None:
    ignored_outputs = [
        ".coverage",
        "coverage.xml",
        "dist",
        "docs/_build",
        "build",
    ]
    tracked = subprocess.run(
        ["git", "ls-files", *ignored_outputs],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    if tracked:
        paths = "\n".join(f"  - {path}" for path in tracked)
        raise SystemExit(f"Generated release artifacts are tracked:\n{paths}")


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
    args = parser.parse_args(argv)

    _check_git_hygiene()
    _require_module("ruff", "lint")
    _run(_python_module("ruff", "check", "."))
    _run(_python_module("ruff", "format", "--check", "src", "tests"))
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


if __name__ == "__main__":
    main()
