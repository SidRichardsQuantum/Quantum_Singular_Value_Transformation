import importlib.util
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_text(relative_path):
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _single_match(pattern, text, source):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    assert len(matches) == 1, f"expected one release marker in {source}, got {matches}"
    return matches[0]


def _first_match(pattern, text, source):
    match = re.search(pattern, text, flags=re.MULTILINE)
    assert match is not None, f"expected release marker in {source}"
    return match.group(1)


def test_release_metadata_markers_match_project_version():
    project = tomllib.loads(_read_text("pyproject.toml"))["project"]
    version = project["version"]

    markers = {
        "README.md": _single_match(
            r"^Current release: `([^`]+)`$",
            _read_text("README.md"),
            "README.md",
        ),
        "RESULTS.md snapshots": _single_match(
            r"^These snapshots were refreshed for package version `([^`]+)`\.$",
            _read_text("RESULTS.md"),
            "RESULTS.md snapshots",
        ),
        "RESULTS.md benchmarks": _single_match(
            r"^Benchmark artefacts were refreshed for package version `([^`]+)`\.$",
            _read_text("RESULTS.md"),
            "RESULTS.md benchmarks",
        ),
        "docs/qsvt/results.md": _single_match(
            r'<span class="metric-value">([^<]+)</span>\n'
            r'\s*<span class="metric-label">current release marker</span>',
            _read_text("docs/qsvt/results.md"),
            "docs/qsvt/results.md",
        ),
        "CHANGELOG.md": _first_match(
            r"^## \[([^\]]+)\] .*$",
            _read_text("CHANGELOG.md"),
            "CHANGELOG.md top entry",
        ),
    }

    assert markers == {source: version for source in markers}


def test_release_check_build_command_supports_local_no_isolation_mode():
    module_path = REPO_ROOT / "scripts" / "release_check.py"
    spec = importlib.util.spec_from_file_location("release_check", module_path)
    assert spec is not None
    assert spec.loader is not None
    release_check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release_check)

    assert release_check._build_command(no_isolation=False) == [
        release_check.sys.executable,
        "-m",
        "build",
    ]
    assert release_check._build_command(no_isolation=True) == [
        release_check.sys.executable,
        "-m",
        "build",
        "--no-isolation",
    ]


def test_release_extra_includes_no_isolation_build_requirements():
    project = tomllib.loads(_read_text("pyproject.toml"))["project"]
    release_deps = set(project["optional-dependencies"]["release"])

    assert {"build", "twine", "wheel"} <= release_deps


def test_sdist_manifest_keeps_large_repo_artifacts_out_of_package():
    manifest = _read_text("MANIFEST.in")

    assert "graft src" in manifest
    assert "include ROADMAP.md" in manifest
    assert "prune notebooks" in manifest
    assert "prune results" in manifest
    assert "prune docs" in manifest
    assert "prune tests" in manifest
