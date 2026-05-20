import re
from pathlib import Path

import tomllib

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
