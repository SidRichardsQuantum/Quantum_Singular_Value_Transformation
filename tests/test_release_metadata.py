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
    project = tomllib.loads(_read_text("pyproject.toml"))["project"]
    assert release_check._project_version() == project["version"]


def test_release_artifact_selection_uses_current_project_version():
    source = _read_text("scripts/release_check.py")
    project = tomllib.loads(_read_text("pyproject.toml"))["project"]

    assert 'glob(f"*{version}*")' in source
    assert 'glob(f"*{version}*.whl")' in source
    assert f'version = "{project["version"]}"' in _read_text("pyproject.toml")


def test_release_preflight_detects_generated_artifact_patterns():
    module_path = REPO_ROOT / "scripts" / "release_check.py"
    spec = importlib.util.spec_from_file_location("release_check", module_path)
    assert spec is not None
    assert spec.loader is not None
    release_check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release_check)

    assert release_check._matches_generated_artifact(".venv/bin/python", ".venv/**")
    assert release_check._matches_generated_artifact(
        "src/qsvt/__pycache__/api.pyc",
        "**/__pycache__/**",
    )
    assert release_check._matches_generated_artifact("coverage.xml", "coverage.xml")
    assert release_check._matches_generated_artifact("dist/package.whl", "dist/**")
    assert not release_check._matches_generated_artifact(
        "results/report.json", "dist/**"
    )


def test_release_preflight_exposes_full_notebook_gate():
    source = _read_text("scripts/release_check.py")

    assert '"--include-notebooks"' in source
    assert '"tests/test_real_example_notebooks.py"' in source


def test_release_preflight_runs_built_wheel_smoke_by_default():
    source = _read_text("scripts/release_check.py")

    assert '"--skip-wheel-smoke"' in source
    assert "def _run_wheel_smoke" in source
    assert '"--system-site-packages"' in source
    assert '"--no-deps"' in source
    assert '"qsvt"' in source
    assert '"scalar"' in source
    assert '"report-schema-manifest"' in source


def test_release_preflight_validates_report_schema_fixtures():
    module_path = REPO_ROOT / "scripts" / "release_check.py"
    spec = importlib.util.spec_from_file_location("release_check", module_path)
    assert spec is not None
    assert spec.loader is not None
    release_check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release_check)

    release_check._check_report_schema_fixtures()
    source = _read_text("scripts/release_check.py")
    assert "report_schema_manifest" in source
    assert '"report-schema-manifest"' in source
    assert '"--fail-on-unsupported"' in source
    assert '"PYTHONPATH"' in source
    assert 'tests" / "fixtures" / "reports' in source


def test_release_preflight_always_runs_cookbook_integration_tests():
    source = _read_text("scripts/release_check.py")

    assert '"integration"' in source
    assert '"tests/test_cookbook_examples.py"' in source


def test_release_extra_includes_no_isolation_build_requirements():
    project = tomllib.loads(_read_text("pyproject.toml"))["project"]
    release_deps = set(project["optional-dependencies"]["release"])

    assert {"build", "twine", "wheel"} <= release_deps


def test_mypy_uses_runtime_python_for_current_dependency_stubs():
    config = tomllib.loads(_read_text("pyproject.toml"))["tool"]["mypy"]

    assert "python_version" not in config


def test_cli_tutorial_uses_active_python_interpreter():
    notebook = _read_text("notebooks/tutorials/10_QSVT_Reports_CLI_and_Artifacts.ipynb")

    assert '"import sys\\n"' in notebook
    assert '"            sys.executable,\\n"' in notebook
    assert '"            \\"python\\",\\n"' not in notebook


def test_sdist_manifest_keeps_large_repo_artifacts_out_of_package():
    manifest = _read_text("MANIFEST.in")

    assert "graft src" in manifest
    assert "include ROADMAP.md" in manifest
    assert "include RELEASING.md" in manifest
    assert "prune notebooks" in manifest
    assert "prune results" in manifest
    assert "prune docs" in manifest
    assert "prune tests" in manifest
    assert "prune .mypy_cache" in manifest
    assert "prune .venv" in manifest
    assert "prune venv" in manifest
    assert "prune env" in manifest
    assert "global-exclude .coverage" in manifest
    assert "global-exclude coverage.xml" in manifest


def test_stable_algorithm_workflows_have_theory_pages():
    import qsvt
    import qsvt.algorithms as algorithms

    expected_pages = {
        "block_encoded_qsvt_workflow": "workflow_block_encoded_qsvt.md",
        "fermi_dirac_occupation_workflow": "workflow_fermi_dirac.md",
        "fixed_point_amplification_workflow": ("workflow_fixed_point_amplification.md"),
        "ground_state_filtering_workflow": "workflow_ground_state_filtering.md",
        "hamiltonian_simulation_workflow": "workflow_hamiltonian_simulation.md",
        "linear_system_comparison_workflow": ("workflow_linear_system_comparison.md"),
        "linear_system_workflow": "workflow_linear_system.md",
        "matrix_log_entropy_workflow": "workflow_matrix_log_entropy.md",
        "quantum_walk_search_workflow": "workflow_quantum_walk_search.md",
        "resolvent_workflow": "workflow_resolvent.md",
        "singular_value_filtering_workflow": ("workflow_singular_value_filtering.md"),
        "singular_value_pseudoinverse_workflow": (
            "workflow_singular_value_pseudoinverse.md"
        ),
        "spectral_counting_workflow": "workflow_spectral_counting.md",
        "spectral_density_workflow": "workflow_spectral_density.md",
        "spectral_thresholding_workflow": "workflow_spectral_thresholding.md",
        "thermal_gibbs_workflow": "workflow_thermal_gibbs.md",
    }
    stable_algorithm_workflows = {
        name
        for name, status in qsvt.__api_statuses__.items()
        if status == qsvt.API_STATUS_STABLE
        and name.endswith("_workflow")
        and hasattr(algorithms, name)
    }

    assert stable_algorithm_workflows == set(expected_pages)

    algorithms_doc = _read_text("docs/qsvt/algorithms.md")
    api_doc = _read_text("docs/qsvt/api_reference.md")
    for workflow_name, page_name in expected_pages.items():
        page_path = REPO_ROOT / "docs" / "qsvt" / page_name
        assert page_path.is_file(), workflow_name
        page_text = page_path.read_text(encoding="utf-8")
        assert workflow_name in page_text
        assert page_name in algorithms_doc
        assert page_name in api_doc
