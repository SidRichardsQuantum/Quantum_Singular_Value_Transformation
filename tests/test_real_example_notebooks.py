from __future__ import annotations

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient
from notebooks._support import benchmark_output_dirs

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_notebook_artifact_paths_are_sandboxed(tmp_path, monkeypatch):
    output_root = tmp_path / "sandbox"
    monkeypatch.setenv("QSVT_NOTEBOOK_OUTPUT_ROOT", str(output_root))

    found_root, benchmark_dir, table_dir = benchmark_output_dirs(REPO_ROOT)

    assert found_root == output_root
    assert benchmark_dir == output_root / "results" / "benchmarks"
    assert table_dir == output_root / "results" / "tables"
    assert benchmark_dir.is_dir()
    assert table_dir.is_dir()


def _execute_notebooks(notebooks, tmp_path, monkeypatch):
    """Execute each notebook in a fresh Jupyter kernel with isolated artifacts."""
    assert notebooks

    output_root = tmp_path / "notebook-output"
    mpl_config = tmp_path / "matplotlib"
    mpl_config.mkdir()
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(mpl_config))
    monkeypatch.setenv("QSVT_NOTEBOOK_OUTPUT_ROOT", str(output_root))

    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=300,
            kernel_name="python3",
            resources={"metadata": {"path": str(REPO_ROOT)}},
            record_timing=False,
        )
        client.execute()

    return output_root


@pytest.mark.notebook
def test_introductory_notebooks_execute(tmp_path, monkeypatch):
    notebooks = sorted((REPO_ROOT / "notebooks" / "tutorials").glob("*.ipynb"))

    _execute_notebooks(notebooks, tmp_path, monkeypatch)


@pytest.mark.notebook
def test_real_example_notebooks_execute(tmp_path, monkeypatch):
    notebooks = sorted((REPO_ROOT / "notebooks" / "real_examples").glob("*.ipynb"))

    _execute_notebooks(notebooks, tmp_path, monkeypatch)


@pytest.mark.notebook
def test_benchmark_notebooks_execute(tmp_path, monkeypatch):
    notebooks = sorted((REPO_ROOT / "notebooks" / "benchmarks").glob("*.ipynb"))

    output_root = _execute_notebooks(notebooks, tmp_path, monkeypatch)

    assert list((output_root / "results" / "benchmarks").glob("*.json"))
    assert list((output_root / "results" / "tables").glob("*.csv"))
