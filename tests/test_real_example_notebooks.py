import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import pytest


def _sandboxed_benchmark_output_dirs(repo_root, artifact_root):
    benchmark_dir = artifact_root / "benchmarks"
    table_dir = artifact_root / "tables"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    def output_dirs(root=None):
        del root
        return repo_root, benchmark_dir, table_dir

    return output_dirs


def test_benchmark_notebook_artifact_paths_are_sandboxed(tmp_path):
    repo_root = tmp_path / "repository"
    output_dirs = _sandboxed_benchmark_output_dirs(
        repo_root,
        tmp_path / "sandbox",
    )
    found_root, benchmark_dir, table_dir = output_dirs()

    assert found_root == repo_root
    assert benchmark_dir == tmp_path / "sandbox" / "benchmarks"
    assert table_dir == tmp_path / "sandbox" / "tables"
    assert benchmark_dir.is_dir()
    assert table_dir.is_dir()


def _snapshot_files(*directories):
    return {
        path: path.read_bytes()
        for directory in directories
        for path in directory.rglob("*")
        if path.is_file()
    }


def _execute_notebooks(notebooks):
    matplotlib.use("Agg")

    assert notebooks

    repo_root = Path(__file__).resolve().parents[1]
    committed_artifacts = _snapshot_files(
        repo_root / "results" / "benchmarks",
        repo_root / "results" / "tables",
    )

    with tempfile.TemporaryDirectory() as mpl_config_dir:
        artifact_root = Path(mpl_config_dir) / "artifacts"
        output_dirs = _sandboxed_benchmark_output_dirs(repo_root, artifact_root)
        old_mpl_config = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = mpl_config_dir
        try:
            with patch("qsvt.notebook.benchmark_output_dirs", new=output_dirs):
                for path in notebooks:
                    namespace = {
                        "__name__": "__notebook_test__",
                        "__notebook_python__": sys.executable,
                    }

                    notebook = json.loads(path.read_text())
                    for cell in notebook["cells"]:
                        if cell.get("cell_type") == "code":
                            source = "".join(cell["source"])
                            source = source.replace(
                                '"python",\n            "-m",\n            "qsvt",',
                                (
                                    "__notebook_python__,\n"
                                    '            "-m",\n'
                                    '            "qsvt",'
                                ),
                            )
                            exec(source, namespace)
                    plt.close("all")
        finally:
            if old_mpl_config is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = old_mpl_config

    assert (
        _snapshot_files(
            repo_root / "results" / "benchmarks",
            repo_root / "results" / "tables",
        )
        == committed_artifacts
    )


@pytest.mark.notebook
def test_introductory_notebooks_execute():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_dir = repo_root / "notebooks" / "tutorials"
    notebooks = sorted(notebook_dir.glob("*.ipynb"))

    _execute_notebooks(notebooks)


@pytest.mark.notebook
def test_real_example_notebooks_execute():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_dir = repo_root / "notebooks" / "real_examples"
    notebooks = sorted(notebook_dir.glob("*.ipynb"))

    _execute_notebooks(notebooks)


@pytest.mark.notebook
def test_benchmark_notebooks_execute():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_dir = repo_root / "notebooks" / "benchmarks"
    notebooks = sorted(notebook_dir.glob("*.ipynb"))

    _execute_notebooks(notebooks)
