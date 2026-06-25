import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest


def _sandbox_notebook_source(source, path, artifact_root):
    if "notebooks/benchmarks" not in path.as_posix():
        return source

    benchmark_dir = artifact_root / "benchmarks"
    table_dir = artifact_root / "tables"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    return source.replace(
        'ARTIFACT_DIR = ROOT / "results/benchmarks"',
        f"ARTIFACT_DIR = Path({str(benchmark_dir)!r})",
    ).replace(
        'TABLE_DIR = ROOT / "results/tables"',
        f"TABLE_DIR = Path({str(table_dir)!r})",
    )


def test_benchmark_notebook_artifact_paths_are_sandboxed(tmp_path):
    source = "\n".join(
        [
            'ARTIFACT_DIR = ROOT / "results/benchmarks"',
            'TABLE_DIR = ROOT / "results/tables"',
        ]
    )

    sandboxed = _sandbox_notebook_source(
        source,
        Path("notebooks/benchmarks/example.ipynb"),
        tmp_path,
    )

    assert 'ROOT / "results/benchmarks"' not in sandboxed
    assert 'ROOT / "results/tables"' not in sandboxed
    assert str(tmp_path / "benchmarks") in sandboxed
    assert str(tmp_path / "tables") in sandboxed


def _execute_notebooks(notebooks):
    matplotlib.use("Agg")

    assert notebooks

    with tempfile.TemporaryDirectory() as mpl_config_dir:
        artifact_root = Path(mpl_config_dir) / "artifacts"
        old_mpl_config = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = mpl_config_dir
        try:
            for path in notebooks:
                namespace = {
                    "__name__": "__notebook_test__",
                    "__notebook_python__": sys.executable,
                }

                notebook = json.loads(path.read_text())
                for cell in notebook["cells"]:
                    if cell.get("cell_type") == "code":
                        source = _sandbox_notebook_source(
                            "".join(cell["source"]),
                            path,
                            artifact_root,
                        )
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
