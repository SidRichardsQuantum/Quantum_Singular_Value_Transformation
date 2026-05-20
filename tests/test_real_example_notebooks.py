import json
import os
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest


def _execute_notebooks(notebooks):
    matplotlib.use("Agg")

    assert notebooks

    with tempfile.TemporaryDirectory() as mpl_config_dir:
        old_mpl_config = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = mpl_config_dir
        try:
            for path in notebooks:
                namespace = {"__name__": "__notebook_test__"}

                notebook = json.loads(path.read_text())
                for cell in notebook["cells"]:
                    if cell.get("cell_type") == "code":
                        exec("".join(cell["source"]), namespace)
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
