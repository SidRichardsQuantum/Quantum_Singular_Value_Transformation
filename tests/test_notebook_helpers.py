import json
import re
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest
from notebooks._support import (
    benchmark_output_dirs,
    display_table,
    find_repo_root,
    format_value,
    print_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRS = (
    REPO_ROOT / "notebooks" / "tutorials",
    REPO_ROOT / "notebooks" / "real_examples",
    REPO_ROOT / "notebooks" / "benchmarks",
)

NOTEBOOK_ROLES = {
    "tutorials": "tutorial",
    "real_examples": "real-example",
    "benchmarks": "benchmark",
}

DIRECT_ADDRESS_PATTERN = re.compile(
    r"\b(?:we|our|ours|you|your|yours|let['’]?s)\b",
    flags=re.IGNORECASE,
)


def test_notebook_support_is_repository_only():
    assert find_spec("qsvt.notebook") is None


def test_real_example_gallery_is_curated():
    notebooks = {
        path.name
        for path in (REPO_ROOT / "notebooks" / "real_examples").glob("*.ipynb")
    }

    assert notebooks == {
        "01_poisson_equation_pde.ipynb",
        "02_hamiltonian_simulation_schrodinger_dynamics.ipynb",
        "03_greens_function_response.ipynb",
        "04_ising_phase_transition_filtering.ipynb",
        "05_fermi_dirac_electronic_occupations.ipynb",
        "06_topological_band_projector_chern_marker.ipynb",
        "07_singular_value_pseudoinverse_deblurring.ipynb",
        "08_matrix_log_entropy_graph_laplacian.ipynb",
    }


def test_every_notebook_has_concise_variable_definitions():
    notebook_paths = sorted(
        path for directory in NOTEBOOK_DIRS for path in directory.glob("*.ipynb")
    )

    assert len(notebook_paths) == 35
    for path in notebook_paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        markdown_cells = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        ]
        definition_sections = [
            source.split("## Variable definitions", maxsplit=1)[1].split(
                "\n## ", maxsplit=1
            )[0]
            for source in markdown_cells
            if "## Variable definitions" in source
        ]

        assert definition_sections, f"{path} has no variable-definition block"
        definition_lines = [
            line
            for section in definition_sections
            for line in section.splitlines()
            if line.startswith("- ")
        ]
        assert definition_lines, f"{path} has an empty variable-definition block"
        assert len(definition_lines) <= 8, (
            f"{path} inventories implementation details instead of keeping a "
            "compact mathematical glossary"
        )
        assert not any(
            phrase in line.lower()
            for line in definition_lines
            for phrase in (
                "plotting helper",
                "output directories",
                "artifact destinations",
            )
        ), f"{path} includes implementation-local names in its definition block"


def test_notebook_prose_avoids_direct_reader_or_author_address():
    for directory in NOTEBOOK_DIRS:
        for path in sorted(directory.glob("*.ipynb")):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            markdown = "\n".join(
                "".join(cell["source"])
                for cell in notebook["cells"]
                if cell["cell_type"] == "markdown"
            )
            match = DIRECT_ADDRESS_PATTERN.search(markdown)
            assert (
                match is None
            ), f"{path} directly addresses the reader: {match.group()}"


def test_every_notebook_follows_the_visible_navigation_contract():
    for directory in NOTEBOOK_DIRS:
        paths = sorted(directory.glob("*.ipynb"))
        for index, path in enumerate(paths):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            cells = notebook["cells"]

            assert cells[0]["cell_type"] == "markdown", path
            assert "".join(cells[0]["source"]).startswith("# "), path

            guide = "".join(cells[1]["source"])
            assert guide.startswith("## Notebook guide\n"), path
            for label in (
                "**Learning objective:**",
                "**Prerequisites:**",
                "**Estimated runtime:**",
                "**Navigation:**",
            ):
                assert label in guide, f"{path} is missing {label}"
            assert "[collection index](README.md)" in guide, path
            if index:
                assert f"[previous]({paths[index - 1].name})" in guide, path
            else:
                assert "[previous](" not in guide, path
            if index + 1 < len(paths):
                assert f"[next]({paths[index + 1].name})" in guide, path
            else:
                assert "[next](" not in guide, path

            takeaways = "".join(cells[-1]["source"])
            assert takeaways.startswith("## Takeaways and next steps\n"), path
            assert "**Result:**" in takeaways, path
            assert "**Interpretation boundary:**" in takeaways, path
            assert "**Continue:**" in takeaways, path


def test_every_notebook_uses_canonical_kernel_and_schema_metadata():
    for directory in NOTEBOOK_DIRS:
        for path in sorted(directory.glob("*.ipynb")):
            metadata = json.loads(path.read_text(encoding="utf-8"))["metadata"]

            assert metadata["kernelspec"] == {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }, path
            assert metadata["language_info"] == {"name": "python"}, path
            assert metadata["qsvt_notebook"] == {
                "role": NOTEBOOK_ROLES[directory.name],
                "schema_version": "1.0",
            }, path


def test_notebook_normalization_is_idempotent():
    subprocess.run(
        [sys.executable, "scripts/normalize_notebooks.py", "--check"],
        cwd=REPO_ROOT,
        check=True,
    )


def test_find_repo_root_and_benchmark_output_dirs(tmp_path):
    root = tmp_path / "project"
    nested = root / "notebooks" / "benchmarks"
    nested.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")

    assert find_repo_root(nested) == root

    found_root, artifact_dir, table_dir = benchmark_output_dirs(nested)
    assert found_root == root
    assert artifact_dir == root / "results" / "benchmarks"
    assert table_dir == root / "results" / "tables"
    assert artifact_dir.is_dir()
    assert table_dir.is_dir()

    with pytest.raises(RuntimeError, match="could not locate"):
        find_repo_root(tmp_path / "missing")


def test_benchmark_output_dirs_support_an_isolated_output_root(
    tmp_path,
    monkeypatch,
):
    repo_root = tmp_path / "project"
    nested = repo_root / "notebooks" / "benchmarks"
    nested.mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "isolated-output"
    monkeypatch.setenv("QSVT_NOTEBOOK_OUTPUT_ROOT", str(output_root))

    found_root, artifact_dir, table_dir = benchmark_output_dirs(nested)

    assert found_root == output_root
    assert artifact_dir == output_root / "results" / "benchmarks"
    assert table_dir == output_root / "results" / "tables"


def test_format_value_matches_notebook_table_conventions():
    assert format_value(None) == "n/a"
    assert format_value(0.0) == "0"
    assert format_value(1.25) == "1.25"
    assert format_value(1e-5) == "1.00e-05"
    assert format_value(1e5) == "1.00e+05"
    assert format_value([1.0, None, 1e-5]) == "1, n/a, 1.00e-05"
    assert (
        format_value(1e5, float_digits=6, sci_digits=3, scientific_large=None)
        == "100000"
    )
    assert format_value(1 + 2j) == "1+2j"


def test_display_table_column_layout(capsys):
    display_table(
        "Readout",
        [{"a": 1.0, "b": 1e-5}],
        [
            ("A", lambda row: row["a"]),
            ("B", lambda row: row["b"]),
        ],
    )

    assert capsys.readouterr().out == ("Readout\n-------\nA : 1\nB        : 1.00e-05\n")


def test_display_table_row_layout(capsys):
    display_table(
        "Rows",
        [{"a": 1.0, "b": 1e-5}],
        [
            ("A", lambda row: row["a"]),
            ("B", lambda row: row["b"]),
        ],
        layout="rows",
        rule="=",
    )

    assert capsys.readouterr().out == (
        "Rows\n====\nA  B       \n-  --------\n1  1.00e-05\n"
    )


def test_display_table_rejects_unknown_layout():
    with pytest.raises(ValueError, match="layout must be"):
        display_table(
            "Bad",
            [{"a": 1}],
            [("A", lambda row: row["a"])],
            layout="diagonal",
        )


def test_display_table_empty_rows_keep_headers(capsys):
    display_table(
        "Empty",
        [],
        [
            ("A", lambda row: row["a"]),
            ("B", lambda row: row["b"]),
        ],
        layout="rows",
    )

    assert capsys.readouterr().out == "Empty\n-----\nA  B\n-  -\n"


def test_print_rows_uses_linear_system_notebook_format(capsys):
    print_rows(
        [{"value": None, "residual": 1e-5, "phase": 1 + 2j}],
        ["value", "residual", "phase"],
    )

    assert capsys.readouterr().out == (
        "value  residual   phase\n-----  ---------  -----\n-      1.000e-05  1+2j \n"
    )


def test_print_rows_empty_table_keeps_header(capsys):
    print_rows([], ["value", "residual"])

    assert capsys.readouterr().out == "value  residual\n-----  --------\n"
