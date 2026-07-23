import json
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


def test_every_notebook_defines_markdown_variables_and_parameters():
    notebook_paths = sorted(
        path for directory in NOTEBOOK_DIRS for path in directory.glob("*.ipynb")
    )

    assert len(notebook_paths) == 33
    for path in notebook_paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        markdown_cells = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        ]
        definition_cells = [
            source for source in markdown_cells if "## Variable definitions" in source
        ]

        assert definition_cells, f"{path} has no variable-definition block"
        assert any(
            line.startswith("- ")
            for source in definition_cells
            for line in source.splitlines()
        ), f"{path} has an empty variable-definition block"


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
