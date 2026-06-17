"""
Small presentation and path helpers shared by the example notebooks.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

TableLayout = Literal["columns", "rows"]
ColumnSpec = tuple[str, Callable[[Any], Any]]


def find_repo_root(start: str | Path | None = None) -> Path:
    """
    Find the nearest ancestor containing ``pyproject.toml``.
    """
    current = Path.cwd() if start is None else Path(start)
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("could not locate repository root")


def benchmark_output_dirs(root: str | Path | None = None) -> tuple[Path, Path, Path]:
    """
    Return ``(root, artifact_dir, table_dir)`` and create the output directories.
    """
    repo_root = find_repo_root(root)
    artifact_dir = repo_root / "results/benchmarks"
    table_dir = repo_root / "results/tables"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    return repo_root, artifact_dir, table_dir


def format_value(
    value: Any,
    *,
    none: str = "n/a",
    float_digits: int = 4,
    sci_digits: int = 2,
    scientific_small: float = 1e-3,
    scientific_large: float | None = 1e4,
    sequence_separator: str = ", ",
    complex_digits: int = 6,
) -> str:
    """
    Format scalar values compactly for notebook text tables.
    """
    if value is None:
        return none
    if isinstance(value, (list, tuple, set)):
        return sequence_separator.join(
            format_value(
                item,
                none=none,
                float_digits=float_digits,
                sci_digits=sci_digits,
                scientific_small=scientific_small,
                scientific_large=scientific_large,
                sequence_separator=sequence_separator,
                complex_digits=complex_digits,
            )
            for item in value
        )
    if isinstance(value, (float, np.floating)):
        return _format_float(
            float(value),
            float_digits=float_digits,
            sci_digits=sci_digits,
            scientific_small=scientific_small,
            scientific_large=scientific_large,
        )
    if isinstance(value, (complex, np.complexfloating)):
        return f"{value.real:.{complex_digits}g}{value.imag:+.{complex_digits}g}j"
    return str(value)


def display_table(
    title: str,
    rows: Sequence[Any],
    columns: Sequence[ColumnSpec],
    *,
    layout: TableLayout = "columns",
    rule: str = "-",
    formatter: Callable[[Any], str] = format_value,
) -> None:
    """
    Print a simple text table for notebook output cells.

    ``layout="columns"`` treats each column specification as a displayed row
    and each input row as a comparison item. ``layout="rows"`` prints a
    conventional header row followed by one displayed row per input item.
    """
    print(title)
    print(rule * len(title))
    if layout == "columns":
        _display_columns(rows, columns, formatter)
    elif layout == "rows":
        _display_rows(rows, columns, formatter)
    else:
        raise ValueError("layout must be 'columns' or 'rows'.")


def print_rows(
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str],
    *,
    formatter: Callable[[Any], str] | None = None,
) -> None:
    """
    Print a conventional fixed-width text table from dict rows.
    """
    value_formatter = formatter or (
        lambda value: format_value(
            value,
            none="-",
            float_digits=6,
            sci_digits=3,
            scientific_large=None,
        )
    )
    widths = {
        column: max(
            [len(column)] + [len(value_formatter(row.get(column))) for row in rows]
        )
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print(
            "  ".join(
                value_formatter(row.get(column)).ljust(widths[column])
                for column in columns
            )
        )


def _display_columns(
    rows: Sequence[Any],
    columns: Sequence[ColumnSpec],
    formatter: Callable[[Any], str],
) -> None:
    rendered = [
        [str(header), *[formatter(accessor(row)) for row in rows]]
        for header, accessor in columns
    ]
    widths = [max(len(item) for item in row) for row in rendered]
    for row, width in zip(rendered, widths, strict=True):
        print(f"{row[0]:<{width}} : " + " | ".join(row[1:]))


def _display_rows(
    rows: Sequence[Any],
    columns: Sequence[ColumnSpec],
    formatter: Callable[[Any], str],
) -> None:
    rendered = [
        [str(name) for name, _ in columns],
        *[[formatter(getter(row)) for _, getter in columns] for row in rows],
    ]
    widths = [max(len(row[index]) for row in rendered) for index in range(len(columns))]
    print(
        "  ".join(
            value.ljust(width) for value, width in zip(rendered[0], widths, strict=True)
        )
    )
    print("  ".join("-" * width for width in widths))
    for row in rendered[1:]:
        print(
            "  ".join(
                value.ljust(width) for value, width in zip(row, widths, strict=True)
            )
        )


def _format_float(
    value: float,
    *,
    float_digits: int,
    sci_digits: int,
    scientific_small: float,
    scientific_large: float | None,
) -> str:
    if value == 0.0:
        return "0"
    use_scientific = abs(value) < scientific_small
    if scientific_large is not None:
        use_scientific = use_scientific or abs(value) >= scientific_large
    if use_scientific:
        return f"{value:.{sci_digits}e}"
    return f"{value:.{float_digits}g}"


__all__ = [
    "benchmark_output_dirs",
    "display_table",
    "find_repo_root",
    "format_value",
    "print_rows",
]
