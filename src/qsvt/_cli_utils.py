"""
Shared helpers for the qsvt command-line interface.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from .reports import report_to_jsonable, save_report, save_report_plot


def parse_float_list(text: str) -> list[float]:
    """
    Parse a comma-separated list of floats.
    """
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_complex_list(text: str) -> list[complex]:
    """
    Parse a comma-separated list of complex values.
    """
    return [complex(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    """
    Parse a comma-separated list of integers.
    """
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("expected at least one integer.")
    return values


def parse_poly(text: str) -> list[float]:
    """
    Parse polynomial coefficients.
    """
    return parse_float_list(text)


def parse_matrix(text: str) -> list[list[complex]]:
    """
    Parse a semicolon-separated matrix.
    """
    rows = [
        [complex(x.strip()) for x in row.split(",") if x.strip()]
        for row in text.split(";")
        if row.strip()
    ]
    if not rows:
        raise ValueError("matrix must contain at least one row.")

    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("matrix rows must all have the same nonzero length.")

    return rows


def add_report_output_args(
    parser: argparse.ArgumentParser,
    *,
    include_plot: bool = False,
) -> None:
    """
    Add common output arguments for report-oriented CLI commands.
    """
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path for writing the report JSON.",
    )
    if include_plot:
        parser.add_argument(
            "--plot",
            type=str,
            help="Optional path for writing a target-vs-polynomial plot.",
        )
    parser.add_argument(
        "--print-report",
        action="store_true",
        help=(
            "Print the full report JSON to stdout even when --output or --plot is used."
        ),
    )


def emit_cli_result(args: argparse.Namespace, result: dict[str, Any]) -> None:
    """
    Persist optional report/plot outputs and print the CLI JSON payload.
    """
    output_path = getattr(args, "output", None)
    plot_path = getattr(args, "plot", None)
    rows_output_path = getattr(args, "rows_output", None)

    if output_path:
        save_report(result, output_path)
    if plot_path:
        save_report_plot(result, plot_path)

    should_print_report = getattr(args, "print_report", False) or (
        output_path is None and plot_path is None
    )
    if should_print_report:
        payload = report_to_jsonable(result)
    else:
        payload = {
            "mode": result.get("mode", args.command),
            "report_written": output_path is not None,
            "plot_written": plot_path is not None,
            "rows_written": rows_output_path is not None,
        }
        if output_path is not None:
            payload["output"] = output_path
        if plot_path is not None:
            payload["plot"] = plot_path
        if rows_output_path is not None:
            payload["rows_output"] = rows_output_path

    print(json.dumps(payload, indent=2))
