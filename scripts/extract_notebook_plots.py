#!/usr/bin/env python3
"""Extract embedded notebook outputs into result artefacts and docs pages.

The notebooks are the source of truth for these figures. This script refreshes
plot directories with stable filenames so ``RESULTS.md`` and the documentation
can link to committed plot artefacts. It can also regenerate the result pages
that display every embedded plot and plain-text result from the notebooks.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_NOTEBOOK_GLOB = "notebooks/tutorials/*.ipynb"
DEFAULT_OUTPUT_DIR = "results/plots/notebooks"
TUTORIAL_DOC = Path("docs/qsvt/tutorial_results.md")
REAL_EXAMPLES_DOC = Path("docs/qsvt/real_example_results.md")
REAL_EXAMPLES_MANIFEST = Path("results/tables/real_examples_plot_manifest.csv")


@dataclass(frozen=True)
class TextOutput:
    cell_index: int
    text: str


@dataclass(frozen=True)
class PlotOutput:
    path: Path
    cell_index: int
    plot_index: int
    width_px: int
    height_px: int


@dataclass
class NotebookResult:
    path: Path
    title: str
    plots: list[PlotOutput] = field(default_factory=list)
    text_outputs: list[TextOutput] = field(default_factory=list)


def _decode_png_payload(payload: str | list[str]) -> bytes:
    if isinstance(payload, list):
        payload = "".join(payload)
    return base64.b64decode(payload)


def _png_dimensions(payload: bytes) -> tuple[int, int]:
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("embedded image payload is not a PNG")
    return struct.unpack(">II", payload[16:24])


def _coerce_text(payload: str | list[str]) -> str:
    if isinstance(payload, list):
        return "".join(payload)
    return payload


def _is_noise_text(text: str) -> bool:
    stripped = text.strip()
    return not stripped or stripped.startswith("<Figure size ")


def _title_from_notebook(path: Path, notebook: dict) -> str:
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = _coerce_text(cell.get("source", []))
        for line in source.splitlines():
            clean = line.strip()
            if clean.startswith("#"):
                return clean.lstrip("#").strip()
    return path.stem.replace("_", " ")


def _extract_text_output(cell_index: int, output: dict) -> TextOutput | None:
    if "text" in output:
        text = _coerce_text(output["text"])
    else:
        text = _coerce_text(output.get("data", {}).get("text/plain", ""))

    if _is_noise_text(text):
        return None
    return TextOutput(cell_index=cell_index, text=_format_text_output(text))


def _format_text_output(text: str) -> str:
    formatted = text.rstrip()
    formatted = _unwrap_numpy_scalars(formatted)
    formatted = _unwrap_numpy_array_reprs(formatted)
    return _strip_standalone_array_repr(formatted)


def _unwrap_numpy_scalars(text: str) -> str:
    scalar_pattern = re.compile(
        r"\b(?:np\.)?(?:float(?:16|32|64)|int(?:8|16|32|64)|bool_)\(([^()]*)\)"
    )
    previous = None
    while previous != text:
        previous = text
        text = scalar_pattern.sub(r"\1", text)
    return text


def _strip_standalone_array_repr(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("array(") or not stripped.endswith(")"):
        return text
    inner = stripped[len("array(") : -1]
    if ", dtype=" in inner:
        return text
    if not inner.lstrip().startswith("["):
        return text
    return inner


def _unwrap_numpy_array_reprs(text: str) -> str:
    cursor = 0
    chunks: list[str] = []
    while True:
        start = text.find("array(", cursor)
        if start == -1:
            chunks.append(text[cursor:])
            return "".join(chunks)

        list_start = start + len("array(")
        while list_start < len(text) and text[list_start].isspace():
            list_start += 1
        if list_start >= len(text) or text[list_start] != "[":
            chunks.append(text[cursor : start + len("array(")])
            cursor = start + len("array(")
            continue

        list_end = _matching_bracket_index(text, list_start)
        if list_end is None:
            chunks.append(text[cursor : start + len("array(")])
            cursor = start + len("array(")
            continue

        close_index = list_end + 1
        while close_index < len(text) and text[close_index].isspace():
            close_index += 1
        if text.startswith(", dtype=", close_index):
            chunks.append(text[cursor : list_end + 1])
            cursor = list_end + 1
            continue
        if close_index >= len(text) or text[close_index] != ")":
            chunks.append(text[cursor : start + len("array(")])
            cursor = start + len("array(")
            continue

        chunks.append(text[cursor:start])
        chunks.append(text[list_start : list_end + 1])
        cursor = close_index + 1


def _matching_bracket_index(text: str, start: int) -> int | None:
    depth = 0
    for index in range(start, len(text)):
        character = text[index]
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                return index
    return None


def extract_notebook_plots(
    *,
    notebook_glob: str = DEFAULT_NOTEBOOK_GLOB,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    clean: bool = True,
) -> list[NotebookResult]:
    notebooks = sorted(Path().glob(notebook_glob))
    if not notebooks:
        raise FileNotFoundError(f"no notebooks matched {notebook_glob!r}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    if clean:
        for old_plot in destination.glob("*.png"):
            old_plot.unlink()

    results: list[NotebookResult] = []
    for notebook_path in notebooks:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        result = NotebookResult(
            path=notebook_path,
            title=_title_from_notebook(notebook_path, notebook),
        )
        plot_index = 0

        for cell_index, cell in enumerate(notebook.get("cells", []), start=1):
            for output in cell.get("outputs", []):
                png_payload = output.get("data", {}).get("image/png")
                if png_payload:
                    plot_index += 1
                    plot_path = (
                        destination / f"{notebook_path.stem}-plot-{plot_index:02d}.png"
                    )
                    plot_bytes = _decode_png_payload(png_payload)
                    plot_path.write_bytes(plot_bytes)
                    width_px, height_px = _png_dimensions(plot_bytes)
                    result.plots.append(
                        PlotOutput(
                            path=plot_path,
                            cell_index=cell_index,
                            plot_index=plot_index,
                            width_px=width_px,
                            height_px=height_px,
                        )
                    )

                text_output = _extract_text_output(cell_index, output)
                if text_output is not None:
                    result.text_outputs.append(text_output)

        results.append(result)

    return results


def execute_notebooks(notebook_glob: str) -> None:
    notebooks = sorted(Path().glob(notebook_glob))
    if not notebooks:
        raise FileNotFoundError(f"no notebooks matched {notebook_glob!r}")

    for notebook in notebooks:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--ExecutePreprocessor.kernel_name=python3",
                "--inplace",
                str(notebook),
            ],
            check=True,
        )


def _path_from_doc(doc_path: Path, artefact: Path) -> str:
    del doc_path
    return (Path("../../") / artefact).as_posix()


def _display_notebook_path(path: Path) -> str:
    return path.as_posix()


def _doc_image_width(plot: PlotOutput) -> int:
    if plot.width_px >= 1000 or plot.width_px / max(plot.height_px, 1) >= 2.0:
        return 760
    if plot.width_px >= 800:
        return 640
    return 520


def write_plot_manifest(results: list[NotebookResult], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            [
                "artefact",
                "notebook",
                "title",
                "plot_index",
                "cell_index",
                "width_px",
                "height_px",
                "result_type",
                "notes",
            ]
        )
        for result in results:
            for plot in result.plots:
                writer.writerow(
                    [
                        plot.path.as_posix(),
                        result.path.as_posix(),
                        result.title,
                        plot.plot_index,
                        plot.cell_index,
                        plot.width_px,
                        plot.height_px,
                        "plot",
                        "embedded notebook PNG output",
                    ]
                )


def write_results_page(
    *,
    results: list[NotebookResult],
    doc_path: Path,
    title: str,
    intro: str,
    manifest_path: Path | None = None,
) -> None:
    doc_path.parent.mkdir(parents=True, exist_ok=True)

    plot_count = sum(len(result.plots) for result in results)
    text_count = sum(len(result.text_outputs) for result in results)

    lines = [
        f"# {title}",
        "",
        "<!-- AUTO-GENERATED by scripts/extract_notebook_plots.py. -->",
        "<!-- Re-run after executing notebooks; do not edit ledgers by hand. -->",
        "",
        intro,
        "",
        "## Current Status",
        "",
        f"- Source notebooks: `{results[0].path.parent.as_posix()}/`",
        f"- Notebooks displayed: `{len(results)}`",
        f"- Embedded plot artefacts displayed: `{plot_count}`",
        f"- Plain-text notebook results displayed: `{text_count}`",
    ]
    if manifest_path is not None:
        manifest_link = _path_from_doc(doc_path, manifest_path)
        lines.append(
            f"- Plot manifest: [`{manifest_path.as_posix()}`]({manifest_link})"
        )

    lines.extend(
        [
            "",
            "## Related Pages",
            "",
            "- [Results summary](results.md)",
            "- [Notebook index](notebooks.md)",
        ]
    )
    if doc_path != TUTORIAL_DOC:
        lines.append("- [Tutorial notebook outputs](tutorial_results.md)")
    if doc_path != REAL_EXAMPLES_DOC:
        lines.append("- [Real-example notebook outputs](real_example_results.md)")

    lines.extend(
        [
            "",
            "## Regeneration",
            "",
            "Execute notebooks, extract their embedded outputs, "
            "and refresh this page with:",
            "",
            "```bash",
            _regeneration_command(results[0].path.parent, doc_path),
            "```",
            "",
            "## Notebook Results",
            "",
        ]
    )

    for result in results:
        source_path = _display_notebook_path(result.path)
        source_link = f"../../{result.path.as_posix()}"
        lines.extend(
            [
                f"### `{result.path.name}`",
                "",
                f"Source: [`{source_path}`]({source_link})",
                "",
            ]
        )
        if not result.plots and not result.text_outputs:
            lines.extend(["No embedded plot or text outputs were found.", ""])
            continue

        for index, plot in enumerate(result.plots, start=1):
            lines.extend(
                [
                    f"```{{image}} {_path_from_doc(doc_path, plot.path)}",
                    f":alt: {result.title} plot {index}",
                    f":width: {_doc_image_width(plot)}px",
                    "```",
                    "",
                ]
            )

        for index, output in enumerate(result.text_outputs, start=1):
            lines.extend(
                [
                    f"Output {index} (cell {output.cell_index}):",
                    "",
                    "```text",
                    output.text,
                    "```",
                    "",
                ]
            )

    doc_path.write_text("\n".join(lines), encoding="utf-8")


def _regeneration_command(notebook_dir: Path, doc_path: Path) -> str:
    if notebook_dir == Path("notebooks/tutorials"):
        return (
            "python scripts/extract_notebook_plots.py "
            "--preset tutorials --execute --write-docs"
        )
    if notebook_dir == Path("notebooks/real_examples"):
        return (
            "python scripts/extract_notebook_plots.py "
            "--preset real-examples --execute --write-docs"
        )
    return (
        "python scripts/extract_notebook_plots.py "
        f'--notebook-glob "{notebook_dir.as_posix()}/*.ipynb" '
        f"--doc-output {doc_path.as_posix()} --execute --write-docs"
    )


def _preset_args(preset: str) -> list[tuple[str, str, Path, str, str, Path | None]]:
    if preset == "tutorials":
        return [
            (
                "notebooks/tutorials/*.ipynb",
                "results/plots/notebooks",
                TUTORIAL_DOC,
                "Tutorial Results",
                (
                    "This generated page displays the embedded plots and text "
                    "outputs from every tutorial notebook."
                ),
                None,
            )
        ]
    if preset == "real-examples":
        return [
            (
                "notebooks/real_examples/*.ipynb",
                "results/plots/real_examples",
                REAL_EXAMPLES_DOC,
                "Real-Example Results",
                (
                    "This generated page displays embedded setup schematics, "
                    "diagnostic plots, and text outputs from every "
                    "real-example notebook."
                ),
                REAL_EXAMPLES_MANIFEST,
            )
        ]
    if preset == "all":
        return _preset_args("tutorials") + _preset_args("real-examples")
    raise ValueError(f"unknown preset: {preset}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract embedded notebook outputs into result artefacts.",
    )
    parser.add_argument(
        "--preset",
        choices=["tutorials", "real-examples", "all"],
        help="Use the repository's standard notebook/result-page locations.",
    )
    parser.add_argument(
        "--notebook-glob",
        default=DEFAULT_NOTEBOOK_GLOB,
        help=f"Notebook glob to read. Defaults to {DEFAULT_NOTEBOOK_GLOB!r}.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write PNG files. Defaults to {DEFAULT_OUTPUT_DIR!r}.",
    )
    parser.add_argument(
        "--doc-output",
        type=Path,
        help="Optional Markdown page to regenerate from notebook outputs.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional CSV plot manifest to regenerate.",
    )
    parser.add_argument(
        "--write-docs",
        action="store_true",
        help="Regenerate the configured Markdown result page(s).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute notebooks in place before extracting embedded outputs.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not remove existing PNG files from the output directory first.",
    )
    args = parser.parse_args()

    groups = (
        _preset_args(args.preset)
        if args.preset
        else [
            (
                args.notebook_glob,
                args.output_dir,
                args.doc_output,
                "Notebook Results",
                "This generated page displays embedded notebook outputs.",
                args.manifest,
            )
        ]
    )

    all_results: list[NotebookResult] = []
    for notebook_glob, output_dir, doc_output, title, intro, manifest in groups:
        if args.execute:
            execute_notebooks(notebook_glob)

        results = extract_notebook_plots(
            notebook_glob=notebook_glob,
            output_dir=output_dir,
            clean=not args.no_clean,
        )
        all_results.extend(results)

        if manifest is not None:
            write_plot_manifest(results, manifest)

        if args.write_docs:
            if doc_output is None:
                raise ValueError("--write-docs requires --doc-output or --preset")
            write_results_page(
                results=results,
                doc_path=doc_output,
                title=title,
                intro=intro,
                manifest_path=manifest,
            )

    plots = [plot.path for result in all_results for plot in result.plots]
    for path in plots:
        print(path)
    print(f"Wrote {len(plots)} notebook plot(s) from {len(all_results)} notebook(s).")


if __name__ == "__main__":
    main()
