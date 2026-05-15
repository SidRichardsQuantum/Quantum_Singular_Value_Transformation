#!/usr/bin/env python3
"""Extract embedded PNG outputs from introductory notebooks.

The introductory notebooks are the source of truth for these figures. This
script refreshes ``results/plots/notebooks/`` with stable filenames so
``RESULTS.md`` can link to committed plot artefacts.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

DEFAULT_NOTEBOOK_GLOB = "notebooks/*.ipynb"
DEFAULT_OUTPUT_DIR = "results/plots/notebooks"


def _decode_png_payload(payload: str | list[str]) -> bytes:
    if isinstance(payload, list):
        payload = "".join(payload)
    return base64.b64decode(payload)


def extract_notebook_plots(
    *,
    notebook_glob: str = DEFAULT_NOTEBOOK_GLOB,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    clean: bool = True,
) -> list[Path]:
    notebooks = sorted(Path().glob(notebook_glob))
    if not notebooks:
        raise FileNotFoundError(f"no notebooks matched {notebook_glob!r}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    if clean:
        for old_plot in destination.glob("*.png"):
            old_plot.unlink()

    written: list[Path] = []
    for notebook_path in notebooks:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        plot_index = 0

        for cell in notebook.get("cells", []):
            for output in cell.get("outputs", []):
                png_payload = output.get("data", {}).get("image/png")
                if not png_payload:
                    continue

                plot_index += 1
                plot_path = (
                    destination / f"{notebook_path.stem}-plot-{plot_index:02d}.png"
                )
                plot_path.write_bytes(_decode_png_payload(png_payload))
                written.append(plot_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract embedded PNG plots from introductory notebooks.",
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
        "--no-clean",
        action="store_true",
        help="Do not remove existing PNG files from the output directory first.",
    )
    args = parser.parse_args()

    written = extract_notebook_plots(
        notebook_glob=args.notebook_glob,
        output_dir=args.output_dir,
        clean=not args.no_clean,
    )

    for path in written:
        print(path)
    print(f"Wrote {len(written)} notebook plot(s).")


if __name__ == "__main__":
    main()
