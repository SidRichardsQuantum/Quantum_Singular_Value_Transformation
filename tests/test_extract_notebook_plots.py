import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "extract_notebook_plots.py"
)
SPEC = importlib.util.spec_from_file_location("extract_notebook_plots", SCRIPT_PATH)
assert SPEC is not None
extract_notebook_plots = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = extract_notebook_plots
SPEC.loader.exec_module(extract_notebook_plots)

NotebookResult = extract_notebook_plots.NotebookResult
PlotOutput = extract_notebook_plots.PlotOutput
_doc_image_width = extract_notebook_plots._doc_image_width
_format_text_output = extract_notebook_plots._format_text_output
write_plot_manifest = extract_notebook_plots.write_plot_manifest


def test_doc_image_width_expands_wide_plots():
    wide = PlotOutput(
        path=Path("wide.png"),
        cell_index=3,
        plot_index=1,
        width_px=1100,
        height_px=360,
    )
    medium = PlotOutput(
        path=Path("medium.png"),
        cell_index=4,
        plot_index=2,
        width_px=850,
        height_px=700,
    )
    compact = PlotOutput(
        path=Path("compact.png"),
        cell_index=5,
        plot_index=3,
        width_px=640,
        height_px=480,
    )

    assert _doc_image_width(wide) == 760
    assert _doc_image_width(medium) == 640
    assert _doc_image_width(compact) == 520


def test_write_plot_manifest_includes_plot_metadata(tmp_path):
    result = NotebookResult(
        path=Path("notebooks/demo.ipynb"),
        title="Demo Notebook",
        plots=[
            PlotOutput(
                path=Path("results/plots/demo-plot-01.png"),
                cell_index=7,
                plot_index=1,
                width_px=900,
                height_px=400,
            )
        ],
    )

    manifest = tmp_path / "manifest.csv"
    write_plot_manifest([result], manifest)

    assert manifest.read_text(encoding="utf-8").splitlines() == [
        "artefact,notebook,title,plot_index,cell_index,width_px,height_px,result_type,notes",
        (
            "results/plots/demo-plot-01.png,notebooks/demo.ipynb,Demo Notebook,"
            "1,7,900,400,plot,embedded notebook PNG output"
        ),
    ]


def test_format_text_output_removes_noisy_numpy_reprs():
    assert _format_text_output("np.float64(1.25)") == "1.25"
    assert _format_text_output("(np.float64(0.5), np.int64(2))") == "(0.5, 2)"
    assert _format_text_output("array([1., 2., 3.])") == "[1., 2., 3.]"
    assert _format_text_output("(array([[1, 2], [3, 4]]), np.float64(0.5))") == (
        "([[1, 2], [3, 4]], 0.5)"
    )
