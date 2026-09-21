"""Render stored package arrays. No fitting, synthesis, or metric computation."""

from __future__ import annotations

import io
import threading

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from qsvt.reports import plot_approximation_report  # noqa: E402

PLOT_LOCK = threading.Lock()


def render(runs) -> bytes:
    with PLOT_LOCK, plt.style.context("dark_background"):
        report = runs[0]["report"]
        if report.get("mode") == "hamiltonian-simulation-workflow":
            fig = _hamiltonian_plot(runs)
        elif len(runs) == 1 and "diagnostics" in report:
            fig, _ = plot_approximation_report(report["diagnostics"])
        else:
            fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), layout="constrained")
            for run in runs:
                r = run["report"]
                label = run["id"][:8]
                if "diagnostics" in r:
                    d = r["diagnostics"]
                    axes[0].plot(d["xs"], d["polynomial_values"], label=label)
                    axes[1].plot(d["xs"], d["errors"], label=label)
                    axes[0].set_ylabel("Polynomial response")
                    axes[1].set_ylabel("Signed sampled error")
                    axes[1].set_xlabel("Normalized x")
                elif "grid" in r:
                    axes[0].plot(
                        r["grid"],
                        r["polynomial_solution"],
                        "o-",
                        label=label + " polynomial",
                    )
                    if r.get("circuit_solution") is not None:
                        axes[0].plot(
                            r["grid"],
                            r["circuit_solution"],
                            "x:",
                            label=label + " QNode",
                        )
                    axes[0].set_xlabel("Position")
                    axes[0].set_ylabel("Solution u(x)")
                else:
                    axes[0].plot(
                        r["polynomial_state"], "o-", label=label + " polynomial"
                    )
                    axes[0].set_xlabel("Logical basis index")
                    axes[0].set_ylabel("Postselected state amplitude")
                if "degree_search" in r:
                    candidates = [
                        c
                        for c in r["degree_search"]["candidates"]
                        if isinstance(c["error"], (int, float))
                    ]
                    axes[1].plot(
                        [c["requested_degree"] for c in candidates],
                        [c["error"] for c in candidates],
                        "o-",
                        label=label,
                    )
                    axes[1].set_xlabel("Requested degree")
                    axes[1].set_ylabel(r["degree_search"]["metric"])
            if "diagnostics" in report:
                d = report["diagnostics"]
                axes[0].plot(
                    d["xs"],
                    d["target_values"],
                    "--",
                    color="white",
                    label="package fitting target",
                )
            elif "grid" in report:
                axes[0].plot(
                    report["grid"],
                    report["direct_solution"],
                    "--",
                    color="white",
                    label="dense direct reference",
                )
            else:
                axes[0].plot(
                    report["reference_state"],
                    "--",
                    color="white",
                    label="hard-projector reference",
                )
            for ax in axes:
                ax.legend(fontsize=8)
                ax.grid(alpha=0.15)
        fig.set_facecolor("#111b29")
        output = io.BytesIO()
        try:
            fig.savefig(
                output,
                format="png",
                dpi=130,
                metadata={"Software": "QSVT Experiment Studio"},
            )
        finally:
            plt.close(fig)
        return output.getvalue()


def _hamiltonian_plot(runs):
    """Plot real/imaginary stored amplitudes without normalizing or recomputing them."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), layout="constrained")
    for run in runs:
        report = run["report"]
        label = run["id"][:8]
        series = [("polynomial", report["evolved_state"], "o-")]
        execution = report.get("qsvt_execution") or {}
        if execution.get("logical_output") is not None:
            series.append(
                ("QNode recovered (LCU rescaled)", execution["logical_output"], "x:")
            )
        for name, values, style in series:
            for part, ax in zip(("real", "imag"), axes, strict=True):
                ax.plot(values[part], style, label=f"{label} {name}")
    for part, ax in zip(("real", "imag"), axes, strict=True):
        ax.plot(
            runs[0]["report"]["reference_state"][part],
            "--",
            color="white",
            label="dense exponential reference",
        )
        ax.set_ylabel(f"State amplitude ({part})")
        ax.set_xlabel("Site index (zero-based)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)
    axes[0].set_title("Hamiltonian evolution · complex state amplitudes")
    return fig
