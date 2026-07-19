"""Declarative research-sweep CLI commands."""

from __future__ import annotations

import argparse

from ._cli_utils import parse_float_list, parse_int_list
from .research import load_research_sweep_spec
from .research_frontier import (
    accuracy_resource_frontier_spec,
    run_accuracy_resource_frontier,
)


def cmd_research_sweep(args: argparse.Namespace) -> dict:
    """Run a supported declarative study from JSON or YAML."""
    spec = load_research_sweep_spec(args.config)
    if spec.study != "accuracy-resource-frontier":
        raise ValueError(
            "the CLI currently supports study='accuracy-resource-frontier'."
        )
    result = run_accuracy_resource_frontier(
        spec,
        output_dir=args.output_dir,
        resume=not args.no_resume,
        fail_fast=args.fail_fast,
    )
    return result.as_report()


def cmd_accuracy_resource_frontier(args: argparse.Namespace) -> dict:
    """Run the built-in accuracy-resource frontier configuration."""
    spec = accuracy_resource_frontier_spec(
        degrees=tuple(parse_int_list(args.degrees)),
        tolerances=tuple(parse_float_list(args.tolerances)),
        attempt_synthesis=args.attempt_synthesis,
    )
    result = run_accuracy_resource_frontier(
        spec,
        output_dir=args.output_dir,
        resume=not args.no_resume,
        fail_fast=args.fail_fast,
    )
    return result.as_report()


def register_research_commands(sub) -> None:
    """Register research sweep and built-in frontier commands."""
    p_sweep = sub.add_parser(
        "research-sweep",
        help="Run a declarative, resumable research experiment sweep",
    )
    p_sweep.add_argument(
        "--config",
        required=True,
        help="JSON or YAML research sweep specification.",
    )
    p_sweep.add_argument(
        "--output-dir",
        required=True,
        help="Directory for normalized config, trials, manifest, and CSV artifacts.",
    )
    p_sweep.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute trials even when matching trial reports already exist.",
    )
    p_sweep.add_argument(
        "--fail-fast",
        action="store_true",
        help="Raise the first evaluator error instead of retaining a failed trial.",
    )
    p_sweep.set_defaults(func=cmd_research_sweep)

    p_frontier = sub.add_parser(
        "accuracy-resource-frontier",
        help="Compare QSVT accuracy and logical resources across access models",
    )
    p_frontier.add_argument(
        "--output-dir",
        default="results/research/accuracy_resource_frontier",
        help="Directory for trial, frontier, Pareto, and manifest artifacts.",
    )
    p_frontier.add_argument(
        "--degrees",
        default="3,5,7",
        help="Comma-separated requested polynomial degrees.",
    )
    p_frontier.add_argument(
        "--tolerances",
        default="0.2",
        help="Comma-separated positive relative-error targets.",
    )
    p_frontier.add_argument(
        "--attempt-synthesis",
        action="store_true",
        help="Attempt phase synthesis for every polynomial component.",
    )
    p_frontier.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute trials even when matching trial reports already exist.",
    )
    p_frontier.add_argument(
        "--fail-fast",
        action="store_true",
        help="Raise the first evaluator error instead of retaining a failed trial.",
    )
    p_frontier.set_defaults(func=cmd_accuracy_resource_frontier)


__all__ = [
    "cmd_accuracy_resource_frontier",
    "cmd_research_sweep",
    "register_research_commands",
]
