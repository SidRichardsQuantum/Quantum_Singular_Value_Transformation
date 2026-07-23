"""CLI commands for planning, tolerance search, and flagship workflows."""

from __future__ import annotations

import argparse
from functools import reduce

import numpy as np
import pennylane as qml

from ._cli_utils import (
    add_report_output_args,
    parse_complex_list,
    parse_float_list,
    parse_matrix,
)
from .algorithms import hamiltonian_simulation_workflow
from .degree import search_design_degree
from .flagship import poisson_qsvt_workflow, spectral_filter_qsvt_workflow
from .planning import (
    QSVTExecutionConfig,
    QSVTProblemSpec,
    QSVTTransformSpec,
    plan_qsvt,
    run_qsvt_plan,
)


def cmd_plan_qsvt(args: argparse.Namespace) -> dict[str, object]:
    """Plan, and optionally execute, an accuracy-driven finite workflow."""
    parameters = {
        name: value
        for name, value in {
            "gamma": args.gamma,
            "lower": args.lower,
            "upper": args.upper,
            "cutoff": args.cutoff,
            "time": args.time,
            "omega": args.omega,
            "eta": args.eta,
            "width": args.width,
            "center": args.center,
            "sharpness": args.sharpness,
            "num_points": args.num_points,
            "bounded_num_points": args.bounded_num_points,
        }.items()
        if value is not None
    }
    problem = QSVTProblemSpec(
        np.asarray(parse_matrix(args.matrix)),
        rhs=_optional_vector(args.rhs),
        state=_optional_vector(args.state),
        source=_optional_vector(args.source),
        name=args.name,
    )
    transform = QSVTTransformSpec(
        args.target,
        tolerance=args.tolerance,
        min_degree=args.min_degree,
        max_degree=args.max_degree,
        degree_step=args.degree_step,
        degree=args.degree,
        parameters=parameters,
    )
    config = QSVTExecutionConfig(
        execute=args.execute,
        device_name=args.device,
        shots=args.shots,
        block_encoding=args.block_encoding,
    )
    plan = plan_qsvt(problem, transform, config)
    return run_qsvt_plan(plan).as_report() if args.execute else plan.as_report()


def cmd_spectral_filter_qsvt(args: argparse.Namespace) -> dict[str, object]:
    """Run the Pauli-Hamiltonian spectral-filter flagship."""
    operator = _parse_pauli_hamiltonian(args.pauli_terms)
    state = np.asarray(parse_complex_list(args.state), dtype=complex)
    result = spectral_filter_qsvt_workflow(
        operator,
        state,
        lower=args.lower,
        upper=args.upper,
        tolerance=args.tolerance,
        min_degree=args.min_degree,
        max_degree=args.max_degree,
        degree_step=args.degree_step,
        sharpness=args.sharpness,
        block_encoding=args.block_encoding,
        execute=args.execute,
        device_name=args.device,
        shots=args.shots,
        num_points=args.num_points,
    )
    return result.as_report()


def cmd_poisson_qsvt(args: argparse.Namespace) -> dict[str, object]:
    """Run the Poisson linear-system flagship."""
    source = None if args.source is None else np.asarray(parse_float_list(args.source))
    result = poisson_qsvt_workflow(
        args.n_points,
        length=args.length,
        source=source,
        tolerance=args.tolerance,
        min_degree=args.min_degree,
        max_degree=args.max_degree,
        degree_step=args.degree_step,
        access_model=args.access_model,
        execute=args.execute,
        device_name=args.device,
        shots=args.shots,
        num_points=args.num_points,
    )
    return result.as_report()


def cmd_hamiltonian_simulation(args: argparse.Namespace) -> dict[str, object]:
    """Run the finite coherent-QSVT Hamiltonian-simulation flagship."""
    result = hamiltonian_simulation_workflow(
        np.asarray(parse_matrix(args.matrix), dtype=complex),
        np.asarray(parse_complex_list(args.state), dtype=complex),
        time=args.time,
        degree=args.degree,
        num_points=args.num_points,
        acceptance_tolerance=args.acceptance_tolerance,
        phase_reconstruction_tolerance=args.phase_reconstruction_tolerance,
        execute_qsvt=args.execute,
        block_encoding=args.block_encoding,
        device_name=args.device,
        shots=args.shots,
    )
    return result.as_report()


def cmd_degree_search(args: argparse.Namespace) -> dict[str, object]:
    """Search a public design target for the requested approximation error."""
    kwargs = {
        name: value
        for name, value in {
            "gamma": args.gamma,
            "a": args.a,
            "alpha": args.alpha,
            "cutoff": args.cutoff,
            "lower": args.lower,
            "upper": args.upper,
            "sharpness": args.sharpness,
            "num_points": args.num_points,
            "bounded_num_points": args.bounded_num_points,
        }.items()
        if value is not None
    }
    return search_design_degree(
        args.kind,
        tolerance=args.tolerance,
        min_degree=args.min_degree,
        max_degree=args.max_degree,
        degree_step=args.degree_step,
        **kwargs,
    ).as_report()


def register_flagship_commands(
    sub: argparse._SubParsersAction,
    problem_targets: list[str],
    design_kinds: list[str],
) -> None:
    """Register accuracy-driven and flagship commands."""
    p_plan = sub.add_parser(
        "plan-workflow",
        help="Plan and optionally execute a tolerance-driven QSVT workflow",
    )
    p_plan.add_argument("--target", choices=problem_targets, required=True)
    p_plan.add_argument("--matrix", required=True)
    p_plan.add_argument("--rhs", default=None)
    p_plan.add_argument("--state", default=None)
    p_plan.add_argument("--source", default=None)
    p_plan.add_argument("--name", default="cli-qsvt-problem")
    _add_accuracy_args(p_plan)
    _add_transform_args(p_plan)
    _add_execution_args(p_plan, block_choices=("embedding", "fable"))
    p_plan.add_argument("--degree", type=int, default=None)
    add_report_output_args(p_plan)
    p_plan.set_defaults(func=cmd_plan_qsvt)

    p_filter = sub.add_parser(
        "spectral-filter-qsvt",
        help="Run a Pauli-LCU spectral-filter QSVT workflow",
    )
    p_filter.add_argument(
        "--pauli-terms",
        required=True,
        help='Comma-separated coefficient:word terms, e.g. "0.4:ZI,0.3:IZ".',
    )
    p_filter.add_argument("--state", required=True)
    p_filter.add_argument("--lower", type=float, required=True)
    p_filter.add_argument("--upper", type=float, required=True)
    p_filter.add_argument("--sharpness", type=float, default=8.0)
    _add_accuracy_args(p_filter)
    _add_execution_args(
        p_filter,
        block_choices=("prepselprep", "qubitization"),
    )
    add_report_output_args(p_filter)
    p_filter.set_defaults(func=cmd_spectral_filter_qsvt)

    p_poisson = sub.add_parser(
        "poisson-qsvt",
        help="Run the finite-difference Poisson QSVT flagship",
    )
    p_poisson.add_argument("--n-points", type=int, default=4)
    p_poisson.add_argument("--length", type=float, default=1.0)
    p_poisson.add_argument("--source", default=None)
    p_poisson.add_argument(
        "--access-model",
        choices=("dense", "fable", "prepselprep", "qubitization"),
        default="prepselprep",
    )
    _add_accuracy_args(
        p_poisson,
        tolerance=0.2,
        min_degree=3,
        max_degree=31,
        degree_step=2,
    )
    _add_execution_args(p_poisson, block_choices=None)
    add_report_output_args(p_poisson)
    p_poisson.set_defaults(func=cmd_poisson_qsvt)

    p_hamiltonian = sub.add_parser(
        "hamiltonian-simulation",
        help="Run the finite coherent-QSVT Hamiltonian-simulation flagship",
    )
    p_hamiltonian.add_argument(
        "--matrix",
        required=True,
        help='Semicolon-separated Hermitian matrix, e.g. "0,1;1,0".',
    )
    p_hamiltonian.add_argument(
        "--state",
        required=True,
        help='Comma-separated input state, e.g. "1,0".',
    )
    p_hamiltonian.add_argument("--time", type=float, required=True)
    p_hamiltonian.add_argument("--degree", type=int, required=True)
    p_hamiltonian.add_argument("--num-points", type=int, default=1001)
    p_hamiltonian.add_argument(
        "--acceptance-tolerance",
        type=float,
        default=1e-6,
        help="Maximum polynomial, state, and norm errors for stated-scope acceptance.",
    )
    p_hamiltonian.add_argument(
        "--phase-reconstruction-tolerance",
        type=float,
        default=1e-6,
    )
    _add_execution_args(
        p_hamiltonian,
        block_choices=("embedding", "fable"),
    )
    add_report_output_args(p_hamiltonian)
    p_hamiltonian.set_defaults(func=cmd_hamiltonian_simulation)

    p_degree = sub.add_parser(
        "degree-search",
        help="Choose polynomial degree from an approximation tolerance",
    )
    p_degree.add_argument("--kind", choices=design_kinds, required=True)
    _add_accuracy_args(p_degree)
    _add_transform_args(p_degree)
    add_report_output_args(p_degree)
    p_degree.set_defaults(func=cmd_degree_search)


def _add_accuracy_args(
    parser: argparse.ArgumentParser,
    *,
    tolerance: float = 0.02,
    min_degree: int = 2,
    max_degree: int = 24,
    degree_step: int = 2,
) -> None:
    parser.add_argument("--tolerance", type=float, default=tolerance)
    parser.add_argument("--min-degree", type=int, default=min_degree)
    parser.add_argument("--max-degree", type=int, default=max_degree)
    parser.add_argument("--degree-step", type=int, default=degree_step)
    parser.add_argument("--num-points", type=int, default=2001)
    parser.add_argument("--bounded-num-points", type=int, default=4001)


def _add_transform_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--a", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--cutoff", type=float, default=None)
    parser.add_argument("--lower", type=float, default=None)
    parser.add_argument("--upper", type=float, default=None)
    parser.add_argument("--sharpness", type=float, default=None)
    parser.add_argument("--width", type=float, default=None)
    parser.add_argument("--center", type=float, default=None)
    parser.add_argument("--time", type=float, default=None)
    parser.add_argument("--omega", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)


def _add_execution_args(
    parser: argparse.ArgumentParser,
    *,
    block_choices: tuple[str, ...] | None,
) -> None:
    parser.add_argument("--no-execute", dest="execute", action="store_false")
    parser.add_argument("--execute", dest="execute", action="store_true")
    parser.set_defaults(execute=True)
    parser.add_argument("--device", default="default.qubit")
    parser.add_argument("--shots", type=int, default=None)
    if block_choices is not None:
        parser.add_argument(
            "--block-encoding",
            choices=block_choices,
            default=block_choices[0],
        )


def _optional_vector(text: str | None) -> np.ndarray | None:
    return None if text is None else np.asarray(parse_complex_list(text))


def _parse_pauli_hamiltonian(text: str) -> qml.operation.Operator:
    coefficients: list[float] = []
    words: list[str] = []
    for item in text.split(","):
        coefficient, separator, word = item.strip().partition(":")
        if not separator or not word:
            raise ValueError("Pauli terms must use coefficient:word syntax.")
        coefficients.append(float(coefficient))
        words.append(word.strip().upper())
    if not words or len({len(word) for word in words}) != 1:
        raise ValueError("Pauli words must be non-empty and have equal length.")
    operators = [_pauli_word(word) for word in words]
    return qml.dot(coefficients, operators)


def _pauli_word(word: str) -> qml.operation.Operator:
    factors = []
    for wire, letter in enumerate(word):
        if letter == "I":
            continue
        gate = {"X": qml.X, "Y": qml.Y, "Z": qml.Z}.get(letter)
        if gate is None:
            raise ValueError("Pauli words may contain only I, X, Y, and Z.")
        factors.append(gate(wire))
    if not factors:
        return qml.I(0)
    return reduce(lambda left, right: left @ right, factors)


__all__ = [
    "cmd_degree_search",
    "cmd_plan_qsvt",
    "cmd_poisson_qsvt",
    "cmd_spectral_filter_qsvt",
    "register_flagship_commands",
]
