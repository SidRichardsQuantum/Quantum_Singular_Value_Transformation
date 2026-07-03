"""
Hardware-oriented QSVT execution helpers.

The helpers in this module accept a caller-created PennyLane device and a
caller-supplied preparation circuit. They are intentionally narrower than the
simulator helpers in :mod:`qsvt.execution`: only finite-shot probability
measurements are returned, preflight checks run before execution, and reports
keep hardware compilation and provider submission separate from local QNode
construction.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence, Sized
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any, cast

import numpy as np
import pennylane as qml

from .block_encoding import BlockEncodingSpec, build_block_encoding_operator
from .execution import _validate_coefficients

DEFAULT_PROVIDER_PLUGIN_PACKAGES = (
    "pennylane-qiskit",
    "qiskit",
    "qiskit-ibm-runtime",
)


@dataclass(frozen=True)
class ProviderPluginReport:
    """Credential-free provider, backend, and plugin metadata for a device."""

    provider_name: str | None
    backend_name: str | None
    backend_version: str | None
    plugin_name: str | None
    plugin_version: str | None
    installed_packages: dict[str, str | None]
    is_fake_backend: bool
    native_gate_set: tuple[str, ...] | None
    min_shots: int | None
    max_shots: int | None
    metadata: dict[str, object]

    def as_report(self) -> dict[str, object]:
        """Return provider/plugin metadata as plain containers."""
        return {
            "provider_name": self.provider_name,
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "plugin_name": self.plugin_name,
            "plugin_version": self.plugin_version,
            "installed_packages": self.installed_packages,
            "is_fake_backend": self.is_fake_backend,
            "native_gate_set": self.native_gate_set,
            "min_shots": self.min_shots,
            "max_shots": self.max_shots,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class HardwareQSVTPreflightResult:
    """Device compatibility checks performed before finite-shot execution."""

    passed: bool
    device_name: str
    wire_order: tuple[Any, ...]
    shots: int | None
    operation_names: tuple[str, ...]
    measurement_names: tuple[str, ...]
    unsupported_operations: tuple[str, ...]
    unsupported_measurements: tuple[str, ...]
    checks: dict[str, bool]
    reasons: tuple[str, ...]
    metadata: dict[str, object]

    def as_report(self) -> dict[str, object]:
        """Return a JSON-safe preflight report."""
        return {
            "mode": "hardware-qsvt-preflight-report",
            "passed": self.passed,
            "device_name": self.device_name,
            "wire_order": self.wire_order,
            "shots": self.shots,
            "operation_names": self.operation_names,
            "measurement_names": self.measurement_names,
            "unsupported_operations": self.unsupported_operations,
            "unsupported_measurements": self.unsupported_measurements,
            "checks": self.checks,
            "reasons": self.reasons,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class HardwareQSVTExecutionResult:
    """Finite-shot QSVT execution result from a caller-supplied device."""

    spec: BlockEncodingSpec
    poly: np.ndarray
    wire_order: tuple[Any, ...]
    projector_source: str
    angle_solver: str
    device_name: str
    shots: int
    succeeded: bool
    preflight: HardwareQSVTPreflightResult
    probabilities: np.ndarray | None
    logical_probabilities: np.ndarray | None
    logical_success_probability: float | None
    logical_success_standard_error: float | None
    maximum_probability_standard_error: float | None
    probability_normalization_error: float | None
    resource_summary: dict[str, object]
    error_type: str | None = None
    error: str | None = None

    def as_report(self) -> dict[str, object]:
        """Return execution data with hardware claim boundaries."""
        return {
            "schema_name": "hardware-qsvt-execution",
            "schema_version": "1.0",
            "mode": "hardware-qsvt-execution-report",
            "implementation_kind": "pennylane-device-finite-shot-qsvt-execution",
            "is_end_to_end_quantum_algorithm": False,
            "succeeded": self.succeeded,
            "block_encoding_spec": self.spec.as_report(),
            "poly": self.poly,
            "wire_order": self.wire_order,
            "projector_source": self.projector_source,
            "angle_solver": self.angle_solver,
            "device_name": self.device_name,
            "shots": self.shots,
            "preflight": self.preflight.as_report(),
            "probabilities": self.probabilities,
            "logical_probabilities": self.logical_probabilities,
            "logical_success_probability": self.logical_success_probability,
            "logical_success_standard_error": self.logical_success_standard_error,
            "maximum_probability_standard_error": (
                self.maximum_probability_standard_error
            ),
            "probability_normalization_error": self.probability_normalization_error,
            "resource_summary": self.resource_summary,
            "error_type": self.error_type,
            "error": self.error,
            "truth_contract": {
                "implemented_components": [
                    "caller_supplied_pennylane_device",
                    "caller_supplied_preparation_circuit",
                    "caller_selected_block_encoding_access_model",
                    "finite_shot_probability_measurement",
                    "preflight_device_compatibility_checks",
                    "logical_circuit_resource_summary",
                ],
                "omitted_quantum_components": [
                    "provider_account_or_credential_management",
                    "paid_hardware_submission_limits",
                    "provider_job_persistence",
                    "provider_native_compilation",
                    "provider_calibration_capture",
                    "error_mitigation",
                    "application_specific_readout_or_tomography",
                    "fault_tolerant_synthesis",
                ],
                "validation_scope": (
                    "The helper verifies local QNode construction and finite-shot "
                    "probability execution on the supplied device. It does not "
                    "claim scalability, fault tolerance, or quantum advantage."
                ),
            },
        }


@dataclass(frozen=True)
class HardwareQSVTCircuitReport:
    """Non-executing logical/decomposed hardware circuit audit report."""

    spec: BlockEncodingSpec
    poly: np.ndarray
    wire_order: tuple[Any, ...]
    projector_source: str
    angle_solver: str
    device_name: str
    shots: int
    provider_plugin: ProviderPluginReport
    preflight: HardwareQSVTPreflightResult
    logical_operations: tuple[str, ...]
    decomposed_operations: tuple[str, ...]
    measurements: tuple[str, ...]
    unsupported_logical_operations: tuple[str, ...]
    unsupported_decomposed_operations: tuple[str, ...]
    logical_resource_summary: dict[str, object]
    decomposed_resource_summary: dict[str, object]
    decomposition_status: str
    decomposition_error_type: str | None = None
    decomposition_error: str | None = None

    def as_report(self) -> dict[str, object]:
        """Return a JSON-safe non-execution hardware circuit report."""
        return {
            "schema_name": "hardware-qsvt-circuit",
            "schema_version": "1.0",
            "mode": "hardware-qsvt-circuit-report",
            "implementation_kind": "pennylane-hardware-qsvt-circuit-audit",
            "executed": False,
            "is_end_to_end_quantum_algorithm": False,
            "block_encoding_spec": self.spec.as_report(),
            "poly": self.poly,
            "wire_order": self.wire_order,
            "projector_source": self.projector_source,
            "angle_solver": self.angle_solver,
            "device_name": self.device_name,
            "shots": self.shots,
            "provider_plugin": self.provider_plugin.as_report(),
            "preflight": self.preflight.as_report(),
            "logical_operations": self.logical_operations,
            "decomposed_operations": self.decomposed_operations,
            "measurements": self.measurements,
            "unsupported_logical_operations": self.unsupported_logical_operations,
            "unsupported_decomposed_operations": (
                self.unsupported_decomposed_operations
            ),
            "logical_resource_summary": self.logical_resource_summary,
            "decomposed_resource_summary": self.decomposed_resource_summary,
            "decomposition_status": self.decomposition_status,
            "decomposition_error_type": self.decomposition_error_type,
            "decomposition_error": self.decomposition_error,
            "truth_contract": {
                "implemented_components": [
                    "logical_circuit_construction",
                    "pennylane_decomposition_attempt",
                    "provider_or_fake_backend_metadata_capture",
                    "native_operation_compatibility_check",
                ],
                "omitted_quantum_components": [
                    "circuit_execution",
                    "provider_job_submission",
                    "provider_native_compilation",
                    "provider_calibration_capture",
                    "error_mitigation",
                    "fault_tolerant_synthesis",
                ],
            },
        }


def qsvt_hardware_preflight(
    spec: BlockEncodingSpec,
    poly: Iterable[float],
    preparation: Callable[[], None],
    device: qml.devices.Device,
    *,
    wire_order: Iterable[Any] | None = None,
    projectors: Sequence[qml.operation.Operator] | None = None,
    angle_solver: str = "root-finding",
    shots: int,
    allow_stateprep: bool = False,
) -> HardwareQSVTPreflightResult:
    """
    Check whether a hardware-oriented QSVT circuit is suitable for a device.

    The checks are conservative and local: they verify finite-shot intent,
    wire coverage, basic device operation/measurement support when advertised,
    and reject ``StatePrep`` by default so hardware examples use explicit
    preparation circuits.
    """
    _validate_hardware_inputs(spec, preparation, device)
    coeffs = _validate_coefficients(poly)
    resolved_shots = _validate_hardware_shots(shots)
    order = _resolve_hardware_wire_order(spec, device, wire_order)
    operation_factory, projector_source = _hardware_operation_factory(
        spec,
        coeffs,
        projectors=projectors,
        angle_solver=angle_solver,
    )
    tape = _make_hardware_tape(preparation, operation_factory, order)
    operation_names = tuple(op.name for op in tape.operations)
    measurement_names = tuple(
        measurement.__class__.__name__ for measurement in tape.measurements
    )
    unsupported_ops = _unsupported_operations(device, operation_names)
    unsupported_measurements = _unsupported_measurements(device, measurement_names)
    provider_report = qsvt_provider_plugin_report(device)
    shots_within_limits = _shots_within_limits(resolved_shots, provider_report)
    rejected_stateprep = not allow_stateprep and any(
        name in {"StatePrep", "QubitStateVector"} for name in operation_names
    )

    checks = {
        "finite_shots": True,
        "device_wires_cover_circuit": _device_covers_wires(device, order),
        "preparation_is_callable": callable(preparation),
        "shots_within_device_limits": shots_within_limits,
        "stateprep_allowed": not rejected_stateprep,
        "operations_supported_when_known": not unsupported_ops,
        "measurements_supported_when_known": not unsupported_measurements,
    }
    reasons = []
    if not checks["device_wires_cover_circuit"]:
        reasons.append("device wires do not cover the requested wire_order.")
    if not shots_within_limits:
        reasons.append(
            "shots must satisfy device limits: "
            f"min={provider_report.min_shots}, max={provider_report.max_shots}."
        )
    if rejected_stateprep:
        reasons.append(
            "StatePrep-style operations are rejected by default for hardware paths."
        )
    if unsupported_ops:
        reasons.append(
            "device reports unsupported operations: " + ", ".join(unsupported_ops)
        )
    if unsupported_measurements:
        reasons.append(
            "device reports unsupported measurements: "
            + ", ".join(unsupported_measurements)
        )
    passed = all(checks.values())
    return HardwareQSVTPreflightResult(
        passed=passed,
        device_name=_device_name(device),
        wire_order=tuple(order),
        shots=resolved_shots,
        operation_names=operation_names,
        measurement_names=measurement_names,
        unsupported_operations=unsupported_ops,
        unsupported_measurements=unsupported_measurements,
        checks=checks,
        reasons=tuple(reasons),
        metadata={
            "projector_source": projector_source,
            "block_encoding_kind": spec.kind,
            "block_encoding_method": spec.block_encoding,
            "logical_shape": spec.logical_shape,
            "device_wire_count": _safe_len(getattr(device, "wires", ())),
            "provider_plugin": provider_report.as_report(),
        },
    )


def execute_qsvt_on_device(
    spec: BlockEncodingSpec,
    poly: Iterable[float],
    preparation: Callable[[], None],
    device: qml.devices.Device,
    *,
    wire_order: Iterable[Any] | None = None,
    projectors: Sequence[qml.operation.Operator] | None = None,
    angle_solver: str = "root-finding",
    shots: int,
    allow_stateprep: bool = False,
    raise_on_failure: bool = False,
) -> HardwareQSVTExecutionResult:
    """
    Execute finite-shot QSVT on a caller-supplied PennyLane device.

    The caller is responsible for creating the device, configuring provider
    credentials outside the package, and supplying a hardware-compatible
    preparation circuit. The helper returns probabilities only; it never
    requests a statevector from the device.
    """
    _validate_hardware_inputs(spec, preparation, device)
    coeffs = _validate_coefficients(poly)
    resolved_shots = _validate_hardware_shots(shots)
    order = _resolve_hardware_wire_order(spec, device, wire_order)
    operation_factory, projector_source = _hardware_operation_factory(
        spec,
        coeffs,
        projectors=projectors,
        angle_solver=angle_solver,
    )
    preflight = qsvt_hardware_preflight(
        spec,
        coeffs,
        preparation,
        device,
        wire_order=order,
        projectors=projectors,
        angle_solver=angle_solver,
        shots=resolved_shots,
        allow_stateprep=allow_stateprep,
    )
    base_summary = _hardware_resource_summary(
        spec,
        coeffs,
        preparation,
        operation_factory,
        device,
        order,
        shots=resolved_shots,
        projector_source=projector_source,
    )

    probabilities: np.ndarray | None = None
    logical_probabilities: np.ndarray | None = None
    logical_success_probability: float | None = None
    logical_success_standard_error: float | None = None
    maximum_probability_standard_error: float | None = None
    probability_normalization_error: float | None = None
    error_type: str | None = None
    error: str | None = None

    if not preflight.passed:
        error_type = "HardwarePreflightError"
        error = "; ".join(preflight.reasons) or "hardware preflight failed."
        if raise_on_failure:
            raise RuntimeError(error)
    else:
        try:
            probabilities = _execute_hardware_probability_qnode(
                preparation,
                operation_factory,
                device,
                order,
                shots=resolved_shots,
            )
            rows = spec.logical_shape[0]
            logical_probabilities = probabilities[:rows]
            logical_success_probability = float(np.sum(logical_probabilities))
            logical_success_standard_error = float(
                np.sqrt(
                    logical_success_probability
                    * max(0.0, 1.0 - logical_success_probability)
                    / resolved_shots
                )
            )
            maximum_probability_standard_error = float(
                np.max(np.sqrt(probabilities * (1.0 - probabilities) / resolved_shots))
            )
            probability_normalization_error = abs(float(np.sum(probabilities)) - 1.0)
        except Exception as exc:
            error_type = type(exc).__name__
            error = str(exc)
            if raise_on_failure:
                raise

    return HardwareQSVTExecutionResult(
        spec=spec,
        poly=coeffs,
        wire_order=tuple(order),
        projector_source=projector_source,
        angle_solver=str(angle_solver),
        device_name=_device_name(device),
        shots=resolved_shots,
        succeeded=error is None,
        preflight=preflight,
        probabilities=probabilities,
        logical_probabilities=logical_probabilities,
        logical_success_probability=logical_success_probability,
        logical_success_standard_error=logical_success_standard_error,
        maximum_probability_standard_error=maximum_probability_standard_error,
        probability_normalization_error=probability_normalization_error,
        resource_summary=base_summary,
        error_type=error_type,
        error=error,
    )


def qsvt_hardware_circuit_report(
    spec: BlockEncodingSpec,
    poly: Iterable[float],
    preparation: Callable[[], None],
    device: qml.devices.Device,
    *,
    wire_order: Iterable[Any] | None = None,
    projectors: Sequence[qml.operation.Operator] | None = None,
    angle_solver: str = "root-finding",
    shots: int,
    allow_stateprep: bool = False,
    max_decomposition_depth: int | None = 1,
) -> HardwareQSVTCircuitReport:
    """
    Build a non-executing hardware circuit audit report.

    The report constructs the same logical QSVT tape as
    :func:`execute_qsvt_on_device`, attempts PennyLane decomposition, and
    compares both operation lists against device-advertised native operations.
    No QNode is executed and no provider job is submitted.
    """
    _validate_hardware_inputs(spec, preparation, device)
    coeffs = _validate_coefficients(poly)
    resolved_shots = _validate_hardware_shots(shots)
    order = _resolve_hardware_wire_order(spec, device, wire_order)
    operation_factory, projector_source = _hardware_operation_factory(
        spec,
        coeffs,
        projectors=projectors,
        angle_solver=angle_solver,
    )
    provider_report = qsvt_provider_plugin_report(device)
    preflight = qsvt_hardware_preflight(
        spec,
        coeffs,
        preparation,
        device,
        wire_order=order,
        projectors=projectors,
        angle_solver=angle_solver,
        shots=resolved_shots,
        allow_stateprep=allow_stateprep,
    )
    tape = _make_hardware_tape(preparation, operation_factory, order)
    logical_operations = tuple(op.name for op in tape.operations)
    measurements = tuple(
        measurement.__class__.__name__ for measurement in tape.measurements
    )
    decomposed_operations = logical_operations
    decomposition_status = "not_attempted"
    decomposition_error_type: str | None = None
    decomposition_error: str | None = None
    try:
        decomposed_operations = tuple(
            op.name
            for op in _decompose_operations(
                tape.operations,
                max_depth=max_decomposition_depth,
            )
        )
        decomposition_status = "succeeded"
    except Exception as exc:
        decomposition_status = "failed"
        decomposition_error_type = type(exc).__name__
        decomposition_error = str(exc)

    return HardwareQSVTCircuitReport(
        spec=spec,
        poly=coeffs,
        wire_order=tuple(order),
        projector_source=projector_source,
        angle_solver=str(angle_solver),
        device_name=_device_name(device),
        shots=resolved_shots,
        provider_plugin=provider_report,
        preflight=preflight,
        logical_operations=logical_operations,
        decomposed_operations=decomposed_operations,
        measurements=measurements,
        unsupported_logical_operations=_unsupported_operations(
            device,
            logical_operations,
        ),
        unsupported_decomposed_operations=_unsupported_operations(
            device,
            decomposed_operations,
        ),
        logical_resource_summary=_operation_resource_summary(logical_operations),
        decomposed_resource_summary=_operation_resource_summary(
            decomposed_operations,
        ),
        decomposition_status=decomposition_status,
        decomposition_error_type=decomposition_error_type,
        decomposition_error=decomposition_error,
    )


def qsvt_hardware_truth_contract() -> dict[str, object]:
    """Return the claim boundary for hardware-oriented QSVT helpers."""
    return {
        "implementation_kind": "pennylane-device-finite-shot-qsvt-execution",
        "truth_status": "finite_shot_device_execution_with_preflight",
        "is_end_to_end_quantum_algorithm": False,
        "implemented_components": [
            "caller_supplied_device",
            "caller_supplied_preparation_circuit",
            "preflight_capability_checks",
            "finite_shot_probability_measurement",
            "logical_resource_summary",
        ],
        "omitted_quantum_components": [
            "provider_credentials",
            "paid_hardware_limits",
            "provider_job_lifecycle",
            "native_provider_compilation",
            "error_mitigation",
            "fault_tolerant_synthesis",
        ],
    }


def qsvt_provider_plugin_report(
    device: qml.devices.Device,
    *,
    plugin_packages: Iterable[str] = DEFAULT_PROVIDER_PLUGIN_PACKAGES,
) -> ProviderPluginReport:
    """
    Return credential-free provider and plugin metadata for a PennyLane device.

    The report is intentionally duck-typed so fake/local backends and optional
    provider plugins can expose metadata through attributes or ``capabilities``
    dictionaries without becoming required package dependencies.
    """
    capabilities = _device_capabilities(device)
    provider_name = _metadata_string(device, capabilities, "provider_name")
    backend_name = _metadata_string(device, capabilities, "backend_name")
    if backend_name is None:
        backend = getattr(device, "backend", None)
        backend_value = _call_or_value(getattr(backend, "name", None))
        backend_name = None if backend_value is None else str(backend_value)
    backend_version = _metadata_string(device, capabilities, "backend_version")
    plugin_name = _metadata_string(device, capabilities, "plugin_name")
    plugin_version = _metadata_string(device, capabilities, "plugin_version")
    native_gate_set = _metadata_string_tuple(
        _metadata_value(device, capabilities, "native_gate_set")
        or _metadata_value(device, capabilities, "basis_gates")
        or _metadata_value(device, capabilities, "operations")
    )
    min_shots = _metadata_int(device, capabilities, "min_shots")
    max_shots = _metadata_int(device, capabilities, "max_shots")
    installed_packages = {
        package: _installed_package_version(package) for package in plugin_packages
    }
    is_fake_backend = bool(
        _metadata_value(device, capabilities, "is_fake_backend")
        or _metadata_value(device, capabilities, "fake_backend")
        or (
            backend_name is not None
            and ("fake" in backend_name.lower() or "simulator" in backend_name.lower())
        )
    )
    return ProviderPluginReport(
        provider_name=provider_name,
        backend_name=backend_name,
        backend_version=backend_version,
        plugin_name=plugin_name,
        plugin_version=plugin_version,
        installed_packages=installed_packages,
        is_fake_backend=is_fake_backend,
        native_gate_set=native_gate_set,
        min_shots=min_shots,
        max_shots=max_shots,
        metadata={
            "device_name": _device_name(device),
            "metadata_source": "device-attributes-or-capabilities",
        },
    )


def _validate_hardware_inputs(
    spec: BlockEncodingSpec,
    preparation: Callable[[], None],
    device: qml.devices.Device,
) -> None:
    if not isinstance(spec, BlockEncodingSpec):
        raise TypeError("spec must be a BlockEncodingSpec.")
    if not callable(preparation):
        raise TypeError("preparation must be callable.")
    if not hasattr(device, "wires"):
        raise TypeError("device must be a PennyLane device with wires.")


def _validate_hardware_shots(shots: int | None) -> int:
    if shots is None:
        raise ValueError("hardware execution requires a positive finite shot count.")
    resolved = int(shots)
    if resolved <= 0:
        raise ValueError("shots must be positive.")
    return resolved


def _resolve_hardware_wire_order(
    spec: BlockEncodingSpec,
    device: qml.devices.Device,
    wire_order: Iterable[Any] | None,
) -> list[Any]:
    if wire_order is None:
        order = list(device.wires)
    else:
        order = list(wire_order)
    if not order:
        raise ValueError("wire_order must contain at least one wire.")
    if len(set(order)) != len(order):
        raise ValueError("wire_order must contain distinct wires.")
    if not set(spec.encoding_wires).issubset(set(order)):
        raise ValueError("wire_order must contain all encoding_wires.")
    return order


def _hardware_operation_factory(
    spec: BlockEncodingSpec,
    coeffs: np.ndarray,
    *,
    projectors: Sequence[qml.operation.Operator] | None,
    angle_solver: str,
):
    explicit_projectors = None if projectors is None else tuple(projectors)
    projector_source = (
        "pennylane-poly-to-angles"
        if explicit_projectors is None
        else "caller-supplied-projectors"
    )
    angles = (
        None
        if explicit_projectors is not None
        else np.asarray(
            qml.poly_to_angles(coeffs, "QSVT", angle_solver=angle_solver),
            dtype=float,
        )
    )

    def operation():
        block_encoding = build_block_encoding_operator(spec)
        resolved_projectors = (
            list(explicit_projectors)
            if explicit_projectors is not None
            else _projectors_from_angles(spec, cast(np.ndarray, angles))
        )
        return qml.QSVT(block_encoding, resolved_projectors)

    return operation, projector_source


def _projectors_from_angles(
    spec: BlockEncodingSpec,
    angles: np.ndarray,
) -> list[qml.operation.Operator]:
    rows, columns = spec.logical_shape
    if spec.kind == "pennylane-operator":
        source = cast(Any, spec.source)
        wires = list(spec.encoding_wires) + list(source.wires)
        dimensions = [rows] * len(angles)
    elif spec.block_encoding == "fable":
        wires = list(spec.encoding_wires)
        dimensions = [rows] * len(angles)
    else:
        wires = list(spec.encoding_wires)
        dimensions = [
            rows if index % 2 == 0 else columns for index in range(len(angles))
        ]
    return [
        qml.PCPhase(float(angle), dim=int(dimension), wires=wires)
        for angle, dimension in zip(angles, dimensions, strict=True)
    ]


def _make_hardware_tape(
    preparation: Callable[[], None],
    operation_factory: Callable[[], qml.operation.Operator],
    wire_order: list[Any],
) -> qml.tape.QuantumTape:
    with qml.tape.QuantumTape() as tape:
        preparation()
        operation_factory()
        qml.probs(wires=wire_order)
    return tape


def _execute_hardware_probability_qnode(
    preparation: Callable[[], None],
    operation_factory: Callable[[], qml.operation.Operator],
    device: qml.devices.Device,
    wire_order: list[Any],
    *,
    shots: int,
) -> np.ndarray:
    @qml.qnode(device)
    def circuit():
        preparation()
        operation_factory()
        return qml.probs(wires=wire_order)

    executable = qml.set_shots(circuit, shots=shots)
    return np.asarray(executable(), dtype=float)


def _hardware_resource_summary(
    spec: BlockEncodingSpec,
    coeffs: np.ndarray,
    preparation: Callable[[], None],
    operation_factory: Callable[[], qml.operation.Operator],
    device: qml.devices.Device,
    wire_order: list[Any],
    *,
    shots: int,
    projector_source: str,
) -> dict[str, object]:
    tape = _make_hardware_tape(preparation, operation_factory, wire_order)
    operation_names = [op.name for op in tape.operations]
    summary: dict[str, object] = {
        "device_name": _device_name(device),
        "provider_plugin": qsvt_provider_plugin_report(device).as_report(),
        "shots": shots,
        "block_encoding_kind": spec.kind,
        "block_encoding_method": spec.block_encoding,
        "normalization_alpha": spec.alpha,
        "logical_shape": spec.logical_shape,
        "encoding_wire_count": len(spec.encoding_wires),
        "total_wire_count": len(wire_order),
        "polynomial_degree": int(coeffs.size - 1),
        "projector_source": projector_source,
        "phase_count": (
            int(coeffs.size)
            if projector_source == "pennylane-poly-to-angles"
            else sum(1 for name in operation_names if name == "PCPhase")
        ),
        "signal_operator_calls": max(0, int(coeffs.size) - 1),
        "logical_num_gates": len(operation_names),
        "logical_depth": None,
        "logical_gate_types": _counts(operation_names),
        "logical_measurement_count": len(tape.measurements),
        "compiled_num_gates": None,
        "compiled_depth": None,
        "compiled_gate_types": {},
        "compilation_status": "not_requested",
        "native_gate_set": None,
        "wire_mapping": {str(wire): wire for wire in wire_order},
    }
    try:

        @qml.qnode(device)
        def circuit():
            preparation()
            operation_factory()
            return qml.probs(wires=wire_order)

        specs = qml.specs(qml.set_shots(circuit, shots=shots))()
        resources = _spec_value(specs, "resources")
        summary.update(
            {
                "logical_num_gates": _object_to_int(
                    _resource_value(
                        resources,
                        "num_gates",
                        summary["logical_num_gates"],
                    )
                ),
                "logical_depth": _optional_int(
                    _resource_value(resources, "depth", None)
                ),
                "logical_gate_types": _object_to_dict(
                    _resource_value(
                        resources,
                        "gate_types",
                        summary["logical_gate_types"],
                    )
                ),
                "logical_measurement_count": _object_len(
                    _resource_value(resources, "measurements", tape.measurements)
                ),
            }
        )
    except Exception as exc:
        summary["resource_summary_error"] = f"{type(exc).__name__}: {exc}"
    return summary


def _decompose_operations(
    operations: Sequence[qml.operation.Operator],
    *,
    max_depth: int | None,
) -> tuple[qml.operation.Operator, ...]:
    decomposed: list[qml.operation.Operator] = []
    for operation in operations:
        decomposed.extend(_decompose_operation(operation, max_depth=max_depth))
    return tuple(decomposed)


def _decompose_operation(
    operation: qml.operation.Operator,
    *,
    max_depth: int | None,
) -> tuple[qml.operation.Operator, ...]:
    if max_depth is not None and max_depth <= 0:
        return (operation,)
    if not bool(getattr(operation, "has_decomposition", False)):
        return (operation,)
    children = tuple(operation.decomposition())
    if max_depth == 1:
        return children
    next_depth = None if max_depth is None else max_depth - 1
    decomposed: list[qml.operation.Operator] = []
    for child in children:
        decomposed.extend(_decompose_operation(child, max_depth=next_depth))
    return tuple(decomposed)


def _operation_resource_summary(
    operation_names: Sequence[str],
) -> dict[str, object]:
    return {
        "num_gates": len(operation_names),
        "depth": None,
        "gate_types": _counts(operation_names),
        "operation_sequence": tuple(operation_names),
    }


def _unsupported_operations(
    device: qml.devices.Device,
    operation_names: tuple[str, ...],
) -> tuple[str, ...]:
    supported = _device_supported_names(device, "operations")
    if supported is None:
        return ()
    return tuple(sorted({name for name in operation_names if name not in supported}))


def _unsupported_measurements(
    device: qml.devices.Device,
    measurement_names: tuple[str, ...],
) -> tuple[str, ...]:
    supported = _device_supported_names(device, "measurements")
    if supported is None:
        return ()
    aliases = {"ProbabilityMP": "Probability"}
    return tuple(
        sorted(
            {
                name
                for name in measurement_names
                if name not in supported and aliases.get(name, name) not in supported
            }
        )
    )


def _device_supported_names(
    device: qml.devices.Device,
    attr: str,
) -> set[str] | None:
    names = getattr(device, attr, None)
    if names is not None:
        return {str(name) for name in names}
    capabilities = _device_capabilities(device)
    values = capabilities.get(attr)
    if values is not None:
        return _string_set(values)
    if attr == "operations":
        native_values = (
            capabilities.get("native_gate_set")
            or capabilities.get("basis_gates")
            or getattr(device, "native_gate_set", None)
            or getattr(device, "basis_gates", None)
        )
        if native_values is not None:
            return _string_set(native_values)
    return None


def _device_capabilities(device: qml.devices.Device) -> Mapping[str, object]:
    capabilities = getattr(device, "capabilities", None)
    if callable(capabilities):
        try:
            caps = capabilities()
        except TypeError:
            caps = None
        if isinstance(caps, Mapping):
            return caps
    if isinstance(capabilities, Mapping):
        return capabilities
    return {}


def _device_covers_wires(device: qml.devices.Device, wire_order: list[Any]) -> bool:
    device_wires = set(getattr(device, "wires", ()))
    return set(wire_order).issubset(device_wires)


def _device_name(device: qml.devices.Device) -> str:
    return str(getattr(device, "name", device.__class__.__name__))


def _counts(names: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    return counts


def _spec_value(specs: object, key: str, default: object | None = None) -> object:
    if hasattr(specs, key):
        return getattr(specs, key)
    if isinstance(specs, dict):
        return specs.get(key, default)
    get_value = getattr(specs, "get", None)
    if callable(get_value):
        return get_value(key, default)
    to_dict = getattr(specs, "to_dict", None)
    if callable(to_dict):
        return to_dict().get(key, default)
    return default


def _resource_value(
    resources: object,
    key: str,
    default: object | None = None,
) -> object:
    if hasattr(resources, key):
        return getattr(resources, key)
    if isinstance(resources, dict):
        return resources.get(key, default)
    get_value = getattr(resources, "get", None)
    if callable(get_value):
        return get_value(key, default)
    return default


def _object_to_int(value: object) -> int:
    return int(cast(Any, value))


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(cast(Any, value))


def _object_to_dict(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    return dict(cast(Any, value))


def _object_len(value: object) -> int:
    if isinstance(value, Sized):
        return len(value)
    return len(cast(Any, value))


def _safe_len(value: object) -> int | None:
    try:
        return _object_len(value)
    except TypeError:
        return None


def _metadata_value(
    device: qml.devices.Device,
    capabilities: Mapping[str, object],
    key: str,
) -> object | None:
    if hasattr(device, key):
        return getattr(device, key)
    return capabilities.get(key)


def _metadata_string(
    device: qml.devices.Device,
    capabilities: Mapping[str, object],
    key: str,
) -> str | None:
    value = _metadata_value(device, capabilities, key)
    if value is None:
        return None
    resolved = _call_or_value(value)
    if resolved is None:
        return None
    return str(resolved)


def _metadata_string_tuple(value: object | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in cast(Iterable[object], value))


def _string_set(value: object) -> set[str]:
    if isinstance(value, str):
        return {value}
    return {str(item) for item in cast(Iterable[object], value)}


def _metadata_int(
    device: qml.devices.Device,
    capabilities: Mapping[str, object],
    key: str,
) -> int | None:
    value = _metadata_value(device, capabilities, key)
    if value is None:
        return None
    return int(cast(Any, _call_or_value(value)))


def _call_or_value(value: object) -> object:
    if callable(value):
        return value()
    return value


def _installed_package_version(package: str) -> str | None:
    try:
        return package_version(package)
    except PackageNotFoundError:
        return None


def _shots_within_limits(shots: int, report: ProviderPluginReport) -> bool:
    if report.min_shots is not None and shots < report.min_shots:
        return False
    if report.max_shots is not None and shots > report.max_shots:
        return False
    return True


__all__ = [
    "HardwareQSVTCircuitReport",
    "HardwareQSVTExecutionResult",
    "HardwareQSVTPreflightResult",
    "ProviderPluginReport",
    "execute_qsvt_on_device",
    "qsvt_hardware_circuit_report",
    "qsvt_hardware_preflight",
    "qsvt_provider_plugin_report",
    "qsvt_hardware_truth_contract",
]
