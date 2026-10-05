"""Device inventory and structured per-device memory estimation services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import cast

from llama_profile_lab.db import (
    AcceleratorDeviceRepository,
    CandidateRepository,
    Database,
    EnvironmentRepository,
    MemoryEstimateRepository,
)
from llama_profile_lab.db.connection import transaction
from llama_profile_lab.db.records import BinaryRecord, MemoryEstimateAttemptStatus
from llama_profile_lab.domain import (
    AcceleratorDevice,
    MemoryEstimateOutput,
    PlacementConstraints,
)
from llama_profile_lab.execution.host import BasicHostInfo, detect_basic_host
from llama_profile_lab.execution.process import (
    ProcessResult,
    ProcessRunner,
    ProcessRunnerError,
)
from llama_profile_lab.llama import (
    BinaryKind,
    CapabilitySet,
    LlamaDeviceListAdapter,
    LlamaDeviceListError,
    MemoryEstimatorAdapter,
    MemoryEstimatorConfigurationError,
    MemoryEstimatorParseError,
    correlate_physical_devices,
    sha256_file,
)


class DeviceInventoryError(RuntimeError):
    """Raised when exact-binary logical-device discovery fails."""


class MemoryEstimatorError(RuntimeError):
    """Raised when a structured memory estimate cannot be completed."""


@dataclass(frozen=True, slots=True)
class DeviceInventoryResult:
    """One persisted exact-binary device observation."""

    host_id: str
    binary_id: str
    devices: tuple[AcceleratorDevice, ...]
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class MemoryEstimateObservation:
    """One successful memory estimate, possibly reused from cache."""

    estimate_id: str
    attempt_id: str
    output: MemoryEstimateOutput
    cache_hit: bool


class DeviceInventoryService:
    """Discover and persist logical devices exposed by one registered binary."""

    def __init__(
        self,
        database: Database,
        *,
        runner: ProcessRunner | None = None,
        adapter: LlamaDeviceListAdapter | None = None,
    ) -> None:
        self.database = database
        self.runner = runner or ProcessRunner()
        self.adapter = adapter or LlamaDeviceListAdapter()

    def inspect(
        self,
        binary_id: str,
        *,
        timeout_seconds: float | None = 30.0,
        cancel_event: Event | None = None,
    ) -> DeviceInventoryResult:
        host = detect_basic_host()
        with self.database.session() as connection:
            environment = EnvironmentRepository(connection)
            binary = environment.get_binary(binary_id)
            if binary is None:
                raise DeviceInventoryError(f"binary not found: {binary_id}")
            _verify_registered_binary(binary)
            host_id = _persist_host(environment, host)
            capabilities = _capabilities(binary)
            try:
                argv = self.adapter.build_argv(
                    binary_path=Path(binary.path),
                    capabilities=capabilities,
                )
            except LlamaDeviceListError as exc:
                raise DeviceInventoryError(str(exc)) from exc

        try:
            process = self.runner.run(
                argv,
                timeout_seconds=timeout_seconds,
                cancel_event=cancel_event,
            )
        except ProcessRunnerError as exc:
            raise DeviceInventoryError(str(exc)) from exc

        if process.interrupted:
            raise DeviceInventoryError("device inventory was interrupted")
        if process.cancelled:
            raise DeviceInventoryError("device inventory was cancelled")
        if process.timed_out:
            raise DeviceInventoryError("device inventory timed out")
        if process.exit_code != 0:
            raise DeviceInventoryError(
                f"device inventory exited with code {process.exit_code}"
            )

        try:
            inventory = self.adapter.parse_output(process.stdout, process.stderr)
        except LlamaDeviceListError as exc:
            raise DeviceInventoryError(str(exc)) from exc
        devices = correlate_physical_devices(inventory.devices, host.gpus)
        raw_output = "\n".join(
            part for part in (process.stdout, process.stderr) if part
        )
        with self.database.session() as connection:
            AcceleratorDeviceRepository(connection).put_inventory(
                host_id=host_id,
                binary_id=binary_id,
                devices=devices,
                raw_output=raw_output,
            )
        return DeviceInventoryResult(
            host_id=host_id,
            binary_id=binary_id,
            devices=devices,
            stdout=process.stdout,
            stderr=process.stderr,
        )


class MemoryEstimatorService:
    """Execute, persist, and cache structured per-device memory estimates."""

    def __init__(
        self,
        database: Database,
        *,
        runner: ProcessRunner | None = None,
        adapter: MemoryEstimatorAdapter | None = None,
    ) -> None:
        self.database = database
        self.runner = runner or ProcessRunner()
        self.adapter = adapter or MemoryEstimatorAdapter()

    def estimate(
        self,
        candidate_id: str,
        *,
        helper_binary_id: str,
        model_path: Path,
        selected_devices: tuple[str, ...] | None = None,
        placement_constraints: PlacementConstraints | None = None,
        timeout_seconds: float | None = 300.0,
        cancel_event: Event | None = None,
    ) -> MemoryEstimateObservation:
        host = detect_basic_host()
        with self.database.session() as connection:
            candidates = CandidateRepository(connection)
            candidate = candidates.get(candidate_id)
            if candidate is None:
                raise MemoryEstimatorError(f"Candidate not found: {candidate_id}")
            environment = EnvironmentRepository(connection)
            helper = environment.get_binary(helper_binary_id)
            if helper is None:
                raise MemoryEstimatorError(
                    f"memory estimator binary not found: {helper_binary_id}"
                )
            try:
                _verify_registered_binary(
                    helper,
                    expected_kind="llama-memory-estimator",
                    verify_hash=False,
                )
            except DeviceInventoryError as exc:
                raise MemoryEstimatorError(str(exc)) from exc
            host_id = _persist_host(environment, host)
            capabilities = _capabilities(helper)
            try:
                invocation = self.adapter.build_invocation(
                    binary_path=Path(helper.path),
                    capabilities=capabilities,
                    helper_sha256=helper.sha256,
                    host_id=host_id,
                    model_path=model_path,
                    candidate=candidate,
                    placement_constraints=placement_constraints,
                    selected_devices=selected_devices,
                )
            except MemoryEstimatorConfigurationError as exc:
                raise MemoryEstimatorError(str(exc)) from exc

            estimates = MemoryEstimateRepository(connection)
            actual_sha256 = sha256_file(Path(helper.path))
            if actual_sha256 != helper.sha256:
                attempt_id = estimates.create_attempt(
                    identity=invocation.identity,
                    candidate_id=candidate_id,
                    host_id=host_id,
                    helper_binary_id=helper_binary_id,
                    model_artifact_id=candidate.model.target_model_id,
                    argv=invocation.argv,
                )
                message = (
                    "registered memory estimator binary changed on disk; "
                    "re-run binary inspect"
                )
                estimates.finish_attempt(
                    attempt_id,
                    status="binary_changed",
                    duration_ns=0,
                    exit_code=None,
                    stdout="",
                    stderr=message,
                    failure_details={
                        "expected_sha256": helper.sha256,
                        "actual_sha256": actual_sha256,
                    },
                )
                raise MemoryEstimatorError(message)

            cached = estimates.find_by_cache_hash(
                invocation.identity.content_hash()
            )
            if cached is not None:
                output = estimates.get_result(cached.id)
                if output is None:
                    raise MemoryEstimatorError(
                        f"cached memory estimate is incomplete: {cached.id}"
                    )
                return MemoryEstimateObservation(
                    estimate_id=cached.id,
                    attempt_id=cached.attempt_id,
                    output=output,
                    cache_hit=True,
                )

            attempt_id = estimates.create_attempt(
                identity=invocation.identity,
                candidate_id=candidate_id,
                host_id=host_id,
                helper_binary_id=helper_binary_id,
                model_artifact_id=candidate.model.target_model_id,
                argv=invocation.argv,
            )

        try:
            process = self.runner.run(
                invocation.argv,
                timeout_seconds=timeout_seconds,
                cancel_event=cancel_event,
            )
        except ProcessRunnerError as exc:
            self._finish_failure(
                attempt_id,
                status="failed",
                duration_ns=0,
                exit_code=None,
                stdout="",
                stderr=str(exc),
                details={"error": str(exc)},
            )
            raise MemoryEstimatorError(str(exc)) from exc

        failure_status = _process_failure_status(process)
        if failure_status is not None:
            self._finish_failure(
                attempt_id,
                status=failure_status,
                duration_ns=process.duration_ns,
                exit_code=process.exit_code,
                stdout=process.stdout,
                stderr=process.stderr,
                details={"forced_kill": process.forced_kill},
                finished_at=process.finished_at,
            )
            raise MemoryEstimatorError(
                _process_failure_message(failure_status, process.exit_code)
            )

        try:
            output = self.adapter.parse_output(process.stdout)
            if (
                invocation.identity.selected_devices
                and output.resolved.devices
                != invocation.identity.selected_devices
            ):
                raise MemoryEstimatorParseError(
                    "resolved device order does not match the requested device order"
                )
        except MemoryEstimatorParseError as exc:
            self._finish_failure(
                attempt_id,
                status="parser_failed",
                duration_ns=process.duration_ns,
                exit_code=process.exit_code,
                stdout=process.stdout,
                stderr=process.stderr,
                details={"error": str(exc)},
                finished_at=process.finished_at,
            )
            raise MemoryEstimatorError(str(exc)) from exc

        with self.database.session() as connection:
            estimates = MemoryEstimateRepository(connection)
            with transaction(connection, immediate=True):
                estimates.finish_attempt(
                    attempt_id,
                    status="completed",
                    duration_ns=process.duration_ns,
                    exit_code=process.exit_code,
                    stdout=process.stdout,
                    stderr=process.stderr,
                    finished_at=process.finished_at,
                )
                estimate_id = estimates.put_success(
                    attempt_id=attempt_id,
                    identity=invocation.identity,
                    result=output,
                    candidate_id=candidate_id,
                    host_id=host_id,
                    helper_binary_id=helper_binary_id,
                    model_artifact_id=candidate.model.target_model_id,
                )
        return MemoryEstimateObservation(
            estimate_id=estimate_id,
            attempt_id=attempt_id,
            output=output,
            cache_hit=False,
        )

    def _finish_failure(
        self,
        attempt_id: str,
        *,
        status: MemoryEstimateAttemptStatus,
        duration_ns: int,
        exit_code: int | None,
        stdout: str,
        stderr: str,
        details: dict[str, object],
        finished_at: str | None = None,
    ) -> None:
        with self.database.session() as connection:
            MemoryEstimateRepository(connection).finish_attempt(
                attempt_id,
                status=status,
                duration_ns=duration_ns,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                failure_details=details,
                finished_at=finished_at,
            )


def _persist_host(
    repository: EnvironmentRepository,
    host: BasicHostInfo,
) -> str:
    return repository.put_host(
        hostname=host.hostname,
        hardware_fingerprint=host.hardware_fingerprint,
        cpu=host.cpu,
        ram_bytes=host.ram_bytes,
        gpus=host.gpus,
        os_info=host.os_info,
    )


def _verify_registered_binary(
    binary: BinaryRecord,
    *,
    expected_kind: str | None = None,
    verify_hash: bool = True,
) -> None:
    if expected_kind is not None and binary.kind != expected_kind:
        raise DeviceInventoryError(
            f"binary {binary.id} is {binary.kind}, not {expected_kind}"
        )
    path = Path(binary.path)
    if not path.is_file():
        raise DeviceInventoryError(
            f"registered binary path does not exist: {path}"
        )
    if verify_hash and sha256_file(path) != binary.sha256:
        raise DeviceInventoryError(
            "registered binary changed on disk; re-run binary inspect"
        )


def _capabilities(binary: BinaryRecord) -> CapabilitySet:
    return CapabilitySet.from_mapping(
        dict(binary.capabilities),
        fallback_kind=_binary_kind(binary.kind),
    )


def _binary_kind(value: str) -> BinaryKind:
    allowed = {
        "llama-bench",
        "llama-fit-params",
        "llama-memory-estimator",
        "llama-server",
        "speed-bench",
    }
    if value not in allowed:
        raise DeviceInventoryError(f"unknown persisted binary kind: {value}")
    return cast(BinaryKind, value)


def _process_failure_status(
    process: ProcessResult,
) -> MemoryEstimateAttemptStatus | None:
    if process.interrupted:
        return "interrupted"
    if process.cancelled:
        return "cancelled"
    if process.timed_out:
        return "timeout"
    if process.exit_code != 0:
        return "failed"
    return None


def _process_failure_message(
    status: MemoryEstimateAttemptStatus,
    exit_code: int | None,
) -> str:
    if status == "timeout":
        return "memory estimator timed out"
    if status == "cancelled":
        return "memory estimator was cancelled"
    if status == "interrupted":
        return "memory estimator was interrupted"
    return f"memory estimator exited with code {exit_code}"
