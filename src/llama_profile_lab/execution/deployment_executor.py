"""Simultaneous multi-server deployment lifecycle orchestration."""

from __future__ import annotations

import socket
import time
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Protocol

from llama_profile_lab.db import (
    CandidateRepository,
    Database,
    DeploymentCandidateRepository,
    DeploymentPlacementRepository,
    DeploymentRunRepository,
    EnvironmentRepository,
    PlacementRepository,
)
from llama_profile_lab.db.records import (
    BinaryRecord,
    DeploymentDeviceAllocationRecord,
    DeploymentMemberStatus,
)
from llama_profile_lab.domain import (
    DeploymentFailureKind,
    GpuTelemetrySample,
)
from llama_profile_lab.execution.host import BasicHostInfo, detect_basic_host
from llama_profile_lab.execution.lock import HostLock, HostLockError
from llama_profile_lab.execution.placement import (
    PlacementConfigurationError,
    resolved_placement_from_record,
    validate_fixed_placement,
)
from llama_profile_lab.execution.server_process import (
    ManagedServerProcess,
    ServerProcessError,
    ServerProcessOutcome,
)
from llama_profile_lab.execution.telemetry import AutoGpuTelemetryProvider
from llama_profile_lab.llama import (
    CapabilitySet,
    LlamaServerAdapter,
    LlamaServerConfigurationError,
    sha256_file,
)


class DeploymentExecutionError(RuntimeError):
    """Raised after a deployment execution has been persisted as non-successful."""

    def __init__(
        self,
        message: str,
        *,
        run_id: str | None = None,
        status: str = "failed",
        failure_kind: DeploymentFailureKind | None = None,
    ) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.status = status
        self.failure_kind = failure_kind


@dataclass(frozen=True, slots=True)
class DeploymentServerInput:
    """Filesystem inputs required to launch one deployment instance."""

    instance_id: str
    model_path: Path
    draft_model_path: Path | None = None


@dataclass(frozen=True, slots=True)
class DeploymentExecutionMember:
    """One persisted member outcome returned by a successful residency run."""

    instance_id: str
    endpoint: str
    pid: int | None
    ready_at: str | None
    exit_code: int | None
    forced_kill: bool


@dataclass(frozen=True, slots=True)
class DeploymentExecutionSummary:
    """Result of one successful simultaneous-residency execution."""

    run_id: str
    deployment_candidate_id: str
    deployment_placement_id: str
    status: str
    members: tuple[DeploymentExecutionMember, ...]
    runtime_memory: Mapping[str, object]


class GpuSnapshotProvider(Protocol):
    """Minimal provider contract required for runtime headroom validation."""

    def sample(self) -> tuple[GpuTelemetrySample, ...]:
        """Return one current GPU snapshot."""


@dataclass(frozen=True, slots=True)
class _MemberPlan:
    instance_id: str
    endpoint: str
    health_url: str
    argv: tuple[str, ...]
    model_path: Path
    draft_model_path: Path | None


@dataclass(frozen=True, slots=True)
class _RuntimeMemoryCheck:
    details: dict[str, object]
    violations: tuple[dict[str, object], ...]


class DeploymentProcessRegistry:
    """Track all member processes and stop every registered process group."""

    def __init__(self) -> None:
        self._processes: dict[str, ManagedServerProcess] = {}
        self._order: list[str] = []

    def register(
        self,
        instance_id: str,
        process: ManagedServerProcess,
    ) -> None:
        if instance_id in self._processes:
            raise ValueError(f"duplicate deployment process: {instance_id}")
        self._processes[instance_id] = process
        self._order.append(instance_id)

    def get(self, instance_id: str) -> ManagedServerProcess | None:
        return self._processes.get(instance_id)

    def items(self) -> tuple[tuple[str, ManagedServerProcess], ...]:
        return tuple((item, self._processes[item]) for item in self._order)

    def stop_all(
        self,
    ) -> dict[str, tuple[ServerProcessOutcome | None, str | None]]:
        results: dict[str, tuple[ServerProcessOutcome | None, str | None]] = {}
        for instance_id in reversed(self._order):
            process = self._processes[instance_id]
            try:
                outcome = process.stop()
                results[instance_id] = (outcome, None)
            except Exception as exc:
                results[instance_id] = (None, str(exc))
            finally:
                try:
                    process.close()
                except Exception as exc:
                    outcome, error = results.get(instance_id, (None, None))
                    message = str(exc) if error is None else f"{error}; {exc}"
                    results[instance_id] = (outcome, message)
        return results


class _PortReservationPool:
    """Reserve unique local TCP ports until each corresponding process starts."""

    def __init__(self, host: str, instance_ids: tuple[str, ...]) -> None:
        self.host = host
        self._sockets: dict[str, socket.socket] = {}
        self._ports: dict[str, int] = {}
        try:
            for instance_id in instance_ids:
                sock = _bind_ephemeral(host)
                self._sockets[instance_id] = sock
                self._ports[instance_id] = int(sock.getsockname()[1])
        except BaseException:
            self.close()
            raise

    def port(self, instance_id: str) -> int:
        return self._ports[instance_id]

    def release(self, instance_id: str) -> None:
        sock = self._sockets.pop(instance_id, None)
        if sock is not None:
            sock.close()

    def close(self) -> None:
        for sock in self._sockets.values():
            sock.close()
        self._sockets.clear()


class DeploymentExecutor:
    """Launch all servers in one immutable DeploymentPlacement as one lifecycle."""

    def __init__(
        self,
        database: Database,
        *,
        server_adapter: LlamaServerAdapter | None = None,
        server_process_factory: Callable[
            [tuple[str, ...]], ManagedServerProcess
        ] | None = None,
        host_detector: Callable[[], BasicHostInfo] = detect_basic_host,
        gpu_provider: GpuSnapshotProvider | None = None,
    ) -> None:
        self.database = database
        self.server_adapter = server_adapter or LlamaServerAdapter()
        self.server_process_factory = (
            server_process_factory or ManagedServerProcess
        )
        self.host_detector = host_detector
        self.gpu_provider = gpu_provider or AutoGpuTelemetryProvider()

    def execute(
        self,
        deployment_placement_id: str,
        inputs: tuple[DeploymentServerInput, ...],
        *,
        host: str = "127.0.0.1",
        readiness_timeout_seconds: float = 300.0,
        residency_hold_seconds: float = 0.0,
        cancel_event: Event | None = None,
    ) -> DeploymentExecutionSummary:
        if not host:
            raise DeploymentExecutionError("server host must not be empty")
        if readiness_timeout_seconds <= 0:
            raise DeploymentExecutionError(
                "readiness timeout must be positive"
            )
        if residency_hold_seconds < 0:
            raise DeploymentExecutionError(
                "residency hold duration must be non-negative"
            )
        try:
            with HostLock(_lock_path(self.database)):
                return self._execute_locked(
                    deployment_placement_id,
                    inputs,
                    host=host,
                    readiness_timeout_seconds=readiness_timeout_seconds,
                    residency_hold_seconds=residency_hold_seconds,
                    cancel_event=cancel_event,
                )
        except HostLockError as exc:
            raise DeploymentExecutionError(str(exc)) from exc

    def _execute_locked(
        self,
        deployment_placement_id: str,
        inputs: tuple[DeploymentServerInput, ...],
        *,
        host: str,
        readiness_timeout_seconds: float,
        residency_hold_seconds: float,
        cancel_event: Event | None,
    ) -> DeploymentExecutionSummary:
        started_ns = time.monotonic_ns()
        started_at = _utc_now()
        host_info = self.host_detector()
        input_map = _server_input_map(inputs)

        with self.database.session() as connection:
            placements = DeploymentPlacementRepository(connection)
            deployment_placement = placements.get(deployment_placement_id)
            if deployment_placement is None:
                raise DeploymentExecutionError(
                    f"deployment placement not found: {deployment_placement_id}"
                )
            if deployment_placement.feasibility != "feasible":
                raise DeploymentExecutionError(
                    "deployment placement is not feasible"
                )
            deployment_id = deployment_placement.deployment_candidate_id
            deployment = DeploymentCandidateRepository(connection).get(
                deployment_id
            )
            if deployment is None:
                raise DeploymentExecutionError(
                    f"deployment Candidate not found: {deployment_id}"
                )
            expected = {item.instance_id for item in deployment.instances}
            if set(input_map) != expected:
                raise DeploymentExecutionError(
                    "server inputs must contain exactly the deployment instances"
                )
            environment = EnvironmentRepository(connection)
            host_id = environment.put_host(
                hostname=host_info.hostname,
                hardware_fingerprint=host_info.hardware_fingerprint,
                cpu=host_info.cpu,
                ram_bytes=host_info.ram_bytes,
                gpus=host_info.gpus,
                os_info=host_info.os_info,
            )
            if host_id != deployment_placement.host_id:
                raise DeploymentExecutionError(
                    "deployment placement belongs to a different host identity"
                )
            runs = DeploymentRunRepository(connection)
            runs.recover_orphaned()
            run_id = runs.create(
                deployment_candidate_id=deployment_id,
                deployment_placement_id=deployment_placement_id,
                status="starting",
                started_at=started_at,
            )

        registry = DeploymentProcessRegistry()
        reservations: _PortReservationPool | None = None
        member_plans: tuple[_MemberPlan, ...] = ()
        failure_kind: DeploymentFailureKind | None = None
        failure_message: str | None = None
        cancelled = False
        runtime_check = _RuntimeMemoryCheck(
            details={"validated_devices": [], "missing_devices": []},
            violations=(),
        )

        try:
            member_plans = self._prepare_members(
                deployment_placement_id,
                deployment,
                input_map,
                host_id=host_id,
                host=host,
                run_id=run_id,
            )
            reservations = _PortReservationPool(
                host,
                tuple(item.instance_id for item in member_plans),
            )
            member_plans = self._assign_reserved_ports(
                member_plans,
                reservations,
                host=host,
                run_id=run_id,
            )

            for plan in member_plans:
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    failure_message = "deployment startup cancelled"
                    break
                reservations.release(plan.instance_id)
                process = self.server_process_factory(plan.argv)
                registry.register(plan.instance_id, process)
                self._set_member_status(
                    run_id,
                    plan.instance_id,
                    "starting",
                    started_at=_utc_now(),
                )
                try:
                    process.start()
                except ServerProcessError as exc:
                    failure_kind = "server_start_failed"
                    failure_message = (
                        f"{plan.instance_id} failed to start: {exc}"
                    )
                    break
                self._set_member_status(
                    run_id,
                    plan.instance_id,
                    "starting",
                    pid=process.pid,
                )

            if failure_message is None:
                readiness_error = self._wait_ready_barrier(
                    run_id,
                    member_plans,
                    registry,
                    timeout_seconds=readiness_timeout_seconds,
                    cancel_event=cancel_event,
                )
                if readiness_error is not None:
                    instance_id, error = readiness_error
                    if error.kind == "cancelled":
                        cancelled = True
                    elif error.kind == "readiness_failed":
                        failure_kind = "member_timeout"
                    else:
                        failure_kind = "server_start_failed"
                    failure_message = f"{instance_id}: {error}"

            if failure_message is None:
                with self.database.session() as connection:
                    DeploymentRunRepository(connection).set_status(
                        run_id,
                        "ready",
                    )

                try:
                    runtime_check = self._runtime_memory_check(
                        deployment_placement_id
                    )
                except Exception as exc:
                    failure_kind = "telemetry_incomplete"
                    failure_message = (
                        f"runtime memory telemetry failed: {exc}"
                    )
                else:
                    if runtime_check.violations:
                        failure_kind = "runtime_memory_margin_violated"
                        failure_message = (
                            "runtime GPU memory headroom violated planned margin"
                        )

            if failure_message is None:
                probe_error = self._residency_probe(
                    member_plans,
                    registry,
                    residency_hold_seconds=residency_hold_seconds,
                    cancel_event=cancel_event,
                )
                if probe_error is not None:
                    instance_id, error = probe_error
                    if error.kind == "cancelled":
                        cancelled = True
                    else:
                        failure_kind = "member_crash"
                    failure_message = f"{instance_id}: {error}"
        except Exception as exc:
            failure_kind = failure_kind or "server_start_failed"
            failure_message = str(exc)
        finally:
            if reservations is not None:
                reservations.close()

        cleanup = registry.stop_all()
        cleanup_failed = any(error is not None for _, error in cleanup.values())
        if cleanup_failed and failure_message is None:
            failure_kind = "member_crash"
            failure_message = "one or more deployment members failed cleanup"

        terminal_member_status: DeploymentMemberStatus = (
            "cancelled"
            if cancelled
            else "failed"
            if failure_message is not None
            else "stopped"
        )
        self._finalize_members(
            run_id,
            member_plans,
            cleanup,
            status=terminal_member_status,
            failure_message=failure_message,
        )

        duration_ns = max(0, time.monotonic_ns() - started_ns)
        if cancelled:
            with self.database.session() as connection:
                DeploymentRunRepository(connection).finish(
                    run_id,
                    status="cancelled",
                    duration_ns=duration_ns,
                    quality_details={
                        "reason": failure_message or "cancelled",
                        "runtime_memory": runtime_check.details,
                    },
                )
            raise DeploymentExecutionError(
                failure_message or "deployment execution cancelled",
                run_id=run_id,
                status="cancelled",
            )

        if failure_message is not None:
            failure_kind = _oom_override(failure_kind, cleanup)
            with self.database.session() as connection:
                DeploymentRunRepository(connection).finish(
                    run_id,
                    status="failed",
                    duration_ns=duration_ns,
                    failure_kind=failure_kind or "member_crash",
                    failure_details={
                        "error": failure_message,
                        "runtime_memory": runtime_check.details,
                        "cleanup_failed": cleanup_failed,
                    },
                )
            raise DeploymentExecutionError(
                failure_message,
                run_id=run_id,
                status="failed",
                failure_kind=failure_kind,
            )

        with self.database.session() as connection:
            DeploymentRunRepository(connection).finish(
                run_id,
                status="completed",
                duration_ns=duration_ns,
                quality_details={"runtime_memory": runtime_check.details},
            )
            persisted_members = DeploymentRunRepository(connection).members(
                run_id
            )

        return DeploymentExecutionSummary(
            run_id=run_id,
            deployment_candidate_id=deployment_id,
            deployment_placement_id=deployment_placement_id,
            status="completed",
            members=tuple(
                DeploymentExecutionMember(
                    instance_id=item.instance_id,
                    endpoint=item.endpoint or "",
                    pid=item.pid,
                    ready_at=item.ready_at,
                    exit_code=item.exit_code,
                    forced_kill=item.forced_kill,
                )
                for item in persisted_members
            ),
            runtime_memory=runtime_check.details,
        )

    def _prepare_members(
        self,
        deployment_placement_id: str,
        deployment,
        inputs: Mapping[str, DeploymentServerInput],
        *,
        host_id: str,
        host: str,
        run_id: str,
    ) -> tuple[_MemberPlan, ...]:
        del host, run_id
        instance_placement_ids: dict[str, str]
        with self.database.session() as connection:
            placement = DeploymentPlacementRepository(connection).get(
                deployment_placement_id
            )
            if placement is None:
                raise DeploymentExecutionError(
                    f"deployment placement not found: {deployment_placement_id}"
                )
            instance_placement_ids = {
                item.instance_id: item.resolved_placement_id
                for item in placement.instance_placements
            }
            candidates = CandidateRepository(connection)
            resolved = PlacementRepository(connection)
            environment = EnvironmentRepository(connection)
            plans: list[_MemberPlan] = []
            for instance in deployment.instances:
                candidate = candidates.get(instance.candidate_id)
                if candidate is None:
                    raise DeploymentExecutionError(
                        f"Candidate not found: {instance.candidate_id}"
                    )
                binary = environment.get_binary(instance.binary_id)
                if binary is None:
                    raise DeploymentExecutionError(
                        f"server binary not found: {instance.binary_id}"
                    )
                _verify_server_binary(binary)
                server_input = inputs[instance.instance_id]
                model_path = _resolve_file(server_input.model_path, "model")
                draft_model_path = (
                    None
                    if server_input.draft_model_path is None
                    else _resolve_file(
                        server_input.draft_model_path,
                        "draft model",
                    )
                )
                resolved_record = resolved.get(
                    instance_placement_ids[instance.instance_id]
                )
                if resolved_record is None:
                    raise DeploymentExecutionError(
                        "resolved placement not found for instance "
                        f"{instance.instance_id}"
                    )
                validate_fixed_placement(
                    resolved_record,
                    candidate=candidate,
                    host_id=host_id,
                )
                plans.append(
                    _MemberPlan(
                        instance_id=instance.instance_id,
                        endpoint="",
                        health_url="",
                        argv=(),
                        model_path=model_path,
                        draft_model_path=draft_model_path,
                    )
                )
        return tuple(plans)

    def _assign_reserved_ports(
        self,
        plans: tuple[_MemberPlan, ...],
        reservations: _PortReservationPool,
        *,
        host: str,
        run_id: str,
    ) -> tuple[_MemberPlan, ...]:
        with self.database.session() as connection:
            placement = DeploymentPlacementRepository(connection).get(
                self._run_placement_id(connection, run_id)
            )
            if placement is None:
                raise DeploymentExecutionError(
                    "deployment placement disappeared during execution"
                )
            deployment = DeploymentCandidateRepository(connection).get(
                placement.deployment_candidate_id
            )
            if deployment is None:
                raise DeploymentExecutionError(
                    "deployment Candidate disappeared during execution"
                )
            candidates = CandidateRepository(connection)
            resolved_repo = PlacementRepository(connection)
            environment = EnvironmentRepository(connection)
            placement_ids = {
                item.instance_id: item.resolved_placement_id
                for item in placement.instance_placements
            }
            instances = {item.instance_id: item for item in deployment.instances}
            result: list[_MemberPlan] = []
            for plan in plans:
                instance = instances[plan.instance_id]
                candidate = candidates.get(instance.candidate_id)
                binary = environment.get_binary(instance.binary_id)
                resolved_record = resolved_repo.get(
                    placement_ids[plan.instance_id]
                )
                if candidate is None or binary is None or resolved_record is None:
                    raise DeploymentExecutionError(
                        f"deployment member disappeared: {plan.instance_id}"
                    )
                resolved_placement = resolved_placement_from_record(
                    resolved_record
                )
                port = reservations.port(plan.instance_id)
                endpoint = f"http://{_health_host(host)}:{port}"
                capabilities = _capabilities(binary)
                try:
                    argv = self.server_adapter.build_argv(
                        binary_path=Path(binary.path),
                        capabilities=capabilities,
                        model_path=plan.model_path,
                        candidate=candidate,
                        placement=resolved_placement,
                        host=host,
                        port=port,
                        draft_model_path=plan.draft_model_path,
                        model_alias=(
                            instance.server_identity
                            if capabilities.supports("--alias")
                            else None
                        ),
                    )
                except LlamaServerConfigurationError as exc:
                    raise DeploymentExecutionError(str(exc)) from exc
                result.append(
                    _MemberPlan(
                        instance_id=plan.instance_id,
                        endpoint=endpoint,
                        health_url=f"{endpoint}/health",
                        argv=argv,
                        model_path=plan.model_path,
                        draft_model_path=plan.draft_model_path,
                    )
                )

            runs = DeploymentRunRepository(connection)
            for plan in result:
                instance = instances[plan.instance_id]
                runs.add_member(
                    run_id,
                    instance_id=plan.instance_id,
                    endpoint=plan.endpoint,
                    argv=plan.argv,
                    target_model_path=str(plan.model_path),
                    draft_model_path=(
                        None
                        if plan.draft_model_path is None
                        else str(plan.draft_model_path)
                    ),
                    result={"server_identity": instance.server_identity},
                )
            return tuple(result)

    def _wait_ready_barrier(
        self,
        run_id: str,
        plans: tuple[_MemberPlan, ...],
        registry: DeploymentProcessRegistry,
        *,
        timeout_seconds: float,
        cancel_event: Event | None,
    ) -> tuple[str, ServerProcessError] | None:
        local_cancel = Event()
        futures: dict[Future[str], str] = {}
        with ThreadPoolExecutor(max_workers=max(1, len(plans))) as executor:
            for plan in plans:
                process = registry.get(plan.instance_id)
                if process is None:
                    continue
                futures[
                    executor.submit(
                        process.wait_ready,
                        plan.health_url,
                        timeout_seconds=timeout_seconds,
                        cancel_event=local_cancel,
                    )
                ] = plan.instance_id

            pending = set(futures)
            failure: tuple[str, ServerProcessError] | None = None
            while pending:
                if cancel_event is not None and cancel_event.is_set():
                    local_cancel.set()
                done, pending = wait(
                    pending,
                    timeout=0.05,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    instance_id = futures[future]
                    try:
                        ready_at = future.result()
                    except ServerProcessError as exc:
                        if failure is None:
                            failure = (instance_id, exc)
                        local_cancel.set()
                    else:
                        if failure is None:
                            self._set_member_status(
                                run_id,
                                instance_id,
                                "ready",
                                ready_at=ready_at,
                            )
            return failure

    def _runtime_memory_check(
        self,
        deployment_placement_id: str,
    ) -> _RuntimeMemoryCheck:
        with self.database.session() as connection:
            allocations = DeploymentPlacementRepository(
                connection
            ).allocations(deployment_placement_id)
        samples = self.gpu_provider.sample()
        return _validate_runtime_margins(allocations, samples)

    def _residency_probe(
        self,
        plans: tuple[_MemberPlan, ...],
        registry: DeploymentProcessRegistry,
        *,
        residency_hold_seconds: float,
        cancel_event: Event | None,
    ) -> tuple[str, ServerProcessError] | None:
        deadline = time.monotonic() + residency_hold_seconds
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                return (
                    plans[0].instance_id,
                    ServerProcessError(
                        "deployment residency cancelled",
                        kind="cancelled",
                    ),
                )
            for plan in plans:
                process = registry.get(plan.instance_id)
                if process is None or not process.alive:
                    return (
                        plan.instance_id,
                        ServerProcessError(
                            "server exited during residency hold",
                            kind="start_failed",
                        ),
                    )
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

        for plan in plans:
            process = registry.get(plan.instance_id)
            if process is None:
                return (
                    plan.instance_id,
                    ServerProcessError(
                        "server process is missing",
                        kind="start_failed",
                    ),
                )
            try:
                process.wait_ready(
                    plan.health_url,
                    timeout_seconds=1.0,
                    cancel_event=cancel_event,
                )
            except ServerProcessError as exc:
                return plan.instance_id, exc
        return None

    def _set_member_status(
        self,
        run_id: str,
        instance_id: str,
        status: DeploymentMemberStatus,
        *,
        pid: int | None = None,
        started_at: str | None = None,
        ready_at: str | None = None,
    ) -> None:
        with self.database.session() as connection:
            DeploymentRunRepository(connection).set_member_status(
                run_id,
                instance_id,
                status,
                pid=pid,
                started_at=started_at,
                ready_at=ready_at,
            )

    def _finalize_members(
        self,
        run_id: str,
        plans: tuple[_MemberPlan, ...],
        cleanup: Mapping[
            str, tuple[ServerProcessOutcome | None, str | None]
        ],
        *,
        status: DeploymentMemberStatus,
        failure_message: str | None,
    ) -> None:
        with self.database.session() as connection:
            runs = DeploymentRunRepository(connection)
            for plan in plans:
                outcome, cleanup_error = cleanup.get(
                    plan.instance_id,
                    (None, None),
                )
                runs.finish_member(
                    run_id,
                    plan.instance_id,
                    status=status,
                    pid=None if outcome is None else outcome.pid,
                    finished_at=(
                        None if outcome is None else outcome.finished_at
                    ),
                    exit_code=(
                        None if outcome is None else outcome.exit_code
                    ),
                    stdout="" if outcome is None else outcome.stdout,
                    stderr="" if outcome is None else outcome.stderr,
                    forced_kill=(
                        False if outcome is None else outcome.forced_kill
                    ),
                    cleanup_error=cleanup_error,
                    result={
                        "endpoint": plan.endpoint,
                        "failure": failure_message,
                    },
                )

    @staticmethod
    def _run_placement_id(connection, run_id: str) -> str:
        row = connection.execute(
            """
            SELECT deployment_placement_id
            FROM deployment_run
            WHERE id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None or row["deployment_placement_id"] is None:
            raise DeploymentExecutionError(
                f"deployment run placement not found: {run_id}"
            )
        return str(row["deployment_placement_id"])


def _validate_runtime_margins(
    allocations: tuple[DeploymentDeviceAllocationRecord, ...],
    samples: tuple[GpuTelemetrySample, ...],
) -> _RuntimeMemoryCheck:
    by_key = {
        sample.stable_device_key: sample
        for sample in samples
        if sample.stable_device_key is not None
    }
    validated: list[dict[str, object]] = []
    missing: list[str] = []
    violations: list[dict[str, object]] = []
    for allocation in allocations:
        sample = by_key.get(allocation.device_id)
        if (
            sample is None
            or sample.vram_total_bytes is None
            or sample.vram_used_bytes is None
        ):
            missing.append(allocation.device_id)
            continue
        free_bytes = max(
            0,
            sample.vram_total_bytes - sample.vram_used_bytes,
        )
        detail = {
            "device_id": allocation.device_id,
            "observed_total_bytes": sample.vram_total_bytes,
            "observed_used_bytes": sample.vram_used_bytes,
            "observed_free_bytes": free_bytes,
            "reserved_margin_bytes": allocation.reserved_margin_bytes,
        }
        validated.append(detail)
        if free_bytes < allocation.reserved_margin_bytes:
            violations.append(detail)
    return _RuntimeMemoryCheck(
        details={
            "validated_devices": validated,
            "missing_devices": missing,
            "violations": violations,
        },
        violations=tuple(violations),
    )


def _server_input_map(
    values: tuple[DeploymentServerInput, ...],
) -> dict[str, DeploymentServerInput]:
    result = {item.instance_id: item for item in values}
    if len(result) != len(values):
        raise DeploymentExecutionError(
            "deployment server inputs must have unique instance IDs"
        )
    return result


def _verify_server_binary(binary: BinaryRecord) -> None:
    if binary.kind != "llama-server":
        raise DeploymentExecutionError(
            f"binary {binary.id} is {binary.kind}, not llama-server"
        )
    path = Path(binary.path)
    if not path.is_file():
        raise DeploymentExecutionError(
            f"registered llama-server path does not exist: {path}"
        )
    if sha256_file(path) != binary.sha256:
        raise DeploymentExecutionError(
            "registered llama-server binary changed on disk; "
            "re-run binary inspect"
        )


def _resolve_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise DeploymentExecutionError(
            f"{label} path does not exist or is not a file: {resolved}"
        )
    return resolved


def _capabilities(binary: BinaryRecord) -> CapabilitySet:
    return CapabilitySet.from_mapping(
        dict(binary.capabilities),
        fallback_kind="llama-server",
    )


def _bind_ephemeral(host: str) -> socket.socket:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.bind((host, 0))
        return sock
    except BaseException:
        sock.close()
        raise


def _health_host(bind_host: str) -> str:
    if bind_host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return bind_host


def _oom_override(
    failure_kind: DeploymentFailureKind | None,
    cleanup: Mapping[
        str, tuple[ServerProcessOutcome | None, str | None]
    ],
) -> DeploymentFailureKind | None:
    for outcome, _ in cleanup.values():
        if outcome is None:
            continue
        text = f"{outcome.stdout}\n{outcome.stderr}".lower()
        if "out of memory" in text or "oom" in text:
            return "server_oom"
    return failure_kind


def _lock_path(database: Database) -> Path:
    raw = str(database.path)
    if raw == ":memory:":
        raise DeploymentExecutionError(
            "deployment execution requires a file-backed SQLite database"
        )
    path = Path(raw)
    return path.with_suffix(path.suffix + ".host.lock")


def _utc_now() -> str:
    return (
        datetime.now(UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
