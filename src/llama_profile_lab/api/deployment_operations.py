"""Durable deployment execution control shared by HTTP and CLI."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Any, Literal, cast

from llama_profile_lab.db import (
    Database,
    DeploymentOperationRepository,
    transaction,
)
from llama_profile_lab.db.records import DeploymentOperationRecord
from llama_profile_lab.domain import ConcurrentWorkloadMode
from llama_profile_lab.execution import (
    ConcurrentDeploymentExecutor,
    ConcurrentDeploymentSummary,
    DeploymentExecutionError,
    DeploymentServerInput,
    StandaloneBaselineInput,
)


DeploymentOperationStatus = Literal[
    "running",
    "pausing",
    "cancelling",
    "completed",
    "paused",
    "cancelled",
    "failed",
]
DeploymentRequestedAction = Literal["pause", "cancel"]


class DeploymentOperationError(RuntimeError):
    """Raised when a durable deployment operation transition is invalid."""


@dataclass(frozen=True, slots=True)
class DeploymentOperationSpec:
    """Exact concurrent deployment request persisted for resume."""

    deployment_placement_id: str
    inputs: tuple[DeploymentServerInput, ...]
    standalone_baselines: tuple[StandaloneBaselineInput, ...] = ()
    host: str = "127.0.0.1"
    readiness_timeout_seconds: float = 300.0

    def to_mapping(self) -> dict[str, object]:
        return {
            "deployment_placement_id": self.deployment_placement_id,
            "instances": [
                {
                    "instance_id": item.instance_id,
                    "model_path": str(item.model_path),
                    "draft_model_path": (
                        None
                        if item.draft_model_path is None
                        else str(item.draft_model_path)
                    ),
                }
                for item in self.inputs
            ],
            "standalone_baselines": [
                {
                    "instance_id": item.instance_id,
                    "mode": item.mode,
                    "prompt_tokens": item.prompt_tokens,
                    "generate_tokens": item.generate_tokens,
                    "depth_tokens": item.depth_tokens,
                    "throughput_tps": item.throughput_tps,
                    "latency_ms": item.latency_ms,
                }
                for item in self.standalone_baselines
            ],
            "host": self.host,
            "readiness_timeout_seconds": self.readiness_timeout_seconds,
        }

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "DeploymentOperationSpec":
        placement_id = payload.get("deployment_placement_id")
        if not isinstance(placement_id, str) or not placement_id:
            raise DeploymentOperationError(
                "persisted deployment operation has invalid placement ID"
            )
        instance_rows = payload.get("instances")
        if not isinstance(instance_rows, list) or not instance_rows:
            raise DeploymentOperationError(
                "persisted deployment operation has no instances"
            )
        inputs: list[DeploymentServerInput] = []
        for row in instance_rows:
            if not isinstance(row, dict):
                raise DeploymentOperationError(
                    "persisted deployment operation instance is invalid"
                )
            instance_id = row.get("instance_id")
            model_path = row.get("model_path")
            draft_path = row.get("draft_model_path")
            if (
                not isinstance(instance_id, str)
                or not instance_id
                or not isinstance(model_path, str)
                or not model_path
                or (
                    draft_path is not None
                    and (
                        not isinstance(draft_path, str)
                        or not draft_path
                    )
                )
            ):
                raise DeploymentOperationError(
                    "persisted deployment operation instance is incomplete"
                )
            inputs.append(
                DeploymentServerInput(
                    instance_id=instance_id,
                    model_path=Path(model_path),
                    draft_model_path=(
                        None if draft_path is None else Path(draft_path)
                    ),
                )
            )

        raw_baselines = payload.get("standalone_baselines", [])
        if not isinstance(raw_baselines, list):
            raise DeploymentOperationError(
                "persisted standalone baselines must be a list"
            )
        baselines: list[StandaloneBaselineInput] = []
        for row in raw_baselines:
            if not isinstance(row, dict):
                raise DeploymentOperationError(
                    "persisted standalone baseline is invalid"
                )
            try:
                instance_id = row["instance_id"]
                mode = row["mode"]
                prompt_tokens = row["prompt_tokens"]
                generate_tokens = row["generate_tokens"]
                depth_tokens = row["depth_tokens"]
                throughput_tps = row["throughput_tps"]
                latency_ms = row.get("latency_ms")
                if not isinstance(instance_id, str) or not instance_id:
                    raise ValueError("baseline instance_id is invalid")
                if mode not in {"prefill", "decode"}:
                    raise ValueError("baseline mode is invalid")
                for value in (
                    prompt_tokens,
                    generate_tokens,
                    depth_tokens,
                ):
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                    ):
                        raise ValueError("baseline token count is invalid")
                if (
                    isinstance(throughput_tps, bool)
                    or not isinstance(throughput_tps, (int, float))
                    or throughput_tps <= 0
                ):
                    raise ValueError("baseline throughput is invalid")
                if (
                    latency_ms is not None
                    and (
                        isinstance(latency_ms, bool)
                        or not isinstance(latency_ms, (int, float))
                        or latency_ms < 0
                    )
                ):
                    raise ValueError("baseline latency is invalid")
                baselines.append(
                    StandaloneBaselineInput(
                        instance_id=instance_id,
                        mode=cast(ConcurrentWorkloadMode, mode),
                        prompt_tokens=prompt_tokens,
                        generate_tokens=generate_tokens,
                        depth_tokens=depth_tokens,
                        throughput_tps=float(throughput_tps),
                        latency_ms=(
                            None
                            if latency_ms is None
                            else float(latency_ms)
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise DeploymentOperationError(
                    "persisted standalone baseline is incomplete"
                ) from exc

        host = payload.get("host", "127.0.0.1")
        readiness = payload.get("readiness_timeout_seconds", 300.0)
        if not isinstance(host, str) or not host:
            raise DeploymentOperationError(
                "persisted deployment host is invalid"
            )
        if (
            isinstance(readiness, bool)
            or not isinstance(readiness, (int, float))
            or readiness <= 0
        ):
            raise DeploymentOperationError(
                "persisted readiness timeout is invalid"
            )
        return cls(
            deployment_placement_id=placement_id,
            inputs=tuple(inputs),
            standalone_baselines=tuple(baselines),
            host=host,
            readiness_timeout_seconds=float(readiness),
        )


@dataclass(frozen=True, slots=True)
class DeploymentOperationSnapshot:
    """Stable application snapshot of one durable operation."""

    id: str
    deployment_candidate_id: str
    deployment_placement_id: str
    deployment_run_id: str | None
    status: DeploymentOperationStatus
    requested_action: DeploymentRequestedAction | None
    started_at: str
    finished_at: str | None
    error: str | None


class _PersistentControlSignal(Event):
    """Threading Event whose observed state is backed by SQLite."""

    def __init__(self, database: Database, operation_id: str) -> None:
        super().__init__()
        self.database = database
        self.operation_id = operation_id

    def is_set(self) -> bool:
        with self.database.session() as connection:
            return DeploymentOperationRepository(
                connection
            ).is_control_requested(self.operation_id)


class DeploymentOperationManager:
    """Start and control deployment benchmarks across API or CLI processes."""

    def __init__(
        self,
        database: Database,
        *,
        executor_factory: Callable[
            [Database], ConcurrentDeploymentExecutor
        ] | None = None,
    ) -> None:
        self.database = database
        self.executor_factory = (
            executor_factory or ConcurrentDeploymentExecutor
        )
        self._lock = RLock()
        self._threads: dict[str, Thread] = {}

    def start(
        self,
        deployment_candidate_id: str,
        spec: DeploymentOperationSpec,
        *,
        background: bool = True,
    ) -> DeploymentOperationSnapshot:
        with self._lock:
            with self.database.session() as connection:
                with transaction(connection, immediate=True):
                    try:
                        operation_id = DeploymentOperationRepository(
                            connection
                        ).create(
                            base_deployment_candidate_id=(
                                deployment_candidate_id
                            ),
                            deployment_placement_id=(
                                spec.deployment_placement_id
                            ),
                            request=spec.to_mapping(),
                        )
                    except ValueError as exc:
                        raise DeploymentOperationError(str(exc)) from exc
            if background:
                thread = Thread(
                    target=self._run,
                    args=(operation_id, spec),
                    name=f"llprof-{operation_id}",
                    daemon=True,
                )
                self._threads[operation_id] = thread
                thread.start()
                return self.snapshot_by_id(operation_id)
        self._run(operation_id, spec)
        return self.snapshot_by_id(operation_id)

    def resume(
        self,
        deployment_candidate_id: str,
        spec: DeploymentOperationSpec | None = None,
        *,
        background: bool = True,
    ) -> DeploymentOperationSnapshot:
        with self.database.session() as connection:
            latest = DeploymentOperationRepository(
                connection
            ).latest_for_deployment(deployment_candidate_id)
        if latest is None or latest.status != "paused":
            raise DeploymentOperationError(
                "deployment has no paused operation to resume"
            )
        if spec is None:
            spec = DeploymentOperationSpec.from_mapping(latest.request)
        return self.start(
            deployment_candidate_id,
            spec,
            background=background,
        )

    def pause(
        self,
        deployment_candidate_id: str,
    ) -> DeploymentOperationSnapshot:
        return self._request_action(deployment_candidate_id, "pause")

    def cancel(
        self,
        deployment_candidate_id: str,
    ) -> DeploymentOperationSnapshot:
        return self._request_action(deployment_candidate_id, "cancel")

    def snapshot(
        self,
        deployment_candidate_id: str,
    ) -> DeploymentOperationSnapshot | None:
        with self.database.session() as connection:
            record = DeploymentOperationRepository(
                connection
            ).latest_for_deployment(deployment_candidate_id)
        return None if record is None else _snapshot(record)

    def snapshot_by_id(
        self,
        operation_id: str,
    ) -> DeploymentOperationSnapshot:
        with self.database.session() as connection:
            record = DeploymentOperationRepository(connection).get(
                operation_id
            )
        if record is None:
            raise DeploymentOperationError(
                f"deployment operation not found: {operation_id}"
            )
        return _snapshot(record)

    def _request_action(
        self,
        deployment_candidate_id: str,
        action: str,
    ) -> DeploymentOperationSnapshot:
        with self.database.session() as connection:
            try:
                record = DeploymentOperationRepository(
                    connection
                ).request_action(deployment_candidate_id, action)
            except ValueError as exc:
                raise DeploymentOperationError(str(exc)) from exc
        return _snapshot(record)

    def _run(
        self,
        operation_id: str,
        spec: DeploymentOperationSpec,
    ) -> None:
        summary: ConcurrentDeploymentSummary | None = None
        error: str | None = None
        run_id: str | None = None
        try:
            summary = self.executor_factory(self.database).execute(
                spec.deployment_placement_id,
                spec.inputs,
                standalone_baselines=spec.standalone_baselines,
                host=spec.host,
                readiness_timeout_seconds=spec.readiness_timeout_seconds,
                cancel_event=_PersistentControlSignal(
                    self.database,
                    operation_id,
                ),
            )
            run_id = summary.deployment_run_id
        except DeploymentExecutionError as exc:
            error = str(exc)
            run_id = exc.run_id
        except Exception as exc:
            error = str(exc)

        with self.database.session() as connection:
            repository = DeploymentOperationRepository(connection)
            record = repository.get(operation_id)
            if record is None:
                return
            requested = record.requested_action
            if summary is not None:
                terminal = "completed"
            elif requested == "pause":
                terminal = "paused"
            elif requested == "cancel":
                terminal = "cancelled"
            elif error is not None:
                terminal = "failed"
            else:
                terminal = "completed"
            try:
                repository.finish(
                    operation_id,
                    status=terminal,
                    deployment_run_id=run_id,
                    error=(
                        None
                        if terminal in {"completed", "paused", "cancelled"}
                        else error
                    ),
                )
            except ValueError:
                return

        with self._lock:
            self._threads.pop(operation_id, None)


def _snapshot(
    record: DeploymentOperationRecord,
) -> DeploymentOperationSnapshot:
    return DeploymentOperationSnapshot(
        id=record.id,
        deployment_candidate_id=record.base_deployment_candidate_id,
        deployment_placement_id=record.deployment_placement_id,
        deployment_run_id=record.deployment_run_id,
        status=cast(DeploymentOperationStatus, record.status),
        requested_action=cast(
            DeploymentRequestedAction | None,
            record.requested_action,
        ),
        started_at=record.started_at,
        finished_at=record.finished_at,
        error=record.error,
    )
