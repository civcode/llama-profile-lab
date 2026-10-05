"""In-process execution control for V2 deployment HTTP operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Event, RLock, Thread
from typing import Literal
from uuid import uuid4

from llama_profile_lab.db import Database
from llama_profile_lab.execution import (
    ConcurrentDeploymentExecutor,
    ConcurrentDeploymentSummary,
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
RequestedAction = Literal["pause", "cancel"]


class DeploymentOperationError(RuntimeError):
    """Raised when a deployment API operation conflicts with current state."""


@dataclass(frozen=True, slots=True)
class DeploymentExecutionSpec:
    """Immutable inputs needed to replay one concurrent deployment run."""

    deployment_placement_id: str
    inputs: tuple[DeploymentServerInput, ...]
    standalone_baselines: tuple[StandaloneBaselineInput, ...]
    host: str
    readiness_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class DeploymentOperationSnapshot:
    """Thread-safe public snapshot of one API-started deployment operation."""

    id: str
    deployment_candidate_id: str
    status: DeploymentOperationStatus
    started_at: str
    finished_at: str | None
    requested_action: RequestedAction | None
    deployment_run_id: str | None
    deployment_placement_id: str
    phase_count: int | None
    error: str | None


@dataclass(slots=True)
class _DeploymentOperation:
    id: str
    deployment_candidate_id: str
    spec: DeploymentExecutionSpec
    cancel_event: Event
    status: DeploymentOperationStatus
    started_at: str
    finished_at: str | None = None
    requested_action: RequestedAction | None = None
    summary: ConcurrentDeploymentSummary | None = None
    error: str | None = None
    thread: Thread | None = None


class DeploymentOperationManager:
    """Own deployment executions started through the local HTTP API."""

    def __init__(
        self,
        database: Database,
        *,
        executor_factory: (
            Callable[[Database], ConcurrentDeploymentExecutor] | None
        ) = None,
    ) -> None:
        self.database = database
        self.executor_factory = (
            executor_factory or ConcurrentDeploymentExecutor
        )
        self._lock = RLock()
        self._operations: dict[str, _DeploymentOperation] = {}
        self._last_specs: dict[str, DeploymentExecutionSpec] = {}

    def start(
        self,
        deployment_candidate_id: str,
        *,
        spec: DeploymentExecutionSpec,
    ) -> DeploymentOperationSnapshot:
        with self._lock:
            current = self._operations.get(deployment_candidate_id)
            if current is not None and _active(current.status):
                raise DeploymentOperationError(
                    f"deployment {deployment_candidate_id} already has "
                    "an active API operation"
                )
            for operation in self._operations.values():
                if _active(operation.status):
                    raise DeploymentOperationError(
                        "another deployment operation is already active "
                        "on this host"
                    )

            operation = _DeploymentOperation(
                id=f"deployop_{uuid4().hex}",
                deployment_candidate_id=deployment_candidate_id,
                spec=spec,
                cancel_event=Event(),
                status="running",
                started_at=_utc_now(),
            )
            thread = Thread(
                target=self._run,
                args=(operation,),
                name=f"llprof-{operation.id}",
                daemon=True,
            )
            operation.thread = thread
            self._operations[deployment_candidate_id] = operation
            self._last_specs[deployment_candidate_id] = spec
            thread.start()
            return _snapshot(operation)

    def resume(
        self,
        deployment_candidate_id: str,
        *,
        spec: DeploymentExecutionSpec | None = None,
    ) -> DeploymentOperationSnapshot:
        with self._lock:
            selected = spec or self._last_specs.get(deployment_candidate_id)
        if selected is None:
            raise DeploymentOperationError(
                "deployment has no prior API execution specification to resume"
            )
        return self.start(
            deployment_candidate_id,
            spec=selected,
        )

    def pause(
        self,
        deployment_candidate_id: str,
    ) -> DeploymentOperationSnapshot:
        return self._request_stop(deployment_candidate_id, "pause")

    def cancel(
        self,
        deployment_candidate_id: str,
    ) -> DeploymentOperationSnapshot:
        return self._request_stop(deployment_candidate_id, "cancel")

    def snapshot(
        self,
        deployment_candidate_id: str,
    ) -> DeploymentOperationSnapshot | None:
        with self._lock:
            operation = self._operations.get(deployment_candidate_id)
            return None if operation is None else _snapshot(operation)

    def _request_stop(
        self,
        deployment_candidate_id: str,
        action: RequestedAction,
    ) -> DeploymentOperationSnapshot:
        with self._lock:
            operation = self._operations.get(deployment_candidate_id)
            if operation is None or not _active(operation.status):
                raise DeploymentOperationError(
                    f"deployment {deployment_candidate_id} has no "
                    f"active API operation to {action}"
                )
            operation.requested_action = action
            operation.status = (
                "pausing" if action == "pause" else "cancelling"
            )
            operation.cancel_event.set()
            return _snapshot(operation)

    def _run(self, operation: _DeploymentOperation) -> None:
        summary: ConcurrentDeploymentSummary | None = None
        error: str | None = None
        try:
            spec = operation.spec
            summary = self.executor_factory(self.database).execute(
                spec.deployment_placement_id,
                spec.inputs,
                standalone_baselines=spec.standalone_baselines,
                host=spec.host,
                readiness_timeout_seconds=spec.readiness_timeout_seconds,
                cancel_event=operation.cancel_event,
            )
        except Exception as exc:
            error = str(exc)

        with self._lock:
            operation.summary = summary
            operation.error = error
            operation.finished_at = _utc_now()
            if operation.requested_action == "pause":
                operation.status = "paused"
                return
            if operation.requested_action == "cancel":
                operation.status = "cancelled"
                return
            if error is not None:
                operation.status = "failed"
                return
            operation.status = "completed"


def _snapshot(
    operation: _DeploymentOperation,
) -> DeploymentOperationSnapshot:
    summary = operation.summary
    return DeploymentOperationSnapshot(
        id=operation.id,
        deployment_candidate_id=operation.deployment_candidate_id,
        status=operation.status,
        started_at=operation.started_at,
        finished_at=operation.finished_at,
        requested_action=operation.requested_action,
        deployment_run_id=(
            None if summary is None else summary.deployment_run_id
        ),
        deployment_placement_id=operation.spec.deployment_placement_id,
        phase_count=None if summary is None else len(summary.phases),
        error=operation.error,
    )


def _active(status: DeploymentOperationStatus) -> bool:
    return status in {"running", "pausing", "cancelling"}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")
