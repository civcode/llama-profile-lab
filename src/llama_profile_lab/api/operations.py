"""In-process execution control for the local HTTP API."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Literal
from uuid import uuid4

from llama_profile_lab.db import Database, ExperimentRepository
from llama_profile_lab.execution import (
    ExecutionSummary,
    ExperimentExecutor,
    HostLockError,
)


OperationStatus = Literal[
    "running",
    "pausing",
    "cancelling",
    "completed",
    "paused",
    "cancelled",
    "failed",
]
RequestedAction = Literal["pause", "cancel"]


class OperationError(RuntimeError):
    """Raised when an API execution action conflicts with current state."""


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    binary_id: str
    model_path: Path
    fit_binary_id: str | None
    timeout_seconds: float | None
    fit_timeout_seconds: float | None
    limit: int | None
    telemetry_interval_seconds: float


@dataclass(frozen=True, slots=True)
class OperationSnapshot:
    id: str
    experiment_id: str
    status: OperationStatus
    started_at: str
    finished_at: str | None
    requested_action: RequestedAction | None
    summary: ExecutionSummary | None
    error: str | None


@dataclass(slots=True)
class _Operation:
    id: str
    experiment_id: str
    cancel_event: Event
    status: OperationStatus
    started_at: str
    finished_at: str | None = None
    requested_action: RequestedAction | None = None
    summary: ExecutionSummary | None = None
    error: str | None = None
    thread: Thread | None = None


class OperationManager:
    """Own API-started benchmark execution and cooperative pause/cancel signals."""

    def __init__(
        self,
        database: Database,
        *,
        executor_factory: Callable[[Database], ExperimentExecutor] | None = None,
    ) -> None:
        self.database = database
        self.executor_factory = executor_factory or ExperimentExecutor
        self._lock = RLock()
        self._operations: dict[str, _Operation] = {}

    def start(
        self,
        experiment_id: str,
        *,
        spec: ExecutionSpec,
        resume: bool,
    ) -> OperationSnapshot:
        with self._lock:
            current = self._operations.get(experiment_id)
            if current is not None and _active(current.status):
                raise OperationError(
                    f"experiment {experiment_id} already has an active API operation"
                )
            for operation in self._operations.values():
                if _active(operation.status):
                    raise OperationError(
                        "another benchmark operation is already active on this host"
                    )

            operation = _Operation(
                id=f"op_{uuid4().hex}",
                experiment_id=experiment_id,
                cancel_event=Event(),
                status="running",
                started_at=_utc_now(),
            )
            thread = Thread(
                target=self._run,
                args=(operation, spec, resume),
                name=f"llprof-{operation.id}",
                daemon=True,
            )
            operation.thread = thread
            self._operations[experiment_id] = operation
            thread.start()
            return _snapshot(operation)

    def pause(self, experiment_id: str) -> OperationSnapshot:
        with self._lock:
            operation = self._operations.get(experiment_id)
            if operation is None or not _active(operation.status):
                with self.database.session() as connection:
                    record = ExperimentRepository(connection).get(experiment_id)
                if record is None:
                    raise OperationError(f"experiment not found: {experiment_id}")
                if record.status == "paused":
                    raise OperationError(f"experiment {experiment_id} is already paused")
                raise OperationError(
                    f"experiment {experiment_id} has no active API operation to pause"
                )
            operation.requested_action = "pause"
            operation.status = "pausing"
            operation.cancel_event.set()
            return _snapshot(operation)

    def cancel(self, experiment_id: str) -> OperationSnapshot | None:
        with self._lock:
            operation = self._operations.get(experiment_id)
            if operation is not None and _active(operation.status):
                operation.requested_action = "cancel"
                operation.status = "cancelling"
                operation.cancel_event.set()
                return _snapshot(operation)

        with self.database.session() as connection:
            experiments = ExperimentRepository(connection)
            record = experiments.get(experiment_id)
            if record is None:
                raise OperationError(f"experiment not found: {experiment_id}")
            if record.status == "cancelled":
                return None
            if record.status == "completed":
                raise OperationError(
                    f"experiment {experiment_id} is completed and cannot be cancelled"
                )
            if record.status == "running":
                raise OperationError(
                    "experiment is running outside this API process; "
                    "resume it to recover stale work before cancelling"
                )
            try:
                experiments.mark_cancelled(experiment_id)
            except ValueError as exc:
                raise OperationError(str(exc)) from exc
        return None

    def snapshot(self, experiment_id: str) -> OperationSnapshot | None:
        with self._lock:
            operation = self._operations.get(experiment_id)
            return None if operation is None else _snapshot(operation)

    def _run(
        self,
        operation: _Operation,
        spec: ExecutionSpec,
        resume: bool,
    ) -> None:
        summary: ExecutionSummary | None = None
        error: str | None = None
        try:
            summary = self.executor_factory(self.database).execute(
                operation.experiment_id,
                binary_id=spec.binary_id,
                model_path=spec.model_path,
                fit_binary_id=spec.fit_binary_id,
                timeout_seconds=spec.timeout_seconds,
                fit_timeout_seconds=spec.fit_timeout_seconds,
                limit=spec.limit,
                resume=resume,
                telemetry_interval_seconds=spec.telemetry_interval_seconds,
                cancel_event=operation.cancel_event,
            )
        except (Exception, HostLockError) as exc:
            error = str(exc)
            self._mark_failed_if_running(operation.experiment_id)

        with self._lock:
            operation.summary = summary
            operation.error = error
            operation.finished_at = _utc_now()

            with self.database.session() as connection:
                experiments = ExperimentRepository(connection)
                record = experiments.get(operation.experiment_id)
                if record is None:
                    operation.status = "failed"
                    operation.error = operation.error or "experiment disappeared"
                    return

                if error is not None:
                    operation.status = "failed"
                    return

                if operation.requested_action == "cancel" and record.status != "completed":
                    try:
                        experiments.mark_cancelled(operation.experiment_id)
                    except ValueError:
                        refreshed = experiments.get(operation.experiment_id)
                        if refreshed is None or refreshed.status != "cancelled":
                            operation.status = "failed"
                            operation.error = "failed to persist experiment cancellation"
                            return
                    operation.status = "cancelled"
                    return

                refreshed = experiments.get(operation.experiment_id)
                status = record.status if refreshed is None else refreshed.status
                if status == "completed":
                    operation.status = "completed"
                elif status == "cancelled":
                    operation.status = "cancelled"
                elif status == "paused":
                    operation.status = "paused"
                else:
                    operation.status = "failed"
                    operation.error = (
                        operation.error
                        or f"execution ended with unexpected experiment status {status}"
                    )

    def _mark_failed_if_running(self, experiment_id: str) -> None:
        with self.database.session() as connection:
            experiments = ExperimentRepository(connection)
            record = experiments.get(experiment_id)
            if record is not None and record.status == "running":
                try:
                    experiments.mark_failed(experiment_id)
                except ValueError:
                    return


def _snapshot(operation: _Operation) -> OperationSnapshot:
    return OperationSnapshot(
        id=operation.id,
        experiment_id=operation.experiment_id,
        status=operation.status,
        started_at=operation.started_at,
        finished_at=operation.finished_at,
        requested_action=operation.requested_action,
        summary=operation.summary,
        error=operation.error,
    )


def _active(status: OperationStatus) -> bool:
    return status in {"running", "pausing", "cancelling"}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
