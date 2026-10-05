"""Durable deployment operation control tests."""

from __future__ import annotations

import time
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

from llama_profile_lab.api.deployment_operations import (
    DeploymentOperationManager,
    DeploymentOperationSpec,
)
from llama_profile_lab.db import (
    Database,
    DeploymentPlacementRepository,
    DeploymentRunRepository,
)
from llama_profile_lab.execution import DeploymentExecutionError
from tests.test_concurrent_workload_execution import _seed


class ControlledExecutor:
    """First call blocks for control; second call completes."""

    def __init__(
        self,
        database: Database,
        state: dict[str, Any],
    ) -> None:
        self.database = database
        self.state = state

    def execute(
        self,
        placement_id,
        inputs,
        *,
        standalone_baselines,
        host,
        readiness_timeout_seconds,
        cancel_event,
    ):
        del inputs, standalone_baselines, host, readiness_timeout_seconds
        self.state["calls"] += 1
        if self.state["calls"] == 1:
            self.state["started"].set()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if cancel_event.is_set():
                    raise DeploymentExecutionError(
                        "synthetic controlled stop",
                        status="cancelled",
                    )
                time.sleep(0.01)
            raise AssertionError("durable control signal was not observed")

        with self.database.session() as connection:
            placement = DeploymentPlacementRepository(connection).record(
                placement_id
            )
            assert placement is not None
            runs = DeploymentRunRepository(connection)
            run_id = runs.create(
                deployment_candidate_id=placement.deployment_candidate_id,
                deployment_placement_id=placement_id,
                status="ready",
            )
            runs.finish(
                run_id,
                status="completed",
                duration_ns=1,
            )
        return SimpleNamespace(deployment_run_id=run_id)


def _manager(
    database: Database,
    state: dict[str, Any],
) -> DeploymentOperationManager:
    return DeploymentOperationManager(
        database,
        executor_factory=lambda db: ControlledExecutor(db, state),
    )


def _wait_status(
    manager: DeploymentOperationManager,
    deployment_id: str,
    expected: str,
) -> None:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        snapshot = manager.snapshot(deployment_id)
        if snapshot is not None and snapshot.status == expected:
            return
        time.sleep(0.02)
    snapshot = manager.snapshot(deployment_id)
    raise AssertionError(
        f"deployment did not reach {expected}; snapshot={snapshot}"
    )


def test_pause_and_resume_across_manager_instances(tmp_path: Path) -> None:
    database, placement_id, inputs = _seed(tmp_path)
    with database.session() as connection:
        placement = DeploymentPlacementRepository(connection).record(
            placement_id
        )
    assert placement is not None
    deployment_id = placement.deployment_candidate_id

    state: dict[str, Any] = {
        "calls": 0,
        "started": Event(),
    }
    spec = DeploymentOperationSpec(
        deployment_placement_id=placement_id,
        inputs=inputs,
    )

    first = _manager(database, state)
    started = first.start(deployment_id, spec, background=True)
    assert started.status == "running"
    assert state["started"].wait(timeout=2.0)

    second = _manager(database, state)
    pausing = second.pause(deployment_id)
    assert pausing.status == "pausing"
    _wait_status(second, deployment_id, "paused")

    third = _manager(database, state)
    resumed = third.resume(
        deployment_id,
        background=False,
    )
    assert resumed.status == "completed"
    assert resumed.deployment_run_id is not None
    assert state["calls"] == 2


def test_cancel_active_operation_from_separate_manager(
    tmp_path: Path,
) -> None:
    database, placement_id, inputs = _seed(tmp_path)
    with database.session() as connection:
        placement = DeploymentPlacementRepository(connection).record(
            placement_id
        )
    assert placement is not None
    deployment_id = placement.deployment_candidate_id
    state: dict[str, Any] = {
        "calls": 0,
        "started": Event(),
    }

    first = _manager(database, state)
    first.start(
        deployment_id,
        DeploymentOperationSpec(
            deployment_placement_id=placement_id,
            inputs=inputs,
        ),
        background=True,
    )
    assert state["started"].wait(timeout=2.0)

    second = _manager(database, state)
    cancelling = second.cancel(deployment_id)
    assert cancelling.status == "cancelling"
    _wait_status(second, deployment_id, "cancelled")
