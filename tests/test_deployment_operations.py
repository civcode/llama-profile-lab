"""Durable deployment operation control tests."""

from __future__ import annotations

import time
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

from llama_profile_lab.api.deployment_operations import (
    DeploymentOperationError,
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


def test_resume_with_replacement_spec_still_requires_paused_state(
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
    manager = _manager(database, state)

    try:
        manager.resume(
            deployment_id,
            DeploymentOperationSpec(
                deployment_placement_id=placement_id,
                inputs=inputs,
            ),
            background=False,
        )
    except DeploymentOperationError as exc:
        assert "no paused operation" in str(exc)
    else:
        raise AssertionError("resume unexpectedly started without paused state")


def test_persisted_baseline_parser_rejects_type_coercion() -> None:
    payload = {
        "deployment_placement_id": "placement-test",
        "instances": [
            {
                "instance_id": "qwen",
                "model_path": "/models/qwen.gguf",
                "draft_model_path": None,
            }
        ],
        "standalone_baselines": [
            {
                "instance_id": "qwen",
                "mode": "decode",
                "prompt_tokens": "128",
                "generate_tokens": 16,
                "depth_tokens": 128,
                "throughput_tps": 20.0,
                "latency_ms": 1.0,
            }
        ],
        "host": "127.0.0.1",
        "readiness_timeout_seconds": 30.0,
    }

    try:
        DeploymentOperationSpec.from_mapping(payload)
    except DeploymentOperationError as exc:
        assert "baseline" in str(exc)
    else:
        raise AssertionError("invalid persisted baseline was coerced")

