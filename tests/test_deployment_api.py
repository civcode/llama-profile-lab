"""HTTP acceptance coverage for V2-M8 deployment resources."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import llama_profile_lab.api.service as api_service_module
from llama_profile_lab.api import create_app
from llama_profile_lab.api.deployment_operations import (
    DeploymentOperationSnapshot,
)
from llama_profile_lab.db import (
    Database,
    DeploymentCandidateRepository,
    DeploymentPlanRepository,
    DeploymentPlacementRepository,
    DeploymentRunRepository,
)
from llama_profile_lab.domain import (
    AcceleratorDevice,
    DeploymentSearchDimension,
    DeploymentSearchSpace,
)
from llama_profile_lab.execution import DeviceInventoryResult
from llama_profile_lab.planning import DeploymentPlanSummary
from tests.test_api import api_request
from tests.test_deployment_analysis import _seed


class FakePlanner:
    def __init__(self, database: Database) -> None:
        self.database = database

    @staticmethod
    def _summary(deployment_id: str, *, plan_id: str | None) -> DeploymentPlanSummary:
        return DeploymentPlanSummary(
            base_deployment_candidate_id=deployment_id,
            host_id="host_api",
            raw_combinations=2,
            rejected_by_constraints=0,
            duplicate_candidates=0,
            symmetry_reduced=0,
            capability_rejected=0,
            estimate_failed=0,
            memory_rejected=1,
            valid_count=1,
            plan_id=plan_id,
        )

    def preview(
        self,
        deployment_id,
        search_space,
        inputs,
        *,
        timeout_seconds,
    ) -> DeploymentPlanSummary:
        assert deployment_id
        assert len(search_space.dimensions) == 1
        assert len(inputs) == 2
        assert timeout_seconds == 30.0
        return self._summary(deployment_id, plan_id=None)

    def plan(
        self,
        deployment_id,
        search_space,
        inputs,
        *,
        timeout_seconds,
    ) -> DeploymentPlanSummary:
        assert deployment_id
        assert len(search_space.dimensions) == 1
        assert len(inputs) == 2
        assert timeout_seconds == 30.0
        return self._summary(deployment_id, plan_id="deployplan_api")


class FakeDeviceInventory:
    def __init__(self, database: Database) -> None:
        self.database = database

    def inspect(
        self,
        binary_id: str,
        *,
        timeout_seconds: float | None = 30.0,
        cancel_event=None,
    ) -> DeviceInventoryResult:
        del cancel_event
        assert binary_id == "server-ui"
        assert timeout_seconds == 12.0
        return DeviceInventoryResult(
            host_id="host-ui",
            binary_id=binary_id,
            devices=(
                AcceleratorDevice(
                    logical_device_name="CUDA0",
                    backend="CUDA",
                    mapping_status="mapped",
                    physical_device_key="uuid:gpu-0",
                    uuid="gpu-0",
                    vendor="NVIDIA",
                    product_name="RTX Test",
                    total_memory_bytes=24 * 1024**3,
                    free_memory_bytes=20 * 1024**3,
                ),
            ),
            stdout="",
            stderr="",
        )


class SequencedDeploymentOperations:
    def __init__(
        self,
        snapshots: list[DeploymentOperationSnapshot] | None = None,
    ) -> None:
        self.snapshots = snapshots or []
        self.current: DeploymentOperationSnapshot | None = None

    def _next(
        self,
        deployment_id: str,
    ) -> DeploymentOperationSnapshot | None:
        if self.snapshots:
            self.current = self.snapshots.pop(0)
        if (
            self.current is not None
            and self.current.deployment_candidate_id != deployment_id
        ):
            raise AssertionError("unexpected deployment ID")
        return self.current

    def snapshot(
        self,
        deployment_id: str,
    ) -> DeploymentOperationSnapshot | None:
        return self._next(deployment_id)

    def start(self, deployment_id, spec, *, background=True):
        del background
        self.current = DeploymentOperationSnapshot(
            id="deployop_run",
            deployment_candidate_id=deployment_id,
            deployment_placement_id=spec.deployment_placement_id,
            deployment_run_id=None,
            status="running",
            requested_action=None,
            started_at="2026-10-05T12:00:00Z",
            finished_at=None,
            error=None,
        )
        return self.current

    def resume(self, deployment_id, spec=None, *, background=True):
        del spec, background
        if self.current is None:
            raise AssertionError("resume expected prior operation")
        self.current = DeploymentOperationSnapshot(
            id="deployop_resume",
            deployment_candidate_id=deployment_id,
            deployment_placement_id=self.current.deployment_placement_id,
            deployment_run_id=self.current.deployment_run_id,
            status="running",
            requested_action=None,
            started_at=self.current.started_at,
            finished_at=None,
            error=None,
        )
        return self.current

    def pause(self, deployment_id):
        if self.current is None:
            raise AssertionError("pause expected active operation")
        self.current = DeploymentOperationSnapshot(
            id=self.current.id,
            deployment_candidate_id=deployment_id,
            deployment_placement_id=self.current.deployment_placement_id,
            deployment_run_id=self.current.deployment_run_id,
            status="pausing",
            requested_action="pause",
            started_at=self.current.started_at,
            finished_at=None,
            error=None,
        )
        return self.current

    def cancel(self, deployment_id):
        if self.current is None:
            raise AssertionError("cancel expected active operation")
        self.current = DeploymentOperationSnapshot(
            id=self.current.id,
            deployment_candidate_id=deployment_id,
            deployment_placement_id=self.current.deployment_placement_id,
            deployment_run_id=self.current.deployment_run_id,
            status="cancelling",
            requested_action="cancel",
            started_at=self.current.started_at,
            finished_at=None,
            error=None,
        )
        return self.current


def _link_plan(
    database: Database,
    base_id: str,
    subjects: dict[str, tuple[str, ...]],
) -> str:
    with database.session() as connection:
        first = DeploymentPlacementRepository(connection).record(
            subjects["a"][1]
        )
        assert first is not None
        repository = DeploymentPlanRepository(connection)
        plan_id = repository.create(
            base_deployment_candidate_id=base_id,
            host_id=first.host_id,
            search_space=DeploymentSearchSpace(
                dimensions=(
                    DeploymentSearchDimension(
                        path="instances.qwen.context.size",
                        values=(6000,),
                    ),
                )
            ),
            request={"fixture": True},
            raw_combinations=len(subjects),
            rejected_by_constraints=0,
            duplicate_candidates=0,
            symmetry_reduced=0,
            capability_rejected=0,
            estimate_failed=0,
            memory_rejected=0,
            valid_count=len(subjects),
        )
        for ordinal, (label, subject) in enumerate(
            sorted(subjects.items())
        ):
            repository.add_case(
                plan_id,
                ordinal=ordinal,
                deployment_candidate_id=subject[0],
                deployment_placement_id=subject[1],
                generation={"label": label},
            )
    return plan_id


def _run_body(placement_id: str) -> dict[str, object]:
    return {
        "deployment_placement_id": placement_id,
        "instances": [
            {
                "instance_id": "qwen",
                "model_path": "/models/qwen.gguf",
            },
            {
                "instance_id": "flash",
                "model_path": "/models/flash.gguf",
            },
        ],
        "readiness_timeout_seconds": 30,
    }


def test_deployment_routes_preserve_v1_api_surface(tmp_path: Path) -> None:
    app = create_app(tmp_path / "api.db")
    paths = {route.path for route in app.routes}

    assert "/api/experiments" in paths
    assert "/api/experiments/{experiment_id}" in paths
    assert "/api/experiments/{experiment_id}/events" in paths
    assert "/api/runs/{run_id}" in paths
    assert "/api/runs/{run_id}/telemetry" in paths
    assert "/api/placements" in paths
    assert "/api/deployments" in paths
    assert "/api/deployments/{deployment_id}/events" in paths


def test_binary_device_inventory_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_service_module,
        "DeviceInventoryService",
        FakeDeviceInventory,
    )
    app = create_app(tmp_path / "api.db")

    response = api_request(
        app,
        "POST",
        "/api/binaries/server-ui/devices",
        query=[("timeout_seconds", "12")],
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["host_id"] == "host-ui"
    assert payload["items"][0]["logical_device_name"] == "CUDA0"
    assert payload["items"][0]["physical_device_key"] == "uuid:gpu-0"
    assert payload["items"][0]["free_memory_bytes"] == 20 * 1024**3


def test_deployment_list_returns_persisted_definitions(tmp_path: Path) -> None:
    database, subjects = _seed(tmp_path)
    app = create_app(
        database.path,
        deployment_operation_manager=SequencedDeploymentOperations(),
    )

    response = api_request(app, "GET", "/api/deployments")
    assert response.status_code == 200
    ids = {item["id"] for item in response.json()["items"]}
    assert subjects["a"][0] in ids
    assert subjects["b"][0] in ids


def test_deployment_list_hides_generated_plan_candidates(tmp_path: Path) -> None:
    database, subjects = _seed(tmp_path)
    base_id = subjects["a"][0]
    _link_plan(database, base_id, subjects)
    app = create_app(
        database.path,
        deployment_operation_manager=SequencedDeploymentOperations(),
    )

    response = api_request(app, "GET", "/api/deployments")
    assert response.status_code == 200
    ids = {item["id"] for item in response.json()["items"]}
    assert base_id in ids
    assert subjects["b"][0] not in ids
    assert subjects["c"][0] not in ids
    assert subjects["d"][0] not in ids


def test_deployment_create_plan_get_and_invalid_reference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database, subjects = _seed(tmp_path)
    base_id = subjects["a"][0]
    with database.session() as connection:
        definition = DeploymentCandidateRepository(connection).get(base_id)
    assert definition is not None

    operations = SequencedDeploymentOperations()
    app = create_app(
        database.path,
        deployment_operation_manager=operations,
    )

    created = api_request(
        app,
        "POST",
        "/api/deployments",
        body={
            "deployment": definition.model_dump(
                mode="json",
                by_alias=True,
            )
        },
    )
    assert created.status_code == 201
    assert created.json()["id"] == base_id
    assert created.json()["status"] == "completed"
    assert created.json()["run_count"] >= 1

    fetched = api_request(
        app,
        "GET",
        f"/api/deployments/{base_id}",
    )
    assert fetched.status_code == 200
    assert fetched.json()["definition"]["instances"][0]["instance_id"] == "flash"

    invalid = definition.model_dump(mode="json", by_alias=True)
    invalid["instances"][0]["candidate_id"] = "cand_missing"
    rejected = api_request(
        app,
        "POST",
        "/api/deployments",
        body={"deployment": invalid},
    )
    assert rejected.status_code == 409

    monkeypatch.setattr(
        api_service_module,
        "DeploymentPlannerService",
        FakePlanner,
    )
    plan_body = {
        "search_space": {
            "dimensions": [
                {
                    "path": "instances.qwen.context.size",
                    "values": [6000],
                }
            ]
        },
        "instances": [
            {
                "instance_id": "qwen",
                "helper_binary_id": "helper_qwen",
                "model_path": "/models/qwen.gguf",
            },
            {
                "instance_id": "flash",
                "helper_binary_id": "helper_flash",
                "model_path": "/models/flash.gguf",
            },
        ],
        "timeout_seconds": 30,
    }
    previewed = api_request(
        app,
        "POST",
        f"/api/deployments/{base_id}/preview",
        body=plan_body,
    )
    assert previewed.status_code == 200
    assert previewed.json()["plan_id"] is None
    assert previewed.json()["raw_combinations"] == 2
    assert previewed.json()["memory_rejected"] == 1
    assert previewed.json()["valid_count"] == 1

    planned = api_request(
        app,
        "POST",
        f"/api/deployments/{base_id}/plan",
        body=plan_body,
    )
    assert planned.status_code == 200
    assert planned.json()["raw_combinations"] == 2
    assert planned.json()["memory_rejected"] == 1
    assert planned.json()["valid_count"] == 1


def test_deployment_inspection_results_and_pareto(tmp_path: Path) -> None:
    database, subjects = _seed(tmp_path)
    base_id = subjects["a"][0]
    _link_plan(database, base_id, subjects)
    with database.session() as connection:
        DeploymentCandidateRepository(connection).add_rejection(
            subjects["b"][0],
            stage="memory",
            reason="device_memory_exceeded",
            details={"device_id": "GPU0", "projected_bytes": 123},
        )
        failed_runs = DeploymentRunRepository(connection)
        failed_runs.add_member(
            subjects["d"][2],
            instance_id="qwen",
            member_status="starting",
            endpoint="http://127.0.0.1:49999",
        )
        failed_runs.finish_member(
            subjects["d"][2],
            "qwen",
            status="failed",
            exit_code=7,
            stderr="synthetic member failure",
            result={"failure": "synthetic member failure"},
        )
    app = create_app(
        database.path,
        deployment_operation_manager=SequencedDeploymentOperations(),
    )

    candidates = api_request(
        app,
        "GET",
        f"/api/deployments/{base_id}/candidates",
    )
    assert candidates.status_code == 200
    candidate_items = candidates.json()["items"]
    assert len(candidate_items) == len(subjects)
    rejected_candidate = next(
        item for item in candidate_items
        if item["id"] == subjects["b"][0]
    )
    assert rejected_candidate["rejection_count"] == 1
    assert rejected_candidate["rejections"][0]["reason"] == "device_memory_exceeded"
    assert rejected_candidate["rejections"][0]["details"]["device_id"] == "GPU0"

    placements = api_request(
        app,
        "GET",
        f"/api/deployments/{base_id}/placements",
    )
    assert placements.status_code == 200
    placement_items = placements.json()["items"]
    assert len(placement_items) == len(subjects)
    first_rows = {
        item["key"]: item
        for item in placement_items[0]["memory"]["rows"]
    }
    assert first_rows["projected_free"]["source"] == "projected"
    assert first_rows["runtime_peak"]["source"] == "runtime"

    runs = api_request(
        app,
        "GET",
        f"/api/deployments/{base_id}/runs",
    )
    assert runs.status_code == 200
    failed_items = [
        item for item in runs.json()["items"]
        if item["status"] == "failed"
    ]
    assert failed_items
    assert any(
        member["status"] == "failed"
        for item in failed_items
        for member in item["members"]
    )

    results = api_request(
        app,
        "GET",
        f"/api/deployments/{base_id}/results",
    )
    assert results.status_code == 200
    assert any(
        row["deployment_status"] == "failed"
        for row in results.json()["rows"]
    )
    assert any(
        row["correctness_valid"] is False
        for row in results.json()["rows"]
        if row["correctness_valid"] is not None
    )

    pareto = api_request(
        app,
        "GET",
        f"/api/deployments/{base_id}/pareto",
        query=[
            (
                "objective",
                "dd:max:deployment.combined_tg_tps"
                "@workload.phase=dd;"
                "workload.member.qwen.depth_tokens=128",
            ),
            (
                "objective",
                "pp:max:deployment.combined_pp_tps"
                "@workload.phase=pp",
            ),
            (
                "objective",
                "retention:max:deployment.min_retention",
            ),
            (
                "objective",
                "context:max:"
                "deployment.total_validated_context_tokens",
            ),
            (
                "objective",
                "headroom:max:"
                "deployment.min_device_headroom_bytes",
            ),
            (
                "objective",
                "power:min:deployment.total_power_avg_w",
            ),
            (
                "constraint",
                "deployment.min_retention:ge:0.75",
            ),
        ],
    )
    assert pareto.status_code == 200
    payload = pareto.json()["result"]
    assert payload["evaluated_count"] >= 2
    assert all(
        item["deployment_placement_id"]
        in {subjects["a"][1], subjects["b"][1], subjects["c"][1]}
        for item in payload["frontier"]
    )


def test_deployment_progress_does_not_double_count_recovered_candidate(
    tmp_path: Path,
) -> None:
    database, subjects = _seed(tmp_path)
    base_id = subjects["a"][0]
    _link_plan(database, base_id, {"a": subjects["a"]})
    with database.session() as connection:
        runs = DeploymentRunRepository(connection)
        failed_run_id = runs.create(
            deployment_candidate_id=subjects["a"][0],
            deployment_placement_id=subjects["a"][1],
            status="ready",
        )
        runs.finish(
            failed_run_id,
            status="failed",
            duration_ns=1,
            failure_kind="server_start_failed",
            failure_details={"error": "recovered later"},
        )

    app = create_app(
        database.path,
        deployment_operation_manager=SequencedDeploymentOperations(),
    )
    response = api_request(
        app,
        "GET",
        f"/api/deployments/{base_id}/progress",
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["planned_candidates"] == 1
    assert payload["completed_candidates"] == 1
    assert payload["failed_candidates"] == 0


def test_deployment_run_control_routes_and_sse(tmp_path: Path) -> None:
    database, subjects = _seed(tmp_path)
    base_id = subjects["a"][0]
    _link_plan(database, base_id, {"a": subjects["a"]})

    running = DeploymentOperationSnapshot(
        id="deployop_sse",
        deployment_candidate_id=base_id,
        deployment_placement_id=subjects["a"][1],
        deployment_run_id=None,
        status="running",
        requested_action=None,
        started_at="2026-10-05T12:00:00Z",
        finished_at=None,
        error=None,
    )
    completed = DeploymentOperationSnapshot(
        id="deployop_sse",
        deployment_candidate_id=base_id,
        deployment_placement_id=subjects["a"][1],
        deployment_run_id=subjects["a"][2],
        status="completed",
        requested_action=None,
        started_at="2026-10-05T12:00:00Z",
        finished_at="2026-10-05T12:00:01Z",
        error=None,
    )
    operations = SequencedDeploymentOperations(
        [running, running, completed]
    )
    app = create_app(
        database.path,
        deployment_operation_manager=operations,
    )

    events = api_request(
        app,
        "GET",
        f"/api/deployments/{base_id}/events",
    )
    assert events.status_code == 200
    assert events.headers["content-type"].startswith("text/event-stream")
    assert events.text.count("event: progress") == 2
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in events.text.splitlines()
        if line.startswith("data: ")
    ]
    assert [item["deployment_status"] for item in payloads] == [
        "running",
        "completed",
    ]
    assert payloads[0]["memory"] is None
    assert payloads[-1]["current_placement_id"] == subjects["a"][1]
    assert payloads[-1]["current_deployment_candidate_id"] == subjects["a"][0]
    assert payloads[-1]["memory"]["deployment_placement_id"] == subjects["a"][1]
    assert "combined_prompt_tps" in payloads[-1]
    assert "combined_decode_tps" in payloads[-1]
    assert "current_combined_prompt_tps" in payloads[-1]
    assert "current_combined_decode_tps" in payloads[-1]
    assert "current_min_retention" in payloads[-1]

    operations = SequencedDeploymentOperations()
    app = create_app(
        database.path,
        deployment_operation_manager=operations,
    )
    started = api_request(
        app,
        "POST",
        f"/api/deployments/{base_id}/run",
        body=_run_body(subjects["a"][1]),
    )
    assert started.status_code == 202
    assert started.json()["operation"]["status"] == "running"

    paused = api_request(
        app,
        "POST",
        f"/api/deployments/{base_id}/pause",
    )
    assert paused.status_code == 202
    assert paused.json()["operation"]["status"] == "pausing"

    resumed = api_request(
        app,
        "POST",
        f"/api/deployments/{base_id}/resume",
        body=_run_body(subjects["a"][1]),
    )
    assert resumed.status_code == 202
    assert resumed.json()["operation"]["status"] == "running"

    cancelled = api_request(
        app,
        "POST",
        f"/api/deployments/{base_id}/cancel",
    )
    assert cancelled.status_code == 202
    assert cancelled.json()["operation"]["status"] == "cancelling"
