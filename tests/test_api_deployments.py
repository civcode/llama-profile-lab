"""HTTP acceptance coverage for V2 deployment resources."""

from __future__ import annotations

from pathlib import Path

from llama_profile_lab.api import create_app
from llama_profile_lab.api.deployment_operations import (
    DeploymentOperationSnapshot,
)
from llama_profile_lab.planning import (
    DeploymentPlanCase,
    DeploymentPlanSummary,
)
from tests.test_api import api_request
from tests.test_deployment_analysis import _seed


class FakeDeploymentPlanner:
    """Return one already-persisted placement without invoking helpers."""

    def __init__(self, placement_id: str, deployment_id: str) -> None:
        self.placement_id = placement_id
        self.deployment_id = deployment_id

    def plan(
        self,
        base_deployment_candidate_id,
        search_space,
        estimator_inputs,
        *,
        timeout_seconds,
    ) -> DeploymentPlanSummary:
        assert base_deployment_candidate_id == self.deployment_id
        assert len(search_space.dimensions) == 1
        assert len(estimator_inputs) == 2
        assert timeout_seconds == 30.0
        return DeploymentPlanSummary(
            base_deployment_candidate_id=self.deployment_id,
            host_id="host-api",
            raw_combinations=1,
            rejected_by_constraints=0,
            duplicate_candidates=0,
            symmetry_reduced=0,
            capability_rejected=0,
            estimate_failed=0,
            memory_rejected=0,
            valid_count=1,
            plan_id="deployplan_api",
            cases=(
                DeploymentPlanCase(
                    deployment_candidate_id=self.deployment_id,
                    deployment_placement_id=self.placement_id,
                    generation={"fixture": "api"},
                ),
            ),
        )


class FakeDeploymentOperations:
    """Deterministic lifecycle snapshots for HTTP control tests."""

    def __init__(self) -> None:
        self.current: DeploymentOperationSnapshot | None = None

    def _set(
        self,
        deployment_id: str,
        placement_id: str,
        *,
        status: str,
        action: str | None = None,
    ) -> DeploymentOperationSnapshot:
        self.current = DeploymentOperationSnapshot(
            id="deployop_api",
            deployment_candidate_id=deployment_id,
            status=status,
            started_at="2026-10-05T18:00:00Z",
            finished_at=(
                "2026-10-05T18:00:01Z"
                if status in {"completed", "paused", "cancelled", "failed"}
                else None
            ),
            requested_action=action,
            deployment_run_id="deployrun_api",
            deployment_placement_id=placement_id,
            phase_count=2,
            error=None,
        )
        return self.current

    def start(self, deployment_id, *, spec):
        return self._set(
            deployment_id,
            spec.deployment_placement_id,
            status="running",
        )

    def pause(self, deployment_id):
        assert self.current is not None
        return self._set(
            deployment_id,
            self.current.deployment_placement_id,
            status="pausing",
            action="pause",
        )

    def cancel(self, deployment_id):
        assert self.current is not None
        return self._set(
            deployment_id,
            self.current.deployment_placement_id,
            status="cancelling",
            action="cancel",
        )

    def resume(self, deployment_id, *, spec=None):
        placement_id = (
            spec.deployment_placement_id
            if spec is not None
            else self.current.deployment_placement_id
        )
        return self._set(
            deployment_id,
            placement_id,
            status="running",
        )

    def snapshot(self, deployment_id):
        if self.current is None:
            return None
        assert self.current.deployment_candidate_id == deployment_id
        return self.current


def _run_request(placement_id: str) -> dict[str, object]:
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
        "readiness_timeout_seconds": 30.0,
    }


def test_deployment_routes_are_registered_without_removing_v1(tmp_path) -> None:
    app = create_app(tmp_path / "api.db")
    paths = {route.path for route in app.routes}

    assert "/api/experiments" in paths
    assert "/api/experiments/{experiment_id}/events" in paths
    assert "/api/deployments" in paths
    assert "/api/deployments/{deployment_id}" in paths
    assert "/api/deployments/{deployment_id}/plan" in paths
    assert "/api/deployments/{deployment_id}/run" in paths
    assert "/api/deployments/{deployment_id}/pause" in paths
    assert "/api/deployments/{deployment_id}/resume" in paths
    assert "/api/deployments/{deployment_id}/cancel" in paths
    assert "/api/deployments/{deployment_id}/progress" in paths
    assert "/api/deployments/{deployment_id}/events" in paths
    assert "/api/deployments/{deployment_id}/candidates" in paths
    assert "/api/deployments/{deployment_id}/placements" in paths
    assert "/api/deployments/{deployment_id}/runs" in paths
    assert "/api/deployments/{deployment_id}/results" in paths
    assert "/api/deployments/{deployment_id}/pareto" in paths


def test_deployment_create_plan_inspection_analysis_and_memory(
    tmp_path: Path,
) -> None:
    database, subjects = _seed(tmp_path)
    deployment_id, placement_id, run_id, _ = subjects["b"]
    with database.session() as connection:
        definition = connection.execute(
            """
            SELECT definition_json
            FROM deployment_candidate
            WHERE id = ?
            """,
            (deployment_id,),
        ).fetchone()
    assert definition is not None

    app = create_app(database.path)
    service = app.state.deployment_api_service
    service.planner = FakeDeploymentPlanner(
        placement_id,
        deployment_id,
    )

    created = api_request(
        app,
        "POST",
        "/api/deployments",
        body={"deployment": __import__("json").loads(definition["definition_json"])},
    )
    assert created.status_code == 201
    assert created.json()["id"] == deployment_id

    fetched = api_request(
        app,
        "GET",
        f"/api/deployments/{deployment_id}",
    )
    assert fetched.status_code == 200
    assert fetched.json()["placement_count"] == 1
    assert fetched.json()["run_count"] == 1

    plan = api_request(
        app,
        "POST",
        f"/api/deployments/{deployment_id}/plan",
        body={
            "search_space": {
                "dimensions": [
                    {
                        "path": "resource_policy.host_ram_reserve_bytes",
                        "values": [0],
                    }
                ]
            },
            "instances": [
                {
                    "instance_id": "qwen",
                    "helper_binary_id": "helper",
                    "model_path": "/models/qwen.gguf",
                },
                {
                    "instance_id": "flash",
                    "helper_binary_id": "helper",
                    "model_path": "/models/flash.gguf",
                },
            ],
            "timeout_seconds": 30.0,
        },
    )
    assert plan.status_code == 200
    assert plan.json()["valid_count"] == 1
    assert plan.json()["cases"][0]["deployment_placement_id"] == placement_id

    placements = api_request(
        app,
        "GET",
        f"/api/deployments/{deployment_id}/placements",
    )
    assert placements.status_code == 200
    assert placements.json()["items"][0]["id"] == placement_id

    memory = api_request(
        app,
        "GET",
        f"/api/deployments/{deployment_id}/placements/{placement_id}/memory",
        query=[("deployment_run_id", run_id)],
    )
    assert memory.status_code == 200
    memory_rows = {
        item["key"]: item for item in memory.json()["rows"]
    }
    assert memory_rows["projected_free"]["source"] == "projected"
    assert memory_rows["runtime_peak"]["source"] == "runtime"

    results = api_request(
        app,
        "GET",
        f"/api/deployments/{deployment_id}/results",
    )
    assert results.status_code == 200
    assert results.json()["rows"]

    pareto = api_request(
        app,
        "GET",
        f"/api/deployments/{deployment_id}/pareto",
        query=[
            (
                "objective",
                "context:max:deployment.total_validated_context_tokens",
            )
        ],
    )
    assert pareto.status_code == 200
    assert pareto.json()["frontier"][0]["deployment_placement_id"] == placement_id


def test_deployment_run_pause_cancel_resume_progress_and_sse(
    tmp_path: Path,
) -> None:
    database, subjects = _seed(tmp_path)
    deployment_id, placement_id, _, _ = subjects["c"]
    app = create_app(database.path)
    operations = FakeDeploymentOperations()
    app.state.deployment_api_service.operations = operations

    run = api_request(
        app,
        "POST",
        f"/api/deployments/{deployment_id}/run",
        body=_run_request(placement_id),
    )
    assert run.status_code == 202
    progress = run.json()
    assert progress["deployment_status"] == "running"
    assert progress["operation"]["deployment_placement_id"] == placement_id
    assert progress["memory"]["deployment_placement_id"] == placement_id
    assert progress["combined_prompt_tps"] is not None
    assert progress["combined_decode_tps"] is not None

    paused = api_request(
        app,
        "POST",
        f"/api/deployments/{deployment_id}/pause",
    )
    assert paused.status_code == 202
    assert paused.json()["operation"]["status"] == "pausing"

    cancelled = api_request(
        app,
        "POST",
        f"/api/deployments/{deployment_id}/cancel",
    )
    assert cancelled.status_code == 202
    assert cancelled.json()["operation"]["status"] == "cancelling"

    resumed = api_request(
        app,
        "POST",
        f"/api/deployments/{deployment_id}/resume",
        body=_run_request(placement_id),
    )
    assert resumed.status_code == 202
    assert resumed.json()["operation"]["status"] == "running"

    operations._set(
        deployment_id,
        placement_id,
        status="completed",
    )
    events = api_request(
        app,
        "GET",
        f"/api/deployments/{deployment_id}/events",
    )
    assert events.status_code == 200
    assert events.headers["content-type"].startswith("text/event-stream")
    assert "event: progress" in events.text
    assert '"deployment_status":"completed"' in events.text


def test_deployment_plan_rejects_empty_instance_inputs(tmp_path: Path) -> None:
    database, subjects = _seed(tmp_path)
    deployment_id = subjects["a"][0]
    app = create_app(database.path)

    response = api_request(
        app,
        "POST",
        f"/api/deployments/{deployment_id}/plan",
        body={
            "search_space": {
                "dimensions": [
                    {
                        "path": "resource_policy.host_ram_reserve_bytes",
                        "values": [0],
                    }
                ]
            },
            "instances": [],
        },
    )
    assert response.status_code == 422
