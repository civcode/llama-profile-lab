"""CLI coverage for deployment planner preview and persistence."""

import json
from pathlib import Path

import pytest

import llama_profile_lab.cli.main as cli_module
from llama_profile_lab.cli.main import main
from llama_profile_lab.execution import (
    DeploymentExecutionMember,
    DeploymentExecutionSummary,
)
from llama_profile_lab.planning import DeploymentPlanSummary


class FakeDeploymentExecutor:
    def __init__(self, database) -> None:
        self.database = database

    def execute(
        self,
        placement_id,
        inputs,
        *,
        host,
        readiness_timeout_seconds,
        residency_hold_seconds,
    ) -> DeploymentExecutionSummary:
        assert placement_id == "deployplace_1"
        assert [item.instance_id for item in inputs] == ["qwen", "flash"]
        assert [str(item.model_path) for item in inputs] == [
            "/models/qwen.gguf",
            "/models/flash.gguf",
        ]
        assert host == "127.0.0.1"
        assert readiness_timeout_seconds == 45.0
        assert residency_hold_seconds == 2.5
        return DeploymentExecutionSummary(
            run_id="deployrun_1",
            deployment_candidate_id="deploy_1",
            deployment_placement_id=placement_id,
            status="completed",
            members=(
                DeploymentExecutionMember(
                    instance_id="qwen",
                    endpoint="http://127.0.0.1:41001",
                    pid=101,
                    ready_at="2026-10-05T12:00:00Z",
                    exit_code=0,
                    forced_kill=False,
                ),
                DeploymentExecutionMember(
                    instance_id="flash",
                    endpoint="http://127.0.0.1:41002",
                    pid=102,
                    ready_at="2026-10-05T12:00:01Z",
                    exit_code=0,
                    forced_kill=False,
                ),
            ),
            runtime_memory={},
        )


class FakeDeploymentPlannerService:
    def __init__(self, database) -> None:
        self.database = database

    def preview(
        self,
        base_id,
        search_space,
        inputs,
        *,
        timeout_seconds,
    ) -> DeploymentPlanSummary:
        assert base_id == "deploy_base"
        assert len(search_space.dimensions) == 1
        assert [item.instance_id for item in inputs] == ["qwen", "flash"]
        assert timeout_seconds == 120.0
        return _summary(plan_id=None)

    def plan(
        self,
        base_id,
        search_space,
        inputs,
        *,
        timeout_seconds,
    ) -> DeploymentPlanSummary:
        assert base_id == "deploy_base"
        assert len(search_space.dimensions) == 1
        assert len(inputs) == 2
        assert timeout_seconds == 120.0
        return _summary(plan_id="deployplan_1")


def _summary(*, plan_id: str | None) -> DeploymentPlanSummary:
    return DeploymentPlanSummary(
        base_deployment_candidate_id="deploy_base",
        host_id="host_test",
        raw_combinations=4,
        rejected_by_constraints=1,
        duplicate_candidates=0,
        symmetry_reduced=0,
        capability_rejected=1,
        estimate_failed=0,
        memory_rejected=1,
        valid_count=1,
        plan_id=plan_id,
    )


def _write_spec(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "base_deployment_candidate_id": "deploy_base",
                "search_space": {
                    "dimensions": [
                        {
                            "path": "instances.qwen.context.size",
                            "values": [8192, 16384],
                        }
                    ]
                },
                "instances": [
                    {
                        "instance_id": "qwen",
                        "helper_binary_id": "bin_helper",
                        "model_path": "/models/qwen.gguf",
                    },
                    {
                        "instance_id": "flash",
                        "helper_binary_id": "bin_helper",
                        "model_path": "/models/flash.gguf",
                    },
                ],
                "timeout_seconds": 120,
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("command", "expected"),
    (
        ("preview", "Valid: 1"),
        ("plan", "Plan: deployplan_1"),
    ),
)
def test_deployment_cli_preview_and_plan(
    tmp_path: Path,
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    expected: str,
) -> None:
    spec = tmp_path / "deployment.json"
    _write_spec(spec)
    monkeypatch.setattr(
        cli_module,
        "DeploymentPlannerService",
        FakeDeploymentPlannerService,
    )

    assert main(
        [
            "deployment",
            command,
            str(spec),
            "--database",
            str(tmp_path / "benchmarks.db"),
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "Raw combinations: 4" in output
    assert "Capability rejected: 1" in output
    assert "Memory rejected: 1" in output
    assert expected in output


def test_deployment_cli_execute(
    tmp_path: Path,
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = tmp_path / "execute.json"
    spec.write_text(
        json.dumps(
            {
                "deployment_placement_id": "deployplace_1",
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
                "readiness_timeout_seconds": 45,
                "residency_hold_seconds": 2.5,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli_module,
        "DeploymentExecutor",
        FakeDeploymentExecutor,
    )

    assert main(
        [
            "deployment",
            "execute",
            str(spec),
            "--database",
            str(tmp_path / "benchmarks.db"),
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "Deployment run: deployrun_1" in output
    assert "Status: completed" in output
    assert "qwen: http://127.0.0.1:41001" in output
    assert "flash: http://127.0.0.1:41002" in output


def test_deployment_cli_rejects_invalid_spec(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = tmp_path / "invalid.json"
    spec.write_text("{}", encoding="utf-8")

    assert main(
        [
            "deployment",
            "preview",
            str(spec),
            "--database",
            str(tmp_path / "benchmarks.db"),
        ]
    ) == 2
    assert "base_deployment_candidate_id" in capsys.readouterr().err
