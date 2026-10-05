"""CLI coverage for deployment planner preview and persistence."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import llama_profile_lab.cli.main as cli_module
from llama_profile_lab.cli.main import main
from llama_profile_lab.execution import (
    ConcurrentDeploymentSummary,
    ConcurrentPhaseSummary,
    DeploymentExecutionMember,
    DeploymentExecutionSummary,
)
from llama_profile_lab.planning import DeploymentPlanSummary
from tests.test_deployment_analysis import _seed


class FakeConcurrentDeploymentExecutor:
    def __init__(self, database) -> None:
        self.database = database

    def execute(
        self,
        placement_id,
        inputs,
        *,
        standalone_baselines,
        host,
        readiness_timeout_seconds,
        cancel_event=None,
    ) -> ConcurrentDeploymentSummary:
        assert placement_id == "deployplace_1"
        assert [item.instance_id for item in inputs] == ["qwen", "flash"]
        assert len(standalone_baselines) == 2
        assert standalone_baselines[0].instance_id == "qwen"
        assert standalone_baselines[0].latency_ms == 1000.0
        assert host == "127.0.0.1"
        assert readiness_timeout_seconds == 45.0
        execution = FakeDeploymentExecutor(self.database).execute(
            placement_id,
            inputs,
            host=host,
            readiness_timeout_seconds=readiness_timeout_seconds,
            residency_hold_seconds=0.0,
        )
        return ConcurrentDeploymentSummary(
            deployment_run_id=execution.run_id,
            deployment_placement_id=placement_id,
            phases=(
                ConcurrentPhaseSummary(
                    run_id="conc_1",
                    workload_case_id="work_dd",
                    phase="dd",
                    quality="clean",
                    combined_prompt_tps=None,
                    combined_decode_tps=42.0,
                    min_retention=0.8,
                ),
                ConcurrentPhaseSummary(
                    run_id="conc_2",
                    workload_case_id="work_pp",
                    phase="pp",
                    quality="clean",
                    combined_prompt_tps=123.0,
                    combined_decode_tps=None,
                    min_retention=0.75,
                ),
            ),
            execution=execution,
        )


class FakeDeploymentPlacementRepository:
    def __init__(self, connection) -> None:
        self.connection = connection

    def record(self, placement_id):
        assert placement_id == "deployplace_1"
        return SimpleNamespace(deployment_candidate_id="deploy_1")


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


def test_deployment_cli_benchmark(
    tmp_path: Path,
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = tmp_path / "benchmark.json"
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
                "standalone_baselines": [
                    {
                        "instance_id": "qwen",
                        "mode": "decode",
                        "prompt_tokens": 0,
                        "generate_tokens": 32,
                        "depth_tokens": 128,
                        "throughput_tps": 20.0,
                        "latency_ms": 1000.0,
                    },
                    {
                        "instance_id": "flash",
                        "mode": "decode",
                        "prompt_tokens": 0,
                        "generate_tokens": 32,
                        "depth_tokens": 128,
                        "throughput_tps": 22.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli_module,
        "ConcurrentDeploymentExecutor",
        FakeConcurrentDeploymentExecutor,
    )

    assert main(
        [
            "deployment",
            "benchmark",
            str(spec),
            "--database",
            str(tmp_path / "benchmarks.db"),
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "Concurrent phases: 2" in output
    assert "DD: quality=clean" in output
    assert "tg_tps=42.000" in output
    assert "PP: quality=clean" in output
    assert "pp_tps=123.000" in output


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


@pytest.mark.parametrize("command", ("run", "resume"))
def test_deployment_cli_run_and_resume_aliases(
    tmp_path: Path,
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    spec = tmp_path / f"{command}.json"
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
                "standalone_baselines": [
                    {
                        "instance_id": "qwen",
                        "mode": "decode",
                        "prompt_tokens": 0,
                        "generate_tokens": 32,
                        "depth_tokens": 128,
                        "throughput_tps": 20.0,
                        "latency_ms": 1000.0,
                    },
                    {
                        "instance_id": "flash",
                        "mode": "decode",
                        "prompt_tokens": 0,
                        "generate_tokens": 32,
                        "depth_tokens": 128,
                        "throughput_tps": 22.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli_module,
        "ConcurrentDeploymentExecutor",
        FakeConcurrentDeploymentExecutor,
    )
    monkeypatch.setattr(
        cli_module,
        "DeploymentPlacementRepository",
        FakeDeploymentPlacementRepository,
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
    assert "Concurrent phases: 2" in output
    assert "DD: quality=clean" in output


@pytest.mark.parametrize("action", ("pause", "cancel"))
def test_deployment_cli_cross_process_control(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    action: str,
) -> None:
    database, subjects = _seed(tmp_path)
    deployment_id, _, run_id, _ = subjects["b"]
    with database.session() as connection:
        connection.execute(
            """
            UPDATE deployment_run
            SET status = 'running', finished_at = NULL
            WHERE id = ?
            """,
            (run_id,),
        )

    assert main(
        [
            "deployment",
            action,
            deployment_id,
            "--database",
            str(database.path),
        ]
    ) == 0
    output = capsys.readouterr().out
    assert f"Requested {action}" in output

    with database.session() as connection:
        row = connection.execute(
            """
            SELECT action
            FROM deployment_control_request
            WHERE deployment_candidate_id = ?
            """,
            (deployment_id,),
        ).fetchone()
    assert row is not None
    assert row["action"] == action


def test_deployment_cli_control_requires_active_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database, subjects = _seed(tmp_path)
    deployment_id = subjects["c"][0]

    assert main(
        [
            "deployment",
            "pause",
            deployment_id,
            "--database",
            str(database.path),
        ]
    ) == 2
    assert "no active deployment run" in capsys.readouterr().err


def test_deployment_cli_show_placement_results_and_pareto(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database, subjects = _seed(tmp_path)
    deployment_id, placement_id, run_id, _ = subjects["b"]

    assert main(
        [
            "deployment",
            "show",
            deployment_id,
            "--format",
            "json",
            "--database",
            str(database.path),
        ]
    ) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["id"] == deployment_id
    assert shown["placement_count"] == 1

    assert main(
        [
            "deployment",
            "placement",
            placement_id,
            "--run",
            run_id,
            "--format",
            "json",
            "--database",
            str(database.path),
        ]
    ) == 0
    placement = json.loads(capsys.readouterr().out)
    assert placement["record"]["id"] == placement_id
    assert placement["runtime_matrix"]["deployment_run_id"] == run_id

    assert main(
        [
            "deployment",
            "results",
            deployment_id,
            "--format",
            "json",
            "--database",
            str(database.path),
        ]
    ) == 0
    results = json.loads(capsys.readouterr().out)
    assert results
    assert all(
        item["deployment_candidate_id"] == deployment_id
        for item in results
    )

    assert main(
        [
            "deployment",
            "pareto",
            deployment_id,
            "--objective",
            "context:max:deployment.total_validated_context_tokens",
            "--format",
            "json",
            "--database",
            str(database.path),
        ]
    ) == 0
    pareto = json.loads(capsys.readouterr().out)
    assert pareto["evaluated_count"] == 1
    assert pareto["frontier"][0]["deployment_placement_id"] == placement_id


def test_deployment_cli_create_is_idempotent_for_existing_definition(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database, subjects = _seed(tmp_path)
    deployment_id = subjects["a"][0]
    with database.session() as connection:
        row = connection.execute(
            """
            SELECT definition_json
            FROM deployment_candidate
            WHERE id = ?
            """,
            (deployment_id,),
        ).fetchone()
    assert row is not None
    spec = tmp_path / "deployment-create.json"
    spec.write_text(str(row["definition_json"]), encoding="utf-8")

    assert main(
        [
            "deployment",
            "create",
            str(spec),
            "--database",
            str(database.path),
        ]
    ) == 0
    assert capsys.readouterr().out.strip() == deployment_id


def test_deployment_cli_help_lists_m8_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["deployment", "--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    for command in (
        "create",
        "plan",
        "placement",
        "run",
        "pause",
        "resume",
        "cancel",
        "show",
        "results",
        "pareto",
    ):
        assert command in output
