"""CLI acceptance coverage for V2-M8 deployment workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import llama_profile_lab.cli.main as cli_module
from llama_profile_lab.api.deployment_operations import (
    DeploymentOperationSnapshot,
)
from llama_profile_lab.cli.main import main
from llama_profile_lab.db import (
    DeploymentCandidateRepository,
    DeploymentPlanRepository,
    DeploymentPlacementRepository,
)
from llama_profile_lab.domain import (
    DeploymentSearchDimension,
    DeploymentSearchSpace,
)
from tests.test_deployment_analysis import _seed


def _link_plan(database, base_id, subjects) -> None:
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


class FakeManager:
    def __init__(self, database) -> None:
        self.database = database

    @staticmethod
    def _snapshot(
        deployment_id: str,
        placement_id: str,
        *,
        status: str,
        action: str | None = None,
    ) -> DeploymentOperationSnapshot:
        return DeploymentOperationSnapshot(
            id="deployop_cli",
            deployment_candidate_id=deployment_id,
            deployment_placement_id=placement_id,
            deployment_run_id="deployrun_cli" if status == "completed" else None,
            status=status,
            requested_action=action,
            started_at="2026-10-05T12:00:00Z",
            finished_at=(
                "2026-10-05T12:00:01Z"
                if status == "completed"
                else None
            ),
            error=None,
        )

    def start(self, deployment_id, spec, *, background=True):
        assert background is False
        return self._snapshot(
            deployment_id,
            spec.deployment_placement_id,
            status="completed",
        )

    def resume(self, deployment_id, spec=None, *, background=True):
        assert background is False
        placement_id = (
            "deployplace_resume"
            if spec is None
            else spec.deployment_placement_id
        )
        return self._snapshot(
            deployment_id,
            placement_id,
            status="completed",
        )

    def pause(self, deployment_id):
        return self._snapshot(
            deployment_id,
            "deployplace_control",
            status="pausing",
            action="pause",
        )

    def cancel(self, deployment_id):
        return self._snapshot(
            deployment_id,
            "deployplace_control",
            status="cancelling",
            action="cancel",
        )

    def snapshot(self, deployment_id):
        del deployment_id
        return None


def _run_spec(path: Path, placement_id: str) -> None:
    path.write_text(
        json.dumps(
            {
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
            }
        ),
        encoding="utf-8",
    )


def test_deployment_cli_help_lists_m8_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as captured:
        main(["deployment", "--help"])
    assert captured.value.code == 0
    output = capsys.readouterr().out
    for command in (
        "create",
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


def test_deployment_cli_create_show_placement_results_and_pareto(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database, subjects = _seed(tmp_path)
    base_id = subjects["a"][0]
    _link_plan(database, base_id, subjects)
    with database.session() as connection:
        definition = DeploymentCandidateRepository(connection).get(base_id)
    assert definition is not None

    create_spec = tmp_path / "create.json"
    create_spec.write_text(
        json.dumps(
            definition.model_dump(mode="json", by_alias=True)
        ),
        encoding="utf-8",
    )
    assert main(
        [
            "deployment",
            "create",
            str(create_spec),
            "--database",
            str(database.path),
        ]
    ) == 0
    assert f"Deployment: {base_id}" in capsys.readouterr().out

    assert main(
        [
            "deployment",
            "show",
            base_id,
            "--format",
            "json",
            "--database",
            str(database.path),
        ]
    ) == 0
    assert f'"id":"{base_id}"' in capsys.readouterr().out.replace(" ", "")

    assert main(
        [
            "deployment",
            "placement",
            base_id,
            "--database",
            str(database.path),
        ]
    ) == 0
    placement_output = capsys.readouterr().out
    assert "Memory matrix:" in placement_output
    assert "/projected" in placement_output
    assert "/runtime" in placement_output

    assert main(
        [
            "deployment",
            "results",
            base_id,
            "--format",
            "json",
            "--database",
            str(database.path),
        ]
    ) == 0
    results_output = capsys.readouterr().out
    assert '"deployment_status": "failed"' in results_output
    assert '"correctness_valid": false' in results_output

    objectives = [
        "dd:max:deployment.combined_tg_tps"
        "@workload.phase=dd;workload.member.qwen.depth_tokens=128",
        "pp:max:deployment.combined_pp_tps@workload.phase=pp",
        "retention:max:deployment.min_retention",
        "context:max:deployment.total_validated_context_tokens",
        "headroom:max:deployment.min_device_headroom_bytes",
        "power:min:deployment.total_power_avg_w",
    ]
    argv = [
        "deployment",
        "pareto",
        base_id,
        "--constraint",
        "deployment.min_retention:ge:0.75",
        "--database",
        str(database.path),
    ]
    for objective in objectives:
        argv.extend(["--objective", objective])
    assert main(argv) == 0
    pareto_output = capsys.readouterr().out
    assert "Frontier:" in pareto_output
    assert "Excluded:" in pareto_output


def test_deployment_cli_durable_control_dispatch(
    tmp_path: Path,
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli_module,
        "DeploymentOperationManager",
        FakeManager,
    )
    spec = tmp_path / "run.json"
    _run_spec(spec, "deployplace_cli")

    assert main(
        [
            "deployment",
            "run",
            "deploy_cli",
            str(spec),
            "--database",
            str(tmp_path / "cli.db"),
        ]
    ) == 0
    assert "Status: completed" in capsys.readouterr().out

    assert main(
        [
            "deployment",
            "pause",
            "deploy_cli",
            "--database",
            str(tmp_path / "cli.db"),
        ]
    ) == 0
    assert "Status: pausing" in capsys.readouterr().out

    assert main(
        [
            "deployment",
            "resume",
            "deploy_cli",
            "--spec",
            str(spec),
            "--database",
            str(tmp_path / "cli.db"),
        ]
    ) == 0
    assert "Status: completed" in capsys.readouterr().out

    assert main(
        [
            "deployment",
            "cancel",
            "deploy_cli",
            "--database",
            str(tmp_path / "cli.db"),
        ]
    ) == 0
    assert "Status: cancelling" in capsys.readouterr().out


def test_deployment_cli_missing_resource_exits_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(
        [
            "deployment",
            "show",
            "deploy_missing",
            "--database",
            str(tmp_path / "missing.db"),
        ]
    ) == 2
    assert "deployment not found" in capsys.readouterr().err
