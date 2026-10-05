"""Coordinated V2 deployment promotion coverage."""

from __future__ import annotations

import json
from pathlib import Path

from llama_profile_lab.api import create_app
from llama_profile_lab.db import (
    Database,
    DeploymentCandidateRepository,
    DeploymentPlacementRepository,
    DeploymentPromotionRepository,
    DeploymentRunRepository,
    EnvironmentRepository,
    PlacementRepository,
)
from llama_profile_lab.domain import (
    AbsoluteDepth,
    DeploymentCandidate,
    DeploymentDeviceAllocation,
    DeploymentInstancePlacement,
    DeploymentPlacement,
    DeploymentWorkloadMix,
    GpuTelemetrySample,
    HostResourcePolicy,
    MeasurementPolicy,
    ModelInstanceCandidate,
    PlacementDeviceMemory,
    PrefillSuiteCase,
    ResolvedPlacement,
    SearchDimension,
    SearchSpace,
    WorkloadSuite,
)
from llama_profile_lab.llama import sha256_file
from tests.test_api import api_request


def _source_experiment(app, profile_id: str) -> tuple[str, str, str]:
    profile = api_request(app, "GET", f"/api/profiles/{profile_id}").json()
    batch_size = profile["candidate"]["compute"]["batch_size"]
    response = api_request(
        app,
        "POST",
        "/api/experiments",
        body={
            "name": f"{profile_id} source",
            "base_candidate": profile["candidate"],
            "search_space": SearchSpace(
                dimensions=(
                    SearchDimension(
                        path="compute.batch_size",
                        values=(batch_size,),
                    ),
                )
            ).model_dump(mode="json", by_alias=True),
            "workload_suite": WorkloadSuite(
                id=f"suite-{profile_id}",
                cases=(
                    PrefillSuiteCase(
                        prompt_tokens=64,
                        depth=AbsoluteDepth(tokens=0),
                    ),
                ),
            ).model_dump(mode="json", by_alias=True),
            "measurement_policy": MeasurementPolicy(
                repetitions=1
            ).model_dump(mode="json", by_alias=True),
        },
    )
    assert response.status_code == 201
    experiment_id = response.json()["id"]
    assert api_request(
        app,
        "POST",
        f"/api/experiments/{experiment_id}/plan",
    ).status_code == 200
    candidate = api_request(
        app,
        "GET",
        f"/api/experiments/{experiment_id}/candidates",
    ).json()["items"][0]
    return experiment_id, candidate["id"], response.json()["workload_suite_id"]


def _seed_deployment(
    database: Database,
    tmp_path: Path,
    *,
    qwen_candidate_id: str,
    flash_candidate_id: str,
    workload_suite_id: str,
) -> tuple[str, str, str, str]:
    server = tmp_path / "llama-server"
    helper = tmp_path / "llama-memory-estimator"
    server.write_bytes(b"server")
    helper.write_bytes(b"helper")

    with database.session() as connection:
        environment = EnvironmentRepository(connection)
        server_id = environment.put_binary(
            sha256=sha256_file(server),
            kind="llama-server",
            path=str(server),
            size_bytes=server.stat().st_size,
            mtime_ns=server.stat().st_mtime_ns,
            capabilities={},
        )
        helper_id = environment.put_binary(
            sha256=sha256_file(helper),
            kind="llama-memory-estimator",
            path=str(helper),
            size_bytes=helper.stat().st_size,
            mtime_ns=helper.stat().st_mtime_ns,
            capabilities={},
        )
        host_id = environment.put_host(
            hostname="promotion-host",
            hardware_fingerprint="promotion-host-v2",
            cpu={},
            ram_bytes=64 * 1024**3,
            gpus=[],
            os_info={},
        )

        deployment = DeploymentCandidate(
            instances=(
                ModelInstanceCandidate(
                    instance_id="qwen",
                    candidate_id=qwen_candidate_id,
                    role="primary",
                    model_artifact_id="artifact:qwen",
                    binary_id=server_id,
                    server_identity="qwen",
                ),
                ModelInstanceCandidate(
                    instance_id="flash",
                    candidate_id=flash_candidate_id,
                    role="secondary",
                    model_artifact_id="artifact:flash",
                    binary_id=server_id,
                    server_identity="flash",
                ),
            ),
            resource_policy=HostResourcePolicy(),
            workload_mix=DeploymentWorkloadMix(
                workload_suite_id=workload_suite_id,
            ),
        )
        deployment_id = DeploymentCandidateRepository(connection).put(deployment)

        placements = PlacementRepository(connection)
        qwen_resolved = placements.put_resolved(
            placement_hash="q" * 64,
            candidate_id=qwen_candidate_id,
            host_id=host_id,
            binary_id=server_id,
            fit_attempt_id=None,
            placement=ResolvedPlacement(
                production_context_size=8192,
                n_gpu_layers=32,
                devices=("CUDA0", "Vulkan0"),
                tensor_split=(0.7, 0.3),
            ),
            request={},
            argv=(),
            stdout="",
            stderr="",
            exit_code=0,
        )
        flash_resolved = placements.put_resolved(
            placement_hash="f" * 64,
            candidate_id=flash_candidate_id,
            host_id=host_id,
            binary_id=server_id,
            fit_attempt_id=None,
            placement=ResolvedPlacement(
                production_context_size=4096,
                n_gpu_layers=20,
                devices=("CUDA0", "Vulkan0"),
                tensor_split=(0.3, 0.7),
            ),
            request={},
            argv=(),
            stdout="",
            stderr="",
            exit_code=0,
        )

        per_instance = 2 * 1024**3
        total = 16 * 1024**3
        memory = tuple(
            PlacementDeviceMemory(
                instance_id=instance_id,
                device_id=device_id,
                model_bytes=per_instance // 2,
                context_bytes=per_instance // 4,
                compute_bytes=per_instance // 4,
                total_bytes=per_instance,
                device_total_bytes=total,
                device_free_bytes=8 * 1024**3,
                source="test-estimator",
            )
            for instance_id in ("qwen", "flash")
            for device_id in ("gpu0", "gpu1")
        )
        placement = DeploymentPlacement(
            deployment_candidate_id=deployment_id,
            host_id=host_id,
            instance_placements=(
                DeploymentInstancePlacement(
                    instance_id="qwen",
                    resolved_placement_id=qwen_resolved,
                ),
                DeploymentInstancePlacement(
                    instance_id="flash",
                    resolved_placement_id=flash_resolved,
                ),
            ),
            device_memory=memory,
            device_allocations=(
                DeploymentDeviceAllocation(
                    device_id="gpu0",
                    projected_bytes=4 * 1024**3,
                    reserved_margin_bytes=1024**3,
                    device_total_bytes=total,
                    projected_free_bytes=11 * 1024**3,
                ),
                DeploymentDeviceAllocation(
                    device_id="gpu1",
                    projected_bytes=4 * 1024**3,
                    reserved_margin_bytes=1024**3,
                    device_total_bytes=total,
                    projected_free_bytes=11 * 1024**3,
                ),
            ),
            feasibility="feasible",
        )
        placement_id = DeploymentPlacementRepository(connection).put(placement)

        for ordinal, candidate_id in enumerate(
            (qwen_candidate_id, flash_candidate_id)
        ):
            attempt_id = f"memattempt-{ordinal}"
            cache_hash = f"memory-cache-{ordinal}"
            connection.execute(
                """
                INSERT INTO memory_estimate_attempt(
                    id, cache_hash, candidate_id, host_id, helper_binary_id,
                    model_artifact_id, request_json, argv_json, status,
                    started_at, finished_at, duration_ns, exit_code
                )
                VALUES (?, ?, ?, ?, ?, ?, '{}', '[]', 'completed',
                        '2026-10-05T00:00:00Z', '2026-10-05T00:00:01Z', 1, 0)
                """,
                (
                    attempt_id,
                    cache_hash,
                    candidate_id,
                    host_id,
                    helper_id,
                    f"artifact:{ordinal}",
                ),
            )
            connection.execute(
                """
                INSERT INTO memory_estimate(
                    id, cache_hash, attempt_id, candidate_id, host_id,
                    helper_binary_id, model_artifact_id, identity_json,
                    result_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, '{}', '{}')
                """,
                (
                    f"mem-{ordinal}",
                    cache_hash,
                    attempt_id,
                    candidate_id,
                    host_id,
                    helper_id,
                    f"artifact:{ordinal}",
                ),
            )

        runs = DeploymentRunRepository(connection)
        run_id = runs.create(
            deployment_candidate_id=deployment_id,
            deployment_placement_id=placement_id,
            status="ready",
        )
        runs.add_gpu_sample(
            run_id,
            timestamp_ns=1_000_000_000,
            gpus=(
                GpuTelemetrySample(
                    device="CUDA0",
                    stable_device_key="gpu0",
                    vram_total_bytes=total,
                    vram_used_bytes=5 * 1024**3,
                    power_w=120.0,
                ),
                GpuTelemetrySample(
                    device="Vulkan0",
                    stable_device_key="gpu1",
                    vram_total_bytes=total,
                    vram_used_bytes=4 * 1024**3,
                    power_w=80.0,
                ),
            ),
        )
        for phase in ("dd", "pp", "pd", "dp"):
            case_id = f"case-{phase}"
            connection.execute(
                """
                INSERT INTO deployment_concurrent_workload_case(
                    id, deployment_candidate_id, case_hash, phase,
                    definition_json
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    deployment_id,
                    f"hash-{phase}",
                    phase,
                    json.dumps({"phase": phase}),
                ),
            )
            connection.execute(
                """
                INSERT INTO deployment_workload_run(
                    id, deployment_run_id, workload_case_id, phase, status,
                    quality, correctness_valid, combined_prompt_tps,
                    combined_decode_tps, min_retention
                )
                VALUES (?, ?, ?, ?, 'completed', 'clean', 1, 100, 25, 0.8)
                """,
                (f"work-{phase}", run_id, case_id, phase),
            )
        runs.finish(run_id, status="completed", duration_ns=1_000_000)

    return deployment_id, placement_id, server_id, helper_id


def test_coordinated_deployment_promotion_requires_complete_sources_and_persists(
    tmp_path: Path,
) -> None:
    launcher = tmp_path / "workstation.json"
    source_payload = {
        "binaries": {"custom": "/opt/llama-server"},
        "models": {
            "qwen": {
                "binary": "custom",
                "profiles": [],
                "model": "/models/qwen.gguf",
                "args": {
                    "--ctx-size": 8192,
                    "--batch-size": 2048,
                    "--ubatch-size": 512,
                },
            },
            "flash": {
                "binary": "custom",
                "profiles": [],
                "model": "/models/flash.gguf",
                "args": {
                    "--ctx-size": 4096,
                    "--batch-size": 1024,
                    "--ubatch-size": 256,
                },
            },
        },
    }
    launcher.write_text(json.dumps(source_payload), encoding="utf-8")
    database_path = tmp_path / "deployment-promotion.db"
    app = create_app(database_path, launcher_config_path=launcher)

    qwen_exp, qwen_candidate, suite_id = _source_experiment(app, "qwen")
    flash_exp, flash_candidate, _ = _source_experiment(app, "flash")
    deployment_id, placement_id, _, helper_id = _seed_deployment(
        Database(database_path),
        tmp_path,
        qwen_candidate_id=qwen_candidate,
        flash_candidate_id=flash_candidate,
        workload_suite_id=suite_id,
    )

    partial = api_request(
        app,
        "POST",
        f"/api/deployments/{deployment_id}/promote",
        body={
            "deployment_placement_id": placement_id,
            "sources": [
                {
                    "instance_id": "qwen",
                    "experiment_id": qwen_exp,
                    "source_profile_id": "qwen",
                }
            ],
        },
    )
    assert partial.status_code == 409
    assert "source provenance for every instance" in partial.json()["detail"]

    response = api_request(
        app,
        "POST",
        f"/api/deployments/{deployment_id}/promote",
        body={
            "deployment_placement_id": placement_id,
            "sources": [
                {
                    "instance_id": "qwen",
                    "experiment_id": qwen_exp,
                    "source_profile_id": "qwen",
                },
                {
                    "instance_id": "flash",
                    "experiment_id": flash_exp,
                    "source_profile_id": "flash",
                },
            ],
        },
    )
    assert response.status_code == 201
    proposal = response.json()
    assert {item["instance_id"] for item in proposal["sources"]} == {
        "qwen",
        "flash",
    }
    assert proposal["deployment_placement_id"] == placement_id
    assert proposal["evidence"]["deployment_run_id"]
    assert {item["helper_binary_id"] for item in proposal["evidence"]["binaries"]} == {
        helper_id
    }
    assert len(proposal["evidence"]["concurrent_validation"]) == 4
    assert all(
        item["correctness_valid"]
        for item in proposal["evidence"]["concurrent_validation"]
    )
    assert {item["instance_id"] for item in proposal["changes"]} == {
        "qwen",
        "flash",
    }
    assert all(item["changes"] for item in proposal["changes"])
    assert '"--device"' in proposal["patch"]
    assert '"--tensor-split"' in proposal["patch"]
    assert json.loads(launcher.read_text(encoding="utf-8")) == source_payload

    with Database(database_path).session() as connection:
        saved = DeploymentPromotionRepository(connection).get(proposal["id"])
    assert saved is not None
    assert saved.deployment_placement_id == placement_id
    assert len(saved.sources) == 2

    with Database(database_path).session() as connection:
        connection.execute(
            """
            UPDATE deployment_workload_run
            SET correctness_valid = 0,
                status = 'invalid',
                quality = 'correctness_invalid',
                failure_kind = 'output_validation_failed'
            WHERE deployment_run_id = ?
              AND phase = 'dp'
            """,
            (proposal["evidence"]["deployment_run_id"],),
        )
    invalid = api_request(
        app,
        "POST",
        f"/api/deployments/{deployment_id}/promote",
        body={
            "deployment_placement_id": placement_id,
            "sources": [
                {
                    "instance_id": "qwen",
                    "experiment_id": qwen_exp,
                    "source_profile_id": "qwen",
                },
                {
                    "instance_id": "flash",
                    "experiment_id": flash_exp,
                    "source_profile_id": "flash",
                },
            ],
        },
    )
    assert invalid.status_code == 409
    assert "correctness-valid completed" in invalid.json()["detail"]
    assert "dp" in invalid.json()["detail"]

    with Database(database_path).session() as connection:
        connection.execute(
            """
            UPDATE deployment_workload_run
            SET correctness_valid = 1,
                status = 'completed',
                quality = 'clean',
                failure_kind = NULL
            WHERE deployment_run_id = ?
              AND phase = 'dp'
            """,
            (proposal["evidence"]["deployment_run_id"],),
        )
    drifted_payload = json.loads(json.dumps(source_payload))
    drifted_payload["models"]["qwen"]["args"]["--ctx-size"] = 16384
    launcher.write_text(json.dumps(drifted_payload), encoding="utf-8")
    drifted = api_request(
        app,
        "POST",
        f"/api/deployments/{deployment_id}/promote",
        body={
            "deployment_placement_id": placement_id,
            "sources": [
                {
                    "instance_id": "qwen",
                    "experiment_id": qwen_exp,
                    "source_profile_id": "qwen",
                },
                {
                    "instance_id": "flash",
                    "experiment_id": flash_exp,
                    "source_profile_id": "flash",
                },
            ],
        },
    )
    assert drifted.status_code == 409
    assert "changed since source experiment" in drifted.json()["detail"]
