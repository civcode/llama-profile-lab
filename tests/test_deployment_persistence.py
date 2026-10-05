"""V2 deployment persistence and relationship tests."""

from pathlib import Path
from sqlite3 import IntegrityError

import pytest

from llama_profile_lab.db import (
    CandidateRepository,
    Database,
    DeploymentCandidateRepository,
    DeploymentPlacementRepository,
    DeploymentRunRepository,
    EnvironmentRepository,
    PlacementRepository,
    WorkloadSuiteRepository,
)
from llama_profile_lab.domain import (
    AbsoluteDepth,
    Candidate,
    ComputeConfig,
    ContextConfig,
    DeploymentCandidate,
    DeploymentDeviceAllocation,
    DeploymentInstancePlacement,
    DeploymentPlacement,
    DeploymentPlacementRequest,
    DeploymentWorkloadMix,
    FitConfig,
    HostResourcePolicy,
    ModelInstanceCandidate,
    ModelSelection,
    PlacementConfig,
    PlacementDeviceMemory,
    PrefillSuiteCase,
    ResolvedPlacement,
    WorkloadSuite,
)


def make_candidate(model_id: str) -> Candidate:
    return Candidate(
        model=ModelSelection(target_model_id=model_id),
        context=ContextConfig(
            size=131072,
            cache_type_k="f16",
            cache_type_v="f16",
        ),
        compute=ComputeConfig(
            flash_attn="on",
            batch_size=4096,
            ubatch_size=2048,
        ),
        placement=PlacementConfig(
            mode="fit",
            fit=FitConfig(target_mib=256, min_context=4096),
        ),
    )


def seed_deployment(
    database: Database,
) -> tuple[str, DeploymentCandidate, str, str, str]:
    with database.session() as connection:
        candidates = CandidateRepository(connection)
        qwen_id = candidates.put(make_candidate("model:qwen"))
        flash_id = candidates.put(make_candidate("model:flash"))

        suite_id = WorkloadSuiteRepository(connection).put(
            WorkloadSuite(
                id="deployment-base",
                cases=(
                    PrefillSuiteCase(
                        prompt_tokens=2048,
                        depth=AbsoluteDepth(tokens=0),
                    ),
                ),
            )
        )

        binary_id = EnvironmentRepository(connection).put_binary(
            sha256="d" * 64,
            kind="llama-server",
            path="/opt/llama-server",
            size_bytes=1234,
            mtime_ns=5678,
        )

        deployment = DeploymentCandidate(
            instances=(
                ModelInstanceCandidate(
                    instance_id="qwen",
                    candidate_id=qwen_id,
                    role="primary",
                    model_artifact_id="artifact:qwen",
                    binary_id=binary_id,
                    requested_placement=DeploymentPlacementRequest(
                        devices=("CUDA0", "Vulkan0"),
                        tensor_split=(3.0, 1.0),
                    ),
                    server_identity="qwen",
                ),
                ModelInstanceCandidate(
                    instance_id="flash",
                    candidate_id=flash_id,
                    role="fast",
                    model_artifact_id="artifact:flash",
                    binary_id=binary_id,
                    requested_placement=DeploymentPlacementRequest(
                        devices=("CUDA0", "Vulkan0"),
                        tensor_split=(1.0, 3.0),
                    ),
                    server_identity="flash",
                ),
            ),
            resource_policy=HostResourcePolicy(
                device_memory_margin_bytes={
                    "CUDA0": 100,
                    "Vulkan0": 100,
                },
                allowed_devices=("CUDA0", "Vulkan0"),
            ),
            workload_mix=DeploymentWorkloadMix(workload_suite_id=suite_id),
        )
        deployment_id = DeploymentCandidateRepository(connection).put(deployment)

    return deployment_id, deployment, qwen_id, flash_id, binary_id


def test_deployment_candidate_round_trips_across_reopen(tmp_path: Path) -> None:
    database = Database(tmp_path / "deployment.db")
    deployment_id, deployment, _, _, _ = seed_deployment(database)

    with database.session() as connection:
        repository = DeploymentCandidateRepository(connection)

        assert repository.put(deployment) == deployment_id
        assert repository.get(deployment_id) == deployment

        record = repository.record(deployment_id)
        assert record is not None
        assert record.deployment_hash == deployment.content_hash()

        instances = repository.instances(deployment_id)
        assert [item.instance_id for item in instances] == ["flash", "qwen"]
        assert [item.ordinal for item in instances] == [0, 1]


def test_deployment_insert_rolls_back_on_invalid_binary_reference(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "deployment-invalid.db")
    deployment_id, deployment, _, _, _ = seed_deployment(database)
    bad = deployment.model_copy(
        update={
            "instances": tuple(
                instance.model_copy(update={"binary_id": "bin_missing"})
                for instance in deployment.instances
            )
        }
    )

    with database.session() as connection:
        repository = DeploymentCandidateRepository(connection)
        with pytest.raises(IntegrityError):
            repository.put(bad)

        assert repository.find_by_hash(bad.content_hash()) is None
        assert repository.get(deployment_id) == deployment


def test_deployment_placement_memory_and_run_members_persist(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "deployment-placement.db")
    deployment_id, deployment, qwen_id, flash_id, binary_id = seed_deployment(
        database
    )

    with database.session() as connection:
        environment = EnvironmentRepository(connection)
        host_id = environment.put_host(
            hostname="mixed-host",
            hardware_fingerprint="cpu:test|cuda:test|amd:test",
            cpu={"model": "test"},
            ram_bytes=128 * 1024**3,
            gpus=[
                {"device": "CUDA0", "name": "NVIDIA"},
                {"device": "Vulkan0", "name": "AMD"},
            ],
            os_info={"kernel": "test"},
        )

        placements = PlacementRepository(connection)
        qwen_placement_id = placements.put_resolved(
            placement_hash="1" * 64,
            candidate_id=qwen_id,
            host_id=host_id,
            binary_id=binary_id,
            fit_attempt_id=None,
            placement=ResolvedPlacement(
                production_context_size=131072,
                n_gpu_layers=48,
                devices=("CUDA0", "Vulkan0"),
                tensor_split=(3.0, 1.0),
            ),
            request={},
            argv=(),
            stdout="",
            stderr="",
            exit_code=0,
        )
        flash_placement_id = placements.put_resolved(
            placement_hash="2" * 64,
            candidate_id=flash_id,
            host_id=host_id,
            binary_id=binary_id,
            fit_attempt_id=None,
            placement=ResolvedPlacement(
                production_context_size=131072,
                n_gpu_layers=32,
                devices=("CUDA0", "Vulkan0"),
                tensor_split=(1.0, 3.0),
            ),
            request={},
            argv=(),
            stdout="",
            stderr="",
            exit_code=0,
        )

        placement = DeploymentPlacement(
            deployment_candidate_id=deployment_id,
            host_id=host_id,
            instance_placements=(
                DeploymentInstancePlacement(
                    instance_id="qwen",
                    resolved_placement_id=qwen_placement_id,
                ),
                DeploymentInstancePlacement(
                    instance_id="flash",
                    resolved_placement_id=flash_placement_id,
                ),
            ),
            device_memory=(
                PlacementDeviceMemory(
                    instance_id="qwen",
                    device_id="CUDA0",
                    model_bytes=300,
                    context_bytes=100,
                    compute_bytes=40,
                    total_bytes=440,
                    device_total_bytes=1000,
                    device_free_bytes=900,
                    source="fixture",
                ),
                PlacementDeviceMemory(
                    instance_id="qwen",
                    device_id="Vulkan0",
                    model_bytes=180,
                    context_bytes=80,
                    compute_bytes=20,
                    total_bytes=280,
                    device_total_bytes=1000,
                    device_free_bytes=900,
                    source="fixture",
                ),
                PlacementDeviceMemory(
                    instance_id="flash",
                    device_id="CUDA0",
                    model_bytes=100,
                    context_bytes=40,
                    compute_bytes=20,
                    total_bytes=160,
                    device_total_bytes=1000,
                    device_free_bytes=900,
                    source="fixture",
                ),
                PlacementDeviceMemory(
                    instance_id="flash",
                    device_id="Vulkan0",
                    model_bytes=140,
                    context_bytes=60,
                    compute_bytes=20,
                    total_bytes=220,
                    device_total_bytes=1000,
                    device_free_bytes=900,
                    source="fixture",
                ),
            ),
            device_allocations=(
                DeploymentDeviceAllocation(
                    device_id="CUDA0",
                    projected_bytes=600,
                    reserved_margin_bytes=100,
                    device_total_bytes=1000,
                    projected_free_bytes=300,
                ),
                DeploymentDeviceAllocation(
                    device_id="Vulkan0",
                    projected_bytes=500,
                    reserved_margin_bytes=100,
                    device_total_bytes=1000,
                    projected_free_bytes=400,
                ),
            ),
            feasibility="feasible",
            provenance={"estimator": "fixture"},
        )

        deployment_placements = DeploymentPlacementRepository(connection)
        placement_id = deployment_placements.put(
            placement,
            request={"strategy": "fixture"},
        )
        assert deployment_placements.get(placement_id) == placement
        assert len(deployment_placements.memory(placement_id)) == 4
        assert len(deployment_placements.allocations(placement_id)) == 2

        mismatched = placement.model_copy(
            update={
                "instance_placements": (
                    DeploymentInstancePlacement(
                        instance_id="qwen",
                        resolved_placement_id=qwen_placement_id,
                    ),
                    DeploymentInstancePlacement(
                        instance_id="flash",
                        resolved_placement_id=qwen_placement_id,
                    ),
                )
            }
        )
        with pytest.raises(ValueError, match="Candidate does not match"):
            deployment_placements.put(mismatched)

        runs = DeploymentRunRepository(connection)
        run_id = runs.create(
            deployment_candidate_id=deployment_id,
            deployment_placement_id=placement_id,
        )
        runs.add_member(
            run_id,
            instance_id="qwen",
            client_run_id="client-qwen",
            result={"ready": True},
        )
        runs.add_member(
            run_id,
            instance_id="flash",
            client_run_id="client-flash",
            result={"ready": True},
        )
        with pytest.raises(ValueError, match="deployment instance"):
            runs.add_member(run_id, instance_id="unknown")

        runs.set_status(
            run_id,
            "running",
            started_at="2026-10-05T12:00:00Z",
        )
        runs.finish(
            run_id,
            status="failed",
            duration_ns=123,
            failure_kind="member_crash",
            failure_details={"instance_id": "flash"},
        )

        record = runs.get(run_id)
        assert record is not None
        assert record.status == "failed"
        assert record.failure_kind == "member_crash"
        assert record.failure_details == {"instance_id": "flash"}
        assert [item.instance_id for item in runs.members(run_id)] == [
            "flash",
            "qwen",
        ]

        with pytest.raises(ValueError, match="already terminal"):
            runs.finish(
                run_id,
                status="completed",
                duration_ns=456,
            )

        assert DeploymentCandidateRepository(connection).get(deployment_id) == deployment
