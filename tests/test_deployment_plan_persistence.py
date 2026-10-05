"""Deployment planner summary/case persistence tests."""

from pathlib import Path

from llama_profile_lab.db import (
    Database,
    DeploymentPlanRepository,
    DeploymentPlacementRepository,
    EnvironmentRepository,
    PlacementRepository,
)
from llama_profile_lab.domain import (
    DeploymentInstancePlacement,
    DeploymentPlacement,
    DeploymentSearchDimension,
    DeploymentSearchSpace,
    ResolvedPlacement,
)
from tests.test_deployment_persistence import seed_deployment


def test_deployment_plan_persists_deterministic_cases(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "deployment-plan.db")
    deployment_id, deployment, qwen_id, flash_id, binary_id = (
        seed_deployment(database)
    )

    with database.session() as connection:
        host_id = EnvironmentRepository(connection).put_host(
            hostname="test",
            hardware_fingerprint="planner-host",
            cpu={},
            ram_bytes=1024,
            gpus=[],
            os_info={},
        )
        placements = PlacementRepository(connection)
        qwen_placement_id = placements.put_resolved(
            placement_hash="a" * 64,
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
            placement_hash="b" * 64,
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
        deployment_placement_id = DeploymentPlacementRepository(
            connection
        ).put(
            DeploymentPlacement(
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
                feasibility="feasible",
            )
        )

        search = DeploymentSearchSpace(
            dimensions=(
                DeploymentSearchDimension(
                    path="instances.qwen.context.size",
                    values=(131072,),
                ),
            )
        )
        plans = DeploymentPlanRepository(connection)
        plan_id = plans.create(
            base_deployment_candidate_id=deployment_id,
            host_id=host_id,
            search_space=search,
            request={"fixture": True},
            raw_combinations=1,
            rejected_by_constraints=0,
            duplicate_candidates=0,
            symmetry_reduced=0,
            capability_rejected=0,
            estimate_failed=0,
            memory_rejected=0,
            valid_count=1,
        )
        first_hash = plans.add_case(
            plan_id,
            ordinal=0,
            deployment_candidate_id=deployment_id,
            deployment_placement_id=deployment_placement_id,
            generation={"assignments": {}},
        )
        record = plans.record(plan_id)
        cases = plans.cases(plan_id)

    assert deployment.workload_mix.workload_suite_id
    assert record is not None
    assert record.valid_count == 1
    assert len(cases) == 1
    assert cases[0].case_hash == first_hash
