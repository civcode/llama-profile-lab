"""Deployment-wide deterministic search expansion tests."""

import pytest

from llama_profile_lab.domain import (
    Candidate,
    ComputeConfig,
    ContextConfig,
    DeploymentCandidate,
    DeploymentPlacementRequest,
    DeploymentSearchDimension,
    DeploymentSearchSpace,
    DeploymentWorkloadMix,
    FitConfig,
    ModelInstanceCandidate,
    ModelSelection,
    PlacementConfig,
)
from llama_profile_lab.planning import (
    DeploymentPlanningError,
    expand_deployment_search,
)


def candidate(model_id: str, *, context: int = 8192) -> Candidate:
    return Candidate(
        model=ModelSelection(target_model_id=model_id),
        context=ContextConfig(
            size=context,
            cache_type_k="f16",
            cache_type_v="f16",
        ),
        compute=ComputeConfig(batch_size=2048, ubatch_size=512),
        placement=PlacementConfig(
            mode="fit",
            fit=FitConfig(target_mib=256, min_context=4096),
        ),
    )


def base() -> tuple[DeploymentCandidate, dict[str, Candidate]]:
    qwen = candidate("model:qwen")
    flash = candidate("model:flash")
    deployment = DeploymentCandidate(
        instances=(
            ModelInstanceCandidate(
                instance_id="qwen",
                candidate_id=f"cand_{qwen.content_hash()}",
                role="primary",
                model_artifact_id="model:qwen",
                binary_id="bin_server",
                requested_placement=DeploymentPlacementRequest(
                    devices=("GPU0",)
                ),
                server_identity="qwen",
            ),
            ModelInstanceCandidate(
                instance_id="flash",
                candidate_id=f"cand_{flash.content_hash()}",
                role="secondary",
                model_artifact_id="model:flash",
                binary_id="bin_server",
                requested_placement=DeploymentPlacementRequest(
                    devices=("GPU1",)
                ),
                server_identity="flash",
            ),
        ),
        workload_mix=DeploymentWorkloadMix(workload_suite_id="suite_1"),
    )
    return deployment, {"qwen": qwen, "flash": flash}


def test_expansion_addresses_instance_candidates_placement_and_margins() -> None:
    deployment, candidates = base()
    search = DeploymentSearchSpace(
        dimensions=(
            DeploymentSearchDimension(
                path="instances.qwen.context.size",
                values=(8192, 16384),
            ),
            DeploymentSearchDimension(
                path="instances.flash.requested_placement.devices",
                values=(("GPU1",), ("GPU0", "GPU1")),
            ),
            DeploymentSearchDimension(
                path="resource_policy.device_memory_margin_bytes.GPU0",
                values=(0, 1024),
            ),
        )
    )

    expansion = expand_deployment_search(deployment, candidates, search)

    assert expansion.raw_combinations == 8
    assert len(expansion.points) == 8
    assert {
        point.candidate_for("qwen").context.size
        for point in expansion.points
    } == {8192, 16384}


def test_constraints_duplicate_elimination_and_symmetry_are_deterministic() -> None:
    deployment, candidates = base()
    search = DeploymentSearchSpace(
        dimensions=(
            DeploymentSearchDimension(
                path="instances.qwen.context.size",
                values=(8192, 16384),
            ),
            DeploymentSearchDimension(
                path="instances.flash.context.size",
                values=(8192, 16384),
            ),
        ),
        constraints=(
            "instances.qwen.context.size >= instances.flash.context.size",
        ),
    )
    expansion = expand_deployment_search(deployment, candidates, search)

    assert expansion.raw_combinations == 4
    assert expansion.rejected_by_constraints == 1
    assert len(expansion.points) == 3

    duplicate_search = DeploymentSearchSpace(
        dimensions=(
            DeploymentSearchDimension(
                path="instances.qwen.requested_placement.split_mode",
                values=(None, "layer"),
            ),
        )
    )
    duplicates = expand_deployment_search(
        deployment,
        candidates,
        duplicate_search,
    )
    assert duplicates.duplicate_candidates == 1
    assert len(duplicates.points) == 1

    symmetric = expand_deployment_search(
        deployment,
        candidates,
        DeploymentSearchSpace(
            dimensions=(
                DeploymentSearchDimension(
                    path="instances.qwen.context.size",
                    values=(8192, 16384),
                ),
            )
        ),
        symmetry_key=lambda item: "equivalent",
    )
    assert symmetric.symmetry_reduced == 1
    assert len(symmetric.points) == 1


def test_model_artifact_identity_can_differ_from_candidate_model_id() -> None:
    deployment, candidates = base()
    deployment = deployment.model_copy(
        update={
            "instances": tuple(
                item.model_copy(
                    update={
                        "model_artifact_id": f"artifact:{item.instance_id}"
                    }
                )
                for item in deployment.instances
            )
        }
    )
    search = DeploymentSearchSpace(
        dimensions=(
            DeploymentSearchDimension(
                path="instances.qwen.context.size",
                values=(8192,),
            ),
        )
    )

    expansion = expand_deployment_search(deployment, candidates, search)

    assert len(expansion.points) == 1
    assert (
        expansion.points[0].deployment.instances[1].model_artifact_id
        == "artifact:qwen"
    )


def test_invalid_instance_path_fails_closed() -> None:
    deployment, candidates = base()
    search = DeploymentSearchSpace(
        dimensions=(
            DeploymentSearchDimension(
                path="instances.missing.context.size",
                values=(8192,),
            ),
        )
    )

    with pytest.raises(DeploymentPlanningError, match="unknown deployment instance"):
        expand_deployment_search(deployment, candidates, search)
