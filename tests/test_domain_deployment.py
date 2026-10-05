"""Deployment-domain validation and identity tests."""

import pytest
from pydantic import ValidationError

from llama_profile_lab.domain import (
    BackendPair,
    DeploymentCandidate,
    DeploymentDeviceAllocation,
    DeploymentInstancePlacement,
    DeploymentPlacement,
    DeploymentPlacementRequest,
    DeploymentWorkloadMix,
    HostResourcePolicy,
    ModelInstanceCandidate,
    PlacementDeviceMemory,
)


def instance(
    instance_id: str,
    *,
    devices: tuple[str, ...] = ("CUDA0", "Vulkan0"),
) -> ModelInstanceCandidate:
    return ModelInstanceCandidate(
        instance_id=instance_id,
        candidate_id=f"cand_{instance_id}",
        role=instance_id,
        model_artifact_id=f"model_{instance_id}",
        binary_id="bin_llama",
        requested_placement=DeploymentPlacementRequest(
            devices=devices,
            tensor_split=(3.0, 1.0),
        ),
        server_identity=f"srv-{instance_id}",
    )


def candidate(
    instances: tuple[ModelInstanceCandidate, ...],
) -> DeploymentCandidate:
    return DeploymentCandidate(
        instances=instances,
        resource_policy=HostResourcePolicy(
            device_memory_margin_bytes={
                "Vulkan0": 1024,
                "CUDA0": 2048,
            },
            allowed_devices=("Vulkan0", "CUDA0"),
        ),
        workload_mix=DeploymentWorkloadMix(workload_suite_id="suite_1"),
    )


def test_deployment_requires_two_instances() -> None:
    with pytest.raises(ValidationError):
        candidate((instance("qwen"),))


def test_deployment_rejects_duplicate_instance_ids() -> None:
    with pytest.raises(ValidationError, match="instance IDs"):
        candidate((instance("qwen"), instance("qwen")))


def test_deployment_identity_is_independent_of_instance_order() -> None:
    left = candidate((instance("qwen"), instance("flash")))
    right = candidate((instance("flash"), instance("qwen")))

    assert left.content_hash() == right.content_hash()
    assert [item.instance_id for item in left.instances] == ["flash", "qwen"]


def test_resource_policy_normalizes_unordered_values() -> None:
    item = candidate((instance("qwen"), instance("flash")))

    dumped = item.model_dump(mode="json")

    assert dumped["resource_policy"]["device_memory_margin_bytes"] == {
        "CUDA0": 2048,
        "Vulkan0": 1024,
    }
    assert dumped["resource_policy"]["allowed_devices"] == [
        "CUDA0",
        "Vulkan0",
    ]


def test_placement_device_order_remains_semantic() -> None:
    left = instance("qwen", devices=("CUDA0", "Vulkan0"))
    right = instance("qwen", devices=("Vulkan0", "CUDA0"))

    assert left != right


def test_tensor_split_matches_selected_devices() -> None:
    with pytest.raises(ValidationError, match="one value per selected device"):
        DeploymentPlacementRequest(
            devices=("CUDA0", "Vulkan0"),
            tensor_split=(1.0,),
        )


def test_memory_record_checks_total() -> None:
    with pytest.raises(ValidationError, match="total_bytes"):
        PlacementDeviceMemory(
            instance_id="qwen",
            device_id="CUDA0",
            model_bytes=4,
            context_bytes=3,
            compute_bytes=2,
            total_bytes=8,
            device_total_bytes=100,
            device_free_bytes=50,
            source="fixture",
        )


def test_homogeneous_backend_pair_is_valid() -> None:
    policy = HostResourcePolicy(
        allowed_backend_pairs=(BackendPair(left="CUDA", right="CUDA"),)
    )

    assert policy.allowed_backend_pairs == (
        BackendPair(left="CUDA", right="CUDA"),
    )


def test_performance_relevant_change_changes_deployment_hash() -> None:
    original = candidate((instance("qwen"), instance("flash")))
    changed_instance = instance("qwen").model_copy(
        update={
            "requested_placement": DeploymentPlacementRequest(
                devices=("CUDA0", "Vulkan0"),
                tensor_split=(1.0, 1.0),
            )
        }
    )
    changed = candidate((changed_instance, instance("flash")))

    assert original.content_hash() != changed.content_hash()


def test_deployment_round_trips_through_canonical_json() -> None:
    original = candidate((instance("qwen"), instance("flash")))

    restored = DeploymentCandidate.model_validate_json(
        original.canonical_identity_json()
    )

    assert restored == original
    assert restored.content_hash() == original.content_hash()


def test_resource_policy_rejects_duplicate_devices() -> None:
    with pytest.raises(ValidationError, match="allowed_devices"):
        HostResourcePolicy(allowed_devices=("CUDA0", "CUDA0"))


def test_placement_identity_ignores_memory_observations_and_provenance() -> None:
    base_memory = PlacementDeviceMemory(
        instance_id="qwen",
        device_id="CUDA0",
        model_bytes=4,
        context_bytes=3,
        compute_bytes=2,
        total_bytes=9,
        device_total_bytes=100,
        device_free_bytes=50,
        source="fixture",
        measured_at="2026-10-05T12:00:00Z",
    )
    left = DeploymentPlacement(
        deployment_candidate_id="deploy_1",
        host_id="host_1",
        instance_placements=(
            DeploymentInstancePlacement(
                instance_id="qwen",
                resolved_placement_id="place_1",
            ),
        ),
        device_memory=(base_memory,),
        device_allocations=(
            DeploymentDeviceAllocation(
                device_id="CUDA0",
                projected_bytes=9,
                reserved_margin_bytes=10,
                device_total_bytes=100,
                projected_free_bytes=81,
            ),
        ),
        feasibility="feasible",
        provenance={"estimator": "first"},
    )
    right = left.model_copy(
        update={
            "device_memory": (
                base_memory.model_copy(
                    update={
                        "device_free_bytes": 40,
                        "measured_at": "2026-10-05T12:01:00Z",
                    }
                ),
            ),
            "device_allocations": (
                DeploymentDeviceAllocation(
                    device_id="CUDA0",
                    projected_bytes=10,
                    reserved_margin_bytes=10,
                    device_total_bytes=100,
                    projected_free_bytes=80,
                ),
            ),
            "feasibility": "pending",
            "provenance": {"estimator": "second"},
        }
    )

    assert left != right
    assert left.content_hash() == right.content_hash()

    changed_placement = left.model_copy(
        update={
            "instance_placements": (
                DeploymentInstancePlacement(
                    instance_id="qwen",
                    resolved_placement_id="place_2",
                ),
            )
        }
    )
    assert changed_placement.content_hash() != left.content_hash()

