"""Accelerator identity and memory-estimator domain tests."""

import pytest
from pydantic import ValidationError

from llama_profile_lab.domain import (
    AcceleratorDevice,
    MemoryEstimateDevice,
    MemoryEstimateIdentity,
    MemoryEstimateOutput,
    MemoryEstimateResolved,
    physical_device_key,
)


def identity(*, devices: tuple[str, ...] = ("CUDA0", "Vulkan0")) -> MemoryEstimateIdentity:
    return MemoryEstimateIdentity(
        candidate_hash="a" * 64,
        model_artifact_id="model:qwen",
        helper_sha256="b" * 64,
        selected_devices=devices,
        n_gpu_layers="all",
        tensor_split=(3.0, 1.0),
    )


def test_physical_device_key_prefers_uuid_over_pci() -> None:
    assert physical_device_key(
        pci_bus_id="0000:01:00.0",
        uuid="GPU-ABC",
    ) == "uuid:gpu-abc"
    assert physical_device_key(pci_bus_id="0000:01:00.0") == "pci:0000:01:00.0"


def test_mapped_accelerator_requires_physical_key() -> None:
    with pytest.raises(ValidationError, match="physical_device_key"):
        AcceleratorDevice(
            logical_device_name="CUDA0",
            backend="CUDA",
            mapping_status="mapped",
        )


def test_accelerator_memory_bounds_are_validated() -> None:
    with pytest.raises(ValidationError, match="free accelerator memory"):
        AcceleratorDevice(
            logical_device_name="CUDA0",
            backend="CUDA",
            total_memory_bytes=100,
            free_memory_bytes=101,
        )


def test_memory_estimate_identity_preserves_device_order() -> None:
    left = identity(devices=("CUDA0", "Vulkan0"))
    right = identity(devices=("Vulkan0", "CUDA0"))

    assert left.content_hash() != right.content_hash()


def test_memory_estimate_identity_rejects_split_length_mismatch() -> None:
    with pytest.raises(ValidationError, match="one value per selected device"):
        MemoryEstimateIdentity(
            candidate_hash="a" * 64,
            model_artifact_id="model:qwen",
            helper_sha256="b" * 64,
            selected_devices=("CUDA0", "Vulkan0"),
            tensor_split=(1.0,),
        )


def test_memory_estimate_device_rejects_impossible_totals() -> None:
    with pytest.raises(ValidationError, match="total_bytes"):
        MemoryEstimateDevice(
            logical_device_name="CUDA0",
            model_bytes=4,
            context_bytes=3,
            compute_bytes=2,
            total_bytes=8,
            device_total_bytes=100,
            device_free_bytes=50,
        )


def test_memory_output_requires_rows_for_resolved_devices() -> None:
    with pytest.raises(ValidationError, match="must match"):
        MemoryEstimateOutput(
            devices=(
                MemoryEstimateDevice(
                    logical_device_name="CUDA0",
                    model_bytes=4,
                    context_bytes=3,
                    compute_bytes=2,
                    total_bytes=9,
                    device_total_bytes=100,
                    device_free_bytes=50,
                ),
            ),
            resolved=MemoryEstimateResolved(
                n_gpu_layers=1,
                devices=("CUDA0", "Vulkan0"),
            ),
        )
