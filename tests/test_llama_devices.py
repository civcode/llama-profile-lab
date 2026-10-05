"""llama.cpp logical-device inventory adapter tests."""

import pytest

from llama_profile_lab.domain import AcceleratorDevice
from llama_profile_lab.llama import (
    CapabilitySet,
    LlamaDeviceListAdapter,
    LlamaDeviceListError,
    correlate_physical_devices,
    parse_device_list,
)


def capabilities(*options: str) -> CapabilitySet:
    return CapabilitySet(
        kind="llama-server",
        options=frozenset(options),
        help_stdout="",
        help_stderr="",
        help_exit_code=0,
    )


def test_parse_device_list_handles_heterogeneous_devices() -> None:
    devices = parse_device_list(
        """
Available devices:
  CUDA0: NVIDIA GeForce RTX 4070 Ti SUPER (16376 MiB, 15120 MiB free)
  Vulkan0: AMD Radeon RX 9700 (16368 MiB, 16000 MiB free)
"""
    )

    assert [item.logical_device_name for item in devices] == ["CUDA0", "Vulkan0"]
    assert [item.backend for item in devices] == ["CUDA", "VULKAN"]
    assert devices[0].total_memory_bytes == 16376 * 1024 * 1024
    assert devices[1].free_memory_bytes == 16000 * 1024 * 1024
    assert all(item.mapping_status == "unresolved" for item in devices)



@pytest.mark.parametrize(
    ("text", "names", "backends"),
    (
        (
            """
Available devices:
  CUDA0: NVIDIA Test GPU (16384 MiB, 15000 MiB free)
""",
            ["CUDA0"],
            ["CUDA"],
        ),
        (
            """
Available devices:
  CUDA0: NVIDIA Test GPU A (16384 MiB, 15000 MiB free)
  CUDA1: NVIDIA Test GPU B (16384 MiB, 14900 MiB free)
""",
            ["CUDA0", "CUDA1"],
            ["CUDA", "CUDA"],
        ),
    ),
)
def test_parse_device_list_handles_single_and_homogeneous_devices(
    text: str,
    names: list[str],
    backends: list[str],
) -> None:
    devices = parse_device_list(text)

    assert [item.logical_device_name for item in devices] == names
    assert [item.backend for item in devices] == backends

def test_device_list_adapter_requires_advertised_option(tmp_path) -> None:
    adapter = LlamaDeviceListAdapter()

    with pytest.raises(LlamaDeviceListError, match="--list-devices"):
        adapter.build_argv(
            binary_path=tmp_path / "llama-server",
            capabilities=capabilities("--model"),
        )


def test_duplicate_logical_device_is_rejected() -> None:
    with pytest.raises(LlamaDeviceListError, match="duplicate logical device"):
        parse_device_list(
            """
Available devices:
  CUDA0: First GPU (100 MiB, 90 MiB free)
  CUDA0: Second GPU (100 MiB, 80 MiB free)
"""
        )


def test_physical_correlation_requires_strong_identity() -> None:
    unresolved = AcceleratorDevice(
        logical_device_name="CUDA0",
        backend="CUDA",
        product_name="Example GPU",
    )
    assert correlate_physical_devices(
        (unresolved,),
        ({"pci_address": "0000:01:00.0", "driver": "nvidia"},),
    )[0].mapping_status == "unresolved"

    explicit = unresolved.model_copy(
        update={"pci_bus_id": "0000:01:00.0"}
    )
    mapped = correlate_physical_devices(
        (explicit,),
        ({"pci_address": "0000:01:00.0", "driver": "nvidia"},),
    )[0]
    assert mapped.mapping_status == "mapped"
    assert mapped.physical_device_key == "pci:0000:01:00.0"
    assert mapped.driver == "nvidia"
