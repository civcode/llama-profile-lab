"""Stable host-identity compatibility tests."""

from llama_profile_lab.execution import host as host_module


def test_enriched_gpu_metadata_does_not_change_legacy_host_fingerprint(
    monkeypatch,
) -> None:
    legacy_gpu = {
        "drm_card": "card0",
        "pci_address": "0000:01:00.0",
        "vendor": "0x10de",
        "device": "0x2782",
        "subsystem_vendor": "0x1043",
        "subsystem_device": "0x88a8",
        "driver": "nvidia",
    }
    monkeypatch.setattr(
        host_module,
        "_linux_gpu_inventory",
        lambda: [legacy_gpu],
    )
    legacy = host_module.detect_basic_host()

    enriched_gpu = {
        **legacy_gpu,
        "pci_bus_id": "0000:01:00.0",
        "stable_device_key": "pci:0000:01:00.0",
        "numa_node": "0",
        "iommu_group": "17",
    }
    monkeypatch.setattr(
        host_module,
        "_linux_gpu_inventory",
        lambda: [enriched_gpu],
    )
    enriched = host_module.detect_basic_host()

    assert enriched.hardware_fingerprint == legacy.hardware_fingerprint
    assert enriched.gpus[0]["stable_device_key"] == "pci:0000:01:00.0"
