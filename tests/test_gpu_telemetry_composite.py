"""Mixed-vendor GPU telemetry composition and per-device summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llama_profile_lab.domain.telemetry import GpuTelemetrySample, TelemetrySample
from llama_profile_lab.execution.telemetry import (
    CompositeGpuTelemetryProvider,
    SysfsGpuTelemetryProvider,
    normalize_pci_bus_id,
    summarize_telemetry,
    summary_metrics,
)


@dataclass
class StaticProvider:
    samples: tuple[GpuTelemetrySample, ...]

    def sample(self) -> tuple[GpuTelemetrySample, ...]:
        return self.samples


class FailingProvider:
    def sample(self) -> tuple[GpuTelemetrySample, ...]:
        raise RuntimeError("synthetic provider failure")


def test_normalize_pci_bus_id_handles_nvidia_domain_width() -> None:
    assert normalize_pci_bus_id("00000000:01:00.0") == "0000:01:00.0"
    assert normalize_pci_bus_id("0000:0A:00.1") == "0000:0a:00.1"
    assert normalize_pci_bus_id("not-a-pci-id") is None


def test_composite_returns_mixed_vendor_devices_and_deduplicates_nvidia() -> None:
    nvidia = StaticProvider(
        (
            GpuTelemetrySample(
                device="00000000:01:00.0",
                name="NVIDIA Test",
                uuid="GPU-ABC",
                utilization_pct=80,
                vram_used_bytes=800,
                vram_total_bytes=1000,
                temperature_c=70,
            ),
        )
    )
    sysfs = StaticProvider(
        (
            GpuTelemetrySample(
                device="0000:01:00.0",
                utilization_pct=60,
                power_w=150,
            ),
            GpuTelemetrySample(
                device="0000:02:00.0",
                name="AMD Test",
                utilization_pct=55,
                vram_used_bytes=600,
                vram_total_bytes=1000,
                power_w=120,
            ),
        )
    )

    samples = CompositeGpuTelemetryProvider(
        (
            ("nvidia-smi", nvidia),
            ("sysfs", sysfs),
        )
    ).sample()

    assert len(samples) == 2
    by_key = {sample.stable_device_key: sample for sample in samples}

    nvidia_sample = by_key["pci:0000:01:00.0"]
    assert nvidia_sample.utilization_pct == 80
    assert nvidia_sample.power_w == 150
    assert nvidia_sample.sources == ("nvidia-smi", "sysfs")

    amd_sample = by_key["pci:0000:02:00.0"]
    assert amd_sample.utilization_pct == 55
    assert amd_sample.sources == ("sysfs",)


def test_composite_with_synthetic_amd_sysfs_tree_returns_both(
    tmp_path: Path,
) -> None:
    sysroot = tmp_path / "sys"
    device = sysroot / "class/drm/card0/device"
    hwmon = device / "hwmon/hwmon0"
    hwmon.mkdir(parents=True)
    (device / "gpu_busy_percent").write_text("51\n", encoding="utf-8")
    (device / "mem_info_vram_used").write_text("600\n", encoding="utf-8")
    (device / "mem_info_vram_total").write_text("1000\n", encoding="utf-8")
    (hwmon / "temp1_input").write_text("65000\n", encoding="utf-8")
    (hwmon / "power1_average").write_text("110000000\n", encoding="utf-8")

    nvidia = StaticProvider(
        (
            GpuTelemetrySample(
                device="0000:01:00.0",
                name="NVIDIA Test",
                utilization_pct=72,
            ),
        )
    )

    samples = CompositeGpuTelemetryProvider(
        (
            ("nvidia-smi", nvidia),
            ("sysfs", SysfsGpuTelemetryProvider(sys_root=sysroot)),
        )
    ).sample()

    assert len(samples) == 2
    assert {sample.sources[0] for sample in samples} == {
        "nvidia-smi",
        "sysfs",
    }
    assert sorted(
        sample.utilization_pct
        for sample in samples
        if sample.utilization_pct is not None
    ) == [51, 72]


def test_composite_survives_partial_provider_failure() -> None:
    amd = StaticProvider(
        (
            GpuTelemetrySample(
                device="0000:02:00.0",
                utilization_pct=42,
            ),
        )
    )

    samples = CompositeGpuTelemetryProvider(
        (
            ("nvidia-smi", FailingProvider()),
            ("sysfs", amd),
        )
    ).sample()

    assert len(samples) == 1
    assert samples[0].stable_device_key == "pci:0000:02:00.0"


def test_explicit_stable_mapping_can_correlate_devices_without_pci() -> None:
    first = StaticProvider(
        (
            GpuTelemetrySample(
                device="logical0",
                stable_device_key="mapped:gpu-a",
                utilization_pct=75,
            ),
        )
    )
    second = StaticProvider(
        (
            GpuTelemetrySample(
                device="logical-amd-bridge",
                stable_device_key="mapped:gpu-a",
                power_w=95,
            ),
        )
    )

    samples = CompositeGpuTelemetryProvider(
        (("provider-a", first), ("provider-b", second))
    ).sample()

    assert len(samples) == 1
    assert samples[0].utilization_pct == 75
    assert samples[0].power_w == 95
    assert samples[0].sources == ("provider-a", "provider-b")


def test_same_product_name_without_strong_identity_is_not_merged() -> None:
    first = StaticProvider(
        (
            GpuTelemetrySample(
                device="logical0",
                name="Same GPU Name",
                utilization_pct=10,
            ),
        )
    )
    second = StaticProvider(
        (
            GpuTelemetrySample(
                device="logical1",
                name="Same GPU Name",
                utilization_pct=20,
            ),
        )
    )

    samples = CompositeGpuTelemetryProvider(
        (("provider-a", first), ("provider-b", second))
    ).sample()

    assert len(samples) == 2
    assert {sample.device for sample in samples} == {"logical0", "logical1"}


def test_per_device_summary_and_generic_metrics_remain_distinct() -> None:
    samples = (
        TelemetrySample(
            timestamp_ns=1,
            phase="during",
            gpus=(
                GpuTelemetrySample(
                    device="0000:01:00.0",
                    pci_bus_id="0000:01:00.0",
                    stable_device_key="pci:0000:01:00.0",
                    sources=("nvidia-smi",),
                    utilization_pct=70,
                    vram_used_bytes=700,
                    power_w=140,
                ),
                GpuTelemetrySample(
                    device="0000:02:00.0",
                    pci_bus_id="0000:02:00.0",
                    stable_device_key="pci:0000:02:00.0",
                    sources=("sysfs",),
                    utilization_pct=40,
                    vram_used_bytes=500,
                    power_w=100,
                ),
            ),
        ),
        TelemetrySample(
            timestamp_ns=2,
            phase="during",
            gpus=(
                GpuTelemetrySample(
                    device="0000:01:00.0",
                    pci_bus_id="0000:01:00.0",
                    stable_device_key="pci:0000:01:00.0",
                    sources=("nvidia-smi",),
                    utilization_pct=90,
                    vram_used_bytes=800,
                    power_w=160,
                ),
                GpuTelemetrySample(
                    device="0000:02:00.0",
                    pci_bus_id="0000:02:00.0",
                    stable_device_key="pci:0000:02:00.0",
                    sources=("sysfs",),
                    utilization_pct=60,
                    vram_used_bytes=650,
                    power_w=110,
                ),
            ),
        ),
    )

    summary = summarize_telemetry(samples)

    assert summary.gpu_utilization_peak_pct == 90
    assert len(summary.gpu_devices) == 2
    by_key = {item.stable_device_key: item for item in summary.gpu_devices}
    assert by_key["pci:0000:01:00.0"].utilization_avg_pct == 80
    assert by_key["pci:0000:02:00.0"].vram_used_peak_bytes == 650

    metrics = summary_metrics(summary)
    assert metrics[
        "telemetry.gpu.pci_0000_01_00_0.utilization_peak_pct"
    ] == 90
    assert metrics[
        "telemetry.gpu.pci_0000_02_00_0.power_avg_w"
    ] == 105
