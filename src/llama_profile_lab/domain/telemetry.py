"""Runtime telemetry observations and run-quality DTOs."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, NonNegativeFloat, NonNegativeInt

from llama_profile_lab.domain.base import FrozenModel

RunQuality = Literal[
    "clean",
    "noisy",
    "external_cpu_load",
    "external_gpu_load",
    "thermal_throttle",
    "telemetry_incomplete",
]
TelemetryPhase = Literal["before", "during", "after"]


class GpuTelemetrySample(FrozenModel):
    """One GPU observation with explicit physical identity and provider provenance."""

    device: str
    name: str | None = None
    uuid: str | None = None
    pci_bus_id: str | None = None
    stable_device_key: str | None = None
    sources: tuple[str, ...] = ()
    utilization_pct: Annotated[float, Field(ge=0, le=100)] | None = None
    vram_used_bytes: NonNegativeInt | None = None
    vram_total_bytes: NonNegativeInt | None = None
    temperature_c: float | None = None
    power_w: NonNegativeFloat | None = None
    graphics_clock_hz: NonNegativeInt | None = None
    memory_clock_hz: NonNegativeInt | None = None
    thermal_throttled: bool | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class TelemetrySample(FrozenModel):
    """One timestamped CPU, process, memory, and GPU observation."""

    timestamp_ns: NonNegativeInt
    phase: TelemetryPhase
    cpu_system_pct: Annotated[float, Field(ge=0, le=100)] | None = None
    cpu_user_pct: Annotated[float, Field(ge=0, le=100)] | None = None
    cpu_system_mode_pct: Annotated[float, Field(ge=0, le=100)] | None = None
    cpu_iowait_pct: Annotated[float, Field(ge=0, le=100)] | None = None
    process_cpu_pct_normalized: Annotated[float, Field(ge=0)] | None = None
    process_cpu_pct_raw: Annotated[float, Field(ge=0)] | None = None
    process_user_time_ns: NonNegativeInt | None = None
    process_system_time_ns: NonNegativeInt | None = None
    process_threads: NonNegativeInt | None = None
    cpu_freq_avg_hz: NonNegativeInt | None = None
    cpu_freq_min_hz: NonNegativeInt | None = None
    cpu_freq_max_hz: NonNegativeInt | None = None
    cpu_temperature_c: float | None = None
    load_avg_1m: NonNegativeFloat | None = None
    load_avg_5m: NonNegativeFloat | None = None
    ram_used_bytes: NonNegativeInt | None = None
    ram_available_bytes: NonNegativeInt | None = None
    swap_used_bytes: NonNegativeInt | None = None
    process_rss_bytes: NonNegativeInt | None = None
    gpus: tuple[GpuTelemetrySample, ...] = ()
    cpu_per_core_pct: tuple[Annotated[float, Field(ge=0, le=100)], ...] = ()
    extra: dict[str, Any] = Field(default_factory=dict)


class GpuTelemetrySummary(FrozenModel):
    """Normalized per-device GPU summary across one telemetry window."""

    stable_device_key: str
    metric_device_id: str
    device: str
    name: str | None = None
    sources: tuple[str, ...] = ()
    sample_count: NonNegativeInt
    utilization_avg_pct: NonNegativeFloat | None = None
    utilization_peak_pct: NonNegativeFloat | None = None
    vram_used_peak_bytes: NonNegativeInt | None = None
    temperature_peak_c: float | None = None
    power_avg_w: NonNegativeFloat | None = None
    power_peak_w: NonNegativeFloat | None = None


class TelemetrySummary(FrozenModel):
    """Normalized scalar summary derived from raw telemetry samples."""

    sample_count: NonNegativeInt
    during_sample_count: NonNegativeInt
    cpu_system_avg_pct: NonNegativeFloat | None = None
    cpu_system_peak_pct: NonNegativeFloat | None = None
    process_cpu_avg_pct_normalized: NonNegativeFloat | None = None
    process_cpu_peak_pct_normalized: NonNegativeFloat | None = None
    process_cpu_avg_pct_raw: NonNegativeFloat | None = None
    process_cpu_peak_pct_raw: NonNegativeFloat | None = None
    cpu_freq_avg_hz: NonNegativeInt | None = None
    cpu_freq_min_hz: NonNegativeInt | None = None
    cpu_freq_max_hz: NonNegativeInt | None = None
    cpu_temperature_peak_c: float | None = None
    process_user_time_ns: NonNegativeInt | None = None
    process_system_time_ns: NonNegativeInt | None = None
    ram_used_peak_bytes: NonNegativeInt | None = None
    ram_available_min_bytes: NonNegativeInt | None = None
    swap_used_peak_bytes: NonNegativeInt | None = None
    process_rss_peak_bytes: NonNegativeInt | None = None
    gpu_utilization_avg_pct: NonNegativeFloat | None = None
    gpu_utilization_peak_pct: NonNegativeFloat | None = None
    gpu_temperature_peak_c: float | None = None
    gpu_vram_used_peak_bytes: NonNegativeInt | None = None
    gpu_power_avg_w: NonNegativeFloat | None = None
    gpu_power_peak_w: NonNegativeFloat | None = None
    gpu_devices: tuple[GpuTelemetrySummary, ...] = ()
    external_cpu_peak_pct: NonNegativeFloat | None = None
    baseline_gpu_utilization_peak_pct: NonNegativeFloat | None = None


class RunQualityAssessment(FrozenModel):
    """Primary quality label plus all observed quality signals."""

    quality: RunQuality
    reasons: tuple[str, ...] = ()
    telemetry_complete: bool
    summary: TelemetrySummary
