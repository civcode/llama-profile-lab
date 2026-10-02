"""Metric registry and stable statistical reductions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean, median, stdev
from typing import Literal, Protocol

MetricAggregation = Literal["mean", "max", "min"]
DerivedMetric = Literal[
    "throughput_mean",
    "throughput_median",
    "throughput_stddev",
    "throughput_cv",
    "throughput_min",
    "throughput_max",
    "throughput_sample_count",
    "cpu_seconds_per_1k_tokens",
    "gpu_energy_j",
]


class MetricRun(Protocol):
    """Minimal observation shape consumed by MetricRegistry."""

    samples: tuple[float, ...]
    metrics: Mapping[str, int | float]
    duration_ns: int | None
    token_count: int


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    """Display and reduction metadata for one analysis metric."""

    name: str
    label: str
    unit: str
    source_metric: str | None = None
    aggregation: MetricAggregation = "mean"
    derived: DerivedMetric | None = None


class MetricRegistry:
    """Evaluate pooled benchmark samples and normalized scalar run metrics."""

    def __init__(self, definitions: Sequence[MetricDefinition]) -> None:
        self._definitions = {definition.name: definition for definition in definitions}
        if len(self._definitions) != len(definitions):
            raise ValueError("metric definitions must have unique names")

    def get(self, name: str) -> MetricDefinition:
        """Return a registered metric or a generic scalar-metric fallback."""
        definition = self._definitions.get(name)
        if definition is not None:
            return definition
        return MetricDefinition(
            name=name,
            label=name,
            unit="",
            source_metric=name,
            aggregation="mean",
        )

    def definitions(self) -> tuple[MetricDefinition, ...]:
        """Return registered metrics deterministically."""
        return tuple(self._definitions[name] for name in sorted(self._definitions))

    def evaluate(self, name: str, runs: Sequence[MetricRun]) -> float | None:
        """Evaluate one metric over repeated successful runs."""
        definition = self.get(name)
        if definition.derived is not None:
            return _evaluate_derived(definition.derived, runs)
        if definition.source_metric is None:
            return None
        values = [
            float(run.metrics[definition.source_metric])
            for run in runs
            if definition.source_metric in run.metrics
        ]
        return _aggregate(values, definition.aggregation)


def _evaluate_derived(
    kind: DerivedMetric,
    runs: Sequence[MetricRun],
) -> float | None:
    samples = [sample for run in runs for sample in run.samples]
    if kind == "throughput_mean":
        return fmean(samples) if samples else None
    if kind == "throughput_median":
        return median(samples) if samples else None
    if kind == "throughput_stddev":
        if not samples:
            return None
        return stdev(samples) if len(samples) > 1 else 0.0
    if kind == "throughput_cv":
        if not samples:
            return None
        mean = fmean(samples)
        if mean == 0:
            return None
        deviation = stdev(samples) if len(samples) > 1 else 0.0
        return deviation / mean
    if kind == "throughput_min":
        return min(samples) if samples else None
    if kind == "throughput_max":
        return max(samples) if samples else None
    if kind == "throughput_sample_count":
        return float(len(samples))
    if kind == "cpu_seconds_per_1k_tokens":
        values: list[float] = []
        for run in runs:
            user = run.metrics.get("telemetry.process_user_time_ns")
            system = run.metrics.get("telemetry.process_system_time_ns")
            if user is None or system is None or run.token_count <= 0:
                continue
            cpu_seconds = (float(user) + float(system)) / 1_000_000_000
            values.append(cpu_seconds / (run.token_count / 1000.0))
        return fmean(values) if values else None
    if kind == "gpu_energy_j":
        values = []
        for run in runs:
            power = run.metrics.get("telemetry.gpu_power_avg_w")
            if power is None or run.duration_ns is None:
                continue
            values.append(float(power) * (run.duration_ns / 1_000_000_000))
        return fmean(values) if values else None
    raise AssertionError(f"unknown derived metric: {kind}")


def _aggregate(values: Sequence[float], aggregation: MetricAggregation) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None
    if aggregation == "mean":
        return fmean(finite)
    if aggregation == "max":
        return max(finite)
    return min(finite)


DEFAULT_METRIC_REGISTRY = MetricRegistry(
    (
        MetricDefinition(
            "throughput.mean",
            "Mean throughput",
            "tokens/s",
            derived="throughput_mean",
        ),
        MetricDefinition(
            "throughput.median",
            "Median throughput",
            "tokens/s",
            derived="throughput_median",
        ),
        MetricDefinition(
            "throughput.stddev",
            "Throughput standard deviation",
            "tokens/s",
            derived="throughput_stddev",
        ),
        MetricDefinition(
            "throughput.cv",
            "Throughput coefficient of variation",
            "ratio",
            derived="throughput_cv",
        ),
        MetricDefinition(
            "throughput.min",
            "Minimum throughput",
            "tokens/s",
            derived="throughput_min",
        ),
        MetricDefinition(
            "throughput.max",
            "Maximum throughput",
            "tokens/s",
            derived="throughput_max",
        ),
        MetricDefinition(
            "throughput.sample_count",
            "Timed sample count",
            "samples",
            derived="throughput_sample_count",
        ),
        MetricDefinition(
            "cpu.process.avg_pct",
            "Average benchmark-process CPU",
            "%",
            "telemetry.process_cpu_avg_pct_normalized",
        ),
        MetricDefinition(
            "cpu.process.peak_pct",
            "Peak benchmark-process CPU",
            "%",
            "telemetry.process_cpu_peak_pct_normalized",
            "max",
        ),
        MetricDefinition(
            "cpu.system.avg_pct",
            "Average system CPU",
            "%",
            "telemetry.cpu_system_avg_pct",
        ),
        MetricDefinition(
            "cpu.system.peak_pct",
            "Peak system CPU",
            "%",
            "telemetry.cpu_system_peak_pct",
            "max",
        ),
        MetricDefinition(
            "cpu.temperature.peak_c",
            "Peak CPU temperature",
            "C",
            "telemetry.cpu_temperature_peak_c",
            "max",
        ),
        MetricDefinition(
            "cpu.seconds_per_1k_tokens",
            "CPU seconds per 1K tokens",
            "cpu-s/1K tokens",
            derived="cpu_seconds_per_1k_tokens",
        ),
        MetricDefinition(
            "memory.ram_used.peak_bytes",
            "Peak RAM used",
            "bytes",
            "telemetry.ram_used_peak_bytes",
            "max",
        ),
        MetricDefinition(
            "memory.process_rss.peak_bytes",
            "Peak benchmark RSS",
            "bytes",
            "telemetry.process_rss_peak_bytes",
            "max",
        ),
        MetricDefinition(
            "gpu.utilization.avg_pct",
            "Average GPU utilization",
            "%",
            "telemetry.gpu_utilization_avg_pct",
        ),
        MetricDefinition(
            "gpu.utilization.peak_pct",
            "Peak GPU utilization",
            "%",
            "telemetry.gpu_utilization_peak_pct",
            "max",
        ),
        MetricDefinition(
            "gpu.vram.peak_bytes",
            "Peak VRAM used",
            "bytes",
            "telemetry.gpu_vram_used_peak_bytes",
            "max",
        ),
        MetricDefinition(
            "gpu.temperature.peak_c",
            "Peak GPU temperature",
            "C",
            "telemetry.gpu_temperature_peak_c",
            "max",
        ),
        MetricDefinition(
            "gpu.power.avg_w",
            "Average GPU power",
            "W",
            "telemetry.gpu_power_avg_w",
        ),
        MetricDefinition(
            "gpu.power.peak_w",
            "Peak GPU power",
            "W",
            "telemetry.gpu_power_peak_w",
            "max",
        ),
        MetricDefinition(
            "gpu.energy_j",
            "Estimated GPU energy",
            "J",
            derived="gpu_energy_j",
        ),
    )
)
