"""Analysis metric registry tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pytest

from llama_profile_lab.analysis import DEFAULT_METRIC_REGISTRY


@dataclass(frozen=True)
class FakeRun:
    samples: tuple[float, ...]
    metrics: Mapping[str, int | float]
    duration_ns: int | None
    token_count: int


def test_throughput_statistics_pool_individual_repetitions() -> None:
    runs = (
        FakeRun((10.0, 20.0), {}, 1_000_000_000, 1000),
        FakeRun((30.0, 40.0), {}, 1_000_000_000, 1000),
    )

    assert DEFAULT_METRIC_REGISTRY.evaluate("throughput.mean", runs) == 25.0
    assert DEFAULT_METRIC_REGISTRY.evaluate("throughput.median", runs) == 25.0
    assert DEFAULT_METRIC_REGISTRY.evaluate("throughput.min", runs) == 10.0
    assert DEFAULT_METRIC_REGISTRY.evaluate("throughput.max", runs) == 40.0
    assert DEFAULT_METRIC_REGISTRY.evaluate("throughput.sample_count", runs) == 4.0
    assert DEFAULT_METRIC_REGISTRY.evaluate(
        "throughput.stddev", runs
    ) == pytest.approx(12.9099444874)
    assert DEFAULT_METRIC_REGISTRY.evaluate(
        "throughput.cv", runs
    ) == pytest.approx(12.9099444874 / 25.0)


def test_resource_and_derived_efficiency_metrics() -> None:
    runs = (
        FakeRun(
            (100.0,),
            {
                "telemetry.process_cpu_avg_pct_normalized": 25.0,
                "telemetry.process_user_time_ns": 1_500_000_000,
                "telemetry.process_system_time_ns": 500_000_000,
                "telemetry.gpu_power_avg_w": 200.0,
            },
            2_000_000_000,
            2000,
        ),
    )

    assert DEFAULT_METRIC_REGISTRY.evaluate("cpu.process.avg_pct", runs) == 25.0
    assert DEFAULT_METRIC_REGISTRY.evaluate(
        "cpu.seconds_per_1k_tokens", runs
    ) == pytest.approx(1.0)
    assert DEFAULT_METRIC_REGISTRY.evaluate("gpu.energy_j", runs) == 400.0
