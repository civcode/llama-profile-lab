"""Telemetry summarization and run-quality tests."""

from llama_profile_lab.domain.telemetry import GpuTelemetrySample, TelemetrySample
from llama_profile_lab.execution.telemetry import (
    RunQualityPolicy,
    classify_run_quality,
    summarize_telemetry,
)


def sample(
    timestamp: int,
    *,
    phase: str = "during",
    system_cpu: float = 60.0,
    process_cpu: float | None = 50.0,
    gpu: float | None = None,
    temp: float = 70.0,
) -> TelemetrySample:
    gpus = (
        (
            GpuTelemetrySample(
                device="0000:01:00.0",
                utilization_pct=gpu,
                temperature_c=temp,
                vram_used_bytes=8_000,
            ),
        )
        if gpu is not None
        else ()
    )
    return TelemetrySample.model_validate(
        {
            "timestamp_ns": timestamp,
            "phase": phase,
            "cpu_system_pct": system_cpu,
            "process_cpu_pct_normalized": process_cpu,
            "process_cpu_pct_raw": (
                None if process_cpu is None else process_cpu * 16
            ),
            "process_user_time_ns": timestamp * 10,
            "process_system_time_ns": timestamp * 2,
            "ram_used_bytes": 1_000,
            "ram_available_bytes": 9_000,
            "process_rss_bytes": 500,
            "gpus": gpus,
        }
    )


def test_summary_separates_process_and_system_cpu() -> None:
    summary = summarize_telemetry(
        (
            sample(1, system_cpu=55, process_cpu=45),
            sample(2, system_cpu=70, process_cpu=50),
        )
    )

    assert summary.cpu_system_avg_pct == 62.5
    assert summary.process_cpu_avg_pct_normalized == 47.5
    assert summary.external_cpu_peak_pct == 20.0
    assert summary.process_rss_peak_bytes == 500


def test_external_cpu_load_is_classified_without_discarding_telemetry() -> None:
    assessment = classify_run_quality(
        (
            sample(1, system_cpu=80, process_cpu=40),
            sample(2, system_cpu=85, process_cpu=45),
        ),
        expect_gpu=False,
    )

    assert assessment.quality == "external_cpu_load"
    assert assessment.telemetry_complete
    assert assessment.summary.sample_count == 2


def test_moderate_background_cpu_load_is_classified_as_noisy() -> None:
    assessment = classify_run_quality(
        (
            sample(1, system_cpu=60, process_cpu=48),
            sample(2, system_cpu=62, process_cpu=50),
        ),
        expect_gpu=False,
    )

    assert assessment.quality == "noisy"
    assert assessment.telemetry_complete
    assert any("noise threshold" in reason for reason in assessment.reasons)


def test_gpu_baseline_load_is_external_gpu_load() -> None:
    assessment = classify_run_quality(
        (
            sample(1, phase="before", system_cpu=5, process_cpu=None, gpu=35),
            sample(2, system_cpu=50, process_cpu=45, gpu=95),
            sample(3, phase="after", system_cpu=5, process_cpu=None, gpu=30),
        ),
        expect_gpu=True,
    )

    assert assessment.quality == "external_gpu_load"


def test_explicit_thermal_signal_takes_priority() -> None:
    thermal_gpu = GpuTelemetrySample(
        device="0000:01:00.0",
        utilization_pct=90,
        temperature_c=80,
        thermal_throttled=True,
    )
    assessment = classify_run_quality(
        (
            TelemetrySample(
                timestamp_ns=1,
                phase="during",
                cpu_system_pct=50,
                process_cpu_pct_normalized=45,
                gpus=(thermal_gpu,),
            ),
        ),
        expect_gpu=True,
    )

    assert assessment.quality == "thermal_throttle"


def test_missing_process_cpu_marks_telemetry_incomplete() -> None:
    assessment = classify_run_quality(
        (sample(1, process_cpu=None),),
        expect_gpu=False,
        policy=RunQualityPolicy(),
    )

    assert assessment.quality == "telemetry_incomplete"
    assert not assessment.telemetry_complete

def test_high_during_gpu_load_is_not_external_gpu_contamination() -> None:
    assessment = classify_run_quality(
        (
            sample(1, phase="before", system_cpu=5, process_cpu=None, gpu=0),
            sample(2, system_cpu=50, process_cpu=45, gpu=99),
            sample(3, phase="after", system_cpu=5, process_cpu=None, gpu=0),
        ),
        expect_gpu=True,
    )

    assert assessment.quality == "clean"
    assert assessment.telemetry_complete

