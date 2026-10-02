"""SPEED-Bench adapter and parser tests."""

import json
from pathlib import Path

import pytest

from llama_profile_lab.domain import SpeedBenchConfig, SpeedBenchWorkloadCase
from llama_profile_lab.llama import (
    CapabilitySet,
    SpeedBenchAdapter,
    SpeedBenchParseError,
    parse_speed_bench_json,
)


def capabilities() -> CapabilitySet:
    return CapabilitySet(
        kind="speed-bench",
        options=frozenset(
            {
                "--url",
                "--model",
                "--bench",
                "--category",
                "--osl",
                "--extra-inputs",
                "--concurrency",
                "--limit",
                "--timeout",
                "--output",
            }
        ),
        help_stdout="",
        help_stderr="",
        help_exit_code=0,
    )


def workload() -> SpeedBenchWorkloadCase:
    return SpeedBenchWorkloadCase(
        speed_bench=SpeedBenchConfig(
            bench="throughput_1k",
            categories=("all",),
            output_tokens=256,
            concurrency=1,
            limit=2,
            request={"temperature": 0},
        )
    )


def test_speed_bench_argv_uses_current_long_form_surface(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    argv = SpeedBenchAdapter().build_argv(
        binary_path=Path("/opt/speed_bench.py"),
        capabilities=capabilities(),
        server_url="http://127.0.0.1:8080",
        workload=workload(),
        category="all",
        output_path=output,
        model_name="test",
        request_timeout_seconds=30,
    )

    assert argv[argv.index("--bench") + 1] == "throughput_1k"
    assert argv[argv.index("--osl") + 1] == "256"
    assert argv[argv.index("--concurrency") + 1] == "1"
    assert argv[argv.index("--extra-inputs") + 1] == '{"temperature":0}'
    assert argv[argv.index("--output") + 1] == str(output)


def test_speed_bench_parser_normalizes_overall_summary() -> None:
    payload = {
        "selected_samples": 2,
        "completed_samples": 2,
        "failed_samples": 0,
        "summary": [
            {
                "category": "overall",
                "requests": 2,
                "turns": 2,
                "failed": 0,
                "avg_prompt_t_s": 123.5,
                "avg_pred_t_s": 67.25,
                "avg_latency": 1.5,
                "draft_n": 100,
                "accepted": 60,
                "accept_rate": 0.6,
            }
        ],
        "results": [],
    }

    result = parse_speed_bench_json(json.dumps(payload))

    assert result.overall.avg_prompt_ts == 123.5
    assert result.overall.avg_pred_ts == 67.25
    assert result.overall.avg_latency_ms == 1500.0
    assert result.overall.accepted_n == 60
    assert result.overall.accept_rate == 0.6


def test_speed_bench_parser_requires_overall_summary() -> None:
    with pytest.raises(SpeedBenchParseError, match="overall"):
        parse_speed_bench_json('{"summary": []}')
