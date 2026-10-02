"""SPEED-Bench argv adapter and current upstream JSON normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llama_profile_lab.domain import canonical_json
from llama_profile_lab.domain.workload import SpeedBenchWorkloadCase
from llama_profile_lab.llama.capabilities import CapabilitySet


class SpeedBenchConfigurationError(ValueError):
    """Raised when a SPEED-Bench workload cannot be represented."""


class SpeedBenchParseError(ValueError):
    """Raised when SPEED-Bench output JSON is malformed."""


@dataclass(frozen=True, slots=True)
class SpeedBenchSummary:
    """Normalized overall result for one SPEED-Bench invocation."""

    requests: int
    turns: int
    failed: int
    avg_prompt_ts: float | None
    avg_pred_ts: float | None
    avg_latency_ms: float | None
    draft_n: int
    accepted_n: int
    accept_rate: float | None


@dataclass(frozen=True, slots=True)
class SpeedBenchResult:
    """Raw current-upstream SPEED-Bench output plus normalized overall row."""

    raw: dict[str, Any]
    overall: SpeedBenchSummary
    selected_samples: int | None
    completed_samples: int | None
    failed_samples: int | None


class SpeedBenchAdapter:
    """Build and parse one invocation of tools/server/bench/speed-bench."""

    def build_argv(
        self,
        *,
        binary_path: Path,
        capabilities: CapabilitySet,
        server_url: str,
        workload: SpeedBenchWorkloadCase,
        category: str,
        output_path: Path,
        model_name: str | None = None,
        request_timeout_seconds: float = 600.0,
    ) -> tuple[str, ...]:
        if capabilities.kind != "speed-bench":
            raise SpeedBenchConfigurationError(
                f"selected binary is {capabilities.kind}, not speed-bench"
            )
        if category not in workload.speed_bench.categories:
            raise SpeedBenchConfigurationError(
                f"category {category!r} is not part of the workload"
            )
        if request_timeout_seconds <= 0:
            raise SpeedBenchConfigurationError("request timeout must be positive")

        config = workload.speed_bench
        extra_inputs = {
            parameter.name: parameter.value
            for parameter in config.request
        }
        argv: list[str] = [str(binary_path)]
        _append(argv, capabilities, "--url", server_url)
        if model_name is not None:
            _append(argv, capabilities, "--model", model_name)
        _append(argv, capabilities, "--bench", config.bench)
        _append(argv, capabilities, "--category", category)
        _append(argv, capabilities, "--osl", str(config.output_tokens))
        _append(argv, capabilities, "--extra-inputs", canonical_json(extra_inputs))
        _append(argv, capabilities, "--concurrency", str(config.concurrency))
        if config.limit is not None:
            _append(argv, capabilities, "--limit", str(config.limit))
        _append(
            argv,
            capabilities,
            "--timeout",
            _format_number(request_timeout_seconds),
        )
        _append(argv, capabilities, "--output", str(output_path))
        return tuple(argv)

    def parse_file(self, path: Path) -> SpeedBenchResult:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SpeedBenchParseError(f"cannot read SPEED-Bench output: {exc}") from exc
        return parse_speed_bench_json(text)


def parse_speed_bench_json(text: str) -> SpeedBenchResult:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpeedBenchParseError(f"invalid SPEED-Bench JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SpeedBenchParseError("SPEED-Bench output must be a JSON object")

    summary = payload.get("summary")
    if not isinstance(summary, list):
        raise SpeedBenchParseError("SPEED-Bench output is missing summary rows")
    overall_rows = [
        row
        for row in summary
        if isinstance(row, dict) and row.get("category") == "overall"
    ]
    if len(overall_rows) != 1:
        raise SpeedBenchParseError("SPEED-Bench output must contain one overall summary")
    row = overall_rows[0]

    normalized = SpeedBenchSummary(
        requests=_required_int(row, "requests"),
        turns=_required_int(row, "turns"),
        failed=_required_int(row, "failed"),
        avg_prompt_ts=_optional_float(row, "avg_prompt_t_s"),
        avg_pred_ts=_optional_float(row, "avg_pred_t_s"),
        avg_latency_ms=(
            None
            if _optional_float(row, "avg_latency") is None
            else _optional_float(row, "avg_latency") * 1000.0
        ),
        draft_n=_required_int(row, "draft_n"),
        accepted_n=_required_int(row, "accepted"),
        accept_rate=_optional_float(row, "accept_rate"),
    )
    if normalized.accept_rate is not None and not 0 <= normalized.accept_rate <= 1:
        raise SpeedBenchParseError("accept_rate must be between 0 and 1")
    if normalized.accepted_n > normalized.draft_n and normalized.draft_n >= 0:
        raise SpeedBenchParseError("accepted draft count exceeds drafted token count")

    return SpeedBenchResult(
        raw=payload,
        overall=normalized,
        selected_samples=_optional_int(payload, "selected_samples"),
        completed_samples=_optional_int(payload, "completed_samples"),
        failed_samples=_optional_int(payload, "failed_samples"),
    )


def _append(
    argv: list[str],
    capabilities: CapabilitySet,
    option: str,
    value: str,
) -> None:
    if not capabilities.supports(option):
        raise SpeedBenchConfigurationError(
            f"SPEED-Bench executable does not advertise required option {option}"
        )
    argv.extend((option, value))


def _required_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SpeedBenchParseError(f"{key} must be a non-negative integer")
    return value


def _optional_int(mapping: dict[str, Any], key: str) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SpeedBenchParseError(f"{key} must be a non-negative integer")
    return value


def _optional_float(mapping: dict[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SpeedBenchParseError(f"{key} must be numeric")
    return float(value)


def _format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)
