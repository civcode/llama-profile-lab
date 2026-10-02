"""llama-bench argv adapter and JSON result parser."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llama_profile_lab.domain import Candidate, MeasurementPolicy
from llama_profile_lab.domain.candidate import ExtraArgument
from llama_profile_lab.domain.workload import (
    CombinedWorkloadCase,
    DecodeWorkloadCase,
    PrefillWorkloadCase,
    SpeedBenchWorkloadCase,
    WorkloadCase,
)
from llama_profile_lab.llama.capabilities import CapabilitySet


class LlamaBenchConfigurationError(ValueError):
    """Raised when a Candidate cannot be represented by the selected binary."""


class LlamaBenchParseError(ValueError):
    """Raised when llama-bench JSON cannot be normalized safely."""


@dataclass(frozen=True, slots=True)
class LlamaBenchSample:
    """One timed llama-bench repetition."""

    elapsed_ns: int
    tokens_per_second: float


@dataclass(frozen=True, slots=True)
class LlamaBenchResult:
    """One normalized single-case llama-bench result."""

    raw: list[dict[str, Any]]
    record: dict[str, Any]
    samples: tuple[LlamaBenchSample, ...]
    avg_ns: int | None
    stddev_ns: int | None
    avg_ts: float | None
    stddev_ts: float | None


class LlamaBenchAdapter:
    """Translate typed Candidate/WorkloadCase values into one llama-bench argv."""

    def build_argv(
        self,
        *,
        binary_path: Path,
        capabilities: CapabilitySet,
        model_path: Path,
        candidate: Candidate,
        workload: WorkloadCase,
        measurement_policy: MeasurementPolicy,
    ) -> tuple[str, ...]:
        """Build a single-workload argv, preferring long-form options."""
        if capabilities.kind != "llama-bench":
            raise LlamaBenchConfigurationError(
                f"selected binary is {capabilities.kind}, not llama-bench"
            )
        if isinstance(workload, SpeedBenchWorkloadCase):
            raise LlamaBenchConfigurationError(
                "speed-bench workloads cannot run through llama-bench"
            )
        if measurement_policy.adaptive is not None:
            raise LlamaBenchConfigurationError(
                "adaptive repetition is not supported by the M5 executor"
            )
        if measurement_policy.repetitions is None:
            raise LlamaBenchConfigurationError("measurement repetitions are missing")

        prompt_tokens, generate_tokens, depth_tokens = _workload_tokens(workload)
        argv: list[str] = [str(binary_path)]

        self._append(argv, capabilities, "--model", ("-m",), str(model_path))
        self._append(
            argv,
            capabilities,
            "--n-prompt",
            ("-p",),
            str(prompt_tokens),
        )
        self._append(argv, capabilities, "--n-gen", ("-n",), str(generate_tokens))
        self._append(argv, capabilities, "--n-depth", ("-d",), str(depth_tokens))
        self._append(
            argv,
            capabilities,
            "--batch-size",
            ("-b",),
            str(candidate.compute.batch_size),
        )
        self._append(
            argv,
            capabilities,
            "--ubatch-size",
            ("-ub",),
            str(candidate.compute.ubatch_size),
        )
        self._append(
            argv,
            capabilities,
            "--cache-type-k",
            ("-ctk",),
            candidate.context.cache_type_k,
        )
        self._append(
            argv,
            capabilities,
            "--cache-type-v",
            ("-ctv",),
            candidate.context.cache_type_v,
        )

        if candidate.compute.threads is not None:
            self._append(
                argv,
                capabilities,
                "--threads",
                ("-t",),
                str(candidate.compute.threads),
            )
        if candidate.compute.flash_attn != "auto":
            self._append(
                argv,
                capabilities,
                "--flash-attn",
                ("-fa",),
                candidate.compute.flash_attn,
            )
        if candidate.compute.load_mode != "auto":
            self._append(
                argv,
                capabilities,
                "--load-mode",
                (),
                candidate.compute.load_mode,
            )
        if candidate.compute.lazy_mode != "auto":
            self._append(
                argv,
                capabilities,
                "--lazy-mode",
                (),
                candidate.compute.lazy_mode,
            )

        constraints = candidate.placement.constraints
        if constraints.n_gpu_layers is not None:
            self._append(
                argv,
                capabilities,
                "--n-gpu-layers",
                ("-ngl",),
                str(constraints.n_gpu_layers),
            )
        if constraints.n_cpu_moe:
            self._append(
                argv,
                capabilities,
                "--n-cpu-moe",
                ("-ncmoe",),
                str(constraints.n_cpu_moe),
            )
        if constraints.split_mode != "layer":
            self._append(
                argv,
                capabilities,
                "--split-mode",
                ("-sm",),
                constraints.split_mode,
            )
        if constraints.main_gpu:
            self._append(
                argv,
                capabilities,
                "--main-gpu",
                ("-mg",),
                str(constraints.main_gpu),
            )
        if constraints.tensor_split is not None:
            self._append(
                argv,
                capabilities,
                "--tensor-split",
                ("-ts",),
                ",".join(str(value) for value in constraints.tensor_split),
            )
        for override in constraints.override_tensor:
            self._append(
                argv,
                capabilities,
                "--override-tensor",
                ("-ot",),
                override,
            )

        self._append_optional_boolean(
            argv,
            capabilities,
            "--no-kv-offload",
            not candidate.context.kv_offload,
        )
        self._append_optional_boolean(
            argv,
            capabilities,
            "--no-op-offload",
            candidate.compute.no_op_offload,
        )
        self._append_optional_boolean(
            argv,
            capabilities,
            "--no-host",
            candidate.compute.no_host,
        )
        self._append_optional_boolean(
            argv,
            capabilities,
            "--repack",
            candidate.compute.repack,
        )

        self._append(
            argv,
            capabilities,
            "--repetitions",
            ("-r",),
            str(measurement_policy.repetitions),
        )
        self._append(argv, capabilities, "--output", ("-o",), "json")
        if not measurement_policy.warmup:
            self._append_flag(argv, capabilities, "--no-warmup", ())
        if measurement_policy.delay_seconds:
            self._append(
                argv,
                capabilities,
                "--delay",
                (),
                _format_number(measurement_policy.delay_seconds),
            )

        for extra_argument in candidate.extra_args:
            self._append_extra(argv, capabilities, extra_argument)

        return tuple(argv)

    @staticmethod
    def _append(
        argv: list[str],
        capabilities: CapabilitySet,
        preferred: str,
        fallbacks: tuple[str, ...],
        value: str,
    ) -> None:
        option = _select_option(capabilities, preferred, fallbacks)
        argv.extend((option, value))

    @staticmethod
    def _append_flag(
        argv: list[str],
        capabilities: CapabilitySet,
        preferred: str,
        fallbacks: tuple[str, ...],
    ) -> None:
        option = _select_option(capabilities, preferred, fallbacks)
        argv.append(option)

    @staticmethod
    def _append_optional_boolean(
        argv: list[str],
        capabilities: CapabilitySet,
        option: str,
        value: bool,
    ) -> None:
        if capabilities.supports(option):
            argv.extend((option, "1" if value else "0"))
        elif value:
            raise LlamaBenchConfigurationError(
                f"binary does not support required option {option}"
            )

    @staticmethod
    def _append_extra(
        argv: list[str],
        capabilities: CapabilitySet,
        extra: ExtraArgument,
    ) -> None:
        if not capabilities.supports(extra.name):
            raise LlamaBenchConfigurationError(
                f"binary does not advertise Candidate extra arg {extra.name}"
            )
        argv.append(extra.name)
        value = extra.value
        if value is None:
            return
        if isinstance(value, tuple):
            if any(item is None for item in value):
                raise LlamaBenchConfigurationError(
                    f"Candidate extra arg {extra.name} cannot contain null tuple values"
                )
            argv.append(
                ",".join(_scalar_arg(item) for item in value if item is not None)
            )
            return
        argv.append(_scalar_arg(value))


def parse_llama_bench_json(
    text: str,
    *,
    expected_repetitions: int | None = None,
) -> LlamaBenchResult:
    """Parse JSON output from one single-workload llama-bench invocation."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LlamaBenchParseError(f"invalid llama-bench JSON: {exc}") from exc

    if not isinstance(payload, list) or len(payload) != 1:
        raise LlamaBenchParseError(
            "expected exactly one llama-bench result object"
        )
    record = payload[0]
    if not isinstance(record, dict):
        raise LlamaBenchParseError("llama-bench result is not an object")

    samples_ns = record.get("samples_ns")
    samples_ts = record.get("samples_ts")
    if not isinstance(samples_ns, list) or not isinstance(samples_ts, list):
        raise LlamaBenchParseError("llama-bench result is missing sample arrays")
    if len(samples_ns) != len(samples_ts):
        raise LlamaBenchParseError("llama-bench sample arrays have different lengths")
    if not samples_ns:
        raise LlamaBenchParseError("llama-bench returned no timed samples")
    if expected_repetitions is not None and len(samples_ns) != expected_repetitions:
        raise LlamaBenchParseError(
            "llama-bench sample count does not match MeasurementPolicy repetitions"
        )

    samples: list[LlamaBenchSample] = []
    for elapsed, throughput in zip(samples_ns, samples_ts, strict=True):
        if isinstance(elapsed, bool) or not isinstance(elapsed, int) or elapsed < 0:
            raise LlamaBenchParseError("invalid elapsed sample in samples_ns")
        if (
            isinstance(throughput, bool)
            or not isinstance(throughput, int | float)
            or throughput < 0
        ):
            raise LlamaBenchParseError("invalid throughput sample in samples_ts")
        samples.append(
            LlamaBenchSample(
                elapsed_ns=elapsed,
                tokens_per_second=float(throughput),
            )
        )

    return LlamaBenchResult(
        raw=payload,
        record=record,
        samples=tuple(samples),
        avg_ns=_optional_int(record, "avg_ns"),
        stddev_ns=_optional_int(record, "stddev_ns"),
        avg_ts=_optional_float(record, "avg_ts"),
        stddev_ts=_optional_float(record, "stddev_ts"),
    )


def _workload_tokens(
    workload: PrefillWorkloadCase | DecodeWorkloadCase | CombinedWorkloadCase,
) -> tuple[int, int, int]:
    return (
        workload.prompt_tokens,
        workload.generate_tokens,
        workload.depth_tokens,
    )


def _select_option(
    capabilities: CapabilitySet,
    preferred: str,
    fallbacks: tuple[str, ...],
) -> str:
    if capabilities.supports(preferred):
        return preferred
    for fallback in fallbacks:
        if capabilities.supports(fallback):
            return fallback
    raise LlamaBenchConfigurationError(
        f"binary does not support required option {preferred}"
    )


def _scalar_arg(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def _optional_int(record: dict[str, Any], key: str) -> int | None:
    value = record.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise LlamaBenchParseError(f"{key} is not an integer")
    return int(value)


def _optional_float(record: dict[str, Any], key: str) -> float | None:
    value = record.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise LlamaBenchParseError(f"{key} is not numeric")
    return float(value)
