"""Typed persistence records that are not immutable domain definitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from llama_profile_lab.domain.telemetry import RunQuality

ExperimentStatus = Literal[
    "draft",
    "planned",
    "running",
    "paused",
    "completed",
    "cancelled",
    "failed",
]

PlacementAttemptStatus = Literal[
    "running",
    "completed",
    "fit_failed",
    "timeout",
    "parser_failed",
    "interrupted",
    "cancelled",
]

ServerRunStatus = Literal[
    "starting",
    "ready",
    "completed",
    "start_failed",
    "readiness_failed",
    "benchmark_failed",
    "interrupted",
    "cancelled",
]

ServerBenchmarkStatus = Literal[
    "running",
    "completed",
    "benchmark_failed",
    "timeout",
    "parser_failed",
    "interrupted",
    "cancelled",
]

RunStatus = Literal[
    "planned",
    "running",
    "completed",
    "oom",
    "timeout",
    "invalid",
    "fit_failed",
    "load_failed",
    "benchmark_failed",
    "parser_failed",
    "interrupted",
    "cancelled",
]


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """Persisted experiment event plus its immutable definition."""

    id: str
    status: ExperimentStatus
    name: str
    base_candidate_id: str
    search_space_id: str
    workload_suite_id: str
    measurement_policy_id: str
    created_at: str
    frozen_at: str | None
    completed_at: str | None


@dataclass(frozen=True, slots=True)
class BenchmarkCaseRecord:
    """One planned Candidate × WorkloadCase observation."""

    id: str
    experiment_id: str
    candidate_id: str
    workload_case_id: str
    placement_id: str | None
    case_hash: str
    status: RunStatus
    ordinal: int


@dataclass(frozen=True, slots=True)
class BenchmarkRunRecord:
    """One append-only attempt to execute a benchmark case."""

    id: str
    benchmark_case_id: str
    host_id: str
    binary_id: str
    measurement_policy_id: str
    started_at: str
    finished_at: str | None
    duration_ns: int | None
    status: RunStatus
    exit_code: int | None
    quality: RunQuality | None
    quality_details: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class BinaryRecord:
    """Persisted identity and discovered capabilities for one executable."""

    id: str
    sha256: str
    kind: str
    path: str
    size_bytes: int
    mtime_ns: int
    git_commit: str | None
    git_branch: str | None
    git_dirty: bool | None
    build_number: str | None
    build_info: Mapping[str, Any]
    capabilities: Mapping[str, Any]
    created_at: str


@dataclass(frozen=True, slots=True)
class ResolvedPlacementRecord:
    """One immutable successful placement resolution."""

    id: str
    placement_hash: str
    candidate_id: str
    host_id: str
    binary_id: str
    fit_attempt_id: str | None
    production_context_size: int
    n_gpu_layers: int
    n_cpu_moe: int
    split_mode: str
    main_gpu: int
    devices: str | tuple[str, ...]
    tensor_split: tuple[float, ...] | None
    override_tensor: tuple[str, ...]
    request: Mapping[str, Any]
    created_at: str


@dataclass(frozen=True, slots=True)
class PlacementAttemptRecord:
    """One append-only llama-fit-params execution attempt."""

    id: str
    placement_hash: str
    candidate_id: str
    host_id: str
    binary_id: str
    model_path: str
    started_at: str
    finished_at: str | None
    duration_ns: int | None
    status: PlacementAttemptStatus
    exit_code: int | None


@dataclass(frozen=True, slots=True)
class ServerRunRecord:
    """One append-only llama-server finalist-validation attempt."""

    id: str
    experiment_id: str
    candidate_id: str
    placement_id: str | None
    host_id: str
    server_binary_id: str
    target_model_id: str
    draft_model_id: str | None
    target_model_path: str | None
    draft_model_path: str | None
    spec_type: str | None
    spec_draft_n_max: int | None
    bind_host: str
    bind_port: int | None
    started_at: str
    ready_at: str | None
    finished_at: str | None
    duration_ns: int | None
    status: ServerRunStatus
    exit_code: int | None


@dataclass(frozen=True, slots=True)
class ServerBenchmarkRecord:
    """One SPEED-Bench invocation against a managed llama-server."""

    id: str
    server_run_id: str
    workload_case_id: str
    speed_bench_binary_id: str | None
    category: str
    status: ServerBenchmarkStatus
    duration_ns: int | None
    exit_code: int | None
    requests: int | None
    failed: int | None
    turns: int | None
    avg_prompt_ts: float | None
    avg_pred_ts: float | None
    avg_latency_ms: float | None
    draft_n: int | None
    accepted_n: int | None
    accept_rate: float | None
    created_at: str


@dataclass(frozen=True, slots=True)
class CandidateEvaluationRecord:
    """One append-only Candidate stage/decision event."""

    id: str
    experiment_id: str
    candidate_id: str
    stage: str
    decision: str
    reason: str | None
    metrics: Mapping[str, Any]
    created_at: str
