"""Typed persistence records that are not immutable domain definitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from llama_profile_lab.domain.concurrent_workload import (
    ConcurrentMemberStatus,
    ConcurrentQuality,
    ConcurrentRunStatus,
    ConcurrentWorkloadMode,
)
from llama_profile_lab.domain.deployment import (
    DeploymentFailureKind,
    DeploymentFeasibility,
    DeploymentRunStatus,
)
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

MemoryEstimateAttemptStatus = Literal[
    "running",
    "completed",
    "failed",
    "parser_failed",
    "timeout",
    "interrupted",
    "cancelled",
    "binary_changed",
]

DeploymentMemberStatus = Literal[
    "planned",
    "starting",
    "ready",
    "stopped",
    "failed",
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


@dataclass(frozen=True, slots=True)
class DeploymentCandidateRecord:
    """Persisted metadata for one immutable deployment Candidate."""

    id: str
    deployment_hash: str
    workload_suite_id: str
    created_at: str


@dataclass(frozen=True, slots=True)
class DeploymentRejectionRecord:
    """One persisted planner/feasibility rejection for a deployment Candidate."""

    id: str
    deployment_candidate_id: str
    stage: str
    reason: str
    details: Mapping[str, Any]
    created_at: str


@dataclass(frozen=True, slots=True)
class DeploymentInstanceRecord:
    """One model/server instance belonging to a deployment Candidate."""

    deployment_candidate_id: str
    instance_id: str
    candidate_id: str
    role: str
    model_artifact_id: str
    binary_id: str
    requested_placement: Mapping[str, Any]
    server_identity: str
    ordinal: int


@dataclass(frozen=True, slots=True)
class DeploymentPlacementRecord:
    """Persisted metadata for one immutable joint deployment placement."""

    id: str
    placement_hash: str
    deployment_candidate_id: str
    host_id: str
    feasibility: DeploymentFeasibility
    request: Mapping[str, Any]
    provenance: Mapping[str, Any]
    created_at: str


@dataclass(frozen=True, slots=True)
class PlacementDeviceMemoryRecord:
    """Normalized per-instance memory estimate on one device."""

    id: str
    deployment_placement_id: str
    deployment_candidate_id: str
    instance_id: str
    device_id: str
    model_bytes: int
    context_bytes: int
    compute_bytes: int
    total_bytes: int
    device_total_bytes: int
    device_free_bytes: int
    source: str
    measured_at: str | None


@dataclass(frozen=True, slots=True)
class DeploymentDeviceAllocationRecord:
    """Projected aggregate memory allocation on one physical device."""

    deployment_placement_id: str
    device_id: str
    projected_bytes: int
    reserved_margin_bytes: int
    device_total_bytes: int
    projected_free_bytes: int


@dataclass(frozen=True, slots=True)
class DeploymentRunRecord:
    """One append-only deployment execution attempt."""

    id: str
    deployment_candidate_id: str
    deployment_placement_id: str | None
    workload_case_id: str | None
    status: DeploymentRunStatus
    quality: RunQuality | None
    quality_details: Mapping[str, Any] | None
    failure_kind: DeploymentFailureKind | None
    failure_details: Mapping[str, Any] | None
    started_at: str | None
    finished_at: str | None
    duration_ns: int | None
    created_at: str


@dataclass(frozen=True, slots=True)
class DeploymentGpuSampleRecord:
    """One timestamped per-device GPU snapshot during deployment residency."""

    deployment_run_id: str
    timestamp_ns: int
    gpus: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class DeploymentRunMemberRecord:
    """One model-instance member of a deployment run."""

    deployment_run_id: str
    deployment_candidate_id: str
    instance_id: str
    server_run_id: str | None
    client_run_id: str | None
    endpoint: str | None
    member_status: DeploymentMemberStatus
    pid: int | None
    argv: tuple[str, ...]
    target_model_path: str | None
    draft_model_path: str | None
    started_at: str | None
    ready_at: str | None
    finished_at: str | None
    exit_code: int | None
    stdout: str
    stderr: str
    forced_kill: bool
    cleanup_error: str | None
    result: Mapping[str, Any]

@dataclass(frozen=True, slots=True)
class AcceleratorDeviceRecord:
    """Persisted logical-device inventory for one host/binary pair."""

    id: str
    host_id: str
    binary_id: str
    logical_device_name: str
    backend: str
    mapping_status: str
    physical_device_key: str | None
    pci_bus_id: str | None
    uuid: str | None
    vendor: str | None
    product_name: str | None
    total_memory_bytes: int | None
    free_memory_bytes: int | None
    driver: str | None
    runtime_metadata: Mapping[str, Any]
    raw_output: str
    observed_at: str


@dataclass(frozen=True, slots=True)
class MemoryEstimateAttemptRecord:
    """One append-only invocation of the structured memory helper."""

    id: str
    cache_hash: str
    candidate_id: str
    host_id: str
    helper_binary_id: str
    model_artifact_id: str
    request: Mapping[str, Any]
    argv: tuple[str, ...]
    status: MemoryEstimateAttemptStatus
    started_at: str
    finished_at: str | None
    duration_ns: int | None
    exit_code: int | None
    stdout: str
    stderr: str
    failure_details: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class MemoryEstimateRecord:
    """One immutable successful memory estimate reused by cache identity."""

    id: str
    cache_hash: str
    attempt_id: str
    candidate_id: str
    host_id: str
    helper_binary_id: str
    model_artifact_id: str
    identity: Mapping[str, Any]
    result: Mapping[str, Any]
    created_at: str


@dataclass(frozen=True, slots=True)
class MemoryEstimateDeviceRecord:
    """One normalized per-device row from a successful memory estimate."""

    memory_estimate_id: str
    ordinal: int
    logical_device_name: str
    model_bytes: int
    context_bytes: int
    compute_bytes: int
    total_bytes: int
    device_total_bytes: int
    device_free_bytes: int

@dataclass(frozen=True, slots=True)
class DeploymentPlanRecord:
    """Persisted counts and provenance for one deployment planner pass."""

    id: str
    base_deployment_candidate_id: str
    host_id: str
    search_hash: str
    raw_combinations: int
    rejected_by_constraints: int
    duplicate_candidates: int
    symmetry_reduced: int
    capability_rejected: int
    estimate_failed: int
    memory_rejected: int
    valid_count: int
    request: Mapping[str, Any]
    created_at: str


@dataclass(frozen=True, slots=True)
class DeploymentPlanCaseRecord:
    """One deterministic feasible deployment case in a persisted plan."""

    deployment_plan_id: str
    ordinal: int
    case_hash: str
    deployment_candidate_id: str
    deployment_placement_id: str
    generation: Mapping[str, Any]

@dataclass(frozen=True, slots=True)
class ConcurrentWorkloadCaseRecord:
    """Persisted immutable concurrent workload case."""

    id: str
    deployment_candidate_id: str
    case_hash: str
    phase: str
    definition: Mapping[str, Any]
    created_at: str


@dataclass(frozen=True, slots=True)
class DeploymentWorkloadRunRecord:
    """One synchronized concurrent workload phase."""

    id: str
    deployment_run_id: str
    workload_case_id: str
    phase: str
    status: ConcurrentRunStatus
    quality: ConcurrentQuality | None
    correctness_valid: bool
    barrier_release_ns: int | None
    overlap_start_ns: int | None
    overlap_end_ns: int | None
    overlap_duration_ns: int | None
    prompt_tokens: int
    decode_tokens: int
    combined_prompt_tps: float | None
    combined_decode_tps: float | None
    min_retention: float | None
    failure_kind: str | None
    failure_details: Mapping[str, Any] | None
    created_at: str


@dataclass(frozen=True, slots=True)
class StandaloneBaselineRecord:
    """One exact standalone throughput denominator."""

    id: str
    candidate_id: str
    resolved_placement_id: str
    host_id: str
    binary_id: str
    mode: ConcurrentWorkloadMode
    prompt_tokens: int
    generate_tokens: int
    depth_tokens: int
    throughput_tps: float
    latency_ms: float | None
    source: Mapping[str, Any]
    created_at: str


@dataclass(frozen=True, slots=True)
class DeploymentWorkloadMemberRecord:
    """One instance result inside a concurrent workload phase."""

    deployment_workload_run_id: str
    instance_id: str
    ordinal: int
    mode: ConcurrentWorkloadMode
    status: ConcurrentMemberStatus
    client_ready_ns: int
    barrier_release_ns: int
    first_request_ns: int | None
    first_token_ns: int | None
    last_token_ns: int | None
    finished_ns: int | None
    prompt_tokens: int
    decode_tokens: int
    native_prompt_tps: float | None
    native_decode_tps: float | None
    overlap_prompt_tokens: int
    overlap_decode_tokens: int
    overlap_prompt_tps: float | None
    overlap_decode_tps: float | None
    latency_ms: float | None
    standalone_baseline_id: str | None
    standalone_tps: float | None
    retention: float | None
    throughput_loss_pct: float | None
    baseline_latency_ms: float | None
    latency_increase_pct: float | None
    correctness_valid: bool
    raw: Mapping[str, Any]
    failure_details: Mapping[str, Any] | None

