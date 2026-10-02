"""Stable HTTP request/response DTOs for the local API."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt

from llama_profile_lab.domain import (
    BaseCandidateBaseline,
    Candidate,
    CandidateBaseline,
    ExperimentDefinition,
    FixedPlacementPolicy,
    MeasurementPolicy,
    PerCandidatePlacementPolicy,
    SearchSpace,
    TelemetrySample,
    WorkloadSuite,
)
from llama_profile_lab.domain.base import JsonScalar


class ApiModel(BaseModel):
    """Strict mutable API model; domain models stay independently immutable."""

    model_config = ConfigDict(extra="forbid")


class HealthResponse(ApiModel):
    status: Literal["ok"] = "ok"
    schema_version: NonNegativeInt


class LauncherProfileDTO(ApiModel):
    id: str
    binary_key: str
    binary_path: str
    profiles: tuple[str, ...]
    model_path: str
    draft_model_path: str | None = None
    server_alias: str | None = None
    args: dict[str, JsonScalar]


class ProfileListResponse(ApiModel):
    configured: bool
    source_path: str | None
    items: tuple[LauncherProfileDTO, ...]


class BinaryDTO(ApiModel):
    id: str
    sha256: str
    kind: str
    path: str
    size_bytes: NonNegativeInt
    mtime_ns: NonNegativeInt
    git_commit: str | None
    git_branch: str | None
    git_dirty: bool | None
    build_number: str | None
    build_info: dict[str, Any]
    capabilities: dict[str, Any]
    created_at: str


class BinaryListResponse(ApiModel):
    items: tuple[BinaryDTO, ...]


class BinaryInspectRequest(ApiModel):
    paths: Annotated[tuple[str, ...], Field(min_length=1)]
    kind: Literal[
        "auto",
        "llama-bench",
        "llama-fit-params",
        "llama-server",
        "speed-bench",
    ] = "auto"


class ModelFileDTO(ApiModel):
    id: str
    part_index: NonNegativeInt
    path: str
    sha256: str
    size_bytes: NonNegativeInt


class ModelDTO(ApiModel):
    id: str
    identity_hash: str
    architecture: str | None
    parameter_count: NonNegativeInt | None
    quantization: str | None
    size_bytes: NonNegativeInt
    metadata: dict[str, Any]
    created_at: str
    files: tuple[ModelFileDTO, ...]


class ModelListResponse(ApiModel):
    items: tuple[ModelDTO, ...]


type PlacementPolicyDTO = Annotated[
    PerCandidatePlacementPolicy | FixedPlacementPolicy,
    Field(discriminator="type"),
]
type BaselinePolicyDTO = Annotated[
    BaseCandidateBaseline | CandidateBaseline,
    Field(discriminator="type"),
]


class ExperimentCreateRequest(ApiModel):
    name: Annotated[str, Field(min_length=1)]
    base_candidate: Candidate
    search_space: SearchSpace
    workload_suite: WorkloadSuite
    measurement_policy: MeasurementPolicy
    placement_policy: PlacementPolicyDTO = Field(
        default_factory=PerCandidatePlacementPolicy
    )
    baseline: BaselinePolicyDTO = Field(default_factory=BaseCandidateBaseline)


class ExperimentCloneRequest(ApiModel):
    name: Annotated[str, Field(min_length=1)] | None = None


class ExperimentDTO(ApiModel):
    id: str
    status: str
    name: str
    base_candidate_id: str
    search_space_id: str
    workload_suite_id: str
    measurement_policy_id: str
    created_at: str
    frozen_at: str | None
    completed_at: str | None
    candidate_count: NonNegativeInt
    workload_count: NonNegativeInt
    benchmark_case_count: NonNegativeInt
    incomplete_case_count: NonNegativeInt
    definition: ExperimentDefinition


class ExperimentListResponse(ApiModel):
    items: tuple[ExperimentDTO, ...]


class PlanSummaryDTO(ApiModel):
    experiment_id: str
    experiment_name: str
    raw_combinations: NonNegativeInt
    rejected_by_constraints: NonNegativeInt
    duplicate_candidates: NonNegativeInt
    candidate_count: NonNegativeInt
    workloads_per_candidate: NonNegativeInt | None
    benchmark_case_count: NonNegativeInt
    unique_workload_count: NonNegativeInt


class ExecutionRequest(ApiModel):
    binary_id: str
    model_path: str
    fit_binary_id: str | None = None
    timeout_seconds: Annotated[float, Field(gt=0)] | None = None
    fit_timeout_seconds: Annotated[float, Field(gt=0)] | None = None
    limit: PositiveInt | None = None
    telemetry_interval_ms: Annotated[int, Field(ge=500)] = 1000


class ExecutionSummaryDTO(ApiModel):
    experiment_id: str
    attempted: NonNegativeInt
    completed: NonNegativeInt
    failed: NonNegativeInt
    remaining: NonNegativeInt
    interrupted: bool
    limited: bool


OperationStatus = Literal[
    "running",
    "pausing",
    "cancelling",
    "completed",
    "paused",
    "cancelled",
    "failed",
]


class OperationDTO(ApiModel):
    id: str
    experiment_id: str
    status: OperationStatus
    started_at: str
    finished_at: str | None
    requested_action: Literal["pause", "cancel"] | None
    summary: ExecutionSummaryDTO | None
    error: str | None


class ExperimentProgressDTO(ApiModel):
    experiment_id: str
    experiment_status: str
    total_cases: NonNegativeInt
    completed_cases: NonNegativeInt
    incomplete_cases: NonNegativeInt
    case_status_counts: dict[str, NonNegativeInt]
    operation: OperationDTO | None


class CandidateDTO(ApiModel):
    id: str
    ordinal: NonNegativeInt
    generation_metadata: dict[str, Any]
    candidate: Candidate
    workload_count: NonNegativeInt
    benchmark_case_count: NonNegativeInt
    completed_case_count: NonNegativeInt
    server_validation_count: NonNegativeInt


class CandidateListResponse(ApiModel):
    items: tuple[CandidateDTO, ...]


class BenchmarkSampleDTO(ApiModel):
    sample_index: NonNegativeInt
    elapsed_ns: NonNegativeInt
    tokens_per_second: float


class RunSummaryDTO(ApiModel):
    id: str
    benchmark_case_id: str
    candidate_id: str
    workload_case_id: str
    placement_id: str | None
    workload_kind: str
    host_id: str
    binary_id: str
    measurement_policy_id: str
    started_at: str
    finished_at: str | None
    duration_ns: NonNegativeInt | None
    status: str
    exit_code: int | None
    quality: str | None
    quality_details: dict[str, Any] | None


class RunListResponse(ApiModel):
    items: tuple[RunSummaryDTO, ...]


class RunDetailDTO(RunSummaryDTO):
    samples: tuple[BenchmarkSampleDTO, ...]
    metrics: dict[str, int | float]
    stdout: str
    stderr: str


class TelemetryResponse(ApiModel):
    run_id: str
    samples: tuple[TelemetrySample, ...]


class ResultsResponse(ApiModel):
    experiment_id: str
    rows: tuple[dict[str, Any], ...]


class ServerValidationRequest(ApiModel):
    experiment_id: str
    server_binary_id: str
    speed_bench_binary_id: str
    model_path: str
    draft_model_path: str | None = None
    placement_id: str | None = None
    model_name: str | None = None
    host: str = "127.0.0.1"
    port: Annotated[int, Field(ge=1, le=65535)] = 8080
    readiness_timeout_seconds: Annotated[float, Field(gt=0)] = 300.0
    request_timeout_seconds: Annotated[float, Field(gt=0)] = 600.0
    benchmark_timeout_seconds: Annotated[float, Field(gt=0)] | None = None
    workload_case_id: str | None = None


class ServerValidationResponse(ApiModel):
    experiment_id: str
    candidate_id: str
    server_run_id: str
    benchmark_ids: tuple[str, ...]
    completed: bool
    speculative: bool
