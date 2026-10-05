"""Stable HTTP request/response DTOs for the local API."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt

from llama_profile_lab.analysis import (
    DeploymentMemoryMatrix,
    DeploymentParetoResult,
    ParetoObjective,
)
from llama_profile_lab.domain import (
    BaseCandidateBaseline,
    Candidate,
    CandidateBaseline,
    DeploymentCandidate,
    DeploymentPlacement,
    DeploymentSearchSpace,
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
    candidate: Candidate


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
        "llama-memory-estimator",
        "llama-server",
        "speed-bench",
    ] = "auto"


class AcceleratorDeviceDTO(ApiModel):
    logical_device_name: str
    backend: str
    mapping_status: str
    physical_device_key: str | None
    pci_bus_id: str | None
    uuid: str | None
    vendor: str | None
    product_name: str | None
    total_memory_bytes: NonNegativeInt | None
    free_memory_bytes: NonNegativeInt | None
    driver: str | None
    runtime_metadata: dict[str, JsonScalar]


class DeviceInventoryResponse(ApiModel):
    host_id: str
    binary_id: str
    items: tuple[AcceleratorDeviceDTO, ...]


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


class ExperimentPreviewRequest(ApiModel):
    base_candidate: Candidate
    search_space: SearchSpace
    workload_suite: WorkloadSuite


class PlanPreviewDTO(ApiModel):
    raw_combinations: NonNegativeInt
    rejected_by_constraints: NonNegativeInt
    duplicate_candidates: NonNegativeInt
    candidate_count: NonNegativeInt
    workloads_per_candidate: NonNegativeInt | None
    benchmark_case_count: NonNegativeInt
    unique_workload_count: NonNegativeInt


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
    base_candidate: Candidate
    search_space: SearchSpace
    workload_suite: WorkloadSuite
    measurement_policy: MeasurementPolicy


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
    current_candidate_id: str | None = None
    current_candidate_ordinal: NonNegativeInt | None = None
    current_workload_case_id: str | None = None
    current_suite_case_index: NonNegativeInt | None = None
    latest_run_id: str | None = None
    latest_tokens_per_second: float | None = None
    latest_metrics: dict[str, int | float] = Field(default_factory=dict)


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


class PromotionRequest(ApiModel):
    experiment_id: str
    source_profile_id: str | None = None


class LauncherArgChangeDTO(ApiModel):
    path: str
    argument: str
    before: JsonScalar
    after: JsonScalar


class PromotionResponse(ApiModel):
    id: str
    experiment_id: str
    candidate_id: str
    source_profile: str
    changes: tuple[LauncherArgChangeDTO, ...]
    patch: str
    source_snapshot: dict[str, Any]
    proposed_snapshot: dict[str, Any]
    validation: dict[str, Any]


class DeploymentPromotionSourceDTO(ApiModel):
    instance_id: Annotated[str, Field(min_length=1)]
    experiment_id: Annotated[str, Field(min_length=1)]
    source_profile_id: Annotated[str, Field(min_length=1)] | None = None


class DeploymentPromotionRequest(ApiModel):
    deployment_placement_id: Annotated[str, Field(min_length=1)]
    sources: Annotated[
        tuple[DeploymentPromotionSourceDTO, ...],
        Field(min_length=2),
    ]


class DeploymentPromotionResponse(ApiModel):
    id: str
    base_deployment_candidate_id: str
    deployment_candidate_id: str
    deployment_placement_id: str
    sources: tuple[dict[str, Any], ...]
    changes: tuple[dict[str, Any], ...]
    patch: str
    source_snapshot: dict[str, Any]
    proposed_snapshot: dict[str, Any]
    evidence: dict[str, Any]


class ParameterDefinitionDTO(ApiModel):
    path: str
    label: str
    category: str
    value_types: tuple[str, ...]
    cli_argument: str | None
    affects_placement: bool
    supported_by: tuple[str, ...]
    minimum: int | float | None
    maximum: int | float | None
    string_choices: tuple[str, ...] | None


class ParameterListResponse(ApiModel):
    items: tuple[ParameterDefinitionDTO, ...]


class MetricDefinitionDTO(ApiModel):
    name: str
    label: str
    unit: str


class MetricListResponse(ApiModel):
    items: tuple[MetricDefinitionDTO, ...]


class PlacementDTO(ApiModel):
    id: str
    candidate_id: str
    host_id: str
    binary_id: str
    fit_attempt_id: str | None
    production_context_size: PositiveInt
    n_gpu_layers: NonNegativeInt
    n_cpu_moe: NonNegativeInt
    split_mode: str
    main_gpu: NonNegativeInt
    devices: str | tuple[str, ...]
    tensor_split: tuple[float, ...] | None
    override_tensor: tuple[str, ...]
    request: dict[str, Any]
    created_at: str


class PlacementListResponse(ApiModel):
    items: tuple[PlacementDTO, ...]


class ParetoRequestDTO(ApiModel):
    objectives: Annotated[tuple[ParetoObjective, ...], Field(min_length=1)]
    filters: tuple[str, ...] = ()
    qualities: tuple[str, ...] = ()


class CandidateEvaluationDTO(ApiModel):
    id: str
    stage: str
    decision: str
    reason: str | None
    metrics: dict[str, Any]
    created_at: str


class ServerBenchmarkDTO(ApiModel):
    id: str
    server_run_id: str
    workload_case_id: str
    category: str
    status: str
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


class CandidateValidationHistoryDTO(ApiModel):
    experiment_id: str
    candidate_id: str
    evaluations: tuple[CandidateEvaluationDTO, ...]
    benchmarks: tuple[ServerBenchmarkDTO, ...]

class DeploymentCreateRequest(ApiModel):
    """Create one immutable base deployment definition."""

    deployment: DeploymentCandidate


class DeploymentDTO(ApiModel):
    """Deployment definition plus computed persisted state counts."""

    id: str
    status: str
    created_at: str
    definition: DeploymentCandidate
    plan_count: NonNegativeInt
    placement_count: NonNegativeInt
    run_count: NonNegativeInt
    latest_plan_id: str | None = None


class DeploymentListResponse(ApiModel):
    items: tuple[DeploymentDTO, ...]


class DeploymentEstimatorInputDTO(ApiModel):
    instance_id: Annotated[str, Field(min_length=1)]
    helper_binary_id: Annotated[str, Field(min_length=1)]
    model_path: Annotated[str, Field(min_length=1)]


class DeploymentPlanRequest(ApiModel):
    search_space: DeploymentSearchSpace
    instances: Annotated[
        tuple[DeploymentEstimatorInputDTO, ...],
        Field(min_length=1),
    ]
    timeout_seconds: Annotated[float, Field(gt=0)] | None = 300.0


class DeploymentPlanCaseDTO(ApiModel):
    deployment_candidate_id: str
    deployment_placement_id: str
    generation: dict[str, Any]


class DeploymentPlanResponse(ApiModel):
    base_deployment_candidate_id: str
    host_id: str
    raw_combinations: NonNegativeInt
    rejected_by_constraints: NonNegativeInt
    duplicate_candidates: NonNegativeInt
    symmetry_reduced: NonNegativeInt
    capability_rejected: NonNegativeInt
    estimate_failed: NonNegativeInt
    memory_rejected: NonNegativeInt
    valid_count: NonNegativeInt
    plan_id: str | None
    cases: tuple[DeploymentPlanCaseDTO, ...]


class DeploymentRejectionDTO(ApiModel):
    id: str
    stage: str
    reason: str
    details: dict[str, Any]
    created_at: str


class DeploymentCandidateItemDTO(ApiModel):
    id: str
    definition: DeploymentCandidate
    generation: dict[str, Any]
    rejection_count: NonNegativeInt
    rejections: tuple[DeploymentRejectionDTO, ...] = ()
    placement_ids: tuple[str, ...]


class DeploymentCandidateListResponse(ApiModel):
    items: tuple[DeploymentCandidateItemDTO, ...]


class DeploymentPlacementDTO(ApiModel):
    id: str
    deployment_candidate_id: str
    host_id: str
    feasibility: str
    placement: DeploymentPlacement
    memory: DeploymentMemoryMatrix


class DeploymentPlacementListResponse(ApiModel):
    items: tuple[DeploymentPlacementDTO, ...]


class DeploymentServerInputDTO(ApiModel):
    instance_id: Annotated[str, Field(min_length=1)]
    model_path: Annotated[str, Field(min_length=1)]
    draft_model_path: Annotated[str, Field(min_length=1)] | None = None


class DeploymentStandaloneBaselineDTO(ApiModel):
    instance_id: Annotated[str, Field(min_length=1)]
    mode: Literal["prefill", "decode"]
    prompt_tokens: NonNegativeInt
    generate_tokens: NonNegativeInt
    depth_tokens: NonNegativeInt
    throughput_tps: Annotated[float, Field(gt=0)]
    latency_ms: Annotated[float, Field(ge=0)] | None = None


class DeploymentRunRequest(ApiModel):
    deployment_placement_id: Annotated[str, Field(min_length=1)]
    instances: Annotated[
        tuple[DeploymentServerInputDTO, ...],
        Field(min_length=1),
    ]
    standalone_baselines: tuple[DeploymentStandaloneBaselineDTO, ...] = ()
    host: Annotated[str, Field(min_length=1)] = "127.0.0.1"
    readiness_timeout_seconds: Annotated[float, Field(gt=0)] = 300.0


DeploymentOperationStatus = Literal[
    "running",
    "pausing",
    "cancelling",
    "completed",
    "paused",
    "cancelled",
    "failed",
]


class DeploymentOperationDTO(ApiModel):
    id: str
    deployment_candidate_id: str
    deployment_placement_id: str
    deployment_run_id: str | None
    status: DeploymentOperationStatus
    requested_action: Literal["pause", "cancel"] | None
    started_at: str
    finished_at: str | None
    error: str | None


class DeploymentMemberStateDTO(ApiModel):
    instance_id: str
    status: str
    endpoint: str | None
    pid: int | None
    ready_at: str | None
    exit_code: int | None


class DeploymentProgressDTO(ApiModel):
    deployment_id: str
    deployment_status: str
    planned_candidates: NonNegativeInt
    completed_candidates: NonNegativeInt
    failed_candidates: NonNegativeInt
    active_deployment_run: str | None
    current_deployment_candidate_id: str | None = None
    current_placement_id: str | None = None
    member_states: tuple[DeploymentMemberStateDTO, ...]
    current_workload_phase: str | None
    combined_prompt_tps: float | None = None
    combined_decode_tps: float | None = None
    memory: DeploymentMemoryMatrix | None = None
    failure_kind: str | None = None
    failure_details: dict[str, Any] | None = None
    operation: DeploymentOperationDTO | None


class DeploymentWorkloadPhaseDTO(ApiModel):
    id: str
    phase: str
    status: str
    quality: str | None
    correctness_valid: bool
    combined_prompt_tps: float | None
    combined_decode_tps: float | None
    min_retention: float | None
    failure_kind: str | None


class DeploymentRunMemberDTO(ApiModel):
    instance_id: str
    status: str
    endpoint: str | None
    pid: int | None
    ready_at: str | None
    finished_at: str | None
    exit_code: int | None
    forced_kill: bool
    cleanup_error: str | None


class DeploymentRunDTO(ApiModel):
    id: str
    deployment_candidate_id: str
    deployment_placement_id: str | None
    status: str
    quality: str | None
    failure_kind: str | None
    started_at: str | None
    finished_at: str | None
    duration_ns: NonNegativeInt | None
    members: tuple[DeploymentRunMemberDTO, ...]
    phases: tuple[DeploymentWorkloadPhaseDTO, ...]


class DeploymentRunListResponse(ApiModel):
    items: tuple[DeploymentRunDTO, ...]


class DeploymentResultsResponse(ApiModel):
    deployment_id: str
    rows: tuple[dict[str, Any], ...]


class DeploymentParetoResponse(ApiModel):
    deployment_id: str
    result: DeploymentParetoResult

