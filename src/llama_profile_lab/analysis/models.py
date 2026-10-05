"""Typed analysis results used by CLI, API, and future UI layers."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, NonNegativeInt, PositiveInt

from llama_profile_lab.domain.base import FrozenModel, JsonScalar

NonEmptyString = Annotated[str, Field(min_length=1)]
ObjectiveDirection = Literal["maximize", "minimize"]


class AnalysisFilter(FrozenModel):
    """One exact-match filter over Candidate, workload, or run metadata."""

    path: NonEmptyString
    value: JsonScalar


class MatrixCell(FrozenModel):
    """One populated sparse X/Y cell."""

    x: JsonScalar
    y: JsonScalar
    value: float
    candidate_id: NonEmptyString
    workload_case_id: NonEmptyString
    run_count: PositiveInt
    sample_count: NonNegativeInt


class MatrixFacet(FrozenModel):
    """One optional facet slice containing sparse cells."""

    value: JsonScalar = None
    cells: tuple[MatrixCell, ...]


class MatrixProjection(FrozenModel):
    """Sparse two-dimensional projection suitable for CLI/UI rendering."""

    experiment_id: NonEmptyString
    x_path: NonEmptyString
    y_path: NonEmptyString
    metric: NonEmptyString
    facet_path: NonEmptyString | None = None
    x_values: tuple[JsonScalar, ...]
    y_values: tuple[JsonScalar, ...]
    facets: tuple[MatrixFacet, ...]


class BaselineDelta(FrozenModel):
    """Signed metric difference for one suite workload."""

    suite_case_index: NonNegativeInt
    workload_label: NonEmptyString
    metric: NonEmptyString
    baseline_value: float | None
    candidate_value: float | None
    delta: float | None
    percent_delta: float | None


class CandidateComparison(FrozenModel):
    """Candidate versus baseline without an evaluative winner."""

    experiment_id: NonEmptyString
    candidate_id: NonEmptyString
    baseline_candidate_id: NonEmptyString
    deltas: tuple[BaselineDelta, ...]


class ParetoObjective(FrozenModel):
    """One user-selected objective and its workload/environment filters."""

    key: NonEmptyString
    direction: ObjectiveDirection
    metric: NonEmptyString
    filters: tuple[AnalysisFilter, ...] = ()


class ParetoCandidate(FrozenModel):
    """One complete objective vector on the Pareto frontier."""

    candidate_id: NonEmptyString
    candidate_ordinal: NonNegativeInt
    values: dict[str, float]


class ParetoResult(FrozenModel):
    """Non-dominated Candidate set plus incomplete Candidate reasons."""

    experiment_id: NonEmptyString
    objectives: tuple[ParetoObjective, ...]
    evaluated_count: NonNegativeInt
    frontier: tuple[ParetoCandidate, ...]
    excluded: dict[str, str]


class CurvePoint(FrozenModel):
    """Measured throughput curve point used by latency interpolation."""

    tokens: NonNegativeInt
    tokens_per_second: Annotated[float, Field(gt=0)]


class LatencyEstimate(FrozenModel):
    """Compute-only request latency estimated from PP and TG curves."""

    experiment_id: NonEmptyString
    candidate_id: NonEmptyString
    prompt_tokens: PositiveInt
    generate_tokens: PositiveInt
    decode_start_depth_tokens: NonNegativeInt
    prefill_tokens_per_second: Annotated[float, Field(gt=0)]
    decode_average_tokens_per_second: Annotated[float, Field(gt=0)]
    prefill_seconds: Annotated[float, Field(ge=0)]
    decode_seconds: Annotated[float, Field(ge=0)]
    total_seconds: Annotated[float, Field(ge=0)]
    prefill_curve: tuple[CurvePoint, ...]
    decode_curve: tuple[CurvePoint, ...]

class DeploymentAnalysisFilter(FrozenModel):
    """One exact deployment-analysis coordinate filter."""

    path: NonEmptyString
    value: JsonScalar


class DeploymentMatrixCell(FrozenModel):
    """One unambiguous deployment projection cell."""

    x: JsonScalar
    y: JsonScalar
    value: float
    deployment_candidate_id: NonEmptyString
    deployment_placement_id: NonEmptyString
    workload_case_id: NonEmptyString | None = None
    observation_count: PositiveInt


class DeploymentMatrixFacet(FrozenModel):
    """One optional deployment-analysis facet slice."""

    value: JsonScalar = None
    cells: tuple[DeploymentMatrixCell, ...]


class DeploymentMatrixProjection(FrozenModel):
    """Sparse deployment projection with exact hidden-coordinate semantics."""

    x_path: NonEmptyString
    y_path: NonEmptyString
    metric: NonEmptyString
    facet_path: NonEmptyString | None = None
    x_values: tuple[JsonScalar, ...]
    y_values: tuple[JsonScalar, ...]
    facets: tuple[DeploymentMatrixFacet, ...]


class DeploymentMemoryRow(FrozenModel):
    """One projected or runtime row in the device memory matrix."""

    key: NonEmptyString
    source: Literal["projected", "runtime"]
    values: dict[str, NonNegativeInt | None]


class DeploymentMemoryDelta(FrozenModel):
    """Signed projected-versus-runtime memory deltas for one physical device."""

    device_id: NonEmptyString
    projected_bytes: NonNegativeInt
    runtime_peak_used_bytes: NonNegativeInt | None = None
    used_delta_bytes: int | None = None
    projected_free_bytes: NonNegativeInt
    runtime_min_free_bytes: NonNegativeInt | None = None
    free_delta_bytes: int | None = None


class DeploymentMemoryMatrix(FrozenModel):
    """Device-by-instance memory projection with runtime evidence separated."""

    deployment_placement_id: NonEmptyString
    deployment_run_id: NonEmptyString | None = None
    devices: tuple[NonEmptyString, ...]
    rows: tuple[DeploymentMemoryRow, ...]
    deltas: tuple[DeploymentMemoryDelta, ...] = ()


class DeploymentInterferenceMember(FrozenModel):
    """Per-instance standalone/concurrent interference evidence."""

    instance_id: NonEmptyString
    mode: Literal["prefill", "decode"]
    native_tps: Annotated[float, Field(ge=0)] | None = None
    overlap_tps: Annotated[float, Field(ge=0)] | None = None
    standalone_tps: Annotated[float, Field(gt=0)] | None = None
    retention: Annotated[float, Field(ge=0)] | None = None
    throughput_loss_pct: float | None = None
    latency_ms: Annotated[float, Field(ge=0)] | None = None
    baseline_latency_ms: Annotated[float, Field(ge=0)] | None = None
    latency_increase_pct: float | None = None
    correctness_valid: bool


class DeploymentInterferenceView(FrozenModel):
    """One DD/PP/PD/DP phase with aggregate and per-instance interference."""

    deployment_run_id: NonEmptyString
    workload_run_id: NonEmptyString
    deployment_placement_id: NonEmptyString
    phase: Literal["dd", "pp", "pd", "dp"]
    quality: NonEmptyString | None = None
    combined_prompt_tps: Annotated[float, Field(ge=0)] | None = None
    combined_decode_tps: Annotated[float, Field(ge=0)] | None = None
    min_retention: Annotated[float, Field(ge=0)] | None = None
    members: tuple[DeploymentInterferenceMember, ...]


class DeploymentMetricDelta(FrozenModel):
    """Signed placement-versus-baseline scalar metric delta."""

    metric: NonEmptyString
    baseline_value: float | None
    candidate_value: float | None
    delta: float | None
    percent_delta: float | None


class DeploymentComparison(FrozenModel):
    """Placement comparison preserving explicit baseline direction."""

    deployment_placement_id: NonEmptyString
    baseline_placement_id: NonEmptyString
    deltas: tuple[DeploymentMetricDelta, ...]


class DeploymentParetoObjective(FrozenModel):
    """One explicit deployment objective and exact workload filters."""

    key: NonEmptyString
    direction: ObjectiveDirection
    metric: NonEmptyString
    filters: tuple[DeploymentAnalysisFilter, ...] = ()


class DeploymentMetricConstraint(FrozenModel):
    """One metric threshold applied before deployment Pareto dominance."""

    metric: NonEmptyString
    operator: Literal["ge", "gt", "le", "lt", "eq"]
    value: float
    filters: tuple[DeploymentAnalysisFilter, ...] = ()


class DeploymentParetoPoint(FrozenModel):
    """One complete deployment-placement objective vector."""

    deployment_candidate_id: NonEmptyString
    deployment_placement_id: NonEmptyString
    values: dict[str, float]


class DeploymentParetoResult(FrozenModel):
    """Non-dominated deployment placements and explicit exclusion reasons."""

    objectives: tuple[DeploymentParetoObjective, ...]
    constraints: tuple[DeploymentMetricConstraint, ...] = ()
    evaluated_count: NonNegativeInt
    frontier: tuple[DeploymentParetoPoint, ...]
    excluded: dict[str, str]

