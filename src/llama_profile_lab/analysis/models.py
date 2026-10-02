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
