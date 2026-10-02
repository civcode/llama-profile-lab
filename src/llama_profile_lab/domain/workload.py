"""Workload-suite and concrete WorkloadCase definitions."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, NonNegativeInt, PositiveInt, model_validator

from llama_profile_lab.domain.base import ContentAddressedModel, FrozenModel

NonEmptyString = Annotated[str, Field(min_length=1)]


class AbsoluteDepth(FrozenModel):
    """An explicit active-context depth."""

    type: Literal["absolute"] = "absolute"
    tokens: NonNegativeInt


class FractionalDepth(FrozenModel):
    """A depth expressed as a fraction of the available context."""

    type: Literal["fraction"] = "fraction"
    value: Annotated[float, Field(ge=0.0, le=1.0)]


type DepthExpression = Annotated[
    AbsoluteDepth | FractionalDepth,
    Field(discriminator="type"),
]


class SuiteCaseBase(FrozenModel):
    """Human-authored workload-suite case metadata."""

    label: NonEmptyString | None = None
    safety_margin_tokens: NonNegativeInt = 0


class PrefillSuiteCase(SuiteCaseBase):
    """Symbolic prompt-processing workload."""

    kind: Literal["microbench-prefill"] = "microbench-prefill"
    prompt_tokens: PositiveInt
    depth: DepthExpression


class DecodeSuiteCase(SuiteCaseBase):
    """Symbolic token-generation workload."""

    kind: Literal["microbench-decode"] = "microbench-decode"
    generate_tokens: PositiveInt
    depth: DepthExpression


class CombinedSuiteCase(SuiteCaseBase):
    """Symbolic prompt + generation workload."""

    kind: Literal["microbench-combined"] = "microbench-combined"
    prompt_tokens: PositiveInt
    generate_tokens: PositiveInt
    depth: DepthExpression


class SpeedBenchConfig(FrozenModel):
    """End-to-end SPEED-Bench workload settings."""

    bench: NonEmptyString
    categories: Annotated[tuple[NonEmptyString, ...], Field(min_length=1)] = ("all",)
    output_tokens: PositiveInt
    concurrency: PositiveInt = 1
    limit: PositiveInt | None = None
    request: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class SpeedBenchSuiteCase(SuiteCaseBase):
    """Symbolic server/SPEED-Bench workload."""

    kind: Literal["speed-bench"] = "speed-bench"
    speed_bench: SpeedBenchConfig


type SuiteCase = Annotated[
    PrefillSuiteCase | DecodeSuiteCase | CombinedSuiteCase | SpeedBenchSuiteCase,
    Field(discriminator="kind"),
]


class WorkloadSuite(ContentAddressedModel):
    """Human-authored workload collection before Candidate-dependent expansion."""

    identity_exclude = frozenset({"id"})

    schema: Literal["llama-workload-suite"] = "llama-workload-suite"
    version: Literal[1] = 1
    id: NonEmptyString
    description: NonEmptyString | None = None
    cases: Annotated[tuple[SuiteCase, ...], Field(min_length=1)]


class WorkloadCaseBase(ContentAddressedModel):
    """Concrete immutable workload common fields."""

    schema: Literal["llama-workload-case"] = "llama-workload-case"
    version: Literal[1] = 1


class PrefillWorkloadCase(WorkloadCaseBase):
    """Concrete prompt-processing benchmark."""

    kind: Literal["microbench-prefill"] = "microbench-prefill"
    prompt_tokens: PositiveInt
    generate_tokens: Literal[0] = 0
    depth_tokens: NonNegativeInt


class DecodeWorkloadCase(WorkloadCaseBase):
    """Concrete token-generation benchmark."""

    kind: Literal["microbench-decode"] = "microbench-decode"
    prompt_tokens: Literal[0] = 0
    generate_tokens: PositiveInt
    depth_tokens: NonNegativeInt


class CombinedWorkloadCase(WorkloadCaseBase):
    """Concrete prompt + generation benchmark."""

    kind: Literal["microbench-combined"] = "microbench-combined"
    prompt_tokens: PositiveInt
    generate_tokens: PositiveInt
    depth_tokens: NonNegativeInt


class SpeedBenchWorkloadCase(WorkloadCaseBase):
    """Concrete SPEED-Bench case."""

    kind: Literal["speed-bench"] = "speed-bench"
    speed_bench: SpeedBenchConfig


type WorkloadCase = Annotated[
    PrefillWorkloadCase
    | DecodeWorkloadCase
    | CombinedWorkloadCase
    | SpeedBenchWorkloadCase,
    Field(discriminator="kind"),
]


class WorkloadEnvelope(FrozenModel):
    """Optional validation wrapper for a concrete WorkloadCase."""

    case: WorkloadCase
    context_size: PositiveInt

    @model_validator(mode="after")
    def validate_context_fit(self) -> WorkloadEnvelope:
        case = self.case
        if isinstance(
            case,
            PrefillWorkloadCase | DecodeWorkloadCase | CombinedWorkloadCase,
        ):
            occupied = case.depth_tokens + case.prompt_tokens + case.generate_tokens
            if occupied > self.context_size:
                raise ValueError("workload exceeds Candidate context size")
        return self
