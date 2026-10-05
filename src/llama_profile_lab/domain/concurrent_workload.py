"""Concurrent multi-instance workload domain values for V2-M6."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import (
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

from llama_profile_lab.domain.base import ContentAddressedModel, FrozenModel, JsonScalar
from llama_profile_lab.domain.deployment import DeploymentPhase

NonEmptyString = Annotated[str, Field(min_length=1)]
type ConcurrentWorkloadMode = Literal["prefill", "decode"]
type ConcurrentMemberStatus = Literal[
    "ready",
    "completed",
    "failed",
    "timeout",
    "cancelled",
    "invalid",
]
type ConcurrentRunStatus = Literal[
    "running",
    "completed",
    "failed",
    "cancelled",
    "invalid",
]
type ConcurrentQuality = Literal[
    "clean",
    "baseline_missing",
    "baseline_ambiguous",
    "member_failed",
    "member_timeout",
    "no_overlap",
    "correctness_invalid",
]


class ConcurrentWorkloadMemberSpec(FrozenModel):
    """One instance's concrete workload within a synchronized phase."""

    ordinal: NonNegativeInt
    instance_id: NonEmptyString
    mode: ConcurrentWorkloadMode
    prompt_tokens: NonNegativeInt = 0
    generate_tokens: NonNegativeInt = 0
    depth_tokens: NonNegativeInt = 0
    prompt_token_id: NonNegativeInt = 1
    timeout_seconds: Annotated[float, Field(gt=0.0)] = 600.0
    require_exact_token_count: bool = True

    @model_validator(mode="after")
    def validate_mode(self) -> ConcurrentWorkloadMemberSpec:
        if self.mode == "prefill":
            if self.prompt_tokens <= 0 or self.generate_tokens != 0:
                raise ValueError(
                    "prefill workload requires prompt_tokens > 0 and "
                    "generate_tokens = 0"
                )
        elif self.generate_tokens <= 0:
            raise ValueError("decode workload requires generate_tokens > 0")
        return self


class ConcurrentWorkloadCase(ContentAddressedModel):
    """One synchronized workload phase spanning two or more resident instances."""

    schema_name: Literal["llama-concurrent-workload-case"] = Field(
        default="llama-concurrent-workload-case",
        alias="schema",
    )
    version: Literal[1] = 1
    phase: DeploymentPhase
    members: Annotated[
        tuple[ConcurrentWorkloadMemberSpec, ...],
        Field(min_length=2),
    ]
    provenance: Mapping[str, JsonScalar] = Field(default_factory=dict)

    @field_validator("members")
    @classmethod
    def validate_members(
        cls,
        value: tuple[ConcurrentWorkloadMemberSpec, ...],
    ) -> tuple[ConcurrentWorkloadMemberSpec, ...]:
        instance_ids = [item.instance_id for item in value]
        if len(instance_ids) != len(set(instance_ids)):
            raise ValueError("concurrent workload instance IDs must be unique")
        ordinals = [item.ordinal for item in value]
        if sorted(ordinals) != list(range(len(value))):
            raise ValueError(
                "concurrent workload member ordinals must be contiguous from zero"
            )
        return tuple(sorted(value, key=lambda item: item.ordinal))

    @model_validator(mode="after")
    def validate_canonical_phase(self) -> ConcurrentWorkloadCase:
        if len(self.members) != 2:
            return self
        modes = tuple(item.mode for item in self.members)
        expected = {
            "dd": ("decode", "decode"),
            "pp": ("prefill", "prefill"),
            "pd": ("prefill", "decode"),
            "dp": ("decode", "prefill"),
        }[self.phase]
        if modes != expected:
            raise ValueError(
                f"phase {self.phase} requires member modes {expected}, got {modes}"
            )
        return self


class ConcurrentTokenEvent(FrozenModel):
    """Cumulative completed-token counter observed at one monotonic timestamp."""

    kind: ConcurrentWorkloadMode
    timestamp_ns: NonNegativeInt
    cumulative_tokens: NonNegativeInt


class ConcurrentMemberResult(FrozenModel):
    """Normalized result from one synchronized client."""

    instance_id: NonEmptyString
    mode: ConcurrentWorkloadMode
    status: ConcurrentMemberStatus
    client_ready_ns: NonNegativeInt
    barrier_release_ns: NonNegativeInt
    first_request_ns: NonNegativeInt | None = None
    first_token_ns: NonNegativeInt | None = None
    last_token_ns: NonNegativeInt | None = None
    finished_ns: NonNegativeInt | None = None
    prompt_tokens: NonNegativeInt = 0
    decode_tokens: NonNegativeInt = 0
    native_prompt_tps: Annotated[float, Field(ge=0.0)] | None = None
    native_decode_tps: Annotated[float, Field(ge=0.0)] | None = None
    latency_ms: Annotated[float, Field(ge=0.0)] | None = None
    token_events: tuple[ConcurrentTokenEvent, ...] = ()
    correctness_valid: bool = True
    failure: NonEmptyString | None = None
    raw: Mapping[str, JsonScalar] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_timing(self) -> ConcurrentMemberResult:
        if self.client_ready_ns > self.barrier_release_ns:
            raise ValueError("client_ready_ns cannot follow barrier release")
        ordered = [
            item
            for item in (
                self.first_request_ns,
                self.first_token_ns,
                self.last_token_ns,
                self.finished_ns,
            )
            if item is not None
        ]
        if ordered != sorted(ordered):
            raise ValueError("member timing values must be monotonic")
        previous: dict[str, int] = {}
        for event in self.token_events:
            key = event.kind
            last = previous.get(key)
            if last is not None and event.cumulative_tokens < last:
                raise ValueError("token event counters must be cumulative")
            previous[key] = event.cumulative_tokens
        return self


class ConcurrentOverlapResult(FrozenModel):
    """Overlap-normalized deployment throughput for one phase."""

    overlap_start_ns: NonNegativeInt
    overlap_end_ns: NonNegativeInt
    overlap_duration_ns: PositiveInt
    prompt_tokens: NonNegativeInt
    decode_tokens: NonNegativeInt
    combined_prompt_tps: Annotated[float, Field(ge=0.0)] | None = None
    combined_decode_tps: Annotated[float, Field(ge=0.0)] | None = None


class ConcurrentRetention(FrozenModel):
    """Standalone-to-concurrent retention for one deployment member."""

    instance_id: NonEmptyString
    mode: ConcurrentWorkloadMode
    baseline_id: NonEmptyString
    standalone_tps: Annotated[float, Field(gt=0.0)]
    concurrent_tps: Annotated[float, Field(ge=0.0)]
    retention: Annotated[float, Field(ge=0.0)]
    throughput_loss_pct: float
