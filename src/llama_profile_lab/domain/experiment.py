"""Experiment-definition domain models."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from llama_profile_lab.domain.base import FrozenModel

NonEmptyString = Annotated[str, Field(min_length=1)]


class PerCandidatePlacementPolicy(FrozenModel):
    """Resolve placement independently for each Candidate."""

    type: Literal["per-candidate"] = "per-candidate"


class FixedPlacementPolicy(FrozenModel):
    """Reuse one previously resolved placement."""

    type: Literal["fixed"] = "fixed"
    placement_id: NonEmptyString


type PlacementPolicy = Annotated[
    PerCandidatePlacementPolicy | FixedPlacementPolicy,
    Field(discriminator="type"),
]


class BaseCandidateBaseline(FrozenModel):
    """Use the experiment's base Candidate as baseline."""

    type: Literal["base-candidate"] = "base-candidate"


class CandidateBaseline(FrozenModel):
    """Use an explicitly selected Candidate as baseline."""

    type: Literal["candidate"] = "candidate"
    candidate_id: NonEmptyString


type BaselinePolicy = Annotated[
    BaseCandidateBaseline | CandidateBaseline,
    Field(discriminator="type"),
]


class ExperimentDefinition(FrozenModel):
    """Immutable experiment definition once execution begins."""

    schema_name: Literal["llama-tuning-experiment"] = Field(
        default="llama-tuning-experiment",
        alias="schema",
    )
    version: Literal[1] = 1
    name: NonEmptyString
    base_candidate_id: NonEmptyString
    search_space_id: NonEmptyString
    workload_suite_id: NonEmptyString
    measurement_policy_id: NonEmptyString
    placement_policy: PlacementPolicy = Field(default_factory=PerCandidatePlacementPolicy)
    baseline: BaselinePolicy = Field(default_factory=BaseCandidateBaseline)
