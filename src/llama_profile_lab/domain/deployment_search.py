"""Deterministic deployment-wide search-space definitions."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from llama_profile_lab.domain.base import (
    ContentAddressedModel,
    FrozenModel,
    JsonScalar,
    canonical_json,
)

NonEmptyString = Annotated[str, Field(min_length=1)]
type DeploymentSearchValue = JsonScalar | tuple[JsonScalar, ...]


class DeploymentSearchDimension(FrozenModel):
    """One discrete deployment-wide search dimension."""

    path: NonEmptyString
    values: Annotated[tuple[DeploymentSearchValue, ...], Field(min_length=1)]
    condition: NonEmptyString | None = None

    @field_validator("values")
    @classmethod
    def validate_unique_values(
        cls,
        values: tuple[DeploymentSearchValue, ...],
    ) -> tuple[DeploymentSearchValue, ...]:
        keys = [canonical_json(value) for value in values]
        if len(keys) != len(set(keys)):
            raise ValueError("deployment search dimension values must be unique")
        return values


class DeploymentSearchConstraint(FrozenModel):
    """Restricted constraint evaluated against an expanded deployment payload."""

    expression: NonEmptyString


class DeploymentGridStrategy(FrozenModel):
    """Deterministic Cartesian-product deployment expansion."""

    type: Literal["grid"] = "grid"


class DeploymentSearchSpace(ContentAddressedModel):
    """Immutable search dimensions for one base DeploymentCandidate."""

    schema_name: Literal["llama-deployment-search-space"] = Field(
        default="llama-deployment-search-space",
        alias="schema",
    )
    version: Literal[1] = 1
    dimensions: Annotated[tuple[DeploymentSearchDimension, ...], Field(min_length=1)]
    constraints: tuple[DeploymentSearchConstraint, ...] = ()
    strategy: DeploymentGridStrategy = Field(default_factory=DeploymentGridStrategy)

    @field_validator("constraints", mode="before")
    @classmethod
    def normalize_constraints(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return tuple(
                {"expression": item} if isinstance(item, str) else item
                for item in value
            )
        return value

    @model_validator(mode="after")
    def validate_unique_paths(self) -> DeploymentSearchSpace:
        paths = [dimension.path for dimension in self.dimensions]
        if len(paths) != len(set(paths)):
            raise ValueError("deployment search dimension paths must be unique")
        return self
