"""N-dimensional search-space definitions."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, field_serializer, field_validator, model_validator

from llama_profile_lab.domain.base import (
    ContentAddressedModel,
    FrozenModel,
    JsonScalar,
    canonical_json,
)

NonEmptyString = Annotated[str, Field(min_length=1)]


class SearchDimension(FrozenModel):
    """One discrete tunable dimension."""

    path: NonEmptyString
    values: Annotated[tuple[JsonScalar, ...], Field(min_length=1)]
    condition: NonEmptyString | None = None

    @field_validator("values")
    @classmethod
    def validate_unique_values(
        cls,
        values: tuple[JsonScalar, ...],
    ) -> tuple[JsonScalar, ...]:
        keys = [canonical_json(value) for value in values]
        if len(keys) != len(set(keys)):
            raise ValueError("search dimension values must be unique")
        return values


class SearchConstraint(FrozenModel):
    """A restricted constraint expression interpreted by the planner."""

    expression: NonEmptyString


class GridStrategy(FrozenModel):
    """Deterministic Cartesian-product expansion."""

    type: Literal["grid"] = "grid"


class SearchSpace(ContentAddressedModel):
    """An immutable N-dimensional candidate search space."""

    schema: Literal["llama-search-space"] = "llama-search-space"
    version: Literal[1] = 1
    dimensions: Annotated[tuple[SearchDimension, ...], Field(min_length=1)]
    constraints: tuple[SearchConstraint, ...] = ()
    strategy: GridStrategy = Field(default_factory=GridStrategy)

    @field_validator("constraints", mode="before")
    @classmethod
    def normalize_constraints(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return tuple(
                {"expression": item} if isinstance(item, str) else item
                for item in value
            )
        return value

    @field_serializer("constraints")
    def serialize_constraints(
        self,
        value: tuple[SearchConstraint, ...],
    ) -> list[str]:
        return [constraint.expression for constraint in value]

    @model_validator(mode="after")
    def validate_unique_paths(self) -> SearchSpace:
        paths = [dimension.path for dimension in self.dimensions]
        if len(paths) != len(set(paths)):
            raise ValueError("search dimension paths must be unique")
        return self
