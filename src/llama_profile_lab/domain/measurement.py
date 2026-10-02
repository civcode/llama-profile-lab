"""Measurement-policy domain models."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, PositiveInt, model_validator

from llama_profile_lab.domain.base import ContentAddressedModel, FrozenModel


class AdaptiveMeasurementPolicy(FrozenModel):
    """Optional adaptive-repetition policy."""

    minimum_repetitions: PositiveInt
    maximum_repetitions: PositiveInt
    target_relative_error: Annotated[float, Field(gt=0.0, le=1.0)]

    @model_validator(mode="after")
    def validate_bounds(self) -> AdaptiveMeasurementPolicy:
        if self.minimum_repetitions > self.maximum_repetitions:
            raise ValueError(
                "minimum_repetitions must be less than or equal to maximum_repetitions"
            )
        return self


class MeasurementPolicy(ContentAddressedModel):
    """How an already-defined workload is measured."""

    schema_name: Literal["llama-measurement-policy"] = Field(
        default="llama-measurement-policy",
        alias="schema",
    )
    version: Literal[1] = 1
    warmup: bool = True
    repetitions: PositiveInt | None = 3
    delay_seconds: Annotated[float, Field(ge=0.0)] = 0.0
    adaptive: AdaptiveMeasurementPolicy | None = None

    @model_validator(mode="after")
    def validate_repetition_mode(self) -> MeasurementPolicy:
        if self.adaptive is None and self.repetitions is None:
            raise ValueError("fixed measurement policy requires repetitions")
        if self.adaptive is not None and self.repetitions is not None:
            raise ValueError(
                "adaptive measurement policy cannot also define fixed repetitions"
            )
        return self
