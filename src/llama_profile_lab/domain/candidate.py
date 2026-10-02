"""Immutable production Candidate configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import (
    Field,
    NonNegativeInt,
    PositiveInt,
    field_serializer,
    field_validator,
    model_validator,
)

from llama_profile_lab.domain.base import ContentAddressedModel, FrozenModel, JsonScalar

NonEmptyString = Annotated[str, Field(min_length=1)]
ExtraArgValue: TypeAlias = JsonScalar | tuple[JsonScalar, ...]
GpuLayerSetting: TypeAlias = NonNegativeInt | Literal["auto", "all"] | None
DeviceSetting: TypeAlias = Literal["auto"] | tuple[NonEmptyString, ...]


class ModelSelection(FrozenModel):
    """Target and optional draft-model identities."""

    target_model_id: NonEmptyString
    draft_model_id: NonEmptyString | None = None


class ContextConfig(FrozenModel):
    """Context and KV-cache behavior."""

    size: PositiveInt
    cache_type_k: NonEmptyString
    cache_type_v: NonEmptyString
    kv_offload: bool = True
    kv_unified: bool = True


class ComputeConfig(FrozenModel):
    """Performance-relevant compute settings."""

    flash_attn: Literal["on", "off", "auto"] = "auto"
    batch_size: PositiveInt
    ubatch_size: PositiveInt
    threads: PositiveInt | None = None
    load_mode: NonEmptyString = "auto"
    lazy_mode: NonEmptyString = "auto"
    repack: bool = True
    no_host: bool = False
    no_op_offload: bool = False

    @model_validator(mode="after")
    def validate_batch_relationship(self) -> ComputeConfig:
        if self.ubatch_size > self.batch_size:
            raise ValueError("ubatch_size must be less than or equal to batch_size")
        return self


class FitConfig(FrozenModel):
    """Requested automatic-placement fit policy."""

    target_mib: NonNegativeInt
    min_context: PositiveInt


class PlacementConstraints(FrozenModel):
    """Placement constraints supplied before fit resolution."""

    n_gpu_layers: GpuLayerSetting = None
    n_cpu_moe: NonNegativeInt = 0
    split_mode: NonEmptyString = "layer"
    main_gpu: NonNegativeInt = 0
    devices: DeviceSetting = "auto"
    tensor_split: tuple[Annotated[float, Field(ge=0)], ...] | None = None
    override_tensor: tuple[NonEmptyString, ...] = ()


class PlacementConfig(FrozenModel):
    """Requested placement mode and constraints."""

    mode: Literal["fit", "fixed"]
    fit: FitConfig | None = None
    constraints: PlacementConstraints = Field(default_factory=PlacementConstraints)

    @model_validator(mode="after")
    def validate_mode(self) -> PlacementConfig:
        if self.mode == "fit" and self.fit is None:
            raise ValueError("fit placement requires a fit policy")
        if self.mode == "fixed" and self.fit is not None:
            raise ValueError("fixed placement cannot include a fit policy")
        return self


class ServerConfig(FrozenModel):
    """Server-relevant settings that affect deployment behavior."""

    parallel: PositiveInt = 1


class SpeculativeConfig(FrozenModel):
    """Speculative-decoding configuration."""

    enabled: bool = False
    type: NonEmptyString | None = None
    draft_n_max: PositiveInt | None = None

    @model_validator(mode="after")
    def validate_enabled_state(self) -> SpeculativeConfig:
        if self.enabled:
            if self.type is None:
                raise ValueError("enabled speculative decoding requires a type")
            if self.draft_n_max is None:
                raise ValueError("enabled speculative decoding requires draft_n_max")
        elif self.type is not None or self.draft_n_max is not None:
            raise ValueError(
                "disabled speculative decoding cannot define type or draft_n_max"
            )
        return self


class ExtraArgument(FrozenModel):
    """One otherwise-unmodeled llama.cpp argument."""

    name: NonEmptyString
    value: ExtraArgValue


class Candidate(ContentAddressedModel):
    """One immutable point in the tunable production-configuration space."""

    schema: Literal["llama-profile-candidate"] = "llama-profile-candidate"
    version: Literal[1] = 1
    model: ModelSelection
    context: ContextConfig
    compute: ComputeConfig
    placement: PlacementConfig
    server: ServerConfig = Field(default_factory=ServerConfig)
    speculative: SpeculativeConfig = Field(default_factory=SpeculativeConfig)
    extra_args: tuple[ExtraArgument, ...] = ()

    @field_validator("extra_args", mode="before")
    @classmethod
    def normalize_extra_args(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return tuple(
                {"name": name, "value": arg_value}
                for name, arg_value in sorted(value.items())
            )
        return value

    @field_validator("extra_args")
    @classmethod
    def validate_extra_args(
        cls,
        value: tuple[ExtraArgument, ...],
    ) -> tuple[ExtraArgument, ...]:
        names = [item.name for item in value]
        if len(names) != len(set(names)):
            raise ValueError("extra_args cannot contain duplicate argument names")
        return tuple(sorted(value, key=lambda item: item.name))

    @field_serializer("extra_args")
    def serialize_extra_args(
        self,
        value: tuple[ExtraArgument, ...],
    ) -> dict[str, ExtraArgValue]:
        return {item.name: item.value for item in value}
