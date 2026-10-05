"""Stable accelerator identity and structured memory-estimation values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, NonNegativeInt, field_validator, model_validator

from llama_profile_lab.domain.base import ContentAddressedModel, FrozenModel, JsonScalar

NonEmptyString = Annotated[str, Field(min_length=1)]
type DeviceMappingStatus = Literal["mapped", "unresolved"]


def physical_device_key(
    *,
    pci_bus_id: str | None = None,
    uuid: str | None = None,
) -> str | None:
    """Return the strongest durable physical-device key available."""
    if uuid:
        return f"uuid:{uuid.strip().lower()}"
    if pci_bus_id:
        return f"pci:{pci_bus_id.strip().lower()}"
    return None


class AcceleratorDevice(FrozenModel):
    """One llama.cpp logical accelerator and its physical mapping when known."""

    logical_device_name: NonEmptyString
    backend: NonEmptyString
    mapping_status: DeviceMappingStatus = "unresolved"
    physical_device_key: NonEmptyString | None = None
    pci_bus_id: NonEmptyString | None = None
    uuid: NonEmptyString | None = None
    vendor: NonEmptyString | None = None
    product_name: NonEmptyString | None = None
    total_memory_bytes: NonNegativeInt | None = None
    free_memory_bytes: NonNegativeInt | None = None
    driver: NonEmptyString | None = None
    runtime_metadata: Mapping[str, JsonScalar] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_mapping(self) -> AcceleratorDevice:
        if self.mapping_status == "mapped" and self.physical_device_key is None:
            raise ValueError("mapped accelerator requires physical_device_key")
        expected = physical_device_key(
            pci_bus_id=self.pci_bus_id,
            uuid=self.uuid,
        )
        if expected is not None and self.physical_device_key not in {None, expected}:
            raise ValueError("physical_device_key does not match UUID/PCI identity")
        if (
            self.total_memory_bytes is not None
            and self.free_memory_bytes is not None
            and self.free_memory_bytes > self.total_memory_bytes
        ):
            raise ValueError("free accelerator memory cannot exceed total memory")
        return self


class MemoryEstimateIdentity(ContentAddressedModel):
    """All inputs that make a per-Candidate memory estimate reusable."""

    schema_name: Literal["llama-memory-estimate-identity"] = Field(
        default="llama-memory-estimate-identity",
        alias="schema",
    )
    version: Literal[1] = 1
    host_id: NonEmptyString
    candidate_hash: NonEmptyString
    model_artifact_id: NonEmptyString
    helper_sha256: NonEmptyString
    selected_devices: tuple[NonEmptyString, ...] = ()
    n_gpu_layers: NonNegativeInt | Literal["auto", "all"] | None = None
    split_mode: NonEmptyString = "layer"
    main_gpu: NonNegativeInt = 0
    tensor_split: tuple[Annotated[float, Field(ge=0.0)], ...] | None = None
    override_tensor: tuple[NonEmptyString, ...] = ()

    @field_validator("selected_devices")
    @classmethod
    def validate_devices(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("selected_devices must be unique")
        return value

    @model_validator(mode="after")
    def validate_tensor_split(self) -> MemoryEstimateIdentity:
        if (
            self.tensor_split is not None
            and self.selected_devices
            and len(self.tensor_split) != len(self.selected_devices)
        ):
            raise ValueError(
                "tensor_split must contain one value per selected device"
            )
        return self


class MemoryEstimateDevice(FrozenModel):
    """Normalized helper memory result for one selected logical device."""

    logical_device_name: NonEmptyString
    model_bytes: NonNegativeInt
    context_bytes: NonNegativeInt
    compute_bytes: NonNegativeInt
    total_bytes: NonNegativeInt
    device_total_bytes: NonNegativeInt
    device_free_bytes: NonNegativeInt

    @model_validator(mode="after")
    def validate_totals(self) -> MemoryEstimateDevice:
        expected = self.model_bytes + self.context_bytes + self.compute_bytes
        if self.total_bytes != expected:
            raise ValueError("total_bytes must equal model + context + compute bytes")
        if self.device_free_bytes > self.device_total_bytes:
            raise ValueError("device_free_bytes cannot exceed device_total_bytes")
        return self


class MemoryEstimateResolved(FrozenModel):
    """Placement values resolved or confirmed by the estimator helper."""

    n_gpu_layers: NonNegativeInt
    devices: tuple[NonEmptyString, ...]
    split_mode: NonEmptyString = "layer"
    main_gpu: NonNegativeInt = 0
    tensor_split: tuple[Annotated[float, Field(ge=0.0)], ...] | None = None
    override_tensor: tuple[NonEmptyString, ...] = ()

    @field_validator("devices")
    @classmethod
    def validate_devices(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("resolved devices must be unique")
        return value

    @model_validator(mode="after")
    def validate_tensor_split(self) -> MemoryEstimateResolved:
        if self.tensor_split is not None and len(self.tensor_split) != len(self.devices):
            raise ValueError(
                "resolved tensor_split must contain one value per device"
            )
        return self


class MemoryEstimateOutput(FrozenModel):
    """Versioned structured output emitted by the registered helper."""

    schema_name: Literal["llama-memory-estimate"] = Field(
        default="llama-memory-estimate",
        alias="schema",
    )
    version: Literal[1] = 1
    devices: Annotated[tuple[MemoryEstimateDevice, ...], Field(min_length=1)]
    resolved: MemoryEstimateResolved
    metadata: Mapping[str, JsonScalar] = Field(default_factory=dict)

    @field_validator("devices")
    @classmethod
    def validate_device_rows(
        cls,
        value: tuple[MemoryEstimateDevice, ...],
    ) -> tuple[MemoryEstimateDevice, ...]:
        names = [item.logical_device_name for item in value]
        if len(names) != len(set(names)):
            raise ValueError("memory estimate devices must be unique")
        return value

    @model_validator(mode="after")
    def validate_selected_devices(self) -> MemoryEstimateOutput:
        rows = {item.logical_device_name for item in self.devices}
        if set(self.resolved.devices) != rows:
            raise ValueError(
                "memory estimate rows must match the resolved selected devices"
            )
        return self
