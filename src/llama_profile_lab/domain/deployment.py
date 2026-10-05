"""Immutable multi-model deployment definitions and placement values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, ClassVar, Literal

from pydantic import (
    Field,
    NonNegativeInt,
    field_serializer,
    field_validator,
    model_validator,
)

from llama_profile_lab.domain.base import (
    ContentAddressedModel,
    FrozenModel,
    JsonScalar,
)

NonEmptyString = Annotated[str, Field(min_length=1)]
type DeploymentPhase = Literal["dd", "pp", "pd", "dp"]
type DeploymentRunStatus = Literal[
    "planned",
    "starting",
    "ready",
    "running",
    "completed",
    "cancelled",
    "failed",
]
type DeploymentFailureKind = Literal[
    "device_capability_mismatch",
    "memory_projection_failed",
    "memory_infeasible",
    "server_start_failed",
    "server_oom",
    "concurrent_workload_failed",
    "member_timeout",
    "member_crash",
    "telemetry_incomplete",
    "runtime_memory_margin_violated",
    "output_validation_failed",
]
type GpuLayerSetting = NonNegativeInt | Literal["auto", "all"] | None


class DeviceMemoryMargin(FrozenModel):
    """Reserved memory to keep free on one physical accelerator."""

    device_id: NonEmptyString
    margin_bytes: NonNegativeInt = 0


class BackendPair(FrozenModel):
    """One explicitly permitted unordered pair of accelerator backends."""

    left: NonEmptyString
    right: NonEmptyString


class HostResourcePolicy(FrozenModel):
    """Deployment-wide host resource constraints."""

    device_memory_margin_bytes: tuple[DeviceMemoryMargin, ...] = ()
    host_ram_margin_bytes: NonNegativeInt = 0
    allow_cpu_offload: bool = False
    allow_swap: bool = False
    allowed_devices: tuple[NonEmptyString, ...] = ()
    allowed_backend_pairs: tuple[BackendPair, ...] = ()
    maximum_total_power_w: Annotated[float, Field(gt=0.0)] | None = None

    @field_validator("device_memory_margin_bytes", mode="before")
    @classmethod
    def normalize_margins(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return tuple(
                {"device_id": device_id, "margin_bytes": margin}
                for device_id, margin in sorted(value.items())
            )
        return value

    @field_validator("device_memory_margin_bytes")
    @classmethod
    def validate_margins(
        cls,
        value: tuple[DeviceMemoryMargin, ...],
    ) -> tuple[DeviceMemoryMargin, ...]:
        device_ids = [item.device_id for item in value]
        if len(device_ids) != len(set(device_ids)):
            raise ValueError("device memory margins must have unique device IDs")
        return tuple(sorted(value, key=lambda item: item.device_id))

    @field_serializer("device_memory_margin_bytes")
    def serialize_margins(
        self,
        value: tuple[DeviceMemoryMargin, ...],
    ) -> dict[str, int]:
        return {item.device_id: item.margin_bytes for item in value}

    @field_validator("allowed_devices")
    @classmethod
    def normalize_allowed_devices(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("allowed_devices must be unique")
        return tuple(sorted(value))

    @field_validator("allowed_backend_pairs")
    @classmethod
    def normalize_backend_pairs(
        cls,
        value: tuple[BackendPair, ...],
    ) -> tuple[BackendPair, ...]:
        normalized = tuple(
            BackendPair(
                left=min(item.left, item.right),
                right=max(item.left, item.right),
            )
            for item in value
        )
        keys = [(item.left, item.right) for item in normalized]
        if len(keys) != len(set(keys)):
            raise ValueError("allowed_backend_pairs must be unique")
        return tuple(sorted(normalized, key=lambda item: (item.left, item.right)))


class DeploymentPlacementRequest(FrozenModel):
    """Deployment-level placement override for one model instance."""

    devices: tuple[NonEmptyString, ...] | None = None
    n_gpu_layers: GpuLayerSetting = None
    split_mode: NonEmptyString | None = None
    main_gpu: NonNegativeInt | None = None
    tensor_split: tuple[Annotated[float, Field(ge=0.0)], ...] | None = None
    override_tensor: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def validate_request(self) -> DeploymentPlacementRequest:
        if self.devices is not None and len(self.devices) == 0:
            raise ValueError("placement devices cannot be empty")
        if self.devices is not None and len(self.devices) != len(set(self.devices)):
            raise ValueError("placement devices must be unique")
        if (
            self.devices is not None
            and self.tensor_split is not None
            and len(self.devices) != len(self.tensor_split)
        ):
            raise ValueError("tensor_split must contain one value per selected device")
        return self


class DeploymentWorkloadMix(FrozenModel):
    """Reference to the base workload suite and concurrent phases to evaluate."""

    workload_suite_id: NonEmptyString
    phases: Annotated[tuple[DeploymentPhase, ...], Field(min_length=1)] = (
        "dd",
        "pp",
        "pd",
        "dp",
    )

    @field_validator("phases")
    @classmethod
    def validate_phases(
        cls,
        value: tuple[DeploymentPhase, ...],
    ) -> tuple[DeploymentPhase, ...]:
        if len(value) != len(set(value)):
            raise ValueError("deployment workload phases must be unique")
        order = {"dd": 0, "pp": 1, "pd": 2, "dp": 3}
        return tuple(sorted(value, key=order.__getitem__))


class ModelInstanceCandidate(FrozenModel):
    """One named model/server instance participating in a deployment."""

    instance_id: NonEmptyString
    candidate_id: NonEmptyString
    role: NonEmptyString
    model_artifact_id: NonEmptyString
    binary_id: NonEmptyString
    requested_placement: DeploymentPlacementRequest = Field(
        default_factory=DeploymentPlacementRequest
    )
    server_identity: NonEmptyString


class DeploymentCandidate(ContentAddressedModel):
    """One immutable joint multi-model deployment optimization point."""

    schema_name: Literal["llama-profile-deployment-candidate"] = Field(
        default="llama-profile-deployment-candidate",
        alias="schema",
    )
    version: Literal[1] = 1
    instances: Annotated[tuple[ModelInstanceCandidate, ...], Field(min_length=2)]
    resource_policy: HostResourcePolicy = Field(default_factory=HostResourcePolicy)
    workload_mix: DeploymentWorkloadMix

    @field_validator("instances")
    @classmethod
    def normalize_instances(
        cls,
        value: tuple[ModelInstanceCandidate, ...],
    ) -> tuple[ModelInstanceCandidate, ...]:
        ids = [item.instance_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("deployment instance IDs must be unique")
        return tuple(sorted(value, key=lambda item: item.instance_id))


class DeploymentInstancePlacement(FrozenModel):
    """Resolved V1 placement associated with one deployment instance."""

    instance_id: NonEmptyString
    resolved_placement_id: NonEmptyString


class PlacementDeviceMemory(FrozenModel):
    """Normalized memory estimate for one instance on one physical device."""

    instance_id: NonEmptyString
    device_id: NonEmptyString
    model_bytes: NonNegativeInt
    context_bytes: NonNegativeInt
    compute_bytes: NonNegativeInt
    total_bytes: NonNegativeInt
    device_total_bytes: NonNegativeInt
    device_free_bytes: NonNegativeInt
    source: NonEmptyString
    measured_at: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_totals(self) -> PlacementDeviceMemory:
        expected = self.model_bytes + self.context_bytes + self.compute_bytes
        if self.total_bytes != expected:
            raise ValueError("total_bytes must equal model + context + compute bytes")
        if self.device_free_bytes > self.device_total_bytes:
            raise ValueError("device_free_bytes cannot exceed device_total_bytes")
        return self


class DeploymentDeviceAllocation(FrozenModel):
    """Projected aggregate allocation and headroom on one physical device."""

    device_id: NonEmptyString
    projected_bytes: NonNegativeInt
    reserved_margin_bytes: NonNegativeInt = 0
    device_total_bytes: NonNegativeInt
    projected_free_bytes: NonNegativeInt

    @model_validator(mode="after")
    def validate_projection(self) -> DeploymentDeviceAllocation:
        required = self.projected_bytes + self.reserved_margin_bytes
        if required > self.device_total_bytes:
            raise ValueError("projected allocation and margin exceed device total memory")
        expected_free = self.device_total_bytes - required
        if self.projected_free_bytes != expected_free:
            raise ValueError(
                "projected_free_bytes does not match allocation and margin"
            )
        return self


type DeploymentFeasibility = Literal[
    "pending",
    "feasible",
    "infeasible",
    "estimate_failed",
]


class DeploymentPlacement(ContentAddressedModel):
    """Concrete joint placement and memory projection for one deployment Candidate."""

    identity_exclude: ClassVar[frozenset[str]] = frozenset(
        {"provenance", "measured_at"}
    )

    schema_name: Literal["llama-profile-deployment-placement"] = Field(
        default="llama-profile-deployment-placement",
        alias="schema",
    )
    version: Literal[1] = 1
    deployment_candidate_id: NonEmptyString
    host_id: NonEmptyString
    instance_placements: tuple[DeploymentInstancePlacement, ...]
    device_memory: tuple[PlacementDeviceMemory, ...] = ()
    device_allocations: tuple[DeploymentDeviceAllocation, ...] = ()
    feasibility: DeploymentFeasibility = "pending"
    provenance: Mapping[str, JsonScalar] = Field(default_factory=dict)

    @field_validator("instance_placements")
    @classmethod
    def normalize_instance_placements(
        cls,
        value: tuple[DeploymentInstancePlacement, ...],
    ) -> tuple[DeploymentInstancePlacement, ...]:
        ids = [item.instance_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("deployment placement instance IDs must be unique")
        return tuple(sorted(value, key=lambda item: item.instance_id))

    @field_validator("device_memory")
    @classmethod
    def normalize_device_memory(
        cls,
        value: tuple[PlacementDeviceMemory, ...],
    ) -> tuple[PlacementDeviceMemory, ...]:
        keys = [(item.instance_id, item.device_id) for item in value]
        if len(keys) != len(set(keys)):
            raise ValueError(
                "device memory records must be unique per instance/device"
            )
        return tuple(
            sorted(value, key=lambda item: (item.instance_id, item.device_id))
        )

    @field_validator("device_allocations")
    @classmethod
    def normalize_device_allocations(
        cls,
        value: tuple[DeploymentDeviceAllocation, ...],
    ) -> tuple[DeploymentDeviceAllocation, ...]:
        ids = [item.device_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("device allocations must have unique device IDs")
        return tuple(sorted(value, key=lambda item: item.device_id))
