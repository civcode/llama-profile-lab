"""Joint deployment planning, capability pruning, and memory feasibility."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal, Protocol

from llama_profile_lab.db import (
    AcceleratorDeviceRepository,
    CandidateRepository,
    Database,
    DeploymentCandidateRepository,
    DeploymentPlanRepository,
    DeploymentPlacementRepository,
    EnvironmentRepository,
    PlacementRepository,
)
from llama_profile_lab.db.records import AcceleratorDeviceRecord, BinaryRecord
from llama_profile_lab.domain import (
    Candidate,
    DeploymentCandidate,
    DeploymentDeviceAllocation,
    DeploymentInstancePlacement,
    DeploymentPlacement,
    DeploymentPlacementRequest,
    DeploymentSearchSpace,
    HostResourcePolicy,
    PlacementConstraints,
    PlacementDeviceMemory,
    ResolvedPlacement,
    sha256_json,
)
from llama_profile_lab.execution.host import BasicHostInfo, detect_basic_host
from llama_profile_lab.execution.memory_estimator import (
    MemoryEstimateObservation,
    MemoryEstimatorError,
    MemoryEstimatorService,
)
from llama_profile_lab.llama import CapabilitySet
from llama_profile_lab.planning.deployment_search import (
    DeploymentPoint,
    expand_deployment_search,
)

DeploymentRejectionReason = Literal[
    "unsupported_device",
    "unsupported_backend_pair",
    "unsupported_split_mode",
    "invalid_kv_configuration",
    "memory_estimate_failed",
    "device_memory_exceeded",
    "host_memory_policy_violation",
    "candidate_constraint_failed",
]


@dataclass(frozen=True, slots=True)
class DeploymentEstimatorInput:
    """Estimator binary and model path for one deployment instance."""

    instance_id: str
    helper_binary_id: str
    model_path: Path


@dataclass(frozen=True, slots=True)
class DeploymentPlanCase:
    """One feasible deployment point produced by the planner."""

    deployment_candidate_id: str
    deployment_placement_id: str
    generation: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class DeploymentPlanSummary:
    """Stable counts and persisted-case references from one planner pass."""

    base_deployment_candidate_id: str
    host_id: str
    raw_combinations: int
    rejected_by_constraints: int
    duplicate_candidates: int
    symmetry_reduced: int
    capability_rejected: int
    estimate_failed: int
    memory_rejected: int
    valid_count: int
    plan_id: str | None = None
    cases: tuple[DeploymentPlanCase, ...] = ()


class MemoryEstimateProvider(Protocol):
    """Interface consumed by the joint planner."""

    def estimate(
        self,
        candidate_id: str,
        *,
        helper_binary_id: str,
        model_path: Path,
        selected_devices: tuple[str, ...] | None = None,
        placement_constraints: PlacementConstraints | None = None,
        timeout_seconds: float | None = 300.0,
    ) -> MemoryEstimateObservation:
        """Return one structured estimate for explicit placement constraints."""


@dataclass(frozen=True, slots=True)
class _DeviceContext:
    instance_id: str
    binary_id: str
    logical_device_name: str
    device_id: str
    backend: str


@dataclass(frozen=True, slots=True)
class _CapabilityResult:
    constraints: dict[str, PlacementConstraints]
    devices: tuple[_DeviceContext, ...]


@dataclass(frozen=True, slots=True)
class _Rejection:
    reason: DeploymentRejectionReason
    details: dict[str, object]


class DeploymentPlannerService:
    """Plan multi-model placements without launching throughput workloads."""

    def __init__(
        self,
        database: Database,
        *,
        memory_estimator: MemoryEstimateProvider | None = None,
        host_detector: Callable[[], BasicHostInfo] = detect_basic_host,
    ) -> None:
        self.database = database
        self.memory_estimator = memory_estimator or MemoryEstimatorService(database)
        self.host_detector = host_detector

    def preview(
        self,
        base_deployment_candidate_id: str,
        search_space: DeploymentSearchSpace,
        estimator_inputs: tuple[DeploymentEstimatorInput, ...],
        *,
        timeout_seconds: float | None = 300.0,
    ) -> DeploymentPlanSummary:
        """Evaluate counts and feasibility without persisting plan cases."""
        return self._run(
            base_deployment_candidate_id,
            search_space,
            estimator_inputs,
            timeout_seconds=timeout_seconds,
            persist=False,
        )

    def plan(
        self,
        base_deployment_candidate_id: str,
        search_space: DeploymentSearchSpace,
        estimator_inputs: tuple[DeploymentEstimatorInput, ...],
        *,
        timeout_seconds: float | None = 300.0,
    ) -> DeploymentPlanSummary:
        """Persist feasible cases and explainable rejected deployment points."""
        return self._run(
            base_deployment_candidate_id,
            search_space,
            estimator_inputs,
            timeout_seconds=timeout_seconds,
            persist=True,
        )

    def _run(
        self,
        base_deployment_candidate_id: str,
        search_space: DeploymentSearchSpace,
        estimator_inputs: tuple[DeploymentEstimatorInput, ...],
        *,
        timeout_seconds: float | None,
        persist: bool,
    ) -> DeploymentPlanSummary:
        host = self.host_detector()
        input_map = _input_map(estimator_inputs)
        with self.database.session() as connection:
            deployments = DeploymentCandidateRepository(connection)
            base = deployments.get(base_deployment_candidate_id)
            if base is None:
                raise ValueError(
                    "deployment Candidate not found: "
                    f"{base_deployment_candidate_id}"
                )
            candidates = CandidateRepository(connection)
            candidate_map: dict[str, Candidate] = {}
            for instance in base.instances:
                candidate = candidates.get(instance.candidate_id)
                if candidate is None:
                    raise ValueError(
                        f"Candidate not found for instance {instance.instance_id}: "
                        f"{instance.candidate_id}"
                    )
                candidate_map[instance.instance_id] = candidate
            if set(input_map) != set(candidate_map):
                raise ValueError(
                    "estimator inputs must contain exactly the deployment instances"
                )
            host_id = EnvironmentRepository(connection).put_host(
                hostname=host.hostname,
                hardware_fingerprint=host.hardware_fingerprint,
                cpu=host.cpu,
                ram_bytes=host.ram_bytes,
                gpus=host.gpus,
                os_info=host.os_info,
            )

        expansion = expand_deployment_search(base, candidate_map, search_space)
        capability_rejected = 0
        estimate_failed = 0
        memory_rejected = 0
        valid_cases: list[DeploymentPlanCase] = []
        persisted_case_data: list[tuple[str, str, Mapping[str, object]]] = []

        for point in expansion.points:
            self._persist_point_candidates(point)
            capability = self._capability_check(
                point,
                host_id=host_id,
                estimator_inputs=input_map,
                host=host,
            )
            if isinstance(capability, _Rejection):
                capability_rejected += 1
                if persist:
                    self._persist_rejection(
                        point.deployment,
                        capability,
                        "capability",
                    )
                continue

            estimates: dict[str, MemoryEstimateObservation] = {}
            estimation_rejection: _Rejection | None = None
            for instance in point.deployment.instances:
                constraints = capability.constraints[instance.instance_id]
                config = input_map[instance.instance_id]
                try:
                    estimates[instance.instance_id] = self.memory_estimator.estimate(
                        instance.candidate_id,
                        helper_binary_id=config.helper_binary_id,
                        model_path=config.model_path,
                        selected_devices=tuple(constraints.devices),
                        placement_constraints=constraints,
                        timeout_seconds=timeout_seconds,
                    )
                except (MemoryEstimatorError, ValueError) as exc:
                    estimation_rejection = _Rejection(
                        "memory_estimate_failed",
                        {
                            "instance_id": instance.instance_id,
                            "error": str(exc),
                        },
                    )
                    break

            if estimation_rejection is not None:
                estimate_failed += 1
                if persist:
                    self._persist_rejection(
                        point.deployment,
                        estimation_rejection,
                        "estimate",
                    )
                continue

            feasibility = _memory_feasibility(
                point.deployment.resource_policy,
                capability.devices,
                estimates,
            )
            if isinstance(feasibility, _Rejection):
                memory_rejected += 1
                if persist:
                    self._persist_rejection(
                        point.deployment,
                        feasibility,
                        "memory",
                    )
                continue

            if persist:
                deployment_id, placement_id = self._persist_feasible(
                    point,
                    host_id=host_id,
                    constraints=capability.constraints,
                    devices=capability.devices,
                    estimates=estimates,
                    allocations=feasibility,
                    estimator_inputs=input_map,
                )
                generation = point.generation_metadata()
                persisted_case_data.append(
                    (deployment_id, placement_id, generation)
                )
                valid_cases.append(
                    DeploymentPlanCase(
                        deployment_candidate_id=deployment_id,
                        deployment_placement_id=placement_id,
                        generation=generation,
                    )
                )

        valid_count = (
            len(valid_cases)
            if persist
            else len(expansion.points)
            - capability_rejected
            - estimate_failed
            - memory_rejected
        )
        plan_id: str | None = None
        if persist:
            with self.database.session() as connection:
                plans = DeploymentPlanRepository(connection)
                plan_id = plans.create(
                    base_deployment_candidate_id=base_deployment_candidate_id,
                    host_id=host_id,
                    search_space=search_space,
                    request={
                        "estimator_instances": ",".join(sorted(input_map)),
                        "timeout_seconds": timeout_seconds,
                    },
                    raw_combinations=expansion.raw_combinations,
                    rejected_by_constraints=expansion.rejected_by_constraints,
                    duplicate_candidates=expansion.duplicate_candidates,
                    symmetry_reduced=expansion.symmetry_reduced,
                    capability_rejected=capability_rejected,
                    estimate_failed=estimate_failed,
                    memory_rejected=memory_rejected,
                    valid_count=valid_count,
                )
                for ordinal, item in enumerate(persisted_case_data):
                    deployment_id, placement_id, generation = item
                    plans.add_case(
                        plan_id,
                        ordinal=ordinal,
                        deployment_candidate_id=deployment_id,
                        deployment_placement_id=placement_id,
                        generation=generation,
                    )

        return DeploymentPlanSummary(
            base_deployment_candidate_id=base_deployment_candidate_id,
            host_id=host_id,
            raw_combinations=expansion.raw_combinations,
            rejected_by_constraints=expansion.rejected_by_constraints,
            duplicate_candidates=expansion.duplicate_candidates,
            symmetry_reduced=expansion.symmetry_reduced,
            capability_rejected=capability_rejected,
            estimate_failed=estimate_failed,
            memory_rejected=memory_rejected,
            valid_count=valid_count,
            plan_id=plan_id,
            cases=tuple(valid_cases),
        )

    def _persist_point_candidates(self, point: DeploymentPoint) -> None:
        with self.database.session() as connection:
            candidates = CandidateRepository(connection)
            for _, candidate in point.candidates:
                candidates.put(candidate)

    def _persist_rejection(
        self,
        deployment: DeploymentCandidate,
        rejection: _Rejection,
        stage: str,
    ) -> None:
        with self.database.session() as connection:
            deployments = DeploymentCandidateRepository(connection)
            deployment_id = deployments.put(deployment)
            deployments.add_rejection(
                deployment_id,
                stage=stage,
                reason=rejection.reason,
                details=rejection.details,
            )

    def _capability_check(
        self,
        point: DeploymentPoint,
        *,
        host_id: str,
        estimator_inputs: Mapping[str, DeploymentEstimatorInput],
        host: BasicHostInfo,
    ) -> _CapabilityResult | _Rejection:
        policy = point.deployment.resource_policy
        if policy.host_ram_margin_bytes >= host.ram_bytes:
            return _Rejection(
                "host_memory_policy_violation",
                {
                    "host_ram_bytes": host.ram_bytes,
                    "host_ram_margin_bytes": policy.host_ram_margin_bytes,
                },
            )

        constraints_by_instance: dict[str, PlacementConstraints] = {}
        selected: list[_DeviceContext] = []
        with self.database.session() as connection:
            environment = EnvironmentRepository(connection)
            inventory = AcceleratorDeviceRepository(connection)
            for instance in point.deployment.instances:
                candidate = point.candidate_for(instance.instance_id)
                constraints = effective_placement_constraints(
                    candidate,
                    instance.requested_placement,
                )
                constraints_by_instance[instance.instance_id] = constraints

                rejection = _validate_constraints(
                    instance.instance_id,
                    candidate,
                    constraints,
                )
                if rejection is not None:
                    return rejection

                server_binary = environment.get_binary(instance.binary_id)
                if server_binary is None:
                    return _Rejection(
                        "unsupported_device",
                        {
                            "instance_id": instance.instance_id,
                            "binary_id": instance.binary_id,
                            "error": "server binary not found",
                        },
                    )
                binary_rejection = _check_binary_options(
                    server_binary,
                    constraints,
                    instance.instance_id,
                )
                if binary_rejection is not None:
                    return binary_rejection

                rows = inventory.list_for_binary(
                    host_id=host_id,
                    binary_id=instance.binary_id,
                )
                by_name = {row.logical_device_name: row for row in rows}
                helper_id = estimator_inputs[
                    instance.instance_id
                ].helper_binary_id
                if not by_name:
                    helper_rows = inventory.list_for_binary(
                        host_id=host_id,
                        binary_id=helper_id,
                    )
                    by_name = {
                        row.logical_device_name: row
                        for row in helper_rows
                    }

                for logical_name in constraints.devices:
                    row = by_name.get(logical_name)
                    if row is None:
                        return _Rejection(
                            "unsupported_device",
                            {
                                "instance_id": instance.instance_id,
                                "logical_device_name": logical_name,
                                "binary_id": instance.binary_id,
                            },
                        )
                    device_id = _physical_device_id(
                        policy,
                        instance.binary_id,
                        row,
                    )
                    if device_id is None:
                        device_id = _physical_device_id(
                            policy,
                            helper_id,
                            row,
                        )
                    if device_id is None:
                        return _Rejection(
                            "unsupported_device",
                            {
                                "instance_id": instance.instance_id,
                                "logical_device_name": logical_name,
                                "error": "physical device mapping is unresolved",
                            },
                        )
                    if (
                        policy.allowed_devices
                        and logical_name not in policy.allowed_devices
                        and device_id not in policy.allowed_devices
                    ):
                        return _Rejection(
                            "unsupported_device",
                            {
                                "instance_id": instance.instance_id,
                                "logical_device_name": logical_name,
                                "device_id": device_id,
                                "error": "device is outside allowed_devices",
                            },
                        )
                    selected.append(
                        _DeviceContext(
                            instance_id=instance.instance_id,
                            binary_id=instance.binary_id,
                            logical_device_name=logical_name,
                            device_id=device_id,
                            backend=row.backend,
                        )
                    )

        backend_rejection = _check_backend_pairs(policy, tuple(selected))
        if backend_rejection is not None:
            return backend_rejection
        return _CapabilityResult(
            constraints=constraints_by_instance,
            devices=tuple(selected),
        )

    def _persist_feasible(
        self,
        point: DeploymentPoint,
        *,
        host_id: str,
        constraints: Mapping[str, PlacementConstraints],
        devices: tuple[_DeviceContext, ...],
        estimates: Mapping[str, MemoryEstimateObservation],
        allocations: tuple[DeploymentDeviceAllocation, ...],
        estimator_inputs: Mapping[str, DeploymentEstimatorInput],
    ) -> tuple[str, str]:
        device_lookup = {
            (item.instance_id, item.logical_device_name): item
            for item in devices
        }
        with self.database.session() as connection:
            candidates = CandidateRepository(connection)
            deployments = DeploymentCandidateRepository(connection)
            placements = PlacementRepository(connection)
            for _, candidate in point.candidates:
                candidates.put(candidate)
            deployment_id = deployments.put(point.deployment)

            instance_placements: list[DeploymentInstancePlacement] = []
            memory_rows: list[PlacementDeviceMemory] = []
            for instance in point.deployment.instances:
                observation = estimates[instance.instance_id]
                candidate = point.candidate_for(instance.instance_id)
                effective = constraints[instance.instance_id]
                resolved = observation.output.resolved
                placement = ResolvedPlacement(
                    production_context_size=candidate.context.size,
                    n_gpu_layers=resolved.n_gpu_layers,
                    n_cpu_moe=effective.n_cpu_moe,
                    split_mode=resolved.split_mode,
                    main_gpu=resolved.main_gpu,
                    devices=resolved.devices,
                    tensor_split=resolved.tensor_split,
                    override_tensor=resolved.override_tensor,
                )
                helper_id = estimator_inputs[
                    instance.instance_id
                ].helper_binary_id
                placement_hash = sha256_json(
                    {
                        "schema": "deployment-estimated-placement",
                        "version": 1,
                        "candidate_id": instance.candidate_id,
                        "host_id": host_id,
                        "helper_binary_id": helper_id,
                        "memory_estimate_id": observation.estimate_id,
                        "resolved": placement.model_dump(mode="json"),
                    }
                )
                resolved_id = placements.put_resolved(
                    placement_hash=placement_hash,
                    candidate_id=instance.candidate_id,
                    host_id=host_id,
                    binary_id=helper_id,
                    fit_attempt_id=None,
                    placement=placement,
                    request={
                        "source": "deployment-memory-estimator",
                        "memory_estimate_id": observation.estimate_id,
                    },
                    argv=(),
                    stdout="",
                    stderr="",
                    exit_code=0,
                    raw_result={
                        "memory_estimate_id": observation.estimate_id,
                        "cache_hit": observation.cache_hit,
                    },
                )
                instance_placements.append(
                    DeploymentInstancePlacement(
                        instance_id=instance.instance_id,
                        resolved_placement_id=resolved_id,
                    )
                )
                for memory in observation.output.devices:
                    context = device_lookup[
                        (
                            instance.instance_id,
                            memory.logical_device_name,
                        )
                    ]
                    memory_rows.append(
                        PlacementDeviceMemory(
                            instance_id=instance.instance_id,
                            device_id=context.device_id,
                            model_bytes=memory.model_bytes,
                            context_bytes=memory.context_bytes,
                            compute_bytes=memory.compute_bytes,
                            total_bytes=memory.total_bytes,
                            device_total_bytes=memory.device_total_bytes,
                            device_free_bytes=memory.device_free_bytes,
                            source=(
                                "memory-estimate:"
                                f"{observation.estimate_id}"
                            ),
                        )
                    )

            deployment_placement = DeploymentPlacement(
                deployment_candidate_id=deployment_id,
                host_id=host_id,
                instance_placements=tuple(instance_placements),
                device_memory=tuple(memory_rows),
                device_allocations=allocations,
                feasibility="feasible",
                provenance={
                    "planner": "v2-m4",
                    "estimate_count": len(estimates),
                },
            )
            placement_id = DeploymentPlacementRepository(connection).put(
                deployment_placement,
                request=point.generation_metadata(),
            )
        return deployment_id, placement_id


def effective_placement_constraints(
    candidate: Candidate,
    request: DeploymentPlacementRequest,
) -> PlacementConstraints:
    """Overlay deployment placement fields onto V1 Candidate constraints."""
    base = candidate.placement.constraints
    return base.model_copy(
        update={
            "devices": (
                request.devices
                if request.devices is not None
                else base.devices
            ),
            "n_gpu_layers": (
                request.n_gpu_layers
                if request.n_gpu_layers is not None
                else base.n_gpu_layers
            ),
            "split_mode": request.split_mode or base.split_mode,
            "main_gpu": (
                request.main_gpu
                if request.main_gpu is not None
                else base.main_gpu
            ),
            "tensor_split": (
                request.tensor_split
                if request.tensor_split is not None
                else base.tensor_split
            ),
            "override_tensor": (
                request.override_tensor
                if request.override_tensor
                else base.override_tensor
            ),
        }
    )


def _input_map(
    values: tuple[DeploymentEstimatorInput, ...],
) -> dict[str, DeploymentEstimatorInput]:
    result = {item.instance_id: item for item in values}
    if len(result) != len(values):
        raise ValueError(
            "deployment estimator inputs must have unique instance IDs"
        )
    return result


def _validate_constraints(
    instance_id: str,
    candidate: Candidate,
    constraints: PlacementConstraints,
) -> _Rejection | None:
    if constraints.devices == "auto":
        return _Rejection(
            "unsupported_device",
            {
                "instance_id": instance_id,
                "error": "joint planning requires explicit selected devices",
            },
        )
    if constraints.split_mode not in {"none", "layer", "row", "tensor"}:
        return _Rejection(
            "unsupported_split_mode",
            {
                "instance_id": instance_id,
                "split_mode": constraints.split_mode,
            },
        )
    if (
        constraints.tensor_split is not None
        and len(constraints.tensor_split) != len(constraints.devices)
    ):
        return _Rejection(
            "candidate_constraint_failed",
            {
                "instance_id": instance_id,
                "error": (
                    "tensor_split length does not match selected devices"
                ),
            },
        )
    if constraints.split_mode == "tensor":
        if candidate.compute.flash_attn != "on":
            return _Rejection(
                "unsupported_split_mode",
                {
                    "instance_id": instance_id,
                    "split_mode": "tensor",
                    "error": "tensor split requires flash attention on",
                },
            )
        supported_kv = {"f32", "f16", "bf16"}
        if (
            candidate.context.cache_type_k not in supported_kv
            or candidate.context.cache_type_v not in supported_kv
        ):
            return _Rejection(
                "invalid_kv_configuration",
                {
                    "instance_id": instance_id,
                    "cache_type_k": candidate.context.cache_type_k,
                    "cache_type_v": candidate.context.cache_type_v,
                    "split_mode": "tensor",
                },
            )
    return None


def _check_binary_options(
    binary: BinaryRecord,
    constraints: PlacementConstraints,
    instance_id: str,
) -> _Rejection | None:
    if binary.kind != "llama-server":
        return _Rejection(
            "unsupported_device",
            {
                "instance_id": instance_id,
                "binary_id": binary.id,
                "binary_kind": binary.kind,
                "error": "deployment instance binary is not llama-server",
            },
        )
    capabilities = CapabilitySet.from_mapping(
        dict(binary.capabilities),
        fallback_kind="llama-server",
    )
    required = ["--device"]
    if constraints.tensor_split is not None:
        required.append("--tensor-split")
    if constraints.split_mode != "layer":
        required.append("--split-mode")
    missing = [
        option
        for option in required
        if not capabilities.supports(option)
    ]
    if not missing:
        return None
    reason: DeploymentRejectionReason = (
        "unsupported_split_mode"
        if "--tensor-split" in missing or "--split-mode" in missing
        else "unsupported_device"
    )
    return _Rejection(
        reason,
        {
            "instance_id": instance_id,
            "binary_id": binary.id,
            "missing_options": ",".join(sorted(missing)),
        },
    )


def _physical_device_id(
    policy: HostResourcePolicy,
    binary_id: str,
    device: AcceleratorDeviceRecord,
) -> str | None:
    if device.physical_device_key:
        return device.physical_device_key
    for mapping in policy.logical_device_mappings:
        if (
            mapping.binary_id == binary_id
            and mapping.logical_device_name == device.logical_device_name
        ):
            return mapping.device_id
    return None


def _check_backend_pairs(
    policy: HostResourcePolicy,
    devices: tuple[_DeviceContext, ...],
) -> _Rejection | None:
    if not policy.allowed_backend_pairs:
        return None
    by_device: dict[str, set[str]] = {}
    for device in devices:
        by_device.setdefault(device.device_id, set()).add(device.backend)
    allowed = {
        tuple(sorted((item.left, item.right)))
        for item in policy.allowed_backend_pairs
    }
    for left, right in combinations(sorted(by_device), 2):
        for left_backend in sorted(by_device[left]):
            for right_backend in sorted(by_device[right]):
                pair = tuple(sorted((left_backend, right_backend)))
                if pair not in allowed:
                    return _Rejection(
                        "unsupported_backend_pair",
                        {
                            "left_device": left,
                            "right_device": right,
                            "left_backend": left_backend,
                            "right_backend": right_backend,
                        },
                    )
    return None


def _memory_feasibility(
    policy: HostResourcePolicy,
    devices: tuple[_DeviceContext, ...],
    estimates: Mapping[str, MemoryEstimateObservation],
) -> tuple[DeploymentDeviceAllocation, ...] | _Rejection:
    lookup = {
        (item.instance_id, item.logical_device_name): item
        for item in devices
    }
    projected: dict[str, int] = {}
    totals: dict[str, list[int]] = {}
    usable: dict[str, list[int]] = {}
    logical_to_physical: dict[str, set[str]] = {}

    for instance_id, observation in estimates.items():
        for row in observation.output.devices:
            context = lookup.get(
                (instance_id, row.logical_device_name)
            )
            if context is None:
                return _Rejection(
                    "unsupported_device",
                    {
                        "instance_id": instance_id,
                        "logical_device_name": row.logical_device_name,
                        "error": "estimator returned an unmapped device",
                    },
                )
            projected[context.device_id] = (
                projected.get(context.device_id, 0) + row.total_bytes
            )
            totals.setdefault(context.device_id, []).append(
                row.device_total_bytes
            )
            usable.setdefault(context.device_id, []).append(
                row.device_free_bytes
            )
            logical_to_physical.setdefault(
                row.logical_device_name,
                set(),
            ).add(context.device_id)

    margins: dict[str, int] = {}
    for margin in policy.device_memory_margin_bytes:
        target = margin.device_id
        if target in projected:
            physical = target
        else:
            matches = logical_to_physical.get(target, set())
            if len(matches) != 1:
                return _Rejection(
                    "unsupported_device",
                    {
                        "margin_device_id": target,
                        "error": (
                            "memory margin does not resolve to one "
                            "selected device"
                        ),
                    },
                )
            physical = next(iter(matches))
        if physical in margins:
            return _Rejection(
                "candidate_constraint_failed",
                {
                    "device_id": physical,
                    "error": (
                        "multiple memory margins resolve to the same device"
                    ),
                },
            )
        margins[physical] = margin.margin_bytes

    allocations: list[DeploymentDeviceAllocation] = []
    for device_id in sorted(projected):
        projected_bytes = projected[device_id]
        margin_bytes = margins.get(device_id, 0)
        device_total = min(totals[device_id])
        available = min(usable[device_id])
        required = projected_bytes + margin_bytes
        if required > available:
            return _Rejection(
                "device_memory_exceeded",
                {
                    "device_id": device_id,
                    "projected_bytes": projected_bytes,
                    "margin_bytes": margin_bytes,
                    "required_bytes": required,
                    "available_bytes": available,
                    "device_total_bytes": device_total,
                },
            )
        allocations.append(
            DeploymentDeviceAllocation(
                device_id=device_id,
                projected_bytes=projected_bytes,
                reserved_margin_bytes=margin_bytes,
                device_total_bytes=device_total,
                projected_free_bytes=device_total - required,
            )
        )
    return tuple(allocations)
