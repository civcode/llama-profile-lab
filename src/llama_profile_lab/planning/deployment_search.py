"""Deterministic deployment-wide parameter registry and grid expansion."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import ValidationError

from llama_profile_lab.domain import (
    Candidate,
    DeploymentCandidate,
    DeploymentPlacementRequest,
    DeploymentSearchSpace,
    HostResourcePolicy,
    ModelInstanceCandidate,
)
from llama_profile_lab.domain.deployment_search import DeploymentSearchValue
from llama_profile_lab.planning.constraints import (
    ConstraintError,
    evaluate_constraint,
    validate_constraint,
)


class DeploymentPlanningError(ValueError):
    """Raised when a deployment search definition is invalid."""


@dataclass(frozen=True, slots=True)
class DeploymentParameterDefinition:
    """Metadata for one dynamic deployment search path family."""

    path_kind: str
    suffix: str
    value_kinds: tuple[type[object], ...]


class DeploymentParameterRegistry:
    """Validate instance-addressable Candidate and placement search paths."""

    _candidate_paths = frozenset(
        {
            "context.size",
            "context.cache_type_k",
            "context.cache_type_v",
            "compute.batch_size",
            "compute.ubatch_size",
            "compute.flash_attn",
            "placement.constraints.n_gpu_layers",
            "placement.constraints.split_mode",
            "placement.constraints.main_gpu",
            "placement.constraints.tensor_split",
        }
    )
    _request_paths = {
        "devices": (tuple,),
        "n_gpu_layers": (int, str, type(None)),
        "split_mode": (str, type(None)),
        "main_gpu": (int, type(None)),
        "tensor_split": (tuple, type(None)),
        "override_tensor": (tuple,),
    }

    def validate_search_space(
        self,
        base: DeploymentCandidate,
        search_space: DeploymentSearchSpace,
    ) -> None:
        instance_ids = {item.instance_id for item in base.instances}
        for dimension in search_space.dimensions:
            path = dimension.path
            if path.startswith("instances."):
                parts = path.split(".")
                if len(parts) < 3 or parts[1] not in instance_ids:
                    raise DeploymentPlanningError(
                        f"unknown deployment instance path: {path}"
                    )
                suffix = ".".join(parts[2:])
                if suffix.startswith("requested_placement."):
                    field = suffix.removeprefix("requested_placement.")
                    kinds = self._request_paths.get(field)
                    if kinds is None:
                        raise DeploymentPlanningError(
                            f"unsupported deployment placement parameter: {path}"
                        )
                    for value in dimension.values:
                        if type(value) not in kinds:
                            expected = ", ".join(kind.__name__ for kind in kinds)
                            raise DeploymentPlanningError(
                                f"{path} expects {expected}; "
                                f"got {type(value).__name__}"
                            )
                    continue
                if suffix not in self._candidate_paths:
                    raise DeploymentPlanningError(
                        f"unsupported instance Candidate parameter: {path}"
                    )
                continue

            if path.startswith("resource_policy.device_memory_margin_bytes."):
                for value in dimension.values:
                    if type(value) is not int or value < 0:
                        raise DeploymentPlanningError(
                            f"{path} expects a non-negative integer"
                        )
                continue
            if path == "resource_policy.host_ram_margin_bytes":
                for value in dimension.values:
                    if type(value) is not int or value < 0:
                        raise DeploymentPlanningError(
                            f"{path} expects a non-negative integer"
                        )
                continue
            raise DeploymentPlanningError(
                f"unsupported deployment search parameter: {path}"
            )


DEFAULT_DEPLOYMENT_PARAMETER_REGISTRY = DeploymentParameterRegistry()


@dataclass(frozen=True, slots=True)
class DeploymentPoint:
    """One concrete expanded deployment and its generation provenance."""

    deployment: DeploymentCandidate
    candidates: tuple[tuple[str, Candidate], ...]
    assignments: tuple[tuple[str, DeploymentSearchValue], ...]
    skipped_dimensions: tuple[str, ...]

    def candidate_for(self, instance_id: str) -> Candidate:
        for current_id, candidate in self.candidates:
            if current_id == instance_id:
                return candidate
        raise KeyError(instance_id)

    def generation_metadata(self) -> dict[str, object]:
        return {
            "assignments": {path: value for path, value in self.assignments},
            "skipped_dimensions": list(self.skipped_dimensions),
        }


@dataclass(frozen=True, slots=True)
class DeploymentSearchExpansion:
    """Deterministic deployment grid expansion counts and unique points."""

    points: tuple[DeploymentPoint, ...]
    raw_combinations: int
    rejected_by_constraints: int
    duplicate_candidates: int
    symmetry_reduced: int


@dataclass(slots=True)
class _ExpansionState:
    payload: dict[str, Any]
    assignments: list[tuple[str, DeploymentSearchValue]]
    skipped_dimensions: list[str]


def expand_deployment_search(
    base: DeploymentCandidate,
    candidates: dict[str, Candidate],
    search_space: DeploymentSearchSpace,
    registry: DeploymentParameterRegistry = DEFAULT_DEPLOYMENT_PARAMETER_REGISTRY,
    *,
    symmetry_key: Callable[[DeploymentCandidate], str] | None = None,
) -> DeploymentSearchExpansion:
    """Expand one deployment grid without executing estimators or benchmarks."""
    expected = {item.instance_id for item in base.instances}
    if set(candidates) != expected:
        raise DeploymentPlanningError(
            "base Candidate map must contain exactly the deployment instances"
        )

    try:
        registry.validate_search_space(base, search_space)
        for dimension in search_space.dimensions:
            if dimension.condition is not None:
                validate_constraint(dimension.condition)
        for constraint in search_space.constraints:
            validate_constraint(constraint.expression)
    except ConstraintError as exc:
        raise DeploymentPlanningError(str(exc)) from exc

    base_payload = _search_payload(base, candidates)
    states = [_ExpansionState(base_payload, [], [])]
    for dimension in search_space.dimensions:
        next_states: list[_ExpansionState] = []
        for state in states:
            if dimension.condition is not None:
                try:
                    enabled = evaluate_constraint(
                        dimension.condition,
                        state.payload,
                    )
                except ConstraintError as exc:
                    raise DeploymentPlanningError(
                        f"cannot evaluate condition for {dimension.path}: {exc}"
                    ) from exc
                if not enabled:
                    next_states.append(
                        _ExpansionState(
                            deepcopy(state.payload),
                            list(state.assignments),
                            [*state.skipped_dimensions, dimension.path],
                        )
                    )
                    continue
            for value in dimension.values:
                payload = deepcopy(state.payload)
                _set_path(payload, dimension.path, value)
                next_states.append(
                    _ExpansionState(
                        payload,
                        [*state.assignments, (dimension.path, value)],
                        list(state.skipped_dimensions),
                    )
                )
        states = next_states

    rejected = 0
    duplicates = 0
    symmetry_reduced = 0
    points: list[DeploymentPoint] = []
    seen_hashes: set[str] = set()
    seen_symmetry: set[str] = set()

    for state in states:
        try:
            allowed = all(
                evaluate_constraint(constraint.expression, state.payload)
                for constraint in search_space.constraints
            )
        except ConstraintError as exc:
            raise DeploymentPlanningError(
                f"cannot evaluate deployment search constraint: {exc}"
            ) from exc
        if not allowed:
            rejected += 1
            continue

        point = _build_point(base, state)
        digest = point.deployment.content_hash()
        if digest in seen_hashes:
            duplicates += 1
            continue
        seen_hashes.add(digest)
        if symmetry_key is not None:
            key = symmetry_key(point.deployment)
            if key in seen_symmetry:
                symmetry_reduced += 1
                continue
            seen_symmetry.add(key)
        points.append(point)

    return DeploymentSearchExpansion(
        points=tuple(points),
        raw_combinations=len(states),
        rejected_by_constraints=rejected,
        duplicate_candidates=duplicates,
        symmetry_reduced=symmetry_reduced,
    )


def _search_payload(
    base: DeploymentCandidate,
    candidates: dict[str, Candidate],
) -> dict[str, Any]:
    instances: dict[str, Any] = {}
    by_id = {item.instance_id: item for item in base.instances}
    for instance_id in sorted(candidates):
        payload = candidates[instance_id].model_dump(
            mode="python",
            by_alias=False,
        )
        payload["requested_placement"] = by_id[
            instance_id
        ].requested_placement.model_dump(mode="python")
        instances[instance_id] = payload
    return {
        "instances": instances,
        "resource_policy": base.resource_policy.model_dump(mode="python"),
    }


def _set_path(
    payload: dict[str, Any],
    path: str,
    value: DeploymentSearchValue,
) -> None:
    parts = path.split(".")
    current: dict[str, Any] = payload
    for part in parts[:-1]:
        nested = current.get(part)
        if not isinstance(nested, dict):
            if path.startswith(
                "resource_policy.device_memory_margin_bytes."
            ):
                current[part] = {}
                nested = current[part]
            else:
                raise DeploymentPlanningError(
                    f"deployment path {path!r} cannot be applied at {part!r}"
                )
        current = nested
    current[parts[-1]] = value


def _build_point(
    base: DeploymentCandidate,
    state: _ExpansionState,
) -> DeploymentPoint:
    candidate_pairs: list[tuple[str, Candidate]] = []
    instances: list[ModelInstanceCandidate] = []
    base_instances = {item.instance_id: item for item in base.instances}
    try:
        for instance_id in sorted(base_instances):
            raw = deepcopy(state.payload["instances"][instance_id])
            request_raw = raw.pop("requested_placement")
            candidate = Candidate.model_validate(raw)
            request = DeploymentPlacementRequest.model_validate(request_raw)
            base_instance = base_instances[instance_id]
            if candidate.model.target_model_id != base_instance.model_artifact_id:
                raise DeploymentPlanningError(
                    f"instance {instance_id} Candidate target model does not "
                    "match model_artifact_id"
                )
            candidate_pairs.append((instance_id, candidate))
            instances.append(
                base_instance.model_copy(
                    update={
                        "candidate_id": f"cand_{candidate.content_hash()}",
                        "requested_placement": request,
                    }
                )
            )
        resource_policy = HostResourcePolicy.model_validate(
            state.payload["resource_policy"]
        )
        deployment = DeploymentCandidate(
            instances=tuple(instances),
            resource_policy=resource_policy,
            workload_mix=base.workload_mix,
        )
    except ValidationError as exc:
        assignments = ", ".join(
            f"{path}={value!r}" for path, value in state.assignments
        )
        raise DeploymentPlanningError(
            f"deployment search combination is invalid "
            f"({assignments}): {exc}"
        ) from exc

    return DeploymentPoint(
        deployment=deployment,
        candidates=tuple(candidate_pairs),
        assignments=tuple(state.assignments),
        skipped_dimensions=tuple(state.skipped_dimensions),
    )
