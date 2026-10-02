"""Deterministic N-dimensional Candidate expansion."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from llama_profile_lab.domain import Candidate, SearchSpace
from llama_profile_lab.domain.base import JsonScalar
from llama_profile_lab.planning.constraints import (
    ConstraintError,
    evaluate_constraint,
    validate_constraint,
)
from llama_profile_lab.planning.parameters import (
    DEFAULT_PARAMETER_REGISTRY,
    ParameterError,
    ParameterRegistry,
)


class PlanningError(ValueError):
    """Raised when an experiment definition cannot be planned."""


@dataclass(frozen=True, slots=True)
class CandidatePoint:
    """One concrete Candidate plus its generation provenance."""

    candidate: Candidate
    assignments: tuple[tuple[str, JsonScalar], ...]
    skipped_dimensions: tuple[str, ...]

    def generation_metadata(self) -> dict[str, object]:
        """Return persistence-friendly search-generation metadata."""
        return {
            "assignments": dict(self.assignments),
            "skipped_dimensions": list(self.skipped_dimensions),
        }


@dataclass(frozen=True, slots=True)
class SearchExpansion:
    """Result of deterministic search-space expansion."""

    candidates: tuple[CandidatePoint, ...]
    raw_combinations: int
    rejected_by_constraints: int
    duplicate_candidates: int


@dataclass(slots=True)
class _ExpansionState:
    payload: dict[str, Any]
    assignments: list[tuple[str, JsonScalar]]
    skipped_dimensions: list[str]


def expand_search_space(
    base_candidate: Candidate,
    search_space: SearchSpace,
    registry: ParameterRegistry = DEFAULT_PARAMETER_REGISTRY,
) -> SearchExpansion:
    """Expand a discrete SearchSpace into unique immutable Candidates."""
    try:
        registry.validate_search_space(search_space)
        for dimension in search_space.dimensions:
            if dimension.condition is not None:
                validate_constraint(dimension.condition)
        for constraint in search_space.constraints:
            validate_constraint(constraint.expression)
    except (ParameterError, ConstraintError) as exc:
        raise PlanningError(str(exc)) from exc

    base_payload = base_candidate.model_dump(mode="python", by_alias=False)
    states = [_ExpansionState(base_payload, [], [])]

    for dimension in search_space.dimensions:
        next_states: list[_ExpansionState] = []
        for state in states:
            if dimension.condition is not None:
                try:
                    enabled = evaluate_constraint(dimension.condition, state.payload)
                except ConstraintError as exc:
                    raise PlanningError(
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

    raw_combinations = len(states)
    rejected = 0
    duplicates = 0
    points: list[CandidatePoint] = []
    seen_hashes: set[str] = set()

    for state in states:
        try:
            allowed = all(
                evaluate_constraint(constraint.expression, state.payload)
                for constraint in search_space.constraints
            )
        except ConstraintError as exc:
            raise PlanningError(f"cannot evaluate search constraint: {exc}") from exc

        if not allowed:
            rejected += 1
            continue

        try:
            candidate = Candidate.model_validate(state.payload)
        except ValidationError as exc:
            assignments = ", ".join(
                f"{path}={value!r}" for path, value in state.assignments
            )
            raise PlanningError(
                f"search combination produced an invalid Candidate ({assignments}): {exc}"
            ) from exc

        digest = candidate.content_hash()
        if digest in seen_hashes:
            duplicates += 1
            continue
        seen_hashes.add(digest)
        points.append(
            CandidatePoint(
                candidate=candidate,
                assignments=tuple(state.assignments),
                skipped_dimensions=tuple(state.skipped_dimensions),
            )
        )

    return SearchExpansion(
        candidates=tuple(points),
        raw_combinations=raw_combinations,
        rejected_by_constraints=rejected,
        duplicate_candidates=duplicates,
    )


def _set_path(payload: dict[str, Any], path: str, value: JsonScalar) -> None:
    parts = path.split(".")
    current: dict[str, Any] = payload
    for part in parts[:-1]:
        nested = current.get(part)
        if not isinstance(nested, dict):
            raise PlanningError(
                f"Candidate path {path!r} cannot be applied at component {part!r}"
            )
        current = nested

    leaf = parts[-1]
    if leaf not in current:
        raise PlanningError(f"Candidate path does not exist: {path}")
    current[leaf] = value
