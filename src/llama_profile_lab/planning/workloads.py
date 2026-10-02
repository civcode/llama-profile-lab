"""Candidate-dependent WorkloadSuite expansion."""

from __future__ import annotations

import math
from dataclasses import dataclass

from llama_profile_lab.domain import Candidate, WorkloadSuite
from llama_profile_lab.domain.workload import (
    AbsoluteDepth,
    CombinedSuiteCase,
    CombinedWorkloadCase,
    DecodeSuiteCase,
    DecodeWorkloadCase,
    FractionalDepth,
    PrefillSuiteCase,
    PrefillWorkloadCase,
    SpeedBenchSuiteCase,
    SpeedBenchWorkloadCase,
    WorkloadCase,
    WorkloadEnvelope,
)
from llama_profile_lab.planning.expand import PlanningError


@dataclass(frozen=True, slots=True)
class ExpandedWorkload:
    """One concrete Candidate-dependent workload and provenance."""

    case: WorkloadCase
    suite_case_index: int
    provenance: dict[str, object]


def expand_workload_suite(
    candidate: Candidate,
    suite: WorkloadSuite,
) -> tuple[ExpandedWorkload, ...]:
    """Expand symbolic workload cases for one concrete Candidate."""
    expanded: list[ExpandedWorkload] = []
    seen_hashes: set[str] = set()

    for index, suite_case in enumerate(suite.cases):
        workload = _expand_case(candidate, suite, index, suite_case)
        digest = workload.case.content_hash()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        expanded.append(workload)

    return tuple(expanded)


def _expand_case(
    candidate: Candidate,
    suite: WorkloadSuite,
    index: int,
    suite_case: PrefillSuiteCase
    | DecodeSuiteCase
    | CombinedSuiteCase
    | SpeedBenchSuiteCase,
) -> ExpandedWorkload:
    if isinstance(suite_case, SpeedBenchSuiteCase):
        return ExpandedWorkload(
            case=SpeedBenchWorkloadCase(speed_bench=suite_case.speed_bench),
            suite_case_index=index,
            provenance={
                "suite_id": suite.id,
                "suite_case_index": index,
                "candidate_context_size": candidate.context.size,
            },
        )

    prompt_tokens = (
        suite_case.prompt_tokens
        if isinstance(suite_case, PrefillSuiteCase | CombinedSuiteCase)
        else 0
    )
    generate_tokens = (
        suite_case.generate_tokens
        if isinstance(suite_case, DecodeSuiteCase | CombinedSuiteCase)
        else 0
    )
    safety_margin = suite_case.safety_margin_tokens

    if isinstance(suite_case.depth, FractionalDepth):
        available_depth = (
            candidate.context.size
            - prompt_tokens
            - generate_tokens
            - safety_margin
        )
        if available_depth < 0:
            raise PlanningError(
                f"workload suite case {index} exceeds Candidate context before depth"
            )
        depth_tokens = math.floor(available_depth * suite_case.depth.value)
    elif isinstance(suite_case.depth, AbsoluteDepth):
        depth_tokens = suite_case.depth.tokens
    else:
        raise PlanningError(f"unsupported depth expression in suite case {index}")

    occupied_with_margin = (
        depth_tokens + prompt_tokens + generate_tokens + safety_margin
    )
    if occupied_with_margin > candidate.context.size:
        raise PlanningError(
            f"workload suite case {index} exceeds Candidate context size"
        )

    if isinstance(suite_case, PrefillSuiteCase):
        concrete: WorkloadCase = PrefillWorkloadCase(
            prompt_tokens=prompt_tokens,
            depth_tokens=depth_tokens,
        )
    elif isinstance(suite_case, DecodeSuiteCase):
        concrete = DecodeWorkloadCase(
            generate_tokens=generate_tokens,
            depth_tokens=depth_tokens,
        )
    else:
        concrete = CombinedWorkloadCase(
            prompt_tokens=prompt_tokens,
            generate_tokens=generate_tokens,
            depth_tokens=depth_tokens,
        )

    WorkloadEnvelope(case=concrete, context_size=candidate.context.size)

    return ExpandedWorkload(
        case=concrete,
        suite_case_index=index,
        provenance={
            "suite_id": suite.id,
            "suite_case_index": index,
            "candidate_context_size": candidate.context.size,
            "safety_margin_tokens": safety_margin,
            "original_depth": suite_case.depth.model_dump(mode="json"),
        },
    )
