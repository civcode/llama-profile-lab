"""Canonical concurrent workload generation for V2-M6."""

from __future__ import annotations

from itertools import product
from typing import Literal

from llama_profile_lab.domain import (
    Candidate,
    ConcurrentWorkloadCase,
    ConcurrentWorkloadMemberSpec,
    DeploymentCandidate,
    DeploymentPhase,
    WorkloadSuite,
)
from llama_profile_lab.domain.workload import (
    DecodeWorkloadCase,
    PrefillWorkloadCase,
)
from llama_profile_lab.planning.workloads import (
    ExpandedWorkload,
    expand_workload_suite,
)


class ConcurrentWorkloadPlanningError(ValueError):
    """Raised when a deployment workload mix cannot be expanded."""


def generate_concurrent_workloads(
    deployment: DeploymentCandidate,
    *,
    candidates: dict[str, Candidate],
    suite: WorkloadSuite,
) -> tuple[ConcurrentWorkloadCase, ...]:
    """Generate deterministic DD/PP/PD/DP cases for a two-instance deployment."""
    if len(deployment.instances) != 2:
        raise ConcurrentWorkloadPlanningError(
            "canonical DD/PP/PD/DP generation currently requires exactly two "
            "deployment instances"
        )
    expanded: dict[str, tuple[ExpandedWorkload, ...]] = {}
    for instance in deployment.instances:
        candidate = candidates.get(instance.candidate_id)
        if candidate is None:
            raise ConcurrentWorkloadPlanningError(
                f"Candidate not supplied for instance {instance.instance_id}"
            )
        expanded[instance.instance_id] = expand_workload_suite(
            candidate,
            suite,
        )

    cases: list[ConcurrentWorkloadCase] = []
    seen: set[str] = set()
    for phase in deployment.workload_mix.phases:
        modes = _phase_modes(phase)
        choices: list[tuple[ExpandedWorkload, ...]] = []
        for instance, mode in zip(deployment.instances, modes, strict=True):
            selected = tuple(
                item
                for item in expanded[instance.instance_id]
                if _matches_mode(item, mode)
            )
            if not selected:
                raise ConcurrentWorkloadPlanningError(
                    f"workload suite has no {mode} case for "
                    f"instance {instance.instance_id}"
                )
            choices.append(selected)

        for combination in product(*choices):
            members = tuple(
                _member_spec(
                    ordinal=ordinal,
                    instance_id=instance.instance_id,
                    mode=mode,
                    workload=workload,
                )
                for ordinal, (instance, mode, workload) in enumerate(
                    zip(
                        deployment.instances,
                        modes,
                        combination,
                        strict=True,
                    )
                )
            )
            case = ConcurrentWorkloadCase(
                phase=phase,
                members=members,
                provenance={
                    "suite_id": suite.id,
                    "instance_0_suite_case": combination[0].suite_case_index,
                    "instance_1_suite_case": combination[1].suite_case_index,
                },
            )
            digest = case.content_hash()
            if digest in seen:
                continue
            seen.add(digest)
            cases.append(case)

    return tuple(cases)


def _phase_modes(
    phase: DeploymentPhase,
) -> tuple[Literal["prefill", "decode"], Literal["prefill", "decode"]]:
    return {
        "dd": ("decode", "decode"),
        "pp": ("prefill", "prefill"),
        "pd": ("prefill", "decode"),
        "dp": ("decode", "prefill"),
    }[phase]


def _matches_mode(
    workload: ExpandedWorkload,
    mode: Literal["prefill", "decode"],
) -> bool:
    if mode == "prefill":
        return isinstance(workload.case, PrefillWorkloadCase)
    return isinstance(workload.case, DecodeWorkloadCase)


def _member_spec(
    *,
    ordinal: int,
    instance_id: str,
    mode: Literal["prefill", "decode"],
    workload: ExpandedWorkload,
) -> ConcurrentWorkloadMemberSpec:
    case = workload.case
    if mode == "prefill" and isinstance(case, PrefillWorkloadCase):
        return ConcurrentWorkloadMemberSpec(
            ordinal=ordinal,
            instance_id=instance_id,
            mode=mode,
            prompt_tokens=case.prompt_tokens,
            generate_tokens=0,
            depth_tokens=case.depth_tokens,
        )
    if mode == "decode" and isinstance(case, DecodeWorkloadCase):
        return ConcurrentWorkloadMemberSpec(
            ordinal=ordinal,
            instance_id=instance_id,
            mode=mode,
            prompt_tokens=0,
            generate_tokens=case.generate_tokens,
            depth_tokens=case.depth_tokens,
        )
    raise ConcurrentWorkloadPlanningError(
        f"workload {case.kind} cannot be used as {mode}"
    )
