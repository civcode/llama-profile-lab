"""Pure plan construction plus transactional plan persistence."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from llama_profile_lab.db import (
    BenchmarkCaseRepository,
    CandidateRepository,
    ExperimentRepository,
    MeasurementPolicyRepository,
    SearchSpaceRepository,
    WorkloadCaseRepository,
    WorkloadSuiteRepository,
    transaction,
)
from llama_profile_lab.domain import Candidate, SearchSpace, WorkloadSuite
from llama_profile_lab.planning.expand import (
    CandidatePoint,
    PlanningError,
    SearchExpansion,
    expand_search_space,
)
from llama_profile_lab.planning.parameters import (
    DEFAULT_PARAMETER_REGISTRY,
    ParameterRegistry,
)
from llama_profile_lab.planning.workloads import ExpandedWorkload, expand_workload_suite


@dataclass(frozen=True, slots=True)
class PlannedCandidate:
    """One Candidate and all unique concrete workloads planned for it."""

    point: CandidatePoint
    workloads: tuple[ExpandedWorkload, ...]


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    """Pure in-memory experiment plan."""

    search: SearchExpansion
    candidates: tuple[PlannedCandidate, ...]

    @property
    def benchmark_case_count(self) -> int:
        return sum(len(candidate.workloads) for candidate in self.candidates)


@dataclass(frozen=True, slots=True)
class PlanSummary:
    """Human-facing summary of a persisted experiment plan."""

    experiment_id: str
    experiment_name: str
    raw_combinations: int
    rejected_by_constraints: int
    duplicate_candidates: int
    candidate_count: int
    workloads_per_candidate: int | None
    benchmark_case_count: int
    unique_workload_count: int


def build_plan(
    base_candidate: Candidate,
    search_space: SearchSpace,
    workload_suite: WorkloadSuite,
    registry: ParameterRegistry = DEFAULT_PARAMETER_REGISTRY,
) -> ExperimentPlan:
    """Build an experiment plan without persistence or subprocesses."""
    search = expand_search_space(base_candidate, search_space, registry)
    planned = tuple(
        PlannedCandidate(
            point=point,
            workloads=expand_workload_suite(point.candidate, workload_suite),
        )
        for point in search.candidates
    )
    return ExperimentPlan(search=search, candidates=planned)


def plan_experiment(
    connection: sqlite3.Connection,
    experiment_id: str,
    registry: ParameterRegistry = DEFAULT_PARAMETER_REGISTRY,
) -> PlanSummary:
    """Build and atomically persist one draft experiment's complete plan."""
    experiments = ExperimentRepository(connection)
    experiment_record = experiments.get(experiment_id)
    definition = experiments.get_definition(experiment_id)
    if experiment_record is None or definition is None:
        raise PlanningError(f"experiment not found: {experiment_id}")
    if experiment_record.status != "draft":
        raise PlanningError(
            f"experiment {experiment_id} is {experiment_record.status}, not draft"
        )

    candidate_repo = CandidateRepository(connection)
    search_repo = SearchSpaceRepository(connection)
    suite_repo = WorkloadSuiteRepository(connection)
    policy_repo = MeasurementPolicyRepository(connection)

    base_candidate = candidate_repo.get(definition.base_candidate_id)
    search_space = search_repo.get(definition.search_space_id)
    workload_suite = suite_repo.get(definition.workload_suite_id)
    measurement_policy = policy_repo.get(definition.measurement_policy_id)

    if base_candidate is None:
        raise PlanningError(f"base Candidate not found: {definition.base_candidate_id}")
    if search_space is None:
        raise PlanningError(f"SearchSpace not found: {definition.search_space_id}")
    if workload_suite is None:
        raise PlanningError(f"WorkloadSuite not found: {definition.workload_suite_id}")
    if measurement_policy is None:
        raise PlanningError(
            f"MeasurementPolicy not found: {definition.measurement_policy_id}"
        )

    plan = build_plan(base_candidate, search_space, workload_suite, registry)

    workload_repo = WorkloadCaseRepository(connection)
    benchmark_repo = BenchmarkCaseRepository(connection)
    unique_workloads: set[str] = set()
    case_ordinal = 0

    with transaction(connection, immediate=True):
        current = experiments.get(experiment_id)
        if current is None or current.status != "draft":
            state = "missing" if current is None else current.status
            raise PlanningError(
                f"experiment {experiment_id} changed state before planning: {state}"
            )

        for candidate_ordinal, planned_candidate in enumerate(plan.candidates):
            candidate_id = candidate_repo.put(planned_candidate.point.candidate)
            experiments.add_candidate(
                experiment_id=experiment_id,
                candidate_id=candidate_id,
                ordinal=candidate_ordinal,
                generation_metadata=planned_candidate.point.generation_metadata(),
            )

            for workload in planned_candidate.workloads:
                workload_id = workload_repo.put(workload.case)
                unique_workloads.add(workload_id)
                experiments.add_workload(
                    experiment_id=experiment_id,
                    candidate_id=candidate_id,
                    workload_case_id=workload_id,
                    suite_case_index=workload.suite_case_index,
                    expansion_provenance=workload.provenance,
                )
                benchmark_repo.put_planned(
                    experiment_id=experiment_id,
                    candidate_id=candidate_id,
                    workload_case_id=workload_id,
                    ordinal=case_ordinal,
                )
                case_ordinal += 1

        experiments.mark_planned(experiment_id)

    workload_counts = {len(candidate.workloads) for candidate in plan.candidates}
    workloads_per_candidate = (
        workload_counts.pop() if len(workload_counts) == 1 else None
    )
    return PlanSummary(
        experiment_id=experiment_id,
        experiment_name=definition.name,
        raw_combinations=plan.search.raw_combinations,
        rejected_by_constraints=plan.search.rejected_by_constraints,
        duplicate_candidates=plan.search.duplicate_candidates,
        candidate_count=len(plan.candidates),
        workloads_per_candidate=workloads_per_candidate,
        benchmark_case_count=plan.benchmark_case_count,
        unique_workload_count=len(unique_workloads),
    )


def render_plan_summary(summary: PlanSummary) -> str:
    """Render a compact deterministic CLI summary."""
    workloads = (
        str(summary.workloads_per_candidate)
        if summary.workloads_per_candidate is not None
        else "varies"
    )
    return "\n".join(
        (
            f"Experiment: {summary.experiment_name} ({summary.experiment_id})",
            f"Raw combinations: {summary.raw_combinations}",
            f"Rejected by constraints: {summary.rejected_by_constraints}",
            f"Duplicate candidates: {summary.duplicate_candidates}",
            f"Candidates: {summary.candidate_count}",
            f"Workloads per candidate: {workloads}",
            f"Benchmark cases: {summary.benchmark_case_count}",
            f"Unique concrete workloads: {summary.unique_workload_count}",
            "Status: planned",
        )
    )
