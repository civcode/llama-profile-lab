"""Concurrent DD/PP/PD/DP orchestration over resident deployment servers."""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from threading import Barrier, BrokenBarrierError, Event
from typing import Callable

from llama_profile_lab.analysis.concurrent import (
    ConcurrentMetricError,
    MemberOverlap,
    compute_overlap,
    native_throughput,
    retention_for,
)
from llama_profile_lab.db import (
    CandidateRepository,
    ConcurrentWorkloadRepository,
    Database,
    DeploymentCandidateRepository,
    DeploymentPlacementRepository,
    StandaloneBaselineRepository,
    WorkloadSuiteRepository,
)
from llama_profile_lab.domain import (
    ConcurrentMemberResult,
    ConcurrentQuality,
    ConcurrentWorkloadCase,
    ConcurrentWorkloadMemberSpec,
)
from llama_profile_lab.execution.concurrent_client import (
    ConcurrentClient,
    ConcurrentClientError,
    LlamaCompletionConcurrentClient,
    PreparedConcurrentClient,
)
from llama_profile_lab.execution.deployment_executor import (
    DeploymentExecutionSummary,
    DeploymentExecutor,
    DeploymentResidentActionError,
    DeploymentServerInput,
    ResidentDeployment,
)
from llama_profile_lab.planning import generate_concurrent_workloads


class ConcurrentDeploymentError(RuntimeError):
    """Raised when a deployment concurrent workload run cannot be completed."""


@dataclass(frozen=True, slots=True)
class ConcurrentPhaseSummary:
    """One persisted concurrent phase result."""

    run_id: str
    workload_case_id: str
    phase: str
    quality: ConcurrentQuality
    combined_prompt_tps: float | None
    combined_decode_tps: float | None
    min_retention: float | None


@dataclass(frozen=True, slots=True)
class ConcurrentDeploymentSummary:
    """Complete M6 execution summary for one residency session."""

    deployment_run_id: str
    deployment_placement_id: str
    phases: tuple[ConcurrentPhaseSummary, ...]
    execution: DeploymentExecutionSummary


@dataclass(frozen=True, slots=True)
class StandaloneBaselineInput:
    """User/import-facing exact standalone denominator."""

    instance_id: str
    mode: str
    prompt_tokens: int
    generate_tokens: int
    depth_tokens: int
    throughput_tps: float
    latency_ms: float | None = None


@dataclass(frozen=True, slots=True)
class _BaselineIdentity:
    candidate_id: str
    resolved_placement_id: str
    host_id: str
    binary_id: str


class ConcurrentDeploymentExecutor:
    """Run generated concurrent phases while M5 keeps every server resident."""

    def __init__(
        self,
        database: Database,
        *,
        deployment_executor: DeploymentExecutor | None = None,
        client: ConcurrentClient | None = None,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.database = database
        self.deployment_executor = (
            deployment_executor or DeploymentExecutor(database)
        )
        self.client = client or LlamaCompletionConcurrentClient()
        self.clock = clock

    def execute(
        self,
        deployment_placement_id: str,
        inputs: tuple[DeploymentServerInput, ...],
        *,
        standalone_baselines: tuple[StandaloneBaselineInput, ...] = (),
        host: str = "127.0.0.1",
        readiness_timeout_seconds: float = 300.0,
        cancel_event: Event | None = None,
    ) -> ConcurrentDeploymentSummary:
        (
            deployment,
            workload_cases,
            baseline_identity,
        ) = self._prepare_plan(deployment_placement_id)

        if standalone_baselines:
            self._persist_baselines(
                deployment,
                baseline_identity,
                standalone_baselines,
            )

        phase_summaries: list[ConcurrentPhaseSummary] = []

        def resident_action(
            resident: ResidentDeployment,
            resident_cancel: Event | None,
        ) -> dict[str, object]:
            if resident.deployment_candidate_id != deployment.content_id(
                "deploy"
            ):
                # Repository IDs are content-addressed from the same hash.
                raise DeploymentResidentActionError(
                    "resident deployment Candidate does not match workload plan"
                )
            endpoint_by_instance = {
                item.instance_id: item.endpoint
                for item in resident.members
            }
            for workload_case_id, workload in workload_cases:
                if resident_cancel is not None and resident_cancel.is_set():
                    raise DeploymentResidentActionError(
                        "concurrent workload execution cancelled",
                        cancelled=True,
                    )
                summary = self._run_phase(
                    resident,
                    workload_case_id,
                    workload,
                    endpoint_by_instance,
                    baseline_identity,
                    cancel_event=resident_cancel,
                )
                phase_summaries.append(summary)
            return {
                "concurrent_workload_runs": [
                    item.run_id for item in phase_summaries
                ],
                "phase_count": len(phase_summaries),
            }

        execution = self.deployment_executor.execute(
            deployment_placement_id,
            inputs,
            host=host,
            readiness_timeout_seconds=readiness_timeout_seconds,
            cancel_event=cancel_event,
            resident_action=resident_action,
        )
        return ConcurrentDeploymentSummary(
            deployment_run_id=execution.run_id,
            deployment_placement_id=deployment_placement_id,
            phases=tuple(phase_summaries),
            execution=execution,
        )

    def _prepare_plan(
        self,
        deployment_placement_id: str,
    ) -> tuple[
        object,
        tuple[tuple[str, ConcurrentWorkloadCase], ...],
        dict[str, _BaselineIdentity],
    ]:
        with self.database.session() as connection:
            placements = DeploymentPlacementRepository(connection)
            placement = placements.get(deployment_placement_id)
            if placement is None:
                raise ConcurrentDeploymentError(
                    f"deployment placement not found: {deployment_placement_id}"
                )
            deployment = DeploymentCandidateRepository(connection).get(
                placement.deployment_candidate_id
            )
            if deployment is None:
                raise ConcurrentDeploymentError(
                    "deployment Candidate not found for placement"
                )
            suite = WorkloadSuiteRepository(connection).get(
                deployment.workload_mix.workload_suite_id
            )
            if suite is None:
                raise ConcurrentDeploymentError(
                    "deployment workload suite not found"
                )
            candidates_repo = CandidateRepository(connection)
            candidates = {}
            instances = {item.instance_id: item for item in deployment.instances}
            for instance in deployment.instances:
                candidate = candidates_repo.get(instance.candidate_id)
                if candidate is None:
                    raise ConcurrentDeploymentError(
                        f"Candidate not found: {instance.candidate_id}"
                    )
                candidates[instance.candidate_id] = candidate

            generated = generate_concurrent_workloads(
                deployment,
                candidates=candidates,
                suite=suite,
            )
            workloads = ConcurrentWorkloadRepository(connection)
            workload_cases = tuple(
                (
                    workloads.put_case(placement.deployment_candidate_id, case),
                    case,
                )
                for case in generated
            )
            resolved_by_instance = {
                item.instance_id: item.resolved_placement_id
                for item in placement.instance_placements
            }
            baseline_identity = {
                instance_id: _BaselineIdentity(
                    candidate_id=instance.candidate_id,
                    resolved_placement_id=resolved_by_instance[instance_id],
                    host_id=placement.host_id,
                    binary_id=instance.binary_id,
                )
                for instance_id, instance in instances.items()
            }
        return deployment, workload_cases, baseline_identity

    def _persist_baselines(
        self,
        deployment,
        identities: dict[str, _BaselineIdentity],
        values: tuple[StandaloneBaselineInput, ...],
    ) -> None:
        valid_instances = {
            item.instance_id for item in deployment.instances
        }
        with self.database.session() as connection:
            repository = StandaloneBaselineRepository(connection)
            for value in values:
                if value.instance_id not in valid_instances:
                    raise ConcurrentDeploymentError(
                        f"unknown baseline instance: {value.instance_id}"
                    )
                if value.mode not in {"prefill", "decode"}:
                    raise ConcurrentDeploymentError(
                        "baseline mode must be prefill or decode"
                    )
                if value.throughput_tps <= 0:
                    raise ConcurrentDeploymentError(
                        "baseline throughput must be positive"
                    )
                identity = identities[value.instance_id]
                repository.add(
                    candidate_id=identity.candidate_id,
                    resolved_placement_id=identity.resolved_placement_id,
                    host_id=identity.host_id,
                    binary_id=identity.binary_id,
                    mode=value.mode,
                    prompt_tokens=value.prompt_tokens,
                    generate_tokens=value.generate_tokens,
                    depth_tokens=value.depth_tokens,
                    throughput_tps=value.throughput_tps,
                    latency_ms=value.latency_ms,
                    source={"source": "deployment-execution-spec"},
                )

    def _run_phase(
        self,
        resident: ResidentDeployment,
        workload_case_id: str,
        workload: ConcurrentWorkloadCase,
        endpoints: dict[str, str],
        identities: dict[str, _BaselineIdentity],
        *,
        cancel_event: Event | None,
    ) -> ConcurrentPhaseSummary:
        with self.database.session() as connection:
            run_id = ConcurrentWorkloadRepository(connection).create_run(
                deployment_run_id=resident.run_id,
                workload_case_id=workload_case_id,
                phase=workload.phase,
            )

        prepared: dict[str, PreparedConcurrentClient] = {}
        prepare_failures: dict[str, str] = {}
        with ThreadPoolExecutor(
            max_workers=len(workload.members)
        ) as executor:
            futures = {
                executor.submit(
                    self.client.prepare,
                    endpoints[member.instance_id],
                    member,
                    cancel_event=cancel_event,
                ): member
                for member in workload.members
            }
            for future, member in futures.items():
                try:
                    prepared[member.instance_id] = future.result()
                except Exception as exc:
                    prepare_failures[member.instance_id] = str(exc)

        if prepare_failures:
            self._finish_preparation_failure(
                run_id,
                workload,
                prepare_failures,
            )
            raise DeploymentResidentActionError(
                "concurrent workload client preparation failed"
            )

        results, barrier_release_ns = self._run_synchronized(
            workload,
            prepared,
            cancel_event=cancel_event,
        )
        return self._persist_phase_result(
            run_id,
            workload_case_id,
            workload,
            results,
            barrier_release_ns,
            identities,
        )

    def _run_synchronized(
        self,
        workload: ConcurrentWorkloadCase,
        prepared: dict[str, PreparedConcurrentClient],
        *,
        cancel_event: Event | None,
    ) -> tuple[tuple[ConcurrentMemberResult, ...], int]:
        release: dict[str, int] = {}

        def set_release() -> None:
            release["ns"] = self.clock()

        barrier = Barrier(len(workload.members) + 1, action=set_release)
        phase_cancel = Event()

        def worker(
            member: ConcurrentWorkloadMemberSpec,
        ) -> ConcurrentMemberResult:
            try:
                barrier.wait(timeout=member.timeout_seconds)
            except BrokenBarrierError:
                now = self.clock()
                return _synthetic_result(
                    member,
                    status="cancelled",
                    ready_ns=prepared[member.instance_id].client_ready_ns,
                    barrier_release_ns=release.get("ns", now),
                    failure="concurrent start barrier aborted",
                )
            return prepared[member.instance_id].run(
                barrier_release_ns=release["ns"],
                cancel_event=phase_cancel,
            )

        results: dict[str, ConcurrentMemberResult] = {}
        with ThreadPoolExecutor(
            max_workers=len(workload.members)
        ) as executor:
            futures: dict[Future[ConcurrentMemberResult], str] = {
                executor.submit(worker, member): member.instance_id
                for member in workload.members
            }
            try:
                barrier.wait(
                    timeout=max(
                        member.timeout_seconds
                        for member in workload.members
                    )
                )
            except BrokenBarrierError:
                phase_cancel.set()

            pending = set(futures)
            while pending:
                if cancel_event is not None and cancel_event.is_set():
                    phase_cancel.set()
                done, pending = wait(pending, timeout=0.05)
                for future in done:
                    instance_id = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        member = next(
                            item
                            for item in workload.members
                            if item.instance_id == instance_id
                        )
                        result = _synthetic_result(
                            member,
                            status="failed",
                            ready_ns=prepared[instance_id].client_ready_ns,
                            barrier_release_ns=release.get(
                                "ns",
                                self.clock(),
                            ),
                            failure=str(exc),
                        )
                    results[instance_id] = result
                    if result.status != "completed":
                        phase_cancel.set()

        release_ns = release.get("ns", self.clock())
        ordered = tuple(
            results.get(
                member.instance_id,
                _synthetic_result(
                    member,
                    status="cancelled",
                    ready_ns=prepared[member.instance_id].client_ready_ns,
                    barrier_release_ns=release_ns,
                    failure="concurrent phase did not return a result",
                ),
            )
            for member in workload.members
        )
        return ordered, release_ns

    def _persist_phase_result(
        self,
        run_id: str,
        workload_case_id: str,
        workload: ConcurrentWorkloadCase,
        results: tuple[ConcurrentMemberResult, ...],
        barrier_release_ns: int,
        identities: dict[str, _BaselineIdentity],
    ) -> ConcurrentPhaseSummary:
        status_failure = _member_failure(results)
        if status_failure is not None:
            quality, failure_kind, cancelled = status_failure
            self._persist_failed_phase(
                run_id,
                workload,
                results,
                barrier_release_ns,
                quality=quality,
                failure_kind=failure_kind,
            )
            raise DeploymentResidentActionError(
                f"concurrent {workload.phase} phase failed",
                failure_kind=(
                    "member_timeout"
                    if failure_kind == "member_timeout"
                    else "concurrent_workload_failed"
                ),
                cancelled=cancelled,
            )

        if any(not item.correctness_valid for item in results):
            self._persist_failed_phase(
                run_id,
                workload,
                results,
                barrier_release_ns,
                quality="correctness_invalid",
                failure_kind="output_validation_failed",
            )
            raise DeploymentResidentActionError(
                f"concurrent {workload.phase} output validation failed",
                failure_kind="output_validation_failed",
            )

        try:
            overlap, member_overlap = compute_overlap(results)
        except ConcurrentMetricError as exc:
            self._persist_failed_phase(
                run_id,
                workload,
                results,
                barrier_release_ns,
                quality="no_overlap",
                failure_kind="no_overlap",
                failure_details={"error": str(exc)},
            )
            raise DeploymentResidentActionError(
                f"concurrent {workload.phase} has no valid overlap"
            ) from exc

        overlap_by_instance = {
            item.instance_id: item for item in member_overlap
        }
        retentions = {}
        missing = False
        ambiguous = False
        with self.database.session() as connection:
            baseline_repo = StandaloneBaselineRepository(connection)
            for spec, result in zip(
                workload.members,
                results,
                strict=True,
            ):
                identity = identities[result.instance_id]
                matches = baseline_repo.lookup_exact(
                    candidate_id=identity.candidate_id,
                    resolved_placement_id=identity.resolved_placement_id,
                    host_id=identity.host_id,
                    binary_id=identity.binary_id,
                    mode=spec.mode,
                    prompt_tokens=spec.prompt_tokens,
                    generate_tokens=spec.generate_tokens,
                    depth_tokens=spec.depth_tokens,
                )
                if len(matches) == 0:
                    missing = True
                    continue
                if len(matches) > 1:
                    ambiguous = True
                    continue
                retentions[result.instance_id] = retention_for(
                    result,
                    baseline_id=matches[0].id,
                    standalone_tps=matches[0].throughput_tps,
                )

        quality: ConcurrentQuality = "clean"
        if ambiguous:
            quality = "baseline_ambiguous"
        elif missing:
            quality = "baseline_missing"
        min_retention = (
            min(item.retention for item in retentions.values())
            if len(retentions) == len(results)
            else None
        )

        with self.database.session() as connection:
            repository = ConcurrentWorkloadRepository(connection)
            for ordinal, (spec, result) in enumerate(
                zip(workload.members, results, strict=True)
            ):
                normalized = overlap_by_instance[result.instance_id]
                retention = retentions.get(result.instance_id)
                repository.add_member(
                    run_id,
                    ordinal=ordinal,
                    result=result,
                    overlap_prompt_tokens=normalized.prompt_tokens,
                    overlap_decode_tokens=normalized.decode_tokens,
                    overlap_prompt_tps=normalized.prompt_tps,
                    overlap_decode_tps=normalized.decode_tps,
                    standalone_baseline_id=(
                        None if retention is None else retention.baseline_id
                    ),
                    standalone_tps=(
                        None if retention is None else retention.standalone_tps
                    ),
                    retention=(
                        None if retention is None else retention.retention
                    ),
                    throughput_loss_pct=(
                        None
                        if retention is None
                        else retention.throughput_loss_pct
                    ),
                    failure_details=(
                        None
                        if result.failure is None
                        else {"error": result.failure}
                    ),
                )
            repository.finish_run(
                run_id,
                status="completed",
                quality=quality,
                correctness_valid=True,
                barrier_release_ns=barrier_release_ns,
                overlap_start_ns=overlap.overlap_start_ns,
                overlap_end_ns=overlap.overlap_end_ns,
                overlap_duration_ns=overlap.overlap_duration_ns,
                prompt_tokens=overlap.prompt_tokens,
                decode_tokens=overlap.decode_tokens,
                combined_prompt_tps=overlap.combined_prompt_tps,
                combined_decode_tps=overlap.combined_decode_tps,
                min_retention=min_retention,
            )

        return ConcurrentPhaseSummary(
            run_id=run_id,
            workload_case_id=workload_case_id,
            phase=workload.phase,
            quality=quality,
            combined_prompt_tps=overlap.combined_prompt_tps,
            combined_decode_tps=overlap.combined_decode_tps,
            min_retention=min_retention,
        )

    def _persist_failed_phase(
        self,
        run_id: str,
        workload: ConcurrentWorkloadCase,
        results: tuple[ConcurrentMemberResult, ...],
        barrier_release_ns: int,
        *,
        quality: ConcurrentQuality,
        failure_kind: str,
        failure_details: dict[str, object] | None = None,
    ) -> None:
        with self.database.session() as connection:
            repository = ConcurrentWorkloadRepository(connection)
            for ordinal, result in enumerate(results):
                repository.add_member(
                    run_id,
                    ordinal=ordinal,
                    result=result,
                    overlap_prompt_tokens=0,
                    overlap_decode_tokens=0,
                    overlap_prompt_tps=None,
                    overlap_decode_tps=None,
                    standalone_baseline_id=None,
                    standalone_tps=None,
                    retention=None,
                    throughput_loss_pct=None,
                    failure_details=(
                        None
                        if result.failure is None
                        else {"error": result.failure}
                    ),
                )
            repository.finish_run(
                run_id,
                status=(
                    "cancelled"
                    if quality == "member_failed"
                    and any(item.status == "cancelled" for item in results)
                    else "invalid"
                    if quality in {"no_overlap", "correctness_invalid"}
                    else "failed"
                ),
                quality=quality,
                correctness_valid=quality != "correctness_invalid",
                barrier_release_ns=barrier_release_ns,
                overlap_start_ns=None,
                overlap_end_ns=None,
                overlap_duration_ns=None,
                prompt_tokens=0,
                decode_tokens=0,
                combined_prompt_tps=None,
                combined_decode_tps=None,
                min_retention=None,
                failure_kind=failure_kind,
                failure_details=failure_details,
            )

    def _finish_preparation_failure(
        self,
        run_id: str,
        workload: ConcurrentWorkloadCase,
        failures: dict[str, str],
    ) -> None:
        now = self.clock()
        results = tuple(
            _synthetic_result(
                member,
                status=(
                    "failed"
                    if member.instance_id in failures
                    else "cancelled"
                ),
                ready_ns=now,
                barrier_release_ns=now,
                failure=failures.get(
                    member.instance_id,
                    "cancelled after peer preparation failure",
                ),
            )
            for member in workload.members
        )
        self._persist_failed_phase(
            run_id,
            workload,
            results,
            now,
            quality="member_failed",
            failure_kind="member_failed",
            failure_details={"preparation_failures": failures},
        )


def _member_failure(
    results: tuple[ConcurrentMemberResult, ...],
) -> tuple[ConcurrentQuality, str, bool] | None:
    if any(item.status == "timeout" for item in results):
        return "member_timeout", "member_timeout", False
    if any(item.status == "cancelled" for item in results):
        return "member_failed", "cancelled", True
    if any(item.status == "failed" for item in results):
        return "member_failed", "member_failed", False
    if any(item.status == "invalid" for item in results):
        return "correctness_invalid", "output_validation_failed", False
    return None


def _synthetic_result(
    member: ConcurrentWorkloadMemberSpec,
    *,
    status: str,
    ready_ns: int,
    barrier_release_ns: int,
    failure: str,
) -> ConcurrentMemberResult:
    now = max(barrier_release_ns, ready_ns)
    return ConcurrentMemberResult(
        instance_id=member.instance_id,
        mode=member.mode,
        status=status,
        client_ready_ns=min(ready_ns, barrier_release_ns),
        barrier_release_ns=barrier_release_ns,
        first_request_ns=now,
        finished_ns=now,
        correctness_valid=False,
        failure=failure,
        raw={"synthetic": True},
    )
