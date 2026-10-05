"""HTTP boundary for V2 multi-model deployment planning and analysis."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import Field, NonNegativeInt

from llama_profile_lab.analysis import (
    DeploymentAnalysisError,
    DeploymentAnalysisFilter,
    DeploymentAnalysisService,
    DeploymentMemoryMatrix,
    DeploymentMetricConstraint,
    DeploymentParetoObjective,
    DeploymentParetoResult,
)
from llama_profile_lab.api.deployment_operations import (
    DeploymentExecutionSpec,
    DeploymentOperationError,
    DeploymentOperationManager,
    DeploymentOperationSnapshot,
)
from llama_profile_lab.api.dto import ApiModel
from llama_profile_lab.api.service import ApiConflictError, ApiNotFoundError
from llama_profile_lab.db import (
    Database,
    DeploymentCandidateRepository,
    DeploymentPlacementRepository,
    DeploymentRunRepository,
)
from llama_profile_lab.domain import (
    DeploymentCandidate,
    DeploymentPlacement,
    DeploymentSearchSpace,
)
from llama_profile_lab.execution import (
    DeploymentServerInput,
    StandaloneBaselineInput,
)
from llama_profile_lab.planning import (
    DeploymentEstimatorInput,
    DeploymentPlannerService,
)


class DeploymentEstimatorInputDTO(ApiModel):
    """Estimator inputs for one deployment instance."""

    instance_id: Annotated[str, Field(min_length=1)]
    helper_binary_id: Annotated[str, Field(min_length=1)]
    model_path: Annotated[str, Field(min_length=1)]


class DeploymentCreateRequest(ApiModel):
    """Create one immutable deployment Candidate."""

    deployment: DeploymentCandidate


class DeploymentPlanRequest(ApiModel):
    """Plan placements for one persisted deployment Candidate."""

    search_space: DeploymentSearchSpace
    instances: Annotated[
        tuple[DeploymentEstimatorInputDTO, ...],
        Field(min_length=1),
    ]
    timeout_seconds: Annotated[float, Field(gt=0)] | None = 300.0


class DeploymentDTO(ApiModel):
    """Application-facing deployment summary."""

    id: str
    deployment_hash: str
    created_at: str
    instance_count: NonNegativeInt
    placement_count: NonNegativeInt
    run_count: NonNegativeInt
    rejection_count: NonNegativeInt
    definition: DeploymentCandidate


class DeploymentCandidateCaseDTO(ApiModel):
    """One feasible generated deployment Candidate/placement pair."""

    ordinal: NonNegativeInt
    deployment_candidate_id: str
    deployment_placement_id: str
    generation: dict[str, Any]


class DeploymentCandidateListResponse(ApiModel):
    items: tuple[DeploymentCandidateCaseDTO, ...]


class DeploymentPlanSummaryDTO(ApiModel):
    base_deployment_candidate_id: str
    host_id: str
    raw_combinations: NonNegativeInt
    rejected_by_constraints: NonNegativeInt
    duplicate_candidates: NonNegativeInt
    symmetry_reduced: NonNegativeInt
    capability_rejected: NonNegativeInt
    estimate_failed: NonNegativeInt
    memory_rejected: NonNegativeInt
    valid_count: NonNegativeInt
    plan_id: str | None
    cases: tuple[DeploymentCandidateCaseDTO, ...]


class DeploymentMemoryDTO(ApiModel):
    instance_id: str
    device_id: str
    model_bytes: NonNegativeInt
    context_bytes: NonNegativeInt
    compute_bytes: NonNegativeInt
    total_bytes: NonNegativeInt
    device_total_bytes: NonNegativeInt
    device_free_bytes: NonNegativeInt
    source: str
    measured_at: str | None


class DeploymentAllocationDTO(ApiModel):
    device_id: str
    projected_bytes: NonNegativeInt
    reserved_margin_bytes: NonNegativeInt
    device_total_bytes: NonNegativeInt
    projected_free_bytes: NonNegativeInt


class DeploymentPlacementDTO(ApiModel):
    id: str
    deployment_candidate_id: str
    host_id: str
    feasibility: str
    request: dict[str, Any]
    provenance: dict[str, Any]
    created_at: str
    placement: DeploymentPlacement
    memory: tuple[DeploymentMemoryDTO, ...]
    allocations: tuple[DeploymentAllocationDTO, ...]


class DeploymentPlacementListResponse(ApiModel):
    items: tuple[DeploymentPlacementDTO, ...]


class DeploymentRunMemberDTO(ApiModel):
    instance_id: str
    endpoint: str | None
    status: str
    pid: int | None
    started_at: str | None
    ready_at: str | None
    finished_at: str | None
    exit_code: int | None
    forced_kill: bool
    cleanup_error: str | None
    result: dict[str, Any]


class DeploymentRunDTO(ApiModel):
    id: str
    deployment_candidate_id: str
    deployment_placement_id: str | None
    workload_case_id: str | None
    status: str
    quality: str | None
    quality_details: dict[str, Any] | None
    failure_kind: str | None
    failure_details: dict[str, Any] | None
    started_at: str | None
    finished_at: str | None
    duration_ns: NonNegativeInt | None
    created_at: str
    current_workload_phase: str | None
    members: tuple[DeploymentRunMemberDTO, ...]


class DeploymentRunListResponse(ApiModel):
    items: tuple[DeploymentRunDTO, ...]


class DeploymentResultsResponse(ApiModel):
    deployment_candidate_id: str
    rows: tuple[dict[str, Any], ...]


class DeploymentParetoRequest(ApiModel):
    objectives: Annotated[
        tuple[DeploymentParetoObjective, ...],
        Field(min_length=1),
    ]
    constraints: tuple[DeploymentMetricConstraint, ...] = ()
    filters: tuple[DeploymentAnalysisFilter, ...] = ()


class DeploymentServerInputDTO(ApiModel):
    """Filesystem inputs required for one deployment server."""

    instance_id: Annotated[str, Field(min_length=1)]
    model_path: Annotated[str, Field(min_length=1)]
    draft_model_path: Annotated[str, Field(min_length=1)] | None = None


class StandaloneBaselineInputDTO(ApiModel):
    """Exact standalone denominator supplied to concurrent execution."""

    instance_id: Annotated[str, Field(min_length=1)]
    mode: Literal["prefill", "decode"]
    prompt_tokens: NonNegativeInt
    generate_tokens: NonNegativeInt
    depth_tokens: NonNegativeInt
    throughput_tps: Annotated[float, Field(gt=0)]
    latency_ms: Annotated[float, Field(ge=0)] | None = None


class DeploymentRunRequest(ApiModel):
    """Start or replay one persisted deployment placement."""

    deployment_placement_id: Annotated[str, Field(min_length=1)]
    instances: Annotated[
        tuple[DeploymentServerInputDTO, ...],
        Field(min_length=1),
    ]
    standalone_baselines: tuple[StandaloneBaselineInputDTO, ...] = ()
    host: Annotated[str, Field(min_length=1)] = "127.0.0.1"
    readiness_timeout_seconds: Annotated[float, Field(gt=0)] = 300.0


class DeploymentOperationDTO(ApiModel):
    id: str
    deployment_candidate_id: str
    status: str
    started_at: str
    finished_at: str | None
    requested_action: str | None
    deployment_run_id: str | None
    deployment_placement_id: str
    phase_count: NonNegativeInt | None
    error: str | None


class DeploymentProgressDTO(ApiModel):
    deployment_candidate_id: str
    deployment_status: str
    planned_candidates: NonNegativeInt
    completed_candidates: NonNegativeInt
    failed_candidates: NonNegativeInt
    active_deployment_run: str | None
    member_states: tuple[DeploymentRunMemberDTO, ...]
    current_workload_phase: str | None
    operation: DeploymentOperationDTO | None


class DeploymentApiService:
    """Stable application boundary over V2 planning, persistence, and analysis."""

    def __init__(
        self,
        database: Database,
        *,
        operations: DeploymentOperationManager | None = None,
    ) -> None:
        self.database = database
        self.planner = DeploymentPlannerService(database)
        self.analysis = DeploymentAnalysisService(database)
        self.operations = operations or DeploymentOperationManager(database)

    def create(self, request: DeploymentCreateRequest) -> DeploymentDTO:
        try:
            with self.database.session() as connection:
                identifier = DeploymentCandidateRepository(connection).put(
                    request.deployment
                )
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise ApiConflictError(str(exc)) from exc
        return self.get(identifier)

    def get(self, deployment_id: str) -> DeploymentDTO:
        with self.database.session() as connection:
            repository = DeploymentCandidateRepository(connection)
            record = repository.record(deployment_id)
            definition = repository.get(deployment_id)
            if record is None or definition is None:
                raise ApiNotFoundError(
                    f"deployment Candidate not found: {deployment_id}"
                )
            scope = self._scope_ids(connection, deployment_id)
            placeholders = ",".join("?" for _ in scope)
            counts = connection.execute(
                f"""
                SELECT
                    (SELECT COUNT(*) FROM deployment_instance
                     WHERE deployment_candidate_id = ?) AS instances,
                    (SELECT COUNT(*) FROM deployment_placement
                     WHERE deployment_candidate_id IN ({placeholders}))
                        AS placements,
                    (SELECT COUNT(*) FROM deployment_run
                     WHERE deployment_candidate_id IN ({placeholders}))
                        AS runs,
                    (SELECT COUNT(*) FROM deployment_rejection
                     WHERE deployment_candidate_id IN ({placeholders}))
                        AS rejections
                """,
                (
                    deployment_id,
                    *scope,
                    *scope,
                    *scope,
                ),
            ).fetchone()
        if counts is None:
            raise RuntimeError("deployment count query returned no row")
        return DeploymentDTO(
            id=record.id,
            deployment_hash=record.deployment_hash,
            created_at=record.created_at,
            instance_count=int(counts["instances"]),
            placement_count=int(counts["placements"]),
            run_count=int(counts["runs"]),
            rejection_count=int(counts["rejections"]),
            definition=definition,
        )

    def plan(
        self,
        deployment_id: str,
        request: DeploymentPlanRequest,
    ) -> DeploymentPlanSummaryDTO:
        self.get(deployment_id)
        try:
            summary = self.planner.plan(
                deployment_id,
                request.search_space,
                tuple(
                    DeploymentEstimatorInput(
                        instance_id=item.instance_id,
                        helper_binary_id=item.helper_binary_id,
                        model_path=Path(item.model_path),
                    )
                    for item in request.instances
                ),
                timeout_seconds=request.timeout_seconds,
            )
        except ValueError as exc:
            raise ApiConflictError(str(exc)) from exc
        return DeploymentPlanSummaryDTO(
            base_deployment_candidate_id=summary.base_deployment_candidate_id,
            host_id=summary.host_id,
            raw_combinations=summary.raw_combinations,
            rejected_by_constraints=summary.rejected_by_constraints,
            duplicate_candidates=summary.duplicate_candidates,
            symmetry_reduced=summary.symmetry_reduced,
            capability_rejected=summary.capability_rejected,
            estimate_failed=summary.estimate_failed,
            memory_rejected=summary.memory_rejected,
            valid_count=summary.valid_count,
            plan_id=summary.plan_id,
            cases=tuple(
                DeploymentCandidateCaseDTO(
                    ordinal=index,
                    deployment_candidate_id=item.deployment_candidate_id,
                    deployment_placement_id=item.deployment_placement_id,
                    generation=dict(item.generation),
                )
                for index, item in enumerate(summary.cases)
            ),
        )

    def candidates(
        self,
        deployment_id: str,
    ) -> DeploymentCandidateListResponse:
        self.get(deployment_id)
        with self.database.session() as connection:
            rows = connection.execute(
                """
                SELECT dpc.ordinal, dpc.deployment_candidate_id,
                       dpc.deployment_placement_id, dpc.generation_json
                FROM deployment_plan_case AS dpc
                JOIN deployment_plan AS dp ON dp.id = dpc.deployment_plan_id
                WHERE dp.base_deployment_candidate_id = ?
                ORDER BY dp.created_at, dp.id, dpc.ordinal
                """,
                (deployment_id,),
            ).fetchall()
        return DeploymentCandidateListResponse(
            items=tuple(
                DeploymentCandidateCaseDTO(
                    ordinal=int(row["ordinal"]),
                    deployment_candidate_id=str(row["deployment_candidate_id"]),
                    deployment_placement_id=str(row["deployment_placement_id"]),
                    generation=_json_object(row["generation_json"]),
                )
                for row in rows
            )
        )

    def placements(
        self,
        deployment_id: str,
    ) -> DeploymentPlacementListResponse:
        self.get(deployment_id)
        with self.database.session() as connection:
            repository = DeploymentPlacementRepository(connection)
            scope = self._scope_ids(connection, deployment_id)
            placeholders = ",".join("?" for _ in scope)
            rows = connection.execute(
                f"""
                SELECT id FROM deployment_placement
                WHERE deployment_candidate_id IN ({placeholders})
                ORDER BY created_at, id
                """,
                scope,
            ).fetchall()
            items = tuple(
                self._placement_dto(repository, str(row["id"]))
                for row in rows
            )
        return DeploymentPlacementListResponse(items=items)

    def placement(
        self,
        deployment_id: str,
        placement_id: str,
    ) -> DeploymentPlacementDTO:
        self.get(deployment_id)
        with self.database.session() as connection:
            repository = DeploymentPlacementRepository(connection)
            record = repository.record(placement_id)
            scope = self._scope_ids(connection, deployment_id)
            if (
                record is None
                or record.deployment_candidate_id not in scope
            ):
                raise ApiNotFoundError(
                    f"deployment placement not found: {placement_id}"
                )
            return self._placement_dto(repository, placement_id)

    def memory_matrix(
        self,
        deployment_id: str,
        placement_id: str,
        *,
        deployment_run_id: str | None = None,
    ) -> DeploymentMemoryMatrix:
        self.placement(deployment_id, placement_id)
        try:
            return self.analysis.memory_matrix(
                placement_id,
                deployment_run_id=deployment_run_id,
            )
        except DeploymentAnalysisError as exc:
            raise ApiConflictError(str(exc)) from exc

    def runs(self, deployment_id: str) -> DeploymentRunListResponse:
        self.get(deployment_id)
        with self.database.session() as connection:
            repository = DeploymentRunRepository(connection)
            scope = self._scope_ids(connection, deployment_id)
            placeholders = ",".join("?" for _ in scope)
            rows = connection.execute(
                f"""
                SELECT id FROM deployment_run
                WHERE deployment_candidate_id IN ({placeholders})
                ORDER BY created_at, id
                """,
                scope,
            ).fetchall()
            items = tuple(
                self._run_dto(connection, repository, str(row["id"]))
                for row in rows
            )
        return DeploymentRunListResponse(items=items)

    def results(self, deployment_id: str) -> DeploymentResultsResponse:
        self.get(deployment_id)
        with self.database.session() as connection:
            scope = self._scope_ids(connection, deployment_id)
        rendered = self.analysis.export(
            format_name="json",
            deployment_candidate_ids=scope,
        )
        raw = json.loads(rendered)
        if not isinstance(raw, list) or any(
            not isinstance(item, dict) for item in raw
        ):
            raise RuntimeError("deployment analysis returned invalid JSON export")
        return DeploymentResultsResponse(
            deployment_candidate_id=deployment_id,
            rows=tuple(dict(item) for item in raw),
        )

    def pareto(
        self,
        deployment_id: str,
        request: DeploymentParetoRequest,
    ) -> DeploymentParetoResult:
        self.get(deployment_id)
        with self.database.session() as connection:
            scope = self._scope_ids(connection, deployment_id)
        try:
            return self.analysis.pareto(
                objectives=request.objectives,
                constraints=request.constraints,
                filters=request.filters,
                deployment_candidate_ids=scope,
            )
        except DeploymentAnalysisError as exc:
            raise ApiConflictError(str(exc)) from exc

    def run(
        self,
        deployment_id: str,
        request: DeploymentRunRequest,
    ) -> DeploymentProgressDTO:
        self.get(deployment_id)
        spec = self._execution_spec(deployment_id, request)
        try:
            self.operations.start(deployment_id, spec=spec)
        except DeploymentOperationError as exc:
            raise ApiConflictError(str(exc)) from exc
        return self.progress(deployment_id)

    def resume(
        self,
        deployment_id: str,
        request: DeploymentRunRequest | None = None,
    ) -> DeploymentProgressDTO:
        self.get(deployment_id)
        spec = (
            None
            if request is None
            else self._execution_spec(deployment_id, request)
        )
        try:
            self.operations.resume(deployment_id, spec=spec)
        except DeploymentOperationError as exc:
            raise ApiConflictError(str(exc)) from exc
        return self.progress(deployment_id)

    def pause(self, deployment_id: str) -> DeploymentProgressDTO:
        self.get(deployment_id)
        try:
            self.operations.pause(deployment_id)
        except DeploymentOperationError as exc:
            raise ApiConflictError(str(exc)) from exc
        return self.progress(deployment_id)

    def cancel(self, deployment_id: str) -> DeploymentProgressDTO:
        self.get(deployment_id)
        try:
            self.operations.cancel(deployment_id)
        except DeploymentOperationError as exc:
            raise ApiConflictError(str(exc)) from exc
        return self.progress(deployment_id)

    def progress(self, deployment_id: str) -> DeploymentProgressDTO:
        self.get(deployment_id)
        operation = self.operations.snapshot(deployment_id)
        with self.database.session() as connection:
            scope = self._scope_ids(connection, deployment_id)
            placeholders = ",".join("?" for _ in scope)
            counts = connection.execute(
                f"""
                SELECT
                    (SELECT COUNT(DISTINCT dpc.deployment_candidate_id)
                     FROM deployment_plan_case AS dpc
                     JOIN deployment_plan AS dp
                       ON dp.id = dpc.deployment_plan_id
                     WHERE dp.base_deployment_candidate_id = ?) AS planned,
                    (SELECT COUNT(DISTINCT deployment_placement_id)
                     FROM deployment_run
                     WHERE deployment_candidate_id IN ({placeholders})
                       AND status = 'completed') AS completed,
                    (SELECT COUNT(DISTINCT deployment_placement_id)
                     FROM deployment_run
                     WHERE deployment_candidate_id IN ({placeholders})
                       AND status = 'failed') AS failed
                """,
                (deployment_id, *scope, *scope),
            ).fetchone()
            run_row = connection.execute(
                f"""
                SELECT id, status
                FROM deployment_run
                WHERE deployment_candidate_id IN ({placeholders})
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                scope,
            ).fetchone()
            run_id = None if run_row is None else str(run_row["id"])
            phase_row = (
                None
                if run_id is None
                else connection.execute(
                    """
                    SELECT phase
                    FROM deployment_workload_run
                    WHERE deployment_run_id = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """,
                    (run_id,),
                ).fetchone()
            )
            members = (
                ()
                if run_id is None
                else self._run_members(
                    DeploymentRunRepository(connection),
                    run_id,
                )
            )

        if counts is None:
            raise RuntimeError("deployment progress count query returned no row")
        if operation is not None:
            deployment_status = operation.status
        elif run_row is not None:
            deployment_status = str(run_row["status"])
        elif int(counts["planned"]) > 0:
            deployment_status = "planned"
        else:
            deployment_status = "draft"

        active_run = None
        if run_row is not None and str(run_row["status"]) in {
            "starting",
            "ready",
            "running",
        }:
            active_run = run_id
        return DeploymentProgressDTO(
            deployment_candidate_id=deployment_id,
            deployment_status=deployment_status,
            planned_candidates=int(counts["planned"]),
            completed_candidates=int(counts["completed"]),
            failed_candidates=int(counts["failed"]),
            active_deployment_run=active_run,
            member_states=members,
            current_workload_phase=(
                None if phase_row is None else str(phase_row["phase"])
            ),
            operation=_operation_dto(operation),
        )

    @staticmethod
    def _scope_ids(
        connection: Any,
        deployment_id: str,
    ) -> tuple[str, ...]:
        rows = connection.execute(
            """
            SELECT DISTINCT dpc.deployment_candidate_id
            FROM deployment_plan_case AS dpc
            JOIN deployment_plan AS dp
              ON dp.id = dpc.deployment_plan_id
            WHERE dp.base_deployment_candidate_id = ?
            ORDER BY dpc.deployment_candidate_id
            """,
            (deployment_id,),
        ).fetchall()
        derived = tuple(
            str(row["deployment_candidate_id"])
            for row in rows
            if str(row["deployment_candidate_id"]) != deployment_id
        )
        return (deployment_id, *derived)

    def _execution_spec(
        self,
        deployment_id: str,
        request: DeploymentRunRequest,
    ) -> DeploymentExecutionSpec:
        placement = self.placement(
            deployment_id,
            request.deployment_placement_id,
        )
        if placement.feasibility != "feasible":
            raise ApiConflictError(
                "only feasible deployment placements can be executed"
            )
        return DeploymentExecutionSpec(
            deployment_placement_id=request.deployment_placement_id,
            inputs=tuple(
                DeploymentServerInput(
                    instance_id=item.instance_id,
                    model_path=Path(item.model_path),
                    draft_model_path=(
                        None
                        if item.draft_model_path is None
                        else Path(item.draft_model_path)
                    ),
                )
                for item in request.instances
            ),
            standalone_baselines=tuple(
                StandaloneBaselineInput(
                    instance_id=item.instance_id,
                    mode=item.mode,
                    prompt_tokens=item.prompt_tokens,
                    generate_tokens=item.generate_tokens,
                    depth_tokens=item.depth_tokens,
                    throughput_tps=item.throughput_tps,
                    latency_ms=item.latency_ms,
                )
                for item in request.standalone_baselines
            ),
            host=request.host,
            readiness_timeout_seconds=request.readiness_timeout_seconds,
        )

    @staticmethod
    def _run_members(
        repository: DeploymentRunRepository,
        run_id: str,
    ) -> tuple[DeploymentRunMemberDTO, ...]:
        return tuple(
            DeploymentRunMemberDTO(
                instance_id=item.instance_id,
                endpoint=item.endpoint,
                status=item.member_status,
                pid=item.pid,
                started_at=item.started_at,
                ready_at=item.ready_at,
                finished_at=item.finished_at,
                exit_code=item.exit_code,
                forced_kill=item.forced_kill,
                cleanup_error=item.cleanup_error,
                result=dict(item.result),
            )
            for item in repository.members(run_id)
        )

    @staticmethod
    def _placement_dto(
        repository: DeploymentPlacementRepository,
        placement_id: str,
    ) -> DeploymentPlacementDTO:
        record = repository.record(placement_id)
        placement = repository.get(placement_id)
        if record is None or placement is None:
            raise ApiNotFoundError(
                f"deployment placement not found: {placement_id}"
            )
        return DeploymentPlacementDTO(
            id=record.id,
            deployment_candidate_id=record.deployment_candidate_id,
            host_id=record.host_id,
            feasibility=record.feasibility,
            request=dict(record.request),
            provenance=dict(record.provenance),
            created_at=record.created_at,
            placement=placement,
            memory=tuple(
                DeploymentMemoryDTO(
                    instance_id=item.instance_id,
                    device_id=item.device_id,
                    model_bytes=item.model_bytes,
                    context_bytes=item.context_bytes,
                    compute_bytes=item.compute_bytes,
                    total_bytes=item.total_bytes,
                    device_total_bytes=item.device_total_bytes,
                    device_free_bytes=item.device_free_bytes,
                    source=item.source,
                    measured_at=item.measured_at,
                )
                for item in repository.memory(placement_id)
            ),
            allocations=tuple(
                DeploymentAllocationDTO(
                    device_id=item.device_id,
                    projected_bytes=item.projected_bytes,
                    reserved_margin_bytes=item.reserved_margin_bytes,
                    device_total_bytes=item.device_total_bytes,
                    projected_free_bytes=item.projected_free_bytes,
                )
                for item in repository.allocations(placement_id)
            ),
        )

    @staticmethod
    def _run_dto(
        connection: Any,
        repository: DeploymentRunRepository,
        run_id: str,
    ) -> DeploymentRunDTO:
        record = repository.get(run_id)
        if record is None:
            raise ApiNotFoundError(f"deployment run not found: {run_id}")
        phase_row = connection.execute(
            """
            SELECT phase FROM deployment_workload_run
            WHERE deployment_run_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
        return DeploymentRunDTO(
            id=record.id,
            deployment_candidate_id=record.deployment_candidate_id,
            deployment_placement_id=record.deployment_placement_id,
            workload_case_id=record.workload_case_id,
            status=record.status,
            quality=record.quality,
            quality_details=(
                None
                if record.quality_details is None
                else dict(record.quality_details)
            ),
            failure_kind=record.failure_kind,
            failure_details=(
                None
                if record.failure_details is None
                else dict(record.failure_details)
            ),
            started_at=record.started_at,
            finished_at=record.finished_at,
            duration_ns=record.duration_ns,
            created_at=record.created_at,
            current_workload_phase=(
                None if phase_row is None else str(phase_row["phase"])
            ),
            members=tuple(
                DeploymentRunMemberDTO(
                    instance_id=item.instance_id,
                    endpoint=item.endpoint,
                    status=item.member_status,
                    pid=item.pid,
                    started_at=item.started_at,
                    ready_at=item.ready_at,
                    finished_at=item.finished_at,
                    exit_code=item.exit_code,
                    forced_kill=item.forced_kill,
                    cleanup_error=item.cleanup_error,
                    result=dict(item.result),
                )
                for item in repository.members(run_id)
            ),
        )


def register_deployment_routes(
    app: FastAPI,
    service: DeploymentApiService,
) -> None:
    """Register the V2 deployment HTTP resources without changing V1 routes."""

    @app.post(
        "/api/deployments",
        response_model=DeploymentDTO,
        status_code=status.HTTP_201_CREATED,
    )
    def create_deployment(request: DeploymentCreateRequest) -> DeploymentDTO:
        return service.create(request)

    @app.get("/api/deployments/{deployment_id}", response_model=DeploymentDTO)
    def get_deployment(deployment_id: str) -> DeploymentDTO:
        return service.get(deployment_id)

    @app.post(
        "/api/deployments/{deployment_id}/plan",
        response_model=DeploymentPlanSummaryDTO,
    )
    def plan_deployment(
        deployment_id: str,
        request: DeploymentPlanRequest,
    ) -> DeploymentPlanSummaryDTO:
        return service.plan(deployment_id, request)

    @app.get(
        "/api/deployments/{deployment_id}/candidates",
        response_model=DeploymentCandidateListResponse,
    )
    def deployment_candidates(
        deployment_id: str,
    ) -> DeploymentCandidateListResponse:
        return service.candidates(deployment_id)

    @app.get(
        "/api/deployments/{deployment_id}/placements",
        response_model=DeploymentPlacementListResponse,
    )
    def deployment_placements(
        deployment_id: str,
    ) -> DeploymentPlacementListResponse:
        return service.placements(deployment_id)

    @app.get(
        "/api/deployments/{deployment_id}/placements/{placement_id}",
        response_model=DeploymentPlacementDTO,
    )
    def deployment_placement(
        deployment_id: str,
        placement_id: str,
    ) -> DeploymentPlacementDTO:
        return service.placement(deployment_id, placement_id)

    @app.get(
        "/api/deployments/{deployment_id}/placements/{placement_id}/memory",
        response_model=DeploymentMemoryMatrix,
    )
    def deployment_memory(
        deployment_id: str,
        placement_id: str,
        deployment_run_id: str | None = None,
    ) -> DeploymentMemoryMatrix:
        return service.memory_matrix(
            deployment_id,
            placement_id,
            deployment_run_id=deployment_run_id,
        )

    @app.post(
        "/api/deployments/{deployment_id}/run",
        response_model=DeploymentProgressDTO,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def run_deployment(
        deployment_id: str,
        request: DeploymentRunRequest,
    ) -> DeploymentProgressDTO:
        return service.run(deployment_id, request)

    @app.post(
        "/api/deployments/{deployment_id}/pause",
        response_model=DeploymentProgressDTO,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def pause_deployment(deployment_id: str) -> DeploymentProgressDTO:
        return service.pause(deployment_id)

    @app.post(
        "/api/deployments/{deployment_id}/resume",
        response_model=DeploymentProgressDTO,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def resume_deployment(
        deployment_id: str,
        request: DeploymentRunRequest | None = None,
    ) -> DeploymentProgressDTO:
        return service.resume(deployment_id, request)

    @app.post(
        "/api/deployments/{deployment_id}/cancel",
        response_model=DeploymentProgressDTO,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def cancel_deployment(deployment_id: str) -> DeploymentProgressDTO:
        return service.cancel(deployment_id)

    @app.get(
        "/api/deployments/{deployment_id}/progress",
        response_model=DeploymentProgressDTO,
    )
    def deployment_progress(deployment_id: str) -> DeploymentProgressDTO:
        return service.progress(deployment_id)

    @app.get(
        "/api/deployments/{deployment_id}/events",
        response_class=StreamingResponse,
    )
    async def deployment_events(
        deployment_id: str,
        request: Request,
    ) -> StreamingResponse:
        service.get(deployment_id)
        return StreamingResponse(
            _deployment_progress_events(
                service,
                deployment_id,
                request,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get(
        "/api/deployments/{deployment_id}/runs",
        response_model=DeploymentRunListResponse,
    )
    def deployment_runs(deployment_id: str) -> DeploymentRunListResponse:
        return service.runs(deployment_id)

    @app.get(
        "/api/deployments/{deployment_id}/results",
        response_model=DeploymentResultsResponse,
    )
    def deployment_results(deployment_id: str) -> DeploymentResultsResponse:
        return service.results(deployment_id)

    @app.get(
        "/api/deployments/{deployment_id}/pareto",
        response_model=DeploymentParetoResult,
    )
    def deployment_pareto_get(
        deployment_id: str,
        objectives: Annotated[list[str], Query(alias="objective")],
        constraints: Annotated[
            list[str] | None,
            Query(alias="constraint"),
        ] = None,
        filters: Annotated[list[str] | None, Query(alias="filter")] = None,
    ) -> DeploymentParetoResult:
        try:
            request = DeploymentParetoRequest(
                objectives=tuple(
                    _parse_pareto_objective(value)
                    for value in objectives
                ),
                constraints=tuple(
                    _parse_metric_constraint(value)
                    for value in constraints or []
                ),
                filters=tuple(
                    _parse_analysis_filter(value)
                    for value in filters or []
                ),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return service.pareto(deployment_id, request)

    @app.post(
        "/api/deployments/{deployment_id}/pareto",
        response_model=DeploymentParetoResult,
    )
    def deployment_pareto(
        deployment_id: str,
        request: DeploymentParetoRequest,
    ) -> DeploymentParetoResult:
        return service.pareto(deployment_id, request)


def _parse_analysis_filter(value: str) -> DeploymentAnalysisFilter:
    path, separator, raw = value.partition("=")
    if not separator or not path:
        raise ValueError(f"invalid deployment filter {value!r}; expected PATH=VALUE")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw
    if parsed is not None and not isinstance(
        parsed,
        (str, int, float, bool),
    ):
        raise ValueError("deployment filter values must be JSON scalars")
    return DeploymentAnalysisFilter(path=path, value=parsed)


def _parse_pareto_objective(value: str) -> DeploymentParetoObjective:
    key, separator, remainder = value.partition(":")
    direction, second, metric = remainder.partition(":")
    if (
        not separator
        or not second
        or not key
        or direction not in {"maximize", "minimize", "max", "min"}
        or not metric
    ):
        raise ValueError(
            "invalid objective; expected KEY:DIRECTION:METRIC"
        )
    normalized = (
        "maximize" if direction in {"maximize", "max"} else "minimize"
    )
    return DeploymentParetoObjective(
        key=key,
        direction=normalized,
        metric=metric,
    )


def _parse_metric_constraint(value: str) -> DeploymentMetricConstraint:
    metric, separator, remainder = value.partition(":")
    operator, second, raw_value = remainder.partition(":")
    if (
        not separator
        or not second
        or not metric
        or operator not in {"ge", "gt", "le", "lt", "eq"}
    ):
        raise ValueError(
            "invalid constraint; expected METRIC:OPERATOR:VALUE"
        )
    try:
        threshold = float(raw_value)
    except ValueError as exc:
        raise ValueError("constraint VALUE must be numeric") from exc
    return DeploymentMetricConstraint(
        metric=metric,
        operator=operator,
        value=threshold,
    )


async def _deployment_progress_events(
    service: DeploymentApiService,
    deployment_id: str,
    request: Request,
):
    previous: str | None = None
    while not await request.is_disconnected():
        progress = service.progress(deployment_id)
        payload = progress.model_dump_json()
        if payload != previous:
            yield f"event: progress\ndata: {payload}\n\n"
            previous = payload
        operation = progress.operation
        if operation is None or operation.status in {
            "completed",
            "paused",
            "cancelled",
            "failed",
        }:
            return
        await asyncio.sleep(0.25)


def _operation_dto(
    snapshot: DeploymentOperationSnapshot | None,
) -> DeploymentOperationDTO | None:
    if snapshot is None:
        return None
    return DeploymentOperationDTO(
        id=snapshot.id,
        deployment_candidate_id=snapshot.deployment_candidate_id,
        status=snapshot.status,
        started_at=snapshot.started_at,
        finished_at=snapshot.finished_at,
        requested_action=snapshot.requested_action,
        deployment_run_id=snapshot.deployment_run_id,
        deployment_placement_id=snapshot.deployment_placement_id,
        phase_count=snapshot.phase_count,
        error=snapshot.error,
    )


def _json_object(raw: object) -> dict[str, Any]:
    value = json.loads(str(raw))
    if not isinstance(value, dict):
        raise RuntimeError("persisted deployment JSON value must be an object")
    return dict(value)
