"""HTTP boundary for V2 multi-model deployment planning and analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, status
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


class DeploymentApiService:
    """Stable application boundary over V2 planning, persistence, and analysis."""

    def __init__(self, database: Database) -> None:
        self.database = database
        self.planner = DeploymentPlannerService(database)
        self.analysis = DeploymentAnalysisService(database)

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
            counts = connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM deployment_instance
                     WHERE deployment_candidate_id = ?) AS instances,
                    (SELECT COUNT(*) FROM deployment_placement
                     WHERE deployment_candidate_id = ?) AS placements,
                    (SELECT COUNT(*) FROM deployment_run
                     WHERE deployment_candidate_id = ?) AS runs,
                    (SELECT COUNT(*) FROM deployment_rejection
                     WHERE deployment_candidate_id = ?) AS rejections
                """,
                (deployment_id, deployment_id, deployment_id, deployment_id),
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
            rows = connection.execute(
                """
                SELECT id FROM deployment_placement
                WHERE deployment_candidate_id = ?
                ORDER BY created_at, id
                """,
                (deployment_id,),
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
            if record is None or record.deployment_candidate_id != deployment_id:
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
            rows = connection.execute(
                """
                SELECT id FROM deployment_run
                WHERE deployment_candidate_id = ?
                ORDER BY created_at, id
                """,
                (deployment_id,),
            ).fetchall()
            items = tuple(
                self._run_dto(connection, repository, str(row["id"]))
                for row in rows
            )
        return DeploymentRunListResponse(items=items)

    def results(self, deployment_id: str) -> DeploymentResultsResponse:
        self.get(deployment_id)
        rendered = self.analysis.export(
            format_name="json",
            filters=(
                DeploymentAnalysisFilter(
                    path="deployment.candidate_id",
                    value=deployment_id,
                ),
            ),
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
        filters = (
            DeploymentAnalysisFilter(
                path="deployment.candidate_id",
                value=deployment_id,
            ),
            *request.filters,
        )
        try:
            return self.analysis.pareto(
                objectives=request.objectives,
                constraints=request.constraints,
                filters=filters,
            )
        except DeploymentAnalysisError as exc:
            raise ApiConflictError(str(exc)) from exc

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

    @app.post(
        "/api/deployments/{deployment_id}/pareto",
        response_model=DeploymentParetoResult,
    )
    def deployment_pareto(
        deployment_id: str,
        request: DeploymentParetoRequest,
    ) -> DeploymentParetoResult:
        return service.pareto(deployment_id, request)


def _json_object(raw: object) -> dict[str, Any]:
    value = json.loads(str(raw))
    if not isinstance(value, dict):
        raise RuntimeError("persisted deployment JSON value must be an object")
    return dict(value)
