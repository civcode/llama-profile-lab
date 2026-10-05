"""FastAPI application factory for llama-profile-lab."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Never

from fastapi import FastAPI, Query, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from llama_profile_lab.analysis import (
    AnalysisError,
    CandidateComparison,
    DeploymentAnalysisError,
    LatencyEstimate,
    MatrixProjection,
    ParetoResult,
)
from llama_profile_lab.api.dto import (
    BinaryInspectRequest,
    BinaryListResponse,
    CandidateListResponse,
    CandidateValidationHistoryDTO,
    DeploymentCandidateListResponse,
    DeploymentCreateRequest,
    DeploymentDTO,
    DeploymentParetoResponse,
    DeploymentPlacementListResponse,
    DeploymentPlanRequest,
    DeploymentPlanResponse,
    DeploymentProgressDTO,
    DeploymentResultsResponse,
    DeploymentRunListResponse,
    DeploymentRunRequest,
    ExecutionRequest,
    ExperimentCloneRequest,
    ExperimentCreateRequest,
    ExperimentDTO,
    ExperimentListResponse,
    ExperimentPreviewRequest,
    ExperimentProgressDTO,
    HealthResponse,
    LauncherProfileDTO,
    MetricListResponse,
    ModelListResponse,
    ParameterListResponse,
    ParetoRequestDTO,
    PlacementDTO,
    PlacementListResponse,
    PlanPreviewDTO,
    PlanSummaryDTO,
    ProfileListResponse,
    PromotionRequest,
    PromotionResponse,
    ResultsResponse,
    RunDetailDTO,
    RunListResponse,
    ServerValidationRequest,
    ServerValidationResponse,
    TelemetryResponse,
)
from llama_profile_lab.api.deployment_operations import (
    DeploymentOperationManager,
)
from llama_profile_lab.api.operations import OperationError, OperationManager
from llama_profile_lab.api.profiles import LauncherProfileError, LauncherProfileProvider
from llama_profile_lab.api.service import (
    ApiConflictError,
    ApiNotFoundError,
    ApiService,
    parse_deployment_constraints,
    parse_deployment_filters,
    parse_deployment_objectives,
    parse_filters,
)
from llama_profile_lab.db import Database
from llama_profile_lab.execution import HostLockError, ServerValidationError
from llama_profile_lab.llama import BinaryDiscoveryError
from llama_profile_lab.planning import PlanningError
from llama_profile_lab.promotion import PromotionError


def create_app(
    database_path: str | Path = Path("data/benchmarks.db"),
    *,
    launcher_config_path: str | Path | None = None,
    operation_manager: OperationManager | None = None,
    deployment_operation_manager: DeploymentOperationManager | None = None,
    frontend_dist_path: str | Path | None = None,
) -> FastAPI:
    """Build a local API instance with explicit filesystem dependencies."""
    database = Database(Path(database_path))
    with database.session():
        pass

    configured_launcher = (
        None if launcher_config_path is None else Path(launcher_config_path)
    )
    profiles = LauncherProfileProvider(configured_launcher)
    operations = operation_manager or OperationManager(database)
    deployment_operations = (
        deployment_operation_manager
        or DeploymentOperationManager(database)
    )
    service = ApiService(
        database,
        profiles=profiles,
        operations=operations,
        deployment_operations=deployment_operations,
    )

    app = FastAPI(
        title="llama-profile-lab API",
        version="1",
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.database = database
    app.state.api_service = service
    app.state.operation_manager = operations
    app.state.deployment_operation_manager = deployment_operations

    _register_exception_handlers(app)
    _register_routes(app, service)

    frontend_dist = _resolve_frontend_dist(frontend_dist_path)
    if frontend_dist is not None:
        app.mount(
            "/",
            StaticFiles(directory=str(frontend_dist), html=True),
            name="frontend",
        )
        app.state.frontend_dist = frontend_dist
    return app


def create_default_app() -> FastAPI:
    """Create an app from environment variables for uvicorn --factory usage."""
    database = Path(os.environ.get("LLPROF_DATABASE", "data/benchmarks.db"))
    launcher_raw = os.environ.get("LLPROF_LAUNCHER_CONFIG")
    launcher = None if launcher_raw is None else Path(launcher_raw)
    frontend_raw = os.environ.get("LLPROF_FRONTEND_DIST")
    frontend = None if frontend_raw is None else Path(frontend_raw)
    return create_app(
        database,
        launcher_config_path=launcher,
        frontend_dist_path=frontend,
    )


def _resolve_frontend_dist(explicit: str | Path | None) -> Path | None:
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"frontend dist directory does not exist: {path}")
        if not (path / "index.html").is_file():
            raise ValueError(f"frontend dist is missing index.html: {path}")
        return path

    source_tree = Path(__file__).resolve().parents[3] / "frontend" / "dist"
    if source_tree.is_dir() and (source_tree / "index.html").is_file():
        return source_tree
    return None


def _register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApiNotFoundError, _not_found_handler)
    for exception_type in (
        ApiConflictError,
        OperationError,
        PlanningError,
        HostLockError,
        ServerValidationError,
        PromotionError,
    ):
        app.add_exception_handler(exception_type, _conflict_handler)
    app.add_exception_handler(AnalysisError, _bad_request_handler)
    app.add_exception_handler(DeploymentAnalysisError, _bad_request_handler)
    app.add_exception_handler(BinaryDiscoveryError, _bad_request_handler)
    app.add_exception_handler(LauncherProfileError, _unavailable_handler)


async def _not_found_handler(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


async def _conflict_handler(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


async def _bad_request_handler(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


async def _unavailable_handler(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


def _register_routes(app: FastAPI, service: ApiService) -> None:
    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(schema_version=service.health())

    @app.get("/api/parameters", response_model=ParameterListResponse)
    def parameters() -> ParameterListResponse:
        return service.list_parameters()

    @app.get("/api/metrics", response_model=MetricListResponse)
    def metrics() -> MetricListResponse:
        return service.list_metrics()

    @app.get("/api/profiles", response_model=ProfileListResponse)
    def profiles() -> ProfileListResponse:
        return service.list_profiles()

    @app.get("/api/profiles/{profile_id}", response_model=LauncherProfileDTO)
    def profile(profile_id: str) -> LauncherProfileDTO:
        return service.get_profile(profile_id)

    @app.get("/api/binaries", response_model=BinaryListResponse)
    def binaries() -> BinaryListResponse:
        return service.list_binaries()

    @app.post(
        "/api/binaries/inspect",
        response_model=BinaryListResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def inspect_binaries(request: BinaryInspectRequest) -> BinaryListResponse:
        return service.inspect_binaries(request)

    @app.get("/api/models", response_model=ModelListResponse)
    def models() -> ModelListResponse:
        return service.list_models()

    @app.post(
        "/api/experiments/preview",
        response_model=PlanPreviewDTO,
    )
    def preview_experiment(request: ExperimentPreviewRequest) -> PlanPreviewDTO:
        return service.preview_experiment(request)

    @app.post(
        "/api/experiments",
        response_model=ExperimentDTO,
        status_code=status.HTTP_201_CREATED,
    )
    def create_experiment(request: ExperimentCreateRequest) -> ExperimentDTO:
        return service.create_experiment(request)

    @app.get("/api/experiments", response_model=ExperimentListResponse)
    def experiments() -> ExperimentListResponse:
        return service.list_experiments()

    @app.get("/api/experiments/{experiment_id}", response_model=ExperimentDTO)
    def experiment(experiment_id: str) -> ExperimentDTO:
        return service.get_experiment(experiment_id)

    @app.post(
        "/api/experiments/{experiment_id}/clone",
        response_model=ExperimentDTO,
        status_code=status.HTTP_201_CREATED,
    )
    def clone_experiment(
        experiment_id: str,
        request: ExperimentCloneRequest,
    ) -> ExperimentDTO:
        return service.clone_experiment(experiment_id, name=request.name)

    @app.post(
        "/api/experiments/{experiment_id}/plan",
        response_model=PlanSummaryDTO,
    )
    def plan_experiment_route(experiment_id: str) -> PlanSummaryDTO:
        return service.plan(experiment_id)

    @app.post(
        "/api/experiments/{experiment_id}/run",
        response_model=ExperimentProgressDTO,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def run_experiment(
        experiment_id: str,
        request: ExecutionRequest,
    ) -> ExperimentProgressDTO:
        return service.start_execution(experiment_id, request, resume=False)

    @app.post(
        "/api/experiments/{experiment_id}/resume",
        response_model=ExperimentProgressDTO,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def resume_experiment(
        experiment_id: str,
        request: ExecutionRequest,
    ) -> ExperimentProgressDTO:
        return service.start_execution(experiment_id, request, resume=True)

    @app.post(
        "/api/experiments/{experiment_id}/pause",
        response_model=ExperimentProgressDTO,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def pause_experiment(experiment_id: str) -> ExperimentProgressDTO:
        return service.pause(experiment_id)

    @app.post(
        "/api/experiments/{experiment_id}/cancel",
        response_model=ExperimentProgressDTO,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def cancel_experiment(experiment_id: str) -> ExperimentProgressDTO:
        return service.cancel(experiment_id)

    @app.get(
        "/api/experiments/{experiment_id}/progress",
        response_model=ExperimentProgressDTO,
    )
    def experiment_progress(experiment_id: str) -> ExperimentProgressDTO:
        return service.progress(experiment_id)

    @app.get(
        "/api/experiments/{experiment_id}/events",
        response_class=StreamingResponse,
    )
    async def experiment_events(
        experiment_id: str,
        request: Request,
    ) -> StreamingResponse:
        service.get_experiment(experiment_id)
        return StreamingResponse(
            _progress_events(service, experiment_id, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get(
        "/api/experiments/{experiment_id}/candidates",
        response_model=CandidateListResponse,
    )
    def experiment_candidates(experiment_id: str) -> CandidateListResponse:
        return service.list_candidates(experiment_id)

    @app.get(
        "/api/experiments/{experiment_id}/runs",
        response_model=RunListResponse,
    )
    def experiment_runs(experiment_id: str) -> RunListResponse:
        return service.list_runs(experiment_id)

    @app.get(
        "/api/experiments/{experiment_id}/results",
        response_model=ResultsResponse,
    )
    def experiment_results(
        experiment_id: str,
        filters: Annotated[list[str] | None, Query(alias="filter")] = None,
        qualities: Annotated[list[str] | None, Query(alias="quality")] = None,
        metrics: Annotated[list[str] | None, Query(alias="metric")] = None,
    ) -> ResultsResponse:
        try:
            parsed = parse_filters(tuple(filters or ()))
        except ValueError as exc:
            return _raise_bad_request(exc)
        return service.results(
            experiment_id,
            filters=parsed,
            qualities=tuple(qualities or ()),
            metrics=None if metrics is None else tuple(metrics),
        )

    @app.get(
        "/api/experiments/{experiment_id}/matrix",
        response_model=MatrixProjection,
    )
    def experiment_matrix(
        experiment_id: str,
        x_path: Annotated[str, Query(alias="x")],
        y_path: Annotated[str, Query(alias="y")],
        metric: str,
        facet_path: Annotated[str | None, Query(alias="facet")] = None,
        filters: Annotated[list[str] | None, Query(alias="filter")] = None,
        qualities: Annotated[list[str] | None, Query(alias="quality")] = None,
    ) -> MatrixProjection:
        try:
            parsed = parse_filters(tuple(filters or ()))
        except ValueError as exc:
            return _raise_bad_request(exc)
        result = service.matrix(
            experiment_id,
            x_path=x_path,
            y_path=y_path,
            metric=metric,
            facet_path=facet_path,
            filters=parsed,
            qualities=tuple(qualities or ()),
        )
        if not isinstance(result, MatrixProjection):
            raise TypeError("analysis service returned non-matrix result")
        return result

    @app.get(
        "/api/experiments/{experiment_id}/compare",
        response_model=CandidateComparison,
    )
    def experiment_compare(
        experiment_id: str,
        candidate_id: str,
        metrics: Annotated[list[str], Query(alias="metric")],
        baseline_candidate_id: str | None = None,
        filters: Annotated[list[str] | None, Query(alias="filter")] = None,
        qualities: Annotated[list[str] | None, Query(alias="quality")] = None,
    ) -> CandidateComparison:
        try:
            parsed = parse_filters(tuple(filters or ()))
        except ValueError as exc:
            return _raise_bad_request(exc)
        if not metrics:
            return _raise_bad_request(ValueError("at least one metric is required"))
        return service.compare_candidate(
            experiment_id,
            candidate_id=candidate_id,
            metric_names=tuple(metrics),
            baseline_candidate_id=baseline_candidate_id,
            filters=parsed,
            qualities=tuple(qualities or ()),
        )

    @app.post(
        "/api/experiments/{experiment_id}/pareto",
        response_model=ParetoResult,
    )
    def experiment_pareto(
        experiment_id: str,
        request: ParetoRequestDTO,
    ) -> ParetoResult:
        try:
            return service.pareto(experiment_id, request)
        except ValueError as exc:
            return _raise_bad_request(exc)

    @app.get(
        "/api/experiments/{experiment_id}/latency",
        response_model=LatencyEstimate,
    )
    def experiment_latency(
        experiment_id: str,
        candidate_id: str,
        prompt_tokens: Annotated[int, Query(gt=0)],
        generate_tokens: Annotated[int, Query(gt=0)],
        decode_start_depth_tokens: Annotated[int | None, Query(ge=0)] = None,
        filters: Annotated[list[str] | None, Query(alias="filter")] = None,
        qualities: Annotated[list[str] | None, Query(alias="quality")] = None,
    ) -> LatencyEstimate:
        try:
            parsed = parse_filters(tuple(filters or ()))
        except ValueError as exc:
            return _raise_bad_request(exc)
        return service.latency(
            experiment_id,
            candidate_id=candidate_id,
            prompt_tokens=prompt_tokens,
            generate_tokens=generate_tokens,
            decode_start_depth_tokens=decode_start_depth_tokens,
            filters=parsed,
            qualities=tuple(qualities or ()),
        )

    @app.get(
        "/api/deployments",
        response_model=DeploymentListResponse,
    )
    def deployments() -> DeploymentListResponse:
        return service.list_deployments()

    @app.post(
        "/api/deployments",
        response_model=DeploymentDTO,
        status_code=status.HTTP_201_CREATED,
    )
    def create_deployment(
        request: DeploymentCreateRequest,
    ) -> DeploymentDTO:
        return service.create_deployment(request)

    @app.get(
        "/api/deployments/{deployment_id}",
        response_model=DeploymentDTO,
    )
    def deployment(deployment_id: str) -> DeploymentDTO:
        return service.get_deployment(deployment_id)

    @app.post(
        "/api/deployments/{deployment_id}/preview",
        response_model=DeploymentPlanResponse,
    )
    def preview_deployment(
        deployment_id: str,
        request: DeploymentPlanRequest,
    ) -> DeploymentPlanResponse:
        return service.preview_deployment(deployment_id, request)

    @app.post(
        "/api/deployments/{deployment_id}/plan",
        response_model=DeploymentPlanResponse,
    )
    def plan_deployment(
        deployment_id: str,
        request: DeploymentPlanRequest,
    ) -> DeploymentPlanResponse:
        return service.plan_deployment(deployment_id, request)

    @app.post(
        "/api/deployments/{deployment_id}/run",
        response_model=DeploymentProgressDTO,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def run_deployment(
        deployment_id: str,
        request: DeploymentRunRequest,
    ) -> DeploymentProgressDTO:
        return service.start_deployment(
            deployment_id,
            request,
            resume=False,
        )

    @app.post(
        "/api/deployments/{deployment_id}/resume",
        response_model=DeploymentProgressDTO,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def resume_deployment(
        deployment_id: str,
        request: DeploymentRunRequest | None = None,
    ) -> DeploymentProgressDTO:
        return service.start_deployment(
            deployment_id,
            request,
            resume=True,
        )

    @app.post(
        "/api/deployments/{deployment_id}/pause",
        response_model=DeploymentProgressDTO,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def pause_deployment(
        deployment_id: str,
    ) -> DeploymentProgressDTO:
        return service.pause_deployment(deployment_id)

    @app.post(
        "/api/deployments/{deployment_id}/cancel",
        response_model=DeploymentProgressDTO,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def cancel_deployment(
        deployment_id: str,
    ) -> DeploymentProgressDTO:
        return service.cancel_deployment(deployment_id)

    @app.get(
        "/api/deployments/{deployment_id}/progress",
        response_model=DeploymentProgressDTO,
    )
    def deployment_progress(
        deployment_id: str,
    ) -> DeploymentProgressDTO:
        return service.deployment_progress(deployment_id)

    @app.get(
        "/api/deployments/{deployment_id}/events",
        response_class=StreamingResponse,
    )
    async def deployment_events(
        deployment_id: str,
        request: Request,
    ) -> StreamingResponse:
        service.get_deployment(deployment_id)
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
        "/api/deployments/{deployment_id}/candidates",
        response_model=DeploymentCandidateListResponse,
    )
    def deployment_candidates(
        deployment_id: str,
    ) -> DeploymentCandidateListResponse:
        return service.list_deployment_candidates(deployment_id)

    @app.get(
        "/api/deployments/{deployment_id}/placements",
        response_model=DeploymentPlacementListResponse,
    )
    def deployment_placements(
        deployment_id: str,
    ) -> DeploymentPlacementListResponse:
        return service.list_deployment_placements(deployment_id)

    @app.get(
        "/api/deployments/{deployment_id}/runs",
        response_model=DeploymentRunListResponse,
    )
    def deployment_runs(
        deployment_id: str,
    ) -> DeploymentRunListResponse:
        return service.list_deployment_runs(deployment_id)

    @app.get(
        "/api/deployments/{deployment_id}/results",
        response_model=DeploymentResultsResponse,
    )
    def deployment_results(
        deployment_id: str,
        filters: Annotated[
            list[str] | None,
            Query(alias="filter"),
        ] = None,
    ) -> DeploymentResultsResponse:
        try:
            parsed = parse_deployment_filters(tuple(filters or ()))
        except ValueError as exc:
            return _raise_bad_request(exc)
        return service.deployment_results(
            deployment_id,
            filters=parsed,
        )

    @app.get(
        "/api/deployments/{deployment_id}/pareto",
        response_model=DeploymentParetoResponse,
    )
    def deployment_pareto(
        deployment_id: str,
        objectives: Annotated[list[str], Query(alias="objective")],
        constraints: Annotated[
            list[str] | None,
            Query(alias="constraint"),
        ] = None,
        filters: Annotated[
            list[str] | None,
            Query(alias="filter"),
        ] = None,
    ) -> DeploymentParetoResponse:
        try:
            parsed_objectives = parse_deployment_objectives(
                tuple(objectives)
            )
            parsed_constraints = parse_deployment_constraints(
                tuple(constraints or ())
            )
            parsed_filters = parse_deployment_filters(
                tuple(filters or ())
            )
        except ValueError as exc:
            return _raise_bad_request(exc)
        return service.deployment_pareto(
            deployment_id,
            objectives=parsed_objectives,
            constraints=parsed_constraints,
            filters=parsed_filters,
        )

    @app.get("/api/placements", response_model=PlacementListResponse)
    def placements() -> PlacementListResponse:
        return service.list_placements()

    @app.get("/api/placements/{placement_id}", response_model=PlacementDTO)
    def placement(placement_id: str) -> PlacementDTO:
        return service.get_placement(placement_id)

    @app.get(
        "/api/experiments/{experiment_id}/candidates/{candidate_id}/validation",
        response_model=CandidateValidationHistoryDTO,
    )
    def candidate_validation_history(
        experiment_id: str,
        candidate_id: str,
    ) -> CandidateValidationHistoryDTO:
        return service.validation_history(experiment_id, candidate_id)

    @app.get("/api/runs/{run_id}", response_model=RunDetailDTO)
    def run(run_id: str) -> RunDetailDTO:
        return service.get_run(run_id)

    @app.get("/api/runs/{run_id}/telemetry", response_model=TelemetryResponse)
    def run_telemetry(run_id: str) -> TelemetryResponse:
        return service.telemetry(run_id)

    @app.post(
        "/api/candidates/{candidate_id}/validate",
        response_model=ServerValidationResponse,
    )
    def validate_candidate(
        candidate_id: str,
        request: ServerValidationRequest,
    ) -> ServerValidationResponse:
        return service.validate_candidate(candidate_id, request)

    @app.post(
        "/api/candidates/{candidate_id}/promote",
        response_model=PromotionResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def promote_candidate(
        candidate_id: str,
        request: PromotionRequest,
    ) -> PromotionResponse:
        return service.promote_candidate(candidate_id, request)


async def _deployment_progress_events(
    service: ApiService,
    deployment_id: str,
    request: Request,
) -> AsyncIterator[str]:
    previous: str | None = None
    while True:
        progress = service.deployment_progress(deployment_id)
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
        if await request.is_disconnected():
            return
        await asyncio.sleep(0.25)


async def _progress_events(
    service: ApiService,
    experiment_id: str,
    request: Request,
) -> AsyncIterator[str]:
    previous: str | None = None
    while True:
        progress = service.progress(experiment_id)
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
        if await request.is_disconnected():
            return
        await asyncio.sleep(0.25)


def _raise_bad_request(exc: ValueError) -> Never:
    from fastapi import HTTPException

    raise HTTPException(status_code=400, detail=str(exc)) from exc
