"""FastAPI application factory for llama-profile-lab."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Query, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from llama_profile_lab.analysis import AnalysisError, MatrixProjection
from llama_profile_lab.api.dto import (
    BinaryInspectRequest,
    BinaryListResponse,
    CandidateListResponse,
    ExecutionRequest,
    ExperimentCloneRequest,
    ExperimentCreateRequest,
    ExperimentDTO,
    ExperimentListResponse,
    ExperimentProgressDTO,
    HealthResponse,
    LauncherProfileDTO,
    ModelListResponse,
    PlanSummaryDTO,
    ProfileListResponse,
    ResultsResponse,
    RunDetailDTO,
    RunListResponse,
    ServerValidationRequest,
    ServerValidationResponse,
    TelemetryResponse,
)
from llama_profile_lab.api.operations import OperationError, OperationManager
from llama_profile_lab.api.profiles import LauncherProfileError, LauncherProfileProvider
from llama_profile_lab.api.service import (
    ApiConflictError,
    ApiNotFoundError,
    ApiService,
    parse_filters,
)
from llama_profile_lab.db import Database
from llama_profile_lab.execution import HostLockError, ServerValidationError
from llama_profile_lab.llama import BinaryDiscoveryError
from llama_profile_lab.planning import PlanningError


def create_app(
    database_path: str | Path = Path("data/benchmarks.db"),
    *,
    launcher_config_path: str | Path | None = None,
    operation_manager: OperationManager | None = None,
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
    service = ApiService(database, profiles=profiles, operations=operations)

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

    _register_exception_handlers(app)
    _register_routes(app, service)
    return app


def create_default_app() -> FastAPI:
    """Create an app from environment variables for uvicorn --factory usage."""
    database = Path(os.environ.get("LLPROF_DATABASE", "data/benchmarks.db"))
    launcher_raw = os.environ.get("LLPROF_LAUNCHER_CONFIG")
    launcher = None if launcher_raw is None else Path(launcher_raw)
    return create_app(database, launcher_config_path=launcher)


def _register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApiNotFoundError, _not_found_handler)
    for exception_type in (
        ApiConflictError,
        OperationError,
        PlanningError,
        HostLockError,
        ServerValidationError,
    ):
        app.add_exception_handler(exception_type, _conflict_handler)
    app.add_exception_handler(AnalysisError, _bad_request_handler)
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


def _raise_bad_request(exc: ValueError) -> object:
    from fastapi import HTTPException

    raise HTTPException(status_code=400, detail=str(exc)) from exc
