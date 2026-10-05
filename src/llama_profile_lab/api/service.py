"""Application-facing services backing the local HTTP API."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from statistics import fmean

from llama_profile_lab.analysis import (
    DEFAULT_METRIC_REGISTRY,
    DeploymentAnalysisFilter,
    DeploymentAnalysisService,
    DeploymentMetricConstraint,
    DeploymentParetoObjective,
    DeploymentParetoResult,
    AnalysisFilter,
    AnalysisService,
    CandidateComparison,
    LatencyEstimate,
    MatrixProjection,
    ParetoResult,
)
from llama_profile_lab.api.dto import (
    BenchmarkSampleDTO,
    BinaryDTO,
    BinaryInspectRequest,
    BinaryListResponse,
    CandidateDTO,
    CandidateEvaluationDTO,
    CandidateListResponse,
    CandidateValidationHistoryDTO,
    DeploymentCandidateItemDTO,
    DeploymentCandidateListResponse,
    DeploymentCreateRequest,
    DeploymentDTO,
    DeploymentMemberStateDTO,
    DeploymentOperationDTO,
    DeploymentParetoResponse,
    DeploymentPlacementDTO,
    DeploymentPlacementListResponse,
    DeploymentPlanCaseDTO,
    DeploymentPlanRequest,
    DeploymentPlanResponse,
    DeploymentProgressDTO,
    DeploymentResultsResponse,
    DeploymentRunDTO,
    DeploymentRunListResponse,
    DeploymentRunMemberDTO,
    DeploymentRunRequest,
    DeploymentWorkloadPhaseDTO,
    ExecutionRequest,
    ExecutionSummaryDTO,
    ExperimentCreateRequest,
    ExperimentDTO,
    ExperimentListResponse,
    ExperimentPreviewRequest,
    ExperimentProgressDTO,
    LauncherArgChangeDTO,
    LauncherProfileDTO,
    MetricDefinitionDTO,
    MetricListResponse,
    ModelDTO,
    ModelFileDTO,
    ModelListResponse,
    OperationDTO,
    ParameterDefinitionDTO,
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
    RunSummaryDTO,
    ServerBenchmarkDTO,
    ServerValidationRequest,
    ServerValidationResponse,
    TelemetryResponse,
)
from llama_profile_lab.api.deployment_operations import (
    DeploymentOperationError,
    DeploymentOperationManager,
    DeploymentOperationSnapshot,
    DeploymentOperationSpec,
)
from llama_profile_lab.api.operations import (
    ExecutionSpec,
    OperationManager,
    OperationSnapshot,
)
from llama_profile_lab.api.profiles import LauncherProfile, LauncherProfileProvider
from llama_profile_lab.db import (
    BenchmarkCaseRepository,
    BenchmarkRunRepository,
    ConcurrentWorkloadRepository,
    DeploymentCandidateRepository,
    DeploymentPlanRepository,
    DeploymentPlacementRepository,
    DeploymentRunRepository,
    CandidateRepository,
    Database,
    EnvironmentRepository,
    ExperimentRepository,
    MeasurementPolicyRepository,
    PlacementRepository,
    SearchSpaceRepository,
    TelemetryRepository,
    WorkloadSuiteRepository,
    schema_version,
    transaction,
)
from llama_profile_lab.db.records import (
    BinaryRecord,
    ExperimentRecord,
    ResolvedPlacementRecord,
)
from llama_profile_lab.domain import (
    CandidateBaseline,
    ExperimentDefinition,
    FixedPlacementPolicy,
)
from llama_profile_lab.domain.base import JsonScalar
from llama_profile_lab.execution import (
    DeploymentServerInput,
    ServerValidationService,
    StandaloneBaselineInput,
)
from llama_profile_lab.llama import BinaryKind, probe_binary
from llama_profile_lab.planning import (
    DEFAULT_PARAMETER_REGISTRY,
    DeploymentEstimatorInput,
    DeploymentPlannerService,
    PlanSummary,
    build_plan,
    plan_experiment,
)
from llama_profile_lab.promotion import PromotionService


class ApiNotFoundError(RuntimeError):
    """Raised when an API resource cannot be found."""


class ApiConflictError(RuntimeError):
    """Raised when a requested API state transition is not valid."""


class ApiService:
    """Stable DTO boundary over the existing planning/execution/analysis services."""

    def __init__(
        self,
        database: Database,
        *,
        profiles: LauncherProfileProvider,
        operations: OperationManager,
        deployment_operations: DeploymentOperationManager | None = None,
    ) -> None:
        self.database = database
        self.profiles = profiles
        self.operations = operations
        self.deployment_operations = (
            deployment_operations
            or DeploymentOperationManager(database)
        )

    def health(self) -> int:
        with self.database.session() as connection:
            return schema_version(connection)

    def list_parameters(self) -> ParameterListResponse:
        return ParameterListResponse(
            items=tuple(
                ParameterDefinitionDTO(
                    path=definition.path,
                    label=definition.label,
                    category=definition.category,
                    value_types=tuple(
                        "null" if item is type(None) else item.__name__
                        for item in definition.python_types
                    ),
                    cli_argument=definition.cli_argument,
                    affects_placement=definition.affects_placement,
                    supported_by=tuple(sorted(definition.supported_by)),
                    minimum=definition.minimum,
                    maximum=definition.maximum,
                    string_choices=(
                        None
                        if definition.string_choices is None
                        else tuple(sorted(definition.string_choices))
                    ),
                )
                for definition in DEFAULT_PARAMETER_REGISTRY.definitions()
            )
        )

    def list_metrics(self) -> MetricListResponse:
        return MetricListResponse(
            items=tuple(
                MetricDefinitionDTO(
                    name=definition.name,
                    label=definition.label,
                    unit=definition.unit,
                )
                for definition in DEFAULT_METRIC_REGISTRY.definitions()
            )
        )

    def list_profiles(self) -> ProfileListResponse:
        items = tuple(_profile_dto(item) for item in self.profiles.list())
        return ProfileListResponse(
            configured=self.profiles.configured,
            source_path=(
                None
                if self.profiles.config_path is None
                else str(self.profiles.config_path)
            ),
            items=items,
        )

    def get_profile(self, profile_id: str) -> LauncherProfileDTO:
        profile = self.profiles.get(profile_id)
        if profile is None:
            if not self.profiles.configured:
                raise ApiNotFoundError("launcher profile registry is not configured")
            raise ApiNotFoundError(f"launcher profile not found: {profile_id}")
        return _profile_dto(profile)

    def list_binaries(self) -> BinaryListResponse:
        with self.database.session() as connection:
            records = EnvironmentRepository(connection).list_binaries()
        return BinaryListResponse(items=tuple(_binary_dto(record) for record in records))

    def inspect_binaries(self, request: BinaryInspectRequest) -> BinaryListResponse:
        kind: BinaryKind | None = None if request.kind == "auto" else request.kind
        records: list[BinaryRecord] = []
        with self.database.session() as connection:
            repository = EnvironmentRepository(connection)
            for raw_path in request.paths:
                probe = probe_binary(Path(raw_path), kind=kind)
                identifier = repository.put_binary(
                    sha256=probe.sha256,
                    kind=probe.kind,
                    path=str(probe.path),
                    size_bytes=probe.size_bytes,
                    mtime_ns=probe.mtime_ns,
                    git_commit=probe.git_commit,
                    build_number=probe.build_number,
                    build_info=probe.build_info_mapping(),
                    capabilities=probe.capabilities.to_mapping(),
                )
                record = repository.get_binary(identifier)
                if record is None:
                    raise RuntimeError("binary registration did not produce a record")
                records.append(record)
        return BinaryListResponse(items=tuple(_binary_dto(record) for record in records))

    def list_models(self) -> ModelListResponse:
        with self.database.session() as connection:
            rows = connection.execute(
                """
                SELECT id, identity_hash, architecture, parameter_count,
                       quantization, size_bytes, metadata_json, created_at
                FROM model
                ORDER BY created_at, id
                """
            ).fetchall()
            items: list[ModelDTO] = []
            for row in rows:
                files = connection.execute(
                    """
                    SELECT id, part_index, path, sha256, size_bytes
                    FROM model_file
                    WHERE model_id = ?
                    ORDER BY part_index
                    """,
                    (row["id"],),
                ).fetchall()
                items.append(
                    ModelDTO(
                        id=str(row["id"]),
                        identity_hash=str(row["identity_hash"]),
                        architecture=row["architecture"],
                        parameter_count=(
                            None
                            if row["parameter_count"] is None
                            else int(row["parameter_count"])
                        ),
                        quantization=row["quantization"],
                        size_bytes=int(row["size_bytes"]),
                        metadata=_json_object(str(row["metadata_json"])),
                        created_at=str(row["created_at"]),
                        files=tuple(
                            ModelFileDTO(
                                id=str(item["id"]),
                                part_index=int(item["part_index"]),
                                path=str(item["path"]),
                                sha256=str(item["sha256"]),
                                size_bytes=int(item["size_bytes"]),
                            )
                            for item in files
                        ),
                    )
                )
        return ModelListResponse(items=tuple(items))

    def preview_experiment(self, request: ExperimentPreviewRequest) -> PlanPreviewDTO:
        plan = build_plan(
            request.base_candidate,
            request.search_space,
            request.workload_suite,
        )
        workload_counts = {len(candidate.workloads) for candidate in plan.candidates}
        workloads_per_candidate = (
            workload_counts.pop() if len(workload_counts) == 1 else None
        )
        unique_workloads = {
            workload.case.content_hash()
            for candidate in plan.candidates
            for workload in candidate.workloads
        }
        return PlanPreviewDTO(
            raw_combinations=plan.search.raw_combinations,
            rejected_by_constraints=plan.search.rejected_by_constraints,
            duplicate_candidates=plan.search.duplicate_candidates,
            candidate_count=len(plan.candidates),
            workloads_per_candidate=workloads_per_candidate,
            benchmark_case_count=plan.benchmark_case_count,
            unique_workload_count=len(unique_workloads),
        )

    def create_experiment(self, request: ExperimentCreateRequest) -> ExperimentDTO:
        with self.database.session() as connection:
            with transaction(connection, immediate=True):
                candidates = CandidateRepository(connection)
                searches = SearchSpaceRepository(connection)
                suites = WorkloadSuiteRepository(connection)
                policies = MeasurementPolicyRepository(connection)
                experiments = ExperimentRepository(connection)

                base_candidate_id = candidates.put(request.base_candidate)
                search_space_id = searches.put(request.search_space)
                workload_suite_id = suites.put(request.workload_suite)
                measurement_policy_id = policies.put(request.measurement_policy)

                if isinstance(request.placement_policy, FixedPlacementPolicy):
                    placement = PlacementRepository(connection).get(
                        request.placement_policy.placement_id
                    )
                    if placement is None:
                        raise ApiNotFoundError(
                            "fixed placement not found: "
                            f"{request.placement_policy.placement_id}"
                        )
                if isinstance(request.baseline, CandidateBaseline):
                    baseline = candidates.get(request.baseline.candidate_id)
                    if baseline is None:
                        raise ApiNotFoundError(
                            f"baseline Candidate not found: {request.baseline.candidate_id}"
                        )

                definition = ExperimentDefinition(
                    name=request.name,
                    base_candidate_id=base_candidate_id,
                    search_space_id=search_space_id,
                    workload_suite_id=workload_suite_id,
                    measurement_policy_id=measurement_policy_id,
                    placement_policy=request.placement_policy,
                    baseline=request.baseline,
                )
                experiment_id = experiments.create(definition)
        return self.get_experiment(experiment_id)

    def list_experiments(self) -> ExperimentListResponse:
        with self.database.session() as connection:
            records = ExperimentRepository(connection).list()
            items = tuple(
                self._experiment_dto(connection, record)
                for record in records
            )
        return ExperimentListResponse(items=items)

    def get_experiment(self, experiment_id: str) -> ExperimentDTO:
        with self.database.session() as connection:
            record = ExperimentRepository(connection).get(experiment_id)
            if record is None:
                raise ApiNotFoundError(f"experiment not found: {experiment_id}")
            return self._experiment_dto(connection, record)

    def clone_experiment(
        self,
        experiment_id: str,
        *,
        name: str | None,
    ) -> ExperimentDTO:
        with self.database.session() as connection:
            experiments = ExperimentRepository(connection)
            definition = experiments.get_definition(experiment_id)
            if definition is None:
                raise ApiNotFoundError(f"experiment not found: {experiment_id}")
            clone = definition.model_copy(
                update={"name": name or f"{definition.name} (copy)"}
            )
            clone_id = experiments.create(clone)
        return self.get_experiment(clone_id)

    def plan(self, experiment_id: str) -> PlanSummaryDTO:
        with self.database.session() as connection:
            summary = plan_experiment(connection, experiment_id)
        return _plan_dto(summary)

    def start_execution(
        self,
        experiment_id: str,
        request: ExecutionRequest,
        *,
        resume: bool,
    ) -> ExperimentProgressDTO:
        self._require_experiment(experiment_id)
        spec = ExecutionSpec(
            binary_id=request.binary_id,
            model_path=Path(request.model_path),
            fit_binary_id=request.fit_binary_id,
            timeout_seconds=request.timeout_seconds,
            fit_timeout_seconds=request.fit_timeout_seconds,
            limit=request.limit,
            telemetry_interval_seconds=request.telemetry_interval_ms / 1000.0,
        )
        self.operations.start(experiment_id, spec=spec, resume=resume)
        return self.progress(experiment_id)

    def pause(self, experiment_id: str) -> ExperimentProgressDTO:
        self.operations.pause(experiment_id)
        return self.progress(experiment_id)

    def cancel(self, experiment_id: str) -> ExperimentProgressDTO:
        self.operations.cancel(experiment_id)
        return self.progress(experiment_id)

    def progress(self, experiment_id: str) -> ExperimentProgressDTO:
        with self.database.session() as connection:
            experiments = ExperimentRepository(connection)
            record = experiments.get(experiment_id)
            if record is None:
                raise ApiNotFoundError(f"experiment not found: {experiment_id}")
            row = connection.execute(
                """
                SELECT COUNT(*) AS total
                FROM benchmark_case
                WHERE experiment_id = ?
                """,
                (experiment_id,),
            ).fetchone()
            total = 0 if row is None else int(row["total"])
            incomplete = BenchmarkCaseRepository(connection).count_incomplete(
                experiment_id
            )
            status_rows = connection.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM benchmark_case
                WHERE experiment_id = ?
                GROUP BY status
                ORDER BY status
                """,
                (experiment_id,),
            ).fetchall()
            counts = {
                str(item["status"]): int(item["count"])
                for item in status_rows
            }
            running = connection.execute(
                """
                SELECT bc.candidate_id, bc.workload_case_id,
                       ec.ordinal AS candidate_ordinal,
                       ew.suite_case_index
                FROM benchmark_case AS bc
                JOIN experiment_candidate AS ec
                  ON ec.experiment_id = bc.experiment_id
                 AND ec.candidate_id = bc.candidate_id
                JOIN experiment_workload AS ew
                  ON ew.experiment_id = bc.experiment_id
                 AND ew.candidate_id = bc.candidate_id
                 AND ew.workload_case_id = bc.workload_case_id
                WHERE bc.experiment_id = ? AND bc.status = 'running'
                ORDER BY bc.ordinal, ew.suite_case_index
                LIMIT 1
                """,
                (experiment_id,),
            ).fetchone()
            latest = connection.execute(
                """
                SELECT br.id
                FROM benchmark_run AS br
                JOIN benchmark_case AS bc ON bc.id = br.benchmark_case_id
                WHERE bc.experiment_id = ?
                  AND br.status = 'completed'
                ORDER BY br.finished_at DESC, br.started_at DESC, br.id DESC
                LIMIT 1
                """,
                (experiment_id,),
            ).fetchone()
            latest_run_id = None if latest is None else str(latest["id"])
            latest_metrics: dict[str, int | float] = {}
            latest_tokens_per_second: float | None = None
            if latest_run_id is not None:
                latest_metrics = BenchmarkRunRepository(connection).metrics(latest_run_id)
                raw_ts = latest_metrics.get("avg_ts")
                if raw_ts is not None:
                    latest_tokens_per_second = float(raw_ts)
                else:
                    samples = BenchmarkRunRepository(connection).samples(latest_run_id)
                    if samples:
                        latest_tokens_per_second = fmean(
                            float(tokens_per_second)
                            for _, _, tokens_per_second in samples
                        )

        return ExperimentProgressDTO(
            experiment_id=experiment_id,
            experiment_status=record.status,
            total_cases=total,
            completed_cases=max(0, total - incomplete),
            incomplete_cases=incomplete,
            case_status_counts=counts,
            operation=_operation_dto(self.operations.snapshot(experiment_id)),
            current_candidate_id=(
                None if running is None else str(running["candidate_id"])
            ),
            current_candidate_ordinal=(
                None if running is None else int(running["candidate_ordinal"])
            ),
            current_workload_case_id=(
                None if running is None else str(running["workload_case_id"])
            ),
            current_suite_case_index=(
                None if running is None else int(running["suite_case_index"])
            ),
            latest_run_id=latest_run_id,
            latest_tokens_per_second=latest_tokens_per_second,
            latest_metrics=latest_metrics,
        )

    def list_candidates(self, experiment_id: str) -> CandidateListResponse:
        with self.database.session() as connection:
            self._require_experiment_connection(connection, experiment_id)
            rows = connection.execute(
                """
                SELECT candidate_id, ordinal, generation_metadata_json
                FROM experiment_candidate
                WHERE experiment_id = ?
                ORDER BY ordinal
                """,
                (experiment_id,),
            ).fetchall()
            candidates = CandidateRepository(connection)
            items: list[CandidateDTO] = []
            for row in rows:
                candidate_id = str(row["candidate_id"])
                candidate = candidates.get(candidate_id)
                if candidate is None:
                    raise RuntimeError(
                        f"experiment references missing Candidate {candidate_id}"
                    )
                counts = connection.execute(
                    """
                    SELECT
                        (
                            SELECT COUNT(*)
                            FROM experiment_workload
                            WHERE experiment_id = ? AND candidate_id = ?
                        ) AS workloads,
                        (
                            SELECT COUNT(*)
                            FROM benchmark_case
                            WHERE experiment_id = ? AND candidate_id = ?
                        ) AS cases,
                        (
                            SELECT COUNT(*)
                            FROM benchmark_case AS bc
                            WHERE bc.experiment_id = ?
                              AND bc.candidate_id = ?
                              AND EXISTS (
                                  SELECT 1
                                  FROM benchmark_run AS br
                                  WHERE br.benchmark_case_id = bc.id
                                    AND br.status = 'completed'
                              )
                        ) AS completed,
                        (
                            SELECT COUNT(*)
                            FROM server_run
                            WHERE experiment_id = ? AND candidate_id = ?
                              AND status = 'completed'
                        ) AS validations
                    """,
                    (
                        experiment_id,
                        candidate_id,
                        experiment_id,
                        candidate_id,
                        experiment_id,
                        candidate_id,
                        experiment_id,
                        candidate_id,
                    ),
                ).fetchone()
                if counts is None:
                    raise RuntimeError("candidate count query returned no row")
                items.append(
                    CandidateDTO(
                        id=candidate_id,
                        ordinal=int(row["ordinal"]),
                        generation_metadata=_json_object(
                            str(row["generation_metadata_json"])
                        ),
                        candidate=candidate,
                        workload_count=int(counts["workloads"]),
                        benchmark_case_count=int(counts["cases"]),
                        completed_case_count=int(counts["completed"]),
                        server_validation_count=int(counts["validations"]),
                    )
                )
        return CandidateListResponse(items=tuple(items))

    def list_runs(self, experiment_id: str) -> RunListResponse:
        with self.database.session() as connection:
            self._require_experiment_connection(connection, experiment_id)
            repository = BenchmarkRunRepository(connection)
            rows = connection.execute(
                """
                SELECT br.id, bc.candidate_id, bc.workload_case_id,
                       bc.placement_id, wc.kind
                FROM benchmark_run AS br
                JOIN benchmark_case AS bc ON bc.id = br.benchmark_case_id
                JOIN workload_case AS wc ON wc.id = bc.workload_case_id
                WHERE bc.experiment_id = ?
                ORDER BY br.started_at, br.id
                """,
                (experiment_id,),
            ).fetchall()
            items: list[RunSummaryDTO] = []
            for row in rows:
                record = repository.get(str(row["id"]))
                if record is None:
                    raise RuntimeError("run disappeared while listing experiment")
                items.append(
                    _run_summary_dto(
                        record=record,
                        candidate_id=str(row["candidate_id"]),
                        workload_case_id=str(row["workload_case_id"]),
                        placement_id=row["placement_id"],
                        workload_kind=str(row["kind"]),
                    )
                )
        return RunListResponse(items=tuple(items))

    def get_run(self, run_id: str) -> RunDetailDTO:
        with self.database.session() as connection:
            runs = BenchmarkRunRepository(connection)
            record = runs.get(run_id)
            if record is None:
                raise ApiNotFoundError(f"run not found: {run_id}")
            row = connection.execute(
                """
                SELECT bc.candidate_id, bc.workload_case_id,
                       bc.placement_id, wc.kind
                FROM benchmark_case AS bc
                JOIN workload_case AS wc ON wc.id = bc.workload_case_id
                WHERE bc.id = ?
                """,
                (record.benchmark_case_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("run references missing benchmark case")
            summary = _run_summary_dto(
                record=record,
                candidate_id=str(row["candidate_id"]),
                workload_case_id=str(row["workload_case_id"]),
                placement_id=row["placement_id"],
                workload_kind=str(row["kind"]),
            )
            stdout, stderr = runs.logs(run_id)
            return RunDetailDTO(
                **summary.model_dump(),
                samples=tuple(
                    BenchmarkSampleDTO(
                        sample_index=index,
                        elapsed_ns=elapsed_ns,
                        tokens_per_second=tokens_per_second,
                    )
                    for index, elapsed_ns, tokens_per_second in runs.samples(run_id)
                ),
                metrics=runs.metrics(run_id),
                stdout=stdout,
                stderr=stderr,
            )

    def telemetry(self, run_id: str) -> TelemetryResponse:
        with self.database.session() as connection:
            if BenchmarkRunRepository(connection).get(run_id) is None:
                raise ApiNotFoundError(f"run not found: {run_id}")
            samples = TelemetryRepository(connection).samples(run_id)
        return TelemetryResponse(run_id=run_id, samples=samples)

    def results(
        self,
        experiment_id: str,
        *,
        filters: tuple[AnalysisFilter, ...],
        qualities: tuple[str, ...],
        metrics: tuple[str, ...] | None,
    ) -> ResultsResponse:
        self._require_experiment(experiment_id)
        rows = AnalysisService(self.database).export_rows(
            experiment_id,
            filters=filters,
            qualities=qualities,
            metric_names=metrics,
        )
        return ResultsResponse(experiment_id=experiment_id, rows=rows)

    def matrix(
        self,
        experiment_id: str,
        *,
        x_path: str,
        y_path: str,
        metric: str,
        facet_path: str | None,
        filters: tuple[AnalysisFilter, ...],
        qualities: tuple[str, ...],
    ) -> MatrixProjection:
        self._require_experiment(experiment_id)
        return AnalysisService(self.database).matrix(
            experiment_id,
            x_path=x_path,
            y_path=y_path,
            metric=metric,
            facet_path=facet_path,
            filters=filters,
            qualities=qualities,
        )

    def list_placements(self) -> PlacementListResponse:
        with self.database.session() as connection:
            records = PlacementRepository(connection).list()
        return PlacementListResponse(
            items=tuple(_placement_dto(record) for record in records)
        )

    def get_placement(self, placement_id: str) -> PlacementDTO:
        with self.database.session() as connection:
            record = PlacementRepository(connection).get(placement_id)
        if record is None:
            raise ApiNotFoundError(f"placement not found: {placement_id}")
        return _placement_dto(record)

    def compare_candidate(
        self,
        experiment_id: str,
        *,
        candidate_id: str,
        metric_names: tuple[str, ...],
        baseline_candidate_id: str | None,
        filters: tuple[AnalysisFilter, ...],
        qualities: tuple[str, ...],
    ) -> CandidateComparison:
        self._require_experiment(experiment_id)
        return AnalysisService(self.database).compare(
            experiment_id,
            candidate_id=candidate_id,
            metric_names=metric_names,
            filters=filters,
            qualities=qualities,
            baseline_candidate_id=baseline_candidate_id,
        )

    def pareto(
        self,
        experiment_id: str,
        request: ParetoRequestDTO,
    ) -> ParetoResult:
        self._require_experiment(experiment_id)
        return AnalysisService(self.database).pareto(
            experiment_id,
            objectives=request.objectives,
            filters=parse_filters(request.filters),
            qualities=request.qualities,
        )

    def latency(
        self,
        experiment_id: str,
        *,
        candidate_id: str,
        prompt_tokens: int,
        generate_tokens: int,
        decode_start_depth_tokens: int | None,
        filters: tuple[AnalysisFilter, ...],
        qualities: tuple[str, ...],
    ) -> LatencyEstimate:
        self._require_experiment(experiment_id)
        return AnalysisService(self.database).latency(
            experiment_id,
            candidate_id=candidate_id,
            prompt_tokens=prompt_tokens,
            generate_tokens=generate_tokens,
            decode_start_depth_tokens=decode_start_depth_tokens,
            filters=filters,
            qualities=qualities,
        )

    def validation_history(
        self,
        experiment_id: str,
        candidate_id: str,
    ) -> CandidateValidationHistoryDTO:
        from llama_profile_lab.db import ServerValidationRepository

        with self.database.session() as connection:
            self._require_experiment_connection(connection, experiment_id)
            if CandidateRepository(connection).get(candidate_id) is None:
                raise ApiNotFoundError(f"Candidate not found: {candidate_id}")
            repository = ServerValidationRepository(connection)
            evaluations = repository.evaluations(
                experiment_id=experiment_id,
                candidate_id=candidate_id,
            )
            benchmarks = repository.benchmarks_for_candidate(
                experiment_id=experiment_id,
                candidate_id=candidate_id,
                completed_only=False,
            )

        return CandidateValidationHistoryDTO(
            experiment_id=experiment_id,
            candidate_id=candidate_id,
            evaluations=tuple(
                CandidateEvaluationDTO(
                    id=item.id,
                    stage=item.stage,
                    decision=item.decision,
                    reason=item.reason,
                    metrics=dict(item.metrics),
                    created_at=item.created_at,
                )
                for item in evaluations
            ),
            benchmarks=tuple(
                ServerBenchmarkDTO(
                    id=item.id,
                    server_run_id=item.server_run_id,
                    workload_case_id=item.workload_case_id,
                    category=item.category,
                    status=item.status,
                    requests=item.requests,
                    failed=item.failed,
                    turns=item.turns,
                    avg_prompt_ts=item.avg_prompt_ts,
                    avg_pred_ts=item.avg_pred_ts,
                    avg_latency_ms=item.avg_latency_ms,
                    draft_n=item.draft_n,
                    accepted_n=item.accepted_n,
                    accept_rate=item.accept_rate,
                    created_at=item.created_at,
                )
                for item in benchmarks
            ),
        )

    def validate_candidate(
        self,
        candidate_id: str,
        request: ServerValidationRequest,
    ) -> ServerValidationResponse:
        summary = ServerValidationService(self.database).validate(
            request.experiment_id,
            candidate_id=candidate_id,
            server_binary_id=request.server_binary_id,
            speed_bench_binary_id=request.speed_bench_binary_id,
            model_path=Path(request.model_path),
            placement_id=request.placement_id,
            draft_model_path=(
                None
                if request.draft_model_path is None
                else Path(request.draft_model_path)
            ),
            model_name=request.model_name,
            host=request.host,
            port=request.port,
            readiness_timeout_seconds=request.readiness_timeout_seconds,
            request_timeout_seconds=request.request_timeout_seconds,
            benchmark_timeout_seconds=request.benchmark_timeout_seconds,
            workload_case_id=request.workload_case_id,
        )
        return ServerValidationResponse(
            experiment_id=summary.experiment_id,
            candidate_id=summary.candidate_id,
            server_run_id=summary.server_run_id,
            benchmark_ids=summary.benchmark_ids,
            completed=summary.completed,
            speculative=summary.speculative,
        )

    def promote_candidate(
        self,
        candidate_id: str,
        request: PromotionRequest,
    ) -> PromotionResponse:
        proposal = PromotionService(self.database, self.profiles).propose(
            request.experiment_id,
            candidate_id,
            source_profile_id=request.source_profile_id,
        )
        return PromotionResponse(
            id=proposal.id,
            experiment_id=proposal.experiment_id,
            candidate_id=proposal.candidate_id,
            source_profile=proposal.source_profile,
            changes=tuple(
                LauncherArgChangeDTO(
                    path=item.path,
                    argument=item.argument,
                    before=item.before,
                    after=item.after,
                )
                for item in proposal.changes
            ),
            patch=proposal.patch,
            source_snapshot=proposal.source_snapshot,
            proposed_snapshot=proposal.proposed_snapshot,
            validation=proposal.validation,
        )

    def _experiment_dto(
        self,
        connection: sqlite3.Connection,
        record: ExperimentRecord,
    ) -> ExperimentDTO:
        definition = ExperimentRepository(connection).get_definition(record.id)
        if definition is None:
            raise RuntimeError(f"experiment {record.id} has no definition")
        counts = connection.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM experiment_candidate
                 WHERE experiment_id = ?) AS candidates,
                (SELECT COUNT(*) FROM experiment_workload
                 WHERE experiment_id = ?) AS workloads,
                (SELECT COUNT(*) FROM benchmark_case
                 WHERE experiment_id = ?) AS cases
            """,
            (record.id, record.id, record.id),
        ).fetchone()
        if counts is None:
            raise RuntimeError("experiment count query returned no row")
        incomplete = BenchmarkCaseRepository(connection).count_incomplete(record.id)
        base_candidate = CandidateRepository(connection).get(record.base_candidate_id)
        search_space = SearchSpaceRepository(connection).get(record.search_space_id)
        workload_suite = WorkloadSuiteRepository(connection).get(record.workload_suite_id)
        measurement_policy = MeasurementPolicyRepository(connection).get(
            record.measurement_policy_id
        )
        if (
            base_candidate is None
            or search_space is None
            or workload_suite is None
            or measurement_policy is None
        ):
            raise RuntimeError(f"experiment {record.id} references missing immutable data")
        return ExperimentDTO(
            id=record.id,
            status=record.status,
            name=record.name,
            base_candidate_id=record.base_candidate_id,
            search_space_id=record.search_space_id,
            workload_suite_id=record.workload_suite_id,
            measurement_policy_id=record.measurement_policy_id,
            created_at=record.created_at,
            frozen_at=record.frozen_at,
            completed_at=record.completed_at,
            candidate_count=int(counts["candidates"]),
            workload_count=int(counts["workloads"]),
            benchmark_case_count=int(counts["cases"]),
            incomplete_case_count=incomplete,
            definition=definition,
            base_candidate=base_candidate,
            search_space=search_space,
            workload_suite=workload_suite,
            measurement_policy=measurement_policy,
        )

    def create_deployment(
        self,
        request: DeploymentCreateRequest,
    ) -> DeploymentDTO:
        try:
            with self.database.session() as connection:
                identifier = DeploymentCandidateRepository(
                    connection
                ).put(request.deployment)
        except sqlite3.IntegrityError as exc:
            raise ApiConflictError(
                "deployment references an unknown Candidate, binary, "
                "or workload suite"
            ) from exc
        return self.get_deployment(identifier)

    def get_deployment(self, deployment_id: str) -> DeploymentDTO:
        operation = self.deployment_operations.snapshot(deployment_id)
        with self.database.session() as connection:
            repository = DeploymentCandidateRepository(connection)
            record = repository.record(deployment_id)
            definition = repository.get(deployment_id)
            if record is None or definition is None:
                raise ApiNotFoundError(
                    f"deployment not found: {deployment_id}"
                )
            counts = connection.execute(
                """
                SELECT
                    (
                        SELECT COUNT(*)
                        FROM deployment_plan
                        WHERE base_deployment_candidate_id = ?
                    ) AS plans,
                    (
                        SELECT COUNT(DISTINCT dp.id)
                        FROM deployment_placement AS dp
                        WHERE dp.deployment_candidate_id = ?
                           OR EXISTS (
                                SELECT 1
                                FROM deployment_plan AS plan
                                JOIN deployment_plan_case AS pc
                                  ON pc.deployment_plan_id = plan.id
                                WHERE plan.base_deployment_candidate_id = ?
                                  AND pc.deployment_placement_id = dp.id
                           )
                    ) AS placements,
                    (
                        SELECT COUNT(*)
                        FROM deployment_run AS dr
                        WHERE dr.deployment_candidate_id = ?
                           OR EXISTS (
                                SELECT 1
                                FROM deployment_plan AS plan
                                JOIN deployment_plan_case AS pc
                                  ON pc.deployment_plan_id = plan.id
                                WHERE plan.base_deployment_candidate_id = ?
                                  AND pc.deployment_candidate_id =
                                      dr.deployment_candidate_id
                           )
                    ) AS runs
                """,
                (
                    deployment_id,
                    deployment_id,
                    deployment_id,
                    deployment_id,
                    deployment_id,
                ),
            ).fetchone()
            latest_plan = connection.execute(
                """
                SELECT id
                FROM deployment_plan
                WHERE base_deployment_candidate_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (deployment_id,),
            ).fetchone()
            latest_run = connection.execute(
                """
                SELECT status
                FROM deployment_run AS dr
                WHERE dr.deployment_candidate_id = ?
                   OR EXISTS (
                        SELECT 1
                        FROM deployment_plan AS plan
                        JOIN deployment_plan_case AS pc
                          ON pc.deployment_plan_id = plan.id
                        WHERE plan.base_deployment_candidate_id = ?
                          AND pc.deployment_candidate_id =
                              dr.deployment_candidate_id
                   )
                ORDER BY dr.created_at DESC, dr.id DESC
                LIMIT 1
                """,
                (deployment_id, deployment_id),
            ).fetchone()
        if counts is None:
            raise RuntimeError("deployment count query returned no row")
        return DeploymentDTO(
            id=deployment_id,
            status=_deployment_status(
                operation,
                None if latest_run is None else str(latest_run["status"]),
                int(counts["plans"]),
            ),
            created_at=record.created_at,
            definition=definition,
            plan_count=int(counts["plans"]),
            placement_count=int(counts["placements"]),
            run_count=int(counts["runs"]),
            latest_plan_id=(
                None if latest_plan is None else str(latest_plan["id"])
            ),
        )

    def plan_deployment(
        self,
        deployment_id: str,
        request: DeploymentPlanRequest,
    ) -> DeploymentPlanResponse:
        self._require_deployment(deployment_id)
        inputs = tuple(
            DeploymentEstimatorInput(
                instance_id=item.instance_id,
                helper_binary_id=item.helper_binary_id,
                model_path=Path(item.model_path),
            )
            for item in request.instances
        )
        try:
            summary = DeploymentPlannerService(self.database).plan(
                deployment_id,
                request.search_space,
                inputs,
                timeout_seconds=request.timeout_seconds,
            )
        except ValueError as exc:
            raise ApiConflictError(str(exc)) from exc
        return _deployment_plan_response(summary)

    def list_deployment_candidates(
        self,
        deployment_id: str,
    ) -> DeploymentCandidateListResponse:
        self._require_deployment(deployment_id)
        with self.database.session() as connection:
            rows = connection.execute(
                """
                SELECT pc.deployment_candidate_id,
                       pc.deployment_placement_id,
                       pc.generation_json
                FROM deployment_plan AS plan
                JOIN deployment_plan_case AS pc
                  ON pc.deployment_plan_id = plan.id
                WHERE plan.base_deployment_candidate_id = ?
                ORDER BY plan.created_at DESC, plan.id DESC, pc.ordinal
                """,
                (deployment_id,),
            ).fetchall()
            grouped: dict[str, dict[str, object]] = {}
            for row in rows:
                candidate_id = str(row["deployment_candidate_id"])
                item = grouped.setdefault(
                    candidate_id,
                    {
                        "generation": _json_object(
                            str(row["generation_json"])
                        ),
                        "placements": [],
                    },
                )
                placements = item["placements"]
                if not isinstance(placements, list):
                    raise RuntimeError("deployment placement accumulator is invalid")
                placement_id = str(row["deployment_placement_id"])
                if placement_id not in placements:
                    placements.append(placement_id)

            repository = DeploymentCandidateRepository(connection)
            items: list[DeploymentCandidateItemDTO] = []
            for candidate_id, item in grouped.items():
                definition = repository.get(candidate_id)
                if definition is None:
                    raise RuntimeError(
                        f"plan references missing deployment Candidate "
                        f"{candidate_id}"
                    )
                generation = item["generation"]
                placements = item["placements"]
                if (
                    not isinstance(generation, dict)
                    or not isinstance(placements, list)
                ):
                    raise RuntimeError("deployment candidate aggregate is invalid")
                items.append(
                    DeploymentCandidateItemDTO(
                        id=candidate_id,
                        definition=definition,
                        generation={
                            str(key): value
                            for key, value in generation.items()
                        },
                        rejection_count=len(
                            repository.rejections(candidate_id)
                        ),
                        placement_ids=tuple(
                            str(value) for value in placements
                        ),
                    )
                )
        return DeploymentCandidateListResponse(items=tuple(items))

    def list_deployment_placements(
        self,
        deployment_id: str,
    ) -> DeploymentPlacementListResponse:
        self._require_deployment(deployment_id)
        with self.database.session() as connection:
            placement_ids = _deployment_placement_ids(
                connection,
                deployment_id,
            )
            repository = DeploymentPlacementRepository(connection)
            records = []
            for placement_id in placement_ids:
                record = repository.record(placement_id)
                placement = repository.get(placement_id)
                if record is None or placement is None:
                    raise RuntimeError(
                        f"deployment placement disappeared: {placement_id}"
                    )
                records.append((record, placement))
        analysis = DeploymentAnalysisService(self.database)
        return DeploymentPlacementListResponse(
            items=tuple(
                DeploymentPlacementDTO(
                    id=record.id,
                    deployment_candidate_id=(
                        record.deployment_candidate_id
                    ),
                    host_id=record.host_id,
                    feasibility=record.feasibility,
                    placement=placement,
                    memory=analysis.memory_matrix(record.id),
                )
                for record, placement in records
            )
        )

    def start_deployment(
        self,
        deployment_id: str,
        request: DeploymentRunRequest | None,
        *,
        resume: bool,
    ) -> DeploymentProgressDTO:
        self._require_deployment(deployment_id)
        spec = (
            None
            if request is None
            else _deployment_operation_spec(request)
        )
        try:
            if resume:
                self.deployment_operations.resume(
                    deployment_id,
                    spec,
                    background=True,
                )
            else:
                if spec is None:
                    raise ApiConflictError(
                        "deployment run requires an execution request"
                    )
                self.deployment_operations.start(
                    deployment_id,
                    spec,
                    background=True,
                )
        except DeploymentOperationError as exc:
            raise ApiConflictError(str(exc)) from exc
        return self.deployment_progress(deployment_id)

    def pause_deployment(
        self,
        deployment_id: str,
    ) -> DeploymentProgressDTO:
        self._require_deployment(deployment_id)
        try:
            self.deployment_operations.pause(deployment_id)
        except DeploymentOperationError as exc:
            raise ApiConflictError(str(exc)) from exc
        return self.deployment_progress(deployment_id)

    def cancel_deployment(
        self,
        deployment_id: str,
    ) -> DeploymentProgressDTO:
        self._require_deployment(deployment_id)
        try:
            self.deployment_operations.cancel(deployment_id)
        except DeploymentOperationError as exc:
            raise ApiConflictError(str(exc)) from exc
        return self.deployment_progress(deployment_id)

    def deployment_progress(
        self,
        deployment_id: str,
    ) -> DeploymentProgressDTO:
        self._require_deployment(deployment_id)
        operation = self.deployment_operations.snapshot(deployment_id)
        with self.database.session() as connection:
            latest_plan = connection.execute(
                """
                SELECT id
                FROM deployment_plan
                WHERE base_deployment_candidate_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (deployment_id,),
            ).fetchone()
            plan_id = None if latest_plan is None else str(latest_plan["id"])
            planned = 0
            completed = 0
            failed = 0
            if plan_id is not None:
                row = connection.execute(
                    """
                    SELECT
                        COUNT(DISTINCT pc.deployment_candidate_id) AS planned,
                        COUNT(DISTINCT CASE
                            WHEN EXISTS (
                                SELECT 1
                                FROM deployment_run AS dr
                                WHERE dr.deployment_placement_id =
                                      pc.deployment_placement_id
                                  AND dr.status = 'completed'
                            )
                            THEN pc.deployment_candidate_id
                        END) AS completed,
                        COUNT(DISTINCT CASE
                            WHEN EXISTS (
                                SELECT 1
                                FROM deployment_run AS dr
                                WHERE dr.deployment_placement_id =
                                      pc.deployment_placement_id
                                  AND dr.status = 'failed'
                            )
                            THEN pc.deployment_candidate_id
                        END) AS failed
                    FROM deployment_plan_case AS pc
                    WHERE pc.deployment_plan_id = ?
                    """,
                    (plan_id,),
                ).fetchone()
                if row is not None:
                    planned = int(row["planned"])
                    completed = int(row["completed"])
                    failed = int(row["failed"])

            active = connection.execute(
                """
                SELECT dr.id
                FROM deployment_run AS dr
                WHERE dr.status IN ('starting', 'ready', 'running')
                  AND (
                      dr.deployment_candidate_id = ?
                      OR EXISTS (
                          SELECT 1
                          FROM deployment_plan AS plan
                          JOIN deployment_plan_case AS pc
                            ON pc.deployment_plan_id = plan.id
                          WHERE plan.base_deployment_candidate_id = ?
                            AND pc.deployment_placement_id =
                                dr.deployment_placement_id
                      )
                  )
                ORDER BY dr.created_at DESC, dr.id DESC
                LIMIT 1
                """,
                (deployment_id, deployment_id),
            ).fetchone()
            active_run_id = (
                None if active is None else str(active["id"])
            )
            if (
                active_run_id is None
                and operation is not None
                and operation.status in {
                    "running",
                    "pausing",
                    "cancelling",
                }
            ):
                active_run_id = operation.deployment_run_id

            members: tuple[DeploymentMemberStateDTO, ...] = ()
            current_phase: str | None = None
            if active_run_id is not None:
                run_repository = DeploymentRunRepository(connection)
                members = tuple(
                    DeploymentMemberStateDTO(
                        instance_id=item.instance_id,
                        status=item.member_status,
                        endpoint=item.endpoint,
                        pid=item.pid,
                        ready_at=item.ready_at,
                        exit_code=item.exit_code,
                    )
                    for item in run_repository.members(active_run_id)
                )
                phase_row = connection.execute(
                    """
                    SELECT phase
                    FROM deployment_workload_run
                    WHERE deployment_run_id = ? AND status = 'running'
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """,
                    (active_run_id,),
                ).fetchone()
                current_phase = (
                    None
                    if phase_row is None
                    else str(phase_row["phase"])
                )

            latest_run = connection.execute(
                """
                SELECT dr.status
                FROM deployment_run AS dr
                WHERE dr.deployment_candidate_id = ?
                   OR EXISTS (
                        SELECT 1
                        FROM deployment_plan AS plan
                        JOIN deployment_plan_case AS pc
                          ON pc.deployment_plan_id = plan.id
                        WHERE plan.base_deployment_candidate_id = ?
                          AND pc.deployment_placement_id =
                              dr.deployment_placement_id
                   )
                ORDER BY dr.created_at DESC, dr.id DESC
                LIMIT 1
                """,
                (deployment_id, deployment_id),
            ).fetchone()

        return DeploymentProgressDTO(
            deployment_id=deployment_id,
            deployment_status=_deployment_status(
                operation,
                None if latest_run is None else str(latest_run["status"]),
                0 if plan_id is None else 1,
            ),
            planned_candidates=planned,
            completed_candidates=completed,
            failed_candidates=failed,
            active_deployment_run=active_run_id,
            member_states=members,
            current_workload_phase=current_phase,
            operation=_deployment_operation_dto(operation),
        )

    def list_deployment_runs(
        self,
        deployment_id: str,
    ) -> DeploymentRunListResponse:
        self._require_deployment(deployment_id)
        with self.database.session() as connection:
            rows = connection.execute(
                """
                SELECT dr.id
                FROM deployment_run AS dr
                WHERE dr.deployment_candidate_id = ?
                   OR EXISTS (
                        SELECT 1
                        FROM deployment_plan AS plan
                        JOIN deployment_plan_case AS pc
                          ON pc.deployment_plan_id = plan.id
                        WHERE plan.base_deployment_candidate_id = ?
                          AND pc.deployment_placement_id =
                              dr.deployment_placement_id
                   )
                ORDER BY dr.created_at, dr.id
                """,
                (deployment_id, deployment_id),
            ).fetchall()
            repository = DeploymentRunRepository(connection)
            items: list[DeploymentRunDTO] = []
            for row in rows:
                record = repository.get(str(row["id"]))
                if record is None:
                    raise RuntimeError("deployment run disappeared")
                items.append(
                    _deployment_run_dto(connection, record)
                )
        return DeploymentRunListResponse(items=tuple(items))

    def deployment_results(
        self,
        deployment_id: str,
        *,
        filters: tuple[DeploymentAnalysisFilter, ...] = (),
    ) -> DeploymentResultsResponse:
        self._require_deployment(deployment_id)
        with self.database.session() as connection:
            placement_ids = _deployment_placement_ids(
                connection,
                deployment_id,
            )
        raw = DeploymentAnalysisService(self.database).export(
            format_name="json",
            filters=filters,
            placement_ids=placement_ids,
        )
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise RuntimeError("deployment analysis export is not a list")
        rows: list[dict[str, object]] = []
        for item in parsed:
            if not isinstance(item, dict):
                raise RuntimeError(
                    "deployment analysis export row is not an object"
                )
            rows.append({str(key): value for key, value in item.items()})
        return DeploymentResultsResponse(
            deployment_id=deployment_id,
            rows=tuple(rows),
        )

    def deployment_pareto(
        self,
        deployment_id: str,
        *,
        objectives: tuple[DeploymentParetoObjective, ...],
        constraints: tuple[DeploymentMetricConstraint, ...] = (),
        filters: tuple[DeploymentAnalysisFilter, ...] = (),
    ) -> DeploymentParetoResponse:
        self._require_deployment(deployment_id)
        with self.database.session() as connection:
            placement_ids = _deployment_placement_ids(
                connection,
                deployment_id,
            )
        result = DeploymentAnalysisService(self.database).pareto(
            objectives=objectives,
            constraints=constraints,
            filters=filters,
            placement_ids=placement_ids,
        )
        return DeploymentParetoResponse(
            deployment_id=deployment_id,
            result=result,
        )

    def _require_deployment(self, deployment_id: str) -> None:
        with self.database.session() as connection:
            if DeploymentCandidateRepository(connection).get(
                deployment_id
            ) is None:
                raise ApiNotFoundError(
                    f"deployment not found: {deployment_id}"
                )

    def _require_experiment(self, experiment_id: str) -> None:
        with self.database.session() as connection:
            self._require_experiment_connection(connection, experiment_id)

    @staticmethod
    def _require_experiment_connection(
        connection: sqlite3.Connection,
        experiment_id: str,
    ) -> None:
        if ExperimentRepository(connection).get(experiment_id) is None:
            raise ApiNotFoundError(f"experiment not found: {experiment_id}")


def parse_filters(values: tuple[str, ...]) -> tuple[AnalysisFilter, ...]:
    return tuple(_parse_filter(value) for value in values)


def _parse_filter(value: str) -> AnalysisFilter:
    path, separator, raw_value = value.partition("=")
    if not separator or not path:
        raise ValueError(f"invalid filter {value!r}; expected PATH=VALUE")
    return AnalysisFilter(path=path, value=_parse_scalar(raw_value))


def _parse_scalar(value: str) -> JsonScalar:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    if parsed is None or isinstance(parsed, str | int | float | bool):
        return parsed
    raise ValueError("analysis filter values must be JSON scalars")


def _profile_dto(profile: LauncherProfile) -> LauncherProfileDTO:
    return LauncherProfileDTO(
        id=profile.id,
        binary_key=profile.binary_key,
        binary_path=profile.binary_path,
        profiles=profile.profiles,
        model_path=profile.model_path,
        draft_model_path=profile.draft_model_path,
        server_alias=profile.server_alias,
        args=profile.args,
        candidate=profile.candidate,
    )


def _binary_dto(record: BinaryRecord) -> BinaryDTO:
    return BinaryDTO(
        id=record.id,
        sha256=record.sha256,
        kind=record.kind,
        path=record.path,
        size_bytes=record.size_bytes,
        mtime_ns=record.mtime_ns,
        git_commit=record.git_commit,
        git_branch=record.git_branch,
        git_dirty=record.git_dirty,
        build_number=record.build_number,
        build_info=dict(record.build_info),
        capabilities=dict(record.capabilities),
        created_at=record.created_at,
    )


def _placement_dto(record: ResolvedPlacementRecord) -> PlacementDTO:
    return PlacementDTO(
        id=record.id,
        candidate_id=record.candidate_id,
        host_id=record.host_id,
        binary_id=record.binary_id,
        fit_attempt_id=record.fit_attempt_id,
        production_context_size=record.production_context_size,
        n_gpu_layers=record.n_gpu_layers,
        n_cpu_moe=record.n_cpu_moe,
        split_mode=record.split_mode,
        main_gpu=record.main_gpu,
        devices=record.devices,
        tensor_split=record.tensor_split,
        override_tensor=record.override_tensor,
        request=dict(record.request),
        created_at=record.created_at,
    )


def _plan_dto(summary: PlanSummary) -> PlanSummaryDTO:
    return PlanSummaryDTO(
        experiment_id=summary.experiment_id,
        experiment_name=summary.experiment_name,
        raw_combinations=summary.raw_combinations,
        rejected_by_constraints=summary.rejected_by_constraints,
        duplicate_candidates=summary.duplicate_candidates,
        candidate_count=summary.candidate_count,
        workloads_per_candidate=summary.workloads_per_candidate,
        benchmark_case_count=summary.benchmark_case_count,
        unique_workload_count=summary.unique_workload_count,
    )


def _operation_dto(snapshot: OperationSnapshot | None) -> OperationDTO | None:
    if snapshot is None:
        return None
    summary = snapshot.summary
    return OperationDTO(
        id=snapshot.id,
        experiment_id=snapshot.experiment_id,
        status=snapshot.status,
        started_at=snapshot.started_at,
        finished_at=snapshot.finished_at,
        requested_action=snapshot.requested_action,
        summary=(
            None
            if summary is None
            else ExecutionSummaryDTO(
                experiment_id=summary.experiment_id,
                attempted=summary.attempted,
                completed=summary.completed,
                failed=summary.failed,
                remaining=summary.remaining,
                interrupted=summary.interrupted,
                limited=summary.limited,
            )
        ),
        error=snapshot.error,
    )


def _run_summary_dto(
    *,
    record: object,
    candidate_id: str,
    workload_case_id: str,
    placement_id: str | None,
    workload_kind: str,
) -> RunSummaryDTO:
    from llama_profile_lab.db.records import BenchmarkRunRecord

    if not isinstance(record, BenchmarkRunRecord):
        raise TypeError("expected BenchmarkRunRecord")
    return RunSummaryDTO(
        id=record.id,
        benchmark_case_id=record.benchmark_case_id,
        candidate_id=candidate_id,
        workload_case_id=workload_case_id,
        placement_id=placement_id,
        workload_kind=workload_kind,
        host_id=record.host_id,
        binary_id=record.binary_id,
        measurement_policy_id=record.measurement_policy_id,
        started_at=record.started_at,
        finished_at=record.finished_at,
        duration_ns=record.duration_ns,
        status=record.status,
        exit_code=record.exit_code,
        quality=record.quality,
        quality_details=(
            None if record.quality_details is None else dict(record.quality_details)
        ),
    )


def _json_object(value: str) -> dict[str, object]:
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError("persisted JSON must be an object")
    return {str(key): item for key, item in loaded.items()}
