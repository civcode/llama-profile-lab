"""llama-server + SPEED-Bench finalist validation orchestration."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

from llama_profile_lab.db import (
    CandidateRepository,
    Database,
    EnvironmentRepository,
    PlacementRepository,
    ServerValidationRepository,
    WorkloadCaseRepository,
)
from llama_profile_lab.db.records import (
    BinaryRecord,
    ResolvedPlacementRecord,
    ServerBenchmarkRecord,
    ServerBenchmarkStatus,
    ServerRunStatus,
)
from llama_profile_lab.domain.workload import SpeedBenchWorkloadCase
from llama_profile_lab.execution.host import BasicHostInfo, detect_basic_host
from llama_profile_lab.execution.lock import HostLock
from llama_profile_lab.execution.placement import (
    PlacementConfigurationError,
    resolved_placement_from_record,
    validate_fixed_placement,
)
from llama_profile_lab.execution.process import ProcessRunner, ProcessRunnerError
from llama_profile_lab.execution.server_process import (
    ManagedServerProcess,
    ServerProcessError,
)
from llama_profile_lab.llama import (
    CapabilitySet,
    LlamaServerAdapter,
    LlamaServerConfigurationError,
    SpeedBenchAdapter,
    SpeedBenchConfigurationError,
    SpeedBenchParseError,
    sha256_file,
)


class ServerValidationError(RuntimeError):
    """Raised when finalist validation cannot be executed safely."""


@dataclass(frozen=True, slots=True)
class ServerValidationSummary:
    """User-facing result for one Candidate server-validation session."""

    experiment_id: str
    candidate_id: str
    server_run_id: str
    benchmark_ids: tuple[str, ...]
    completed: bool
    speculative: bool


@dataclass(frozen=True, slots=True)
class ServerComparison:
    """Descriptive baseline-versus-speculative server measurements."""

    experiment_id: str
    baseline_candidate_id: str
    speculative_candidate_id: str
    workload_case_id: str
    category: str
    baseline_prompt_ts: float | None
    speculative_prompt_ts: float | None
    baseline_pred_ts: float | None
    speculative_pred_ts: float | None
    baseline_latency_ms: float | None
    speculative_latency_ms: float | None
    decode_speedup: float | None
    latency_speedup: float | None
    draft_n: int | None
    accepted_n: int | None
    accept_rate: float | None


class ServerValidationService:
    """Launch a resolved finalist Candidate and execute persisted SPEED-Bench workloads."""

    def __init__(
        self,
        database: Database,
        *,
        process_runner: ProcessRunner | None = None,
        server_adapter: LlamaServerAdapter | None = None,
        speed_bench_adapter: SpeedBenchAdapter | None = None,
        host_detector: Callable[[], BasicHostInfo] = detect_basic_host,
        server_process_factory: Callable[[tuple[str, ...]], ManagedServerProcess] | None = None,
    ) -> None:
        self.database = database
        self.process_runner = process_runner or ProcessRunner()
        self.server_adapter = server_adapter or LlamaServerAdapter()
        self.speed_bench_adapter = speed_bench_adapter or SpeedBenchAdapter()
        self.host_detector = host_detector
        self.server_process_factory = server_process_factory or ManagedServerProcess

    def validate(
        self,
        experiment_id: str,
        *,
        candidate_id: str,
        server_binary_id: str,
        speed_bench_binary_id: str,
        model_path: Path,
        placement_id: str | None = None,
        draft_model_path: Path | None = None,
        model_name: str | None = None,
        host: str = "127.0.0.1",
        port: int = 8080,
        readiness_timeout_seconds: float = 300.0,
        request_timeout_seconds: float = 600.0,
        benchmark_timeout_seconds: float | None = None,
        workload_case_id: str | None = None,
        cancel_event: Event | None = None,
    ) -> ServerValidationSummary:
        if readiness_timeout_seconds <= 0:
            raise ServerValidationError("readiness timeout must be positive")
        if request_timeout_seconds <= 0:
            raise ServerValidationError("request timeout must be positive")
        lock_path = _lock_path(self.database)
        with HostLock(lock_path):
            return self._validate_locked(
                experiment_id,
                candidate_id=candidate_id,
                server_binary_id=server_binary_id,
                speed_bench_binary_id=speed_bench_binary_id,
                model_path=model_path,
                placement_id=placement_id,
                draft_model_path=draft_model_path,
                model_name=model_name,
                host=host,
                port=port,
                readiness_timeout_seconds=readiness_timeout_seconds,
                request_timeout_seconds=request_timeout_seconds,
                benchmark_timeout_seconds=benchmark_timeout_seconds,
                workload_case_id=workload_case_id,
                cancel_event=cancel_event,
            )

    def compare(
        self,
        experiment_id: str,
        *,
        baseline_candidate_id: str,
        speculative_candidate_id: str,
        workload_case_id: str | None = None,
        category: str = "all",
    ) -> ServerComparison:
        with self.database.session() as connection:
            repository = ServerValidationRepository(connection)
            baseline = _latest_benchmarks(
                repository.benchmarks_for_candidate(
                    experiment_id=experiment_id,
                    candidate_id=baseline_candidate_id,
                )
            )
            speculative = _latest_benchmarks(
                repository.benchmarks_for_candidate(
                    experiment_id=experiment_id,
                    candidate_id=speculative_candidate_id,
                )
            )

        common = {
            key
            for key in baseline
            if key in speculative and key[1] == category
        }
        if workload_case_id is not None:
            common = {key for key in common if key[0] == workload_case_id}
        if len(common) != 1:
            raise ServerValidationError(
                "server comparison requires exactly one common workload/category; "
                "specify --workload-case or --category"
            )
        key = next(iter(common))
        base = baseline[key]
        spec = speculative[key]
        return ServerComparison(
            experiment_id=experiment_id,
            baseline_candidate_id=baseline_candidate_id,
            speculative_candidate_id=speculative_candidate_id,
            workload_case_id=key[0],
            category=key[1],
            baseline_prompt_ts=base.avg_prompt_ts,
            speculative_prompt_ts=spec.avg_prompt_ts,
            baseline_pred_ts=base.avg_pred_ts,
            speculative_pred_ts=spec.avg_pred_ts,
            baseline_latency_ms=base.avg_latency_ms,
            speculative_latency_ms=spec.avg_latency_ms,
            decode_speedup=_ratio(spec.avg_pred_ts, base.avg_pred_ts),
            latency_speedup=_ratio(base.avg_latency_ms, spec.avg_latency_ms),
            draft_n=spec.draft_n,
            accepted_n=spec.accepted_n,
            accept_rate=spec.accept_rate,
        )

    def _validate_locked(
        self,
        experiment_id: str,
        *,
        candidate_id: str,
        server_binary_id: str,
        speed_bench_binary_id: str,
        model_path: Path,
        placement_id: str | None,
        draft_model_path: Path | None,
        model_name: str | None,
        host: str,
        port: int,
        readiness_timeout_seconds: float,
        request_timeout_seconds: float,
        benchmark_timeout_seconds: float | None,
        workload_case_id: str | None,
        cancel_event: Event | None,
    ) -> ServerValidationSummary:
        with self.database.session() as connection:
            candidates = CandidateRepository(connection)
            placements = PlacementRepository(connection)
            environment = EnvironmentRepository(connection)
            server_runs = ServerValidationRepository(connection)
            workloads = WorkloadCaseRepository(connection)

            candidate = candidates.get(candidate_id)
            if candidate is None:
                raise ServerValidationError(f"candidate not found: {candidate_id}")
            linked = connection.execute(
                """
                SELECT 1 FROM experiment_candidate
                WHERE experiment_id = ? AND candidate_id = ?
                """,
                (experiment_id, candidate_id),
            ).fetchone()
            if linked is None:
                raise ServerValidationError(
                    f"candidate {candidate_id} is not part of experiment {experiment_id}"
                )

            server_binary = environment.get_binary(server_binary_id)
            speed_binary = environment.get_binary(speed_bench_binary_id)
            if server_binary is None:
                raise ServerValidationError(f"binary not found: {server_binary_id}")
            if speed_binary is None:
                raise ServerValidationError(f"binary not found: {speed_bench_binary_id}")
            _verify_binary(server_binary, "llama-server")
            _verify_binary(speed_binary, "speed-bench")

            target_path = _resolve_file(model_path, "model")
            draft_path = (
                None
                if draft_model_path is None
                else _resolve_file(draft_model_path, "draft model")
            )

            host_info = self.host_detector()
            host_id = environment.put_host(
                hostname=host_info.hostname,
                hardware_fingerprint=host_info.hardware_fingerprint,
                cpu=host_info.cpu,
                ram_bytes=host_info.ram_bytes,
                gpus=host_info.gpus,
                os_info=host_info.os_info,
            )
            selected_placement = _select_placement(
                connection,
                placements,
                experiment_id=experiment_id,
                candidate_id=candidate_id,
                placement_id=placement_id,
            )
            try:
                validate_fixed_placement(
                    selected_placement,
                    candidate=candidate,
                    host_id=host_id,
                )
            except PlacementConfigurationError as exc:
                raise ServerValidationError(str(exc)) from exc

            selected_workloads = _speed_workloads(
                connection,
                workloads,
                experiment_id=experiment_id,
                candidate_id=candidate_id,
                workload_case_id=workload_case_id,
            )
            if not selected_workloads:
                raise ServerValidationError(
                    "candidate has no SPEED-Bench workload selected in this experiment"
                )

            placement = resolved_placement_from_record(selected_placement)
            try:
                server_argv = self.server_adapter.build_argv(
                    binary_path=Path(server_binary.path),
                    capabilities=_capabilities(server_binary, "llama-server"),
                    model_path=target_path,
                    candidate=candidate,
                    placement=placement,
                    host=host,
                    port=port,
                    draft_model_path=draft_path,
                    model_alias=model_name,
                )
            except LlamaServerConfigurationError as exc:
                raise ServerValidationError(str(exc)) from exc

            server_runs.recover_orphaned(experiment_id)
            server_runs.add_evaluation(
                experiment_id=experiment_id,
                candidate_id=candidate_id,
                stage="finalist",
                decision="validate",
                metrics={"placement_id": selected_placement.id},
            )
            server_run_id = server_runs.create_run(
                experiment_id=experiment_id,
                candidate_id=candidate_id,
                placement_id=selected_placement.id,
                host_id=host_id,
                server_binary_id=server_binary.id,
                target_model_id=candidate.model.target_model_id,
                draft_model_id=candidate.model.draft_model_id,
                target_model_path=str(target_path),
                draft_model_path=None if draft_path is None else str(draft_path),
                spec_type=candidate.speculative.type,
                spec_draft_n_max=candidate.speculative.draft_n_max,
                bind_host=host,
                bind_port=port,
                argv=server_argv,
                environment=_captured_environment(),
            )

            server = self.server_process_factory(server_argv)
            benchmark_ids: list[str] = []
            terminal_status: ServerRunStatus = "completed"
            failure_reason: str | None = None
            outcome = None
            try:
                server.start()
                ready_at = server.wait_ready(
                    f"http://{host}:{port}/health",
                    timeout_seconds=readiness_timeout_seconds,
                    cancel_event=cancel_event,
                )
                server_runs.mark_ready(server_run_id, ready_at=ready_at)

                with tempfile.TemporaryDirectory(prefix="llprof-speed-bench-") as tempdir:
                    output_dir = Path(tempdir)
                    for selected_id, workload in selected_workloads:
                        for category in workload.speed_bench.categories:
                            if cancel_event is not None and cancel_event.is_set():
                                terminal_status = "cancelled"
                                failure_reason = "validation cancelled"
                                break
                            output_path = output_dir / (
                                f"{selected_id.replace(':', '_')}-{category}.json"
                            )
                            try:
                                argv = self.speed_bench_adapter.build_argv(
                                    binary_path=Path(speed_binary.path),
                                    capabilities=_capabilities(speed_binary, "speed-bench"),
                                    server_url=f"http://{host}:{port}",
                                    workload=workload,
                                    category=category,
                                    output_path=output_path,
                                    model_name=model_name,
                                    request_timeout_seconds=request_timeout_seconds,
                                )
                            except SpeedBenchConfigurationError as exc:
                                terminal_status = "benchmark_failed"
                                failure_reason = str(exc)
                                break

                            benchmark_id = server_runs.create_benchmark(
                                server_run_id=server_run_id,
                                workload_case_id=selected_id,
                                speed_bench_binary_id=speed_binary.id,
                                category=category,
                                argv=argv,
                            )
                            benchmark_ids.append(benchmark_id)
                            status, reason = self._run_speed_bench(
                                server_runs,
                                benchmark_id=benchmark_id,
                                argv=argv,
                                output_path=output_path,
                                timeout_seconds=benchmark_timeout_seconds,
                                cancel_event=cancel_event,
                            )
                            if status != "completed":
                                terminal_status = (
                                    "interrupted"
                                    if status == "interrupted"
                                    else "cancelled"
                                    if status == "cancelled"
                                    else "benchmark_failed"
                                )
                                failure_reason = reason
                                break
                            if not server.alive:
                                terminal_status = "benchmark_failed"
                                failure_reason = "llama-server exited during SPEED-Bench"
                                break
                        if terminal_status != "completed":
                            break
            except ServerProcessError as exc:
                terminal_status = _server_error_status(exc)
                failure_reason = str(exc)
            finally:
                outcome = server.stop()
                server.close()

            server_runs.finish_run(
                server_run_id,
                status=terminal_status,
                duration_ns=outcome.duration_ns,
                exit_code=outcome.exit_code,
                stdout=outcome.stdout,
                stderr=outcome.stderr,
                finished_at=outcome.finished_at,
            )
            completed = terminal_status == "completed"
            server_runs.add_evaluation(
                experiment_id=experiment_id,
                candidate_id=candidate_id,
                stage="server-validated",
                decision="completed" if completed else "failed",
                reason=failure_reason,
                metrics={
                    "server_run_id": server_run_id,
                    "benchmark_ids": benchmark_ids,
                    "speculative": candidate.speculative.enabled,
                },
            )

            return ServerValidationSummary(
                experiment_id=experiment_id,
                candidate_id=candidate_id,
                server_run_id=server_run_id,
                benchmark_ids=tuple(benchmark_ids),
                completed=completed,
                speculative=candidate.speculative.enabled,
            )

    def _run_speed_bench(
        self,
        repository: ServerValidationRepository,
        *,
        benchmark_id: str,
        argv: tuple[str, ...],
        output_path: Path,
        timeout_seconds: float | None,
        cancel_event: Event | None,
    ) -> tuple[ServerBenchmarkStatus, str | None]:
        try:
            process = self.process_runner.run(
                argv,
                timeout_seconds=timeout_seconds,
                cancel_event=cancel_event,
            )
        except ProcessRunnerError as exc:
            repository.finish_benchmark(
                benchmark_id,
                status="benchmark_failed",
                duration_ns=0,
                exit_code=None,
                stdout="",
                stderr=str(exc),
            )
            return "benchmark_failed", str(exc)

        process_status = _speed_process_status(process)
        parsed = None
        parse_error: str | None = None
        if output_path.is_file():
            try:
                parsed = self.speed_bench_adapter.parse_file(output_path)
            except SpeedBenchParseError as exc:
                parse_error = str(exc)

        if parsed is not None:
            overall = parsed.overall
            repository.finish_benchmark(
                benchmark_id,
                status=process_status,
                duration_ns=process.duration_ns,
                exit_code=process.exit_code,
                stdout=process.stdout,
                stderr=process.stderr,
                raw_result=parsed.raw,
                requests=overall.requests,
                failed=overall.failed,
                turns=overall.turns,
                avg_prompt_ts=overall.avg_prompt_ts,
                avg_pred_ts=overall.avg_pred_ts,
                avg_latency_ms=overall.avg_latency_ms,
                draft_n=overall.draft_n,
                accepted_n=overall.accepted_n,
                accept_rate=overall.accept_rate,
            )
            return process_status, None if process_status == "completed" else process.stderr

        final_status: ServerBenchmarkStatus
        if process_status == "completed":
            final_status = "parser_failed"
        else:
            final_status = process_status
        reason = parse_error or process.stderr or "SPEED-Bench produced no output JSON"
        repository.finish_benchmark(
            benchmark_id,
            status=final_status,
            duration_ns=process.duration_ns,
            exit_code=process.exit_code,
            stdout=process.stdout,
            stderr=reason,
        )
        return final_status, reason


def _speed_process_status(process: Any) -> ServerBenchmarkStatus:
    if process.interrupted:
        return "interrupted"
    if process.cancelled:
        return "cancelled"
    if process.timed_out:
        return "timeout"
    if process.exit_code == 0:
        return "completed"
    return "benchmark_failed"


def _server_error_status(exc: ServerProcessError) -> ServerRunStatus:
    if exc.kind == "cancelled":
        return "cancelled"
    if exc.kind == "interrupted":
        return "interrupted"
    if exc.kind == "readiness_failed":
        return "readiness_failed"
    return "start_failed"


def _select_placement(
    connection: Any,
    repository: PlacementRepository,
    *,
    experiment_id: str,
    candidate_id: str,
    placement_id: str | None,
) -> ResolvedPlacementRecord:
    if placement_id is not None:
        record = repository.get(placement_id)
        if record is None:
            raise ServerValidationError(f"placement not found: {placement_id}")
        if record.candidate_id != candidate_id:
            raise ServerValidationError("placement belongs to a different Candidate")
        return record

    rows = connection.execute(
        """
        SELECT DISTINCT placement_id
        FROM benchmark_case
        WHERE experiment_id = ? AND candidate_id = ? AND placement_id IS NOT NULL
        ORDER BY placement_id
        """,
        (experiment_id, candidate_id),
    ).fetchall()
    ids = [str(row["placement_id"]) for row in rows]
    if len(ids) != 1:
        raise ServerValidationError(
            "could not infer one resolved placement for Candidate; pass --placement"
        )
    record = repository.get(ids[0])
    if record is None:
        raise ServerValidationError(f"resolved placement not found: {ids[0]}")
    return record


def _speed_workloads(
    connection: Any,
    repository: WorkloadCaseRepository,
    *,
    experiment_id: str,
    candidate_id: str,
    workload_case_id: str | None,
) -> tuple[tuple[str, SpeedBenchWorkloadCase], ...]:
    query = """
        SELECT ew.workload_case_id
        FROM experiment_workload AS ew
        JOIN workload_case AS wc ON wc.id = ew.workload_case_id
        WHERE ew.experiment_id = ?
          AND ew.candidate_id = ?
          AND wc.kind = 'speed-bench'
    """
    parameters: list[Any] = [experiment_id, candidate_id]
    if workload_case_id is not None:
        query += " AND ew.workload_case_id = ?"
        parameters.append(workload_case_id)
    query += " ORDER BY ew.suite_case_index, ew.workload_case_id"
    rows = connection.execute(query, tuple(parameters)).fetchall()
    result: list[tuple[str, SpeedBenchWorkloadCase]] = []
    for row in rows:
        identifier = str(row["workload_case_id"])
        workload = repository.get(identifier)
        if isinstance(workload, SpeedBenchWorkloadCase):
            result.append((identifier, workload))
    return tuple(result)


def _latest_benchmarks(
    records: tuple[ServerBenchmarkRecord, ...],
) -> dict[tuple[str, str], ServerBenchmarkRecord]:
    result: dict[tuple[str, str], ServerBenchmarkRecord] = {}
    for record in records:
        result[(record.workload_case_id, record.category)] = record
    return result


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _verify_binary(binary: BinaryRecord, expected_kind: str) -> None:
    if binary.kind != expected_kind:
        raise ServerValidationError(
            f"binary {binary.id} is {binary.kind}, not {expected_kind}"
        )
    path = Path(binary.path)
    if not path.is_file():
        raise ServerValidationError(
            f"registered {expected_kind} path does not exist: {path}"
        )
    if sha256_file(path) != binary.sha256:
        raise ServerValidationError(
            f"registered {expected_kind} executable changed on disk; re-run binary inspect"
        )


def _capabilities(binary: BinaryRecord, expected_kind: str) -> CapabilitySet:
    if expected_kind not in {"llama-server", "speed-bench"}:
        raise ValueError(f"unsupported capability kind: {expected_kind}")
    return CapabilitySet.from_mapping(
        dict(binary.capabilities),
        fallback_kind=expected_kind,
    )


def _resolve_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ServerValidationError(f"{label} does not exist: {resolved}")
    return resolved


def _captured_environment() -> dict[str, str]:
    keys = ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
    return {key: value for key in keys if (value := os.environ.get(key)) is not None}


def _lock_path(database: Database) -> Path:
    raw = str(database.path)
    if raw == ":memory:":
        raise ServerValidationError("server validation requires a file-backed SQLite database")
    path = Path(raw)
    return path.with_suffix(path.suffix + ".host.lock")
