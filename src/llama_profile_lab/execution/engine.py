"""Sequential llama-bench execution with full-context placement resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from llama_profile_lab.db import (
    BenchmarkCaseRepository,
    BenchmarkRunRepository,
    CandidateRepository,
    Database,
    EnvironmentRepository,
    ExperimentRepository,
    MeasurementPolicyRepository,
    PlacementRepository,
    WorkloadCaseRepository,
)
from llama_profile_lab.db.records import (
    BinaryRecord,
    ResolvedPlacementRecord,
    RunStatus,
)
from llama_profile_lab.execution.host import detect_basic_host
from llama_profile_lab.execution.lock import HostLock
from llama_profile_lab.execution.placement import (
    PlacementConfigurationError,
    PlacementResolutionFailure,
    PlacementResolver,
    resolved_placement_from_record,
    validate_fixed_placement,
)
from llama_profile_lab.execution.process import (
    ProcessResult,
    ProcessRunner,
    ProcessRunnerError,
)
from llama_profile_lab.llama import CapabilitySet, sha256_file
from llama_profile_lab.llama.bench import (
    LlamaBenchAdapter,
    LlamaBenchConfigurationError,
    LlamaBenchParseError,
    parse_llama_bench_json,
)


class ExecutionError(RuntimeError):
    """Raised when an experiment cannot be executed safely."""


@dataclass(frozen=True, slots=True)
class ExecutionSummary:
    """Result of one run/resume invocation."""

    experiment_id: str
    attempted: int
    completed: int
    failed: int
    remaining: int
    interrupted: bool
    limited: bool


class ExperimentExecutor:
    """Execute incomplete cases sequentially with one placement per Candidate."""

    def __init__(
        self,
        database: Database,
        *,
        process_runner: ProcessRunner | None = None,
        bench_adapter: LlamaBenchAdapter | None = None,
    ) -> None:
        self.database = database
        self.process_runner = process_runner or ProcessRunner()
        self.bench_adapter = bench_adapter or LlamaBenchAdapter()

    def execute(
        self,
        experiment_id: str,
        *,
        binary_id: str,
        model_path: Path,
        fit_binary_id: str | None = None,
        timeout_seconds: float | None = None,
        fit_timeout_seconds: float | None = None,
        limit: int | None = None,
        resume: bool = False,
        cancel_event: Event | None = None,
    ) -> ExecutionSummary:
        """Execute only cases without a successful prior run."""
        if limit is not None and limit <= 0:
            raise ExecutionError("limit must be positive")
        lock_path = _lock_path(self.database)
        with HostLock(lock_path):
            return self._execute_locked(
                experiment_id,
                binary_id=binary_id,
                model_path=model_path,
                fit_binary_id=fit_binary_id,
                timeout_seconds=timeout_seconds,
                fit_timeout_seconds=fit_timeout_seconds,
                limit=limit,
                resume=resume,
                cancel_event=cancel_event,
            )

    def _execute_locked(
        self,
        experiment_id: str,
        *,
        binary_id: str,
        model_path: Path,
        fit_binary_id: str | None,
        timeout_seconds: float | None,
        fit_timeout_seconds: float | None,
        limit: int | None,
        resume: bool,
        cancel_event: Event | None,
    ) -> ExecutionSummary:
        with self.database.session() as connection:
            experiments = ExperimentRepository(connection)
            cases = BenchmarkCaseRepository(connection)
            runs = BenchmarkRunRepository(connection)
            placements = PlacementRepository(connection)
            environment = EnvironmentRepository(connection)
            candidates = CandidateRepository(connection)
            workloads = WorkloadCaseRepository(connection)
            policies = MeasurementPolicyRepository(connection)

            experiment = experiments.get(experiment_id)
            definition = experiments.get_definition(experiment_id)
            if experiment is None or definition is None:
                raise ExecutionError(f"experiment not found: {experiment_id}")

            allowed_states = {"planned"} if not resume else {
                "planned",
                "paused",
                "failed",
                "running",
            }
            if experiment.status not in allowed_states:
                raise ExecutionError(
                    f"experiment {experiment_id} is {experiment.status}; "
                    f"allowed states are {sorted(allowed_states)}"
                )

            bench_binary = environment.get_binary(binary_id)
            if bench_binary is None:
                raise ExecutionError(f"binary not found: {binary_id}")
            _verify_binary(bench_binary, expected_kind="llama-bench")

            fit_binary: BinaryRecord | None = None
            fixed_placement: ResolvedPlacementRecord | None = None
            if definition.placement_policy.type == "per-candidate":
                if fit_binary_id is None:
                    raise ExecutionError(
                        "per-candidate placement requires --fit-binary"
                    )
                fit_binary = environment.get_binary(fit_binary_id)
                if fit_binary is None:
                    raise ExecutionError(f"fit binary not found: {fit_binary_id}")
                _verify_binary(fit_binary, expected_kind="llama-fit-params")
            else:
                fixed_placement = placements.get(
                    definition.placement_policy.placement_id
                )
                if fixed_placement is None:
                    raise ExecutionError(
                        "fixed placement not found: "
                        f"{definition.placement_policy.placement_id}"
                    )

            resolved_model_path = model_path.expanduser().resolve()
            if not resolved_model_path.is_file():
                raise ExecutionError(f"model does not exist: {resolved_model_path}")

            policy = policies.get(definition.measurement_policy_id)
            if policy is None:
                raise ExecutionError(
                    f"measurement policy not found: {definition.measurement_policy_id}"
                )

            host = detect_basic_host()
            host_id = environment.put_host(
                hostname=host.hostname,
                hardware_fingerprint=host.hardware_fingerprint,
                cpu=host.cpu,
                ram_bytes=host.ram_bytes,
                gpus=host.gpus,
                os_info=host.os_info,
            )

            placements.recover_orphaned()
            runs.recover_orphaned(experiment_id)
            experiments.mark_running(experiment_id)

            resolver = PlacementResolver(
                placements,
                process_runner=self.process_runner,
            )
            placement_cache: dict[str, ResolvedPlacementRecord] = {}
            placement_failures: dict[str, RunStatus] = {}

            pending = cases.list_incomplete(experiment_id)
            attempted = 0
            completed = 0
            failed = 0
            interrupted = False
            limited = False

            for case in pending:
                if limit is not None and attempted >= limit:
                    limited = True
                    break

                candidate = candidates.get(case.candidate_id)
                workload = workloads.get(case.workload_case_id)
                if candidate is None or workload is None:
                    raise ExecutionError(
                        f"planned case {case.id} references missing immutable data"
                    )

                attempted += 1

                prior_placement_failure = placement_failures.get(case.candidate_id)
                if prior_placement_failure is not None:
                    cases.set_status(case.id, prior_placement_failure)
                    failed += 1
                    continue

                placement_record = placement_cache.get(case.candidate_id)
                if placement_record is None:
                    try:
                        if fixed_placement is not None:
                            validate_fixed_placement(
                                fixed_placement,
                                candidate=candidate,
                                host_id=host_id,
                            )
                            placement_record = fixed_placement
                        else:
                            if fit_binary is None:
                                raise ExecutionError(
                                    "fit binary disappeared during execution"
                                )
                            resolution = resolver.resolve_per_candidate(
                                candidate_id=case.candidate_id,
                                candidate=candidate,
                                host_id=host_id,
                                hardware_fingerprint=host.hardware_fingerprint,
                                fit_binary=fit_binary,
                                model_path=resolved_model_path,
                                timeout_seconds=fit_timeout_seconds,
                                cancel_event=cancel_event,
                            )
                            placement_record = resolution.record

                        cases.bind_placement_for_candidate(
                            experiment_id=experiment_id,
                            candidate_id=case.candidate_id,
                            placement_id=placement_record.id,
                        )
                        placement_cache[case.candidate_id] = placement_record
                    except PlacementConfigurationError:
                        placement_failures[case.candidate_id] = "invalid"
                        cases.set_status(case.id, "invalid")
                        failed += 1
                        continue
                    except PlacementResolutionFailure as exc:
                        case_status = _placement_failure_case_status(exc)
                        cases.set_status(case.id, case_status)
                        failed += 1
                        if case_status in {"interrupted", "cancelled"}:
                            interrupted = True
                            break
                        placement_failures[case.candidate_id] = case_status
                        continue

                placement = resolved_placement_from_record(placement_record)
                try:
                    argv = self.bench_adapter.build_argv(
                        binary_path=Path(bench_binary.path),
                        capabilities=_binary_capabilities(bench_binary),
                        model_path=resolved_model_path,
                        candidate=candidate,
                        workload=workload,
                        measurement_policy=policy,
                        placement=placement,
                    )
                except LlamaBenchConfigurationError as exc:
                    run_id = runs.create(
                        benchmark_case_id=case.id,
                        host_id=host_id,
                        binary_id=bench_binary.id,
                        measurement_policy_id=definition.measurement_policy_id,
                        argv=(),
                        environment=_captured_environment(),
                    )
                    runs.finish(
                        run_id,
                        status="invalid",
                        duration_ns=0,
                        exit_code=None,
                        stderr=str(exc),
                    )
                    cases.set_status(case.id, "invalid")
                    failed += 1
                    continue

                run_id = runs.create(
                    benchmark_case_id=case.id,
                    host_id=host_id,
                    binary_id=bench_binary.id,
                    measurement_policy_id=definition.measurement_policy_id,
                    argv=argv,
                    environment=_captured_environment(),
                )
                cases.set_status(case.id, "running")
                try:
                    process_result = self.process_runner.run(
                        argv,
                        timeout_seconds=timeout_seconds,
                        cancel_event=cancel_event,
                    )
                except ProcessRunnerError as exc:
                    runs.finish(
                        run_id,
                        status="benchmark_failed",
                        duration_ns=0,
                        exit_code=None,
                        stderr=str(exc),
                    )
                    cases.set_status(case.id, "benchmark_failed")
                    failed += 1
                    continue

                terminal_status = _process_failure_status(process_result)
                if terminal_status is not None:
                    runs.finish(
                        run_id,
                        status=terminal_status,
                        duration_ns=process_result.duration_ns,
                        exit_code=process_result.exit_code,
                        stdout=process_result.stdout,
                        stderr=process_result.stderr,
                        finished_at=process_result.finished_at,
                    )
                    cases.set_status(case.id, terminal_status)
                    failed += 1
                    if terminal_status in {"interrupted", "cancelled"}:
                        interrupted = True
                        break
                    continue

                try:
                    result = parse_llama_bench_json(
                        process_result.stdout,
                        expected_repetitions=policy.repetitions,
                    )
                except LlamaBenchParseError as exc:
                    runs.finish(
                        run_id,
                        status="parser_failed",
                        duration_ns=process_result.duration_ns,
                        exit_code=process_result.exit_code,
                        stdout=process_result.stdout,
                        stderr=_append_error(process_result.stderr, str(exc)),
                        finished_at=process_result.finished_at,
                    )
                    cases.set_status(case.id, "parser_failed")
                    failed += 1
                    continue

                runs.add_samples(
                    run_id,
                    tuple(
                        (sample.elapsed_ns, sample.tokens_per_second)
                        for sample in result.samples
                    ),
                )
                metrics: dict[str, int | float] = {}
                for name, value in (
                    ("avg_ns", result.avg_ns),
                    ("stddev_ns", result.stddev_ns),
                    ("avg_ts", result.avg_ts),
                    ("stddev_ts", result.stddev_ts),
                ):
                    if value is not None:
                        metrics[name] = value
                runs.add_metrics(run_id, metrics)
                runs.finish(
                    run_id,
                    status="completed",
                    duration_ns=process_result.duration_ns,
                    exit_code=process_result.exit_code,
                    stdout=process_result.stdout,
                    stderr=process_result.stderr,
                    raw_result=result.raw,
                    finished_at=process_result.finished_at,
                )
                cases.set_status(case.id, "completed")
                completed += 1

            remaining = cases.count_incomplete(experiment_id)
            if interrupted or limited:
                experiments.mark_paused(experiment_id)
            elif remaining == 0:
                experiments.mark_completed(experiment_id)
            else:
                experiments.mark_paused(experiment_id)

            return ExecutionSummary(
                experiment_id=experiment_id,
                attempted=attempted,
                completed=completed,
                failed=failed,
                remaining=remaining,
                interrupted=interrupted,
                limited=limited,
            )


def _verify_binary(binary: BinaryRecord, *, expected_kind: str) -> None:
    if binary.kind != expected_kind:
        raise ExecutionError(
            f"binary {binary.id} is {binary.kind}, not {expected_kind}"
        )
    path = Path(binary.path)
    if not path.is_file():
        raise ExecutionError(
            f"registered {expected_kind} path does not exist: {path}"
        )
    current_hash = sha256_file(path)
    if current_hash != binary.sha256:
        raise ExecutionError(
            f"registered {expected_kind} binary changed on disk; "
            "re-run binary inspect"
        )


def _binary_capabilities(binary: BinaryRecord) -> CapabilitySet:
    return CapabilitySet.from_mapping(
        dict(binary.capabilities),
        fallback_kind="llama-bench",
    )


def _placement_failure_case_status(
    failure: PlacementResolutionFailure,
) -> RunStatus:
    if failure.kind == "interrupted":
        return "interrupted"
    if failure.kind == "cancelled":
        return "cancelled"
    return "fit_failed"


def _process_failure_status(result: ProcessResult) -> RunStatus | None:
    if result.interrupted:
        return "interrupted"
    if result.cancelled:
        return "cancelled"
    if result.timed_out:
        return "timeout"
    if result.exit_code == 0:
        return None

    combined = f"{result.stdout}\n{result.stderr}".lower()
    if any(
        marker in combined
        for marker in (
            "out of memory",
            "cuda error: out of memory",
            "failed to allocate",
            "cannot allocate memory",
        )
    ):
        return "oom"
    if any(
        marker in combined
        for marker in (
            "failed to load model",
            "error loading model",
            "failed loading model",
        )
    ):
        return "load_failed"
    return "benchmark_failed"


_ENV_ALLOWLIST = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GGML_CUDA_ENABLE_UNIFIED_MEMORY",
    "OMP_NUM_THREADS",
)


def _captured_environment() -> dict[str, str]:
    return {
        name: os.environ[name]
        for name in _ENV_ALLOWLIST
        if name in os.environ
    }


def _lock_path(database: Database) -> Path:
    raw = str(database.path)
    if raw == ":memory:":
        raise ExecutionError("benchmark execution requires a file-backed SQLite database")
    runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
    return runtime_dir / f"llama-profile-lab-{os.getuid()}.host.lock"


def _append_error(stderr: str, error: str) -> str:
    if not stderr:
        return error
    return f"{stderr.rstrip()}\n{error}"
