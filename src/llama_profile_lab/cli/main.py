"""Top-level llprof command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from llama_profile_lab import __version__
from llama_profile_lab.analysis import (
    DEFAULT_METRIC_REGISTRY,
    AnalysisError,
    AnalysisFilter,
    AnalysisService,
    ParetoObjective,
    serialize_export,
)
from llama_profile_lab.db import (
    BenchmarkRunRepository,
    Database,
    EnvironmentRepository,
    PlacementRepository,
    TelemetryRepository,
)
from llama_profile_lab.db.records import BinaryRecord
from llama_profile_lab.domain.base import JsonScalar
from llama_profile_lab.execution import (
    ExecutionError,
    ExecutionSummary,
    ExperimentExecutor,
    HostLockError,
    ServerValidationError,
    ServerValidationService,
)
from llama_profile_lab.llama import (
    BinaryDiscoveryError,
    BinaryKind,
    BinaryProbe,
    CapabilitySet,
    compare_capabilities,
    discover_binary_paths,
    probe_binary,
)
from llama_profile_lab.planning import PlanningError, plan_experiment, render_plan_summary

_BINARY_KIND_CHOICES = (
    "auto",
    "llama-bench",
    "llama-fit-params",
    "llama-server",
    "speed-bench",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command-line parser."""
    parser = argparse.ArgumentParser(
        prog="llprof",
        description="Experiment, benchmark, and tune llama.cpp profiles.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    commands = parser.add_subparsers(dest="command")
    _add_experiment_parser(commands)
    _add_binary_parser(commands)
    _add_run_parser(commands)
    _add_placement_parser(commands)
    _add_results_parser(commands)
    _add_server_parser(commands)
    _add_api_parser(commands)
    return parser


def _add_experiment_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    experiment = commands.add_parser(
        "experiment",
        help="Create, plan, execute, and resume experiments.",
    )
    experiment_commands = experiment.add_subparsers(dest="experiment_command")

    plan = experiment_commands.add_parser(
        "plan",
        help="Expand and persist a draft experiment without executing benchmarks.",
    )
    plan.add_argument("experiment_id", help="Persisted experiment ID to plan.")
    _add_database_argument(plan)

    run = experiment_commands.add_parser(
        "run",
        help="Execute incomplete planned cases with a registered llama-bench binary.",
    )
    _add_execution_arguments(run)

    resume = experiment_commands.add_parser(
        "resume",
        help="Recover stale attempts and execute only cases without a successful run.",
    )
    _add_execution_arguments(resume)


def _add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("experiment_id")
    parser.add_argument(
        "--binary",
        required=True,
        dest="binary_id",
        help="Registered llama-bench binary ID.",
    )
    parser.add_argument(
        "--fit-binary",
        dest="fit_binary_id",
        default=None,
        help=(
            "Registered llama-fit-params binary ID. Required for "
            "per-candidate placement policy; unused for fixed placement."
        ),
    )
    parser.add_argument(
        "--model-path",
        required=True,
        type=Path,
        help=(
            "Target GGUF path for benchmark execution and placement fitting. "
            "Model-registry path resolution is introduced later."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Optional timeout for each llama-bench case.",
    )
    parser.add_argument(
        "--fit-timeout-seconds",
        type=float,
        default=None,
        help="Optional timeout for each llama-fit-params invocation.",
    )
    parser.add_argument(
        "--telemetry-interval-ms",
        type=int,
        default=1000,
        help="Telemetry sampling interval in milliseconds (minimum 500; default 1000).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Execute at most this many incomplete cases, then pause.",
    )
    _add_database_argument(parser)


def _add_binary_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    binary = commands.add_parser(
        "binary",
        help="Discover and inspect llama.cpp executables.",
    )
    binary_commands = binary.add_subparsers(dest="binary_command")

    inspect = binary_commands.add_parser(
        "inspect",
        help="Probe and register one or more explicit executable paths.",
    )
    inspect.add_argument("paths", type=Path, nargs="+")
    inspect.add_argument(
        "--kind",
        choices=_BINARY_KIND_CHOICES,
        default="auto",
        help="Override tool kind when it cannot be inferred from the filename.",
    )
    _add_database_argument(inspect)

    discover = binary_commands.add_parser(
        "discover",
        help="Find known llama.cpp tools in search directories and PATH.",
    )
    discover.add_argument(
        "--search-dir",
        action="append",
        type=Path,
        default=[],
        help="Directory containing llama.cpp binaries; may be repeated.",
    )
    discover.add_argument(
        "--no-path",
        action="store_true",
        help="Do not search the process PATH.",
    )
    _add_database_argument(discover)

    listing = binary_commands.add_parser(
        "list",
        help="List executables already registered in SQLite.",
    )
    _add_database_argument(listing)

    compare = binary_commands.add_parser(
        "compare",
        help="Compare supported arguments for two registered executables.",
    )
    compare.add_argument("left_id")
    compare.add_argument("right_id")
    _add_database_argument(compare)


def _add_run_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    run = commands.add_parser(
        "run",
        help="Inspect persisted benchmark runs.",
    )
    run_commands = run.add_subparsers(dest="run_command")
    show = run_commands.add_parser(
        "show",
        help="Show one benchmark attempt with samples and normalized metrics.",
    )
    show.add_argument("run_id")
    show.add_argument(
        "--logs",
        action="store_true",
        help="Include captured stdout/stderr.",
    )
    _add_database_argument(show)


def _add_placement_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    placement = commands.add_parser(
        "placement",
        help="Inspect cached resolved placements.",
    )
    placement_commands = placement.add_subparsers(dest="placement_command")

    listing = placement_commands.add_parser(
        "list",
        help="List successful cached placements.",
    )
    _add_database_argument(listing)

    show = placement_commands.add_parser(
        "show",
        help="Show one resolved placement.",
    )
    show.add_argument("placement_id")
    _add_database_argument(show)


def _add_results_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    results = commands.add_parser(
        "results",
        help="Analyze completed benchmark observations.",
    )
    result_commands = results.add_subparsers(dest="results_command")

    metrics = result_commands.add_parser(
        "metrics",
        help="List built-in analysis metrics.",
    )
    _add_database_argument(metrics)

    matrix = result_commands.add_parser(
        "matrix",
        help="Project one workload slice over two Candidate dimensions.",
    )
    matrix.add_argument("experiment_id")
    matrix.add_argument("--x", required=True, dest="x_path")
    matrix.add_argument("--y", required=True, dest="y_path")
    matrix.add_argument("--metric", required=True)
    matrix.add_argument("--facet", dest="facet_path", default=None)
    matrix.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        dest="format_name",
    )
    _add_analysis_filters(matrix)
    _add_database_argument(matrix)

    compare = result_commands.add_parser(
        "compare",
        help="Compare one Candidate with the experiment baseline.",
    )
    compare.add_argument("experiment_id")
    compare.add_argument("candidate_id")
    compare.add_argument("--baseline", dest="baseline_candidate_id", default=None)
    compare.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        default=[],
        help="Metric to compare; may be repeated.",
    )
    _add_analysis_filters(compare)
    _add_database_argument(compare)

    pareto = result_commands.add_parser(
        "pareto",
        help="Return the non-dominated set for explicit objectives.",
    )
    pareto.add_argument("experiment_id")
    pareto.add_argument(
        "--objective",
        action="append",
        required=True,
        help=(
            "KEY:DIRECTION:METRIC[@PATH=VALUE;PATH=VALUE], where DIRECTION "
            "is max/min or maximize/minimize."
        ),
    )
    _add_analysis_filters(pareto)
    _add_database_argument(pareto)

    latency = result_commands.add_parser(
        "latency",
        help="Estimate compute-only request latency from PP/TG curves.",
    )
    latency.add_argument("experiment_id")
    latency.add_argument("--candidate", required=True, dest="candidate_id")
    latency.add_argument("--prompt-tokens", type=int, required=True)
    latency.add_argument("--generate-tokens", type=int, required=True)
    latency.add_argument("--decode-start-depth-tokens", type=int, default=None)
    _add_analysis_filters(latency)
    _add_database_argument(latency)

    export = result_commands.add_parser(
        "export",
        help="Export summarized Candidate × workload observations.",
    )
    export.add_argument("experiment_id")
    export.add_argument("--format", choices=("csv", "json"), required=True, dest="format_name")
    export.add_argument("--output", type=Path, default=None)
    export.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        default=[],
        help="Metric to include; may be repeated. Defaults to the standard summary set.",
    )
    _add_analysis_filters(export)
    _add_database_argument(export)


def _add_analysis_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Exact filter PATH=VALUE; VALUE is parsed as JSON when possible.",
    )
    parser.add_argument(
        "--quality",
        action="append",
        default=[],
        help="Include only this run-quality label; may be repeated.",
    )


def _add_server_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    server = commands.add_parser(
        "server",
        help="Validate finalist Candidates with llama-server and SPEED-Bench.",
    )
    server_commands = server.add_subparsers(dest="server_command")

    validate = server_commands.add_parser(
        "validate",
        help="Launch one finalist Candidate and execute its SPEED-Bench workloads.",
    )
    validate.add_argument("experiment_id")
    validate.add_argument("candidate_id")
    validate.add_argument("--server-binary", required=True, dest="server_binary_id")
    validate.add_argument(
        "--speed-bench-binary",
        required=True,
        dest="speed_bench_binary_id",
    )
    validate.add_argument("--model-path", required=True, type=Path)
    validate.add_argument("--draft-model-path", type=Path, default=None)
    validate.add_argument("--placement", dest="placement_id", default=None)
    validate.add_argument("--model-name", default=None)
    validate.add_argument("--host", default="127.0.0.1")
    validate.add_argument("--port", type=int, default=8080)
    validate.add_argument("--readiness-timeout-seconds", type=float, default=300.0)
    validate.add_argument("--request-timeout-seconds", type=float, default=600.0)
    validate.add_argument("--benchmark-timeout-seconds", type=float, default=None)
    validate.add_argument("--workload-case", dest="workload_case_id", default=None)
    _add_database_argument(validate)

    compare = server_commands.add_parser(
        "compare",
        help="Compare completed baseline and speculative server measurements.",
    )
    compare.add_argument("experiment_id")
    compare.add_argument("baseline_candidate_id")
    compare.add_argument("speculative_candidate_id")
    compare.add_argument("--workload-case", dest="workload_case_id", default=None)
    compare.add_argument("--category", default="all")
    _add_database_argument(compare)


def _add_api_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    api = commands.add_parser(
        "api",
        help="Serve the local FastAPI application.",
    )
    api.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address; defaults to loopback only.",
    )
    api.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP port (default 8000).",
    )
    api.add_argument(
        "--launcher-config",
        type=Path,
        default=None,
        help="Optional llama-profile-launcher host JSON for read-only profile endpoints.",
    )
    _add_database_argument(api)


def _add_database_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/benchmarks.db"),
        help="SQLite database path (default: data/benchmarks.db).",
    )


def _plan_command(database_path: Path, experiment_id: str) -> int:
    database = Database(database_path)
    try:
        with database.session() as connection:
            summary = plan_experiment(connection, experiment_id)
    except PlanningError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(render_plan_summary(summary))
    return 0


def _execute_command(
    database_path: Path,
    experiment_id: str,
    *,
    binary_id: str,
    fit_binary_id: str | None,
    model_path: Path,
    timeout_seconds: float | None,
    fit_timeout_seconds: float | None,
    limit: int | None,
    resume: bool,
    telemetry_interval_ms: int,
) -> int:
    try:
        summary = ExperimentExecutor(Database(database_path)).execute(
            experiment_id,
            binary_id=binary_id,
            fit_binary_id=fit_binary_id,
            model_path=model_path,
            timeout_seconds=timeout_seconds,
            fit_timeout_seconds=fit_timeout_seconds,
            limit=limit,
            resume=resume,
            telemetry_interval_seconds=telemetry_interval_ms / 1000.0,
        )
    except (ExecutionError, HostLockError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(_render_execution_summary(summary))
    if summary.interrupted:
        return 130
    if summary.failed and not summary.limited:
        return 1
    return 0


def _render_execution_summary(summary: ExecutionSummary) -> str:
    state = "interrupted" if summary.interrupted else "paused" if summary.limited else "done"
    return "\n".join(
        (
            f"Experiment: {summary.experiment_id}",
            f"Attempted: {summary.attempted}",
            f"Completed: {summary.completed}",
            f"Failed: {summary.failed}",
            f"Remaining: {summary.remaining}",
            f"State: {state}",
        )
    )


def _run_show_command(
    database_path: Path,
    run_id: str,
    *,
    include_logs: bool,
) -> int:
    with Database(database_path).session() as connection:
        repository = BenchmarkRunRepository(connection)
        record = repository.get(run_id)
        if record is None:
            print(f"error: run not found: {run_id}", file=sys.stderr)
            return 2
        samples = repository.samples(run_id)
        metrics = repository.metrics(run_id)
        telemetry_samples = TelemetryRepository(connection).samples(run_id)
        logs = repository.logs(run_id) if include_logs else None

    print(f"Run: {record.id}")
    print(f"Case: {record.benchmark_case_id}")
    print(f"Status: {record.status}")
    print(f"Binary: {record.binary_id}")
    print(f"Host: {record.host_id}")
    print(f"Started: {record.started_at}")
    print(f"Finished: {record.finished_at or '-'}")
    print(f"Duration ns: {record.duration_ns if record.duration_ns is not None else '-'}")
    print(f"Exit code: {record.exit_code if record.exit_code is not None else '-'}")
    print(f"Quality: {record.quality or '-'}")
    print(f"Telemetry samples: {len(telemetry_samples)}")
    if record.quality_details is not None:
        reasons = record.quality_details.get("reasons", [])
        if isinstance(reasons, list) and reasons:
            print("Quality reasons:")
            for reason in reasons:
                print(f"  {reason}")
    print(f"Samples: {len(samples)}")
    for index, elapsed_ns, throughput in samples:
        print(f"  {index}: {throughput:.6f} t/s ({elapsed_ns} ns)")
    if metrics:
        print("Metrics:")
        for name, value in sorted(metrics.items()):
            print(f"  {name}: {value}")
    if logs is not None:
        stdout, stderr = logs
        print("Stdout:")
        print(stdout.rstrip())
        print("Stderr:")
        print(stderr.rstrip())
    return 0


def _results_metrics_command() -> int:
    for definition in DEFAULT_METRIC_REGISTRY.definitions():
        print(
            f"{definition.name}\t{definition.label}\t"
            f"{definition.unit or '-'}"
        )
    return 0


def _results_matrix_command(
    database_path: Path,
    experiment_id: str,
    *,
    x_path: str,
    y_path: str,
    metric: str,
    facet_path: str | None,
    filter_args: Sequence[str],
    qualities: Sequence[str],
    format_name: str,
) -> int:
    try:
        filters = _parse_filters(filter_args)
        projection = AnalysisService(Database(database_path)).matrix(
            experiment_id,
            x_path=x_path,
            y_path=y_path,
            metric=metric,
            filters=filters,
            facet_path=facet_path,
            qualities=qualities,
        )
    except AnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if format_name == "json":
        print(projection.model_dump_json(indent=2))
    else:
        print(_render_matrix(projection))
    return 0


def _results_compare_command(
    database_path: Path,
    experiment_id: str,
    candidate_id: str,
    *,
    baseline_candidate_id: str | None,
    metric_names: Sequence[str],
    filter_args: Sequence[str],
    qualities: Sequence[str],
) -> int:
    metrics = tuple(metric_names) or ("throughput.median",)
    try:
        comparison = AnalysisService(Database(database_path)).compare(
            experiment_id,
            candidate_id=candidate_id,
            baseline_candidate_id=baseline_candidate_id,
            metric_names=metrics,
            filters=_parse_filters(filter_args),
            qualities=qualities,
        )
    except AnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Candidate: {comparison.candidate_id}")
    print(f"Baseline: {comparison.baseline_candidate_id}")
    for delta in comparison.deltas:
        current = "-" if delta.candidate_value is None else _format_number(delta.candidate_value)
        baseline = "-" if delta.baseline_value is None else _format_number(delta.baseline_value)
        percent = "-" if delta.percent_delta is None else f"{delta.percent_delta:+.2f}%"
        print(
            f"[{delta.suite_case_index}] {delta.workload_label}  "
            f"{delta.metric}: {current} vs {baseline} ({percent})"
        )
    return 0


def _results_pareto_command(
    database_path: Path,
    experiment_id: str,
    *,
    objective_args: Sequence[str],
    filter_args: Sequence[str],
    qualities: Sequence[str],
) -> int:
    try:
        objectives = tuple(_parse_objective(value) for value in objective_args)
        result = AnalysisService(Database(database_path)).pareto(
            experiment_id,
            objectives=objectives,
            filters=_parse_filters(filter_args),
            qualities=qualities,
        )
    except (AnalysisError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Evaluated candidates: {result.evaluated_count}")
    print(f"Pareto frontier: {len(result.frontier)}")
    for candidate in result.frontier:
        values = "  ".join(
            f"{key}={_format_number(candidate.values[key])}"
            for key in candidate.values
        )
        print(f"{candidate.candidate_id}  {values}")
    if result.excluded:
        print("Excluded for incomplete objective data:")
        for candidate_id, reason in sorted(result.excluded.items()):
            print(f"  {candidate_id}: {reason}")
    return 0


def _results_latency_command(
    database_path: Path,
    experiment_id: str,
    *,
    candidate_id: str,
    prompt_tokens: int,
    generate_tokens: int,
    decode_start_depth_tokens: int | None,
    filter_args: Sequence[str],
    qualities: Sequence[str],
) -> int:
    try:
        estimate = AnalysisService(Database(database_path)).latency(
            experiment_id,
            candidate_id=candidate_id,
            prompt_tokens=prompt_tokens,
            generate_tokens=generate_tokens,
            decode_start_depth_tokens=decode_start_depth_tokens,
            filters=_parse_filters(filter_args),
            qualities=qualities,
        )
    except AnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Candidate: {estimate.candidate_id}")
    print(f"Prefill: {estimate.prefill_seconds:.6f} s")
    print(f"Decode: {estimate.decode_seconds:.6f} s")
    print(f"Total: {estimate.total_seconds:.6f} s")
    print(f"Interpolated PP: {estimate.prefill_tokens_per_second:.6f} t/s")
    print(
        "Average TG over requested interval: "
        f"{estimate.decode_average_tokens_per_second:.6f} t/s"
    )
    return 0


def _results_export_command(
    database_path: Path,
    experiment_id: str,
    *,
    format_name: str,
    output: Path | None,
    metric_names: Sequence[str],
    filter_args: Sequence[str],
    qualities: Sequence[str],
) -> int:
    try:
        rows = AnalysisService(Database(database_path)).export_rows(
            experiment_id,
            filters=_parse_filters(filter_args),
            qualities=qualities,
            metric_names=tuple(metric_names) or None,
        )
        rendered = serialize_export(rows, format_name=format_name)
    except AnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if output is None:
        print(rendered, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {len(rows)} rows to {output}")
    return 0


def _parse_filters(values: Sequence[str]) -> tuple[AnalysisFilter, ...]:
    return tuple(_parse_filter(value) for value in values)


def _parse_filter(value: str) -> AnalysisFilter:
    path, separator, raw_value = value.partition("=")
    if not separator or not path:
        raise AnalysisError(f"invalid filter {value!r}; expected PATH=VALUE")
    return AnalysisFilter(path=path, value=_parse_scalar(raw_value))


def _parse_scalar(value: str) -> JsonScalar:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    if parsed is None or isinstance(parsed, (str, int, float, bool)):
        return parsed
    raise AnalysisError("analysis filter values must be JSON scalars")


def _parse_objective(value: str) -> ParetoObjective:
    head, separator, raw_filters = value.partition("@")
    parts = head.split(":", 2)
    if len(parts) != 3 or any(not part for part in parts):
        raise AnalysisError(
            f"invalid objective {value!r}; expected KEY:DIRECTION:METRIC[@FILTERS]"
        )
    key, raw_direction, metric = parts
    direction_map: dict[str, Literal["maximize", "minimize"]] = {
        "max": "maximize",
        "maximize": "maximize",
        "min": "minimize",
        "minimize": "minimize",
    }
    direction = direction_map.get(raw_direction)
    if direction is None:
        raise AnalysisError(f"invalid Pareto direction: {raw_direction}")
    objective_filters = (
        _parse_filters(tuple(item for item in raw_filters.split(";") if item))
        if separator
        else ()
    )
    return ParetoObjective(
        key=key,
        direction=direction,
        metric=metric,
        filters=objective_filters,
    )


def _render_matrix(projection: object) -> str:
    from llama_profile_lab.analysis import MatrixProjection

    if not isinstance(projection, MatrixProjection):
        raise TypeError("expected MatrixProjection")
    lines = [
        f"Experiment: {projection.experiment_id}",
        f"Metric: {projection.metric}",
        f"X: {projection.x_path}",
        f"Y: {projection.y_path}",
    ]
    for facet in projection.facets:
        if projection.facet_path is not None:
            lines.append(f"Facet {projection.facet_path}={facet.value}")
        by_coordinate = {(cell.x, cell.y): cell for cell in facet.cells}
        header = ["Y \\ X", *[str(value) for value in projection.x_values]]
        lines.append("\t".join(header))
        for y_value in projection.y_values:
            row = [str(y_value)]
            for x_value in projection.x_values:
                cell = by_coordinate.get((x_value, y_value))
                row.append("" if cell is None else _format_number(cell.value))
            lines.append("\t".join(row))
    return "\n".join(lines)


def _format_number(value: float) -> str:
    return f"{value:.6g}"


def _server_validate_command(
    database_path: Path,
    experiment_id: str,
    candidate_id: str,
    *,
    server_binary_id: str,
    speed_bench_binary_id: str,
    model_path: Path,
    draft_model_path: Path | None,
    placement_id: str | None,
    model_name: str | None,
    host: str,
    port: int,
    readiness_timeout_seconds: float,
    request_timeout_seconds: float,
    benchmark_timeout_seconds: float | None,
    workload_case_id: str | None,
) -> int:
    try:
        summary = ServerValidationService(Database(database_path)).validate(
            experiment_id,
            candidate_id=candidate_id,
            server_binary_id=server_binary_id,
            speed_bench_binary_id=speed_bench_binary_id,
            model_path=model_path,
            draft_model_path=draft_model_path,
            placement_id=placement_id,
            model_name=model_name,
            host=host,
            port=port,
            readiness_timeout_seconds=readiness_timeout_seconds,
            request_timeout_seconds=request_timeout_seconds,
            benchmark_timeout_seconds=benchmark_timeout_seconds,
            workload_case_id=workload_case_id,
        )
    except (ServerValidationError, HostLockError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Server run: {summary.server_run_id}")
    print(f"Candidate: {summary.candidate_id}")
    print(f"SPEED-Bench invocations: {len(summary.benchmark_ids)}")
    print(f"Speculative: {'yes' if summary.speculative else 'no'}")
    print(f"Status: {'completed' if summary.completed else 'failed'}")
    return 0 if summary.completed else 1


def _server_compare_command(
    database_path: Path,
    experiment_id: str,
    baseline_candidate_id: str,
    speculative_candidate_id: str,
    *,
    workload_case_id: str | None,
    category: str,
) -> int:
    try:
        result = ServerValidationService(Database(database_path)).compare(
            experiment_id,
            baseline_candidate_id=baseline_candidate_id,
            speculative_candidate_id=speculative_candidate_id,
            workload_case_id=workload_case_id,
            category=category,
        )
    except ServerValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Workload: {result.workload_case_id}")
    print(f"Category: {result.category}")
    print(f"Baseline Candidate: {result.baseline_candidate_id}")
    print(f"Speculative Candidate: {result.speculative_candidate_id}")
    print(f"Baseline prompt t/s: {_optional_number(result.baseline_prompt_ts)}")
    print(f"Speculative prompt t/s: {_optional_number(result.speculative_prompt_ts)}")
    print(f"Baseline decode t/s: {_optional_number(result.baseline_pred_ts)}")
    print(f"Speculative decode t/s: {_optional_number(result.speculative_pred_ts)}")
    print(f"Baseline latency ms: {_optional_number(result.baseline_latency_ms)}")
    print(f"Speculative latency ms: {_optional_number(result.speculative_latency_ms)}")
    print(f"Decode speedup: {_optional_ratio(result.decode_speedup)}")
    print(f"Latency speedup: {_optional_ratio(result.latency_speedup)}")
    print(f"Drafted tokens: {result.draft_n if result.draft_n is not None else '-'}")
    print(f"Accepted tokens: {result.accepted_n if result.accepted_n is not None else '-'}")
    print(
        "Acceptance rate: "
        + ("-" if result.accept_rate is None else f"{result.accept_rate:.4f}")
    )
    return 0


def _optional_number(value: float | None) -> str:
    return "-" if value is None else _format_number(value)


def _optional_ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}x"


def _api_command(
    database_path: Path,
    *,
    host: str,
    port: int,
    launcher_config: Path | None,
) -> int:
    if not 1 <= port <= 65535:
        print("error: API port must be between 1 and 65535", file=sys.stderr)
        return 2

    import uvicorn

    from llama_profile_lab.api import create_app

    app = create_app(
        database_path,
        launcher_config_path=launcher_config,
    )
    uvicorn.run(app, host=host, port=port)
    return 0


def _placement_list_command(database_path: Path) -> int:
    with Database(database_path).session() as connection:
        records = PlacementRepository(connection).list()

    if not records:
        print("No cached resolved placements.")
        return 0

    for record in records:
        print(
            f"{record.id}  ctx={record.production_context_size}  "
            f"ngl={record.n_gpu_layers}  candidate={record.candidate_id}"
        )
    return 0


def _placement_show_command(database_path: Path, placement_id: str) -> int:
    with Database(database_path).session() as connection:
        record = PlacementRepository(connection).get(placement_id)

    if record is None:
        print(f"error: placement not found: {placement_id}", file=sys.stderr)
        return 2

    print(f"Placement: {record.id}")
    print(f"Candidate: {record.candidate_id}")
    print(f"Host: {record.host_id}")
    print(f"Fit binary: {record.binary_id}")
    print(f"Fit attempt: {record.fit_attempt_id or '-'}")
    print(f"Production context: {record.production_context_size}")
    print(f"GPU layers: {record.n_gpu_layers}")
    print(f"CPU MoE: {record.n_cpu_moe}")
    print(f"Split mode: {record.split_mode}")
    print(f"Main GPU: {record.main_gpu}")
    if record.devices == "auto":
        print("Devices: auto")
    else:
        print(f"Devices: {','.join(record.devices)}")
    print(
        "Tensor split: "
        + (
            "auto"
            if record.tensor_split is None
            else ",".join(str(value) for value in record.tensor_split)
        )
    )
    print(f"Tensor overrides: {len(record.override_tensor)}")
    print(f"Created: {record.created_at}")
    return 0


def _binary_inspect_command(
    database_path: Path,
    paths: Sequence[Path],
    kind_arg: str,
) -> int:
    explicit_kind = _parse_kind_argument(kind_arg)
    database = Database(database_path)
    try:
        probes = tuple(probe_binary(path, kind=explicit_kind) for path in paths)
    except BinaryDiscoveryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    with database.session() as connection:
        repository = EnvironmentRepository(connection)
        for index, probe in enumerate(probes):
            identifier = _persist_probe(repository, probe)
            if index:
                print()
            print(_render_probe(identifier, probe))
    return 0


def _binary_discover_command(
    database_path: Path,
    search_dirs: Sequence[Path],
    *,
    include_path: bool,
) -> int:
    paths = discover_binary_paths(tuple(search_dirs), include_path=include_path)
    if not paths:
        print("No llama.cpp binaries found.", file=sys.stderr)
        return 1
    return _binary_inspect_command(database_path, paths, "auto")


def _binary_list_command(database_path: Path) -> int:
    with Database(database_path).session() as connection:
        records = EnvironmentRepository(connection).list_binaries()

    if not records:
        print("No registered llama.cpp binaries.")
        return 0

    for record in records:
        capabilities = _capabilities_from_record(record)
        print(
            f"{record.id}  {record.kind}  "
            f"{len(capabilities.options)} options  {record.path}"
        )
    return 0


def _binary_compare_command(
    database_path: Path,
    left_id: str,
    right_id: str,
) -> int:
    with Database(database_path).session() as connection:
        repository = EnvironmentRepository(connection)
        left = repository.get_binary(left_id)
        right = repository.get_binary(right_id)

    if left is None:
        print(f"error: binary not found: {left_id}", file=sys.stderr)
        return 2
    if right is None:
        print(f"error: binary not found: {right_id}", file=sys.stderr)
        return 2

    diff = compare_capabilities(
        _capabilities_from_record(left),
        _capabilities_from_record(right),
    )
    print(f"Left:  {left.id} ({left.kind})")
    print(f"Right: {right.id} ({right.kind})")
    print(f"Common: {len(diff.common)}")
    print("Only left:")
    _print_options(diff.only_left)
    print("Only right:")
    _print_options(diff.only_right)
    return 0


def _persist_probe(repository: EnvironmentRepository, probe: BinaryProbe) -> str:
    return repository.put_binary(
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


def _render_probe(identifier: str, probe: BinaryProbe) -> str:
    lines = [
        f"ID: {identifier}",
        f"Kind: {probe.kind}",
        f"Path: {probe.path}",
        f"SHA-256: {probe.sha256}",
        f"Size: {probe.size_bytes} bytes",
        f"Build: {probe.build_number or 'unknown'}",
        f"Commit: {probe.git_commit or 'unknown'}",
        f"Supported options ({len(probe.capabilities.options)}):",
    ]
    lines.extend(f"  {option}" for option in sorted(probe.capabilities.options))
    return "\n".join(lines)


def _capabilities_from_record(record: BinaryRecord) -> CapabilitySet:
    return CapabilitySet.from_mapping(
        dict(record.capabilities),
        fallback_kind=_record_kind(record.kind),
    )


def _record_kind(value: str) -> BinaryKind:
    if value not in {"llama-bench", "llama-fit-params", "llama-server", "speed-bench"}:
        raise ValueError(f"unknown persisted binary kind: {value}")
    return cast(BinaryKind, value)


def _parse_kind_argument(value: str) -> BinaryKind | None:
    if value == "auto":
        return None
    return _record_kind(value)


def _print_options(options: Sequence[str]) -> None:
    if not options:
        print("  (none)")
        return
    for option in options:
        print(f"  {option}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the llprof command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "experiment":
        if args.experiment_command == "plan":
            return _plan_command(args.database, args.experiment_id)
        if args.experiment_command in {"run", "resume"}:
            return _execute_command(
                args.database,
                args.experiment_id,
                binary_id=args.binary_id,
                fit_binary_id=args.fit_binary_id,
                model_path=args.model_path,
                timeout_seconds=args.timeout_seconds,
                fit_timeout_seconds=args.fit_timeout_seconds,
                limit=args.limit,
                resume=args.experiment_command == "resume",
                telemetry_interval_ms=args.telemetry_interval_ms,
            )

    if args.command == "binary":
        if args.binary_command == "inspect":
            return _binary_inspect_command(args.database, args.paths, args.kind)
        if args.binary_command == "discover":
            return _binary_discover_command(
                args.database,
                args.search_dir,
                include_path=not args.no_path,
            )
        if args.binary_command == "list":
            return _binary_list_command(args.database)
        if args.binary_command == "compare":
            return _binary_compare_command(args.database, args.left_id, args.right_id)

    if args.command == "run" and args.run_command == "show":
        return _run_show_command(args.database, args.run_id, include_logs=args.logs)

    if args.command == "placement":
        if args.placement_command == "list":
            return _placement_list_command(args.database)
        if args.placement_command == "show":
            return _placement_show_command(args.database, args.placement_id)

    if args.command == "server":
        if args.server_command == "validate":
            return _server_validate_command(
                args.database,
                args.experiment_id,
                args.candidate_id,
                server_binary_id=args.server_binary_id,
                speed_bench_binary_id=args.speed_bench_binary_id,
                model_path=args.model_path,
                draft_model_path=args.draft_model_path,
                placement_id=args.placement_id,
                model_name=args.model_name,
                host=args.host,
                port=args.port,
                readiness_timeout_seconds=args.readiness_timeout_seconds,
                request_timeout_seconds=args.request_timeout_seconds,
                benchmark_timeout_seconds=args.benchmark_timeout_seconds,
                workload_case_id=args.workload_case_id,
            )
        if args.server_command == "compare":
            return _server_compare_command(
                args.database,
                args.experiment_id,
                args.baseline_candidate_id,
                args.speculative_candidate_id,
                workload_case_id=args.workload_case_id,
                category=args.category,
            )

    if args.command == "api":
        return _api_command(
            args.database,
            host=args.host,
            port=args.port,
            launcher_config=args.launcher_config,
        )

    if args.command == "results":
        if args.results_command == "metrics":
            return _results_metrics_command()
        if args.results_command == "matrix":
            return _results_matrix_command(
                args.database,
                args.experiment_id,
                x_path=args.x_path,
                y_path=args.y_path,
                metric=args.metric,
                facet_path=args.facet_path,
                filter_args=args.filter,
                qualities=args.quality,
                format_name=args.format_name,
            )
        if args.results_command == "compare":
            return _results_compare_command(
                args.database,
                args.experiment_id,
                args.candidate_id,
                baseline_candidate_id=args.baseline_candidate_id,
                metric_names=args.metrics,
                filter_args=args.filter,
                qualities=args.quality,
            )
        if args.results_command == "pareto":
            return _results_pareto_command(
                args.database,
                args.experiment_id,
                objective_args=args.objective,
                filter_args=args.filter,
                qualities=args.quality,
            )
        if args.results_command == "latency":
            return _results_latency_command(
                args.database,
                args.experiment_id,
                candidate_id=args.candidate_id,
                prompt_tokens=args.prompt_tokens,
                generate_tokens=args.generate_tokens,
                decode_start_depth_tokens=args.decode_start_depth_tokens,
                filter_args=args.filter,
                qualities=args.quality,
            )
        if args.results_command == "export":
            return _results_export_command(
                args.database,
                args.experiment_id,
                format_name=args.format_name,
                output=args.output,
                metric_names=args.metrics,
                filter_args=args.filter,
                qualities=args.quality,
            )

    return 0
