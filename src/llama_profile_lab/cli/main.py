"""Top-level llprof command-line interface."""

from __future__ import annotations

import argparse
import csv
import io
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
    DeploymentAnalysisError,
    ParetoObjective,
    serialize_export,
)
from llama_profile_lab.api.deployment_operations import (
    DeploymentOperationError,
    DeploymentOperationManager,
    DeploymentOperationSnapshot,
    DeploymentOperationSpec,
)
from llama_profile_lab.api.dto import DeploymentCreateRequest
from llama_profile_lab.api.operations import OperationManager
from llama_profile_lab.api.service import (
    ApiConflictError,
    ApiNotFoundError,
    ApiService,
    parse_deployment_constraints,
    parse_deployment_filters,
    parse_deployment_objectives,
)
from llama_profile_lab.api.profiles import LauncherProfileError, LauncherProfileProvider
from llama_profile_lab.archive import (
    ArchiveError,
    ArchiveService,
    serialize_deployment_export,
    serialize_experiment_export,
)
from llama_profile_lab.db import (
    BenchmarkRunRepository,
    Database,
    EnvironmentRepository,
    PlacementRepository,
    TelemetryRepository,
)
from llama_profile_lab.db.records import BinaryRecord
from llama_profile_lab.diagnostics import inspect_database
from llama_profile_lab.domain import (
    DeploymentCandidate,
    DeploymentSearchSpace,
)
from llama_profile_lab.domain.base import JsonScalar
from llama_profile_lab.execution import (
    ConcurrentDeploymentError,
    ConcurrentDeploymentExecutor,
    ConcurrentDeploymentSummary,
    DeploymentExecutionError,
    DeploymentExecutionSummary,
    DeploymentExecutor,
    DeploymentServerInput,
    DeviceInventoryError,
    DeviceInventoryService,
    ExecutionError,
    ExecutionSummary,
    ExperimentExecutor,
    HostLockError,
    MemoryEstimatorError,
    MemoryEstimatorService,
    ServerValidationError,
    ServerValidationService,
    StandaloneBaselineInput,
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
from llama_profile_lab.planning import (
    DeploymentEstimatorInput,
    DeploymentPlannerService,
    DeploymentPlanSummary,
    PlanningError,
    plan_experiment,
    render_plan_summary,
)
from llama_profile_lab.promotion import PromotionError, PromotionService

_BINARY_KIND_CHOICES = (
    "auto",
    "llama-bench",
    "llama-fit-params",
    "llama-memory-estimator",
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
    _add_profile_parser(commands)
    _add_binary_parser(commands)
    _add_run_parser(commands)
    _add_placement_parser(commands)
    _add_deployment_parser(commands)
    _add_results_parser(commands)
    _add_server_parser(commands)
    _add_api_parser(commands)
    _add_ui_parser(commands)
    _add_database_parser(commands)
    _add_archive_parser(commands)
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

    export = experiment_commands.add_parser(
        "export",
        help="Export complete persisted provenance for one experiment as JSON.",
    )
    export.add_argument("experiment_id")
    export.add_argument("--output", type=Path, default=None)
    _add_database_argument(export)


def _add_profile_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    profile = commands.add_parser(
        "profile",
        help="Inspect launcher profiles and generate validated promotion patches.",
    )
    profile_commands = profile.add_subparsers(dest="profile_command")

    promote = profile_commands.add_parser(
        "promote",
        help="Persist and render a launcher patch for a server-validated Candidate.",
    )
    promote.add_argument("experiment_id")
    promote.add_argument("candidate_id")
    promote.add_argument(
        "--launcher-config",
        type=Path,
        required=True,
        help="llama-profile-launcher host JSON used as the immutable patch source.",
    )
    promote.add_argument("--source-profile", dest="source_profile_id", default=None)
    promote.add_argument("--output", type=Path, default=None)
    _add_database_argument(promote)


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

    devices = binary_commands.add_parser(
        "devices",
        help="Discover logical devices exposed by one exact registered binary.",
    )
    devices.add_argument("binary_id")
    devices.add_argument("--timeout-seconds", type=float, default=30.0)
    _add_database_argument(devices)


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

    estimate = placement_commands.add_parser(
        "estimate",
        help="Run or reuse a structured per-device memory estimate.",
    )
    estimate.add_argument("candidate_id")
    estimate.add_argument(
        "--helper-binary",
        required=True,
        dest="helper_binary_id",
        help="Registered llama-memory-estimator binary ID.",
    )
    estimate.add_argument("--model-path", required=True, type=Path)
    estimate.add_argument(
        "--device",
        action="append",
        default=None,
        dest="devices",
        help="Ordered logical device name; repeat to select multiple devices.",
    )
    estimate.add_argument("--timeout-seconds", type=float, default=300.0)
    _add_database_argument(estimate)


def _add_deployment_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    deployment = commands.add_parser(
        "deployment",
        help="Create, plan, run, inspect, and analyze multi-model deployments.",
    )
    deployment_commands = deployment.add_subparsers(
        dest="deployment_command"
    )

    create = deployment_commands.add_parser(
        "create",
        help="Persist one immutable base deployment definition.",
    )
    create.add_argument(
        "spec",
        type=Path,
        help="JSON DeploymentCandidate or {deployment: ...} document.",
    )
    _add_database_argument(create)

    preview = deployment_commands.add_parser(
        "preview",
        help="Evaluate deployment search counts without persisting plan cases.",
    )
    preview.add_argument(
        "spec",
        type=Path,
        help="JSON deployment planning specification.",
    )
    _add_database_argument(preview)

    plan = deployment_commands.add_parser(
        "plan",
        help="Persist feasible deployment cases and rejection history.",
    )
    plan.add_argument(
        "spec",
        type=Path,
        help="JSON deployment planning specification.",
    )
    _add_database_argument(plan)

    show = deployment_commands.add_parser(
        "show",
        help="Show one base deployment definition and computed state.",
    )
    show.add_argument("deployment_id")
    show.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        dest="format_name",
    )
    _add_database_argument(show)

    placement = deployment_commands.add_parser(
        "placement",
        help="Show planned placement memory matrices.",
    )
    placement.add_argument("deployment_id")
    placement.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        dest="format_name",
    )
    _add_database_argument(placement)

    run = deployment_commands.add_parser(
        "run",
        help="Run a durable concurrent deployment benchmark.",
    )
    run.add_argument("deployment_id")
    run.add_argument(
        "spec",
        type=Path,
        help="JSON concurrent deployment benchmark specification.",
    )
    _add_database_argument(run)

    pause = deployment_commands.add_parser(
        "pause",
        help="Request cooperative pause of an active deployment run.",
    )
    pause.add_argument("deployment_id")
    _add_database_argument(pause)

    resume = deployment_commands.add_parser(
        "resume",
        help="Resume the latest paused deployment operation.",
    )
    resume.add_argument("deployment_id")
    resume.add_argument(
        "--spec",
        type=Path,
        default=None,
        help="Optional replacement benchmark spec; defaults to persisted request.",
    )
    _add_database_argument(resume)

    cancel = deployment_commands.add_parser(
        "cancel",
        help="Cancel an active or paused deployment operation.",
    )
    cancel.add_argument("deployment_id")
    _add_database_argument(cancel)

    results = deployment_commands.add_parser(
        "results",
        help="Show raw M7 deployment analysis rows.",
    )
    results.add_argument("deployment_id")
    results.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Exact deployment analysis filter PATH=VALUE.",
    )
    results.add_argument(
        "--format",
        choices=("table", "json", "csv"),
        default="table",
        dest="format_name",
    )
    results.add_argument("--output", type=Path, default=None)
    _add_database_argument(results)

    export = deployment_commands.add_parser(
        "export",
        help="Export complete V2 deployment provenance as deterministic JSON.",
    )
    export.add_argument("deployment_id")
    export.add_argument("--output", type=Path, default=None)
    _add_database_argument(export)

    pareto = deployment_commands.add_parser(
        "pareto",
        help="Show the constrained deployment Pareto frontier.",
    )
    pareto.add_argument("deployment_id")
    pareto.add_argument(
        "--objective",
        action="append",
        required=True,
        help="KEY:DIRECTION:METRIC[@PATH=VALUE;PATH=VALUE].",
    )
    pareto.add_argument(
        "--constraint",
        action="append",
        default=[],
        help="METRIC:OP:VALUE[@PATH=VALUE;PATH=VALUE].",
    )
    pareto.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Exact deployment analysis filter PATH=VALUE.",
    )
    pareto.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        dest="format_name",
    )
    _add_database_argument(pareto)

    execute = deployment_commands.add_parser(
        "execute",
        help="Launch every server in one persisted deployment placement.",
    )
    execute.add_argument(
        "spec",
        type=Path,
        help="JSON deployment execution specification.",
    )
    _add_database_argument(execute)

    benchmark = deployment_commands.add_parser(
        "benchmark",
        help="Run synchronized DD/PP/PD/DP workloads while servers remain resident.",
    )
    benchmark.add_argument(
        "spec",
        type=Path,
        help="JSON concurrent deployment benchmark specification.",
    )
    _add_database_argument(benchmark)


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


def _add_ui_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    ui = commands.add_parser(
        "ui",
        help="Serve the built React UI and local API together.",
    )
    ui.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address; defaults to loopback only.",
    )
    ui.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP port (default 8000).",
    )
    ui.add_argument(
        "--launcher-config",
        type=Path,
        default=None,
        help="llama-profile-launcher host JSON exposed read-only to the UI.",
    )
    ui.add_argument(
        "--frontend-dir",
        type=Path,
        default=Path("frontend/dist"),
        help="Built frontend directory (default frontend/dist).",
    )
    _add_database_argument(ui)


def _add_database_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    database = commands.add_parser(
        "database",
        help="Inspect SQLite integrity, growth, and representative query plans.",
    )
    database_commands = database.add_subparsers(dest="database_command")
    check = database_commands.add_parser(
        "check",
        help="Run integrity, foreign-key, size, and index-plan diagnostics.",
    )
    _add_database_argument(check)


def _add_archive_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    archive = commands.add_parser(
        "archive",
        help="Create or restore a verified database snapshot archive.",
    )
    archive.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination .tar.gz archive path when creating an archive.",
    )
    archive.add_argument(
        "--restore",
        type=Path,
        default=None,
        help="Restore this archive into --database instead of creating one.",
    )
    archive.add_argument(
        "--artifact",
        action="append",
        type=Path,
        default=[],
        help="Optional artifact file to include when creating; may be repeated.",
    )
    archive.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Optional directory for restored artifact files.",
    )
    _add_database_argument(archive)


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


def _experiment_export_command(
    database_path: Path,
    experiment_id: str,
    *,
    output: Path | None,
) -> int:
    try:
        rendered = serialize_experiment_export(Database(database_path), experiment_id)
    except ArchiveError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if output is None:
        print(rendered, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"Wrote experiment export to {output}")
    return 0


def _profile_promote_command(
    database_path: Path,
    experiment_id: str,
    candidate_id: str,
    *,
    launcher_config: Path,
    source_profile_id: str | None,
    output: Path | None,
) -> int:
    try:
        proposal = PromotionService(
            Database(database_path),
            LauncherProfileProvider(launcher_config),
        ).propose(
            experiment_id,
            candidate_id,
            source_profile_id=source_profile_id,
        )
    except (PromotionError, LauncherProfileError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rendered = proposal.patch or "# No launcher changes are required.\n"
    if output is None:
        print(rendered, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"Wrote launcher patch to {output}")
    print(
        f"Promotion record: {proposal.id} "
        f"({len(proposal.changes)} launcher argument changes)"
    )
    return 0


def _database_check_command(database_path: Path) -> int:
    if not database_path.expanduser().is_file():
        print(f"error: database does not exist: {database_path}", file=sys.stderr)
        return 2
    try:
        report = inspect_database(Database(database_path))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Schema version: {report.schema_version}")
    print(f"Integrity: {'ok' if report.integrity_ok else 'failed'}")
    print(f"Foreign-key violations: {report.foreign_key_violations}")
    print(f"Database bytes: {report.database_bytes}")
    print(f"WAL bytes: {report.wal_bytes}")
    print(f"Freelist pages: {report.freelist_count}/{report.page_count}")
    print("Rows:")
    for table, count in sorted(report.row_counts.items()):
        print(f"  {table}: {count}")
    print("Representative query plans:")
    for plan in report.query_plans:
        state = "indexed" if plan.uses_index else "scan"
        print(f"  {plan.name}: {state}")
        for detail in plan.details:
            print(f"    {detail}")

    healthy = (
        report.integrity_ok
        and report.foreign_key_violations == 0
        and all(plan.uses_index for plan in report.query_plans)
    )
    return 0 if healthy else 1


def _archive_command(
    database_path: Path,
    *,
    output: Path | None,
    restore: Path | None,
    artifacts: Sequence[Path],
    artifacts_dir: Path | None,
) -> int:
    if restore is not None:
        if output is not None or artifacts:
            print(
                "error: --restore cannot be combined with --output or --artifact",
                file=sys.stderr,
            )
            return 2
        try:
            manifest = ArchiveService.restore(
                restore,
                database_path,
                artifacts_dir=artifacts_dir,
            )
        except ArchiveError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"Restored database: {database_path}")
        print(f"Schema version: {manifest.schema_version}")
        print(f"Verified files: {len(manifest.files)}")
        return 0

    if output is None:
        print("error: --output is required when creating an archive", file=sys.stderr)
        return 2
    if artifacts_dir is not None:
        print("error: --artifacts-dir is only valid with --restore", file=sys.stderr)
        return 2
    try:
        manifest = ArchiveService(Database(database_path)).create(
            output,
            artifacts=artifacts,
        )
    except ArchiveError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Archive: {output}")
    print(f"Schema version: {manifest.schema_version}")
    for item in manifest.files:
        print(f"{item.sha256}  {item.size_bytes}  {item.path}")
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


def _ui_command(
    database_path: Path,
    *,
    host: str,
    port: int,
    launcher_config: Path | None,
    frontend_dir: Path,
) -> int:
    if not 1 <= port <= 65535:
        print("error: UI port must be between 1 and 65535", file=sys.stderr)
        return 2
    resolved_frontend = frontend_dir.expanduser().resolve()
    if not (resolved_frontend / "index.html").is_file():
        print(
            "error: built frontend not found; run 'cd frontend && npm install && npm run build'",
            file=sys.stderr,
        )
        return 2

    import uvicorn

    from llama_profile_lab.api import create_app

    app = create_app(
        database_path,
        launcher_config_path=launcher_config,
        frontend_dist_path=resolved_frontend,
    )
    uvicorn.run(app, host=host, port=port)
    return 0


def _deployment_api_service(database_path: Path) -> ApiService:
    database = Database(database_path)
    return ApiService(
        database,
        profiles=LauncherProfileProvider(None),
        operations=OperationManager(database),
        deployment_operations=DeploymentOperationManager(database),
    )


def _deployment_create_command(
    database_path: Path,
    spec_path: Path,
) -> int:
    try:
        raw = json.loads(
            spec_path.expanduser().resolve().read_text(encoding="utf-8")
        )
        if not isinstance(raw, dict):
            raise ValueError("deployment create spec must be a JSON object")
        deployment_raw = raw.get("deployment", raw)
        if not isinstance(deployment_raw, dict):
            raise ValueError("deployment definition must be a JSON object")
        deployment = DeploymentCandidate.model_validate(deployment_raw)
        result = _deployment_api_service(
            database_path
        ).create_deployment(
            DeploymentCreateRequest(deployment=deployment)
        )
    except (
        ApiConflictError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Deployment: {result.id}")
    print(f"Status: {result.status}")
    print(f"Instances: {len(result.definition.instances)}")
    return 0


def _deployment_show_command(
    database_path: Path,
    deployment_id: str,
    *,
    format_name: str,
) -> int:
    try:
        result = _deployment_api_service(
            database_path
        ).get_deployment(deployment_id)
    except ApiNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if format_name == "json":
        print(result.model_dump_json(indent=2))
        return 0

    print(f"Deployment: {result.id}")
    print(f"Status: {result.status}")
    print(f"Created: {result.created_at}")
    print(f"Plans: {result.plan_count}")
    print(f"Placements: {result.placement_count}")
    print(f"Runs: {result.run_count}")
    print(f"Latest plan: {result.latest_plan_id or '-'}")
    for instance in result.definition.instances:
        print(
            f"{instance.instance_id}: role={instance.role} "
            f"candidate={instance.candidate_id} "
            f"binary={instance.binary_id}"
        )
    return 0


def _deployment_placement_command(
    database_path: Path,
    deployment_id: str,
    *,
    format_name: str,
) -> int:
    try:
        response = _deployment_api_service(
            database_path
        ).list_deployment_placements(deployment_id)
    except (ApiNotFoundError, DeploymentAnalysisError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if format_name == "json":
        print(response.model_dump_json(indent=2))
        return 0
    if not response.items:
        print("No deployment placements.")
        return 0

    for index, item in enumerate(response.items):
        if index:
            print()
        print(
            f"Placement: {item.id} "
            f"candidate={item.deployment_candidate_id} "
            f"feasibility={item.feasibility}"
        )
        devices = item.memory.devices
        print("Memory matrix:")
        print("  row/source  " + "  ".join(devices))
        for row in item.memory.rows:
            values = "  ".join(
                "-"
                if row.values.get(device) is None
                else str(row.values[device])
                for device in devices
            )
            print(f"  {row.key}/{row.source}  {values}")
    return 0


def _operation_spec_from_path(path: Path) -> DeploymentOperationSpec:
    (
        placement_id,
        inputs,
        baselines,
        host,
        readiness_timeout_seconds,
    ) = _load_concurrent_deployment_spec(path)
    return DeploymentOperationSpec(
        deployment_placement_id=placement_id,
        inputs=inputs,
        standalone_baselines=baselines,
        host=host,
        readiness_timeout_seconds=readiness_timeout_seconds,
    )


def _render_deployment_operation(
    snapshot: DeploymentOperationSnapshot,
) -> str:
    return (
        f"Operation: {snapshot.id}\n"
        f"Deployment: {snapshot.deployment_candidate_id}\n"
        f"Placement: {snapshot.deployment_placement_id}\n"
        f"Run: {snapshot.deployment_run_id or '-'}\n"
        f"Status: {snapshot.status}\n"
        f"Action: {snapshot.requested_action or '-'}\n"
        f"Error: {snapshot.error or '-'}"
    )


def _deployment_run_command(
    database_path: Path,
    deployment_id: str,
    spec_path: Path,
) -> int:
    try:
        snapshot = DeploymentOperationManager(
            Database(database_path)
        ).start(
            deployment_id,
            _operation_spec_from_path(spec_path),
            background=False,
        )
    except (
        DeploymentOperationError,
        DeploymentExecutionError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_render_deployment_operation(snapshot))
    return 0 if snapshot.status == "completed" else 2


def _deployment_resume_command(
    database_path: Path,
    deployment_id: str,
    spec_path: Path | None,
) -> int:
    try:
        spec = (
            None
            if spec_path is None
            else _operation_spec_from_path(spec_path)
        )
        snapshot = DeploymentOperationManager(
            Database(database_path)
        ).resume(
            deployment_id,
            spec,
            background=False,
        )
    except (
        DeploymentOperationError,
        DeploymentExecutionError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_render_deployment_operation(snapshot))
    return 0 if snapshot.status == "completed" else 2


def _deployment_control_command(
    database_path: Path,
    deployment_id: str,
    *,
    action: str,
) -> int:
    manager = DeploymentOperationManager(Database(database_path))
    try:
        if action == "pause":
            snapshot = manager.pause(deployment_id)
        elif action == "cancel":
            snapshot = manager.cancel(deployment_id)
        else:
            raise ValueError(f"unsupported deployment action: {action}")
    except (DeploymentOperationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_render_deployment_operation(snapshot))
    return 0


def _deployment_results_command(
    database_path: Path,
    deployment_id: str,
    *,
    filter_args: Sequence[str],
    format_name: str,
    output: Path | None,
) -> int:
    try:
        filters = parse_deployment_filters(tuple(filter_args))
        response = _deployment_api_service(
            database_path
        ).deployment_results(
            deployment_id,
            filters=filters,
        )
    except (
        ApiNotFoundError,
        DeploymentAnalysisError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if format_name == "json":
        rendered = response.model_dump_json(indent=2)
    elif format_name == "csv":
        rendered = _deployment_rows_csv(response.rows)
    else:
        lines = [
            f"Deployment: {response.deployment_id}",
            f"Rows: {len(response.rows)}",
        ]
        for row in response.rows:
            lines.append(
                "  "
                f"placement={row.get('deployment_placement_id', '-')} "
                f"run={row.get('deployment_run_id', '-')} "
                f"phase={row.get('phase', '-')} "
                f"status={row.get('workload_status') or row.get('deployment_status', '-')} "
                f"pp_tps={row.get('combined_pp_tps', '-')} "
                f"tg_tps={row.get('combined_tg_tps', '-')} "
                f"retention={row.get('min_retention', '-')}"
            )
        rendered = "\n".join(lines)
    _write_text_output(rendered, output)
    return 0


def _deployment_export_command(
    database_path: Path,
    deployment_id: str,
    *,
    output: Path | None,
) -> int:
    try:
        rendered = serialize_deployment_export(
            Database(database_path),
            deployment_id,
        )
    except (ArchiveError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _write_text_output(rendered.rstrip("\n"), output)
    return 0


def _deployment_pareto_command(
    database_path: Path,
    deployment_id: str,
    *,
    objective_args: Sequence[str],
    constraint_args: Sequence[str],
    filter_args: Sequence[str],
    format_name: str,
) -> int:
    try:
        response = _deployment_api_service(
            database_path
        ).deployment_pareto(
            deployment_id,
            objectives=parse_deployment_objectives(
                tuple(objective_args)
            ),
            constraints=parse_deployment_constraints(
                tuple(constraint_args)
            ),
            filters=parse_deployment_filters(tuple(filter_args)),
        )
    except (
        ApiNotFoundError,
        DeploymentAnalysisError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if format_name == "json":
        print(response.model_dump_json(indent=2))
        return 0

    result = response.result
    print(f"Deployment: {deployment_id}")
    print(f"Evaluated: {result.evaluated_count}")
    print(f"Frontier: {len(result.frontier)}")
    for point in result.frontier:
        values = " ".join(
            f"{key}={value:.6g}"
            for key, value in point.values.items()
        )
        print(
            f"  {point.deployment_placement_id} "
            f"candidate={point.deployment_candidate_id} {values}"
        )
    if result.excluded:
        print("Excluded:")
        for placement_id, reason in sorted(result.excluded.items()):
            print(f"  {placement_id}: {reason}")
    return 0


def _deployment_rows_csv(
    rows: Sequence[dict[str, object]],
) -> str:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _write_text_output(
    value: str,
    output: Path | None,
) -> None:
    if output is None:
        print(value)
        return
    resolved = output.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(value, encoding="utf-8")
    print(resolved)


def _deployment_benchmark_command(
    database_path: Path,
    spec_path: Path,
) -> int:
    try:
        (
            placement_id,
            inputs,
            baselines,
            host,
            readiness_timeout_seconds,
        ) = _load_concurrent_deployment_spec(spec_path)
        summary = ConcurrentDeploymentExecutor(
            Database(database_path)
        ).execute(
            placement_id,
            inputs,
            standalone_baselines=baselines,
            host=host,
            readiness_timeout_seconds=readiness_timeout_seconds,
        )
    except (
        ConcurrentDeploymentError,
        DeploymentExecutionError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(_render_concurrent_deployment_summary(summary))
    return 0


def _load_concurrent_deployment_spec(
    path: Path,
) -> tuple[
    str,
    tuple[DeploymentServerInput, ...],
    tuple[StandaloneBaselineInput, ...],
    str,
    float,
]:
    (
        placement_id,
        inputs,
        host,
        readiness_timeout_seconds,
        _,
    ) = _load_deployment_execution_spec(path)
    resolved = path.expanduser().resolve()
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("concurrent deployment spec must be a JSON object")
    baseline_rows = raw.get("standalone_baselines", [])
    if not isinstance(baseline_rows, list):
        raise ValueError("standalone_baselines must be a JSON array")

    baselines: list[StandaloneBaselineInput] = []
    for row in baseline_rows:
        if not isinstance(row, dict):
            raise ValueError("standalone baseline entry must be an object")
        required = (
            "instance_id",
            "mode",
            "prompt_tokens",
            "generate_tokens",
            "depth_tokens",
            "throughput_tps",
        )
        if any(name not in row for name in required):
            raise ValueError(
                "standalone baseline requires instance_id, mode, "
                "prompt_tokens, generate_tokens, depth_tokens, and "
                "throughput_tps"
            )
        instance_id = row["instance_id"]
        mode = row["mode"]
        prompt_tokens = row["prompt_tokens"]
        generate_tokens = row["generate_tokens"]
        depth_tokens = row["depth_tokens"]
        throughput_tps = row["throughput_tps"]
        latency_ms = row.get("latency_ms")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError("baseline instance_id must be non-empty")
        if mode not in {"prefill", "decode"}:
            raise ValueError("baseline mode must be prefill or decode")
        for name, value in (
            ("prompt_tokens", prompt_tokens),
            ("generate_tokens", generate_tokens),
            ("depth_tokens", depth_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"baseline {name} must be non-negative integer")
        if (
            isinstance(throughput_tps, bool)
            or not isinstance(throughput_tps, (int, float))
            or throughput_tps <= 0
        ):
            raise ValueError("baseline throughput_tps must be positive")
        if (
            latency_ms is not None
            and (
                isinstance(latency_ms, bool)
                or not isinstance(latency_ms, (int, float))
                or latency_ms < 0
            )
        ):
            raise ValueError("baseline latency_ms must be non-negative")
        baselines.append(
            StandaloneBaselineInput(
                instance_id=instance_id,
                mode=mode,
                prompt_tokens=prompt_tokens,
                generate_tokens=generate_tokens,
                depth_tokens=depth_tokens,
                throughput_tps=float(throughput_tps),
                latency_ms=(
                    None if latency_ms is None else float(latency_ms)
                ),
            )
        )
    return (
        placement_id,
        inputs,
        tuple(baselines),
        host,
        readiness_timeout_seconds,
    )


def _render_concurrent_deployment_summary(
    summary: ConcurrentDeploymentSummary,
) -> str:
    lines = [
        f"Deployment run: {summary.deployment_run_id}",
        f"Placement: {summary.deployment_placement_id}",
        f"Concurrent phases: {len(summary.phases)}",
    ]
    for phase in summary.phases:
        prompt = (
            "-"
            if phase.combined_prompt_tps is None
            else f"{phase.combined_prompt_tps:.3f}"
        )
        decode = (
            "-"
            if phase.combined_decode_tps is None
            else f"{phase.combined_decode_tps:.3f}"
        )
        retention = (
            "-"
            if phase.min_retention is None
            else f"{phase.min_retention:.3f}"
        )
        lines.append(
            f"{phase.phase.upper()}: quality={phase.quality} "
            f"pp_tps={prompt} tg_tps={decode} "
            f"min_retention={retention}"
        )
    return "\n".join(lines)


def _deployment_execute_command(
    database_path: Path,
    spec_path: Path,
) -> int:
    try:
        (
            placement_id,
            inputs,
            host,
            readiness_timeout_seconds,
            residency_hold_seconds,
        ) = _load_deployment_execution_spec(spec_path)
        summary = DeploymentExecutor(Database(database_path)).execute(
            placement_id,
            inputs,
            host=host,
            readiness_timeout_seconds=readiness_timeout_seconds,
            residency_hold_seconds=residency_hold_seconds,
        )
    except (
        DeploymentExecutionError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(_render_deployment_execution_summary(summary))
    return 0


def _load_deployment_execution_spec(
    path: Path,
) -> tuple[
    str,
    tuple[DeploymentServerInput, ...],
    str,
    float,
    float,
]:
    resolved = path.expanduser().resolve()
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("deployment execution spec must be a JSON object")

    placement_id = raw.get("deployment_placement_id")
    if not isinstance(placement_id, str) or not placement_id:
        raise ValueError(
            "deployment execution spec requires deployment_placement_id"
        )

    instance_rows = raw.get("instances")
    if not isinstance(instance_rows, list) or not instance_rows:
        raise ValueError(
            "deployment execution spec requires non-empty instances list"
        )
    inputs: list[DeploymentServerInput] = []
    for row in instance_rows:
        if not isinstance(row, dict):
            raise ValueError("deployment execution instance must be object")
        instance_id = row.get("instance_id")
        model_path = row.get("model_path")
        draft_model_path = row.get("draft_model_path")
        if (
            not isinstance(instance_id, str)
            or not instance_id
            or not isinstance(model_path, str)
            or not model_path
            or (
                draft_model_path is not None
                and (
                    not isinstance(draft_model_path, str)
                    or not draft_model_path
                )
            )
        ):
            raise ValueError(
                "each deployment execution instance requires instance_id "
                "and model_path; draft_model_path must be a non-empty "
                "string when present"
            )
        inputs.append(
            DeploymentServerInput(
                instance_id=instance_id,
                model_path=Path(model_path),
                draft_model_path=(
                    None
                    if draft_model_path is None
                    else Path(draft_model_path)
                ),
            )
        )

    host = raw.get("host", "127.0.0.1")
    if not isinstance(host, str) or not host:
        raise ValueError("deployment execution host must be a non-empty string")

    readiness = raw.get("readiness_timeout_seconds", 300.0)
    if (
        isinstance(readiness, bool)
        or not isinstance(readiness, (int, float))
        or readiness <= 0
    ):
        raise ValueError(
            "readiness_timeout_seconds must be a positive number"
        )

    hold = raw.get("residency_hold_seconds", 0.0)
    if (
        isinstance(hold, bool)
        or not isinstance(hold, (int, float))
        or hold < 0
    ):
        raise ValueError(
            "residency_hold_seconds must be a non-negative number"
        )
    return (
        placement_id,
        tuple(inputs),
        host,
        float(readiness),
        float(hold),
    )


def _render_deployment_execution_summary(
    summary: DeploymentExecutionSummary,
) -> str:
    lines = [
        f"Deployment run: {summary.run_id}",
        f"Placement: {summary.deployment_placement_id}",
        f"Status: {summary.status}",
    ]
    for member in summary.members:
        lines.append(
            f"{member.instance_id}: {member.endpoint} "
            f"pid={member.pid or '-'} ready={member.ready_at or '-'}"
        )
    return "\n".join(lines)


def _deployment_plan_command(
    database_path: Path,
    spec_path: Path,
    *,
    persist: bool,
) -> int:
    try:
        base_id, search_space, inputs, timeout_seconds = (
            _load_deployment_plan_spec(spec_path)
        )
        service = DeploymentPlannerService(Database(database_path))
        summary = (
            service.plan(
                base_id,
                search_space,
                inputs,
                timeout_seconds=timeout_seconds,
            )
            if persist
            else service.preview(
                base_id,
                search_space,
                inputs,
                timeout_seconds=timeout_seconds,
            )
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(_render_deployment_plan_summary(summary))
    return 0


def _load_deployment_plan_spec(
    path: Path,
) -> tuple[
    str,
    DeploymentSearchSpace,
    tuple[DeploymentEstimatorInput, ...],
    float | None,
]:
    resolved = path.expanduser().resolve()
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("deployment planning spec must be a JSON object")

    base_id = raw.get("base_deployment_candidate_id")
    if not isinstance(base_id, str) or not base_id:
        raise ValueError(
            "deployment planning spec requires base_deployment_candidate_id"
        )
    search_raw = raw.get("search_space")
    if not isinstance(search_raw, dict):
        raise ValueError(
            "deployment planning spec requires search_space object"
        )
    search_space = DeploymentSearchSpace.model_validate(search_raw)

    instance_rows = raw.get("instances")
    if not isinstance(instance_rows, list) or not instance_rows:
        raise ValueError(
            "deployment planning spec requires non-empty instances list"
        )
    inputs: list[DeploymentEstimatorInput] = []
    for row in instance_rows:
        if not isinstance(row, dict):
            raise ValueError("deployment instance estimator entry must be object")
        instance_id = row.get("instance_id")
        helper_binary_id = row.get("helper_binary_id")
        model_path = row.get("model_path")
        if (
            not isinstance(instance_id, str)
            or not instance_id
            or not isinstance(helper_binary_id, str)
            or not helper_binary_id
            or not isinstance(model_path, str)
            or not model_path
        ):
            raise ValueError(
                "each deployment estimator entry requires instance_id, "
                "helper_binary_id, and model_path"
            )
        inputs.append(
            DeploymentEstimatorInput(
                instance_id=instance_id,
                helper_binary_id=helper_binary_id,
                model_path=Path(model_path),
            )
        )

    timeout_raw = raw.get("timeout_seconds", 300.0)
    if timeout_raw is None:
        timeout_seconds = None
    elif isinstance(timeout_raw, bool) or not isinstance(
        timeout_raw,
        (int, float),
    ):
        raise ValueError("timeout_seconds must be numeric or null")
    elif timeout_raw <= 0:
        raise ValueError("timeout_seconds must be positive")
    else:
        timeout_seconds = float(timeout_raw)
    return base_id, search_space, tuple(inputs), timeout_seconds


def _render_deployment_plan_summary(
    summary: DeploymentPlanSummary,
) -> str:
    lines = [
        f"Base deployment: {summary.base_deployment_candidate_id}",
        f"Host: {summary.host_id}",
        f"Raw combinations: {summary.raw_combinations}",
        f"Rejected by constraints: {summary.rejected_by_constraints}",
        f"Duplicate candidates: {summary.duplicate_candidates}",
        f"Symmetry reduced: {summary.symmetry_reduced}",
        f"Capability rejected: {summary.capability_rejected}",
        f"Estimate failed: {summary.estimate_failed}",
        f"Memory rejected: {summary.memory_rejected}",
        f"Valid: {summary.valid_count}",
    ]
    if summary.plan_id is not None:
        lines.append(f"Plan: {summary.plan_id}")
    return "\n".join(lines)


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


def _binary_devices_command(
    database_path: Path,
    binary_id: str,
    *,
    timeout_seconds: float,
) -> int:
    try:
        result = DeviceInventoryService(Database(database_path)).inspect(
            binary_id,
            timeout_seconds=timeout_seconds,
        )
    except DeviceInventoryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Host: {result.host_id}")
    print(f"Binary: {result.binary_id}")
    for device in result.devices:
        total = (
            "-"
            if device.total_memory_bytes is None
            else str(device.total_memory_bytes)
        )
        free = (
            "-"
            if device.free_memory_bytes is None
            else str(device.free_memory_bytes)
        )
        physical = device.physical_device_key or "unresolved"
        print(
            f"{device.logical_device_name}\t{device.backend}\t"
            f"{device.product_name or '-'}\t"
            f"total={total}\tfree={free}\tphysical={physical}"
        )
    return 0


def _placement_estimate_command(
    database_path: Path,
    candidate_id: str,
    *,
    helper_binary_id: str,
    model_path: Path,
    devices: Sequence[str] | None,
    timeout_seconds: float,
) -> int:
    try:
        result = MemoryEstimatorService(Database(database_path)).estimate(
            candidate_id,
            helper_binary_id=helper_binary_id,
            model_path=model_path,
            selected_devices=(
                None if devices is None else tuple(devices)
            ),
            timeout_seconds=timeout_seconds,
        )
    except MemoryEstimatorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Estimate: {result.estimate_id}")
    print(f"Attempt: {result.attempt_id}")
    print(f"Cache: {'hit' if result.cache_hit else 'miss'}")
    print(f"GPU layers: {result.output.resolved.n_gpu_layers}")
    print(f"Devices: {','.join(result.output.resolved.devices)}")
    for device in result.output.devices:
        print(
            f"{device.logical_device_name}\t"
            f"model={device.model_bytes}\t"
            f"context={device.context_bytes}\t"
            f"compute={device.compute_bytes}\t"
            f"total={device.total_bytes}\t"
            f"free={device.device_free_bytes}/"
            f"{device.device_total_bytes}"
        )
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
    if value not in {
        "llama-bench",
        "llama-fit-params",
        "llama-memory-estimator",
        "llama-server",
        "speed-bench",
    }:
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
        if args.experiment_command == "export":
            return _experiment_export_command(
                args.database,
                args.experiment_id,
                output=args.output,
            )
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

    if args.command == "profile" and args.profile_command == "promote":
        return _profile_promote_command(
            args.database,
            args.experiment_id,
            args.candidate_id,
            launcher_config=args.launcher_config,
            source_profile_id=args.source_profile_id,
            output=args.output,
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
        if args.binary_command == "devices":
            return _binary_devices_command(
                args.database,
                args.binary_id,
                timeout_seconds=args.timeout_seconds,
            )

    if args.command == "run" and args.run_command == "show":
        return _run_show_command(args.database, args.run_id, include_logs=args.logs)

    if args.command == "placement":
        if args.placement_command == "list":
            return _placement_list_command(args.database)
        if args.placement_command == "show":
            return _placement_show_command(args.database, args.placement_id)
        if args.placement_command == "estimate":
            return _placement_estimate_command(
                args.database,
                args.candidate_id,
                helper_binary_id=args.helper_binary_id,
                model_path=args.model_path,
                devices=args.devices,
                timeout_seconds=args.timeout_seconds,
            )

    if args.command == "deployment":
        if args.deployment_command == "create":
            return _deployment_create_command(
                args.database,
                args.spec,
            )
        if args.deployment_command == "show":
            return _deployment_show_command(
                args.database,
                args.deployment_id,
                format_name=args.format_name,
            )
        if args.deployment_command == "placement":
            return _deployment_placement_command(
                args.database,
                args.deployment_id,
                format_name=args.format_name,
            )
        if args.deployment_command == "run":
            return _deployment_run_command(
                args.database,
                args.deployment_id,
                args.spec,
            )
        if args.deployment_command == "pause":
            return _deployment_control_command(
                args.database,
                args.deployment_id,
                action="pause",
            )
        if args.deployment_command == "resume":
            return _deployment_resume_command(
                args.database,
                args.deployment_id,
                args.spec,
            )
        if args.deployment_command == "cancel":
            return _deployment_control_command(
                args.database,
                args.deployment_id,
                action="cancel",
            )
        if args.deployment_command == "results":
            return _deployment_results_command(
                args.database,
                args.deployment_id,
                filter_args=args.filter,
                format_name=args.format_name,
                output=args.output,
            )
        if args.deployment_command == "export":
            return _deployment_export_command(
                args.database,
                args.deployment_id,
                output=args.output,
            )
        if args.deployment_command == "pareto":
            return _deployment_pareto_command(
                args.database,
                args.deployment_id,
                objective_args=args.objective,
                constraint_args=args.constraint,
                filter_args=args.filter,
                format_name=args.format_name,
            )
        if args.deployment_command == "preview":
            return _deployment_plan_command(
                args.database,
                args.spec,
                persist=False,
            )
        if args.deployment_command == "plan":
            return _deployment_plan_command(
                args.database,
                args.spec,
                persist=True,
            )
        if args.deployment_command == "execute":
            return _deployment_execute_command(
                args.database,
                args.spec,
            )
        if args.deployment_command == "benchmark":
            return _deployment_benchmark_command(
                args.database,
                args.spec,
            )

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

    if args.command == "ui":
        return _ui_command(
            args.database,
            host=args.host,
            port=args.port,
            launcher_config=args.launcher_config,
            frontend_dir=args.frontend_dir,
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

    if args.command == "database" and args.database_command == "check":
        return _database_check_command(args.database)

    if args.command == "archive":
        return _archive_command(
            args.database,
            output=args.output,
            restore=args.restore,
            artifacts=args.artifact,
            artifacts_dir=args.artifacts_dir,
        )

    return 0
