"""Top-level llprof command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from llama_profile_lab import __version__
from llama_profile_lab.db import (
    BenchmarkRunRepository,
    Database,
    EnvironmentRepository,
    PlacementRepository,
    TelemetryRepository,
)
from llama_profile_lab.db.records import BinaryRecord
from llama_profile_lab.execution import (
    ExecutionError,
    ExecutionSummary,
    ExperimentExecutor,
    HostLockError,
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

_BINARY_KIND_CHOICES = ("auto", "llama-bench", "llama-fit-params", "llama-server")


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
    if value not in {"llama-bench", "llama-fit-params", "llama-server"}:
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

    return 0
