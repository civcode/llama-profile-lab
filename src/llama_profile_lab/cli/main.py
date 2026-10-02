"""Top-level llprof command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from llama_profile_lab import __version__
from llama_profile_lab.db import Database, EnvironmentRepository
from llama_profile_lab.db.records import BinaryRecord
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
    return parser


def _add_experiment_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    experiment = commands.add_parser(
        "experiment",
        help="Create, plan, and manage experiments.",
    )
    experiment_commands = experiment.add_subparsers(dest="experiment_command")
    plan = experiment_commands.add_parser(
        "plan",
        help="Expand and persist a draft experiment without executing benchmarks.",
    )
    plan.add_argument("experiment_id", help="Persisted experiment ID to plan.")
    _add_database_argument(plan)


def _add_binary_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
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


def _binary_inspect_command(
    database_path: Path,
    paths: Sequence[Path],
    kind_arg: str,
) -> int:
    explicit_kind = _parse_kind_argument(kind_arg)
    database = Database(database_path)
    try:
        probes = tuple(
            probe_binary(path, kind=explicit_kind)
            for path in paths
        )
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

    if args.command == "experiment" and args.experiment_command == "plan":
        return _plan_command(args.database, args.experiment_id)

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

    return 0
