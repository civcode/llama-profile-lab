"""Top-level llprof command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from llama_profile_lab import __version__
from llama_profile_lab.db import Database
from llama_profile_lab.planning import PlanningError, plan_experiment, render_plan_summary


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
    plan.add_argument(
        "--database",
        type=Path,
        default=Path("data/benchmarks.db"),
        help="SQLite database path (default: data/benchmarks.db).",
    )
    return parser


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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the llprof command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "experiment" and args.experiment_command == "plan":
        return _plan_command(args.database, args.experiment_id)

    return 0
