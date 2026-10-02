"""Top-level llprof command-line interface."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from llama_profile_lab import __version__


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the llprof command-line interface."""
    parser = build_parser()
    parser.parse_args(argv)
    return 0
