"""llama.cpp help-surface parsing and capability comparison."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

BinaryKind = Literal[
    "llama-bench",
    "llama-fit-params",
    "llama-memory-estimator",
    "llama-server",
    "speed-bench",
]

_OPTION_RE = re.compile(r"^(?:--[A-Za-z0-9][A-Za-z0-9_-]*|-[A-Za-z0-9][A-Za-z0-9_-]*)$")


@dataclass(frozen=True, slots=True)
class CapabilitySet:
    """Parsed option surface for one exact llama.cpp executable."""

    kind: BinaryKind
    options: frozenset[str]
    help_stdout: str
    help_stderr: str
    help_exit_code: int | None
    help_timed_out: bool = False

    @property
    def long_options(self) -> tuple[str, ...]:
        """Return deterministic long-form option names."""
        return tuple(sorted(option for option in self.options if option.startswith("--")))

    @property
    def short_options(self) -> tuple[str, ...]:
        """Return deterministic single-dash option names."""
        return tuple(
            sorted(
                option
                for option in self.options
                if option.startswith("-") and not option.startswith("--")
            )
        )

    def supports(self, option: str) -> bool:
        """Return whether the exact executable advertises an option."""
        return option in self.options

    def to_mapping(self) -> dict[str, object]:
        """Return the JSON document persisted with a binary record."""
        return {
            "schema": "llama-binary-capabilities",
            "version": 1,
            "kind": self.kind,
            "options": sorted(self.options),
            "long_options": list(self.long_options),
            "short_options": list(self.short_options),
            "help_stdout": self.help_stdout,
            "help_stderr": self.help_stderr,
            "help_exit_code": self.help_exit_code,
            "help_timed_out": self.help_timed_out,
        }

    @classmethod
    def from_mapping(
        cls,
        mapping: dict[str, Any],
        *,
        fallback_kind: BinaryKind,
    ) -> CapabilitySet:
        """Rehydrate capabilities from SQLite JSON."""
        raw_kind = mapping.get("kind", fallback_kind)
        kind: BinaryKind
        if raw_kind in {
            "llama-bench",
            "llama-fit-params",
            "llama-memory-estimator",
            "llama-server",
            "speed-bench",
        }:
            kind = raw_kind
        else:
            kind = fallback_kind

        raw_options = mapping.get("options", ())
        options = (
            frozenset(str(option) for option in raw_options)
            if isinstance(raw_options, list | tuple)
            else frozenset()
        )
        help_exit_code = mapping.get("help_exit_code")
        return cls(
            kind=kind,
            options=options,
            help_stdout=str(mapping.get("help_stdout", "")),
            help_stderr=str(mapping.get("help_stderr", "")),
            help_exit_code=help_exit_code if isinstance(help_exit_code, int) else None,
            help_timed_out=bool(mapping.get("help_timed_out", False)),
        )


@dataclass(frozen=True, slots=True)
class CapabilityDiff:
    """Difference between two exact executable option surfaces."""

    common: tuple[str, ...]
    only_left: tuple[str, ...]
    only_right: tuple[str, ...]

    @property
    def identical(self) -> bool:
        """Return whether both option sets are identical."""
        return not self.only_left and not self.only_right


def parse_help_options(*texts: str) -> frozenset[str]:
    """Extract advertised option names from llama.cpp help output.

    Only lines whose first non-whitespace token is an option are considered.
    Consecutive option tokens at the beginning of a line are treated as aliases.
    This avoids interpreting option names mentioned later in prose as capabilities.
    """
    options: set[str] = set()
    for text in texts:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("-"):
                continue

            for token in stripped.split():
                cleaned = token.strip(",;")
                if _OPTION_RE.fullmatch(cleaned):
                    options.add(cleaned)
                    continue
                if cleaned in {"", "|"}:
                    continue
                break
    return frozenset(options)


def compare_capabilities(
    left: CapabilitySet,
    right: CapabilitySet,
) -> CapabilityDiff:
    """Compare two executable option surfaces."""
    return CapabilityDiff(
        common=tuple(sorted(left.options & right.options)),
        only_left=tuple(sorted(left.options - right.options)),
        only_right=tuple(sorted(right.options - left.options)),
    )
