"""Discovery, probing, and exact executable fingerprinting for llama.cpp tools."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from llama_profile_lab.llama.capabilities import (
    BinaryKind,
    CapabilitySet,
    parse_help_options,
)

_BINARY_NAME_TO_KIND: dict[str, BinaryKind] = {
    "llama-bench": "llama-bench",
    "llama-fit-params": "llama-fit-params",
    "llama-server": "llama-server",
    "speed-bench": "speed-bench",
    "speed_bench.py": "speed-bench",
}
_VERSION_NUMBER_RE = re.compile(r"\b(?:version|build)\s*[:=]?\s*(\d+)\b", re.IGNORECASE)
_COMMIT_LABEL_RE = re.compile(
    r"\b(?:commit|revision|rev)\s*[:=]\s*([0-9a-f]{7,40})\b",
    re.IGNORECASE,
)
_PAREN_COMMIT_RE = re.compile(r"\(([0-9a-f]{7,40})\)", re.IGNORECASE)


class BinaryDiscoveryError(RuntimeError):
    """Raised when a llama.cpp executable cannot be discovered or probed."""


@dataclass(frozen=True, slots=True)
class CommandCapture:
    """Captured output for one safe, shell-free metadata command."""

    argv: tuple[str, ...]
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def combined_output(self) -> str:
        """Return stdout and stderr without losing either stream."""
        parts = [part.strip() for part in (self.stdout, self.stderr) if part.strip()]
        return "\n".join(parts)


@dataclass(frozen=True, slots=True)
class BinaryProbe:
    """Fingerprint, build metadata, and capabilities for one executable."""

    kind: BinaryKind
    path: Path
    sha256: str
    size_bytes: int
    mtime_ns: int
    version: CommandCapture
    help: CommandCapture
    capabilities: CapabilitySet
    build_number: str | None
    git_commit: str | None

    def build_info_mapping(self) -> dict[str, object]:
        """Return complete version/build probe data for persistence."""
        return {
            "schema": "llama-binary-build-info",
            "version": 1,
            "build_number": self.build_number,
            "git_commit": self.git_commit,
            "version_argv": list(self.version.argv),
            "version_exit_code": self.version.exit_code,
            "version_stdout": self.version.stdout,
            "version_stderr": self.version.stderr,
            "version_timed_out": self.version.timed_out,
        }


def infer_binary_kind(path: Path, explicit: BinaryKind | None = None) -> BinaryKind:
    """Infer tool kind from filename unless explicitly supplied."""
    if explicit is not None:
        return explicit

    name = path.name
    inferred = _BINARY_NAME_TO_KIND.get(name)
    if inferred is not None:
        return inferred
    raise BinaryDiscoveryError(
        f"cannot infer llama.cpp binary kind from filename {name!r}; specify --kind"
    )


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash an executable without loading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def discover_binary_paths(
    search_dirs: tuple[Path, ...] = (),
    *,
    include_path: bool = True,
) -> tuple[Path, ...]:
    """Discover known llama.cpp tool names deterministically."""
    discovered: dict[str, Path] = {}

    for directory in search_dirs:
        expanded = directory.expanduser()
        if expanded.is_file():
            candidate = expanded.resolve()
            if candidate.name in _BINARY_NAME_TO_KIND and os.access(candidate, os.X_OK):
                discovered[str(candidate)] = candidate
            continue

        for name in _BINARY_NAME_TO_KIND:
            candidate = expanded / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                resolved = candidate.resolve()
                discovered[str(resolved)] = resolved

    if include_path:
        for name in _BINARY_NAME_TO_KIND:
            found = shutil.which(name)
            if found is not None:
                resolved = Path(found).resolve()
                discovered[str(resolved)] = resolved

    return tuple(discovered[key] for key in sorted(discovered))


def probe_binary(
    path: Path,
    *,
    kind: BinaryKind | None = None,
    timeout_seconds: float = 10.0,
) -> BinaryProbe:
    """Probe one exact executable for identity, version/build info, and help."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise BinaryDiscoveryError(f"binary does not exist: {resolved}")
    if not os.access(resolved, os.X_OK):
        raise BinaryDiscoveryError(f"binary is not executable: {resolved}")

    resolved_kind = infer_binary_kind(resolved, kind)
    stat = resolved.stat()
    version = _capture((str(resolved), "--version"), timeout_seconds)
    help_capture = _capture((str(resolved), "--help"), timeout_seconds)
    if (
        help_capture.exit_code not in {0, None}
        and not help_capture.combined_output
        and not help_capture.timed_out
    ):
        help_capture = _capture((str(resolved), "-h"), timeout_seconds)

    capabilities = CapabilitySet(
        kind=resolved_kind,
        options=parse_help_options(help_capture.stdout, help_capture.stderr),
        help_stdout=help_capture.stdout,
        help_stderr=help_capture.stderr,
        help_exit_code=help_capture.exit_code,
        help_timed_out=help_capture.timed_out,
    )
    build_number, git_commit = parse_build_metadata(version.combined_output)

    return BinaryProbe(
        kind=resolved_kind,
        path=resolved,
        sha256=sha256_file(resolved),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        version=version,
        help=help_capture,
        capabilities=capabilities,
        build_number=build_number,
        git_commit=git_commit,
    )


def parse_build_metadata(text: str) -> tuple[str | None, str | None]:
    """Extract build/version number and commit hash when advertised."""
    number_match = _VERSION_NUMBER_RE.search(text)
    commit_match = _COMMIT_LABEL_RE.search(text) or _PAREN_COMMIT_RE.search(text)
    return (
        number_match.group(1) if number_match is not None else None,
        commit_match.group(1).lower() if commit_match is not None else None,
    )


def _capture(argv: tuple[str, ...], timeout_seconds: float) -> CommandCapture:
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandCapture(
            argv=argv,
            exit_code=None,
            stdout=_decode_timeout_stream(exc.stdout),
            stderr=_decode_timeout_stream(exc.stderr),
            timed_out=True,
        )
    except OSError as exc:
        raise BinaryDiscoveryError(f"failed to invoke {argv[0]}: {exc}") from exc

    return CommandCapture(
        argv=argv,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _decode_timeout_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value
