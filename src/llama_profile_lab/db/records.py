"""Typed persistence records that are not immutable domain definitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

ExperimentStatus = Literal[
    "draft",
    "planned",
    "running",
    "paused",
    "completed",
    "cancelled",
    "failed",
]

RunStatus = Literal[
    "planned",
    "running",
    "completed",
    "oom",
    "timeout",
    "invalid",
    "fit_failed",
    "load_failed",
    "benchmark_failed",
    "parser_failed",
    "interrupted",
    "cancelled",
]


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """Persisted experiment event plus its immutable definition."""

    id: str
    status: ExperimentStatus
    name: str
    base_candidate_id: str
    search_space_id: str
    workload_suite_id: str
    measurement_policy_id: str
    created_at: str
    frozen_at: str | None
    completed_at: str | None


@dataclass(frozen=True, slots=True)
class BenchmarkCaseRecord:
    """One planned Candidate × WorkloadCase observation."""

    id: str
    experiment_id: str
    candidate_id: str
    workload_case_id: str
    placement_id: str | None
    case_hash: str
    status: RunStatus
    ordinal: int


@dataclass(frozen=True, slots=True)
class BenchmarkRunRecord:
    """One append-only attempt to execute a benchmark case."""

    id: str
    benchmark_case_id: str
    host_id: str
    binary_id: str
    measurement_policy_id: str
    started_at: str
    finished_at: str | None
    duration_ns: int | None
    status: RunStatus
    exit_code: int | None


@dataclass(frozen=True, slots=True)
class BinaryRecord:
    """Persisted identity and discovered capabilities for one executable."""

    id: str
    sha256: str
    kind: str
    path: str
    size_bytes: int
    mtime_ns: int
    git_commit: str | None
    git_branch: str | None
    git_dirty: bool | None
    build_number: str | None
    build_info: Mapping[str, Any]
    capabilities: Mapping[str, Any]
    created_at: str
