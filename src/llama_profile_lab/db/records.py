"""Typed persistence records that are not immutable domain definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
