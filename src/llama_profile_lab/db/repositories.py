"""Thin SQLite repositories for domain objects and experiment records."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import TypeAdapter

from llama_profile_lab.domain import (
    Candidate,
    ExperimentDefinition,
    MeasurementPolicy,
    SearchSpace,
    WorkloadSuite,
    canonical_json,
    sha256_json,
)
from llama_profile_lab.domain.workload import WorkloadCase
from llama_profile_lab.db.records import (
    BenchmarkCaseRecord,
    BenchmarkRunRecord,
    ExperimentRecord,
    ExperimentStatus,
    RunStatus,
)

_WORKLOAD_ADAPTER = TypeAdapter(WorkloadCase)


def _event_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _content_id(prefix: str, digest: str) -> str:
    return f"{prefix}_{digest}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _loads_object(value: str) -> dict[str, Any]:
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError("persisted JSON document must be an object")
    return loaded


class CandidateRepository:
    """Persistence for immutable Candidates."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def put(self, candidate: Candidate) -> str:
        digest = candidate.content_hash()
        identifier = _content_id("cand", digest)
        fit_target = (
            candidate.placement.fit.target_mib
            if candidate.placement.fit is not None
            else None
        )
        self.connection.execute(
            """
            INSERT INTO candidate(
                id, config_hash, target_model_id, draft_model_id,
                context_size, batch_size, ubatch_size,
                cache_type_k, cache_type_v, flash_attn,
                fit_target_mib, config_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(config_hash) DO NOTHING
            """,
            (
                identifier,
                digest,
                candidate.model.target_model_id,
                candidate.model.draft_model_id,
                candidate.context.size,
                candidate.compute.batch_size,
                candidate.compute.ubatch_size,
                candidate.context.cache_type_k,
                candidate.context.cache_type_v,
                candidate.compute.flash_attn,
                fit_target,
                canonical_json(candidate),
            ),
        )
        row = self.connection.execute(
            "SELECT id FROM candidate WHERE config_hash = ?",
            (digest,),
        ).fetchone()
        if row is None:
            raise RuntimeError("candidate insert did not produce a row")
        return str(row["id"])

    def get(self, identifier: str) -> Candidate | None:
        row = self.connection.execute(
            "SELECT config_json FROM candidate WHERE id = ?",
            (identifier,),
        ).fetchone()
        if row is None:
            return None
        return Candidate.model_validate_json(str(row["config_json"]))


class SearchSpaceRepository:
    """Persistence for immutable SearchSpaces."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def put(self, search_space: SearchSpace) -> str:
        digest = search_space.content_hash()
        identifier = _content_id("space", digest)
        self.connection.execute(
            """
            INSERT INTO search_space(id, definition_hash, definition_json)
            VALUES (?, ?, ?)
            ON CONFLICT(definition_hash) DO NOTHING
            """,
            (identifier, digest, canonical_json(search_space)),
        )
        row = self.connection.execute(
            "SELECT id FROM search_space WHERE definition_hash = ?",
            (digest,),
        ).fetchone()
        if row is None:
            raise RuntimeError("search-space insert did not produce a row")
        return str(row["id"])

    def get(self, identifier: str) -> SearchSpace | None:
        row = self.connection.execute(
            "SELECT definition_json FROM search_space WHERE id = ?",
            (identifier,),
        ).fetchone()
        if row is None:
            return None
        return SearchSpace.model_validate_json(str(row["definition_json"]))


class WorkloadSuiteRepository:
    """Persistence for immutable workload suites."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def put(self, suite: WorkloadSuite) -> str:
        digest = suite.content_hash()
        identifier = _content_id("suite", digest)
        self.connection.execute(
            """
            INSERT INTO workload_suite(id, definition_hash, name, definition_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(definition_hash) DO NOTHING
            """,
            (identifier, digest, suite.id, canonical_json(suite)),
        )
        row = self.connection.execute(
            "SELECT id FROM workload_suite WHERE definition_hash = ?",
            (digest,),
        ).fetchone()
        if row is None:
            raise RuntimeError("workload-suite insert did not produce a row")
        return str(row["id"])

    def get(self, identifier: str) -> WorkloadSuite | None:
        row = self.connection.execute(
            "SELECT definition_json FROM workload_suite WHERE id = ?",
            (identifier,),
        ).fetchone()
        if row is None:
            return None
        return WorkloadSuite.model_validate_json(str(row["definition_json"]))


class WorkloadCaseRepository:
    """Persistence for immutable concrete workload cases."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def put(self, workload: WorkloadCase) -> str:
        digest = workload.content_hash()
        identifier = _content_id("work", digest)
        prompt_tokens = getattr(workload, "prompt_tokens", None)
        generate_tokens = getattr(workload, "generate_tokens", None)
        depth_tokens = getattr(workload, "depth_tokens", None)
        self.connection.execute(
            """
            INSERT INTO workload_case(
                id, workload_hash, kind, prompt_tokens,
                generate_tokens, depth_tokens, definition_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(workload_hash) DO NOTHING
            """,
            (
                identifier,
                digest,
                workload.kind,
                prompt_tokens,
                generate_tokens,
                depth_tokens,
                canonical_json(workload),
            ),
        )
        row = self.connection.execute(
            "SELECT id FROM workload_case WHERE workload_hash = ?",
            (digest,),
        ).fetchone()
        if row is None:
            raise RuntimeError("workload insert did not produce a row")
        return str(row["id"])

    def get(self, identifier: str) -> WorkloadCase | None:
        row = self.connection.execute(
            "SELECT definition_json FROM workload_case WHERE id = ?",
            (identifier,),
        ).fetchone()
        if row is None:
            return None
        return _WORKLOAD_ADAPTER.validate_json(str(row["definition_json"]))


class MeasurementPolicyRepository:
    """Persistence for immutable measurement policies."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def put(self, policy: MeasurementPolicy) -> str:
        digest = policy.content_hash()
        identifier = _content_id("measure", digest)
        self.connection.execute(
            """
            INSERT INTO measurement_policy(id, policy_hash, definition_json)
            VALUES (?, ?, ?)
            ON CONFLICT(policy_hash) DO NOTHING
            """,
            (identifier, digest, canonical_json(policy)),
        )
        row = self.connection.execute(
            "SELECT id FROM measurement_policy WHERE policy_hash = ?",
            (digest,),
        ).fetchone()
        if row is None:
            raise RuntimeError("measurement-policy insert did not produce a row")
        return str(row["id"])

    def get(self, identifier: str) -> MeasurementPolicy | None:
        row = self.connection.execute(
            "SELECT definition_json FROM measurement_policy WHERE id = ?",
            (identifier,),
        ).fetchone()
        if row is None:
            return None
        return MeasurementPolicy.model_validate_json(str(row["definition_json"]))


class ExperimentRepository:
    """Persistence for experiment events and their immutable definitions."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def create(
        self,
        definition: ExperimentDefinition,
        *,
        status: ExperimentStatus = "draft",
    ) -> str:
        identifier = _event_id("exp")
        self.connection.execute(
            """
            INSERT INTO experiment(
                id, name, status, base_candidate_id, search_space_id,
                workload_suite_id, measurement_policy_id,
                placement_policy_json, baseline_json, definition_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                identifier,
                definition.name,
                status,
                definition.base_candidate_id,
                definition.search_space_id,
                definition.workload_suite_id,
                definition.measurement_policy_id,
                canonical_json(definition.placement_policy),
                canonical_json(definition.baseline),
                canonical_json(definition),
            ),
        )
        return identifier

    def get_definition(self, identifier: str) -> ExperimentDefinition | None:
        row = self.connection.execute(
            "SELECT definition_json FROM experiment WHERE id = ?",
            (identifier,),
        ).fetchone()
        if row is None:
            return None
        return ExperimentDefinition.model_validate_json(str(row["definition_json"]))

    def get(self, identifier: str) -> ExperimentRecord | None:
        row = self.connection.execute(
            """
            SELECT id, status, name, base_candidate_id, search_space_id,
                   workload_suite_id, measurement_policy_id,
                   created_at, frozen_at, completed_at
            FROM experiment
            WHERE id = ?
            """,
            (identifier,),
        ).fetchone()
        if row is None:
            return None
        return ExperimentRecord(
            id=str(row["id"]),
            status=row["status"],
            name=str(row["name"]),
            base_candidate_id=str(row["base_candidate_id"]),
            search_space_id=str(row["search_space_id"]),
            workload_suite_id=str(row["workload_suite_id"]),
            measurement_policy_id=str(row["measurement_policy_id"]),
            created_at=str(row["created_at"]),
            frozen_at=row["frozen_at"],
            completed_at=row["completed_at"],
        )


class BenchmarkCaseRepository:
    """Persistence for planned benchmark cases."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def put_planned(
        self,
        *,
        experiment_id: str,
        candidate_id: str,
        workload_case_id: str,
        ordinal: int,
        placement_id: str | None = None,
    ) -> str:
        case_hash = sha256_json(
            {
                "experiment_id": experiment_id,
                "candidate_id": candidate_id,
                "workload_case_id": workload_case_id,
                "placement_id": placement_id,
            }
        )
        identifier = _content_id("case", case_hash)
        self.connection.execute(
            """
            INSERT INTO benchmark_case(
                id, experiment_id, candidate_id, workload_case_id,
                placement_id, case_hash, status, ordinal
            )
            VALUES (?, ?, ?, ?, ?, ?, 'planned', ?)
            ON CONFLICT(case_hash) DO NOTHING
            """,
            (
                identifier,
                experiment_id,
                candidate_id,
                workload_case_id,
                placement_id,
                case_hash,
                ordinal,
            ),
        )
        row = self.connection.execute(
            "SELECT id FROM benchmark_case WHERE case_hash = ?",
            (case_hash,),
        ).fetchone()
        if row is None:
            raise RuntimeError("benchmark-case insert did not produce a row")
        return str(row["id"])

    def get(self, identifier: str) -> BenchmarkCaseRecord | None:
        row = self.connection.execute(
            """
            SELECT id, experiment_id, candidate_id, workload_case_id,
                   placement_id, case_hash, status, ordinal
            FROM benchmark_case
            WHERE id = ?
            """,
            (identifier,),
        ).fetchone()
        if row is None:
            return None
        return BenchmarkCaseRecord(
            id=str(row["id"]),
            experiment_id=str(row["experiment_id"]),
            candidate_id=str(row["candidate_id"]),
            workload_case_id=str(row["workload_case_id"]),
            placement_id=row["placement_id"],
            case_hash=str(row["case_hash"]),
            status=row["status"],
            ordinal=int(row["ordinal"]),
        )


class BenchmarkRunRepository:
    """Persistence for append-only benchmark execution attempts."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def create(
        self,
        *,
        benchmark_case_id: str,
        host_id: str,
        binary_id: str,
        measurement_policy_id: str,
        argv: tuple[str, ...],
        environment: Mapping[str, str],
        started_at: str | None = None,
    ) -> str:
        identifier = _event_id("run")
        self.connection.execute(
            """
            INSERT INTO benchmark_run(
                id, benchmark_case_id, host_id, binary_id,
                measurement_policy_id, started_at, status,
                argv_json, environment_json
            )
            VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?)
            """,
            (
                identifier,
                benchmark_case_id,
                host_id,
                binary_id,
                measurement_policy_id,
                started_at or _utc_now(),
                canonical_json(list(argv)),
                canonical_json(dict(environment)),
            ),
        )
        return identifier

    def finish(
        self,
        identifier: str,
        *,
        status: RunStatus,
        duration_ns: int,
        exit_code: int | None,
        stdout: str = "",
        stderr: str = "",
        raw_result: Mapping[str, Any] | None = None,
        finished_at: str | None = None,
    ) -> None:
        if status in {"planned", "running"}:
            raise ValueError("finished run must have a terminal status")
        cursor = self.connection.execute(
            """
            UPDATE benchmark_run
            SET finished_at = ?, duration_ns = ?, status = ?, exit_code = ?,
                stdout = ?, stderr = ?, raw_result_json = ?
            WHERE id = ? AND status = 'running'
            """,
            (
                finished_at or _utc_now(),
                duration_ns,
                status,
                exit_code,
                stdout,
                stderr,
                canonical_json(dict(raw_result)) if raw_result is not None else None,
                identifier,
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError("run does not exist or has already been finalized")

    def get(self, identifier: str) -> BenchmarkRunRecord | None:
        row = self.connection.execute(
            """
            SELECT id, benchmark_case_id, host_id, binary_id,
                   measurement_policy_id, started_at, finished_at,
                   duration_ns, status, exit_code
            FROM benchmark_run
            WHERE id = ?
            """,
            (identifier,),
        ).fetchone()
        if row is None:
            return None
        return BenchmarkRunRecord(
            id=str(row["id"]),
            benchmark_case_id=str(row["benchmark_case_id"]),
            host_id=str(row["host_id"]),
            binary_id=str(row["binary_id"]),
            measurement_policy_id=str(row["measurement_policy_id"]),
            started_at=str(row["started_at"]),
            finished_at=row["finished_at"],
            duration_ns=row["duration_ns"],
            status=row["status"],
            exit_code=row["exit_code"],
        )


class EnvironmentRepository:
    """Minimal host/binary persistence needed by benchmark-run foreign keys."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def put_host(
        self,
        *,
        hostname: str,
        hardware_fingerprint: str,
        cpu: Mapping[str, Any],
        ram_bytes: int,
        gpus: list[Mapping[str, Any]],
        os_info: Mapping[str, Any],
    ) -> str:
        identifier = _content_id("host", sha256_json(hardware_fingerprint))
        self.connection.execute(
            """
            INSERT INTO host(
                id, hostname, hardware_fingerprint, cpu_json,
                ram_bytes, gpu_json, os_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(hardware_fingerprint) DO NOTHING
            """,
            (
                identifier,
                hostname,
                hardware_fingerprint,
                canonical_json(dict(cpu)),
                ram_bytes,
                canonical_json([dict(gpu) for gpu in gpus]),
                canonical_json(dict(os_info)),
            ),
        )
        row = self.connection.execute(
            "SELECT id FROM host WHERE hardware_fingerprint = ?",
            (hardware_fingerprint,),
        ).fetchone()
        if row is None:
            raise RuntimeError("host insert did not produce a row")
        return str(row["id"])

    def put_binary(
        self,
        *,
        sha256: str,
        kind: str,
        path: str,
        size_bytes: int,
        mtime_ns: int,
        build_info: Mapping[str, Any] | None = None,
        capabilities: Mapping[str, Any] | None = None,
    ) -> str:
        identifier = _content_id("bin", sha256)
        self.connection.execute(
            """
            INSERT INTO binary(
                id, sha256, kind, path, size_bytes, mtime_ns,
                build_info_json, capabilities_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sha256) DO NOTHING
            """,
            (
                identifier,
                sha256,
                kind,
                path,
                size_bytes,
                mtime_ns,
                canonical_json(dict(build_info or {})),
                canonical_json(dict(capabilities or {})),
            ),
        )
        row = self.connection.execute(
            "SELECT id FROM binary WHERE sha256 = ?",
            (sha256,),
        ).fetchone()
        if row is None:
            raise RuntimeError("binary insert did not produce a row")
        return str(row["id"])


def decode_json_object(value: str) -> dict[str, Any]:
    """Decode one persisted JSON object; useful to callers inspecting raw fields."""
    return _loads_object(value)
