"""Thin SQLite repositories for domain objects and experiment records."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import TypeAdapter

from llama_profile_lab.db.records import (
    BenchmarkCaseRecord,
    BenchmarkRunRecord,
    BinaryRecord,
    ExperimentRecord,
    ExperimentStatus,
    PlacementAttemptRecord,
    PlacementAttemptStatus,
    ResolvedPlacementRecord,
    RunStatus,
)
from llama_profile_lab.domain import (
    Candidate,
    ExperimentDefinition,
    MeasurementPolicy,
    ResolvedPlacement,
    SearchSpace,
    WorkloadSuite,
    canonical_json,
    sha256_json,
)
from llama_profile_lab.domain.workload import WorkloadCase

_WORKLOAD_ADAPTER: TypeAdapter[WorkloadCase] = TypeAdapter(WorkloadCase)


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

    def add_candidate(
        self,
        *,
        experiment_id: str,
        candidate_id: str,
        ordinal: int,
        generation_metadata: Mapping[str, Any],
    ) -> None:
        """Link one generated Candidate to an experiment."""
        self.connection.execute(
            """
            INSERT INTO experiment_candidate(
                experiment_id, candidate_id, ordinal, generation_metadata_json
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                experiment_id,
                candidate_id,
                ordinal,
                canonical_json(dict(generation_metadata)),
            ),
        )

    def add_workload(
        self,
        *,
        experiment_id: str,
        candidate_id: str,
        workload_case_id: str,
        suite_case_index: int,
        expansion_provenance: Mapping[str, Any],
    ) -> None:
        """Link one Candidate-dependent concrete workload to an experiment."""
        self.connection.execute(
            """
            INSERT INTO experiment_workload(
                experiment_id, candidate_id, workload_case_id,
                suite_case_index, expansion_provenance_json
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                candidate_id,
                workload_case_id,
                suite_case_index,
                canonical_json(dict(expansion_provenance)),
            ),
        )

    def mark_planned(self, identifier: str) -> None:
        """Freeze a draft experiment after its complete plan is persisted."""
        cursor = self.connection.execute(
            """
            UPDATE experiment
            SET status = 'planned', frozen_at = ?
            WHERE id = ? AND status = 'draft'
            """,
            (_utc_now(), identifier),
        )
        if cursor.rowcount != 1:
            raise ValueError("experiment does not exist or is not draft")


    def mark_running(self, identifier: str) -> None:
        """Transition a resumable experiment into running state."""
        cursor = self.connection.execute(
            """
            UPDATE experiment
            SET status = 'running', completed_at = NULL
            WHERE id = ?
              AND status IN ('planned', 'paused', 'failed', 'running')
            """,
            (identifier,),
        )
        if cursor.rowcount != 1:
            raise ValueError("experiment does not exist or is not resumable")

    def mark_paused(self, identifier: str) -> None:
        """Pause an executing experiment while retaining its persisted plan."""
        cursor = self.connection.execute(
            "UPDATE experiment SET status = 'paused' WHERE id = ? AND status = 'running'",
            (identifier,),
        )
        if cursor.rowcount != 1:
            raise ValueError("experiment does not exist or is not running")

    def mark_completed(self, identifier: str) -> None:
        """Mark an experiment completed after every case has a successful run."""
        cursor = self.connection.execute(
            """
            UPDATE experiment
            SET status = 'completed', completed_at = ?
            WHERE id = ? AND status = 'running'
            """,
            (_utc_now(), identifier),
        )
        if cursor.rowcount != 1:
            raise ValueError("experiment does not exist or is not running")

    def mark_failed(self, identifier: str) -> None:
        """Mark an experiment failed while preserving retryable case history."""
        cursor = self.connection.execute(
            "UPDATE experiment SET status = 'failed' WHERE id = ? AND status = 'running'",
            (identifier,),
        )
        if cursor.rowcount != 1:
            raise ValueError("experiment does not exist or is not running")


class PlacementRepository:
    """Persistence for fit attempts and immutable successful placements."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def create_attempt(
        self,
        *,
        placement_hash: str,
        candidate_id: str,
        host_id: str,
        binary_id: str,
        model_path: str,
        argv: tuple[str, ...],
        started_at: str | None = None,
    ) -> str:
        identifier = _event_id("fit")
        self.connection.execute(
            """
            INSERT INTO placement_attempt(
                id, placement_hash, candidate_id, host_id, binary_id,
                model_path, started_at, status, argv_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?)
            """,
            (
                identifier,
                placement_hash,
                candidate_id,
                host_id,
                binary_id,
                model_path,
                started_at or _utc_now(),
                canonical_json(list(argv)),
            ),
        )
        return identifier

    def finish_attempt(
        self,
        identifier: str,
        *,
        status: PlacementAttemptStatus,
        duration_ns: int,
        exit_code: int | None,
        stdout: str = "",
        stderr: str = "",
        raw_result: Any = None,
        finished_at: str | None = None,
    ) -> None:
        if status == "running":
            raise ValueError("finished placement attempt must be terminal")
        cursor = self.connection.execute(
            """
            UPDATE placement_attempt
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
                canonical_json(raw_result) if raw_result is not None else None,
                identifier,
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError("placement attempt does not exist or is already finalized")

    def get_attempt(self, identifier: str) -> PlacementAttemptRecord | None:
        row = self.connection.execute(
            """
            SELECT id, placement_hash, candidate_id, host_id, binary_id,
                   model_path, started_at, finished_at, duration_ns,
                   status, exit_code
            FROM placement_attempt
            WHERE id = ?
            """,
            (identifier,),
        ).fetchone()
        if row is None:
            return None
        return PlacementAttemptRecord(
            id=str(row["id"]),
            placement_hash=str(row["placement_hash"]),
            candidate_id=str(row["candidate_id"]),
            host_id=str(row["host_id"]),
            binary_id=str(row["binary_id"]),
            model_path=str(row["model_path"]),
            started_at=str(row["started_at"]),
            finished_at=row["finished_at"],
            duration_ns=row["duration_ns"],
            status=row["status"],
            exit_code=row["exit_code"],
        )

    def recover_orphaned(self) -> int:
        """Finalize stale fit attempts left running by a previous process."""
        now = _utc_now()
        cursor = self.connection.execute(
            """
            UPDATE placement_attempt
            SET status = 'interrupted', finished_at = ?
            WHERE status = 'running'
            """,
            (now,),
        )
        return cursor.rowcount

    def put_resolved(
        self,
        *,
        placement_hash: str,
        candidate_id: str,
        host_id: str,
        binary_id: str,
        fit_attempt_id: str | None,
        placement: ResolvedPlacement,
        request: Mapping[str, Any],
        argv: tuple[str, ...],
        stdout: str,
        stderr: str,
        exit_code: int | None,
        raw_result: Mapping[str, Any] | None = None,
    ) -> str:
        identifier = _content_id("place", placement_hash)
        self.connection.execute(
            """
            INSERT INTO resolved_placement(
                id, placement_hash, candidate_id, host_id, binary_id,
                production_context_size, n_gpu_layers, n_cpu_moe,
                split_mode, main_gpu, tensor_split_json, override_tensor_json,
                argv_json, stdout, stderr, exit_code, raw_result_json,
                fit_attempt_id, devices_json, request_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(placement_hash) DO NOTHING
            """,
            (
                identifier,
                placement_hash,
                candidate_id,
                host_id,
                binary_id,
                placement.production_context_size,
                placement.n_gpu_layers,
                placement.n_cpu_moe,
                placement.split_mode,
                placement.main_gpu,
                (
                    canonical_json(list(placement.tensor_split))
                    if placement.tensor_split is not None
                    else None
                ),
                canonical_json(list(placement.override_tensor)),
                canonical_json(list(argv)),
                stdout,
                stderr,
                exit_code,
                canonical_json(dict(raw_result or {})),
                fit_attempt_id,
                canonical_json(
                    placement.devices
                    if placement.devices == "auto"
                    else list(placement.devices)
                ),
                canonical_json(dict(request)),
            ),
        )
        row = self.connection.execute(
            "SELECT id FROM resolved_placement WHERE placement_hash = ?",
            (placement_hash,),
        ).fetchone()
        if row is None:
            raise RuntimeError("resolved placement insert did not produce a row")
        return str(row["id"])

    def find_by_hash(self, placement_hash: str) -> ResolvedPlacementRecord | None:
        row = self.connection.execute(
            """
            SELECT id, placement_hash, candidate_id, host_id, binary_id,
                   fit_attempt_id, production_context_size, n_gpu_layers,
                   n_cpu_moe, split_mode, main_gpu, devices_json,
                   tensor_split_json, override_tensor_json, request_json,
                   created_at
            FROM resolved_placement
            WHERE placement_hash = ?
            """,
            (placement_hash,),
        ).fetchone()
        return None if row is None else self._record(row)

    def get(self, identifier: str) -> ResolvedPlacementRecord | None:
        row = self.connection.execute(
            """
            SELECT id, placement_hash, candidate_id, host_id, binary_id,
                   fit_attempt_id, production_context_size, n_gpu_layers,
                   n_cpu_moe, split_mode, main_gpu, devices_json,
                   tensor_split_json, override_tensor_json, request_json,
                   created_at
            FROM resolved_placement
            WHERE id = ?
            """,
            (identifier,),
        ).fetchone()
        return None if row is None else self._record(row)

    def list(self) -> tuple[ResolvedPlacementRecord, ...]:
        rows = self.connection.execute(
            """
            SELECT id, placement_hash, candidate_id, host_id, binary_id,
                   fit_attempt_id, production_context_size, n_gpu_layers,
                   n_cpu_moe, split_mode, main_gpu, devices_json,
                   tensor_split_json, override_tensor_json, request_json,
                   created_at
            FROM resolved_placement
            ORDER BY created_at, id
            """
        ).fetchall()
        return tuple(self._record(row) for row in rows)

    @staticmethod
    def _record(row: sqlite3.Row) -> ResolvedPlacementRecord:
        raw_devices = json.loads(str(row["devices_json"]))
        if raw_devices == "auto":
            devices: str | tuple[str, ...] = "auto"
        elif isinstance(raw_devices, list):
            devices = tuple(str(value) for value in raw_devices)
        else:
            raise ValueError("persisted devices_json is invalid")

        raw_tensor_split = (
            None
            if row["tensor_split_json"] is None
            else json.loads(str(row["tensor_split_json"]))
        )
        tensor_split: tuple[float, ...] | None
        if raw_tensor_split is None:
            tensor_split = None
        elif isinstance(raw_tensor_split, list):
            tensor_split = tuple(float(value) for value in raw_tensor_split)
        else:
            raise ValueError("persisted tensor_split_json is invalid")

        raw_overrides = json.loads(str(row["override_tensor_json"]))
        if not isinstance(raw_overrides, list):
            raise ValueError("persisted override_tensor_json is invalid")

        return ResolvedPlacementRecord(
            id=str(row["id"]),
            placement_hash=str(row["placement_hash"]),
            candidate_id=str(row["candidate_id"]),
            host_id=str(row["host_id"]),
            binary_id=str(row["binary_id"]),
            fit_attempt_id=row["fit_attempt_id"],
            production_context_size=int(row["production_context_size"]),
            n_gpu_layers=int(row["n_gpu_layers"]),
            n_cpu_moe=int(row["n_cpu_moe"] or 0),
            split_mode=str(row["split_mode"] or "layer"),
            main_gpu=int(row["main_gpu"] or 0),
            devices=devices,
            tensor_split=tensor_split,
            override_tensor=tuple(str(value) for value in raw_overrides),
            request=_loads_object(str(row["request_json"])),
            created_at=str(row["created_at"]),
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


    def list_incomplete(self, experiment_id: str) -> tuple[BenchmarkCaseRecord, ...]:
        """List cases with no successful run, ordered by planned ordinal."""
        rows = self.connection.execute(
            """
            SELECT bc.id, bc.experiment_id, bc.candidate_id, bc.workload_case_id,
                   bc.placement_id, bc.case_hash, bc.status, bc.ordinal
            FROM benchmark_case AS bc
            WHERE bc.experiment_id = ?
              AND NOT EXISTS (
                  SELECT 1
                  FROM benchmark_run AS br
                  WHERE br.benchmark_case_id = bc.id
                    AND br.status = 'completed'
              )
            ORDER BY bc.ordinal
            """,
            (experiment_id,),
        ).fetchall()
        return tuple(
            BenchmarkCaseRecord(
                id=str(row["id"]),
                experiment_id=str(row["experiment_id"]),
                candidate_id=str(row["candidate_id"]),
                workload_case_id=str(row["workload_case_id"]),
                placement_id=row["placement_id"],
                case_hash=str(row["case_hash"]),
                status=row["status"],
                ordinal=int(row["ordinal"]),
            )
            for row in rows
        )

    def count_incomplete(self, experiment_id: str) -> int:
        """Count cases that still lack a successful run."""
        row = self.connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM benchmark_case AS bc
            WHERE bc.experiment_id = ?
              AND NOT EXISTS (
                  SELECT 1
                  FROM benchmark_run AS br
                  WHERE br.benchmark_case_id = bc.id
                    AND br.status = 'completed'
              )
            """,
            (experiment_id,),
        ).fetchone()
        if row is None:
            return 0
        return int(row["count"])

    def set_status(self, identifier: str, status: RunStatus) -> None:
        """Update the latest execution state of a planned benchmark case."""
        cursor = self.connection.execute(
            "UPDATE benchmark_case SET status = ? WHERE id = ?",
            (status, identifier),
        )
        if cursor.rowcount != 1:
            raise ValueError(f"benchmark case not found: {identifier}")


    def bind_placement_for_candidate(
        self,
        *,
        experiment_id: str,
        candidate_id: str,
        placement_id: str,
    ) -> int:
        """Bind one concrete placement to all cases for a Candidate."""
        rows = self.connection.execute(
            """
            SELECT id, workload_case_id, placement_id
            FROM benchmark_case
            WHERE experiment_id = ? AND candidate_id = ?
            ORDER BY ordinal
            """,
            (experiment_id, candidate_id),
        ).fetchall()

        updated = 0
        for row in rows:
            case_id = str(row["id"])
            existing = row["placement_id"]
            if existing == placement_id:
                continue
            if existing is not None and existing != placement_id:
                successful = self.connection.execute(
                    """
                    SELECT 1
                    FROM benchmark_run
                    WHERE benchmark_case_id = ? AND status = 'completed'
                    LIMIT 1
                    """,
                    (case_id,),
                ).fetchone()
                if successful is not None:
                    raise ValueError(
                        f"cannot change placement for successfully completed case {case_id}"
                    )

            case_hash = sha256_json(
                {
                    "experiment_id": experiment_id,
                    "candidate_id": candidate_id,
                    "workload_case_id": str(row["workload_case_id"]),
                    "placement_id": placement_id,
                }
            )
            self.connection.execute(
                """
                UPDATE benchmark_case
                SET placement_id = ?, case_hash = ?
                WHERE id = ?
                """,
                (placement_id, case_hash, case_id),
            )
            updated += 1
        return updated


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
        raw_result: Any = None,
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
                canonical_json(raw_result) if raw_result is not None else None,
                identifier,
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError("run does not exist or has already been finalized")

    def add_samples(
        self,
        run_id: str,
        samples: Sequence[tuple[int, float]],
    ) -> None:
        """Persist individual timed repetitions for one run."""
        self.connection.executemany(
            """
            INSERT INTO benchmark_sample(
                run_id, sample_index, elapsed_ns, tokens_per_second
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                (run_id, index, elapsed_ns, tokens_per_second)
                for index, (elapsed_ns, tokens_per_second) in enumerate(samples)
            ),
        )

    def add_metrics(
        self,
        run_id: str,
        metrics: Mapping[str, int | float],
    ) -> None:
        """Persist normalized aggregate metrics from tool output."""
        rows: list[tuple[str, str, float | None, int | None, str, str]] = []
        for name, value in metrics.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                rows.append((run_id, name, None, value, "", "{}"))
            else:
                rows.append((run_id, name, float(value), None, "", "{}"))
        self.connection.executemany(
            """
            INSERT INTO metric(
                run_id, metric_name, value_real, value_integer, unit, dimensions_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def recover_orphaned(self, experiment_id: str) -> int:
        """Convert stale running attempts into interrupted history before resume."""
        rows = self.connection.execute(
            """
            SELECT br.id, br.benchmark_case_id
            FROM benchmark_run AS br
            JOIN benchmark_case AS bc ON bc.id = br.benchmark_case_id
            WHERE bc.experiment_id = ? AND br.status = 'running'
            """,
            (experiment_id,),
        ).fetchall()
        if not rows:
            return 0

        now = _utc_now()
        run_ids = [str(row["id"]) for row in rows]
        case_ids = {str(row["benchmark_case_id"]) for row in rows}
        self.connection.executemany(
            """
            UPDATE benchmark_run
            SET status = 'interrupted', finished_at = ?
            WHERE id = ? AND status = 'running'
            """,
            ((now, run_id) for run_id in run_ids),
        )
        for case_id in case_ids:
            successful = self.connection.execute(
                """
                SELECT 1 FROM benchmark_run
                WHERE benchmark_case_id = ? AND status = 'completed'
                LIMIT 1
                """,
                (case_id,),
            ).fetchone()
            if successful is None:
                self.connection.execute(
                    "UPDATE benchmark_case SET status = 'planned' WHERE id = ?",
                    (case_id,),
                )
        return len(run_ids)

    def samples(self, run_id: str) -> tuple[tuple[int, int, float], ...]:
        """Return sample index, elapsed ns, and throughput for a run."""
        rows = self.connection.execute(
            """
            SELECT sample_index, elapsed_ns, tokens_per_second
            FROM benchmark_sample
            WHERE run_id = ?
            ORDER BY sample_index
            """,
            (run_id,),
        ).fetchall()
        return tuple(
            (
                int(row["sample_index"]),
                int(row["elapsed_ns"]),
                float(row["tokens_per_second"]),
            )
            for row in rows
        )

    def metrics(self, run_id: str) -> dict[str, int | float]:
        """Return normalized scalar metrics for a run."""
        rows = self.connection.execute(
            """
            SELECT metric_name, value_real, value_integer
            FROM metric
            WHERE run_id = ?
            ORDER BY metric_name
            """,
            (run_id,),
        ).fetchall()
        metrics: dict[str, int | float] = {}
        for row in rows:
            integer = row["value_integer"]
            real = row["value_real"]
            if integer is not None:
                metrics[str(row["metric_name"])] = int(integer)
            elif real is not None:
                metrics[str(row["metric_name"])] = float(real)
        return metrics

    def logs(self, run_id: str) -> tuple[str, str]:
        """Return captured stdout and stderr for one run."""
        row = self.connection.execute(
            "SELECT stdout, stderr FROM benchmark_run WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"benchmark run not found: {run_id}")
        return str(row["stdout"]), str(row["stderr"])

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
    """Host and exact executable persistence used by discovery and run provenance."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def put_host(
        self,
        *,
        hostname: str,
        hardware_fingerprint: str,
        cpu: Mapping[str, Any],
        ram_bytes: int,
        gpus: Sequence[Mapping[str, Any]],
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
        git_commit: str | None = None,
        git_branch: str | None = None,
        git_dirty: bool | None = None,
        build_number: str | None = None,
        build_info: Mapping[str, Any] | None = None,
        capabilities: Mapping[str, Any] | None = None,
    ) -> str:
        """Register or refresh metadata for one exact executable hash."""
        identifier = _content_id("bin", sha256)
        self.connection.execute(
            """
            INSERT INTO binary(
                id, sha256, kind, path, size_bytes, mtime_ns,
                git_commit, git_branch, git_dirty, build_number,
                build_info_json, capabilities_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sha256) DO UPDATE SET
                kind = excluded.kind,
                path = excluded.path,
                size_bytes = excluded.size_bytes,
                mtime_ns = excluded.mtime_ns,
                git_commit = excluded.git_commit,
                git_branch = excluded.git_branch,
                git_dirty = excluded.git_dirty,
                build_number = excluded.build_number,
                build_info_json = excluded.build_info_json,
                capabilities_json = excluded.capabilities_json
            """,
            (
                identifier,
                sha256,
                kind,
                path,
                size_bytes,
                mtime_ns,
                git_commit,
                git_branch,
                None if git_dirty is None else int(git_dirty),
                build_number,
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

    def get_binary(self, identifier: str) -> BinaryRecord | None:
        """Load one registered executable."""
        row = self.connection.execute(
            """
            SELECT id, sha256, kind, path, size_bytes, mtime_ns,
                   git_commit, git_branch, git_dirty, build_number,
                   build_info_json, capabilities_json, created_at
            FROM binary
            WHERE id = ?
            """,
            (identifier,),
        ).fetchone()
        if row is None:
            return None
        return self._binary_record(row)

    def list_binaries(self) -> tuple[BinaryRecord, ...]:
        """List registered executables deterministically."""
        rows = self.connection.execute(
            """
            SELECT id, sha256, kind, path, size_bytes, mtime_ns,
                   git_commit, git_branch, git_dirty, build_number,
                   build_info_json, capabilities_json, created_at
            FROM binary
            ORDER BY kind, path, sha256
            """
        ).fetchall()
        return tuple(self._binary_record(row) for row in rows)

    @staticmethod
    def _binary_record(row: sqlite3.Row) -> BinaryRecord:
        dirty_value = row["git_dirty"]
        return BinaryRecord(
            id=str(row["id"]),
            sha256=str(row["sha256"]),
            kind=str(row["kind"]),
            path=str(row["path"]),
            size_bytes=int(row["size_bytes"]),
            mtime_ns=int(row["mtime_ns"]),
            git_commit=row["git_commit"],
            git_branch=row["git_branch"],
            git_dirty=None if dirty_value is None else bool(dirty_value),
            build_number=row["build_number"],
            build_info=_loads_object(str(row["build_info_json"])),
            capabilities=_loads_object(str(row["capabilities_json"])),
            created_at=str(row["created_at"]),
        )


def decode_json_object(value: str) -> dict[str, Any]:
    """Decode one persisted JSON object; useful to callers inspecting raw fields."""
    return _loads_object(value)
