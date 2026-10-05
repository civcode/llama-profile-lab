"""Thin SQLite repositories for domain objects and experiment records."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import TypeAdapter

from llama_profile_lab.db.connection import transaction
from llama_profile_lab.db.records import (
    AcceleratorDeviceRecord,
    BenchmarkCaseRecord,
    BenchmarkRunRecord,
    BinaryRecord,
    CandidateEvaluationRecord,
    DeploymentCandidateRecord,
    DeploymentDeviceAllocationRecord,
    DeploymentInstanceRecord,
    DeploymentPlacementRecord,
    DeploymentRejectionRecord,
    DeploymentRunMemberRecord,
    DeploymentRunRecord,
    ExperimentRecord,
    ExperimentStatus,
    MemoryEstimateAttemptRecord,
    MemoryEstimateAttemptStatus,
    MemoryEstimateDeviceRecord,
    MemoryEstimateRecord,
    PlacementAttemptRecord,
    PlacementAttemptStatus,
    PlacementDeviceMemoryRecord,
    ResolvedPlacementRecord,
    RunStatus,
    ServerBenchmarkRecord,
    ServerBenchmarkStatus,
    ServerRunRecord,
    ServerRunStatus,
)
from llama_profile_lab.domain import (
    AcceleratorDevice,
    Candidate,
    DeploymentCandidate,
    DeploymentFailureKind,
    DeploymentPlacement,
    DeploymentRunStatus,
    ExperimentDefinition,
    MeasurementPolicy,
    MemoryEstimateIdentity,
    MemoryEstimateOutput,
    ResolvedPlacement,
    SearchSpace,
    TelemetrySample,
    WorkloadSuite,
    canonical_json,
    sha256_json,
)
from llama_profile_lab.domain.telemetry import GpuTelemetrySample, RunQuality
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

    def list(self) -> tuple[ExperimentRecord, ...]:
        """List experiments newest first without exposing persistence JSON."""
        rows = self.connection.execute(
            """
            SELECT id, status, name, base_candidate_id, search_space_id,
                   workload_suite_id, measurement_policy_id,
                   created_at, frozen_at, completed_at
            FROM experiment
            ORDER BY created_at DESC, id DESC
            """
        ).fetchall()
        return tuple(
            ExperimentRecord(
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
            for row in rows
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

    def mark_cancelled(self, identifier: str) -> None:
        """Mark unfinished experiment work cancelled without deleting its plan/history."""
        cursor = self.connection.execute(
            """
            UPDATE experiment
            SET status = 'cancelled'
            WHERE id = ?
              AND status IN ('draft', 'planned', 'running', 'paused', 'failed')
            """,
            (identifier,),
        )
        if cursor.rowcount != 1:
            raise ValueError("experiment does not exist or is already terminal")


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

    def set_quality(
        self,
        run_id: str,
        *,
        quality: RunQuality,
        details: Mapping[str, Any],
    ) -> None:
        """Attach telemetry quality without changing benchmark success state."""
        cursor = self.connection.execute(
            """
            UPDATE benchmark_run
            SET quality = ?, quality_details_json = ?
            WHERE id = ?
            """,
            (quality, canonical_json(dict(details)), run_id),
        )
        if cursor.rowcount != 1:
            raise ValueError(f"benchmark run not found: {run_id}")

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
                   duration_ns, status, exit_code, quality, quality_details_json
            FROM benchmark_run
            WHERE id = ?
            """,
            (identifier,),
        ).fetchone()
        if row is None:
            return None
        return self._record(row)

    def list_for_experiment(
        self,
        experiment_id: str,
    ) -> tuple[BenchmarkRunRecord, ...]:
        """List all append-only attempts for an experiment in start order."""
        rows = self.connection.execute(
            """
            SELECT br.id, br.benchmark_case_id, br.host_id, br.binary_id,
                   br.measurement_policy_id, br.started_at, br.finished_at,
                   br.duration_ns, br.status, br.exit_code, br.quality,
                   br.quality_details_json
            FROM benchmark_run AS br
            JOIN benchmark_case AS bc ON bc.id = br.benchmark_case_id
            WHERE bc.experiment_id = ?
            ORDER BY br.started_at, br.id
            """,
            (experiment_id,),
        ).fetchall()
        return tuple(self._record(row) for row in rows)

    @staticmethod
    def _record(row: sqlite3.Row) -> BenchmarkRunRecord:
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
            quality=row["quality"],
            quality_details=(
                None
                if row["quality_details_json"] is None
                else _loads_object(str(row["quality_details_json"]))
            ),
        )


class TelemetryRepository:
    """Persistence for raw run telemetry observations."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def add_samples(
        self,
        run_id: str,
        samples: Sequence[TelemetrySample],
    ) -> None:
        rows = []
        for sample in samples:
            extra = dict(sample.extra)
            extra["phase"] = sample.phase
            rows.append(
                (
                    run_id,
                    sample.timestamp_ns,
                    sample.cpu_system_pct,
                    sample.cpu_user_pct,
                    sample.cpu_system_mode_pct,
                    sample.cpu_iowait_pct,
                    sample.process_cpu_pct_normalized,
                    sample.process_cpu_pct_raw,
                    sample.process_user_time_ns,
                    sample.process_system_time_ns,
                    sample.process_threads,
                    sample.cpu_freq_avg_hz,
                    sample.cpu_freq_min_hz,
                    sample.cpu_freq_max_hz,
                    sample.cpu_temperature_c,
                    sample.load_avg_1m,
                    sample.load_avg_5m,
                    sample.ram_used_bytes,
                    sample.ram_available_bytes,
                    sample.swap_used_bytes,
                    sample.process_rss_bytes,
                    canonical_json(
                        [gpu.model_dump(mode="json") for gpu in sample.gpus]
                    ),
                    canonical_json(list(sample.cpu_per_core_pct)),
                    canonical_json(extra),
                )
            )
        self.connection.executemany(
            """
            INSERT INTO telemetry_sample(
                run_id, timestamp_ns, cpu_system_pct, cpu_user_pct,
                cpu_system_mode_pct, cpu_iowait_pct,
                process_cpu_pct_normalized, process_cpu_pct_raw,
                process_user_time_ns, process_system_time_ns,
                process_threads, cpu_freq_avg_hz, cpu_freq_min_hz,
                cpu_freq_max_hz, cpu_temperature_c, load_avg_1m,
                load_avg_5m, ram_used_bytes, ram_available_bytes,
                swap_used_bytes, process_rss_bytes, gpu_json,
                cpu_per_core_json, extra_json
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?
            )
            """,
            rows,
        )

    def samples(self, run_id: str) -> tuple[TelemetrySample, ...]:
        rows = self.connection.execute(
            """
            SELECT timestamp_ns, cpu_system_pct, cpu_user_pct,
                   cpu_system_mode_pct, cpu_iowait_pct,
                   process_cpu_pct_normalized, process_cpu_pct_raw,
                   process_user_time_ns, process_system_time_ns,
                   process_threads, cpu_freq_avg_hz, cpu_freq_min_hz,
                   cpu_freq_max_hz, cpu_temperature_c, load_avg_1m,
                   load_avg_5m, ram_used_bytes, ram_available_bytes,
                   swap_used_bytes, process_rss_bytes, gpu_json,
                   cpu_per_core_json, extra_json
            FROM telemetry_sample
            WHERE run_id = ?
            ORDER BY timestamp_ns
            """,
            (run_id,),
        ).fetchall()
        result: list[TelemetrySample] = []
        for row in rows:
            extra = _loads_object(str(row["extra_json"]))
            phase = extra.pop("phase", "during")
            raw_gpus = json.loads(str(row["gpu_json"]))
            if not isinstance(raw_gpus, list):
                raise ValueError("persisted gpu_json must be a list")
            raw_cores = json.loads(str(row["cpu_per_core_json"]))
            if not isinstance(raw_cores, list):
                raise ValueError("persisted cpu_per_core_json must be a list")
            result.append(
                TelemetrySample(
                    timestamp_ns=int(row["timestamp_ns"]),
                    phase=phase,
                    cpu_system_pct=row["cpu_system_pct"],
                    cpu_user_pct=row["cpu_user_pct"],
                    cpu_system_mode_pct=row["cpu_system_mode_pct"],
                    cpu_iowait_pct=row["cpu_iowait_pct"],
                    process_cpu_pct_normalized=row["process_cpu_pct_normalized"],
                    process_cpu_pct_raw=row["process_cpu_pct_raw"],
                    process_user_time_ns=row["process_user_time_ns"],
                    process_system_time_ns=row["process_system_time_ns"],
                    process_threads=row["process_threads"],
                    cpu_freq_avg_hz=row["cpu_freq_avg_hz"],
                    cpu_freq_min_hz=row["cpu_freq_min_hz"],
                    cpu_freq_max_hz=row["cpu_freq_max_hz"],
                    cpu_temperature_c=row["cpu_temperature_c"],
                    load_avg_1m=row["load_avg_1m"],
                    load_avg_5m=row["load_avg_5m"],
                    ram_used_bytes=row["ram_used_bytes"],
                    ram_available_bytes=row["ram_available_bytes"],
                    swap_used_bytes=row["swap_used_bytes"],
                    process_rss_bytes=row["process_rss_bytes"],
                    gpus=tuple(
                        GpuTelemetrySample.model_validate(gpu)
                        for gpu in raw_gpus
                    ),
                    cpu_per_core_pct=tuple(float(value) for value in raw_cores),
                    extra=extra,
                )
            )
        return tuple(result)


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


class ServerValidationRepository:
    """Persistence for managed llama-server and SPEED-Bench validation events."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def create_run(
        self,
        *,
        experiment_id: str,
        candidate_id: str,
        placement_id: str | None,
        host_id: str,
        server_binary_id: str,
        target_model_id: str,
        draft_model_id: str | None,
        target_model_path: str,
        draft_model_path: str | None,
        spec_type: str | None,
        spec_draft_n_max: int | None,
        bind_host: str,
        bind_port: int,
        argv: tuple[str, ...],
        environment: Mapping[str, str],
        started_at: str | None = None,
    ) -> str:
        identifier = _event_id("srv")
        self.connection.execute(
            """
            INSERT INTO server_run(
                id, experiment_id, candidate_id, placement_id, host_id,
                server_binary_id, target_model_id, draft_model_id,
                target_model_path, draft_model_path, spec_type, spec_draft_n_max,
                argv_json, started_at, status, bind_host, bind_port,
                environment_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'starting', ?, ?, ?)
            """,
            (
                identifier,
                experiment_id,
                candidate_id,
                placement_id,
                host_id,
                server_binary_id,
                target_model_id,
                draft_model_id,
                target_model_path,
                draft_model_path,
                spec_type,
                spec_draft_n_max,
                canonical_json(list(argv)),
                started_at or _utc_now(),
                bind_host,
                bind_port,
                canonical_json(dict(environment)),
            ),
        )
        return identifier

    def mark_ready(self, identifier: str, *, ready_at: str | None = None) -> None:
        cursor = self.connection.execute(
            """
            UPDATE server_run
            SET status = 'ready', ready_at = ?
            WHERE id = ? AND status = 'starting'
            """,
            (ready_at or _utc_now(), identifier),
        )
        if cursor.rowcount != 1:
            raise ValueError("server run does not exist or is not starting")

    def finish_run(
        self,
        identifier: str,
        *,
        status: ServerRunStatus,
        duration_ns: int,
        exit_code: int | None,
        stdout: str = "",
        stderr: str = "",
        finished_at: str | None = None,
    ) -> None:
        if status in {"starting", "ready"}:
            raise ValueError("finished server run must be terminal")
        cursor = self.connection.execute(
            """
            UPDATE server_run
            SET finished_at = ?, duration_ns = ?, status = ?, exit_code = ?,
                stdout = ?, stderr = ?
            WHERE id = ? AND status IN ('starting', 'ready')
            """,
            (
                finished_at or _utc_now(),
                duration_ns,
                status,
                exit_code,
                stdout,
                stderr,
                identifier,
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError("server run does not exist or is already finalized")

    def get_run(self, identifier: str) -> ServerRunRecord | None:
        row = self.connection.execute(
            """
            SELECT id, experiment_id, candidate_id, placement_id, host_id,
                   server_binary_id, target_model_id, draft_model_id,
                   target_model_path, draft_model_path, spec_type, spec_draft_n_max,
                   bind_host, bind_port, started_at, ready_at, finished_at,
                   duration_ns, status, exit_code
            FROM server_run
            WHERE id = ?
            """,
            (identifier,),
        ).fetchone()
        return None if row is None else self._run_record(row)

    def recover_orphaned(self, experiment_id: str | None = None) -> int:
        query = """
            UPDATE server_run
            SET status = 'interrupted', finished_at = ?
            WHERE status IN ('starting', 'ready')
        """
        parameters: tuple[Any, ...]
        if experiment_id is None:
            parameters = (_utc_now(),)
        else:
            query += " AND experiment_id = ?"
            parameters = (_utc_now(), experiment_id)
        cursor = self.connection.execute(query, parameters)
        self.connection.execute(
            """
            UPDATE server_benchmark
            SET status = 'interrupted'
            WHERE status = 'running'
              AND server_run_id IN (
                  SELECT id FROM server_run WHERE status = 'interrupted'
              )
            """
        )
        return cursor.rowcount

    def create_benchmark(
        self,
        *,
        server_run_id: str,
        workload_case_id: str,
        speed_bench_binary_id: str,
        category: str,
        argv: tuple[str, ...],
    ) -> str:
        identifier = _event_id("srvbench")
        self.connection.execute(
            """
            INSERT INTO server_benchmark(
                id, server_run_id, workload_case_id, speed_bench_binary_id,
                category, status, argv_json, raw_json
            )
            VALUES (?, ?, ?, ?, ?, 'running', ?, '{}')
            """,
            (
                identifier,
                server_run_id,
                workload_case_id,
                speed_bench_binary_id,
                category,
                canonical_json(list(argv)),
            ),
        )
        return identifier

    def finish_benchmark(
        self,
        identifier: str,
        *,
        status: ServerBenchmarkStatus,
        duration_ns: int,
        exit_code: int | None,
        stdout: str,
        stderr: str,
        raw_result: Mapping[str, Any] | None = None,
        requests: int | None = None,
        failed: int | None = None,
        turns: int | None = None,
        avg_prompt_ts: float | None = None,
        avg_pred_ts: float | None = None,
        avg_latency_ms: float | None = None,
        draft_n: int | None = None,
        accepted_n: int | None = None,
        accept_rate: float | None = None,
    ) -> None:
        if status == "running":
            raise ValueError("finished server benchmark must be terminal")
        cursor = self.connection.execute(
            """
            UPDATE server_benchmark
            SET status = ?, duration_ns = ?, exit_code = ?, stdout = ?, stderr = ?,
                raw_json = ?, requests = ?, failed = ?, turns = ?,
                avg_prompt_ts = ?, avg_pred_ts = ?, avg_latency_ms = ?,
                draft_n = ?, accepted_n = ?, accept_rate = ?
            WHERE id = ? AND status = 'running'
            """,
            (
                status,
                duration_ns,
                exit_code,
                stdout,
                stderr,
                canonical_json(dict(raw_result or {})),
                requests,
                failed,
                turns,
                avg_prompt_ts,
                avg_pred_ts,
                avg_latency_ms,
                draft_n,
                accepted_n,
                accept_rate,
                identifier,
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError("server benchmark does not exist or is already finalized")

    def benchmarks_for_candidate(
        self,
        *,
        experiment_id: str,
        candidate_id: str,
        completed_only: bool = True,
    ) -> tuple[ServerBenchmarkRecord, ...]:
        query = """
            SELECT sb.id, sb.server_run_id, sb.workload_case_id,
                   sb.speed_bench_binary_id, sb.category, sb.status,
                   sb.duration_ns, sb.exit_code, sb.requests, sb.failed, sb.turns,
                   sb.avg_prompt_ts, sb.avg_pred_ts, sb.avg_latency_ms,
                   sb.draft_n, sb.accepted_n, sb.accept_rate, sb.created_at
            FROM server_benchmark AS sb
            JOIN server_run AS sr ON sr.id = sb.server_run_id
            WHERE sr.experiment_id = ? AND sr.candidate_id = ?
        """
        parameters: list[Any] = [experiment_id, candidate_id]
        if completed_only:
            query += " AND sb.status = 'completed'"
        query += " ORDER BY sb.created_at, sb.id"
        rows = self.connection.execute(query, tuple(parameters)).fetchall()
        return tuple(self._benchmark_record(row) for row in rows)

    def add_evaluation(
        self,
        *,
        experiment_id: str,
        candidate_id: str,
        stage: str,
        decision: str,
        reason: str | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> str:
        identifier = _event_id("eval")
        self.connection.execute(
            """
            INSERT INTO candidate_evaluation(
                id, experiment_id, candidate_id, stage, decision, reason, metrics_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                identifier,
                experiment_id,
                candidate_id,
                stage,
                decision,
                reason,
                canonical_json(dict(metrics or {})),
            ),
        )
        return identifier

    def evaluations(
        self,
        *,
        experiment_id: str,
        candidate_id: str,
    ) -> tuple[CandidateEvaluationRecord, ...]:
        rows = self.connection.execute(
            """
            SELECT id, experiment_id, candidate_id, stage, decision,
                   reason, metrics_json, created_at
            FROM candidate_evaluation
            WHERE experiment_id = ? AND candidate_id = ?
            ORDER BY created_at, id
            """,
            (experiment_id, candidate_id),
        ).fetchall()
        return tuple(
            CandidateEvaluationRecord(
                id=str(row["id"]),
                experiment_id=str(row["experiment_id"]),
                candidate_id=str(row["candidate_id"]),
                stage=str(row["stage"]),
                decision=str(row["decision"]),
                reason=row["reason"],
                metrics=_loads_object(str(row["metrics_json"])),
                created_at=str(row["created_at"]),
            )
            for row in rows
        )

    @staticmethod
    def _run_record(row: sqlite3.Row) -> ServerRunRecord:
        return ServerRunRecord(
            id=str(row["id"]),
            experiment_id=str(row["experiment_id"]),
            candidate_id=str(row["candidate_id"]),
            placement_id=row["placement_id"],
            host_id=str(row["host_id"]),
            server_binary_id=str(row["server_binary_id"]),
            target_model_id=str(row["target_model_id"]),
            draft_model_id=row["draft_model_id"],
            target_model_path=row["target_model_path"],
            draft_model_path=row["draft_model_path"],
            spec_type=row["spec_type"],
            spec_draft_n_max=row["spec_draft_n_max"],
            bind_host=str(row["bind_host"]),
            bind_port=row["bind_port"],
            started_at=str(row["started_at"]),
            ready_at=row["ready_at"],
            finished_at=row["finished_at"],
            duration_ns=row["duration_ns"],
            status=row["status"],
            exit_code=row["exit_code"],
        )

    @staticmethod
    def _benchmark_record(row: sqlite3.Row) -> ServerBenchmarkRecord:
        return ServerBenchmarkRecord(
            id=str(row["id"]),
            server_run_id=str(row["server_run_id"]),
            workload_case_id=str(row["workload_case_id"]),
            speed_bench_binary_id=row["speed_bench_binary_id"],
            category=str(row["category"]),
            status=row["status"],
            duration_ns=row["duration_ns"],
            exit_code=row["exit_code"],
            requests=row["requests"],
            failed=row["failed"],
            turns=row["turns"],
            avg_prompt_ts=row["avg_prompt_ts"],
            avg_pred_ts=row["avg_pred_ts"],
            avg_latency_ms=row["avg_latency_ms"],
            draft_n=row["draft_n"],
            accepted_n=row["accepted_n"],
            accept_rate=row["accept_rate"],
            created_at=str(row["created_at"]),
        )


class DeploymentCandidateRepository:
    """Persistence for immutable multi-model deployment Candidates."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def put(self, deployment: DeploymentCandidate) -> str:
        """Persist one deployment and its normalized instance rows atomically."""
        digest = deployment.content_hash()
        identifier = _content_id("deploy", digest)
        with transaction(self.connection, immediate=True):
            cursor = self.connection.execute(
                """
                INSERT INTO deployment_candidate(
                    id, deployment_hash, workload_suite_id,
                    resource_policy_json, definition_json
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(deployment_hash) DO NOTHING
                """,
                (
                    identifier,
                    digest,
                    deployment.workload_mix.workload_suite_id,
                    canonical_json(deployment.resource_policy),
                    canonical_json(deployment),
                ),
            )
            row = self.connection.execute(
                "SELECT id FROM deployment_candidate WHERE deployment_hash = ?",
                (digest,),
            ).fetchone()
            if row is None:
                raise RuntimeError("deployment insert did not produce a row")
            persisted_id = str(row["id"])

            if cursor.rowcount == 1:
                self.connection.executemany(
                    """
                    INSERT INTO deployment_instance(
                        deployment_candidate_id, instance_id, candidate_id,
                        role, model_artifact_id, binary_id,
                        requested_placement_json, server_identity, ordinal
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            persisted_id,
                            instance.instance_id,
                            instance.candidate_id,
                            instance.role,
                            instance.model_artifact_id,
                            instance.binary_id,
                            canonical_json(instance.requested_placement),
                            instance.server_identity,
                            ordinal,
                        )
                        for ordinal, instance in enumerate(deployment.instances)
                    ),
                )
            else:
                existing = self.get(persisted_id)
                if existing != deployment:
                    raise RuntimeError(
                        "deployment hash collision or persisted definition mismatch"
                    )

        return persisted_id

    def get(self, identifier: str) -> DeploymentCandidate | None:
        row = self.connection.execute(
            "SELECT definition_json FROM deployment_candidate WHERE id = ?",
            (identifier,),
        ).fetchone()
        if row is None:
            return None
        return DeploymentCandidate.model_validate_json(str(row["definition_json"]))

    def find_by_hash(self, deployment_hash: str) -> DeploymentCandidateRecord | None:
        row = self.connection.execute(
            """
            SELECT id, deployment_hash, workload_suite_id, created_at
            FROM deployment_candidate
            WHERE deployment_hash = ?
            """,
            (deployment_hash,),
        ).fetchone()
        return None if row is None else self._record(row)

    def record(self, identifier: str) -> DeploymentCandidateRecord | None:
        row = self.connection.execute(
            """
            SELECT id, deployment_hash, workload_suite_id, created_at
            FROM deployment_candidate
            WHERE id = ?
            """,
            (identifier,),
        ).fetchone()
        return None if row is None else self._record(row)

    def instances(self, identifier: str) -> tuple[DeploymentInstanceRecord, ...]:
        rows = self.connection.execute(
            """
            SELECT deployment_candidate_id, instance_id, candidate_id,
                   role, model_artifact_id, binary_id,
                   requested_placement_json, server_identity, ordinal
            FROM deployment_instance
            WHERE deployment_candidate_id = ?
            ORDER BY ordinal
            """,
            (identifier,),
        ).fetchall()
        return tuple(
            DeploymentInstanceRecord(
                deployment_candidate_id=str(row["deployment_candidate_id"]),
                instance_id=str(row["instance_id"]),
                candidate_id=str(row["candidate_id"]),
                role=str(row["role"]),
                model_artifact_id=str(row["model_artifact_id"]),
                binary_id=str(row["binary_id"]),
                requested_placement=_loads_object(
                    str(row["requested_placement_json"])
                ),
                server_identity=str(row["server_identity"]),
                ordinal=int(row["ordinal"]),
            )
            for row in rows
        )

    def add_rejection(
        self,
        deployment_candidate_id: str,
        *,
        stage: str,
        reason: str,
        details: Mapping[str, Any] | None = None,
    ) -> str:
        """Persist one explainable rejection for a deployment Candidate."""
        identifier = _event_id("deployreject")
        self.connection.execute(
            """
            INSERT INTO deployment_rejection(
                id, deployment_candidate_id, stage, reason, details_json
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                identifier,
                deployment_candidate_id,
                stage,
                reason,
                canonical_json(dict(details or {})),
            ),
        )
        return identifier

    def rejections(
        self,
        deployment_candidate_id: str,
    ) -> tuple[DeploymentRejectionRecord, ...]:
        rows = self.connection.execute(
            """
            SELECT id, deployment_candidate_id, stage, reason,
                   details_json, created_at
            FROM deployment_rejection
            WHERE deployment_candidate_id = ?
            ORDER BY created_at, id
            """,
            (deployment_candidate_id,),
        ).fetchall()
        return tuple(
            DeploymentRejectionRecord(
                id=str(row["id"]),
                deployment_candidate_id=str(row["deployment_candidate_id"]),
                stage=str(row["stage"]),
                reason=str(row["reason"]),
                details=_loads_object(str(row["details_json"])),
                created_at=str(row["created_at"]),
            )
            for row in rows
        )

    @staticmethod
    def _record(row: sqlite3.Row) -> DeploymentCandidateRecord:
        return DeploymentCandidateRecord(
            id=str(row["id"]),
            deployment_hash=str(row["deployment_hash"]),
            workload_suite_id=str(row["workload_suite_id"]),
            created_at=str(row["created_at"]),
        )


class DeploymentPlacementRepository:
    """Persistence for immutable joint placements and per-device memory."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def put(
        self,
        placement: DeploymentPlacement,
        *,
        request: Mapping[str, Any] | None = None,
    ) -> str:
        """Persist one fully resolved deployment placement atomically."""
        expected_rows = self.connection.execute(
            """
            SELECT instance_id, candidate_id
            FROM deployment_instance
            WHERE deployment_candidate_id = ?
            ORDER BY instance_id
            """,
            (placement.deployment_candidate_id,),
        ).fetchall()
        if not expected_rows:
            raise ValueError(
                f"deployment candidate not found: {placement.deployment_candidate_id}"
            )
        expected = {
            str(row["instance_id"]): str(row["candidate_id"])
            for row in expected_rows
        }
        actual = {item.instance_id for item in placement.instance_placements}
        if actual != set(expected):
            raise ValueError(
                "deployment placement must contain exactly the deployment instances"
            )

        for item in placement.instance_placements:
            row = self.connection.execute(
                """
                SELECT candidate_id, host_id
                FROM resolved_placement
                WHERE id = ?
                """,
                (item.resolved_placement_id,),
            ).fetchone()
            if row is None:
                raise ValueError(
                    f"resolved placement not found: {item.resolved_placement_id}"
                )
            if str(row["candidate_id"]) != expected[item.instance_id]:
                raise ValueError(
                    "resolved placement Candidate does not match deployment instance"
                )
            if str(row["host_id"]) != placement.host_id:
                raise ValueError(
                    "resolved placement host does not match deployment placement host"
                )

        for item in placement.device_memory:
            if item.instance_id not in expected:
                raise ValueError(
                    f"device memory references unknown instance: {item.instance_id}"
                )

        digest = placement.content_hash()
        identifier = _content_id("deployplace", digest)
        with transaction(self.connection, immediate=True):
            cursor = self.connection.execute(
                """
                INSERT INTO deployment_placement(
                    id, placement_hash, deployment_candidate_id, host_id,
                    feasibility, request_json, result_json, provenance_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(placement_hash) DO NOTHING
                """,
                (
                    identifier,
                    digest,
                    placement.deployment_candidate_id,
                    placement.host_id,
                    placement.feasibility,
                    canonical_json(dict(request or {})),
                    canonical_json(placement),
                    canonical_json(dict(placement.provenance)),
                ),
            )
            if cursor.rowcount == 1:
                self.connection.executemany(
                    """
                    INSERT INTO deployment_instance_placement(
                        deployment_placement_id, deployment_candidate_id,
                        instance_id, resolved_placement_id
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        (
                            identifier,
                            placement.deployment_candidate_id,
                            item.instance_id,
                            item.resolved_placement_id,
                        )
                        for item in placement.instance_placements
                    ),
                )
                self.connection.executemany(
                    """
                    INSERT INTO placement_device_memory(
                        id, deployment_placement_id, deployment_candidate_id,
                        instance_id, device_id, model_bytes, context_bytes,
                        compute_bytes, total_bytes, device_total_bytes,
                        device_free_bytes, source, measured_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            _content_id(
                                "depmem",
                                sha256_json(
                                    {
                                        "deployment_placement_id": identifier,
                                        "instance_id": item.instance_id,
                                        "device_id": item.device_id,
                                    }
                                ),
                            ),
                            identifier,
                            placement.deployment_candidate_id,
                            item.instance_id,
                            item.device_id,
                            item.model_bytes,
                            item.context_bytes,
                            item.compute_bytes,
                            item.total_bytes,
                            item.device_total_bytes,
                            item.device_free_bytes,
                            item.source,
                            item.measured_at,
                        )
                        for item in placement.device_memory
                    ),
                )
                self.connection.executemany(
                    """
                    INSERT INTO deployment_device_allocation(
                        deployment_placement_id, device_id, projected_bytes,
                        reserved_margin_bytes, device_total_bytes,
                        projected_free_bytes
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            identifier,
                            item.device_id,
                            item.projected_bytes,
                            item.reserved_margin_bytes,
                            item.device_total_bytes,
                            item.projected_free_bytes,
                        )
                        for item in placement.device_allocations
                    ),
                )
            else:
                existing = self.get(identifier)
                if (
                    existing is None
                    or existing.identity_payload() != placement.identity_payload()
                ):
                    raise RuntimeError(
                        "deployment placement hash collision or persisted mismatch"
                    )

        return identifier

    def get(self, identifier: str) -> DeploymentPlacement | None:
        row = self.connection.execute(
            "SELECT result_json FROM deployment_placement WHERE id = ?",
            (identifier,),
        ).fetchone()
        if row is None:
            return None
        return DeploymentPlacement.model_validate_json(str(row["result_json"]))

    def record(self, identifier: str) -> DeploymentPlacementRecord | None:
        row = self.connection.execute(
            """
            SELECT id, placement_hash, deployment_candidate_id, host_id,
                   feasibility, request_json, provenance_json, created_at
            FROM deployment_placement
            WHERE id = ?
            """,
            (identifier,),
        ).fetchone()
        if row is None:
            return None
        return DeploymentPlacementRecord(
            id=str(row["id"]),
            placement_hash=str(row["placement_hash"]),
            deployment_candidate_id=str(row["deployment_candidate_id"]),
            host_id=str(row["host_id"]),
            feasibility=row["feasibility"],
            request=_loads_object(str(row["request_json"])),
            provenance=_loads_object(str(row["provenance_json"])),
            created_at=str(row["created_at"]),
        )

    def memory(
        self,
        identifier: str,
    ) -> tuple[PlacementDeviceMemoryRecord, ...]:
        rows = self.connection.execute(
            """
            SELECT id, deployment_placement_id, deployment_candidate_id,
                   instance_id, device_id, model_bytes, context_bytes,
                   compute_bytes, total_bytes, device_total_bytes,
                   device_free_bytes, source, measured_at
            FROM placement_device_memory
            WHERE deployment_placement_id = ?
            ORDER BY instance_id, device_id
            """,
            (identifier,),
        ).fetchall()
        return tuple(
            PlacementDeviceMemoryRecord(
                id=str(row["id"]),
                deployment_placement_id=str(row["deployment_placement_id"]),
                deployment_candidate_id=str(row["deployment_candidate_id"]),
                instance_id=str(row["instance_id"]),
                device_id=str(row["device_id"]),
                model_bytes=int(row["model_bytes"]),
                context_bytes=int(row["context_bytes"]),
                compute_bytes=int(row["compute_bytes"]),
                total_bytes=int(row["total_bytes"]),
                device_total_bytes=int(row["device_total_bytes"]),
                device_free_bytes=int(row["device_free_bytes"]),
                source=str(row["source"]),
                measured_at=row["measured_at"],
            )
            for row in rows
        )

    def allocations(
        self,
        identifier: str,
    ) -> tuple[DeploymentDeviceAllocationRecord, ...]:
        rows = self.connection.execute(
            """
            SELECT deployment_placement_id, device_id, projected_bytes,
                   reserved_margin_bytes, device_total_bytes,
                   projected_free_bytes
            FROM deployment_device_allocation
            WHERE deployment_placement_id = ?
            ORDER BY device_id
            """,
            (identifier,),
        ).fetchall()
        return tuple(
            DeploymentDeviceAllocationRecord(
                deployment_placement_id=str(row["deployment_placement_id"]),
                device_id=str(row["device_id"]),
                projected_bytes=int(row["projected_bytes"]),
                reserved_margin_bytes=int(row["reserved_margin_bytes"]),
                device_total_bytes=int(row["device_total_bytes"]),
                projected_free_bytes=int(row["projected_free_bytes"]),
            )
            for row in rows
        )


class DeploymentRunRepository:
    """Persistence for deployment execution attempts and their member processes."""

    _TERMINAL = frozenset({"completed", "cancelled", "failed"})

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def create(
        self,
        *,
        deployment_candidate_id: str,
        deployment_placement_id: str | None = None,
        workload_case_id: str | None = None,
        status: DeploymentRunStatus = "planned",
        started_at: str | None = None,
    ) -> str:
        if deployment_placement_id is not None:
            row = self.connection.execute(
                """
                SELECT deployment_candidate_id
                FROM deployment_placement
                WHERE id = ?
                """,
                (deployment_placement_id,),
            ).fetchone()
            if row is None:
                raise ValueError(
                    f"deployment placement not found: {deployment_placement_id}"
                )
            if str(row["deployment_candidate_id"]) != deployment_candidate_id:
                raise ValueError(
                    "deployment placement belongs to a different deployment Candidate"
                )

        identifier = _event_id("deployrun")
        self.connection.execute(
            """
            INSERT INTO deployment_run(
                id, deployment_candidate_id, deployment_placement_id,
                workload_case_id, status, started_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                identifier,
                deployment_candidate_id,
                deployment_placement_id,
                workload_case_id,
                status,
                started_at,
            ),
        )
        return identifier

    def set_status(
        self,
        identifier: str,
        status: DeploymentRunStatus,
        *,
        started_at: str | None = None,
    ) -> None:
        if status in self._TERMINAL:
            raise ValueError("use finish() for terminal deployment-run states")
        cursor = self.connection.execute(
            """
            UPDATE deployment_run
            SET status = ?,
                started_at = COALESCE(started_at, ?)
            WHERE id = ?
              AND status NOT IN ('completed', 'cancelled', 'failed')
            """,
            (status, started_at, identifier),
        )
        if cursor.rowcount != 1:
            raise ValueError("deployment run does not exist or is already terminal")

    def add_member(
        self,
        deployment_run_id: str,
        *,
        instance_id: str,
        server_run_id: str | None = None,
        client_run_id: str | None = None,
        result: Mapping[str, Any] | None = None,
    ) -> None:
        row = self.connection.execute(
            """
            SELECT deployment_candidate_id
            FROM deployment_run
            WHERE id = ?
            """,
            (deployment_run_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"deployment run not found: {deployment_run_id}")
        deployment_candidate_id = str(row["deployment_candidate_id"])
        member = self.connection.execute(
            """
            SELECT 1
            FROM deployment_instance
            WHERE deployment_candidate_id = ? AND instance_id = ?
            """,
            (deployment_candidate_id, instance_id),
        ).fetchone()
        if member is None:
            raise ValueError(
                f"deployment instance not found for run: {instance_id}"
            )
        self.connection.execute(
            """
            INSERT INTO deployment_run_member(
                deployment_run_id, deployment_candidate_id, instance_id,
                server_run_id, client_run_id, result_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                deployment_run_id,
                deployment_candidate_id,
                instance_id,
                server_run_id,
                client_run_id,
                canonical_json(dict(result or {})),
            ),
        )

    def finish(
        self,
        identifier: str,
        *,
        status: DeploymentRunStatus,
        duration_ns: int,
        quality: RunQuality | None = None,
        quality_details: Mapping[str, Any] | None = None,
        failure_kind: DeploymentFailureKind | None = None,
        failure_details: Mapping[str, Any] | None = None,
        finished_at: str | None = None,
    ) -> None:
        if status not in self._TERMINAL:
            raise ValueError("finished deployment run must have a terminal status")
        if status == "failed" and failure_kind is None:
            raise ValueError("failed deployment run requires failure_kind")
        if status != "failed" and failure_kind is not None:
            raise ValueError("non-failed deployment run cannot set failure_kind")
        cursor = self.connection.execute(
            """
            UPDATE deployment_run
            SET status = ?, duration_ns = ?, quality = ?,
                quality_details_json = ?, failure_kind = ?,
                failure_details_json = ?, finished_at = ?
            WHERE id = ?
              AND status NOT IN ('completed', 'cancelled', 'failed')
            """,
            (
                status,
                duration_ns,
                quality,
                (
                    canonical_json(dict(quality_details))
                    if quality_details is not None
                    else None
                ),
                failure_kind,
                (
                    canonical_json(dict(failure_details))
                    if failure_details is not None
                    else None
                ),
                finished_at or _utc_now(),
                identifier,
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError("deployment run does not exist or is already terminal")

    def get(self, identifier: str) -> DeploymentRunRecord | None:
        row = self.connection.execute(
            """
            SELECT id, deployment_candidate_id, deployment_placement_id,
                   workload_case_id, status, quality, quality_details_json,
                   failure_kind, failure_details_json, started_at, finished_at,
                   duration_ns, created_at
            FROM deployment_run
            WHERE id = ?
            """,
            (identifier,),
        ).fetchone()
        if row is None:
            return None
        return DeploymentRunRecord(
            id=str(row["id"]),
            deployment_candidate_id=str(row["deployment_candidate_id"]),
            deployment_placement_id=row["deployment_placement_id"],
            workload_case_id=row["workload_case_id"],
            status=row["status"],
            quality=row["quality"],
            quality_details=(
                None
                if row["quality_details_json"] is None
                else _loads_object(str(row["quality_details_json"]))
            ),
            failure_kind=row["failure_kind"],
            failure_details=(
                None
                if row["failure_details_json"] is None
                else _loads_object(str(row["failure_details_json"]))
            ),
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            duration_ns=row["duration_ns"],
            created_at=str(row["created_at"]),
        )

    def members(
        self,
        identifier: str,
    ) -> tuple[DeploymentRunMemberRecord, ...]:
        rows = self.connection.execute(
            """
            SELECT deployment_run_id, deployment_candidate_id, instance_id,
                   server_run_id, client_run_id, result_json
            FROM deployment_run_member
            WHERE deployment_run_id = ?
            ORDER BY instance_id
            """,
            (identifier,),
        ).fetchall()
        return tuple(
            DeploymentRunMemberRecord(
                deployment_run_id=str(row["deployment_run_id"]),
                deployment_candidate_id=str(row["deployment_candidate_id"]),
                instance_id=str(row["instance_id"]),
                server_run_id=row["server_run_id"],
                client_run_id=row["client_run_id"],
                result=_loads_object(str(row["result_json"])),
            )
            for row in rows
        )


class AcceleratorDeviceRepository:
    """Current logical-device inventory for exact host/binary pairs."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def put_inventory(
        self,
        *,
        host_id: str,
        binary_id: str,
        devices: Sequence[AcceleratorDevice],
        raw_output: str,
    ) -> tuple[str, ...]:
        identifiers: list[str] = []
        with transaction(self.connection, immediate=True):
            for device in devices:
                identifier = _content_id(
                    "accel",
                    sha256_json(
                        {
                            "host_id": host_id,
                            "binary_id": binary_id,
                            "logical_device_name": device.logical_device_name,
                        }
                    ),
                )
                self.connection.execute(
                    """
                    INSERT INTO accelerator_device(
                        id, host_id, binary_id, logical_device_name, backend,
                        mapping_status, physical_device_key, pci_bus_id, uuid,
                        vendor, product_name, total_memory_bytes,
                        free_memory_bytes, driver, runtime_metadata_json,
                        raw_output
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(host_id, binary_id, logical_device_name)
                    DO UPDATE SET
                        backend = excluded.backend,
                        mapping_status = excluded.mapping_status,
                        physical_device_key = excluded.physical_device_key,
                        pci_bus_id = excluded.pci_bus_id,
                        uuid = excluded.uuid,
                        vendor = excluded.vendor,
                        product_name = excluded.product_name,
                        total_memory_bytes = excluded.total_memory_bytes,
                        free_memory_bytes = excluded.free_memory_bytes,
                        driver = excluded.driver,
                        runtime_metadata_json = excluded.runtime_metadata_json,
                        raw_output = excluded.raw_output,
                        observed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    """,
                    (
                        identifier,
                        host_id,
                        binary_id,
                        device.logical_device_name,
                        device.backend,
                        device.mapping_status,
                        device.physical_device_key,
                        device.pci_bus_id,
                        device.uuid,
                        device.vendor,
                        device.product_name,
                        device.total_memory_bytes,
                        device.free_memory_bytes,
                        device.driver,
                        canonical_json(dict(device.runtime_metadata)),
                        raw_output,
                    ),
                )
                identifiers.append(identifier)
        return tuple(identifiers)

    def list_for_binary(
        self,
        *,
        host_id: str,
        binary_id: str,
    ) -> tuple[AcceleratorDeviceRecord, ...]:
        rows = self.connection.execute(
            """
            SELECT id, host_id, binary_id, logical_device_name, backend,
                   mapping_status, physical_device_key, pci_bus_id, uuid,
                   vendor, product_name, total_memory_bytes,
                   free_memory_bytes, driver, runtime_metadata_json,
                   raw_output, observed_at
            FROM accelerator_device
            WHERE host_id = ? AND binary_id = ?
            ORDER BY logical_device_name
            """,
            (host_id, binary_id),
        ).fetchall()
        return tuple(
            AcceleratorDeviceRecord(
                id=str(row["id"]),
                host_id=str(row["host_id"]),
                binary_id=str(row["binary_id"]),
                logical_device_name=str(row["logical_device_name"]),
                backend=str(row["backend"]),
                mapping_status=str(row["mapping_status"]),
                physical_device_key=row["physical_device_key"],
                pci_bus_id=row["pci_bus_id"],
                uuid=row["uuid"],
                vendor=row["vendor"],
                product_name=row["product_name"],
                total_memory_bytes=row["total_memory_bytes"],
                free_memory_bytes=row["free_memory_bytes"],
                driver=row["driver"],
                runtime_metadata=_loads_object(
                    str(row["runtime_metadata_json"])
                ),
                raw_output=str(row["raw_output"]),
                observed_at=str(row["observed_at"]),
            )
            for row in rows
        )


class MemoryEstimateRepository:
    """Append-only estimator attempts plus immutable successful cache entries."""

    _TERMINAL = frozenset(
        {
            "completed",
            "failed",
            "parser_failed",
            "timeout",
            "interrupted",
            "cancelled",
            "binary_changed",
        }
    )

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def create_attempt(
        self,
        *,
        identity: MemoryEstimateIdentity,
        candidate_id: str,
        host_id: str,
        helper_binary_id: str,
        model_artifact_id: str,
        argv: tuple[str, ...],
        started_at: str | None = None,
    ) -> str:
        identifier = _event_id("memattempt")
        self.connection.execute(
            """
            INSERT INTO memory_estimate_attempt(
                id, cache_hash, candidate_id, host_id, helper_binary_id,
                model_artifact_id, request_json, argv_json, status, started_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)
            """,
            (
                identifier,
                identity.content_hash(),
                candidate_id,
                host_id,
                helper_binary_id,
                model_artifact_id,
                canonical_json(identity),
                canonical_json(list(argv)),
                started_at or _utc_now(),
            ),
        )
        return identifier

    def finish_attempt(
        self,
        identifier: str,
        *,
        status: MemoryEstimateAttemptStatus,
        duration_ns: int,
        exit_code: int | None,
        stdout: str,
        stderr: str,
        failure_details: Mapping[str, Any] | None = None,
        finished_at: str | None = None,
    ) -> None:
        if status not in self._TERMINAL:
            raise ValueError("finished memory estimate attempt must be terminal")
        cursor = self.connection.execute(
            """
            UPDATE memory_estimate_attempt
            SET status = ?, finished_at = ?, duration_ns = ?, exit_code = ?,
                stdout = ?, stderr = ?, failure_details_json = ?
            WHERE id = ? AND status = 'running'
            """,
            (
                status,
                finished_at or _utc_now(),
                duration_ns,
                exit_code,
                stdout,
                stderr,
                (
                    canonical_json(dict(failure_details))
                    if failure_details is not None
                    else None
                ),
                identifier,
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError(
                "memory estimate attempt does not exist or is already terminal"
            )

    def put_success(
        self,
        *,
        attempt_id: str,
        identity: MemoryEstimateIdentity,
        result: MemoryEstimateOutput,
        candidate_id: str,
        host_id: str,
        helper_binary_id: str,
        model_artifact_id: str,
    ) -> str:
        cache_hash = identity.content_hash()
        attempt = self.connection.execute(
            """
            SELECT cache_hash, candidate_id, host_id, helper_binary_id,
                   model_artifact_id, status
            FROM memory_estimate_attempt
            WHERE id = ?
            """,
            (attempt_id,),
        ).fetchone()
        if attempt is None:
            raise ValueError(f"memory estimate attempt not found: {attempt_id}")
        expected = (
            cache_hash,
            candidate_id,
            host_id,
            helper_binary_id,
            model_artifact_id,
        )
        actual = (
            str(attempt["cache_hash"]),
            str(attempt["candidate_id"]),
            str(attempt["host_id"]),
            str(attempt["helper_binary_id"]),
            str(attempt["model_artifact_id"]),
        )
        if actual != expected:
            raise ValueError("memory estimate attempt identity mismatch")
        if str(attempt["status"]) != "completed":
            raise ValueError(
                "successful memory estimate requires a completed attempt"
            )

        identifier = _content_id("memest", cache_hash)
        rows_by_name = {
            item.logical_device_name: item
            for item in result.devices
        }
        with transaction(self.connection, immediate=True):
            cursor = self.connection.execute(
                """
                INSERT INTO memory_estimate(
                    id, cache_hash, attempt_id, candidate_id, host_id,
                    helper_binary_id, model_artifact_id, identity_json,
                    result_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_hash) DO NOTHING
                """,
                (
                    identifier,
                    cache_hash,
                    attempt_id,
                    candidate_id,
                    host_id,
                    helper_binary_id,
                    model_artifact_id,
                    canonical_json(identity),
                    canonical_json(result),
                ),
            )
            if cursor.rowcount == 1:
                self.connection.executemany(
                    """
                    INSERT INTO memory_estimate_device(
                        memory_estimate_id, ordinal, logical_device_name,
                        model_bytes, context_bytes, compute_bytes, total_bytes,
                        device_total_bytes, device_free_bytes
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            identifier,
                            ordinal,
                            logical_name,
                            rows_by_name[logical_name].model_bytes,
                            rows_by_name[logical_name].context_bytes,
                            rows_by_name[logical_name].compute_bytes,
                            rows_by_name[logical_name].total_bytes,
                            rows_by_name[logical_name].device_total_bytes,
                            rows_by_name[logical_name].device_free_bytes,
                        )
                        for ordinal, logical_name in enumerate(
                            result.resolved.devices
                        )
                    ),
                )
            else:
                existing = self.find_by_cache_hash(cache_hash)
                if existing is None or existing.identity != identity.model_dump(
                    mode="json",
                    by_alias=True,
                ):
                    raise RuntimeError(
                        "memory estimate cache hash collision or identity mismatch"
                    )
        return identifier

    def find_by_cache_hash(
        self,
        cache_hash: str,
    ) -> MemoryEstimateRecord | None:
        row = self.connection.execute(
            """
            SELECT id, cache_hash, attempt_id, candidate_id, host_id,
                   helper_binary_id, model_artifact_id, identity_json,
                   result_json, created_at
            FROM memory_estimate
            WHERE cache_hash = ?
            """,
            (cache_hash,),
        ).fetchone()
        return None if row is None else self._record(row)

    def get(self, identifier: str) -> MemoryEstimateRecord | None:
        row = self.connection.execute(
            """
            SELECT id, cache_hash, attempt_id, candidate_id, host_id,
                   helper_binary_id, model_artifact_id, identity_json,
                   result_json, created_at
            FROM memory_estimate
            WHERE id = ?
            """,
            (identifier,),
        ).fetchone()
        return None if row is None else self._record(row)

    def get_result(self, identifier: str) -> MemoryEstimateOutput | None:
        row = self.connection.execute(
            "SELECT result_json FROM memory_estimate WHERE id = ?",
            (identifier,),
        ).fetchone()
        if row is None:
            return None
        return MemoryEstimateOutput.model_validate_json(str(row["result_json"]))

    def devices(
        self,
        identifier: str,
    ) -> tuple[MemoryEstimateDeviceRecord, ...]:
        rows = self.connection.execute(
            """
            SELECT memory_estimate_id, ordinal, logical_device_name,
                   model_bytes, context_bytes, compute_bytes, total_bytes,
                   device_total_bytes, device_free_bytes
            FROM memory_estimate_device
            WHERE memory_estimate_id = ?
            ORDER BY ordinal
            """,
            (identifier,),
        ).fetchall()
        return tuple(
            MemoryEstimateDeviceRecord(
                memory_estimate_id=str(row["memory_estimate_id"]),
                ordinal=int(row["ordinal"]),
                logical_device_name=str(row["logical_device_name"]),
                model_bytes=int(row["model_bytes"]),
                context_bytes=int(row["context_bytes"]),
                compute_bytes=int(row["compute_bytes"]),
                total_bytes=int(row["total_bytes"]),
                device_total_bytes=int(row["device_total_bytes"]),
                device_free_bytes=int(row["device_free_bytes"]),
            )
            for row in rows
        )

    def attempt(self, identifier: str) -> MemoryEstimateAttemptRecord | None:
        row = self.connection.execute(
            """
            SELECT id, cache_hash, candidate_id, host_id, helper_binary_id,
                   model_artifact_id, request_json, argv_json, status,
                   started_at, finished_at, duration_ns, exit_code,
                   stdout, stderr, failure_details_json
            FROM memory_estimate_attempt
            WHERE id = ?
            """,
            (identifier,),
        ).fetchone()
        if row is None:
            return None
        raw_argv = json.loads(str(row["argv_json"]))
        if not isinstance(raw_argv, list):
            raise ValueError("persisted memory estimate argv must be a list")
        return MemoryEstimateAttemptRecord(
            id=str(row["id"]),
            cache_hash=str(row["cache_hash"]),
            candidate_id=str(row["candidate_id"]),
            host_id=str(row["host_id"]),
            helper_binary_id=str(row["helper_binary_id"]),
            model_artifact_id=str(row["model_artifact_id"]),
            request=_loads_object(str(row["request_json"])),
            argv=tuple(str(item) for item in raw_argv),
            status=row["status"],
            started_at=str(row["started_at"]),
            finished_at=row["finished_at"],
            duration_ns=row["duration_ns"],
            exit_code=row["exit_code"],
            stdout=str(row["stdout"]),
            stderr=str(row["stderr"]),
            failure_details=(
                None
                if row["failure_details_json"] is None
                else _loads_object(str(row["failure_details_json"]))
            ),
        )

    @staticmethod
    def _record(row: sqlite3.Row) -> MemoryEstimateRecord:
        return MemoryEstimateRecord(
            id=str(row["id"]),
            cache_hash=str(row["cache_hash"]),
            attempt_id=str(row["attempt_id"]),
            candidate_id=str(row["candidate_id"]),
            host_id=str(row["host_id"]),
            helper_binary_id=str(row["helper_binary_id"]),
            model_artifact_id=str(row["model_artifact_id"]),
            identity=_loads_object(str(row["identity_json"])),
            result=_loads_object(str(row["result_json"])),
            created_at=str(row["created_at"]),
        )

