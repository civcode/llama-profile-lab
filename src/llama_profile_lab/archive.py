"""Consistent archive snapshots and full experiment provenance exports."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from llama_profile_lab.db import Database, schema_version


class ArchiveError(RuntimeError):
    """Raised when an archive or experiment export cannot be created."""


@dataclass(frozen=True, slots=True)
class ArchiveFile:
    """One hashed file included in an archive."""

    path: str
    sha256: str
    size_bytes: int
    kind: str


@dataclass(frozen=True, slots=True)
class ArchiveManifest:
    """Manifest embedded in each llprof archive."""

    format: str
    created_at: str
    schema_version: int
    database_source: str
    files: tuple[ArchiveFile, ...]


class ArchiveService:
    """Create consistent SQLite backup archives with hashed manifests."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def create(
        self,
        output_path: Path,
        *,
        artifacts: Iterable[Path] = (),
    ) -> ArchiveManifest:
        output = output_path.expanduser()
        if output.exists():
            raise ArchiveError(f"archive output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="llprof-archive-") as tempdir:
            root = Path(tempdir)
            snapshot_path = root / "database.sqlite3"
            with self.database.session() as source:
                source.execute("PRAGMA wal_checkpoint(FULL)")
                version = schema_version(source)
                destination = sqlite3.connect(snapshot_path)
                try:
                    source.backup(destination)
                    destination.commit()
                finally:
                    destination.close()

            entries: list[ArchiveFile] = [
                _archive_file(snapshot_path, "database.sqlite3", "database")
            ]
            artifacts_dir = root / "artifacts"
            for index, artifact in enumerate(artifacts):
                source_path = artifact.expanduser()
                if not source_path.is_file():
                    raise ArchiveError(f"artifact does not exist: {source_path}")
                artifacts_dir.mkdir(exist_ok=True)
                archived_name = f"{index:03d}-{source_path.name}"
                copied = artifacts_dir / archived_name
                shutil.copy2(source_path, copied)
                entries.append(
                    _archive_file(
                        copied,
                        f"artifacts/{archived_name}",
                        "artifact",
                    )
                )

            manifest = ArchiveManifest(
                format="llprof-archive-v1",
                created_at=_utc_now(),
                schema_version=version,
                database_source=str(self.database.path),
                files=tuple(entries),
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        **asdict(manifest),
                        "files": [asdict(item) for item in manifest.files],
                    },
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            try:
                with tarfile.open(output, mode="w:gz") as archive:
                    archive.add(snapshot_path, arcname="database.sqlite3")
                    archive.add(manifest_path, arcname="manifest.json")
                    if artifacts_dir.is_dir():
                        for path in sorted(artifacts_dir.iterdir()):
                            archive.add(path, arcname=f"artifacts/{path.name}")
            except OSError as exc:
                output.unlink(missing_ok=True)
                raise ArchiveError(f"cannot create archive {output}: {exc}") from exc

        return manifest


def export_experiment(database: Database, experiment_id: str) -> dict[str, Any]:
    """Export all persisted rows needed to reconstruct one experiment's provenance."""
    with database.session() as connection:
        experiment = connection.execute(
            "SELECT * FROM experiment WHERE id = ?",
            (experiment_id,),
        ).fetchone()
        if experiment is None:
            raise ArchiveError(f"experiment not found: {experiment_id}")

        experiment_candidates = _rows(
            connection,
            "SELECT * FROM experiment_candidate WHERE experiment_id = ? ORDER BY ordinal",
            (experiment_id,),
        )
        experiment_workloads = _rows(
            connection,
            """
            SELECT *
            FROM experiment_workload
            WHERE experiment_id = ?
            ORDER BY candidate_id, suite_case_index
            """,
            (experiment_id,),
        )
        cases = _rows(
            connection,
            "SELECT * FROM benchmark_case WHERE experiment_id = ? ORDER BY ordinal",
            (experiment_id,),
        )
        evaluations = _rows(
            connection,
            """
            SELECT *
            FROM candidate_evaluation
            WHERE experiment_id = ?
            ORDER BY created_at, id
            """,
            (experiment_id,),
        )
        server_runs = _rows(
            connection,
            """
            SELECT *
            FROM server_run
            WHERE experiment_id = ?
            ORDER BY started_at, id
            """,
            (experiment_id,),
        )

        candidate_ids = {
            str(row["candidate_id"]) for row in experiment_candidates
        }
        candidate_ids.add(str(experiment["base_candidate_id"]))
        workload_ids = {
            str(row["workload_case_id"]) for row in experiment_workloads
        }
        case_ids = [str(row["id"]) for row in cases]
        server_run_ids = [str(row["id"]) for row in server_runs]

        runs = _rows_in(
            connection,
            "SELECT * FROM benchmark_run WHERE benchmark_case_id IN ({}) ORDER BY started_at, id",
            case_ids,
        )
        run_ids = [str(row["id"]) for row in runs]
        samples = _rows_in(
            connection,
            "SELECT * FROM benchmark_sample WHERE run_id IN ({}) ORDER BY run_id, sample_index",
            run_ids,
        )
        telemetry = _rows_in(
            connection,
            "SELECT * FROM telemetry_sample WHERE run_id IN ({}) ORDER BY run_id, timestamp_ns",
            run_ids,
        )
        metrics = _rows_in(
            connection,
            "SELECT * FROM metric WHERE run_id IN ({}) ORDER BY run_id, metric_name",
            run_ids,
        )
        server_benchmarks = _rows_in(
            connection,
            "SELECT * FROM server_benchmark WHERE server_run_id IN ({}) ORDER BY created_at, id",
            server_run_ids,
        )

        placement_ids = {
            str(row["placement_id"])
            for row in cases
            if row["placement_id"] is not None
        }
        placement_ids.update(
            str(row["placement_id"])
            for row in server_runs
            if row["placement_id"] is not None
        )
        placements = _rows_in(
            connection,
            "SELECT * FROM resolved_placement WHERE id IN ({}) ORDER BY created_at, id",
            sorted(placement_ids),
        )
        fit_attempt_ids = [
            str(row["fit_attempt_id"])
            for row in placements
            if row["fit_attempt_id"] is not None
        ]
        placement_attempts = _rows_in(
            connection,
            "SELECT * FROM placement_attempt WHERE id IN ({}) ORDER BY started_at, id",
            fit_attempt_ids,
        )

        binary_ids = {
            str(row["binary_id"]) for row in runs if row["binary_id"] is not None
        }
        binary_ids.update(
            str(row["binary_id"]) for row in placements if row["binary_id"] is not None
        )
        binary_ids.update(
            str(row["binary_id"])
            for row in placement_attempts
            if row["binary_id"] is not None
        )
        binary_ids.update(
            str(row["server_binary_id"])
            for row in server_runs
            if row["server_binary_id"] is not None
        )
        binary_ids.update(
            str(row["speed_bench_binary_id"])
            for row in server_benchmarks
            if row["speed_bench_binary_id"] is not None
        )

        host_ids = {
            str(row["host_id"]) for row in runs if row["host_id"] is not None
        }
        host_ids.update(
            str(row["host_id"]) for row in placements if row["host_id"] is not None
        )
        host_ids.update(
            str(row["host_id"])
            for row in placement_attempts
            if row["host_id"] is not None
        )
        host_ids.update(
            str(row["host_id"])
            for row in server_runs
            if row["host_id"] is not None
        )

        payload = {
            "format": "llprof-experiment-export-v1",
            "exported_at": _utc_now(),
            "schema_version": schema_version(connection),
            "experiment": dict(experiment),
            "immutable": {
                "base_candidate": _one(
                    connection,
                    "SELECT * FROM candidate WHERE id = ?",
                    (experiment["base_candidate_id"],),
                ),
                "search_space": _one(
                    connection,
                    "SELECT * FROM search_space WHERE id = ?",
                    (experiment["search_space_id"],),
                ),
                "workload_suite": _one(
                    connection,
                    "SELECT * FROM workload_suite WHERE id = ?",
                    (experiment["workload_suite_id"],),
                ),
                "measurement_policy": _one(
                    connection,
                    "SELECT * FROM measurement_policy WHERE id = ?",
                    (experiment["measurement_policy_id"],),
                ),
                "candidates": _rows_in(
                    connection,
                    "SELECT * FROM candidate WHERE id IN ({}) ORDER BY id",
                    sorted(candidate_ids),
                ),
                "workload_cases": _rows_in(
                    connection,
                    "SELECT * FROM workload_case WHERE id IN ({}) ORDER BY id",
                    sorted(workload_ids),
                ),
            },
            "plan": {
                "experiment_candidates": experiment_candidates,
                "experiment_workloads": experiment_workloads,
                "benchmark_cases": cases,
            },
            "execution": {
                "benchmark_runs": runs,
                "benchmark_samples": samples,
                "telemetry_samples": telemetry,
                "metrics": metrics,
                "placements": placements,
                "placement_attempts": placement_attempts,
            },
            "validation": {
                "server_runs": server_runs,
                "server_benchmarks": server_benchmarks,
                "candidate_evaluations": evaluations,
            },
            "environment": {
                "binaries": _rows_in(
                    connection,
                    "SELECT * FROM binary WHERE id IN ({}) ORDER BY kind, id",
                    sorted(binary_ids),
                ),
                "hosts": _rows_in(
                    connection,
                    "SELECT * FROM host WHERE id IN ({}) ORDER BY id",
                    sorted(host_ids),
                ),
            },
        }
    return payload


def serialize_experiment_export(
    database: Database,
    experiment_id: str,
) -> str:
    """Render one full provenance export as deterministic JSON."""
    return json.dumps(
        export_experiment(database, experiment_id),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"


def _rows(
    connection: sqlite3.Connection,
    query: str,
    params: tuple[Any, ...] = (),
) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(query, params).fetchall()]


def _one(
    connection: sqlite3.Connection,
    query: str,
    params: tuple[Any, ...],
) -> dict[str, Any] | None:
    row = connection.execute(query, params).fetchone()
    return None if row is None else dict(row)


def _rows_in(
    connection: sqlite3.Connection,
    query_template: str,
    identifiers: list[str],
) -> list[dict[str, Any]]:
    if not identifiers:
        return []
    placeholders = ",".join("?" for _ in identifiers)
    return _rows(
        connection,
        query_template.format(placeholders),
        tuple(identifiers),
    )


def _archive_file(path: Path, archived_path: str, kind: str) -> ArchiveFile:
    return ArchiveFile(
        path=archived_path,
        sha256=_sha256(path),
        size_bytes=path.stat().st_size,
        kind=kind,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
