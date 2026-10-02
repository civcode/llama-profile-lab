"""M12 archive and complete experiment export acceptance tests."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tarfile
from pathlib import Path

from llama_profile_lab.archive import (
    ArchiveService,
    export_experiment,
    serialize_experiment_export,
)
from tests.analysis_helpers import seed_analysis_experiment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_archive_contains_consistent_database_snapshot_and_hashed_manifest(
    tmp_path: Path,
) -> None:
    database, _ = seed_analysis_experiment(tmp_path / "benchmarks.db")
    artifact = tmp_path / "notes.txt"
    artifact.write_text("acceptance artifact\n", encoding="utf-8")
    output = tmp_path / "experiment.tar.gz"

    manifest = ArchiveService(database).create(output, artifacts=(artifact,))

    assert manifest.format == "llprof-archive-v1"
    assert manifest.schema_version == 5
    assert output.is_file()

    extraction = tmp_path / "archive"
    extraction.mkdir()
    with tarfile.open(output, "r:gz") as archive:
        assert sorted(archive.getnames()) == [
            "artifacts/000-notes.txt",
            "database.sqlite3",
            "manifest.json",
        ]
        archive.extractall(extraction)

    snapshot = extraction / "database.sqlite3"
    payload = json.loads((extraction / "manifest.json").read_text(encoding="utf-8"))
    entries = {item["path"]: item for item in payload["files"]}
    assert entries["database.sqlite3"]["sha256"] == _sha256(snapshot)
    archived_artifact = extraction / "artifacts" / "000-notes.txt"
    assert entries["artifacts/000-notes.txt"]["sha256"] == _sha256(archived_artifact)

    connection = sqlite3.connect(snapshot)
    try:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert (
            connection.execute("SELECT MAX(version) FROM schema_migration").fetchone()[0]
            == 5
        )
        assert connection.execute("SELECT COUNT(*) FROM benchmark_run").fetchone()[0] > 0
    finally:
        connection.close()


def test_full_experiment_export_contains_plan_runs_and_environment(tmp_path: Path) -> None:
    database, experiment_id = seed_analysis_experiment(tmp_path / "export.db")

    payload = export_experiment(database, experiment_id)

    assert payload["format"] == "llprof-experiment-export-v1"
    assert payload["experiment"]["id"] == experiment_id
    assert len(payload["plan"]["experiment_candidates"]) == 11
    assert len(payload["plan"]["benchmark_cases"]) == 44
    assert len(payload["execution"]["benchmark_runs"]) == 44
    assert len(payload["execution"]["benchmark_samples"]) == 132
    assert payload["environment"]["binaries"]
    assert payload["environment"]["hosts"]

    serialized = serialize_experiment_export(database, experiment_id)
    reparsed = json.loads(serialized)
    assert reparsed["experiment"]["id"] == experiment_id
    assert len(reparsed["execution"]["benchmark_runs"]) == 44
