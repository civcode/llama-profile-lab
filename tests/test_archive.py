"""M12 archive and complete experiment export acceptance tests."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tarfile
from pathlib import Path

import pytest

from llama_profile_lab.archive import (
    ArchiveError,
    ArchiveService,
    export_experiment,
    serialize_experiment_export,
)
from llama_profile_lab.cli.main import main
from llama_profile_lab.db import Database
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
    assert manifest.schema_version == 11
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
            == 11
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


def test_archive_and_full_export_cli_commands(tmp_path: Path) -> None:
    database, experiment_id = seed_analysis_experiment(tmp_path / "cli.db")
    database_path = Path(database.path)
    export_path = tmp_path / "experiment.json"
    archive_path = tmp_path / "experiment.tar.gz"

    assert (
        main(
            [
                "experiment",
                "export",
                experiment_id,
                "--output",
                str(export_path),
                "--database",
                str(database_path),
            ]
        )
        == 0
    )
    assert json.loads(export_path.read_text(encoding="utf-8"))["experiment"]["id"] == experiment_id

    assert (
        main(
            [
                "archive",
                "--output",
                str(archive_path),
                "--database",
                str(database_path),
            ]
        )
        == 0
    )
    assert archive_path.is_file()

    restored_path = tmp_path / "restored-cli.db"
    assert (
        main(
            [
                "archive",
                "--restore",
                str(archive_path),
                "--database",
                str(restored_path),
            ]
        )
        == 0
    )
    assert json.loads(
        serialize_experiment_export(Database(restored_path), experiment_id)
    )["experiment"]["id"] == experiment_id


def test_archive_restore_verifies_snapshot_and_artifacts(tmp_path: Path) -> None:
    database, experiment_id = seed_analysis_experiment(tmp_path / "source.db")
    artifact = tmp_path / "trace.txt"
    artifact.write_text("trace payload\n", encoding="utf-8")
    archive_path = tmp_path / "backup.tar.gz"
    ArchiveService(database).create(archive_path, artifacts=(artifact,))

    restored_path = tmp_path / "restored.db"
    restored_artifacts = tmp_path / "restored-artifacts"
    manifest = ArchiveService.restore(
        archive_path,
        restored_path,
        artifacts_dir=restored_artifacts,
    )

    assert manifest.format == "llprof-archive-v1"
    assert restored_path.is_file()
    assert (restored_artifacts / "000-trace.txt").read_text(encoding="utf-8") == (
        "trace payload\n"
    )
    restored = export_experiment(Database(restored_path), experiment_id)
    assert restored["experiment"]["id"] == experiment_id
    assert len(restored["execution"]["benchmark_runs"]) == 44


def test_archive_restore_rejects_tampered_payload(tmp_path: Path) -> None:
    database, _ = seed_analysis_experiment(tmp_path / "source.db")
    archive_path = tmp_path / "backup.tar.gz"
    ArchiveService(database).create(archive_path)

    unpacked = tmp_path / "unpacked"
    unpacked.mkdir()
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(unpacked)
    with (unpacked / "database.sqlite3").open("ab") as handle:
        handle.write(b"tampered")

    tampered = tmp_path / "tampered.tar.gz"
    with tarfile.open(tampered, "w:gz") as archive:
        archive.add(unpacked / "database.sqlite3", arcname="database.sqlite3")
        archive.add(unpacked / "manifest.json", arcname="manifest.json")

    with pytest.raises(ArchiveError, match="size mismatch|hash mismatch"):
        ArchiveService.restore(tampered, tmp_path / "should-not-exist.db")
