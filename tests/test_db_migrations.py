"""SQLite migration runner tests."""

from pathlib import Path

import pytest

from llama_profile_lab.db import (
    MigrationError,
    connect_database,
    migrate,
    schema_version,
)


def test_initial_migration_creates_schema(tmp_path: Path) -> None:
    connection = connect_database(tmp_path / "benchmarks.db")
    try:
        assert migrate(connection) == 5
        assert schema_version(connection) == 5

        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {
            "schema_migration",
            "candidate",
            "search_space",
            "workload_suite",
            "workload_case",
            "measurement_policy",
            "experiment",
            "benchmark_case",
            "benchmark_run",
            "telemetry_sample",
            "server_run",
            "server_benchmark",
            "candidate_evaluation",
        } <= tables

        assert migrate(connection) == 5
    finally:
        connection.close()


def test_modified_applied_migration_is_rejected(tmp_path: Path) -> None:
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    migration = migrations_dir / "001_test.sql"
    migration.write_text("CREATE TABLE sample (id INTEGER PRIMARY KEY);\n", encoding="utf-8")

    connection = connect_database(tmp_path / "checksum.db")
    try:
        migrate(connection, migrations_dir)
        migration.write_text(
            "CREATE TABLE sample (id INTEGER PRIMARY KEY, name TEXT);\n",
            encoding="utf-8",
        )

        with pytest.raises(MigrationError, match="does not match"):
            migrate(connection, migrations_dir)
    finally:
        connection.close()


def test_non_contiguous_migrations_are_rejected(tmp_path: Path) -> None:
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    (migrations_dir / "001_first.sql").write_text("SELECT 1;\n", encoding="utf-8")
    (migrations_dir / "003_third.sql").write_text("SELECT 3;\n", encoding="utf-8")

    connection = connect_database(tmp_path / "gaps.db")
    try:
        with pytest.raises(MigrationError, match="contiguous"):
            migrate(connection, migrations_dir)
    finally:
        connection.close()


def test_every_historical_schema_prefix_upgrades_to_current(tmp_path: Path) -> None:
    source_dir = Path(__file__).parents[1] / "migrations"
    migration_files = sorted(source_dir.glob("[0-9][0-9][0-9]_*.sql"))
    assert len(migration_files) == 5

    for prefix_length in range(1, len(migration_files)):
        partial_dir = tmp_path / f"migrations-{prefix_length}"
        partial_dir.mkdir()
        for source in migration_files[:prefix_length]:
            (partial_dir / source.name).write_bytes(source.read_bytes())

        connection = connect_database(tmp_path / f"upgrade-{prefix_length}.db")
        try:
            assert migrate(connection, partial_dir) == prefix_length
            assert schema_version(connection) == prefix_length
            assert migrate(connection) == len(migration_files)
            assert schema_version(connection) == len(migration_files)
            assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        finally:
            connection.close()
