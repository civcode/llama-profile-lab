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
        assert migrate(connection) == 3
        assert schema_version(connection) == 3

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
        } <= tables

        assert migrate(connection) == 3
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
