"""Numbered SQLite migration discovery and application."""

from __future__ import annotations

import hashlib
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

_MIGRATION_PATTERN = re.compile(r"^(?P<version>[0-9]{3,})_(?P<name>[a-z0-9_]+)\.sql$")


class MigrationError(RuntimeError):
    """Raised when migration history is invalid or cannot be applied."""


@dataclass(frozen=True, slots=True)
class Migration:
    """One immutable SQL migration."""

    version: int
    name: str
    path: Path
    checksum: str
    sql: str


def default_migrations_dir() -> Path:
    """Locate repository migrations, with an installed-layout fallback."""
    source_tree = Path(__file__).resolve().parents[3] / "migrations"
    if source_tree.is_dir():
        return source_tree

    installed = Path(sys.prefix) / "share" / "llama-profile-lab" / "migrations"
    if installed.is_dir():
        return installed

    raise MigrationError(
        "cannot locate migrations directory; pass migrations_dir explicitly"
    )


def discover_migrations(directory: Path) -> tuple[Migration, ...]:
    """Load and validate ordered migration files."""
    if not directory.is_dir():
        raise MigrationError(f"migration directory does not exist: {directory}")

    migrations: list[Migration] = []
    for path in sorted(directory.iterdir()):
        match = _MIGRATION_PATTERN.match(path.name)
        if match is None:
            continue
        sql = path.read_text(encoding="utf-8")
        checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
        migrations.append(
            Migration(
                version=int(match.group("version")),
                name=match.group("name"),
                path=path,
                checksum=checksum,
                sql=sql,
            )
        )

    if not migrations:
        raise MigrationError(f"no migrations found in {directory}")

    versions = [migration.version for migration in migrations]
    if len(versions) != len(set(versions)):
        raise MigrationError("duplicate migration version")

    expected = list(range(1, max(versions) + 1))
    if versions != expected:
        raise MigrationError(
            f"migration versions must be contiguous from 1; found {versions}"
        )

    return tuple(migrations)


def _ensure_history_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migration (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            checksum TEXT NOT NULL,
            applied_at TEXT NOT NULL
                DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )
        """
    )


def _applied_migrations(connection: sqlite3.Connection) -> dict[int, tuple[str, str]]:
    rows = connection.execute(
        "SELECT version, name, checksum FROM schema_migration ORDER BY version"
    ).fetchall()
    return {
        int(row["version"]): (str(row["name"]), str(row["checksum"]))
        for row in rows
    }


def migrate(
    connection: sqlite3.Connection,
    migrations_dir: Path | None = None,
) -> int:
    """Apply pending migrations and return the resulting schema version."""
    directory = migrations_dir or default_migrations_dir()
    migrations = discover_migrations(directory)
    _ensure_history_table(connection)
    applied = _applied_migrations(connection)

    for migration in migrations:
        recorded = applied.get(migration.version)
        if recorded is not None:
            recorded_name, recorded_checksum = recorded
            if recorded_name != migration.name or recorded_checksum != migration.checksum:
                raise MigrationError(
                    f"applied migration {migration.version:03d}_{recorded_name} "
                    "does not match the migration file on disk"
                )
            continue

        script = (
            "BEGIN IMMEDIATE;\n"
            + migration.sql
            + "\nINSERT INTO schema_migration(version, name, checksum) VALUES ("
            + str(migration.version)
            + ", "
            + repr(migration.name)
            + ", "
            + repr(migration.checksum)
            + ");\nCOMMIT;"
        )
        try:
            connection.executescript(script)
        except sqlite3.Error as exc:
            if connection.in_transaction:
                connection.rollback()
            raise MigrationError(
                f"failed to apply migration {migration.path.name}: {exc}"
            ) from exc

    return schema_version(connection)


def schema_version(connection: sqlite3.Connection) -> int:
    """Return the highest applied migration version, or zero."""
    _ensure_history_table(connection)
    row = connection.execute(
        "SELECT COALESCE(MAX(version), 0) AS version FROM schema_migration"
    ).fetchone()
    if row is None:
        return 0
    return int(row["version"])
