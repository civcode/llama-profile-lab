"""Database integrity, query-plan, and growth diagnostics for release hardening."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from llama_profile_lab.db import Database, schema_version


@dataclass(frozen=True, slots=True)
class QueryPlanCheck:
    """One representative SQLite query plan and its index-use result."""

    name: str
    details: tuple[str, ...]
    uses_index: bool


@dataclass(frozen=True, slots=True)
class DatabaseDiagnostics:
    """Operational health and size summary for one llprof database."""

    schema_version: int
    integrity_ok: bool
    foreign_key_violations: int
    page_count: int
    page_size: int
    freelist_count: int
    database_bytes: int
    wal_bytes: int
    row_counts: dict[str, int]
    query_plans: tuple[QueryPlanCheck, ...]


_REPRESENTATIVE_QUERIES = (
    (
        "incomplete benchmark cases",
        """
        SELECT id
        FROM benchmark_case
        WHERE experiment_id = ? AND status != 'completed'
        ORDER BY ordinal
        """,
        ("exp-diagnostic",),
    ),
    (
        "successful run lookup",
        """
        SELECT id
        FROM benchmark_run
        WHERE benchmark_case_id = ? AND status = 'completed'
        ORDER BY started_at DESC
        """,
        ("case-diagnostic",),
    ),
    (
        "server validation history",
        """
        SELECT id
        FROM candidate_evaluation
        WHERE experiment_id = ? AND candidate_id = ?
        ORDER BY created_at, id
        """,
        ("exp-diagnostic", "cand-diagnostic"),
    ),
    (
        "server benchmark history",
        """
        SELECT id
        FROM server_benchmark
        WHERE server_run_id = ?
        ORDER BY created_at, id
        """,
        ("srv-diagnostic",),
    ),
)


def inspect_database(database: Database) -> DatabaseDiagnostics:
    """Inspect integrity, growth, row counts, and critical query plans."""
    database_path = Path(database.path)
    with database.session() as connection:
        integrity_row = connection.execute("PRAGMA integrity_check").fetchone()
        integrity_ok = integrity_row is not None and integrity_row[0] == "ok"
        foreign_key_violations = len(
            connection.execute("PRAGMA foreign_key_check").fetchall()
        )
        page_count = _pragma_int(connection, "page_count")
        page_size = _pragma_int(connection, "page_size")
        freelist_count = _pragma_int(connection, "freelist_count")
        rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
        row_counts = {
            str(row["name"]): int(
                connection.execute(
                    f'SELECT COUNT(*) FROM "{str(row["name"])}"'
                ).fetchone()[0]
            )
            for row in rows
        }
        plans = tuple(
            _query_plan(connection, name, query, params)
            for name, query, params in _REPRESENTATIVE_QUERIES
        )
        version = schema_version(connection)

    wal_path = Path(str(database_path) + "-wal")
    return DatabaseDiagnostics(
        schema_version=version,
        integrity_ok=integrity_ok,
        foreign_key_violations=foreign_key_violations,
        page_count=page_count,
        page_size=page_size,
        freelist_count=freelist_count,
        database_bytes=page_count * page_size,
        wal_bytes=wal_path.stat().st_size if wal_path.is_file() else 0,
        row_counts=row_counts,
        query_plans=plans,
    )


def _pragma_int(connection: sqlite3.Connection, name: str) -> int:
    row = connection.execute(f"PRAGMA {name}").fetchone()
    if row is None:
        raise RuntimeError(f"SQLite did not return PRAGMA {name}")
    return int(row[0])


def _query_plan(
    connection: sqlite3.Connection,
    name: str,
    query: str,
    params: tuple[str, ...],
) -> QueryPlanCheck:
    rows = connection.execute("EXPLAIN QUERY PLAN " + query, params).fetchall()
    details = tuple(str(row[3]) for row in rows)
    uses_index = any(
        "USING INDEX" in detail.upper() or "USING COVERING INDEX" in detail.upper()
        for detail in details
    )
    return QueryPlanCheck(name=name, details=details, uses_index=uses_index)
