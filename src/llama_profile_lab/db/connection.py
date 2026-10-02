"""SQLite connection and transaction utilities."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

DatabasePath = str | Path


def connect_database(path: DatabasePath) -> sqlite3.Connection:
    """Open a configured SQLite connection for llama-profile-lab."""
    if isinstance(path, Path):
        path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        str(path),
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


@contextmanager
def transaction(
    connection: sqlite3.Connection,
    *,
    immediate: bool = False,
) -> Iterator[sqlite3.Connection]:
    """Run a transaction, using a savepoint when already inside one."""
    if connection.in_transaction:
        savepoint = f"sp_{uuid4().hex}"
        connection.execute(f"SAVEPOINT {savepoint}")
        try:
            yield connection
        except BaseException:
            connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            raise
        else:
            connection.execute(f"RELEASE SAVEPOINT {savepoint}")
        return

    connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
    try:
        yield connection
    except BaseException:
        connection.rollback()
        raise
    else:
        connection.commit()
