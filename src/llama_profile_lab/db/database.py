"""Database facade for configured connections and repositories."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from llama_profile_lab.db.connection import DatabasePath, connect_database
from llama_profile_lab.db.migrations import migrate


class Database:
    """A SQLite database with migrations applied on each opened session."""

    def __init__(
        self,
        path: DatabasePath,
        *,
        migrations_dir: Path | None = None,
    ) -> None:
        self.path = path
        self.migrations_dir = migrations_dir

    def connect(self) -> sqlite3.Connection:
        """Open a configured, migrated connection."""
        connection = connect_database(self.path)
        try:
            migrate(connection, self.migrations_dir)
        except BaseException:
            connection.close()
            raise
        return connection

    @contextmanager
    def session(self) -> Iterator[sqlite3.Connection]:
        """Open and close one configured database connection."""
        connection = self.connect()
        try:
            yield connection
        finally:
            connection.close()
