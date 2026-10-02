"""SQLite connection and transaction tests."""

import sqlite3
from pathlib import Path

import pytest

from llama_profile_lab.db import connect_database, transaction


def test_connection_configures_required_pragmas(tmp_path: Path) -> None:
    connection = connect_database(tmp_path / "benchmarks.db")
    try:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 1
    finally:
        connection.close()


def test_transaction_rolls_back_on_error(tmp_path: Path) -> None:
    connection = connect_database(tmp_path / "transactions.db")
    try:
        connection.execute("CREATE TABLE item (value TEXT NOT NULL)")
        with pytest.raises(RuntimeError):
            with transaction(connection):
                connection.execute("INSERT INTO item(value) VALUES ('before-error')")
                raise RuntimeError("stop")

        count = connection.execute("SELECT COUNT(*) FROM item").fetchone()[0]
        assert count == 0
    finally:
        connection.close()


def test_nested_transaction_uses_savepoint(tmp_path: Path) -> None:
    connection = connect_database(tmp_path / "nested.db")
    try:
        connection.execute("CREATE TABLE item (value TEXT NOT NULL)")
        with transaction(connection):
            connection.execute("INSERT INTO item(value) VALUES ('outer')")
            with pytest.raises(sqlite3.IntegrityError):
                with transaction(connection):
                    connection.execute("CREATE TABLE unique_item (value TEXT UNIQUE)")
                    connection.execute("INSERT INTO unique_item VALUES ('x')")
                    connection.execute("INSERT INTO unique_item VALUES ('x')")
            connection.execute("INSERT INTO item(value) VALUES ('after')")

        values = [
            row[0]
            for row in connection.execute(
                "SELECT value FROM item ORDER BY rowid"
            ).fetchall()
        ]
        assert values == ["outer", "after"]
    finally:
        connection.close()
