"""CLI coverage for llama.cpp binary inspection and comparison."""

from pathlib import Path

import pytest

from llama_profile_lab.cli.main import main
from llama_profile_lab.db import Database, EnvironmentRepository
from tests.test_llama_discovery import write_fake_binary


def test_binary_inspect_registers_and_reports_capabilities(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    binary = tmp_path / "llama-server"
    write_fake_binary(
        binary,
        version="version: 9000 (abcdef1)",
        help_text="  --batch-size N  batch\n  --spec-type T  spec\n",
    )
    database_path = tmp_path / "benchmarks.db"

    result = main(
        [
            "binary",
            "inspect",
            str(binary),
            "--database",
            str(database_path),
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "Kind: llama-server" in output
    assert "--batch-size" in output
    assert "--spec-type" in output

    with Database(database_path).session() as connection:
        records = EnvironmentRepository(connection).list_binaries()
        assert len(records) == 1


def test_binary_compare_reports_capability_difference(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = tmp_path / "a" / "llama-server"
    second = tmp_path / "b" / "llama-server"
    first.parent.mkdir()
    second.parent.mkdir()
    write_fake_binary(first, version="version: 1 (aaaaaaa)", help_text="  --batch-size N")
    write_fake_binary(
        second,
        version="version: 2 (bbbbbbb)",
        help_text="  --batch-size N\n  --spec-type T",
    )
    database_path = tmp_path / "benchmarks.db"

    assert main(["binary", "inspect", str(first), "--database", str(database_path)]) == 0
    first_id = Database(database_path).connect().execute(
        "SELECT id FROM binary ORDER BY created_at LIMIT 1"
    ).fetchone()[0]
    assert main(["binary", "inspect", str(second), "--database", str(database_path)]) == 0

    with Database(database_path).session() as connection:
        records = EnvironmentRepository(connection).list_binaries()
        second_id = next(record.id for record in records if record.id != first_id)

    capsys.readouterr()
    result = main(
        [
            "binary",
            "compare",
            first_id,
            second_id,
            "--database",
            str(database_path),
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "Only right:" in output
    assert "--spec-type" in output
