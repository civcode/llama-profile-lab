"""CLI coverage for persisted experiment planning."""

from pathlib import Path

import pytest
from tests.test_planning_persistence import seed_reference_experiment

from llama_profile_lab.cli.main import main
from llama_profile_lab.db import Database


def test_experiment_plan_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = tmp_path / "benchmarks.db"
    database = Database(database_path)
    experiment_id = seed_reference_experiment(database)

    result = main(
        [
            "experiment",
            "plan",
            experiment_id,
            "--database",
            str(database_path),
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "Candidates: 11" in output
    assert "Benchmark cases: 44" in output
    assert "Rejected by constraints: 1" in output
    assert "Status: planned" in output
