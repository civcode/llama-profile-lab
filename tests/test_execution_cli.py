"""CLI coverage for experiment run/resume and run inspection."""

from pathlib import Path

import pytest

from llama_profile_lab.cli.main import main
from llama_profile_lab.db import Database
from llama_profile_lab.planning import plan_experiment
from tests.test_execution_engine import register_fake_bench, write_fake_llama_bench
from tests.test_planning_persistence import seed_reference_experiment


def test_run_limit_resume_and_show_commands(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = tmp_path / "benchmarks.db"
    database = Database(database_path)
    experiment_id = seed_reference_experiment(database)
    with database.session() as connection:
        plan_experiment(connection, experiment_id)

    binary = tmp_path / "llama-bench"
    write_fake_llama_bench(binary)
    binary_id = register_fake_bench(database, binary)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")

    first = main(
        [
            "experiment",
            "run",
            experiment_id,
            "--binary",
            binary_id,
            "--model-path",
            str(model),
            "--limit",
            "1",
            "--database",
            str(database_path),
        ]
    )
    assert first == 0
    assert "Remaining: 43" in capsys.readouterr().out

    second = main(
        [
            "experiment",
            "resume",
            experiment_id,
            "--binary",
            binary_id,
            "--model-path",
            str(model),
            "--database",
            str(database_path),
        ]
    )
    assert second == 0
    assert "Remaining: 0" in capsys.readouterr().out

    with database.session() as connection:
        run_id = connection.execute(
            "SELECT id FROM benchmark_run ORDER BY started_at LIMIT 1"
        ).fetchone()[0]

    shown = main(
        [
            "run",
            "show",
            str(run_id),
            "--database",
            str(database_path),
        ]
    )
    assert shown == 0
    output = capsys.readouterr().out
    assert "Status: completed" in output
    assert "Samples: 3" in output
    assert "avg_ts" in output
