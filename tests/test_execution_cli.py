"""CLI coverage for experiment run/resume and run inspection."""

from pathlib import Path

import pytest

from llama_profile_lab.cli.main import main
from llama_profile_lab.db import Database
from llama_profile_lab.planning import plan_experiment
from tests.test_execution_engine import (
    register_fake_binary,
    write_fake_fit_params,
    write_fake_llama_bench,
)
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
    fit = tmp_path / "llama-fit-params"
    write_fake_llama_bench(binary)
    write_fake_fit_params(fit)
    binary_id = register_fake_binary(database, binary)
    fit_binary_id = register_fake_binary(database, fit)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")

    first = main(
        [
            "experiment",
            "run",
            experiment_id,
            "--binary",
            binary_id,
            "--fit-binary",
            fit_binary_id,
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
            "--fit-binary",
            fit_binary_id,
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



def test_placement_list_and_show_commands(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = tmp_path / "placements.db"
    database = Database(database_path)
    experiment_id = seed_reference_experiment(database)
    with database.session() as connection:
        plan_experiment(connection, experiment_id)

    bench = tmp_path / "llama-bench"
    fit = tmp_path / "llama-fit-params"
    write_fake_llama_bench(bench)
    write_fake_fit_params(fit)
    bench_id = register_fake_binary(database, bench)
    fit_id = register_fake_binary(database, fit)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")

    assert main(
        [
            "experiment",
            "run",
            experiment_id,
            "--binary",
            bench_id,
            "--fit-binary",
            fit_id,
            "--model-path",
            str(model),
            "--limit",
            "1",
            "--database",
            str(database_path),
        ]
    ) == 0
    capsys.readouterr()

    assert main(["placement", "list", "--database", str(database_path)]) == 0
    listing = capsys.readouterr().out
    assert "ctx=131072" in listing
    assert "ngl=42" in listing

    with database.session() as connection:
        placement_id = str(
            connection.execute("SELECT id FROM resolved_placement LIMIT 1").fetchone()[0]
        )

    assert main(
        [
            "placement",
            "show",
            placement_id,
            "--database",
            str(database_path),
        ]
    ) == 0
    shown = capsys.readouterr().out
    assert "Production context: 131072" in shown
    assert "GPU layers: 42" in shown
