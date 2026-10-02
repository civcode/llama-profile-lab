"""CLI coverage for M8 analysis commands."""

from __future__ import annotations

from pathlib import Path

import pytest

from llama_profile_lab.cli.main import main
from tests.analysis_helpers import candidate_id_for, seed_analysis_experiment


def test_results_matrix_and_compare_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = tmp_path / "analysis-cli.db"
    database, experiment_id = seed_analysis_experiment(database_path)

    assert main(
        [
            "results",
            "matrix",
            experiment_id,
            "--x",
            "compute.batch_size",
            "--y",
            "compute.ubatch_size",
            "--metric",
            "throughput.median",
            "--filter",
            "workload.kind=microbench-prefill",
            "--filter",
            "workload.prompt_tokens=8192",
            "--database",
            str(database_path),
        ]
    ) == 0
    matrix_output = capsys.readouterr().out
    assert "Metric: throughput.median" in matrix_output
    assert "8192" in matrix_output

    candidate_id = candidate_id_for(
        database,
        experiment_id,
        batch=8192,
        ubatch=4096,
    )
    assert main(
        [
            "results",
            "compare",
            experiment_id,
            candidate_id,
            "--metric",
            "throughput.median",
            "--filter",
            "workload.kind=microbench-prefill",
            "--filter",
            "workload.prompt_tokens=8192",
            "--database",
            str(database_path),
        ]
    ) == 0
    comparison_output = capsys.readouterr().out
    assert f"Candidate: {candidate_id}" in comparison_output
    assert "throughput.median" in comparison_output


def test_results_pareto_and_latency_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = tmp_path / "analysis-cli-2.db"
    database, experiment_id = seed_analysis_experiment(database_path)

    assert main(
        [
            "results",
            "pareto",
            experiment_id,
            "--objective",
            "pp:max:throughput.median@workload.kind=microbench-prefill;"
            "workload.prompt_tokens=8192",
            "--objective",
            "tg:max:throughput.median@workload.kind=microbench-decode;"
            "workload.depth_tokens=4096",
            "--database",
            str(database_path),
        ]
    ) == 0
    pareto_output = capsys.readouterr().out
    assert "Pareto frontier:" in pareto_output

    candidate_id = candidate_id_for(
        database,
        experiment_id,
        batch=4096,
        ubatch=512,
    )
    assert main(
        [
            "results",
            "latency",
            experiment_id,
            "--candidate",
            candidate_id,
            "--prompt-tokens",
            "4096",
            "--generate-tokens",
            "64",
            "--decode-start-depth-tokens",
            "8192",
            "--database",
            str(database_path),
        ]
    ) == 0
    latency_output = capsys.readouterr().out
    assert "Total:" in latency_output
    assert "Average TG" in latency_output
