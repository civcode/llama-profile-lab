"""M13 database integrity, growth, and query-plan diagnostics."""

from __future__ import annotations

from pathlib import Path

from llama_profile_lab.cli.main import main
from llama_profile_lab.diagnostics import inspect_database
from tests.analysis_helpers import seed_analysis_experiment


def test_reference_database_diagnostics_are_healthy_and_indexed(tmp_path: Path) -> None:
    database, _ = seed_analysis_experiment(tmp_path / "diagnostics.db")

    report = inspect_database(database)

    assert report.schema_version == 7
    assert report.integrity_ok
    assert report.foreign_key_violations == 0
    assert report.database_bytes > 0
    assert report.page_count > 0
    assert report.page_size > 0
    assert report.row_counts["benchmark_case"] == 44
    assert report.row_counts["benchmark_run"] == 44
    assert report.row_counts["benchmark_sample"] == 132
    assert report.query_plans
    assert all(plan.uses_index for plan in report.query_plans)


def test_database_check_cli_reports_growth_and_query_plans(
    tmp_path: Path,
    capsys,
) -> None:
    database, _ = seed_analysis_experiment(tmp_path / "diagnostics-cli.db")

    assert main(["database", "check", "--database", str(database.path)]) == 0

    output = capsys.readouterr().out
    assert "Integrity: ok" in output
    assert "Database bytes:" in output
    assert "benchmark_run: 44" in output
    assert "Representative query plans:" in output
    assert "indexed" in output
