"""Tests for the bootstrap CLI."""

from pathlib import Path

import pytest

from llama_profile_lab.cli.main import main


def test_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "llprof" in output
    assert "Experiment, benchmark, and tune llama.cpp profiles." in output
    assert "server" in output
    assert "api" in output
    assert "ui" in output


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == "llprof 0.1.0"


def test_ui_requires_built_frontend(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing-ui"
    result = main(["ui", "--frontend-dir", str(missing)])
    assert result == 2
    assert "built frontend not found" in capsys.readouterr().err
