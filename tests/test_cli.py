"""Tests for the bootstrap CLI."""

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


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == "llprof 0.1.0"
