"""Subprocess lifecycle and host-lock tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from llama_profile_lab.execution import HostLock, HostLockError, ProcessRunner


def test_process_runner_captures_stdout_stderr_and_exit_code() -> None:
    result = ProcessRunner().run(
        (
            sys.executable,
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr)",
        )
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "out"
    assert result.stderr.strip() == "err"
    assert result.duration_ns >= 0
    assert not result.timed_out


def test_process_runner_times_out_and_terminates_group() -> None:
    result = ProcessRunner(terminate_grace_seconds=0.1).run(
        (sys.executable, "-c", "import time; time.sleep(10)"),
        timeout_seconds=0.1,
    )

    assert result.timed_out
    assert result.exit_code is not None


def test_host_lock_is_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "bench.lock"
    with HostLock(path):
        with pytest.raises(HostLockError):
            with HostLock(path):
                raise AssertionError("unreachable")

    with HostLock(path):
        assert path.exists()
