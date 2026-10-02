"""Safe subprocess execution with process-group cleanup."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Event


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Complete observation of one subprocess attempt."""

    argv: tuple[str, ...]
    pid: int
    started_at: str
    finished_at: str
    duration_ns: int
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    cancelled: bool = False
    interrupted: bool = False
    forced_kill: bool = False


class ProcessRunnerError(RuntimeError):
    """Raised when a subprocess cannot be launched."""


class ProcessRunner:
    """Execute one command without a shell and clean up its process group."""

    def __init__(
        self,
        *,
        terminate_grace_seconds: float = 2.0,
        poll_seconds: float = 0.1,
    ) -> None:
        if terminate_grace_seconds < 0:
            raise ValueError("terminate_grace_seconds must be non-negative")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self.terminate_grace_seconds = terminate_grace_seconds
        self.poll_seconds = poll_seconds

    def run(
        self,
        argv: tuple[str, ...],
        *,
        timeout_seconds: float | None = None,
        cancel_event: Event | None = None,
    ) -> ProcessResult:
        """Execute argv and return captured stdout/stderr and lifecycle metadata."""
        if not argv:
            raise ValueError("argv must not be empty")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        started_wall = _utc_now()
        started_ns = time.monotonic_ns()
        try:
            process = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
                start_new_session=True,
            )
        except OSError as exc:
            raise ProcessRunnerError(f"failed to launch {argv[0]}: {exc}") from exc

        timed_out = False
        cancelled = False
        interrupted = False
        forced_kill = False

        try:
            stdout, stderr = self._communicate_until_done(
                process,
                started_ns=started_ns,
                timeout_seconds=timeout_seconds,
                cancel_event=cancel_event,
            )
        except _ProcessTimeout:
            timed_out = True
            stdout, stderr, forced_kill = self._terminate(process)
        except _ProcessCancelled:
            cancelled = True
            stdout, stderr, forced_kill = self._terminate(process)
        except KeyboardInterrupt:
            interrupted = True
            stdout, stderr, forced_kill = self._terminate(process)

        finished_ns = time.monotonic_ns()
        return ProcessResult(
            argv=argv,
            pid=process.pid,
            started_at=started_wall,
            finished_at=_utc_now(),
            duration_ns=max(0, finished_ns - started_ns),
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            cancelled=cancelled,
            interrupted=interrupted,
            forced_kill=forced_kill,
        )

    def _communicate_until_done(
        self,
        process: subprocess.Popen[str],
        *,
        started_ns: int,
        timeout_seconds: float | None,
        cancel_event: Event | None,
    ) -> tuple[str, str]:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise _ProcessCancelled

            if timeout_seconds is None:
                wait_seconds = self.poll_seconds
            else:
                elapsed_seconds = (time.monotonic_ns() - started_ns) / 1_000_000_000
                remaining = timeout_seconds - elapsed_seconds
                if remaining <= 0:
                    raise _ProcessTimeout
                wait_seconds = min(self.poll_seconds, remaining)

            try:
                return process.communicate(timeout=wait_seconds)
            except subprocess.TimeoutExpired:
                continue

    def _terminate(
        self,
        process: subprocess.Popen[str],
    ) -> tuple[str, str, bool]:
        forced_kill = False
        _signal_process_group(process, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=self.terminate_grace_seconds)
        except subprocess.TimeoutExpired:
            forced_kill = True
            _signal_process_group(process, signal.SIGKILL)
            stdout, stderr = process.communicate()
        return stdout, stderr, forced_kill


class _ProcessTimeout(Exception):
    pass


class _ProcessCancelled(Exception):
    pass


def _signal_process_group(process: subprocess.Popen[str], sig: signal.Signals) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        return


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
