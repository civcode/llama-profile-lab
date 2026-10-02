"""Managed long-lived llama-server subprocess lifecycle."""

from __future__ import annotations

import http.client
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Event
from typing import Literal


ServerProcessFailureKind = Literal[
    "start_failed",
    "readiness_failed",
    "interrupted",
    "cancelled",
]


class ServerProcessError(RuntimeError):
    """Raised when a managed server cannot start or become ready."""

    def __init__(self, message: str, *, kind: ServerProcessFailureKind) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True, slots=True)
class ServerProcessOutcome:
    """Captured lifecycle result after a managed server is stopped."""

    argv: tuple[str, ...]
    pid: int | None
    started_at: str
    finished_at: str
    duration_ns: int
    exit_code: int | None
    stdout: str
    stderr: str
    forced_kill: bool


class ManagedServerProcess:
    """Start llama-server, poll /health, and terminate its process group cleanly."""

    def __init__(
        self,
        argv: tuple[str, ...],
        *,
        terminate_grace_seconds: float = 5.0,
        poll_seconds: float = 0.2,
    ) -> None:
        if not argv:
            raise ValueError("argv must not be empty")
        if terminate_grace_seconds < 0:
            raise ValueError("terminate_grace_seconds must be non-negative")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self.argv = argv
        self.terminate_grace_seconds = terminate_grace_seconds
        self.poll_seconds = poll_seconds
        self._stdout = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        self._stderr = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        self._process: subprocess.Popen[str] | None = None
        self._started_at = _utc_now()
        self._started_ns: int | None = None
        self._forced_kill = False

    @property
    def pid(self) -> int | None:
        return None if self._process is None else self._process.pid

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if self._process is not None:
            raise ServerProcessError("server process has already been started", kind="start_failed")
        self._started_at = _utc_now()
        self._started_ns = time.monotonic_ns()
        try:
            self._process = subprocess.Popen(
                self.argv,
                stdout=self._stdout,
                stderr=self._stderr,
                text=True,
                shell=False,
                start_new_session=True,
            )
        except OSError as exc:
            raise ServerProcessError(
                f"failed to launch {self.argv[0]}: {exc}",
                kind="start_failed",
            ) from exc

    def wait_ready(
        self,
        health_url: str,
        *,
        timeout_seconds: float,
        cancel_event: Event | None = None,
    ) -> str:
        if timeout_seconds <= 0:
            raise ValueError("readiness timeout must be positive")
        process = self._require_process()
        deadline = time.monotonic() + timeout_seconds
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise ServerProcessError(
                        "server readiness cancelled",
                        kind="cancelled",
                    )
                exit_code = process.poll()
                if exit_code is not None:
                    raise ServerProcessError(
                        f"server exited before readiness with code {exit_code}",
                        kind="start_failed",
                    )
                if _health_ready(health_url):
                    return _utc_now()
                if time.monotonic() >= deadline:
                    raise ServerProcessError(
                        f"server did not become ready within {timeout_seconds:g} seconds",
                        kind="readiness_failed",
                    )
                time.sleep(self.poll_seconds)
        except KeyboardInterrupt as exc:
            raise ServerProcessError(
                "server readiness interrupted",
                kind="interrupted",
            ) from exc

    def stop(self) -> ServerProcessOutcome:
        process = self._process
        if process is not None and process.poll() is None:
            _signal_process_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=self.terminate_grace_seconds)
            except subprocess.TimeoutExpired:
                self._forced_kill = True
                _signal_process_group(process, signal.SIGKILL)
                process.wait()

        finished_ns = time.monotonic_ns()
        started_ns = self._started_ns or finished_ns
        self._stdout.flush()
        self._stderr.flush()
        self._stdout.seek(0)
        self._stderr.seek(0)
        return ServerProcessOutcome(
            argv=self.argv,
            pid=None if process is None else process.pid,
            started_at=self._started_at,
            finished_at=_utc_now(),
            duration_ns=max(0, finished_ns - started_ns),
            exit_code=None if process is None else process.returncode,
            stdout=self._stdout.read(),
            stderr=self._stderr.read(),
            forced_kill=self._forced_kill,
        )

    def close(self) -> None:
        self._stdout.close()
        self._stderr.close()

    def _require_process(self) -> subprocess.Popen[str]:
        if self._process is None:
            raise ServerProcessError("server process has not been started", kind="start_failed")
        return self._process


def _health_ready(url: str) -> bool:
    prefix = "http://"
    if not url.startswith(prefix):
        return False
    authority, separator, raw_path = url.removeprefix(prefix).partition("/")
    if not separator:
        raw_path = ""
    host, colon, raw_port = authority.rpartition(":")
    if not colon or not host:
        return False
    try:
        port = int(raw_port)
    except ValueError:
        return False

    connection = http.client.HTTPConnection(host, port, timeout=1.0)
    try:
        connection.request("GET", "/" + raw_path)
        response = connection.getresponse()
        response.read()
        return response.status == 200
    except (OSError, http.client.HTTPException):
        return False
    finally:
        connection.close()


def _signal_process_group(process: subprocess.Popen[str], sig: signal.Signals) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        return


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
