"""Host-level exclusive execution lock."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from types import TracebackType
from typing import Self


class HostLockError(RuntimeError):
    """Raised when another benchmark executor already owns the host lock."""


class HostLock:
    """Advisory exclusive lock held for a benchmark execution session."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: object | None = None

    def acquire(self) -> None:
        """Acquire the lock without waiting."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise HostLockError(
                f"benchmark host lock is already held: {self.path}"
            ) from exc

        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        self._handle = handle

    def release(self) -> None:
        """Release the lock if held."""
        handle = self._handle
        if handle is None:
            return
        if not hasattr(handle, "fileno") or not hasattr(handle, "close"):
            raise RuntimeError("invalid host lock handle")
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        self._handle = None

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()
