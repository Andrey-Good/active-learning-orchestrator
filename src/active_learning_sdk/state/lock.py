from __future__ import annotations

"""
Project locking.

The engine writes to `state.json` and cache files. If two processes do that at the same
time, the project can get corrupted.

This module provides a simple lock file (`project.lock`) to prevent concurrent runs.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional, Union

from ..exceptions import InfrastructureError, ProjectLockedError


class ProjectLock:
    """
    Cross-platform best-effort file lock using atomic file creation.

    For juniors:
    - When a project is opened with `lock=True`, the engine creates a lock file.
    - If the lock file already exists, it raises `ProjectLockedError`.

    Attributes:
        lock_path (Path):
            Where: the lock file is created at this path.
            What: usually `workdir/project.lock`.
            Why: unique file per project prevents concurrent writers.
        _fd (Optional[int]):
            Where: holds the file descriptor while the lock is acquired.
            What: integer fd or None.
            Why: keeping the fd open reduces the chance of accidental release.
    """

    def __init__(self, lock_path: Union[str, Path]) -> None:
        self.lock_path = Path(lock_path)
        self._fd: Optional[int] = None

    def acquire(self) -> None:
        """Acquire the lock by creating the lock file exclusively."""
        try:
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
            self._fd = os.open(str(self.lock_path), flags, 0o644)
            payload = json.dumps({"pid": os.getpid(), "created_at": time.time()}).encode("utf-8")
            os.write(self._fd, payload)
            os.fsync(self._fd)
        except FileExistsError as error:
            raise ProjectLockedError(f"Project is locked: {self.lock_path}") from error
        except Exception as error:
            raise InfrastructureError(f"Failed to acquire project lock: {error}") from error

    def release(self) -> None:
        """Release the lock by closing the file descriptor and deleting the lock file."""
        try:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            if self.lock_path.exists():
                self.lock_path.unlink()
        except Exception as error:
            raise InfrastructureError(f"Failed to release project lock: {error}") from error

    def __enter__(self) -> "ProjectLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
