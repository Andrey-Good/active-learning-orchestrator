"""
Project locking.

The engine writes to `state.json` and cache files. If two processes do that at the same
time, the project can get corrupted.

This module provides a simple lock file (`project.lock`) to prevent concurrent runs.
"""

from __future__ import annotations


import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Tuple, Union

from ..exceptions import InfrastructureError, ProjectLockedError

_ACQUISITION_GATE_TIMEOUT_SECONDS = 5.0
_MALFORMED_GATE_GRACE_SECONDS = 0.25


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
        self._owner_token: Optional[str] = None

    def acquire(self) -> None:
        """Acquire the lock by creating the lock file exclusively."""
        try:
            with self._acquisition_gate():
                self._acquire_locked()
        except ProjectLockedError:
            raise
        except Exception as error:
            raise InfrastructureError(f"Failed to acquire project lock: {error}") from error

    def _acquire_locked(self) -> None:
        try:
            self._acquire_once()
        except FileExistsError as error:
            if self._remove_stale_lock():
                try:
                    self._acquire_once()
                    return
                except FileExistsError:
                    pass
            raise ProjectLockedError(f"Project is locked: {self.lock_path}") from error

    def _acquire_once(self) -> None:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        self._fd = os.open(str(self.lock_path), flags, 0o644)
        self._owner_token = uuid.uuid4().hex
        payload = json.dumps(
            {"pid": os.getpid(), "created_at": time.time(), "owner_token": self._owner_token}
        ).encode("utf-8")
        os.write(self._fd, payload)
        os.fsync(self._fd)

    @contextmanager
    def _acquisition_gate(self) -> Iterator[None]:
        """
        Serialize acquire/stale-break operations across SDK processes.

        Without this short-lived gate, one process can remove a stale lock while
        another creates a fresh one in the unlink/create window.
        """
        gate_path = self.lock_path.with_name(f"{self.lock_path.name}.acquire")
        token = uuid.uuid4().hex
        fd: Optional[int] = None
        deadline = time.monotonic() + _ACQUISITION_GATE_TIMEOUT_SECONDS
        while fd is None:
            try:
                fd = os.open(str(gate_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                gate_payload = json.dumps(
                    {"pid": os.getpid(), "created_at": time.time(), "owner_token": token}
                ).encode("utf-8")
                os.write(fd, gate_payload)
                os.fsync(fd)
            except FileExistsError as error:
                if self._remove_stale_payload_file(gate_path):
                    continue
                if time.monotonic() >= deadline:
                    raise ProjectLockedError(f"Project lock acquisition gate is busy: {gate_path}") from error
                time.sleep(0.01)
            except Exception:
                if fd is not None:
                    os.close(fd)
                raise

        try:
            yield
        finally:
            try:
                os.close(fd)
                payload = self._read_payload_file(gate_path)
                if payload is not None and payload.get("pid") == os.getpid() and payload.get("owner_token") == token:
                    gate_path.unlink()
            except FileNotFoundError:
                pass

    def _remove_stale_lock(self) -> bool:
        first_read = self._read_lock_payload_with_identity()
        if first_read is None:
            return False
        stale_payload, stale_identity = first_read
        if stale_payload is None or not self._is_stale_payload(stale_payload):
            return False
        current_read = self._read_lock_payload_with_identity()
        if current_read is None:
            return True
        current_payload, current_identity = current_read
        if current_payload != stale_payload or current_identity != stale_identity:
            return False
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            return True
        except Exception:
            return False
        return True

    def _remove_stale_payload_file(self, path: Path) -> bool:
        first_read = self._read_payload_file_with_identity(path)
        if first_read is None:
            return self._remove_malformed_payload_file_if_stable(path)
        stale_payload, stale_identity = first_read
        if not self._is_stale_payload(stale_payload):
            return False
        current_read = self._read_payload_file_with_identity(path)
        if current_read is None:
            return True
        current_payload, current_identity = current_read
        if current_payload != stale_payload or current_identity != stale_identity:
            return False
        try:
            path.unlink()
        except FileNotFoundError:
            return True
        except Exception:
            return False
        return True

    def _remove_malformed_payload_file_if_stable(self, path: Path) -> bool:
        first_identity = self._payload_file_identity(path)
        if first_identity is None:
            return True
        first_age_seconds = max(0.0, time.time() - (first_identity[3] / 1_000_000_000))
        if first_age_seconds < _MALFORMED_GATE_GRACE_SECONDS:
            return False
        current_read = self._read_payload_file_with_identity(path)
        if current_read is not None:
            return False
        current_identity = self._payload_file_identity(path)
        if current_identity is None:
            return True
        if current_identity != first_identity:
            return False
        try:
            path.unlink()
        except FileNotFoundError:
            return True
        except Exception:
            return False
        return True

    def _read_lock_payload(self) -> Optional[Mapping[str, Any]]:
        return self._read_payload_file(self.lock_path)

    def _read_payload_file(self, path: Path) -> Optional[Mapping[str, Any]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _read_lock_payload_with_identity(self) -> Optional[Tuple[Mapping[str, Any], Tuple[int, int, int, int]]]:
        return self._read_payload_file_with_identity(self.lock_path)

    def _read_payload_file_with_identity(
        self, path: Path
    ) -> Optional[Tuple[Mapping[str, Any], Tuple[int, int, int, int]]]:
        try:
            fd = os.open(str(path), os.O_RDONLY)
        except Exception:
            return None
        try:
            stat_result = os.fstat(fd)
            chunks = []
            while True:
                chunk = os.read(fd, 8192)
                if not chunk:
                    break
                chunks.append(chunk)
            payload = json.loads(b"".join(chunks).decode("utf-8"))
        except Exception:
            return None
        finally:
            os.close(fd)
        if not isinstance(payload, dict):
            return None
        identity = (
            int(getattr(stat_result, "st_dev", 0)),
            int(getattr(stat_result, "st_ino", 0)),
            int(getattr(stat_result, "st_size", 0)),
            int(getattr(stat_result, "st_mtime_ns", 0)),
        )
        return payload, identity

    def _payload_file_identity(self, path: Path) -> Optional[Tuple[int, int, int, int]]:
        try:
            stat_result = path.stat()
        except Exception:
            return None
        return (
            int(getattr(stat_result, "st_dev", 0)),
            int(getattr(stat_result, "st_ino", 0)),
            int(getattr(stat_result, "st_size", 0)),
            int(getattr(stat_result, "st_mtime_ns", 0)),
        )

    def _is_stale_lock(self) -> bool:
        payload = self._read_lock_payload()
        return payload is not None and self._is_stale_payload(payload)

    def _is_stale_payload(self, payload: Mapping[str, Any]) -> bool:
        pid = payload.get("pid")
        if not isinstance(pid, (str, bytes, int)):
            return False
        try:
            pid_int = int(pid)
        except Exception:
            return False
        if pid_int <= 0 or pid_int == os.getpid():
            return False
        try:
            os.kill(pid_int, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        except OSError as error:
            if getattr(error, "winerror", None) == 87:
                return True
            if getattr(error, "errno", None) == 22:
                return True
            return False
        return False

    def release(self) -> None:
        """Release the lock by closing the file descriptor and deleting the lock file."""
        if self._fd is None:
            return
        try:
            os.close(self._fd)
            self._fd = None
            if self.lock_path.exists() and self._lock_file_is_owned_by_this_instance():
                self.lock_path.unlink()
            self._owner_token = None
        except Exception as error:
            raise InfrastructureError(f"Failed to release project lock: {error}") from error

    def _lock_file_is_owned_by_this_instance(self) -> bool:
        if self._owner_token is None:
            return False
        payload = self._read_lock_payload()
        if payload is None:
            return False
        return payload.get("pid") == os.getpid() and payload.get("owner_token") == self._owner_token

    def __enter__(self) -> "ProjectLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
