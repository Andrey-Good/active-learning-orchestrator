"""
Small utilities shared by multiple modules.

For juniors:
- This file should stay small and dependency-free.
- Put "generic helpers" here, but avoid dumping large business logic into utils.
"""

from __future__ import annotations


import dataclasses
import enum
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Union


def dataclass_to_dict(obj: Any) -> Any:
    """
    Convert dataclasses/enums into JSON-friendly types (dict/list/str/float/...).

    The state store uses this so it can write dataclass objects into `state.json`.
    """
    if dataclasses.is_dataclass(obj):
        result = {}
        for field_def in dataclasses.fields(obj):
            result[field_def.name] = dataclass_to_dict(getattr(obj, field_def.name))
        return result
    if isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {str(key): dataclass_to_dict(value) for key, value in obj.items()}
    if isinstance(obj, enum.Enum):
        return obj.value
    return obj


def _replace_with_retry(source: Path, target: Path, *, attempts: int = 8, initial_delay_seconds: float = 0.01) -> None:
    delay = initial_delay_seconds
    for attempt in range(attempts):
        try:
            os.replace(str(source), str(target))
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, 0.25)


def atomic_write_text(path: Union[str, Path], payload: str, encoding: str = "utf-8") -> None:
    """
    Write text to a file atomically.

    Why this exists:
    - If the process crashes while writing `state.json`, you do not want a half-written file.
    - We write to a temp file first, fsync it, then replace the target file in one operation.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding=encoding, dir=str(target.parent), delete=False) as tmp:
        tmp.write(payload)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    try:
        _replace_with_retry(tmp_path, target)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def sha256_file(path: Union[str, Path]) -> str:
    """Return the SHA-256 hex digest for a file's current bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Return a deterministic SHA-256 digest for a strict JSON-compatible value."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
