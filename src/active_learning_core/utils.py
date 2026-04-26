from __future__ import annotations

"""
Small utilities shared by multiple modules.

For juniors:
- This file should stay small and dependency-free.
- Put "generic helpers" here, but avoid dumping large business logic into utils.
"""

import dataclasses
import enum
import os
import tempfile
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
    os.replace(str(tmp_path), str(target))
