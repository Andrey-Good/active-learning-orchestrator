"""
Caching layer.

Selection strategies may call the model many times. Caching prevents repeating the
same expensive computation on restart or between rounds.

This file contains:
- CacheStore: a tiny storage interface
- InMemoryCacheStore / JsonlDiskCacheStore: simple implementations
- PredictionCache / EmbeddingCache: helpers to build stable cache keys
"""

from __future__ import annotations


import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Union

from .utils import atomic_write_text


def _scope_value(value: Any) -> str:
    if isinstance(value, str):
        raw = f"str:{value}"
    elif value is None:
        raw = "none:null"
    elif isinstance(value, bool):
        raw = f"bool:{value}"
    elif isinstance(value, int):
        raw = f"int:{value}"
    elif isinstance(value, float):
        raw = f"float:{value}"
    else:
        try:
            raw = f"json:{json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)}"
        except TypeError:
            raw = f"repr:{value!r}"
    return hashlib.blake2b(raw.encode("utf-8", errors="replace"), digest_size=16).hexdigest()


class CacheStore(Protocol):
    """
    Storage interface for cache entries.

    For juniors:
    - A CacheStore is like a dict, but it can be backed by memory or disk.
    - The engine uses it via this interface, so you can swap implementations
      without changing engine code.

    Attributes:
        (implementation-specific):
            Where: concrete classes like `InMemoryCacheStore` and `JsonlDiskCacheStore` store their own data.
            What: for example a dict, file paths, counters.
            Why: the engine depends only on the method contract, not on internal storage details.
    """

    def get(self, key: str) -> Optional[Any]:
        ...

    def set(self, key: str, value: Any) -> None:
        ...

    def delete(self, key: str) -> None:
        ...

    def stats(self, key_prefix: Optional[str] = None) -> Dict[str, Any]:
        ...

    def clear(self, *, reason: str = "manual", kind: str = "all") -> None:
        ...


def _store_stats(store: CacheStore, *, key_prefix: Optional[str] = None) -> Dict[str, Any]:
    try:
        return store.stats(key_prefix=key_prefix)
    except TypeError:
        stats = store.stats()
        if key_prefix is not None:
            stats = dict(stats)
            stats.setdefault("current_reusable_items", stats.get("items"))
        return stats


def _store_clear(store: CacheStore, *, reason: str, kind: str) -> None:
    try:
        store.clear(reason=reason, kind=kind)
    except TypeError:
        store.clear()


def _store_record_invalidation(store: CacheStore, *, reason: str, kind: str) -> None:
    recorder = getattr(store, "record_invalidation", None)
    if callable(recorder):
        recorder(reason=reason, kind=kind)


class InMemoryCacheStore:
    """
    Simple in-memory cache store with optional size cap.

    This is useful for notebooks and quick experiments. It is not crash-safe.

    Attributes:
        _max_items (Optional[int]):
            Where: used during `set()` to decide whether to evict.
            What: maximum number of keys to keep in memory.
            Why: prevents uncontrolled memory usage.
        _data (Dict[str, Any]):
            Where: stores cached values.
            What: in-memory mapping key -> value.
            Why: fast reads/writes.
        _hits/_misses (int):
            Where: returned in `stats()`.
            What: counters.
            Why: helps you see if cache is effective.
    """

    def __init__(self, max_items: Optional[int] = None) -> None:
        self._max_items = max_items
        self._data: Dict[str, Any] = {}
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._deletes = 0
        self._clears = 0
        self._last_clear_reason: Optional[str] = None
        self._last_clear_kind: Optional[str] = None
        self._last_cleared_at: Optional[float] = None
        self._last_cleared_items = 0
        self._invalidations = 0
        self._last_invalidation_reason: Optional[str] = None
        self._last_invalidation_kind: Optional[str] = None
        self._last_invalidated_at: Optional[float] = None

    def get(self, key: str) -> Optional[Any]:
        if key in self._data:
            self._hits += 1
            return self._data[key]
        self._misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        if self._max_items is not None and self._max_items <= 0:
            return
        if self._max_items is not None and len(self._data) >= self._max_items and key not in self._data:
            oldest_key = next(iter(self._data.keys()))
            self._data.pop(oldest_key, None)
        self._data[key] = value
        self._writes += 1

    def delete(self, key: str) -> None:
        if key in self._data:
            self._deletes += 1
            self._data.pop(key, None)

    def stats(self, key_prefix: Optional[str] = None) -> Dict[str, Any]:
        current_reusable_items = (
            sum(1 for key in self._data if key.startswith(key_prefix)) if key_prefix is not None else len(self._data)
        )
        return {
            "items": len(self._data),
            "stored_items": len(self._data),
            "current_reusable_items": current_reusable_items,
            "hits": self._hits,
            "misses": self._misses,
            "writes": self._writes,
            "deletes": self._deletes,
            "clears": self._clears,
            "session_hits": self._hits,
            "session_misses": self._misses,
            "session_writes": self._writes,
            "session_deletes": self._deletes,
            "session_clears": self._clears,
            "lifetime_writes": self._writes,
            "lifetime_deletes": self._deletes,
            "lifetime_clears": self._clears,
            "lifetime_invalidations": self._invalidations,
            "data_bytes": 0,
            "index_bytes": 0,
            "metadata_bytes": 0,
            "last_clear_reason": self._last_clear_reason,
            "last_clear_kind": self._last_clear_kind,
            "last_cleared_at": self._last_cleared_at,
            "last_cleared_items": self._last_cleared_items,
            "last_cleared_bytes": 0,
            "last_invalidation_reason": self._last_invalidation_reason,
            "last_invalidation_kind": self._last_invalidation_kind,
            "last_invalidated_at": self._last_invalidated_at,
        }

    def clear(self, *, reason: str = "manual", kind: str = "all") -> None:
        self._last_clear_reason = reason
        self._last_clear_kind = kind
        self._last_cleared_at = time.time()
        self._last_cleared_items = len(self._data)
        self._data.clear()
        self._clears += 1
        self._hits = 0
        self._misses = 0

    def record_invalidation(self, *, reason: str, kind: str) -> None:
        self._invalidations += 1
        self._last_invalidation_reason = reason
        self._last_invalidation_kind = kind
        self._last_invalidated_at = time.time()


class JsonlDiskCacheStore:
    """
    Append-only JSONL disk cache store.

    Contract:
    - It appends values to a `.jsonl` file.
    - It stores an index (key -> byte offset) in a separate json file.
    - It is a single-process/single-writer cache. Concurrent writers are not supported.
    - On open, a missing/corrupt/legacy index is rebuilt from the JSONL log.
    - If the previous process crashed during append, trailing partial/corrupt records are truncated during rebuild.

    Limitations (important):
    - `set()` is O(1) relative to the number of cache records.
    - `delete()` only removes from the index, it does not shrink the log.
    - No cross-process concurrency support.

    For production, consider SQLite/LMDB.

    Attributes:
        cache_dir (Path):
            Where: created in `__init__` and used to compute `_index_path`/`_data_path`.
            What: directory on disk where this cache namespace lives.
            Why: keeps cache files inside the project workdir (or another chosen folder).
        namespace (str):
            Where: used to build file names and reported via `stats()`.
            What: logical cache name, e.g. "predictions" or "embeddings".
            Why: separates different cache types so keys do not collide.
        _index_path (Path):
            Where: read/written by `_load_index()`/`_save_index()`.
            What: path to JSON file storing key -> line_number.
            Why: allows O(1) lookup of where an entry is in the `.jsonl` log.
        _data_path (Path):
            Where: appended to by `set()` and scanned by `get()`.
            What: path to the `.jsonl` append-only log.
            Why: simplest durable storage format for the scaffold.
        _index (Dict[str, int]):
            Where: in-memory mapping used by `get()`/`set()`/`delete()`.
            What: key -> byte offset in the `.jsonl` file.
            Why: avoids scanning the file for every read/write.
        _next_offset (int):
            Where: updated after index recovery and each append.
            What: expected end-of-log byte offset.
            Why: supports O(1) appends without counting lines.
        _hits/_misses (int):
            Where: returned from `stats()`.
            What: counters for cache effectiveness.
            Why: helps you confirm caching actually improves performance.
    """

    def __init__(self, cache_dir: Union[str, Path], namespace: str) -> None:
        self.cache_dir = Path(cache_dir)
        self.namespace = namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.cache_dir / f"{namespace}.index.json"
        self._data_path = self.cache_dir / f"{namespace}.jsonl"
        self._stats_path = self.cache_dir / f"{namespace}.stats.json"
        self._index: Dict[str, int] = {}
        self._next_offset = 0
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._deletes = 0
        self._clears = 0
        self._metadata = self._load_stats_metadata()
        self._load_index()

    def _load_stats_metadata(self) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {
            "lifetime_writes": 0,
            "lifetime_deletes": 0,
            "lifetime_clears": 0,
            "lifetime_invalidations": 0,
            "last_clear_reason": None,
            "last_clear_kind": None,
            "last_cleared_at": None,
            "last_cleared_items": 0,
            "last_cleared_bytes": 0,
            "last_invalidation_reason": None,
            "last_invalidation_kind": None,
            "last_invalidated_at": None,
        }
        if not self._stats_path.exists():
            return defaults
        try:
            raw = json.loads(self._stats_path.read_text(encoding="utf-8"))
        except Exception:
            return defaults
        if not isinstance(raw, dict):
            return defaults
        metadata = dict(defaults)
        for key, default in defaults.items():
            value = raw.get(key, default)
            if isinstance(default, int):
                metadata[key] = value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else default
            elif isinstance(default, float):
                metadata[key] = value if isinstance(value, (int, float)) and not isinstance(value, bool) else default
            else:
                metadata[key] = value
        return metadata

    def _save_stats_metadata(self) -> None:
        atomic_write_text(
            self._stats_path,
            json.dumps(self._metadata, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _load_index(self) -> None:
        if not self._data_path.exists():
            self._index = {}
            self._next_offset = 0
            return
        if not self._index_path.exists():
            self._rebuild_index_from_log()
            return
        try:
            raw_index = json.loads(self._index_path.read_text(encoding="utf-8"))
            if not isinstance(raw_index, dict):
                raise ValueError("cache index must be a JSON object")
            parsed: Dict[str, int] = {}
            for raw_key, raw_value in raw_index.items():
                if isinstance(raw_value, int) and not isinstance(raw_value, bool):
                    # Legacy indexes stored line numbers as bare integers. Rebuild once
                    # so future gets can seek directly by byte offset.
                    self._rebuild_index_from_log()
                    return
                if not isinstance(raw_value, dict):
                    raise ValueError("cache index values must be offset objects")
                raw_offset = raw_value.get("offset")
                if not isinstance(raw_offset, int) or isinstance(raw_offset, bool) or raw_offset < 0:
                    raise ValueError("cache index offset must be a non-negative integer")
                parsed[str(raw_key)] = raw_offset
            self._index = parsed
            self._next_offset = self._data_path.stat().st_size
        except Exception:
            self._rebuild_index_from_log()

    def _save_index(self) -> None:
        payload = {key: {"offset": offset} for key, offset in sorted(self._index.items())}
        atomic_write_text(
            self._index_path,
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _rebuild_index_from_log(self) -> None:
        rebuilt: Dict[str, int] = {}
        offset = 0
        valid_offset = 0
        try:
            with self._data_path.open("rb") as data_file:
                for line in data_file:
                    line_offset = offset
                    offset += len(line)
                    if not line.strip():
                        valid_offset = offset
                        continue
                    try:
                        record = json.loads(line.decode("utf-8"))
                    except Exception:
                        break
                    if not isinstance(record, dict) or "key" not in record or "value" not in record:
                        break
                    rebuilt[str(record["key"])] = line_offset
                    valid_offset = offset
        except Exception:
            rebuilt = {}
            valid_offset = 0

        file_size = self._data_path.stat().st_size if self._data_path.exists() else 0
        if valid_offset < file_size:
            with self._data_path.open("r+b") as data_file:
                data_file.truncate(valid_offset)

        self._index = rebuilt
        self._next_offset = valid_offset
        self._save_index()

    def get(self, key: str) -> Optional[Any]:
        """Return cached value or None if missing."""
        if key not in self._index:
            self._misses += 1
            return None
        target_line = self._index[key]
        try:
            with self._data_path.open("r", encoding="utf-8") as data_file:
                data_file.seek(target_line)
                line = data_file.readline()
                record = json.loads(line)
                if record.get("key") != key:
                    self._delete_index_entry(key)
                    self._misses += 1
                    return None
                value = record["value"]
                self._hits += 1
                return value
        except Exception:
            self._delete_index_entry(key)
            self._misses += 1
            return None

    def set(self, key: str, value: Any) -> None:
        """Persist a value by appending it to the jsonl file."""
        record = {"key": key, "value": value}
        serialized = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with self._data_path.open("a", encoding="utf-8") as data_file:
            data_file.seek(0, 2)
            offset = data_file.tell()
            data_file.write(serialized + "\n")
            self._next_offset = data_file.tell()
        self._index[key] = offset
        self._writes += 1
        self._metadata["lifetime_writes"] = int(self._metadata.get("lifetime_writes", 0)) + 1
        self._save_index()
        self._save_stats_metadata()

    def delete(self, key: str) -> None:
        if key in self._index:
            self._index.pop(key, None)
            self._deletes += 1
            self._metadata["lifetime_deletes"] = int(self._metadata.get("lifetime_deletes", 0)) + 1
            self._save_index()
            self._save_stats_metadata()

    def _delete_index_entry(self, key: str) -> None:
        if key in self._index:
            self._index.pop(key, None)
            self._save_index()

    def stats(self, key_prefix: Optional[str] = None) -> Dict[str, Any]:
        stored_items = len(self._index)
        current_reusable_items = (
            sum(1 for key in self._index if key.startswith(key_prefix)) if key_prefix is not None else stored_items
        )
        data_bytes = self._data_path.stat().st_size if self._data_path.exists() else 0
        index_bytes = self._index_path.stat().st_size if self._index_path.exists() else 0
        metadata_bytes = self._stats_path.stat().st_size if self._stats_path.exists() else 0
        return {
            "namespace": self.namespace,
            "items": stored_items,
            "stored_items": stored_items,
            "current_reusable_items": current_reusable_items,
            "hits": self._hits,
            "misses": self._misses,
            "writes": int(self._metadata.get("lifetime_writes", 0)),
            "deletes": int(self._metadata.get("lifetime_deletes", 0)),
            "clears": int(self._metadata.get("lifetime_clears", 0)),
            "session_hits": self._hits,
            "session_misses": self._misses,
            "session_writes": self._writes,
            "session_deletes": self._deletes,
            "session_clears": self._clears,
            "lifetime_writes": int(self._metadata.get("lifetime_writes", 0)),
            "lifetime_deletes": int(self._metadata.get("lifetime_deletes", 0)),
            "lifetime_clears": int(self._metadata.get("lifetime_clears", 0)),
            "lifetime_invalidations": int(self._metadata.get("lifetime_invalidations", 0)),
            "data_bytes": data_bytes,
            "index_bytes": index_bytes,
            "metadata_bytes": metadata_bytes,
            "last_clear_reason": self._metadata.get("last_clear_reason"),
            "last_clear_kind": self._metadata.get("last_clear_kind"),
            "last_cleared_at": self._metadata.get("last_cleared_at"),
            "last_cleared_items": self._metadata.get("last_cleared_items", 0),
            "last_cleared_bytes": self._metadata.get("last_cleared_bytes", 0),
            "last_invalidation_reason": self._metadata.get("last_invalidation_reason"),
            "last_invalidation_kind": self._metadata.get("last_invalidation_kind"),
            "last_invalidated_at": self._metadata.get("last_invalidated_at"),
        }

    def clear(self, *, reason: str = "manual", kind: str = "all") -> None:
        cleared_items = len(self._index)
        cleared_bytes = (
            (self._data_path.stat().st_size if self._data_path.exists() else 0)
            + (self._index_path.stat().st_size if self._index_path.exists() else 0)
        )
        self._index.clear()
        if self._data_path.exists():
            self._data_path.unlink()
        if self._index_path.exists():
            self._index_path.unlink()
        self._clears += 1
        self._metadata["lifetime_clears"] = int(self._metadata.get("lifetime_clears", 0)) + 1
        if reason != "manual":
            self._metadata["lifetime_invalidations"] = int(self._metadata.get("lifetime_invalidations", 0)) + 1
        self._metadata["last_clear_reason"] = reason
        self._metadata["last_clear_kind"] = kind
        self._metadata["last_cleared_at"] = time.time()
        self._metadata["last_cleared_items"] = cleared_items
        self._metadata["last_cleared_bytes"] = cleared_bytes
        self._hits = 0
        self._misses = 0
        self._save_stats_metadata()

    def record_invalidation(self, *, reason: str, kind: str) -> None:
        self._metadata["lifetime_invalidations"] = int(self._metadata.get("lifetime_invalidations", 0)) + 1
        self._metadata["last_invalidation_reason"] = reason
        self._metadata["last_invalidation_kind"] = kind
        self._metadata["last_invalidated_at"] = time.time()
        self._save_stats_metadata()


class PredictionCache:
    """
    Cache for model prediction probabilities keyed by model, dataset, and sample.

    The `model_id` is used so that caches do not mix results from different
    versions of the model (after training).

    Attributes:
        _store (CacheStore):
            Where: used by get/set/clear.
            What: underlying storage implementation.
            Why: lets you choose memory vs disk without changing selection code.
    """

    def __init__(self, store: CacheStore) -> None:
        self._store = store

    def get(self, model_id: str, sample_id: str, dataset_fingerprint: Optional[str] = None) -> Optional[Any]:
        return self._store.get(self._key(model_id, sample_id, dataset_fingerprint))

    def set(self, model_id: str, sample_id: str, proba: Any, dataset_fingerprint: Optional[str] = None) -> None:
        self._store.set(self._key(model_id, sample_id, dataset_fingerprint), proba)

    def delete(self, model_id: str, sample_id: str, dataset_fingerprint: Optional[str] = None) -> None:
        self._store.delete(self._key(model_id, sample_id, dataset_fingerprint))

    def clear(self, *, reason: str = "manual", kind: str = "predictions") -> None:
        _store_clear(self._store, reason=reason, kind=kind)

    def stats(self, model_id: Optional[str] = None, dataset_fingerprint: Optional[str] = None) -> Dict[str, Any]:
        key_prefix = self._key_prefix(model_id, dataset_fingerprint) if model_id is not None else None
        return _store_stats(self._store, key_prefix=key_prefix)

    def record_invalidation(self, *, reason: str, kind: str = "predictions") -> None:
        _store_record_invalidation(self._store, reason=reason, kind=kind)

    def _key(self, model_id: str, sample_id: str, dataset_fingerprint: Optional[str] = None) -> str:
        return f"{self._key_prefix(model_id, dataset_fingerprint)}{self._scope_value(sample_id)}"

    def _key_prefix(self, model_id: str, dataset_fingerprint: Optional[str] = None) -> str:
        dataset_scope = self._scope_value(dataset_fingerprint if dataset_fingerprint is not None else "default-dataset")
        return f"pred::{self._scope_value(model_id)}::{dataset_scope}::"

    def _scope_value(self, value: Any) -> str:
        return _scope_value(value)


class EmbeddingCache:
    """
    Cache for model embeddings keyed by model, dataset, embedding config, and sample id.

    Attributes:
        _store (CacheStore):
            Where: used by get/set/clear.
            What: underlying storage implementation.
            Why: same reason as PredictionCache.
    """

    def __init__(self, store: CacheStore) -> None:
        self._store = store

    def get(
        self,
        model_id: str,
        sample_id: str,
        dataset_fingerprint: Optional[str] = None,
        embedding_config: Optional[Any] = None,
    ) -> Optional[Any]:
        return self._store.get(self._key(model_id, sample_id, dataset_fingerprint, embedding_config))

    def set(
        self,
        model_id: str,
        sample_id: str,
        emb: Any,
        dataset_fingerprint: Optional[str] = None,
        embedding_config: Optional[Any] = None,
    ) -> None:
        self._store.set(self._key(model_id, sample_id, dataset_fingerprint, embedding_config), emb)

    def delete(
        self,
        model_id: str,
        sample_id: str,
        dataset_fingerprint: Optional[str] = None,
        embedding_config: Optional[Any] = None,
    ) -> None:
        self._store.delete(self._key(model_id, sample_id, dataset_fingerprint, embedding_config))

    def clear(self, *, reason: str = "manual", kind: str = "embeddings") -> None:
        _store_clear(self._store, reason=reason, kind=kind)

    def stats(
        self,
        model_id: Optional[str] = None,
        dataset_fingerprint: Optional[str] = None,
        embedding_config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        key_prefix = (
            self._key_prefix(model_id, dataset_fingerprint, embedding_config) if model_id is not None else None
        )
        return _store_stats(self._store, key_prefix=key_prefix)

    def record_invalidation(self, *, reason: str, kind: str = "embeddings") -> None:
        _store_record_invalidation(self._store, reason=reason, kind=kind)

    def _key(
        self,
        model_id: str,
        sample_id: str,
        dataset_fingerprint: Optional[str] = None,
        embedding_config: Optional[Any] = None,
    ) -> str:
        return f"{self._key_prefix(model_id, dataset_fingerprint, embedding_config)}{self._scope_value(sample_id)}"

    def _key_prefix(
        self,
        model_id: str,
        dataset_fingerprint: Optional[str] = None,
        embedding_config: Optional[Any] = None,
    ) -> str:
        dataset_scope = self._scope_value(dataset_fingerprint if dataset_fingerprint is not None else "default-dataset")
        config_scope = self._scope_value(embedding_config if embedding_config is not None else "default-embedding")
        return f"emb::{self._scope_value(model_id)}::{dataset_scope}::{config_scope}::"

    def _scope_value(self, value: Any) -> str:
        return _scope_value(value)
