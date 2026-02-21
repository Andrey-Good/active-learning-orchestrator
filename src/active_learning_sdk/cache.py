from __future__ import annotations

"""
Caching layer.

Selection strategies may call the model many times. Caching prevents repeating the
same expensive computation on restart or between rounds.

This file contains:
- CacheStore: a tiny storage interface
- InMemoryCacheStore / JsonlDiskCacheStore: simple implementations
- PredictionCache / EmbeddingCache: helpers to build stable cache keys
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Union


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

    def stats(self) -> Dict[str, Any]:
        ...

    def clear(self) -> None:
        ...


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

    def get(self, key: str) -> Optional[Any]:
        if key in self._data:
            self._hits += 1
            return self._data[key]
        self._misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        if self._max_items is not None and len(self._data) >= self._max_items and key not in self._data:
            oldest_key = next(iter(self._data.keys()))
            self._data.pop(oldest_key, None)
        self._data[key] = value

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def stats(self) -> Dict[str, Any]:
        return {"items": len(self._data), "hits": self._hits, "misses": self._misses}

    def clear(self) -> None:
        self._data.clear()
        self._hits = 0
        self._misses = 0


class JsonlDiskCacheStore:
    """
    Very simple JSONL disk cache store.

    This is a scaffold implementation:
    - It appends values to a `.jsonl` file.
    - It stores an index (key -> line number) in a separate json file.

    Limitations (important):
    - `set()` is O(N) because it counts lines each time.
    - `delete()` only removes from the index, it does not shrink the log.
    - No concurrency support.

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
            What: key -> line number in the `.jsonl` file.
            Why: avoids scanning the file to find the offset for every read.
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
        self._index: Dict[str, int] = {}
        self._hits = 0
        self._misses = 0
        self._load_index()

    def _load_index(self) -> None:
        if not self._index_path.exists():
            return
        try:
            self._index = json.loads(self._index_path.read_text(encoding="utf-8"))
        except Exception:
            self._index = {}

    def _save_index(self) -> None:
        self._index_path.write_text(json.dumps(self._index, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, key: str) -> Optional[Any]:
        """Return cached value or None if missing."""
        if key not in self._index:
            self._misses += 1
            return None
        target_line = self._index[key]
        try:
            with self._data_path.open("r", encoding="utf-8") as data_file:
                for line_number, line in enumerate(data_file):
                    if line_number == target_line:
                        self._hits += 1
                        return json.loads(line)["value"]
        except Exception:
            pass
        self._misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        """Persist a value by appending it to the jsonl file."""
        record = {"key": key, "value": value}
        with self._data_path.open("a", encoding="utf-8") as data_file:
            line_no = sum(1 for _ in self._data_path.open("r", encoding="utf-8")) if self._data_path.exists() else 0
            data_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._index[key] = line_no
        self._save_index()

    def delete(self, key: str) -> None:
        self._index.pop(key, None)
        self._save_index()

    def stats(self) -> Dict[str, Any]:
        return {"namespace": self.namespace, "items": len(self._index), "hits": self._hits, "misses": self._misses}

    def clear(self) -> None:
        self._index.clear()
        if self._data_path.exists():
            self._data_path.unlink()
        if self._index_path.exists():
            self._index_path.unlink()
        self._hits = 0
        self._misses = 0


class PredictionCache:
    """
    Cache for model prediction probabilities keyed by `(model_id, sample_id)`.

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

    def get(self, model_id: str, sample_id: str) -> Optional[Any]:
        return self._store.get(self._key(model_id, sample_id))

    def set(self, model_id: str, sample_id: str, proba: Any) -> None:
        self._store.set(self._key(model_id, sample_id), proba)

    def clear(self) -> None:
        self._store.clear()

    def stats(self) -> Dict[str, Any]:
        return self._store.stats()

    def _key(self, model_id: str, sample_id: str) -> str:
        return f"pred::{model_id}::{sample_id}"


class EmbeddingCache:
    """
    Cache for model embeddings keyed by `(model_id, sample_id)`.

    Attributes:
        _store (CacheStore):
            Where: used by get/set/clear.
            What: underlying storage implementation.
            Why: same reason as PredictionCache.
    """

    def __init__(self, store: CacheStore) -> None:
        self._store = store

    def get(self, model_id: str, sample_id: str) -> Optional[Any]:
        return self._store.get(self._key(model_id, sample_id))

    def set(self, model_id: str, sample_id: str, emb: Any) -> None:
        self._store.set(self._key(model_id, sample_id), emb)

    def clear(self) -> None:
        self._store.clear()

    def stats(self) -> Dict[str, Any]:
        return self._store.stats()

    def _key(self, model_id: str, sample_id: str) -> str:
        return f"emb::{model_id}::{sample_id}"
