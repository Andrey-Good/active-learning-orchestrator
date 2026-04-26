from __future__ import annotations

"""
Dataset access layer.

The engine does not want to know whether your data comes from:
- a pandas DataFrame
- a CSV file
- a database
- an API

Instead it talks to a simple `DatasetProvider` interface.
If you need to support a new dataset source, implement `DatasetProvider`.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Protocol, Sequence, runtime_checkable

from ..exceptions import ConfigurationError
from ..types import DataSample

try:
    import pandas as _pd  # type: ignore
except Exception:
    _pd = None  # type: ignore


@runtime_checkable
class DatasetProvider(Protocol):
    """
    Dataset access contract.

    Minimum required methods:
    - iter_sample_ids(): yields all ids in the dataset
    - get_sample(sample_id): returns one DataSample

    Notes for juniors:
    - sample ids should be stable across runs (do not use row index that can change).
    - `schema()` should return a stable description used in fingerprinting.

    Attributes:
        (implementation-specific):
            Where: a provider stores whatever it needs to fetch samples.
            What: for example an in-memory table, a file path, a DB connection, or API client.
            Why: the engine treats the provider as a black box and only calls its methods.
    """

    def iter_sample_ids(self) -> Iterator[str]:
        ...

    def get_sample(self, sample_id: str) -> DataSample:
        ...

    def get_samples(self, sample_ids: Sequence[str]) -> List[DataSample]:
        return [self.get_sample(sample_id) for sample_id in sample_ids]

    def schema(self) -> Dict[str, str]:
        return {"sample_id": "str", "text": "str"}


class DataFrameDatasetProvider:
    """
    `pandas.DataFrame` backed dataset provider.

    Expected columns by default:
    - sample_id
    - text

    Optional columns:
    - meta (dict-like or JSON string)
    - group_id

    Attributes:
        _df (Any):
            Where: used by `iter_sample_ids()` and `get_sample()`.
            What: the pandas DataFrame that holds data.
            Why: provides fast access to rows/columns.
        _text_column (str):
            Where: used in `get_sample()` to extract text.
            What: name of the text column.
            Why: lets you adapt to different dataset column names.
        _id_column (str):
            Where: used to extract sample_id values.
            What: name of the id column.
            Why: stable ids are required by the engine/state.
        _index (Dict[str, int]):
            Where: used in `get_sample()` for O(1) lookup.
            What: mapping sample_id -> row index.
            Why: makes random access fast when the engine needs a subset of ids.
    """

    def __init__(self, df: Any, text_column: str = "text", id_column: str = "sample_id") -> None:
        if _pd is None:
            raise ConfigurationError("pandas is not installed. Install pandas or provide a custom DatasetProvider.")
        if id_column not in df.columns or text_column not in df.columns:
            raise ConfigurationError(f"DataFrame must contain columns: {id_column!r}, {text_column!r}")
        self._df = df
        self._text_column = text_column
        self._id_column = id_column
        self._index: Dict[str, int] = {}
        for row_index, sample_id in enumerate(df[id_column].astype(str).tolist()):
            if sample_id in self._index:
                raise ConfigurationError(f"Duplicate sample_id in dataset: {sample_id!r}")
            self._index[sample_id] = row_index

    @classmethod
    def from_path(cls, path: Path) -> "DataFrameDatasetProvider":
        """Load CSV/Parquet via pandas and wrap as a provider."""
        if _pd is None:
            raise ConfigurationError("Reading dataset from path requires pandas.")
        if not path.exists():
            raise ConfigurationError(f"Dataset path does not exist: {path}")
        if path.suffix.lower() == ".csv":
            df = _pd.read_csv(path)
            return cls(df)
        if path.suffix.lower() in {".parquet", ".pq"}:
            df = _pd.read_parquet(path)
            return cls(df)
        raise ConfigurationError(f"Unsupported dataset file type: {path.suffix}")

    def iter_sample_ids(self) -> Iterator[str]:
        for sample_id in self._df[self._id_column].astype(str).tolist():
            yield sample_id

    def get_sample(self, sample_id: str) -> DataSample:
        try:
            row_index = self._index[str(sample_id)]
        except KeyError as error:
            raise KeyError(f"Unknown sample_id={sample_id!r}") from error

        row = self._df.iloc[row_index]
        data = {"text": str(row[self._text_column])}
        meta: Dict[str, Any] = {}
        if "meta" in self._df.columns:
            meta_value = row["meta"]
            if isinstance(meta_value, str):
                try:
                    meta = json.loads(meta_value)
                except Exception:
                    meta = {"meta": meta_value}
            elif isinstance(meta_value, dict):
                meta = meta_value
            else:
                meta = {"meta": meta_value}

        group_id = str(row["group_id"]) if "group_id" in self._df.columns and row["group_id"] is not None else None
        return DataSample(sample_id=str(sample_id), data=data, meta=meta, group_id=group_id)

    def schema(self) -> Dict[str, str]:
        return {"sample_id": "str", "text": "str"}
