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

from __future__ import annotations


import json
import math
from collections.abc import Mapping as MappingABC
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Protocol, Sequence, runtime_checkable

from ..exceptions import ConfigurationError
from ..types import DataSample


def _pandas() -> Any:
    try:
        import pandas as pd  # type: ignore
    except Exception as error:
        raise ConfigurationError("pandas is not installed. Install pandas or provide a custom DatasetProvider.") from error
    return pd


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        pd = _pandas()
    except ConfigurationError:
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, bool):
        return missing
    return False


def _json_safe_value(value: Any) -> Any:
    if _is_missing_scalar(value):
        return None
    if isinstance(value, MappingABC):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return _json_safe_value(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        json.dumps(value, allow_nan=False)
        return value
    except (TypeError, ValueError):
        return str(value)


def _validate_dataframe_sample_id(value: Any, *, row_index: int, column: str) -> str:
    if _is_missing_scalar(value):
        raise ConfigurationError(f"DataFrame sample_id column {column!r} contains missing value at row {row_index}.")
    if not isinstance(value, str):
        raise ConfigurationError(
            f"DataFrame sample_id column {column!r} must contain string values; "
            f"row {row_index} has {type(value).__name__}."
        )
    if value == "":
        raise ConfigurationError(f"DataFrame sample_id column {column!r} contains empty string at row {row_index}.")
    return value


def _validate_dataframe_text(value: Any, *, sample_id: str, column: str) -> str:
    if _is_missing_scalar(value):
        raise ConfigurationError(f"DataFrame text column {column!r} contains missing value for sample_id={sample_id!r}.")
    if not isinstance(value, str):
        raise ConfigurationError(
            f"DataFrame text column {column!r} must contain string values; "
            f"sample_id={sample_id!r} has {type(value).__name__}."
        )
    return value


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
    - any additional columns are exposed in `DataSample.data` so public
      contracts such as `SplitConfig(mode="column")` can use them.

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
        if id_column not in df.columns or text_column not in df.columns:
            raise ConfigurationError(f"DataFrame must contain columns: {id_column!r}, {text_column!r}")
        self._df = df
        self._text_column = text_column
        self._id_column = id_column
        self._index: Dict[str, int] = {}
        self._sample_ids: List[str] = []
        for row_index, value in enumerate(df[id_column].tolist()):
            sample_id = _validate_dataframe_sample_id(value, row_index=row_index, column=id_column)
            if sample_id in self._index:
                raise ConfigurationError(f"Duplicate sample_id in dataset: {sample_id!r}")
            self._index[sample_id] = row_index
            self._sample_ids.append(sample_id)

    @classmethod
    def from_path(cls, path: Path) -> "DataFrameDatasetProvider":
        """Load CSV/Parquet via pandas and wrap as a provider."""
        pd = _pandas()
        if not path.exists():
            raise ConfigurationError(f"Dataset path does not exist: {path}")
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, dtype={"sample_id": "string"})
            return cls(df)
        if path.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
            return cls(df)
        raise ConfigurationError(f"Unsupported dataset file type: {path.suffix}")

    def iter_sample_ids(self) -> Iterator[str]:
        for sample_id in self._sample_ids:
            yield sample_id

    def get_sample(self, sample_id: str) -> DataSample:
        try:
            row_index = self._index[sample_id]
        except KeyError as error:
            raise KeyError(f"Unknown sample_id={sample_id!r}") from error

        row = self._df.iloc[row_index]
        data = {"text": _validate_dataframe_text(row[self._text_column], sample_id=sample_id, column=self._text_column)}
        reserved_columns = {self._id_column, "meta", "group_id"}
        for column in self._df.columns:
            if column in reserved_columns or column == self._text_column:
                continue
            data[str(column)] = _json_safe_value(row[column])
        meta: Dict[str, Any] = {}
        if "meta" in self._df.columns:
            meta_value = row["meta"]
            if isinstance(meta_value, str):
                try:
                    meta = json.loads(meta_value)
                except Exception:
                    meta = {"meta": meta_value}
            elif isinstance(meta_value, MappingABC):
                meta = dict(meta_value)
            else:
                meta = {"meta": meta_value}
            meta = _json_safe_value(meta)

        group_value = row["group_id"] if "group_id" in self._df.columns else None
        group_id = None if _is_missing_scalar(group_value) else str(_json_safe_value(group_value))
        return DataSample(sample_id=sample_id, data=data, meta=meta, group_id=group_id)

    def get_samples(self, sample_ids: Sequence[str]) -> List[DataSample]:
        return [self.get_sample(sample_id) for sample_id in sample_ids]

    def schema(self) -> Dict[str, str]:
        return {"sample_id": "str", "text": "str"}
