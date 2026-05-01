"""Validation helpers for ``SelectionContext`` model output caches."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, List, Optional, Sequence

from ..exceptions import ConfigurationError


def validate_predict_proba_rows(
    rows: Sequence[Any],
    sample_ids: Sequence[str],
    *,
    label_width: Optional[int],
) -> List[List[float]]:
    if len(rows) != len(sample_ids):
        raise ConfigurationError(f"model.predict_proba returned {len(rows)} rows for {len(sample_ids)} sample ids.")

    expected_width: Optional[int] = None
    cleaned_rows: List[List[float]] = []
    for row_index, (sample_id, row) in enumerate(zip(sample_ids, rows)):
        values = _coerce_row_values(
            row,
            row_index=row_index,
            sample_id=sample_id,
            method_name="predict_proba",
            non_sequence_suffix=" of probabilities",
        )
        _validate_probability_width(values, row_index, sample_id, label_width, expected_width)
        if expected_width is None:
            expected_width = len(values)
        cleaned_rows.append(_validate_probability_values(values, row_index, sample_id))
    return cleaned_rows


def validate_embedding_rows(rows: Sequence[Any], sample_ids: Sequence[str]) -> List[List[float]]:
    if len(rows) != len(sample_ids):
        raise ConfigurationError(f"model.embed returned {len(rows)} rows for {len(sample_ids)} sample ids.")

    expected_width: Optional[int] = None
    cleaned_rows: List[List[float]] = []
    for row_index, (sample_id, row) in enumerate(zip(sample_ids, rows)):
        values = _coerce_row_values(
            row,
            row_index=row_index,
            sample_id=sample_id,
            method_name="embed",
            non_sequence_suffix=" of numeric values",
        )
        if expected_width is None:
            expected_width = len(values)
        elif len(values) != expected_width:
            raise ConfigurationError(
                f"model.embed row {row_index} for sample {sample_id!r} has width {len(values)}; "
                f"expected {expected_width}."
            )
        cleaned_rows.append(_validate_embedding_values(values, row_index, sample_id))
    return cleaned_rows


def _coerce_row_values(
    row: Any,
    *,
    row_index: int,
    sample_id: str,
    method_name: str,
    non_sequence_suffix: str,
) -> List[Any]:
    if isinstance(row, (str, bytes)):
        raise ConfigurationError(
            f"model.{method_name} row {row_index} for sample {sample_id!r} must be a sequence{non_sequence_suffix}."
        )
    try:
        values = list(row)
    except TypeError as exc:
        raise ConfigurationError(
            f"model.{method_name} row {row_index} for sample {sample_id!r} must be a sequence."
        ) from exc
    if not values:
        raise ConfigurationError(f"model.{method_name} row {row_index} for sample {sample_id!r} must not be empty.")
    return values


def _validate_probability_width(
    values: Sequence[Any],
    row_index: int,
    sample_id: str,
    label_width: Optional[int],
    expected_width: Optional[int],
) -> None:
    if len(values) < 2:
        raise ConfigurationError(
            f"model.predict_proba row {row_index} for sample {sample_id!r} "
            "must have at least 2 probability columns."
        )
    if label_width is not None and len(values) != label_width:
        raise ConfigurationError(
            f"model.predict_proba row {row_index} for sample {sample_id!r} has width {len(values)}; "
            f"label_schema.labels defines {label_width} labels."
        )
    if expected_width is not None and len(values) != expected_width:
        raise ConfigurationError(
            f"model.predict_proba row {row_index} for sample {sample_id!r} has width {len(values)}; "
            f"expected {expected_width}."
        )


def _validate_probability_values(values: Sequence[Any], row_index: int, sample_id: str) -> List[float]:
    probabilities: List[float] = []
    for column_index, value in enumerate(values):
        probability = _coerce_finite_number(
            value,
            method_name="predict_proba",
            row_index=row_index,
            column_index=column_index,
            sample_id=sample_id,
        )
        if probability < 0:
            raise ConfigurationError(
                f"model.predict_proba row {row_index}, column {column_index} for sample {sample_id!r} "
                "must be non-negative."
            )
        probabilities.append(probability)

    row_sum = sum(probabilities)
    if row_sum <= 0:
        raise ConfigurationError(
            f"model.predict_proba row {row_index} for sample {sample_id!r} must have a positive sum."
        )
    if not math.isclose(row_sum, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ConfigurationError(
            f"model.predict_proba row {row_index} for sample {sample_id!r} must sum to 1.0; "
            f"got {row_sum}."
        )
    return probabilities


def _validate_embedding_values(values: Sequence[Any], row_index: int, sample_id: str) -> List[float]:
    embedding_values: List[float] = []
    for column_index, value in enumerate(values):
        embedding_values.append(
            _coerce_finite_number(
                value,
                method_name="embed",
                row_index=row_index,
                column_index=column_index,
                sample_id=sample_id,
            )
        )
    return embedding_values


def _coerce_finite_number(
    value: Any,
    *,
    method_name: str,
    row_index: int,
    column_index: int,
    sample_id: str,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, Real)):
        raise ConfigurationError(
            f"model.{method_name} row {row_index}, column {column_index} for sample {sample_id!r} must be numeric."
        )
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ConfigurationError(
            f"model.{method_name} row {row_index}, column {column_index} for sample {sample_id!r} must be finite."
        )
    return numeric_value
