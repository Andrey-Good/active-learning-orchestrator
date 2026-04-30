from __future__ import annotations

from typing import Sequence

_PROBABILITY_SUPPORT_EPSILON = 1e-12


def label_count(context: object) -> int | None:
    label_schema = getattr(context, "label_schema", None)
    labels = getattr(label_schema, "labels", None)
    if labels is None:
        return None
    try:
        return len(list(labels))
    except TypeError:
        return None


def probability_support_count(probabilities: Sequence[Sequence[float]]) -> int:
    if not probabilities:
        return 0
    width = len(probabilities[0])
    supported = 0
    for column_index in range(width):
        if any(row[column_index] > _PROBABILITY_SUPPORT_EPSILON for row in probabilities):
            supported += 1
    return supported
