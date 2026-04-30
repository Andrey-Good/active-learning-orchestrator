"""BADGE selection over adapter-provided gradient embeddings."""

from __future__ import annotations


import math
from numbers import Real
from typing import Any, List, Sequence, TYPE_CHECKING

import numpy as np

from ..exceptions import ConfigurationError
from .embedding import MaxMinEmbeddingStrategy
from ._shared import label_count as _label_count
from ._shared import probability_support_count as _probability_support_count
from .uncertainty import RandomStrategy, _group_diverse_prefix, _normalize_probability_rows

if TYPE_CHECKING:
    from ..engine import SelectionContext


_COLD_START_SUPPORT_FRACTION = 0.5
_COLD_START_MIN_MISSING_LABELS = 3


def _unique_pool_ids(pool_ids: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for raw_sample_id in pool_ids:
        sample_id = str(raw_sample_id)
        if sample_id in seen:
            continue
        seen.add(sample_id)
        out.append(sample_id)
    return out


def _target_count(sample_ids: Sequence[str], k: int) -> int:
    if k <= 0 or not sample_ids:
        return 0
    return min(k, len(sample_ids))


def _normalize_gradient_embedding_rows(embeddings: Any, sample_ids: Sequence[str], *, strategy_name: str) -> np.ndarray:
    try:
        rows = list(embeddings)
    except TypeError as exc:
        raise ConfigurationError(f"{strategy_name}.gradient_embed output must be row-like.") from exc

    if len(rows) != len(sample_ids):
        raise ConfigurationError(
            f"{strategy_name}.gradient_embed returned {len(rows)} rows for {len(sample_ids)} sample ids."
        )

    normalized_rows: List[List[float]] = []
    expected_width: int | None = None
    for row_index, (sample_id, row) in enumerate(zip(sample_ids, rows)):
        if isinstance(row, (str, bytes)):
            raise ConfigurationError(
                f"{strategy_name}.gradient_embed row {row_index} for sample {sample_id!r} "
                "must be a sequence of numeric values."
            )
        try:
            values = list(row)
        except TypeError as exc:
            raise ConfigurationError(
                f"{strategy_name}.gradient_embed row {row_index} for sample {sample_id!r} must be a sequence."
            ) from exc

        if not values:
            raise ConfigurationError(
                f"{strategy_name}.gradient_embed row {row_index} for sample {sample_id!r} must not be empty."
            )

        if expected_width is None:
            expected_width = len(values)
        elif len(values) != expected_width:
            raise ConfigurationError(
                f"{strategy_name}.gradient_embed row {row_index} for sample {sample_id!r} has width {len(values)}; "
                f"expected {expected_width}."
            )

        cleaned: List[float] = []
        for column_index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ConfigurationError(
                    f"{strategy_name}.gradient_embed value at row {row_index}, column {column_index} must be numeric."
                )
            gradient_value = float(value)
            if not math.isfinite(gradient_value):
                raise ConfigurationError(
                    f"{strategy_name}.gradient_embed value at row {row_index}, column {column_index} must be finite."
                )
            cleaned.append(gradient_value)
        normalized_rows.append(cleaned)

    return np.asarray(normalized_rows, dtype=float)


def _gradient_embedding_matrix(context: "SelectionContext", sample_ids: Sequence[str], *, strategy_name: str) -> np.ndarray:
    if not sample_ids:
        return np.empty((0, 0), dtype=float)
    gradient_embed = getattr(context, "gradient_embed", None)
    if not callable(gradient_embed):
        raise ConfigurationError(f"Strategy {strategy_name!r} requires SelectionContext.gradient_embed.")
    return _normalize_gradient_embedding_rows(gradient_embed(sample_ids), sample_ids, strategy_name=strategy_name)


def _record_cold_start_diagnostic(
    context: "SelectionContext",
    *,
    effective_strategy: str,
    label_count: int,
    support_count: int,
    support_fraction: float,
    missing_label_count: int,
    fallback_mode: str,
    exploration_count: int,
    exploitation_count: int,
) -> None:
    recorder = getattr(context, "record_strategy_diagnostic", None)
    if not callable(recorder):
        return
    recorder(
        "badge",
        {
            "effective_strategy": effective_strategy,
            "fallback_reason": "cold_start_sparse_probability_support",
            "label_count": label_count,
            "support_count": support_count,
            "support_fraction": support_fraction,
            "missing_label_count": missing_label_count,
            "fallback_mode": fallback_mode,
            "exploration_count": exploration_count,
            "exploitation_count": exploitation_count,
        },
    )


def _merge_badge_cold_start_orders(
    exploration_order: Sequence[str],
    exploitation_order: Sequence[str],
    pool_ids: Sequence[str],
    k: int,
    *,
    exploration_count: int,
) -> List[str]:
    selected: List[str] = []
    selected_ids: set[str] = set()

    def append_from(order: Sequence[str], limit: int | None = None) -> None:
        for sample_id in order:
            if len(selected) >= k:
                return
            if limit is not None and len(selected) >= limit:
                return
            if sample_id in selected_ids:
                continue
            selected.append(sample_id)
            selected_ids.add(sample_id)

    append_from(exploration_order, exploration_count)
    append_from(exploitation_order)
    append_from(exploration_order)
    append_from(pool_ids)
    return selected


def _cold_start_probability_state(
    pool_ids: Sequence[str],
    context: "SelectionContext",
) -> dict[str, Any] | None:
    label_count = _label_count(context)
    predict_proba = getattr(context, "predict_proba", None)
    if label_count is None or label_count <= 0 or not callable(predict_proba):
        return None

    probabilities = _normalize_probability_rows(
        predict_proba(pool_ids),
        pool_ids,
        strategy_name="badge",
        label_count=label_count,
    )

    support_count = _probability_support_count(probabilities)
    missing_count = max(0, label_count - support_count)
    support_fraction = support_count / label_count
    if missing_count < _COLD_START_MIN_MISSING_LABELS or support_fraction >= _COLD_START_SUPPORT_FRACTION:
        return None
    return {
        "label_count": label_count,
        "support_count": support_count,
        "support_fraction": support_fraction,
        "missing_count": missing_count,
    }


def _cold_start_blended_selection(
    pool_ids: Sequence[str],
    k: int,
    context: "SelectionContext",
    exploitation_order: Sequence[str],
    cold_start_state: dict[str, Any],
) -> List[str] | None:
    label_count = int(cold_start_state["label_count"])
    support_count = int(cold_start_state["support_count"])
    support_fraction = float(cold_start_state["support_fraction"])
    missing_count = int(cold_start_state["missing_count"])

    target_k = min(k, len(pool_ids))
    if target_k <= 0:
        return []
    exploration_fraction = min(0.6, max(0.4, 1.0 - support_fraction))
    exploration_count = max(1, math.ceil(target_k * exploration_fraction))
    if target_k > 1:
        exploration_count = min(exploration_count, target_k - 1)
    exploitation_count = target_k - exploration_count

    embed = getattr(context, "embed", None)
    if callable(embed):
        try:
            exploration_order = MaxMinEmbeddingStrategy().select(pool_ids, len(pool_ids), context)
            selected = _merge_badge_cold_start_orders(
                exploration_order,
                exploitation_order,
                pool_ids,
                target_k,
                exploration_count=exploration_count,
            )
            selected = _group_diverse_prefix(
                [*selected, *exploitation_order, *exploration_order, *pool_ids],
                pool_ids,
                target_k,
                context,
                strategy_name="badge",
            )
            _record_cold_start_diagnostic(
                context,
                effective_strategy=f"cold_start_blend:{MaxMinEmbeddingStrategy.name}+badge",
                label_count=label_count,
                support_count=support_count,
                support_fraction=support_fraction,
                missing_label_count=missing_count,
                fallback_mode="blend",
                exploration_count=exploration_count,
                exploitation_count=exploitation_count,
            )
            return selected
        except ConfigurationError:
            pass
    exploration_order = RandomStrategy().select(pool_ids, target_k, context)
    selected = _merge_badge_cold_start_orders(
        exploration_order,
        exploitation_order,
        pool_ids,
        target_k,
        exploration_count=exploration_count,
    )
    selected = _group_diverse_prefix(
        [*selected, *exploitation_order, *exploration_order, *pool_ids],
        pool_ids,
        target_k,
        context,
        strategy_name="badge",
    )
    _record_cold_start_diagnostic(
        context,
        effective_strategy=f"cold_start_blend:{RandomStrategy.name}+badge",
        label_count=label_count,
        support_count=support_count,
        support_fraction=support_fraction,
        missing_label_count=missing_count,
        fallback_mode="blend",
        exploration_count=exploration_count,
        exploitation_count=exploitation_count,
    )
    return selected


def _squared_distance_to_center(matrix: np.ndarray, center: np.ndarray) -> np.ndarray:
    diff = matrix - center.reshape(1, -1)
    return np.einsum("ij,ij->i", diff, diff)


def _best_unselected_index(scores: Sequence[float] | np.ndarray, selected_indices: set[int]) -> int | None:
    best_index: int | None = None
    best_score = -math.inf
    for index, raw_score in enumerate(scores):
        if index in selected_indices:
            continue
        score = float(raw_score)
        if not math.isfinite(score):
            continue
        # Ties deliberately keep the earlier unique pool position for stable deterministic behavior.
        if best_index is None or score > best_score:
            best_index = index
            best_score = score
    return best_index


class BadgeStrategy:
    """Deterministic BADGE-style selection using adapter gradient embeddings.

    BADGE gradient embeddings are useful once the model has meaningful support
    across the task's label space. In many-class cold starts, adapters often can
    only produce probabilities/gradients for labels already seen in the seed
    data. In that regime, the strategy first explores by embedding novelty so it
    can discover classes before returning to gradient-space BADGE.
    """

    name = "badge"
    required_capabilities = frozenset({"gradient_embed"})

    def select(self, pool_ids: Sequence[str], k: int, context: "SelectionContext") -> List[str]:
        sample_ids = _unique_pool_ids(pool_ids)
        target_k = _target_count(sample_ids, k)
        if target_k <= 0:
            return []

        cold_start_state = _cold_start_probability_state(sample_ids, context)

        # BADGE requires gradient embeddings. The SDK does not synthesize them from logits;
        # adapters own pseudo-label and gradient-vector details for their model family.
        matrix = _gradient_embedding_matrix(context, sample_ids, strategy_name=self.name)
        selected_indices: set[int] = set()
        selected: List[str] = []

        first_scores = np.einsum("ij,ij->i", matrix, matrix)
        first_index = _best_unselected_index(first_scores, selected_indices)
        if first_index is None:
            return []

        selected_indices.add(first_index)
        selected.append(sample_ids[first_index])
        min_distances = _squared_distance_to_center(matrix, matrix[first_index])

        while len(selected) < target_k:
            index = _best_unselected_index(min_distances, selected_indices)
            if index is None:
                break

            selected_indices.add(index)
            selected.append(sample_ids[index])
            distances_to_new_center = _squared_distance_to_center(matrix, matrix[index])
            min_distances = np.minimum(min_distances, distances_to_new_center)

        if cold_start_state is not None:
            cold_start_selection = _cold_start_blended_selection(
                sample_ids,
                target_k,
                context,
                selected,
                cold_start_state,
            )
            if cold_start_selection is not None:
                return cold_start_selection

        return _group_diverse_prefix([*selected, *sample_ids], sample_ids, target_k, context, strategy_name=self.name)
