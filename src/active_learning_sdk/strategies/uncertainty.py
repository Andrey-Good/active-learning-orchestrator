"""
Built-in sampling strategies (MVP).

These strategies implement common "uncertainty sampling" heuristics:
- entropy: pick items with highest prediction entropy
- margin: pick items where top-2 classes are close
- least_confidence: pick items with low max probability
- random: baseline
"""

from __future__ import annotations


import hashlib
import math
from numbers import Real
from typing import Any, List, Sequence, Tuple, TYPE_CHECKING

from ..exceptions import ConfigurationError
from .embedding import (
    DeduplicateNearNeighborsStrategy,
    DensityWeightedDiversityStrategy,
    EmbeddingKMeansPPStrategy,
    KCenterGreedyStrategy,
    MaxMinEmbeddingStrategy,
)
from ._shared import label_count as _label_count
from ._shared import probability_support_count as _probability_support_count

if TYPE_CHECKING:
    from ..engine import SelectionContext


_PROBABILITY_SUM_REL_TOL = 1e-9
_PROBABILITY_SUM_ABS_TOL = 1e-12
_COLD_START_SUPPORT_FRACTION = 0.95
_COLD_START_MIN_MISSING_LABELS = 1
_SMALL_SEED_MIN_LABELS_PER_CLASS = 8
_SMALL_SEED_MIN_TOTAL_LABELS = 24


def _stable_hash_parts(*parts: object) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _context_model_id(context: "SelectionContext") -> str:
    model_id = getattr(context, "model_id", None)
    if callable(model_id):
        return str(model_id())
    return "unknown"


def _tie_key(strategy_name: str, model_id: str, sample_id: str) -> str:
    return _stable_hash_parts("tie", strategy_name, model_id, sample_id)


def _pool_key(pool_ids: Sequence[str]) -> str:
    return _stable_hash_parts("pool", *sorted(str(sample_id) for sample_id in pool_ids))


def _is_real_number(value: Any) -> bool:
    return not isinstance(value, bool) and (isinstance(value, (int, float)) or isinstance(value, Real))


def _unique_pool_ids(pool_ids: Sequence[str]) -> List[str]:
    seen = set()
    unique_ids: List[str] = []
    for sample_id in pool_ids:
        if sample_id in seen:
            continue
        seen.add(sample_id)
        unique_ids.append(sample_id)
    return unique_ids


def _select_top_scored(
    scored: Sequence[Tuple[str, float]],
    k: int,
    *,
    strategy_name: str,
    context: "SelectionContext",
) -> List[str]:
    if k <= 0 or not scored:
        return []

    ordered_by_score = sorted(scored, key=lambda pair: -pair[1])
    if len(ordered_by_score) <= 1:
        return [sample_id for sample_id, _ in ordered_by_score[:k]]

    selected: List[str] = []
    model_id: str | None = None
    index = 0
    while index < len(ordered_by_score) and len(selected) < k:
        score = ordered_by_score[index][1]
        end = index + 1
        while end < len(ordered_by_score) and ordered_by_score[end][1] == score:
            end += 1

        score_group = ordered_by_score[index:end]
        if len(score_group) > 1:
            if model_id is None:
                model_id = _context_model_id(context)
            score_group = sorted(
                score_group,
                key=lambda pair: (_tie_key(strategy_name, model_id or "unknown", pair[0]), pair[0]),
            )

        for sample_id, _ in score_group:
            selected.append(sample_id)
            if len(selected) >= k:
                break
        index = end

    return selected


def _entropy_scores(
    pool_ids: Sequence[str],
    probabilities: Sequence[Sequence[float]],
) -> List[Tuple[str, float]]:
    scored: List[Tuple[str, float]] = []
    for sample_id, probability in zip(pool_ids, probabilities):
        entropy = 0.0
        for prob in probability:
            if prob > 0:
                entropy -= prob * math.log(prob)
        scored.append((sample_id, entropy))
    return scored


def _record_cold_start_diagnostic(
    context: "SelectionContext",
    *,
    strategy_name: str,
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
        strategy_name,
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


def _record_small_seed_diagnostic(
    context: "SelectionContext",
    *,
    strategy_name: str,
    effective_strategy: str,
    label_count: int,
    labeled_count: int,
    threshold: int,
    exploration_count: int,
    exploitation_count: int,
) -> None:
    recorder = getattr(context, "record_strategy_diagnostic", None)
    if not callable(recorder):
        return
    recorder(
        strategy_name,
        {
            "effective_strategy": effective_strategy,
            "fallback_reason": "small_seed_diversity_exploration",
            "label_count": label_count,
            "labeled_count": labeled_count,
            "small_seed_threshold": threshold,
            "fallback_mode": "blend",
            "exploration_count": exploration_count,
            "exploitation_count": exploitation_count,
        },
    )


def _labeled_count(context: "SelectionContext") -> int | None:
    labeled_ids = getattr(context, "labeled_ids", None)
    if labeled_ids is None:
        return None
    try:
        return len(_unique_pool_ids(labeled_ids))
    except TypeError:
        return None


def _cold_start_blended_selection(
    pool_ids: Sequence[str],
    k: int,
    context: "SelectionContext",
    probabilities: Sequence[Sequence[float]],
    *,
    strategy_name: str,
    exploitation_order: Sequence[str],
    exploration_strategy: Any | None = None,
) -> List[str] | None:
    """Blend exploration with the requested heuristic in sparse-support cold starts.

    In cold-start many-class classification, model adapters often expose a full
    label schema while the fitted classifier only has probability support for
    labels already seen in the seed set. Pure uncertainty then over-samples
    known-class boundaries and under-discovers unseen classes. Full substitution
    makes unrelated strategies alias each other, so this guardrail keeps a
    smaller exploitation slice from the requested heuristic in every non-trivial
    batch.
    """

    label_count = _label_count(context)
    if label_count is None or label_count <= 0:
        return None

    support_count = _probability_support_count(probabilities)
    missing_count = max(0, label_count - support_count)
    support_fraction = support_count / label_count
    if missing_count < _COLD_START_MIN_MISSING_LABELS or support_fraction >= _COLD_START_SUPPORT_FRACTION:
        return None

    target_k = min(k, len(set(str(sample_id) for sample_id in pool_ids)))
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
            explorer = exploration_strategy or KCenterGreedyStrategy()
            effective_strategy = str(getattr(explorer, "name", explorer.__class__.__name__))
            exploration_order = explorer.select(pool_ids, target_k, context)
            selected = _group_diverse_merged_selection(
                exploration_order,
                exploitation_order,
                pool_ids,
                target_k,
                context,
                strategy_name=strategy_name,
                first_order_count=exploration_count,
            )
            _record_cold_start_diagnostic(
                context,
                strategy_name=strategy_name,
                effective_strategy=f"cold_start_blend:{effective_strategy}+{strategy_name}",
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
    effective_strategy = RandomStrategy.name
    exploration_order = RandomStrategy().select(pool_ids, target_k, context)
    selected = _group_diverse_merged_selection(
        exploration_order,
        exploitation_order,
        pool_ids,
        target_k,
        context,
        strategy_name=strategy_name,
        first_order_count=exploration_count,
    )
    _record_cold_start_diagnostic(
        context,
        strategy_name=strategy_name,
        effective_strategy=f"cold_start_blend:{effective_strategy}+{strategy_name}",
        label_count=label_count,
        support_count=support_count,
        support_fraction=support_fraction,
        missing_label_count=missing_count,
        fallback_mode="blend",
        exploration_count=exploration_count,
        exploitation_count=exploitation_count,
    )
    return selected


def _small_seed_diversity_selection(
    pool_ids: Sequence[str],
    k: int,
    context: "SelectionContext",
    *,
    strategy_name: str,
    exploitation_order: Sequence[str],
    exploration_strategy: Any,
) -> List[str] | None:
    label_count = _label_count(context)
    labeled_count = _labeled_count(context)
    if label_count is None or label_count <= 0 or labeled_count is None or labeled_count <= 0:
        return None

    threshold = max(_SMALL_SEED_MIN_TOTAL_LABELS, label_count * _SMALL_SEED_MIN_LABELS_PER_CLASS)
    if labeled_count >= threshold:
        return None

    target_k = min(k, len(set(str(sample_id) for sample_id in pool_ids)))
    if target_k <= 1 or not callable(getattr(context, "embed", None)):
        return None

    early_fraction = 0.5 if labeled_count < (threshold / 2) else 0.35
    exploration_count = max(1, math.ceil(target_k * early_fraction))
    exploration_count = min(exploration_count, target_k - 1)
    exploitation_count = target_k - exploration_count

    try:
        exploration_order = exploration_strategy.select(pool_ids, len(pool_ids), context)
    except ConfigurationError:
        return None

    selected = _group_diverse_merged_selection(
        exploration_order,
        exploitation_order,
        pool_ids,
        target_k,
        context,
        strategy_name=strategy_name,
        first_order_count=exploration_count,
    )
    effective_strategy = str(getattr(exploration_strategy, "name", exploration_strategy.__class__.__name__))
    _record_small_seed_diagnostic(
        context,
        strategy_name=strategy_name,
        effective_strategy=f"small_seed_blend:{effective_strategy}+{strategy_name}",
        label_count=label_count,
        labeled_count=labeled_count,
        threshold=threshold,
        exploration_count=exploration_count,
        exploitation_count=exploitation_count,
    )
    return selected


def _merge_selection_orders(
    first_order: Sequence[str],
    second_order: Sequence[str],
    pool_ids: Sequence[str],
    k: int,
    *,
    first_order_count: int,
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

    append_from(first_order, first_order_count)
    append_from(second_order)
    append_from(first_order)
    append_from(pool_ids)
    return selected


def _group_diverse_merged_selection(
    first_order: Sequence[str],
    second_order: Sequence[str],
    pool_ids: Sequence[str],
    k: int,
    context: "SelectionContext",
    *,
    strategy_name: str,
    first_order_count: int,
) -> List[str]:
    selected = _merge_selection_orders(
        first_order,
        second_order,
        pool_ids,
        k,
        first_order_count=first_order_count,
    )
    return _group_diverse_prefix(
        [*selected, *second_order, *first_order, *pool_ids],
        pool_ids,
        k,
        context,
        strategy_name=strategy_name,
    )


def _group_diverse_prefix(
    ordered_ids: Sequence[str],
    pool_ids: Sequence[str],
    k: int,
    context: "SelectionContext",
    *,
    strategy_name: str,
) -> List[str]:
    if k <= 0:
        return []
    if not callable(getattr(context, "get_samples", None)):
        return _unique_pool_ids(ordered_ids)[: min(k, len(set(str(sample_id) for sample_id in pool_ids)))]

    groups_by_id = _groups_by_id(strategy_name, pool_ids, context)
    selected: List[str] = []
    selected_ids: set[str] = set()
    selected_groups = _labeled_group_keys(strategy_name, context)
    target_k = min(k, len(set(str(sample_id) for sample_id in pool_ids)))

    for sample_id in ordered_ids:
        if len(selected) >= target_k:
            return selected
        if sample_id in selected_ids:
            continue
        group_key = groups_by_id.get(str(sample_id), ("sample", str(sample_id)))
        if group_key in selected_groups:
            continue
        selected.append(sample_id)
        selected_ids.add(sample_id)
        selected_groups.add(group_key)

    for sample_id in ordered_ids:
        if len(selected) >= target_k:
            return selected
        if sample_id in selected_ids:
            continue
        selected.append(sample_id)
        selected_ids.add(sample_id)
    return selected


def _normalize_probability_rows(
    probabilities: Any,
    pool_ids: Sequence[str],
    *,
    strategy_name: str,
    label_count: int | None = None,
) -> List[List[float]]:
    try:
        rows = list(probabilities)
    except TypeError as exc:
        raise ConfigurationError(f"{strategy_name}.predict_proba output must be row-like.") from exc

    if len(rows) != len(pool_ids):
        raise ConfigurationError(
            f"{strategy_name}.predict_proba returned {len(rows)} rows for {len(pool_ids)} pool ids."
        )

    normalized_rows: List[List[float]] = []
    expected_width: int | None = None
    for row_index, (sample_id, row) in enumerate(zip(pool_ids, rows)):
        if isinstance(row, (str, bytes)):
            raise ConfigurationError(
                f"{strategy_name}.predict_proba row {row_index} for sample {sample_id!r} must be a sequence of numeric probabilities."
            )
        try:
            values = list(row)
        except TypeError as exc:
            raise ConfigurationError(
                f"{strategy_name}.predict_proba row {row_index} for sample {sample_id!r} must be a sequence."
            ) from exc

        if not values:
            raise ConfigurationError(
                f"{strategy_name}.predict_proba row {row_index} for sample {sample_id!r} must not be empty."
            )
        if len(values) < 2:
            raise ConfigurationError(
                f"{strategy_name}.predict_proba row {row_index} for sample {sample_id!r} "
                "must have at least 2 probability columns."
            )
        if expected_width is None:
            expected_width = len(values)
            if label_count is not None and expected_width != label_count:
                raise ConfigurationError(
                    f"{strategy_name}.predict_proba returned {expected_width} probability columns, "
                    f"but label_schema.labels defines {label_count} labels."
                )
        elif len(values) != expected_width:
            raise ConfigurationError(
                f"{strategy_name}.predict_proba row {row_index} for sample {sample_id!r} has width {len(values)}; "
                f"expected {expected_width}."
            )

        cleaned: List[float] = []
        for column_index, value in enumerate(values):
            if not _is_real_number(value):
                raise ConfigurationError(
                    f"{strategy_name}.predict_proba value at row {row_index}, column {column_index} must be numeric."
                )
            probability = float(value)
            if not math.isfinite(probability):
                raise ConfigurationError(
                    f"{strategy_name}.predict_proba value at row {row_index}, column {column_index} must be finite."
                )
            if probability < 0:
                raise ConfigurationError(
                    f"{strategy_name}.predict_proba value at row {row_index}, column {column_index} must be non-negative."
                )
            cleaned.append(probability)

        row_sum = sum(cleaned)
        if row_sum <= 0:
            raise ConfigurationError(
                f"{strategy_name}.predict_proba row {row_index} for sample {sample_id!r} must have a positive sum."
            )
        if not math.isclose(row_sum, 1.0, rel_tol=_PROBABILITY_SUM_REL_TOL, abs_tol=_PROBABILITY_SUM_ABS_TOL):
            raise ConfigurationError(
                f"{strategy_name}.predict_proba row {row_index} for sample {sample_id!r} must sum to 1.0; "
                f"got {row_sum}."
            )
        normalized_rows.append(cleaned)

    return normalized_rows


def _probability_rows(
    context: "SelectionContext",
    pool_ids: Sequence[str],
    *,
    strategy_name: str,
) -> List[List[float]]:
    return _normalize_probability_rows(
        context.predict_proba(pool_ids),
        pool_ids,
        strategy_name=strategy_name,
        label_count=_label_count(context),
    )


class RandomStrategy:
    """
    Pick k random ids from the pool.

    Attributes:
        name (str):
            Where: used by `StrategyScheduler._get_strategy()` and by user configs like `SchedulerConfig(strategy=...)`.
            What: stable string identifier for this strategy ("random").
            Why: lets the engine select strategies by name and makes configs serializable.
    """
    name = "random"
    required_capabilities: frozenset[str] = frozenset()

    def select(self, pool_ids: Sequence[str], k: int, context: "SelectionContext") -> List[str]:
        if k <= 0:
            return []
        pool_ids = _unique_pool_ids(pool_ids)
        if len(pool_ids) <= k:
            return list(pool_ids)
        model_id = _context_model_id(context)
        pool_key = _pool_key(pool_ids)
        ordered = sorted(
            pool_ids,
            key=lambda sample_id: (_stable_hash_parts(self.name, model_id, pool_key, sample_id), str(sample_id)),
        )
        return ordered[:k]


class EntropyStrategy:
    """
    Pick items with the highest entropy of predicted probabilities.

    Higher entropy means the model is "more unsure".

    Attributes:
        name (str):
            Where: used by scheduler and configs to refer to this strategy.
            What: stable string identifier ("entropy").
            Why: ensures the strategy can be selected without importing the class directly.
    """
    name = "entropy"
    required_capabilities = frozenset({"predict_proba"})

    def select(self, pool_ids: Sequence[str], k: int, context: "SelectionContext") -> List[str]:
        if k <= 0 or not pool_ids:
            return []
        pool_ids = _unique_pool_ids(pool_ids)
        probabilities = _probability_rows(context, pool_ids, strategy_name=self.name)
        scored = _entropy_scores(pool_ids, probabilities)
        scored_order = _select_top_scored(scored, len(pool_ids), strategy_name=self.name, context=context)
        exploitation_order = _group_diverse_prefix(
            scored_order,
            pool_ids,
            k,
            context,
            strategy_name=self.name,
        )
        cold_start_selection = _cold_start_blended_selection(
            pool_ids,
            k,
            context,
            probabilities,
            strategy_name=self.name,
            exploitation_order=exploitation_order,
            exploration_strategy=KCenterGreedyStrategy(),
        )
        if cold_start_selection is not None:
            return cold_start_selection
        small_seed_selection = _small_seed_diversity_selection(
            pool_ids,
            k,
            context,
            strategy_name=self.name,
            exploitation_order=exploitation_order,
            exploration_strategy=KCenterGreedyStrategy(),
        )
        if small_seed_selection is not None:
            return small_seed_selection
        return exploitation_order


def _predicted_class_buckets(
    scored: Sequence[Tuple[str, float]],
    probabilities: Sequence[Sequence[float]],
    *,
    strategy_name: str,
    model_id: str,
) -> tuple[dict[int, List[Tuple[str, float]]], List[int]]:
    buckets: dict[int, List[Tuple[str, float]]] = {}
    for (sample_id, entropy), probability in zip(scored, probabilities):
        predicted_class = max(
            range(len(probability)),
            key=lambda index: (probability[index], -index),
        )
        buckets.setdefault(predicted_class, []).append((sample_id, entropy))

    for predicted_class, bucket in buckets.items():
        buckets[predicted_class] = sorted(
            bucket,
            key=lambda pair: (-pair[1], _tie_key(strategy_name, model_id, pair[0]), pair[0]),
        )

    class_order = sorted(
        buckets.keys(),
        key=lambda predicted_class: (
            -buckets[predicted_class][0][1],
            predicted_class,
        ),
    )
    return buckets, class_order


def _class_balanced_order(
    buckets: dict[int, List[Tuple[str, float]]],
    class_order: Sequence[int],
    k: int,
    pool_ids: Sequence[str],
) -> List[str]:
    selected: List[str] = []
    selected_ids = set()
    positions = {predicted_class: 0 for predicted_class in class_order}
    target_k = min(k, len(set(str(sample_id) for sample_id in pool_ids)))

    while len(selected) < target_k:
        added_this_round = False
        for predicted_class in class_order:
            bucket = buckets[predicted_class]
            position = positions[predicted_class]
            while position < len(bucket) and bucket[position][0] in selected_ids:
                position += 1
            positions[predicted_class] = position
            if position >= len(bucket):
                continue

            sample_id = bucket[position][0]
            selected.append(sample_id)
            selected_ids.add(sample_id)
            positions[predicted_class] = position + 1
            added_this_round = True
            if len(selected) >= target_k:
                break

        if not added_this_round:
            break

    return selected


def _groups_by_id(strategy_name: str, pool_ids: Sequence[str], context: "SelectionContext") -> dict[str, tuple[str, str]]:
    samples = context.get_samples(pool_ids)
    if len(samples) != len(pool_ids):
        raise ConfigurationError(
            f"{strategy_name}.get_samples returned {len(samples)} samples for {len(pool_ids)} pool ids."
        )

    returned_ids = [str(getattr(sample, "sample_id", "")) for sample in samples]
    expected_ids = [str(sample_id) for sample_id in pool_ids]
    if len(set(returned_ids)) != len(returned_ids):
        raise ConfigurationError(f"{strategy_name}.get_samples returned duplicate sample_id values.")
    missing_ids = sorted(set(expected_ids) - set(returned_ids))
    foreign_ids = sorted(set(returned_ids) - set(expected_ids))
    if missing_ids or foreign_ids:
        raise ConfigurationError(
            f"{strategy_name}.get_samples returned sample_id values that do not match requested pool ids; "
            f"missing={missing_ids}, foreign={foreign_ids}."
        )
    if returned_ids != expected_ids:
        raise ConfigurationError(
            f"{strategy_name}.get_samples must return samples in the same sample_id order as requested."
        )

    groups_by_id = {}
    for sample_id, sample in zip(expected_ids, samples):
        group_id = getattr(sample, "group_id", None)
        if group_id is None:
            groups_by_id[sample_id] = ("sample", sample_id)
        else:
            groups_by_id[sample_id] = ("group", str(group_id))
    return groups_by_id


def _labeled_group_keys(strategy_name: str, context: "SelectionContext") -> set[tuple[str, str]]:
    labeled_ids = _unique_pool_ids(getattr(context, "labeled_ids", []))
    if not labeled_ids or not callable(getattr(context, "get_samples", None)):
        return set()
    return set(_groups_by_id(f"{strategy_name}.labeled_ids", labeled_ids, context).values())


class ClassBalancedEntropyStrategy:
    """
    Pick high-entropy items while round-robin balancing predicted classes.

    Each sample is assigned to the class with the highest predicted probability.
    Within a predicted class, samples are ranked by entropy. The final batch is
    assembled one item per predicted class per pass when multiple classes have
    candidates, so one predicted class cannot monopolize the batch until other
    classes are exhausted.
    """

    name = "class_balanced_entropy"
    required_capabilities = frozenset({"predict_proba"})

    def select(self, pool_ids: Sequence[str], k: int, context: "SelectionContext") -> List[str]:
        if k <= 0 or not pool_ids:
            return []
        pool_ids = _unique_pool_ids(pool_ids)

        probabilities = _probability_rows(context, pool_ids, strategy_name=self.name)
        scored = _entropy_scores(pool_ids, probabilities)
        model_id = _context_model_id(context)

        buckets, class_order = _predicted_class_buckets(
            scored,
            probabilities,
            strategy_name=self.name,
            model_id=model_id,
        )
        exploitation_order = _class_balanced_order(buckets, class_order, k, pool_ids)
        cold_start_selection = _cold_start_blended_selection(
            pool_ids,
            k,
            context,
            probabilities,
            strategy_name=self.name,
            exploitation_order=exploitation_order,
            exploration_strategy=DensityWeightedDiversityStrategy(),
        )
        if cold_start_selection is not None:
            return cold_start_selection
        return exploitation_order


class GroupDiverseEntropyStrategy:
    """
    Pick high-entropy items while avoiding repeated groups when possible.

    The strategy first ranks the pool by entropy using deterministic tie-breaking,
    then greedily takes at most one item per group. If k is larger than the number
    of groups, it fills the rest from the same entropy ranking.
    """

    name = "group_diverse_entropy"
    required_capabilities = frozenset({"predict_proba"})

    def select(self, pool_ids: Sequence[str], k: int, context: "SelectionContext") -> List[str]:
        if k <= 0 or not pool_ids:
            return []
        pool_ids = _unique_pool_ids(pool_ids)

        probabilities = _probability_rows(context, pool_ids, strategy_name=self.name)
        scored = _entropy_scores(pool_ids, probabilities)
        groups_by_id = _groups_by_id(self.name, pool_ids, context)

        model_id = _context_model_id(context)
        ordered = sorted(
            scored,
            key=lambda pair: (-pair[1], _tie_key(self.name, model_id, pair[0]), pair[0]),
        )

        selected: List[str] = []
        selected_ids = set()
        selected_groups = _labeled_group_keys(self.name, context)
        target_k = min(k, len(set(str(sample_id) for sample_id in pool_ids)))

        for sample_id, _ in ordered:
            if len(selected) >= target_k:
                break
            if sample_id in selected_ids:
                continue
            group_key = groups_by_id[sample_id]
            if group_key in selected_groups:
                continue
            selected.append(sample_id)
            selected_ids.add(sample_id)
            selected_groups.add(group_key)

        for sample_id, _ in ordered:
            if len(selected) >= target_k:
                break
            if sample_id in selected_ids:
                continue
            selected.append(sample_id)
            selected_ids.add(sample_id)
        cold_start_selection = _cold_start_blended_selection(
            pool_ids,
            k,
            context,
            probabilities,
            strategy_name=self.name,
            exploitation_order=selected,
            exploration_strategy=DeduplicateNearNeighborsStrategy(),
        )
        if cold_start_selection is not None:
            return cold_start_selection

        return selected


class ClassGroupBalancedEntropyStrategy:
    """
    Pick high-entropy items with predicted-class balance and group diversity.

    The strategy round-robins predicted classes like class_balanced_entropy. For
    each class turn it prefers the highest-entropy candidate whose group has not
    already been selected. When unique groups are exhausted, it fills remaining
    slots with the deterministic class-balanced entropy order.
    """

    name = "class_group_balanced_entropy"
    required_capabilities = frozenset({"predict_proba"})

    def select(self, pool_ids: Sequence[str], k: int, context: "SelectionContext") -> List[str]:
        if k <= 0 or not pool_ids:
            return []
        pool_ids = _unique_pool_ids(pool_ids)

        probabilities = _probability_rows(context, pool_ids, strategy_name=self.name)
        scored = _entropy_scores(pool_ids, probabilities)
        model_id = _context_model_id(context)
        buckets, class_order = _predicted_class_buckets(
            scored,
            probabilities,
            strategy_name=self.name,
            model_id=model_id,
        )
        target_k = min(k, len(set(str(sample_id) for sample_id in pool_ids)))
        class_balanced_order = _class_balanced_order(buckets, class_order, target_k, pool_ids)
        groups_by_id = _groups_by_id(self.name, pool_ids, context)

        selected: List[str] = []
        selected_ids = set()
        selected_groups = _labeled_group_keys(self.name, context)

        while len(selected) < target_k:
            added_this_round = False
            for predicted_class in class_order:
                candidate = None
                for sample_id, _ in buckets[predicted_class]:
                    if sample_id in selected_ids:
                        continue
                    group_key = groups_by_id[sample_id]
                    if group_key in selected_groups:
                        continue
                    candidate = sample_id
                    break
                if candidate is None:
                    continue

                selected.append(candidate)
                selected_ids.add(candidate)
                selected_groups.add(groups_by_id[candidate])
                added_this_round = True
                if len(selected) >= target_k:
                    return selected

            if not added_this_round:
                break

        for sample_id in class_balanced_order:
            if len(selected) >= target_k:
                break
            if sample_id in selected_ids:
                continue
            selected.append(sample_id)
            selected_ids.add(sample_id)

        cold_start_selection = _cold_start_blended_selection(
            pool_ids,
            k,
            context,
            probabilities,
            strategy_name=self.name,
            exploitation_order=selected,
            exploration_strategy=EmbeddingKMeansPPStrategy(),
        )
        if cold_start_selection is not None:
            return cold_start_selection

        return selected


class LeastConfidenceStrategy:
    """
    Pick items where the model is least confident (1 - max_proba).

    Attributes:
        name (str):
            Where: used by scheduler and configs to refer to this strategy.
            What: stable string identifier ("least_confidence").
            Why: makes it possible to choose this heuristic via config.
    """
    name = "least_confidence"
    required_capabilities = frozenset({"predict_proba"})

    def select(self, pool_ids: Sequence[str], k: int, context: "SelectionContext") -> List[str]:
        if k <= 0 or not pool_ids:
            return []
        pool_ids = _unique_pool_ids(pool_ids)
        probabilities = _probability_rows(context, pool_ids, strategy_name=self.name)
        scored: List[Tuple[str, float]] = []
        for sample_id, probability in zip(pool_ids, probabilities):
            score = 1.0 - max(probability)
            scored.append((sample_id, score))
        scored_order = _select_top_scored(scored, len(pool_ids), strategy_name=self.name, context=context)
        exploitation_order = _group_diverse_prefix(
            scored_order,
            pool_ids,
            k,
            context,
            strategy_name=self.name,
        )
        cold_start_selection = _cold_start_blended_selection(
            pool_ids,
            k,
            context,
            probabilities,
            strategy_name=self.name,
            exploitation_order=exploitation_order,
            exploration_strategy=MaxMinEmbeddingStrategy(),
        )
        if cold_start_selection is not None:
            return cold_start_selection
        return exploitation_order


class MarginStrategy:
    """
    Pick items with the smallest margin between top-2 probabilities.

    Small margin means the model is torn between two classes.

    Attributes:
        name (str):
            Where: used by scheduler and configs to refer to this strategy.
            What: stable string identifier ("margin").
            Why: makes this heuristic selectable via `SchedulerConfig`.
    """
    name = "margin"
    required_capabilities = frozenset({"predict_proba"})

    def select(self, pool_ids: Sequence[str], k: int, context: "SelectionContext") -> List[str]:
        if k <= 0 or not pool_ids:
            return []
        pool_ids = _unique_pool_ids(pool_ids)
        probabilities = _probability_rows(context, pool_ids, strategy_name=self.name)
        scored: List[Tuple[str, float]] = []
        for sample_id, probability in zip(pool_ids, probabilities):
            probs = sorted(probability, reverse=True)
            if len(probs) >= 2:
                margin = probs[0] - probs[1]
            elif len(probs) == 1:
                margin = probs[0]
            scored.append((sample_id, -margin))
        scored_order = _select_top_scored(scored, len(pool_ids), strategy_name=self.name, context=context)
        exploitation_order = _group_diverse_prefix(
            scored_order,
            pool_ids,
            k,
            context,
            strategy_name=self.name,
        )
        cold_start_selection = _cold_start_blended_selection(
            pool_ids,
            k,
            context,
            probabilities,
            strategy_name=self.name,
            exploitation_order=exploitation_order,
            exploration_strategy=DensityWeightedDiversityStrategy(),
        )
        if cold_start_selection is not None:
            return cold_start_selection
        small_seed_selection = _small_seed_diversity_selection(
            pool_ids,
            k,
            context,
            strategy_name=self.name,
            exploitation_order=exploitation_order,
            exploration_strategy=DensityWeightedDiversityStrategy(),
        )
        if small_seed_selection is not None:
            return small_seed_selection
        return exploitation_order
