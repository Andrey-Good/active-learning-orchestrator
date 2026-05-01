"""Configurable hybrid uncertainty/diversity acquisition strategies."""

from __future__ import annotations


import hashlib
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Dict, List, Mapping, Sequence, TYPE_CHECKING

import numpy as np

from ..exceptions import ConfigurationError
from .embedding import _normalize_embedding_rows
from .uncertainty import _normalize_probability_rows

if TYPE_CHECKING:
    from ..engine import SelectionContext


HYBRID_MODES = frozenset(
    {
        "weighted",
        "uncertainty_prefilter_diversity",
        "diversity_prefilter_uncertainty",
    }
)
UNCERTAINTY_COMPONENTS = frozenset({"entropy", "margin", "least_confidence"})
DIVERSITY_COMPONENTS = frozenset({"coreset_kcenter", "embedding_kmeans_pp", "max_min_embedding"})

_DEFAULT_HYBRID_CONFIG: Dict[str, Any] = {
    "mode": "weighted",
    "uncertainty": "entropy",
    "diversity": "coreset_kcenter",
    "uncertainty_weight": 0.5,
    "diversity_weight": 0.5,
    "prefilter_multiplier": 3.0,
    "exploration_fraction": 0.0,
    "class_balance": False,
    "group_balance": False,
}


@dataclass(frozen=True)
class HybridSelectionResult:
    selected: List[str]
    snapshot: Dict[str, Any]


def validate_hybrid_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and default a public hybrid config mapping."""
    if not isinstance(config, Mapping):
        raise ConfigurationError("scheduler_config.hybrid must be a mapping.")

    normalized = dict(_DEFAULT_HYBRID_CONFIG)
    normalized.update(dict(config))

    if normalized["mode"] not in HYBRID_MODES:
        raise ConfigurationError(f"Unsupported hybrid.mode={normalized['mode']!r}")
    if normalized["uncertainty"] not in UNCERTAINTY_COMPONENTS:
        raise ConfigurationError(f"Unsupported hybrid.uncertainty={normalized['uncertainty']!r}")
    if normalized["diversity"] not in DIVERSITY_COMPONENTS:
        raise ConfigurationError(f"Unsupported hybrid.diversity={normalized['diversity']!r}")

    normalized["uncertainty_weight"] = _non_negative_float(
        normalized["uncertainty_weight"],
        field_name="hybrid.uncertainty_weight",
    )
    normalized["diversity_weight"] = _non_negative_float(
        normalized["diversity_weight"],
        field_name="hybrid.diversity_weight",
    )
    if normalized["mode"] == "weighted" and (
        normalized["uncertainty_weight"] + normalized["diversity_weight"]
    ) <= 0.0:
        raise ConfigurationError("hybrid weighted mode requires uncertainty_weight + diversity_weight > 0")

    normalized["prefilter_multiplier"] = _positive_float(
        normalized["prefilter_multiplier"],
        field_name="hybrid.prefilter_multiplier",
    )
    normalized["exploration_fraction"] = _bounded_float(
        normalized["exploration_fraction"],
        field_name="hybrid.exploration_fraction",
        lower=0.0,
        upper=1.0,
    )
    normalized["class_balance"] = _bool_value(normalized["class_balance"], field_name="hybrid.class_balance")
    normalized["group_balance"] = _bool_value(normalized["group_balance"], field_name="hybrid.group_balance")
    return normalized


def normalize_scores(scores: Sequence[float], *, field_name: str = "scores") -> List[float]:
    """Min-max normalize finite scores, with constant scores mapping to zero."""
    cleaned: List[float] = []
    for index, score in enumerate(scores):
        if isinstance(score, bool) or not isinstance(score, Real):
            raise ConfigurationError(f"{field_name}[{index}] must be numeric.")
        value = float(score)
        if not math.isfinite(value):
            raise ConfigurationError(f"{field_name}[{index}] must be finite.")
        cleaned.append(value)

    if not cleaned:
        return []

    low = min(cleaned)
    high = max(cleaned)
    if high == low:
        return [0.0 for _ in cleaned]

    scale = high - low
    if not math.isfinite(scale):
        max_abs = max(max(abs(low), abs(high)), 1.0)
        scaled_low = low / max_abs
        scaled_high = high / max_abs
        scale = scaled_high - scaled_low
        if scale == 0.0 or not math.isfinite(scale):
            return [0.0 for _ in cleaned]
        return [(value / max_abs - scaled_low) / scale for value in cleaned]

    return [(value - low) / scale for value in cleaned]


class HybridStrategy:
    """Hybrid acquisition selector used by the scheduler's ``mode='hybrid'`` path."""

    name = "hybrid"
    required_capabilities = frozenset({"predict_proba", "embed"})

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = validate_hybrid_config(config)

    def select(self, pool_ids: Sequence[str], k: int, context: "SelectionContext") -> HybridSelectionResult:
        sample_ids = _unique_pool_ids(pool_ids)
        target_k = min(k, len(sample_ids))
        if target_k <= 0:
            return HybridSelectionResult([], self._snapshot(target_k=target_k, pool_size=len(sample_ids)))

        probabilities = _probability_rows(context, sample_ids, strategy_name=self.name)
        uncertainty_scores = _uncertainty_scores(
            sample_ids,
            probabilities,
            component=self.config["uncertainty"],
        )
        embeddings = _embedding_matrix(context, sample_ids, strategy_name=self.name)

        mode = self.config["mode"]
        if mode == "weighted":
            selected, details = self._select_weighted(
                sample_ids,
                probabilities,
                uncertainty_scores,
                embeddings,
                target_k,
                context,
            )
        elif mode == "uncertainty_prefilter_diversity":
            selected, details = self._select_uncertainty_prefilter_diversity(
                sample_ids,
                probabilities,
                uncertainty_scores,
                embeddings,
                target_k,
                context,
            )
        elif mode == "diversity_prefilter_uncertainty":
            selected, details = self._select_diversity_prefilter_uncertainty(
                sample_ids,
                probabilities,
                uncertainty_scores,
                embeddings,
                target_k,
                context,
            )
        else:
            raise ConfigurationError(f"Unsupported hybrid.mode={mode!r}")

        snapshot = self._snapshot(target_k=target_k, pool_size=len(sample_ids), **details)
        return HybridSelectionResult(selected, snapshot)

    def _select_weighted(
        self,
        sample_ids: Sequence[str],
        probabilities: Sequence[Sequence[float]],
        uncertainty_scores: Sequence[float],
        embeddings: np.ndarray,
        target_k: int,
        context: "SelectionContext",
    ) -> tuple[List[str], Dict[str, Any]]:
        if self.config["uncertainty_weight"] == 0.0 and self.config["diversity_weight"] > 0.0:
            ordered = _diversity_select(
                sample_ids,
                embeddings,
                sample_ids,
                len(sample_ids),
                context,
                component=self.config["diversity"],
                strategy_name=self.name,
            )
        else:
            diversity_scores = _diversity_scores(
                sample_ids,
                embeddings,
                context,
                component=self.config["diversity"],
                strategy_name=self.name,
            )
            normalized_uncertainty = normalize_scores(uncertainty_scores, field_name="hybrid.uncertainty_scores")
            normalized_diversity = normalize_scores(diversity_scores, field_name="hybrid.diversity_scores")
            combined = [
                self.config["uncertainty_weight"] * uncertainty_score
                + self.config["diversity_weight"] * diversity_score
                for uncertainty_score, diversity_score in zip(normalized_uncertainty, normalized_diversity)
            ]
            ordered = _rank_by_scores(sample_ids, combined, strategy_name=self.name, context=context)

        exploration_count = min(target_k, int(math.floor(target_k * self.config["exploration_fraction"])))
        exploration_ids: List[str] = []
        if exploration_count > 0:
            exploration_ids = _stable_random_order(sample_ids, strategy_name=self.name, context=context)[:exploration_count]
            ordered = [sample_id for sample_id in ordered if sample_id not in set(exploration_ids)]

        exploitation_count = target_k - exploration_count
        selected = _apply_guardrails(
            ordered,
            target_k=exploitation_count,
            probabilities_by_id=dict(zip(sample_ids, probabilities)),
            context=context,
            class_balance=self.config["class_balance"],
            group_balance=self.config["group_balance"],
        )
        selected = _dedup_preserve_order(selected + exploration_ids, target_k)
        selected_before_fallback = list(selected)
        if len(selected) < target_k:
            selected = _fill_from_order(selected, ordered + list(sample_ids), target_k)
        fallback_count = len(selected) - len(selected_before_fallback)

        return selected, {
            "exploration_count": exploration_count,
            "fallback_count": fallback_count,
        }

    def _select_uncertainty_prefilter_diversity(
        self,
        sample_ids: Sequence[str],
        probabilities: Sequence[Sequence[float]],
        uncertainty_scores: Sequence[float],
        embeddings: np.ndarray,
        target_k: int,
        context: "SelectionContext",
    ) -> tuple[List[str], Dict[str, Any]]:
        prefilter_count = _prefilter_count(target_k, len(sample_ids), self.config["prefilter_multiplier"])
        uncertainty_order = _rank_by_scores(sample_ids, uncertainty_scores, strategy_name=self.name, context=context)
        candidate_ids = uncertainty_order[:prefilter_count]
        diversity_selected = _diversity_select(
            candidate_ids,
            embeddings,
            sample_ids,
            target_k,
            context,
            component=self.config["diversity"],
            strategy_name=self.name,
        )
        primary_ids = set(_dedup_preserve_order(diversity_selected, target_k))
        selected = _apply_guardrails(
            diversity_selected + candidate_ids,
            target_k=target_k,
            probabilities_by_id=dict(zip(sample_ids, probabilities)),
            context=context,
            class_balance=self.config["class_balance"],
            group_balance=self.config["group_balance"],
        )
        fallback_count = sum(1 for sample_id in selected if sample_id not in primary_ids)
        return selected, {
            "prefilter_count": prefilter_count,
            "exploration_count": 0,
            "fallback_count": fallback_count,
        }

    def _select_diversity_prefilter_uncertainty(
        self,
        sample_ids: Sequence[str],
        probabilities: Sequence[Sequence[float]],
        uncertainty_scores: Sequence[float],
        embeddings: np.ndarray,
        target_k: int,
        context: "SelectionContext",
    ) -> tuple[List[str], Dict[str, Any]]:
        prefilter_count = _prefilter_count(target_k, len(sample_ids), self.config["prefilter_multiplier"])
        candidate_ids = _diversity_select(
            sample_ids,
            embeddings,
            sample_ids,
            prefilter_count,
            context,
            component=self.config["diversity"],
            strategy_name=self.name,
        )
        uncertainty_by_id = dict(zip(sample_ids, uncertainty_scores))
        candidate_scores = [uncertainty_by_id[sample_id] for sample_id in candidate_ids]
        ordered = _rank_by_scores(candidate_ids, candidate_scores, strategy_name=self.name, context=context)
        selected = _apply_guardrails(
            ordered,
            target_k=target_k,
            probabilities_by_id=dict(zip(sample_ids, probabilities)),
            context=context,
            class_balance=self.config["class_balance"],
            group_balance=self.config["group_balance"],
        )
        return selected, {
            "prefilter_count": prefilter_count,
            "exploration_count": 0,
            "fallback_count": target_k - len(selected),
        }

    def _snapshot(self, *, target_k: int, pool_size: int, **extra: Any) -> Dict[str, Any]:
        snapshot = {
            "mode": "hybrid",
            "target_k": target_k,
            "pool_size": pool_size,
            "hybrid": dict(self.config),
            "uncertainty": self.config["uncertainty"],
            "diversity": self.config["diversity"],
        }
        snapshot.update(extra)
        return snapshot


def _non_negative_float(value: Any, *, field_name: str) -> float:
    number = _finite_float(value, field_name=field_name)
    if number < 0.0:
        raise ConfigurationError(f"{field_name} must be >= 0.")
    return number


def _positive_float(value: Any, *, field_name: str) -> float:
    number = _finite_float(value, field_name=field_name)
    if number <= 0.0:
        raise ConfigurationError(f"{field_name} must be > 0.")
    return number


def _bounded_float(value: Any, *, field_name: str, lower: float, upper: float) -> float:
    number = _finite_float(value, field_name=field_name)
    if number < lower or number > upper:
        raise ConfigurationError(f"{field_name} must be in [{lower}, {upper}].")
    return number


def _finite_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigurationError(f"{field_name} must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigurationError(f"{field_name} must be finite.")
    return number


def _bool_value(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{field_name} must be a bool.")
    return value


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


def _probability_rows(
    context: "SelectionContext",
    sample_ids: Sequence[str],
    *,
    strategy_name: str,
) -> List[List[float]]:
    return _normalize_probability_rows(
        context.predict_proba(sample_ids),
        sample_ids,
        strategy_name=strategy_name,
        label_count=_label_count(context),
    )


def _label_count(context: "SelectionContext") -> int | None:
    label_schema = getattr(context, "label_schema", None)
    labels = getattr(label_schema, "labels", None)
    if labels is None:
        return None
    try:
        return len(list(labels))
    except TypeError:
        return None


def _embedding_matrix(context: "SelectionContext", sample_ids: Sequence[str], *, strategy_name: str) -> np.ndarray:
    if not sample_ids:
        return np.empty((0, 0), dtype=float)
    return _normalize_embedding_rows(context.embed(sample_ids), sample_ids, strategy_name=strategy_name)


def _uncertainty_scores(
    sample_ids: Sequence[str],
    probabilities: Sequence[Sequence[float]],
    *,
    component: str,
) -> List[float]:
    scores: List[float] = []
    for sample_id, row in zip(sample_ids, probabilities):
        if component == "entropy":
            entropy = 0.0
            for probability in row:
                if probability > 0.0:
                    entropy -= probability * math.log(probability)
            scores.append(entropy)
        elif component == "margin":
            ordered = sorted(row, reverse=True)
            margin = ordered[0] - ordered[1] if len(ordered) >= 2 else ordered[0]
            scores.append(-margin)
        elif component == "least_confidence":
            scores.append(1.0 - max(row))
        else:
            raise ConfigurationError(f"Unsupported hybrid.uncertainty={component!r} for sample {sample_id!r}")
    return scores


def _diversity_scores(
    sample_ids: Sequence[str],
    matrix: np.ndarray,
    context: "SelectionContext",
    *,
    component: str,
    strategy_name: str,
) -> List[float]:
    if len(sample_ids) == 0:
        return []
    if component == "coreset_kcenter":
        labeled = _labeled_centers(context, matrix.shape[1], strategy_name=strategy_name)
        if labeled is not None and labeled.size:
            return list(_nearest_squared_distances(matrix, labeled))
        return list(_centroid_distance_scores(matrix))
    if component == "max_min_embedding":
        return list(_centroid_distance_scores(matrix))
    if component == "embedding_kmeans_pp":
        return list(_centroid_representative_scores(matrix))
    raise ConfigurationError(f"Unsupported hybrid.diversity={component!r}")


def _diversity_select(
    candidate_ids: Sequence[str],
    full_matrix: np.ndarray,
    full_ids: Sequence[str],
    k: int,
    context: "SelectionContext",
    *,
    component: str,
    strategy_name: str,
) -> List[str]:
    candidate_ids = _dedup_preserve_order(candidate_ids, len(candidate_ids))
    if k <= 0 or not candidate_ids:
        return []
    id_to_index = {sample_id: index for index, sample_id in enumerate(full_ids)}
    candidate_indices = [id_to_index[sample_id] for sample_id in candidate_ids]
    matrix = full_matrix[candidate_indices, :]
    target_k = min(k, len(candidate_ids))

    selected_indices: set[int] = set()
    selected: List[str] = []
    min_distances = None
    if component == "coreset_kcenter":
        labeled = _labeled_centers(context, matrix.shape[1], strategy_name=strategy_name)
        if labeled is not None and labeled.size:
            min_distances = _nearest_squared_distances(matrix, labeled)
    elif component not in DIVERSITY_COMPONENTS:
        raise ConfigurationError(f"Unsupported hybrid.diversity={component!r}")

    while len(selected) < target_k:
        if min_distances is None:
            scores = _diversity_seed_scores(matrix, component=component)
        else:
            scores = min_distances.copy()

        index = _best_index_by_score(
            scores,
            candidate_ids,
            selected_indices=selected_indices,
            strategy_name=strategy_name,
            context=context,
        )
        if index is None:
            break

        selected_indices.add(index)
        selected.append(candidate_ids[index])
        distances_to_new_center = _squared_distances(matrix, matrix[index])
        if min_distances is None:
            min_distances = distances_to_new_center
        else:
            min_distances = np.minimum(min_distances, distances_to_new_center)
    return selected


def _labeled_centers(context: "SelectionContext", width: int, *, strategy_name: str) -> np.ndarray | None:
    labeled_ids = _unique_pool_ids(getattr(context, "labeled_ids", []))
    if not labeled_ids:
        return None
    centers = _embedding_matrix(context, labeled_ids, strategy_name=strategy_name)
    if centers.shape[1] != width:
        raise ConfigurationError(
            f"{strategy_name}.embed returned labeled embeddings with width {centers.shape[1]}; expected {width}."
        )
    return centers


def _squared_distances(matrix: np.ndarray, center: np.ndarray) -> np.ndarray:
    diff = matrix - center.reshape(1, -1)
    return np.einsum("ij,ij->i", diff, diff)


def _centroid_distance_scores(matrix: np.ndarray) -> np.ndarray:
    centroid = np.mean(matrix, axis=0)
    return _squared_distances(matrix, centroid)


def _centroid_representative_scores(matrix: np.ndarray) -> np.ndarray:
    return -_centroid_distance_scores(matrix)


def _diversity_seed_scores(matrix: np.ndarray, *, component: str) -> np.ndarray:
    if component == "embedding_kmeans_pp":
        return _centroid_representative_scores(matrix)
    if component in {"coreset_kcenter", "max_min_embedding"}:
        return _centroid_distance_scores(matrix)
    raise ConfigurationError(f"Unsupported hybrid.diversity={component!r}")


def _nearest_squared_distances(matrix: np.ndarray, centers: np.ndarray) -> np.ndarray:
    best = np.full(matrix.shape[0], np.inf, dtype=float)
    for center in centers:
        best = np.minimum(best, _squared_distances(matrix, center))
    return best


def _rank_by_scores(
    sample_ids: Sequence[str],
    scores: Sequence[float],
    *,
    strategy_name: str,
    context: "SelectionContext",
) -> List[str]:
    if len(sample_ids) != len(scores):
        raise ConfigurationError(f"{strategy_name} got {len(scores)} scores for {len(sample_ids)} sample ids.")
    for index, score in enumerate(scores):
        value = float(score)
        if not math.isfinite(value):
            raise ConfigurationError(f"{strategy_name} score at index {index} must be finite.")
    return [
        sample_id
        for sample_id, _ in sorted(
            zip(sample_ids, scores),
            key=lambda item: (-float(item[1]), _tie_key(strategy_name, context, item[0]), item[0]),
        )
    ]


def _best_index_by_score(
    scores: Sequence[float] | np.ndarray,
    sample_ids: Sequence[str],
    *,
    selected_indices: set[int],
    strategy_name: str,
    context: "SelectionContext",
) -> int | None:
    candidates = [
        (index, float(score))
        for index, score in enumerate(scores)
        if index not in selected_indices and math.isfinite(float(score))
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (-item[1], _tie_key(strategy_name, context, sample_ids[item[0]]), sample_ids[item[0]]),
    )[0][0]


def _apply_guardrails(
    ordered_ids: Sequence[str],
    *,
    target_k: int,
    probabilities_by_id: Mapping[str, Sequence[float]],
    context: "SelectionContext",
    class_balance: bool,
    group_balance: bool,
) -> List[str]:
    ordered = _dedup_preserve_order(ordered_ids, len(ordered_ids))
    if class_balance:
        class_by_id = {
            sample_id: max(range(len(probabilities_by_id[sample_id])), key=lambda index: (probabilities_by_id[sample_id][index], -index))
            for sample_id in ordered
            if sample_id in probabilities_by_id
        }
        ordered = _class_balanced_order(ordered, class_by_id)
    if group_balance:
        ordered = _group_balanced_order(ordered, _group_keys(ordered, context))
    return ordered[:target_k]


def _class_balanced_order(ordered_ids: Sequence[str], class_by_id: Mapping[str, int]) -> List[str]:
    buckets: Dict[int, List[str]] = {}
    class_order: List[int] = []
    for sample_id in ordered_ids:
        predicted_class = class_by_id.get(sample_id)
        if predicted_class is None:
            continue
        if predicted_class not in buckets:
            buckets[predicted_class] = []
            class_order.append(predicted_class)
        buckets[predicted_class].append(sample_id)

    selected: List[str] = []
    positions = {predicted_class: 0 for predicted_class in class_order}
    while len(selected) < len(ordered_ids):
        progressed = False
        for predicted_class in class_order:
            position = positions[predicted_class]
            if position >= len(buckets[predicted_class]):
                continue
            selected.append(buckets[predicted_class][position])
            positions[predicted_class] = position + 1
            progressed = True
        if not progressed:
            break
    return _fill_from_order(selected, ordered_ids, len(ordered_ids))


def _group_balanced_order(ordered_ids: Sequence[str], group_by_id: Mapping[str, tuple[str, str]]) -> List[str]:
    selected: List[str] = []
    selected_ids = set()
    selected_groups = set()
    for sample_id in ordered_ids:
        group_key = group_by_id.get(sample_id, ("sample", sample_id))
        if group_key in selected_groups:
            continue
        selected.append(sample_id)
        selected_ids.add(sample_id)
        selected_groups.add(group_key)
    for sample_id in ordered_ids:
        if sample_id in selected_ids:
            continue
        selected.append(sample_id)
    return selected


def _group_keys(sample_ids: Sequence[str], context: "SelectionContext") -> Dict[str, tuple[str, str]]:
    expected_ids = [str(sample_id) for sample_id in sample_ids]
    keys = {sample_id: ("sample", sample_id) for sample_id in expected_ids}
    try:
        samples = context.get_samples(expected_ids)
    except Exception as error:
        raise ConfigurationError(f"hybrid group_balance could not read sample groups: {error}") from error
    if len(samples) != len(expected_ids):
        raise ConfigurationError(
            f"hybrid group_balance get_samples returned {len(samples)} samples for {len(expected_ids)} sample_id values."
        )
    returned_ids = [str(getattr(sample, "sample_id", "")) for sample in samples]
    if len(set(returned_ids)) != len(returned_ids):
        raise ConfigurationError("hybrid group_balance get_samples returned duplicate sample_id values.")
    missing_ids = sorted(set(expected_ids) - set(returned_ids))
    foreign_ids = sorted(set(returned_ids) - set(expected_ids))
    if missing_ids or foreign_ids:
        raise ConfigurationError(
            "hybrid group_balance get_samples returned sample_id values that do not match requested ids; "
            f"missing={missing_ids}, foreign={foreign_ids}."
        )
    if returned_ids != expected_ids:
        raise ConfigurationError("hybrid group_balance get_samples must return samples in requested sample_id order.")
    for sample in samples:
        sample_id = str(getattr(sample, "sample_id", ""))
        group_id = getattr(sample, "group_id", None)
        if group_id is not None:
            keys[sample_id] = ("group", str(group_id))
    return keys


def _prefilter_count(target_k: int, pool_size: int, multiplier: float) -> int:
    return min(pool_size, max(target_k, int(math.ceil(target_k * multiplier))))


def _fill_from_order(selected: Sequence[str], ordered_ids: Sequence[str], target_k: int) -> List[str]:
    out = _dedup_preserve_order(selected, target_k)
    selected_set = set(out)
    for sample_id in ordered_ids:
        if len(out) >= target_k:
            break
        if sample_id in selected_set:
            continue
        out.append(sample_id)
        selected_set.add(sample_id)
    return out


def _dedup_preserve_order(ids: Sequence[str], k: int) -> List[str]:
    seen = set()
    out: List[str] = []
    for sample_id in ids:
        if sample_id in seen:
            continue
        seen.add(sample_id)
        out.append(sample_id)
        if len(out) >= k:
            break
    return out


def _stable_random_order(
    sample_ids: Sequence[str],
    *,
    strategy_name: str,
    context: "SelectionContext",
) -> List[str]:
    return sorted(sample_ids, key=lambda sample_id: (_stable_hash("explore", strategy_name, _model_id(context), sample_id), sample_id))


def _tie_key(strategy_name: str, context: "SelectionContext", sample_id: str) -> str:
    return _stable_hash("tie", strategy_name, _model_id(context), sample_id)


def _model_id(context: "SelectionContext") -> str:
    model_id = getattr(context, "model_id", None)
    if callable(model_id):
        return str(model_id())
    return "unknown"


def _stable_hash(*parts: object) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()
