"""Embedding-backed deterministic diversity strategies."""

from __future__ import annotations


import hashlib
import math
from numbers import Real
from typing import Any, List, Sequence, TYPE_CHECKING

import numpy as np

from ..exceptions import ConfigurationError

if TYPE_CHECKING:
    from ..engine import SelectionContext


_NEAR_DUPLICATE_DISTANCE = 1e-12
_DENSITY_REFERENCE_LIMIT = 512
_DENSITY_CHUNK_SIZE = 512


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


def _target_count(pool_ids: Sequence[str], k: int) -> int:
    if k <= 0 or not pool_ids:
        return 0
    return min(k, len(pool_ids))


def _normalize_embedding_rows(embeddings: Any, sample_ids: Sequence[str], *, strategy_name: str) -> np.ndarray:
    try:
        rows = list(embeddings)
    except TypeError as exc:
        raise ConfigurationError(f"{strategy_name}.embed output must be row-like.") from exc

    if len(rows) != len(sample_ids):
        raise ConfigurationError(f"{strategy_name}.embed returned {len(rows)} rows for {len(sample_ids)} sample ids.")

    normalized_rows: List[List[float]] = []
    expected_width: int | None = None
    for row_index, (sample_id, row) in enumerate(zip(sample_ids, rows)):
        if isinstance(row, (str, bytes)):
            raise ConfigurationError(
                f"{strategy_name}.embed row {row_index} for sample {sample_id!r} must be a sequence of numeric values."
            )
        try:
            values = list(row)
        except TypeError as exc:
            raise ConfigurationError(
                f"{strategy_name}.embed row {row_index} for sample {sample_id!r} must be a sequence."
            ) from exc

        if not values:
            raise ConfigurationError(f"{strategy_name}.embed row {row_index} for sample {sample_id!r} must not be empty.")

        if expected_width is None:
            expected_width = len(values)
        elif len(values) != expected_width:
            raise ConfigurationError(
                f"{strategy_name}.embed row {row_index} for sample {sample_id!r} has width {len(values)}; "
                f"expected {expected_width}."
            )

        cleaned: List[float] = []
        for column_index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ConfigurationError(
                    f"{strategy_name}.embed value at row {row_index}, column {column_index} must be numeric."
                )
            embedding_value = float(value)
            if not math.isfinite(embedding_value):
                raise ConfigurationError(
                    f"{strategy_name}.embed value at row {row_index}, column {column_index} must be finite."
                )
            cleaned.append(embedding_value)
        normalized_rows.append(cleaned)

    return np.asarray(normalized_rows, dtype=float)


def _embedding_matrix(context: "SelectionContext", sample_ids: Sequence[str], *, strategy_name: str) -> np.ndarray:
    if not sample_ids:
        return np.empty((0, 0), dtype=float)
    return _normalize_embedding_rows(context.embed(sample_ids), sample_ids, strategy_name=strategy_name)


def _squared_distance_to_center(matrix: np.ndarray, center: np.ndarray) -> np.ndarray:
    diff = matrix - center.reshape(1, -1)
    return np.einsum("ij,ij->i", diff, diff)


def _nearest_squared_distances(matrix: np.ndarray, centers: np.ndarray) -> np.ndarray:
    if centers.size == 0:
        return np.full(matrix.shape[0], np.inf, dtype=float)
    best = np.full(matrix.shape[0], np.inf, dtype=float)
    for center in centers:
        best = np.minimum(best, _squared_distance_to_center(matrix, center))
    return best


def _pairwise_squared_distances(matrix: np.ndarray) -> np.ndarray:
    squared_norms = np.einsum("ij,ij->i", matrix, matrix)
    distances = squared_norms[:, None] + squared_norms[None, :] - (2.0 * matrix @ matrix.T)
    return np.maximum(distances, 0.0)


def _best_index_by_score(
    scores: Sequence[float] | np.ndarray,
    sample_ids: Sequence[str],
    *,
    selected_indices: set[int],
    strategy_name: str,
    model_id: str,
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
        key=lambda item: (-item[1], _tie_key(strategy_name, model_id, sample_ids[item[0]]), sample_ids[item[0]]),
    )[0][0]


def _rank_indices_by_score(
    scores: Sequence[float] | np.ndarray,
    sample_ids: Sequence[str],
    *,
    strategy_name: str,
    model_id: str,
) -> List[int]:
    return [
        index
        for index, _ in sorted(
            enumerate(float(score) for score in scores),
            key=lambda item: (-item[1], _tie_key(strategy_name, model_id, sample_ids[item[0]]), sample_ids[item[0]]),
        )
    ]


def _groups_by_id(strategy_name: str, sample_ids: Sequence[str], context: "SelectionContext") -> dict[str, tuple[str, str]]:
    samples = context.get_samples(sample_ids)
    if len(samples) != len(sample_ids):
        raise ConfigurationError(
            f"{strategy_name}.get_samples returned {len(samples)} samples for {len(sample_ids)} sample ids."
        )

    expected_ids = [str(sample_id) for sample_id in sample_ids]
    returned_ids = [str(getattr(sample, "sample_id", "")) for sample in samples]
    if len(set(returned_ids)) != len(returned_ids):
        raise ConfigurationError(f"{strategy_name}.get_samples returned duplicate sample_id values.")
    missing_ids = sorted(set(expected_ids) - set(returned_ids))
    foreign_ids = sorted(set(returned_ids) - set(expected_ids))
    if missing_ids or foreign_ids:
        raise ConfigurationError(
            f"{strategy_name}.get_samples returned sample_id values that do not match requested ids; "
            f"missing={missing_ids}, foreign={foreign_ids}."
        )
    if returned_ids != expected_ids:
        raise ConfigurationError(f"{strategy_name}.get_samples must return samples in requested sample_id order.")

    groups_by_id = {}
    for sample_id, sample in zip(expected_ids, samples):
        group_id = getattr(sample, "group_id", None)
        groups_by_id[sample_id] = ("sample", sample_id) if group_id is None else ("group", str(group_id))
    return groups_by_id


def _labeled_group_keys(strategy_name: str, context: "SelectionContext") -> set[tuple[str, str]]:
    labeled_ids = _unique_pool_ids(getattr(context, "labeled_ids", []))
    if not labeled_ids or not callable(getattr(context, "get_samples", None)):
        return set()
    return set(_groups_by_id(f"{strategy_name}.labeled_ids", labeled_ids, context).values())


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


def _centroid_scores(matrix: np.ndarray) -> np.ndarray:
    centroid = np.mean(matrix, axis=0)
    return _squared_distance_to_center(matrix, centroid)


def _centroid_representative_scores(matrix: np.ndarray) -> np.ndarray:
    return -_centroid_scores(matrix)


def _greedy_max_min(
    sample_ids: Sequence[str],
    matrix: np.ndarray,
    k: int,
    context: "SelectionContext",
    *,
    strategy_name: str,
    initial_centers: np.ndarray | None = None,
    density: np.ndarray | None = None,
    seed_mode: str = "farthest_from_centroid",
) -> List[str]:
    target_k = _target_count(sample_ids, k)
    if target_k <= 0:
        return []

    model_id = _context_model_id(context)
    selected_indices: set[int] = set()
    selected: List[str] = []

    if initial_centers is not None and initial_centers.size:
        min_distances = _nearest_squared_distances(matrix, initial_centers)
    else:
        min_distances = None

    while len(selected) < target_k:
        if min_distances is None:
            if seed_mode == "closest_to_centroid":
                scores = _centroid_representative_scores(matrix)
            elif seed_mode == "farthest_from_centroid":
                scores = _centroid_scores(matrix)
            else:
                raise ConfigurationError(f"Unsupported {strategy_name} seed mode {seed_mode!r}.")
        else:
            scores = min_distances.copy()
        if density is not None:
            scores = scores / np.maximum(density, 1e-12)

        index = _best_index_by_score(
            scores,
            sample_ids,
            selected_indices=selected_indices,
            strategy_name=strategy_name,
            model_id=model_id,
        )
        if index is None:
            break

        selected_indices.add(index)
        selected.append(sample_ids[index])
        distances_to_new_center = _squared_distance_to_center(matrix, matrix[index])
        if min_distances is None:
            min_distances = distances_to_new_center
        else:
            min_distances = np.minimum(min_distances, distances_to_new_center)

    return selected


def _reference_indices(size: int, limit: int = _DENSITY_REFERENCE_LIMIT) -> np.ndarray:
    if size <= limit:
        return np.arange(size, dtype=int)
    return np.linspace(0, size - 1, num=limit, dtype=int)


def _squared_distances_to_references(matrix: np.ndarray, references: np.ndarray) -> np.ndarray:
    matrix_norms = np.einsum("ij,ij->i", matrix, matrix)
    reference_norms = np.einsum("ij,ij->i", references, references)
    distances = matrix_norms[:, None] + reference_norms[None, :] - (2.0 * matrix @ references.T)
    return np.maximum(distances, 0.0)


def _local_density(matrix: np.ndarray) -> np.ndarray:
    reference_matrix = matrix[_reference_indices(matrix.shape[0])]
    reference_pairwise = _pairwise_squared_distances(reference_matrix)
    positive = reference_pairwise[reference_pairwise > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    densities = np.empty(matrix.shape[0], dtype=float)
    for offset in range(0, matrix.shape[0], _DENSITY_CHUNK_SIZE):
        chunk = matrix[offset : offset + _DENSITY_CHUNK_SIZE]
        distances = _squared_distances_to_references(chunk, reference_matrix)
        similarities = np.exp(-distances / scale)
        densities[offset : offset + chunk.shape[0]] = np.mean(similarities, axis=1)
    return densities


class KCenterGreedyStrategy:
    """Greedy CoreSet k-center selection with labeled samples as existing centers."""

    name = "coreset_kcenter"
    required_capabilities = frozenset({"embed"})

    def select(self, pool_ids: Sequence[str], k: int, context: "SelectionContext") -> List[str]:
        sample_ids = _unique_pool_ids(pool_ids)
        if _target_count(sample_ids, k) <= 0:
            return []

        matrix = _embedding_matrix(context, sample_ids, strategy_name=self.name)
        labeled_ids = _unique_pool_ids(getattr(context, "labeled_ids", []))
        initial_centers = None
        if labeled_ids:
            initial_centers = _embedding_matrix(context, labeled_ids, strategy_name=self.name)
            if initial_centers.shape[1] != matrix.shape[1]:
                raise ConfigurationError(
                    f"{self.name}.embed returned labeled embeddings with width {initial_centers.shape[1]}; "
                    f"expected {matrix.shape[1]}."
                )
        selected = _greedy_max_min(sample_ids, matrix, k, context, strategy_name=self.name, initial_centers=initial_centers)
        return _group_diverse_prefix([*selected, *sample_ids], sample_ids, k, context, strategy_name=self.name)


class EmbeddingKMeansPPStrategy:
    """Deterministic k-means++ style farthest-prototype initialization."""

    name = "embedding_kmeans_pp"
    required_capabilities = frozenset({"embed"})

    def select(self, pool_ids: Sequence[str], k: int, context: "SelectionContext") -> List[str]:
        sample_ids = _unique_pool_ids(pool_ids)
        if _target_count(sample_ids, k) <= 0:
            return []
        matrix = _embedding_matrix(context, sample_ids, strategy_name=self.name)
        selected = _greedy_max_min(
            sample_ids,
            matrix,
            k,
            context,
            strategy_name=self.name,
            seed_mode="closest_to_centroid",
        )
        return _group_diverse_prefix([*selected, *sample_ids], sample_ids, k, context, strategy_name=self.name)


class MaxMinEmbeddingStrategy:
    """Select points that maximize the minimum distance to the selected set."""

    name = "max_min_embedding"
    required_capabilities = frozenset({"embed"})

    def select(self, pool_ids: Sequence[str], k: int, context: "SelectionContext") -> List[str]:
        sample_ids = _unique_pool_ids(pool_ids)
        if _target_count(sample_ids, k) <= 0:
            return []
        matrix = _embedding_matrix(context, sample_ids, strategy_name=self.name)
        selected = _greedy_max_min(sample_ids, matrix, k, context, strategy_name=self.name)
        return _group_diverse_prefix([*selected, *sample_ids], sample_ids, k, context, strategy_name=self.name)


class DeduplicateNearNeighborsStrategy:
    """Prefer one sample per near-duplicate embedding before filling remaining slots."""

    name = "deduplicate_near_neighbors"
    required_capabilities = frozenset({"embed"})

    def select(self, pool_ids: Sequence[str], k: int, context: "SelectionContext") -> List[str]:
        sample_ids = _unique_pool_ids(pool_ids)
        target_k = _target_count(sample_ids, k)
        if target_k <= 0:
            return []

        matrix = _embedding_matrix(context, sample_ids, strategy_name=self.name)
        model_id = _context_model_id(context)
        order = _rank_indices_by_score(_centroid_scores(matrix), sample_ids, strategy_name=self.name, model_id=model_id)
        selected_indices: List[int] = []
        deferred: List[int] = []

        for index in order:
            if len(selected_indices) >= target_k:
                break
            if any(float(_squared_distance_to_center(matrix[index : index + 1], matrix[chosen])[0]) <= _NEAR_DUPLICATE_DISTANCE for chosen in selected_indices):
                deferred.append(index)
                continue
            selected_indices.append(index)

        for index in deferred + order:
            if len(selected_indices) >= target_k:
                break
            if index not in selected_indices:
                selected_indices.append(index)

        selected = [sample_ids[index] for index in selected_indices[:target_k]]
        return _group_diverse_prefix([*selected, *sample_ids], sample_ids, k, context, strategy_name=self.name)


class DensityWeightedDiversityStrategy:
    """Greedy diversity that downweights samples from locally dense embedding regions."""

    name = "density_weighted_diversity"
    required_capabilities = frozenset({"embed"})

    def select(self, pool_ids: Sequence[str], k: int, context: "SelectionContext") -> List[str]:
        sample_ids = _unique_pool_ids(pool_ids)
        if _target_count(sample_ids, k) <= 0:
            return []
        matrix = _embedding_matrix(context, sample_ids, strategy_name=self.name)
        density = _local_density(matrix)
        selected = _greedy_max_min(sample_ids, matrix, k, context, strategy_name=self.name, density=density)
        return _group_diverse_prefix([*selected, *sample_ids], sample_ids, k, context, strategy_name=self.name)
