from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from active_learning_sdk import (
    ActiveLearningProject,
    CacheConfig,
    ConfigurationError,
    LabelBackendConfig,
    LabelSchema,
    SchedulerConfig,
    SplitConfig,
)
from active_learning_sdk.backends.base import RoundProgress, RoundPullResult, RoundPushResult
from active_learning_sdk.cache import EmbeddingCache, InMemoryCacheStore
from active_learning_sdk.engine import StrategyScheduler
from active_learning_sdk.strategies import (
    DeduplicateNearNeighborsStrategy,
    DensityWeightedDiversityStrategy,
    EmbeddingKMeansPPStrategy,
    KCenterGreedyStrategy,
    MaxMinEmbeddingStrategy,
)
from active_learning_sdk.strategies.embedding import _local_density
from active_learning_sdk.types import DataSample


class FakeEmbeddingContext:
    def __init__(
        self,
        embeddings: Mapping[str, Any],
        *,
        labeled_ids: Sequence[str] = (),
        model_id: str = "fake-model",
    ) -> None:
        self._embeddings = dict(embeddings)
        self.labeled_ids = list(labeled_ids)
        self._model_id = model_id

    def model_id(self) -> str:
        return self._model_id

    def embed(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[Any]:
        return [self._embeddings[sample_id] for sample_id in sample_ids]


class GroupedEmbeddingContext(FakeEmbeddingContext):
    def __init__(
        self,
        embeddings: Mapping[str, Any],
        *,
        groups: Mapping[str, str | None],
        labeled_ids: Sequence[str] = (),
    ) -> None:
        super().__init__(embeddings, labeled_ids=labeled_ids)
        self._groups = dict(groups)

    def get_samples(self, sample_ids: Sequence[str]) -> list[DataSample]:
        return [
            DataSample(sample_id=str(sample_id), data={"text": str(sample_id)}, group_id=self._groups.get(str(sample_id)))
            for sample_id in sample_ids
        ]


class InMemoryDataset:
    def __init__(self) -> None:
        self._samples = {
            "s1": DataSample(sample_id="s1", data={"text": "one"}),
            "s2": DataSample(sample_id="s2", data={"text": "two"}),
            "s3": DataSample(sample_id="s3", data={"text": "three"}),
        }

    def iter_sample_ids(self):
        yield from self._samples

    def get_sample(self, sample_id: str) -> DataSample:
        return self._samples[sample_id]

    def get_samples(self, sample_ids: Sequence[str]) -> list[DataSample]:
        return [self.get_sample(sample_id) for sample_id in sample_ids]

    def schema(self) -> dict[str, str]:
        return {"sample_id": "str", "text": "str"}


class NoopBackend:
    def ensure_ready(self, label_schema: LabelSchema) -> dict[str, Any]:
        return {"backend": "noop"}

    def push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: dict[str, Any] | None = None,
    ) -> RoundPushResult:
        return RoundPushResult(task_ids={sample.sample_id: sample.sample_id for sample in samples})

    def poll_round(
        self,
        round_id: str,
        task_ids: Mapping[str, str],
        policy: Any,
    ) -> RoundProgress:
        return RoundProgress(total=len(task_ids), done=len(task_ids), ready_sample_ids=list(task_ids))

    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        return RoundPullResult(annotations={})


class ProbabilityModel:
    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [[0.5, 0.5] for _ in texts]

    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs) -> None:
        return None

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> dict[str, float]:
        return {"accuracy": 0.0}


class EmbeddingModel(ProbabilityModel):
    def embed(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [[float(index), 0.0] for index, _ in enumerate(texts)]

    def get_model_id(self) -> str:
        return "embedding-model"

    def get_embedding_config(self) -> str:
        return "embedding-config-v1"


def _configure_project(tmp_path: Path, *, model: Any, strategy_name: str) -> ActiveLearningProject:
    project = ActiveLearningProject("embedding-test", tmp_path, lock=False)
    project.configure(
        dataset=InMemoryDataset(),
        model=model,
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=NoopBackend(),
        scheduler_config=SchedulerConfig(strategy=strategy_name),
        cache_config=CacheConfig(enable=False),
        split_config=SplitConfig(
            mode="explicit",
            explicit_splits={"train": ["s1"], "val": ["s2"], "test": ["s3"]},
        ),
    )
    return project


def _assert_valid_unique_selection(selected: Sequence[str], pool_ids: Sequence[str], k: int) -> None:
    assert len(selected) == min(k, len(set(pool_ids)))
    assert len(selected) == len(set(selected))
    assert set(selected).issubset(set(pool_ids))


def _embedding_strategies():
    return (
        KCenterGreedyStrategy(),
        EmbeddingKMeansPPStrategy(),
        MaxMinEmbeddingStrategy(),
        DeduplicateNearNeighborsStrategy(),
        DensityWeightedDiversityStrategy(),
    )


def test_coreset_kcenter_chooses_diverse_endpoints_on_simple_geometry() -> None:
    context = FakeEmbeddingContext(
        {
            "s1": [0.0, 0.0],
            "s2": [1.0, 0.0],
            "s3": [10.0, 0.0],
            "s4": [11.0, 0.0],
        }
    )

    selected = KCenterGreedyStrategy().select(["s1", "s2", "s3", "s4"], 2, context)

    assert set(selected) == {"s1", "s4"}


def test_coreset_kcenter_uses_labeled_ids_as_existing_centers() -> None:
    context = FakeEmbeddingContext(
        {
            "labeled": [0.0, 0.0],
            "near": [1.0, 0.0],
            "middle": [5.0, 0.0],
            "far": [10.0, 0.0],
        },
        labeled_ids=["labeled"],
    )

    selected = KCenterGreedyStrategy().select(["near", "middle", "far"], 1, context)

    assert selected == ["far"]


def test_embedding_diversity_strategies_are_distinct_on_designed_geometry() -> None:
    embeddings = {
        "labeled": [0.0],
        "a": [3.0],
        "b": [4.0],
        "c": [10.0],
    }
    pool = ["a", "b", "c"]

    max_min = MaxMinEmbeddingStrategy().select(pool, 2, FakeEmbeddingContext(embeddings))
    kmeans_pp = EmbeddingKMeansPPStrategy().select(pool, 2, FakeEmbeddingContext(embeddings))
    coreset = KCenterGreedyStrategy().select(pool, 2, FakeEmbeddingContext(embeddings, labeled_ids=["labeled"]))

    assert max_min == ["c", "a"]
    assert kmeans_pp == ["b", "c"]
    assert coreset == ["c", "b"]
    assert len({tuple(max_min), tuple(kmeans_pp), tuple(coreset)}) == 3


def test_embedding_strategies_return_deterministic_valid_diverse_batches() -> None:
    embeddings = {
        "a": [0.0, 0.0],
        "b": [0.1, 0.0],
        "c": [5.0, 0.0],
        "d": [10.0, 0.0],
        "e": [10.1, 0.0],
    }
    pool = ["a", "b", "c", "d", "e"]

    for strategy in _embedding_strategies():
        context = FakeEmbeddingContext(embeddings)
        first = strategy.select(pool, 3, context)
        second = strategy.select(pool, 3, context)

        assert first == second
        _assert_valid_unique_selection(first, pool, 3)


def test_embedding_strategies_avoid_already_labeled_groups_when_group_ids_are_available() -> None:
    embeddings = {
        "labeled-a": [0.0, 0.0],
        "a2": [0.1, 0.0],
        "b1": [5.0, 0.0],
        "c1": [10.0, 0.0],
    }
    groups = {"labeled-a": "group-a", "a2": "group-a", "b1": "group-b", "c1": "group-c"}

    for strategy in _embedding_strategies():
        context = GroupedEmbeddingContext(embeddings, groups=groups, labeled_ids=["labeled-a"])
        selected = strategy.select(["a2", "b1", "c1"], 2, context)

        assert set(selected) == {"b1", "c1"}


def test_embedding_strategies_return_empty_selection_for_empty_pool() -> None:
    for strategy in _embedding_strategies():
        assert strategy.select([], 3, FakeEmbeddingContext({})) == []


@pytest.mark.parametrize("k", [0, -1])
def test_embedding_strategies_return_empty_selection_for_non_positive_k(k: int) -> None:
    embeddings = {
        "a": [0.0, 0.0],
        "b": [1.0, 0.0],
    }

    for strategy in _embedding_strategies():
        assert strategy.select(["a", "b"], k, FakeEmbeddingContext(embeddings)) == []


def test_embedding_strategies_handle_partial_duplicate_embeddings_deterministically() -> None:
    embeddings = {
        "a": [0.0, 0.0],
        "b": [0.0, 0.0],
        "c": [3.0, 0.0],
        "d": [9.0, 0.0],
    }
    pool = ["a", "b", "c", "d"]

    for strategy in _embedding_strategies():
        context = FakeEmbeddingContext(embeddings)
        first = strategy.select(pool, 3, context)
        second = strategy.select(pool, 3, context)

        assert first == second
        _assert_valid_unique_selection(first, pool, 3)


def test_deduplicate_near_neighbors_prefers_unique_embeddings_before_duplicates() -> None:
    embeddings = {
        "a": [0.0, 0.0],
        "b": [0.0, 0.0],
        "c": [0.0, 0.0],
        "d": [5.0, 0.0],
        "e": [10.0, 0.0],
    }

    selected = DeduplicateNearNeighborsStrategy().select(
        ["a", "b", "c", "d", "e"],
        3,
        FakeEmbeddingContext(embeddings),
    )

    _assert_valid_unique_selection(selected, ["a", "b", "c", "d", "e"], 3)
    assert len({tuple(embeddings[sample_id]) for sample_id in selected}) == 3


def test_identical_embeddings_return_deterministic_unique_ids() -> None:
    context = FakeEmbeddingContext({sample_id: [1.0, 1.0] for sample_id in ("a", "b", "c")})

    first = KCenterGreedyStrategy().select(["a", "b", "c"], 5, context)
    second = KCenterGreedyStrategy().select(["a", "b", "c"], 5, context)

    assert first == second
    _assert_valid_unique_selection(first, ["a", "b", "c"], 5)


@pytest.mark.parametrize(
    ("embeddings", "match"),
    [
        ([["wrong-count"]], "returned 1 rows for 2 sample ids"),
        ([[1.0], [1.0, 2.0]], "expected 1"),
        ([[], []], "must not be empty"),
        ([[1.0], ["bad"]], "must be numeric"),
        ([[1.0], [float("inf")]], "must be finite"),
    ],
)
def test_malformed_embeddings_raise_configuration_error(embeddings: list[Any], match: str) -> None:
    class MalformedContext(FakeEmbeddingContext):
        def embed(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[Any]:
            return embeddings

    with pytest.raises(ConfigurationError, match=match):
        KCenterGreedyStrategy().select(["a", "b"], 1, MalformedContext({}))


def test_configuring_coreset_kcenter_without_embed_fails(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="coreset_kcenter.*embed"):
        _configure_project(tmp_path, model=ProbabilityModel(), strategy_name="coreset_kcenter")


def test_configuring_coreset_kcenter_with_embed_succeeds(tmp_path: Path) -> None:
    project = _configure_project(tmp_path, model=EmbeddingModel(), strategy_name="coreset_kcenter")

    assert project.get_state().scheduler_config["strategy"] == "coreset_kcenter"


def test_embedding_strategies_are_available_through_scheduler() -> None:
    context = FakeEmbeddingContext(
        {
            "a": [0.0, 0.0],
            "b": [1.0, 0.0],
            "c": [2.0, 0.0],
        }
    )

    for strategy_name in (
        "coreset_kcenter",
        "embedding_kmeans_pp",
        "max_min_embedding",
        "deduplicate_near_neighbors",
        "density_weighted_diversity",
    ):
        scheduler = StrategyScheduler(SchedulerConfig(strategy=strategy_name))
        selected, snapshot = scheduler.select_batch(["a", "b", "c"], 2, context, state={})

        _assert_valid_unique_selection(selected, ["a", "b", "c"], 2)
        assert snapshot == {"mode": "single", "strategy": strategy_name}


def test_embedding_cache_keys_differ_by_dataset_fingerprint() -> None:
    cache = EmbeddingCache(InMemoryCacheStore())

    cache.set("model", "sample", [1.0, 2.0], dataset_fingerprint="dataset-a", embedding_config="v1")

    assert cache.get("model", "sample", dataset_fingerprint="dataset-a", embedding_config="v1") == [1.0, 2.0]
    assert cache.get("model", "sample", dataset_fingerprint="dataset-b", embedding_config="v1") is None
    assert cache.get("model", "sample", dataset_fingerprint="dataset-a", embedding_config="v2") is None


def test_embedding_cache_direct_get_set_signature_remains_supported() -> None:
    cache = EmbeddingCache(InMemoryCacheStore())

    cache.set("model", "sample", [3.0])

    assert cache.get("model", "sample") == [3.0]


def test_density_estimate_is_finite_for_large_pool_without_full_tensor_allocation() -> None:
    import numpy as np

    matrix = np.asarray([[float(index), float(index % 7)] for index in range(700)], dtype=float)

    density = _local_density(matrix)

    assert density.shape == (700,)
    assert np.all(np.isfinite(density))
    assert np.all(density > 0.0)
