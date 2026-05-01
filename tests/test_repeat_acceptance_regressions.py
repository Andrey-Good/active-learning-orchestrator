from __future__ import annotations

from typing import Sequence

import pytest

from active_learning_sdk import LabelSchema, SchedulerConfig
from active_learning_sdk.cache import EmbeddingCache, InMemoryCacheStore
from active_learning_sdk.engine import SelectionContext, StrategyScheduler
from active_learning_sdk.strategies.embedding import KCenterGreedyStrategy
from active_learning_sdk.types import DataSample


class EmbeddingProvider:
    def get_samples(self, sample_ids: Sequence[str]) -> list[DataSample]:
        return [DataSample(sample_id=sample_id, data={"text": sample_id}) for sample_id in sample_ids]

    def get_texts(self, sample_ids: Sequence[str]) -> list[str]:
        return list(sample_ids)


class BadEmbeddingModel:
    def __init__(self) -> None:
        self.return_valid = False

    def get_model_id(self) -> str:
        return "repeat-acceptance-bad-embedding"

    def embed(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        if self.return_valid:
            return [[float(index), 1.0] for index, _ in enumerate(texts)]
        return [[0.0, float("nan")] for _ in texts]


class VariableWidthEmbeddingModel:
    def get_model_id(self) -> str:
        return "repeat-acceptance-variable-width-embedding"

    def embed(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


def test_invalid_embedding_rows_do_not_poison_embedding_cache() -> None:
    store = InMemoryCacheStore()
    context = SelectionContext(
        provider=EmbeddingProvider(),
        model=BadEmbeddingModel(),
        label_schema=LabelSchema(task="text_classification", labels=["alpha", "beta"]),
        prediction_cache=None,
        embedding_cache=EmbeddingCache(store),
        labeled_ids=[],
        last_metrics={},
        dataset_fingerprint="repeat-acceptance-fingerprint",
    )

    with pytest.raises(Exception, match="finite"):
        KCenterGreedyStrategy().select(["s1", "s2"], 1, context)

    assert store.stats()["items"] == 0


def test_existing_invalid_embedding_cache_entry_is_evicted_and_recomputed() -> None:
    store = InMemoryCacheStore()
    model = BadEmbeddingModel()
    context = SelectionContext(
        provider=EmbeddingProvider(),
        model=model,
        label_schema=LabelSchema(task="text_classification", labels=["alpha", "beta"]),
        prediction_cache=None,
        embedding_cache=EmbeddingCache(store),
        labeled_ids=[],
        last_metrics={},
        dataset_fingerprint="repeat-acceptance-fingerprint",
    )
    context.embedding_cache.set(
        context.model_id(),
        "s1",
        [0.0, float("nan")],
        dataset_fingerprint=context.dataset_fingerprint,
        embedding_config="default-embedding",
    )

    model.return_valid = True
    selected = KCenterGreedyStrategy().select(["s1", "s2"], 1, context)

    assert len(selected) == 1
    assert selected[0] in {"s1", "s2"}
    assert store.stats()["items"] == 2


def test_existing_embedding_cache_entry_with_incompatible_width_is_evicted_and_recomputed() -> None:
    store = InMemoryCacheStore()
    model = VariableWidthEmbeddingModel()
    context = SelectionContext(
        provider=EmbeddingProvider(),
        model=model,
        label_schema=LabelSchema(task="text_classification", labels=["alpha", "beta"]),
        prediction_cache=None,
        embedding_cache=EmbeddingCache(store),
        labeled_ids=[],
        last_metrics={},
        dataset_fingerprint="repeat-acceptance-fingerprint",
    )
    context.embedding_cache.set(
        context.model_id(),
        "s1",
        [0.0],
        dataset_fingerprint=context.dataset_fingerprint,
        embedding_config="default-embedding",
    )

    embeddings = context.embed(["s1", "s2"])

    assert embeddings == [[0.0, 1.0], [1.0, 1.0]]
    assert store.stats()["items"] == 2


def test_all_cached_embedding_entries_with_incompatible_width_are_evicted_and_recomputed() -> None:
    store = InMemoryCacheStore()
    model = VariableWidthEmbeddingModel()
    context = SelectionContext(
        provider=EmbeddingProvider(),
        model=model,
        label_schema=LabelSchema(task="text_classification", labels=["alpha", "beta"]),
        prediction_cache=None,
        embedding_cache=EmbeddingCache(store),
        labeled_ids=[],
        last_metrics={},
        dataset_fingerprint="repeat-acceptance-fingerprint",
    )
    for sample_id, embedding in {"s1": [0.0], "s2": [0.0, 1.0]}.items():
        context.embedding_cache.set(
            context.model_id(),
            sample_id,
            embedding,
            dataset_fingerprint=context.dataset_fingerprint,
            embedding_config="default-embedding",
        )

    embeddings = context.embed(["s1", "s2"])

    assert embeddings == [[0.0, 1.0], [1.0, 1.0]]
    assert store.stats()["items"] == 2


def test_custom_selector_can_inspect_current_candidate_pool() -> None:
    observed_pool: list[str] = []

    def selector(context: object, k: int, pool_ids: Sequence[str]) -> list[str]:
        del context, k
        observed_pool.extend(pool_ids)
        return [pool_ids[-1]]

    scheduler = StrategyScheduler(SchedulerConfig(mode="custom", custom_selector=selector))

    selected, snapshot = scheduler.select_batch(["dup", "dup", "available"], 1, object(), state={})

    assert observed_pool == ["dup", "available"]
    assert selected == ["available"]
    assert snapshot == {"mode": "custom"}


def test_legacy_two_argument_custom_selector_still_works() -> None:
    scheduler = StrategyScheduler(
        SchedulerConfig(mode="custom", custom_selector=lambda context, k: ["available"])
    )

    selected, snapshot = scheduler.select_batch(["dup", "available"], 1, object(), state={})

    assert selected == ["available"]
    assert snapshot == {"mode": "custom"}
