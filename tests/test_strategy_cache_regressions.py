from __future__ import annotations

from active_learning_sdk.cache import EmbeddingCache, InMemoryCacheStore, PredictionCache


def test_prediction_cache_keys_do_not_alias_model_and_sample_delimiters() -> None:
    cache = PredictionCache(InMemoryCacheStore())

    cache.set("model::alpha", "sample", [0.1, 0.9])

    assert cache.get("model", "alpha::sample") is None


def test_embedding_cache_keys_do_not_alias_model_and_sample_delimiters() -> None:
    cache = EmbeddingCache(InMemoryCacheStore())

    cache.set("model::alpha", "sample", [1.0, 2.0])

    assert cache.get("model", "alpha::sample") is None
