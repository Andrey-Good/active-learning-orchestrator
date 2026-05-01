from __future__ import annotations

from typing import Any, Sequence

import pytest

from active_learning_sdk.cache import EmbeddingCache, InMemoryCacheStore, PredictionCache
from active_learning_sdk.exceptions import ConfigurationError
from active_learning_sdk.strategies import (
    CommitteeVoteEntropyStrategy,
    EntropyStrategy,
    MarginStrategy,
    McDropoutEntropyStrategy,
)


class ProbabilityContext:
    def __init__(self, probabilities: dict[str, Sequence[float]]) -> None:
        self.probabilities = {sample_id: list(row) for sample_id, row in probabilities.items()}

    def model_id(self) -> str:
        return "w99-probability-model"

    def predict_proba(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        return [self.probabilities[str(sample_id)] for sample_id in sample_ids]


class SingleColumnStochasticContext:
    def model_id(self) -> str:
        return "w99-stochastic-model"

    def predict_stochastic(self, sample_ids: Sequence[str], n: int = 10, batch_size: int = 32) -> list[list[list[float]]]:
        del batch_size
        return [[[1.0] for _ in range(n)] for _ in sample_ids]


class SingleColumnCommitteeContext:
    def model_id(self) -> str:
        return "w99-committee-model"

    def predict_committee(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[list[list[float]]]:
        del batch_size
        return [[[1.0], [1.0]] for _ in sample_ids]


def test_entropy_tie_breaking_is_deterministic_and_pool_order_independent() -> None:
    probabilities = {sample_id: [0.5, 0.5] for sample_id in ("a", "b", "c", "d")}
    context = ProbabilityContext(probabilities)
    strategy = EntropyStrategy()

    first = strategy.select(["a", "b", "c", "d"], 3, context)
    second = strategy.select(["d", "c", "b", "a"], 3, context)
    third = strategy.select(["a", "b", "c", "d"], 3, context)

    assert first == second == third
    assert len(first) == 3
    assert len(first) == len(set(first))
    assert set(first).issubset(probabilities)


@pytest.mark.parametrize("strategy", [EntropyStrategy(), MarginStrategy()])
def test_standard_probability_strategies_reject_single_column_probability_rows(strategy: Any) -> None:
    context = ProbabilityContext({"a": [1.0], "b": [1.0]})

    with pytest.raises(ConfigurationError, match="at least 2 probability columns"):
        strategy.select(["a", "b"], 1, context)


def test_mc_dropout_rejects_single_column_probability_cubes() -> None:
    with pytest.raises(ConfigurationError, match="at least 2 probability columns"):
        McDropoutEntropyStrategy().select(["a", "b"], 1, SingleColumnStochasticContext())


def test_committee_vote_entropy_rejects_single_column_probability_cubes() -> None:
    with pytest.raises(ConfigurationError, match="at least 2 probability columns"):
        CommitteeVoteEntropyStrategy().select(["a", "b"], 1, SingleColumnCommitteeContext())


def test_prediction_cache_does_not_alias_numeric_and_string_sample_ids() -> None:
    cache = PredictionCache(InMemoryCacheStore())

    cache.set("model", 1, [0.1, 0.9])  # type: ignore[arg-type]

    assert cache.get("model", "1") is None


def test_embedding_cache_does_not_alias_numeric_and_string_sample_ids() -> None:
    cache = EmbeddingCache(InMemoryCacheStore())

    cache.set("model", 1, [1.0, 2.0])  # type: ignore[arg-type]

    assert cache.get("model", "1") is None


def test_cache_scope_type_tags_do_not_alias_tag_like_strings() -> None:
    prediction_cache = PredictionCache(InMemoryCacheStore())
    embedding_cache = EmbeddingCache(InMemoryCacheStore())

    prediction_cache.set("model", 1, [0.1, 0.9])  # type: ignore[arg-type]
    embedding_cache.set("model", None, [1.0, 2.0])  # type: ignore[arg-type]

    assert prediction_cache.get("model", "int:1") is None
    assert embedding_cache.get("model", "none:null") is None
