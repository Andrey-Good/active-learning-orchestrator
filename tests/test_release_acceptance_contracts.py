from __future__ import annotations

from typing import Sequence

import pytest

from active_learning_sdk.cache import InMemoryCacheStore, PredictionCache
from active_learning_sdk.configs import LabelSchema
from active_learning_sdk.engine import SelectionContext
from active_learning_sdk.exceptions import ConfigurationError
from active_learning_sdk.strategies.uncertainty import EntropyStrategy, LeastConfidenceStrategy, MarginStrategy


class TextProvider:
    def get_texts(self, sample_ids: Sequence[str]) -> list[str]:
        return [str(sample_id) for sample_id in sample_ids]


class StableProbabilityModel:
    def __init__(self, probabilities_by_text: dict[str, list[float]], model_id: str = "stable-model") -> None:
        self.probabilities_by_text = probabilities_by_text
        self._model_id = model_id
        self.calls: list[list[str]] = []

    def get_model_id(self) -> str:
        return self._model_id

    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        text_list = [str(text) for text in texts]
        self.calls.append(text_list)
        return [list(self.probabilities_by_text[text]) for text in text_list]


class RuntimeScopedProbabilityModel:
    def __init__(self, probabilities_by_text: dict[str, list[float]]) -> None:
        self.probabilities_by_text = probabilities_by_text
        self.calls: list[list[str]] = []

    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        text_list = [str(text) for text in texts]
        self.calls.append(text_list)
        return [list(self.probabilities_by_text[text]) for text in text_list]


class ProbabilityContext:
    def __init__(self, probabilities_by_id: dict[str, list[float]]) -> None:
        self.probabilities_by_id = probabilities_by_id

    def model_id(self) -> str:
        return "strategy-contract-model"

    def predict_proba(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        return [list(self.probabilities_by_id[str(sample_id)]) for sample_id in sample_ids]


def _selection_context(model: object, prediction_cache: PredictionCache) -> SelectionContext:
    return SelectionContext(
        provider=TextProvider(),  # type: ignore[arg-type]
        model=model,  # type: ignore[arg-type]
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        prediction_cache=prediction_cache,
        embedding_cache=None,
        labeled_ids=[],
        last_metrics={},
    )


def test_prediction_cache_can_delete_single_entry_without_clearing_namespace() -> None:
    cache = PredictionCache(InMemoryCacheStore())

    cache.set("model", "s1", [0.25, 0.75])
    cache.set("model", "s2", [0.75, 0.25])
    cache.delete("model", "s1")

    assert cache.get("model", "s1") is None
    assert cache.get("model", "s2") == [0.75, 0.25]


def test_selection_context_evicts_and_recomputes_invalid_cached_prediction_rows() -> None:
    cache = PredictionCache(InMemoryCacheStore())
    cache.set("stable-model", "s1", [5.0, 5.0])
    model = StableProbabilityModel({"s1": [0.25, 0.75]})
    context = _selection_context(model, cache)

    assert context.predict_proba(["s1"]) == [[0.25, 0.75]]

    assert model.calls == [["s1"]]
    assert cache.get("stable-model", "s1") == [0.25, 0.75]


def test_runtime_unique_model_id_prevents_aliasing_for_adapters_without_stable_id() -> None:
    store = InMemoryCacheStore()
    first_model = RuntimeScopedProbabilityModel({"s1": [0.9, 0.1]})
    second_model = RuntimeScopedProbabilityModel({"s1": [0.1, 0.9]})
    first_context = _selection_context(first_model, PredictionCache(store))
    second_context = _selection_context(second_model, PredictionCache(store))

    assert first_context.model_id() != "unknown"
    assert second_context.model_id() != "unknown"
    assert first_context.model_id() != second_context.model_id()
    assert first_context.predict_proba(["s1"]) == [[0.9, 0.1]]
    assert second_context.predict_proba(["s1"]) == [[0.1, 0.9]]
    assert first_model.calls == [["s1"]]
    assert second_model.calls == [["s1"]]


@pytest.mark.parametrize("strategy", [EntropyStrategy(), LeastConfidenceStrategy(), MarginStrategy()])
def test_uncertainty_strategies_reject_count_like_probability_rows(strategy: object) -> None:
    context = ProbabilityContext({"s1": [5.0, 5.0]})

    with pytest.raises(ConfigurationError, match="must sum to 1\\.0"):
        strategy.select(["s1"], 1, context)  # type: ignore[attr-defined]


@pytest.mark.parametrize("strategy", [EntropyStrategy(), LeastConfidenceStrategy(), MarginStrategy()])
def test_uncertainty_strategies_reject_one_column_probability_rows(strategy: object) -> None:
    context = ProbabilityContext({"s1": [1.0]})

    with pytest.raises(ConfigurationError, match="at least 2 probability columns"):
        strategy.select(["s1"], 1, context)  # type: ignore[attr-defined]


def test_uncertainty_strategy_accepts_rows_that_sum_to_one_with_float_tolerance() -> None:
    context = ProbabilityContext(
        {
            "s1": [0.5 + 1e-13, 0.5],
            "s2": [0.9, 0.1],
        }
    )

    assert EntropyStrategy().select(["s1", "s2"], 1, context) == ["s1"]
