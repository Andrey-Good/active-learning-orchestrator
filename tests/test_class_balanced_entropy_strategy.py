from __future__ import annotations

from typing import Sequence

import pytest

from active_learning_sdk.configs import LabelSchema, SchedulerConfig
from active_learning_sdk.engine import StrategyScheduler
from active_learning_sdk.exceptions import ConfigurationError
from active_learning_sdk.strategies import ClassBalancedEntropyStrategy, EntropyStrategy


class FakeSelectionContext:
    def __init__(
        self,
        probabilities: dict[str, list[float]],
        *,
        label_schema: LabelSchema | None = None,
        embeddings: dict[str, list[float]] | None = None,
    ) -> None:
        self._probabilities = probabilities
        self.label_schema = label_schema
        self._embeddings = embeddings or {}

    def model_id(self) -> str:
        return "fake-model"

    def predict_proba(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [self._probabilities[sample_id] for sample_id in sample_ids]

    def embed(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [self._embeddings[sample_id] for sample_id in sample_ids]


def test_class_balanced_entropy_round_robins_across_predicted_classes() -> None:
    context = FakeSelectionContext(
        {
            "a1": [0.5, 0.5, 0.0],
            "a2": [0.51, 0.49, 0.0],
            "a3": [0.52, 0.48, 0.0],
            "b1": [0.0, 0.6, 0.4],
        }
    )

    selected = ClassBalancedEntropyStrategy().select(["a1", "a2", "a3", "b1"], 3, context)

    assert selected == ["a1", "b1", "a2"]


def test_class_balanced_entropy_fills_from_remaining_class_when_others_exhaust() -> None:
    context = FakeSelectionContext(
        {
            "a1": [0.5, 0.5],
            "a2": [0.51, 0.49],
            "b1": [0.4, 0.6],
        }
    )

    selected = ClassBalancedEntropyStrategy().select(["a1", "a2", "b1"], 3, context)

    assert selected == ["a1", "b1", "a2"]


def test_class_balanced_entropy_matches_entropy_when_one_predicted_class_exists() -> None:
    context = FakeSelectionContext(
        {
            "s1": [0.6, 0.4],
            "s2": [0.7, 0.3],
            "s3": [0.8, 0.2],
        }
    )

    selected = ClassBalancedEntropyStrategy().select(["s3", "s2", "s1"], 2, context)

    assert selected == ["s1", "s2"]


def test_class_balanced_entropy_is_deterministic_for_ties() -> None:
    context = FakeSelectionContext(
        {
            "a1": [0.5, 0.5, 0.0],
            "a2": [0.5, 0.5, 0.0],
            "b1": [0.0, 0.5, 0.5],
            "b2": [0.0, 0.5, 0.5],
        }
    )
    strategy = ClassBalancedEntropyStrategy()

    first = strategy.select(["a1", "a2", "b1", "b2"], 4, context)
    second = strategy.select(["a1", "a2", "b1", "b2"], 4, context)

    assert first == second
    assert first[0] in {"a1", "a2"}
    assert first[1] in {"b1", "b2"}


def test_class_balanced_entropy_handles_edge_cases_and_duplicate_pool_ids() -> None:
    context = FakeSelectionContext(
        {
            "s1": [0.5, 0.5],
            "s2": [0.6, 0.4],
        }
    )
    strategy = ClassBalancedEntropyStrategy()

    assert strategy.select(["s1", "s2"], 0, context) == []
    assert strategy.select([], 2, context) == []
    assert strategy.select(["s1", "s1", "s2"], 5, context) == ["s1", "s2"]


def test_class_balanced_entropy_rejects_count_like_probability_rows() -> None:
    context = FakeSelectionContext(
        {
            "a1": [5.0, 5.0, 0.0],
            "b1": [0.0, 6.0, 4.0],
        }
    )

    with pytest.raises(ConfigurationError, match="must sum to 1\\.0"):
        ClassBalancedEntropyStrategy().select(["a1", "b1"], 2, context)


def test_class_balanced_entropy_rejects_invalid_probability_rows() -> None:
    context = FakeSelectionContext({"s1": [0.5, "bad"]})  # type: ignore[list-item]

    with pytest.raises(ConfigurationError, match="must be numeric"):
        ClassBalancedEntropyStrategy().select(["s1"], 1, context)


def test_class_balanced_entropy_rejects_inconsistent_probability_widths() -> None:
    context = FakeSelectionContext(
        {
            "s1": [0.5, 0.5],
            "s2": [0.3, 0.3, 0.4],
        }
    )

    with pytest.raises(ConfigurationError, match="has width 3; expected 2"):
        ClassBalancedEntropyStrategy().select(["s1", "s2"], 1, context)


def test_class_balanced_entropy_is_available_through_strategy_scheduler() -> None:
    context = FakeSelectionContext(
        {
            "a1": [0.5, 0.5, 0.0],
            "a2": [0.51, 0.49, 0.0],
            "b1": [0.0, 0.6, 0.4],
        }
    )
    scheduler = StrategyScheduler(SchedulerConfig(mode="single", strategy="class_balanced_entropy"))

    selected, snapshot = scheduler.select_batch(["a1", "a2", "b1"], 2, context, state={})

    assert selected == ["a1", "b1"]
    assert snapshot == {"mode": "single", "strategy": "class_balanced_entropy"}


def test_entropy_uses_diversity_guardrail_when_probability_support_is_sparse() -> None:
    context = FakeSelectionContext(
        {
            "known_1": [0.5, 0.5, 0.0, 0.0, 0.0],
            "known_2": [0.51, 0.49, 0.0, 0.0, 0.0],
            "novel_far": [0.99, 0.01, 0.0, 0.0, 0.0],
        },
        label_schema=LabelSchema(task="text_classification", labels=["a", "b", "c", "d", "e"]),
        embeddings={
            "known_1": [0.0, 0.0],
            "known_2": [0.1, 0.0],
            "novel_far": [10.0, 0.0],
        },
    )

    selected = EntropyStrategy().select(["known_1", "known_2", "novel_far"], 2, context)

    assert selected == ["novel_far", "known_1"]


def test_entropy_keeps_pure_uncertainty_when_probability_support_is_complete() -> None:
    context = FakeSelectionContext(
        {
            "uncertain": [0.5, 0.5],
            "confident": [0.99, 0.01],
            "middle": [0.7, 0.3],
        },
        label_schema=LabelSchema(task="text_classification", labels=["a", "b"]),
        embeddings={
            "uncertain": [0.0, 0.0],
            "confident": [10.0, 0.0],
            "middle": [5.0, 0.0],
        },
    )

    selected = EntropyStrategy().select(["confident", "middle", "uncertain"], 2, context)

    assert selected == ["uncertain", "middle"]


def test_entropy_cold_start_guardrail_falls_back_to_random_without_embeddings() -> None:
    class NoEmbeddingContext(FakeSelectionContext):
        def embed(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[list[float]]:
            raise ConfigurationError("Model does not support embeddings")

    context = NoEmbeddingContext(
        {
            "a": [0.5, 0.5, 0.0, 0.0, 0.0],
            "b": [0.51, 0.49, 0.0, 0.0, 0.0],
            "c": [0.99, 0.01, 0.0, 0.0, 0.0],
        },
        label_schema=LabelSchema(task="text_classification", labels=["a", "b", "c", "d", "e"]),
    )

    selected = EntropyStrategy().select(["a", "b", "c"], 2, context)

    assert len(selected) == 2
    assert len(selected) == len(set(selected))
    assert set(selected).issubset({"a", "b", "c"})
