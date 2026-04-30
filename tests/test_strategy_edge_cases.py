from __future__ import annotations

from typing import Any, Sequence

import pytest

from active_learning_sdk.exceptions import ConfigurationError, StrategyError
from active_learning_sdk.configs import SchedulerConfig
from active_learning_sdk.engine import StrategyScheduler
from active_learning_sdk.strategies import (
    EntropyStrategy,
    LeastConfidenceStrategy,
    MarginStrategy,
    RandomStrategy,
)
from active_learning_sdk.types import DataSample


class FakeProbabilityContext:
    def __init__(self, probabilities: dict[str, Sequence[float]]) -> None:
        self._probabilities = probabilities
        self.calls: list[list[str]] = []

    def model_id(self) -> str:
        return "audit-probability-model"

    def predict_proba(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        ids = [str(sample_id) for sample_id in sample_ids]
        self.calls.append(ids)
        return [list(self._probabilities[sample_id]) for sample_id in ids]


class FakeProbabilityModel:
    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        return [[0.5, 0.5] for _ in texts]


class FakeRuntimeContext(FakeProbabilityContext):
    def __init__(self, probabilities: dict[str, Sequence[float]]) -> None:
        super().__init__(probabilities)
        self.model = FakeProbabilityModel()


class GroupedProbabilityContext(FakeProbabilityContext):
    def __init__(
        self,
        probabilities: dict[str, Sequence[float]],
        groups: dict[str, str],
        labeled_ids: Sequence[str] = (),
    ) -> None:
        super().__init__(probabilities)
        self._groups = groups
        self.labeled_ids = list(labeled_ids)

    def get_samples(self, sample_ids: Sequence[str]) -> list[DataSample]:
        return [
            DataSample(sample_id=str(sample_id), data={"text": str(sample_id)}, group_id=self._groups[str(sample_id)])
            for sample_id in sample_ids
        ]


class SmallSeedDiversityContext(GroupedProbabilityContext):
    label_schema = type("LabelSchemaStub", (), {"labels": ["negative", "positive"]})()

    def __init__(self) -> None:
        super().__init__(
            {
                "labeled": [0.99, 0.01],
                "near_uncertain": [0.5, 0.5],
                "far_diverse": [0.7, 0.3],
                "other": [0.8, 0.2],
            },
            {"labeled": "seen", "near_uncertain": "seen", "far_diverse": "far", "other": "other"},
            labeled_ids=["labeled"],
        )
        self._embeddings = {
            "labeled": [0.0, 0.0],
            "near_uncertain": [0.1, 0.0],
            "far_diverse": [10.0, 0.0],
            "other": [5.0, 0.0],
        }

    def embed(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [self._embeddings[str(sample_id)] for sample_id in sample_ids]


class OutOfPoolStrategy:
    name = "out_of_pool"
    required_capabilities = frozenset()

    def select(self, pool_ids: Sequence[str], k: int, context: object) -> list[str]:
        return ["not-in-pool", "kept", "also-kept"][:k]


class ShortCustomStrategy:
    name = "short_custom"
    required_capabilities = frozenset()

    def select(self, pool_ids: Sequence[str], k: int, context: object) -> list[str]:
        return ["kept"]


class ShortEntropyOverrideStrategy:
    name = "entropy"
    required_capabilities = frozenset()

    def select(self, pool_ids: Sequence[str], k: int, context: object) -> list[str]:
        return ["kept"]


class EmbeddingRequiredEntropyOverrideStrategy:
    name = "entropy"
    required_capabilities = frozenset({"embed"})

    def select(self, pool_ids: Sequence[str], k: int, context: object) -> list[str]:
        return ["kept"]


def _assert_unique_pool_selection(selected: Sequence[str], pool_ids: Sequence[str], k: int) -> None:
    unique_pool = set(pool_ids)
    assert len(selected) == min(k, len(unique_pool))
    assert len(selected) == len(set(selected))
    assert set(selected).issubset(unique_pool)


@pytest.mark.parametrize(
    "strategy",
    [
        EntropyStrategy(),
        LeastConfidenceStrategy(),
        MarginStrategy(),
    ],
)
def test_uncertainty_strategies_do_not_return_duplicate_ids_when_pool_contains_duplicates(strategy: Any) -> None:
    context = FakeProbabilityContext(
        {
            "dup": [0.5, 0.5],
            "other": [0.99, 0.01],
        }
    )

    selected = strategy.select(["dup", "dup", "other"], 2, context)

    _assert_unique_pool_selection(selected, ["dup", "other"], 2)


@pytest.mark.parametrize("strategy", [EntropyStrategy(), LeastConfidenceStrategy(), MarginStrategy()])
def test_uncertainty_strategies_prefer_group_diverse_batches_when_group_ids_are_available(strategy: Any) -> None:
    context = GroupedProbabilityContext(
        {
            "g1_a": [0.5, 0.5],
            "g1_b": [0.5, 0.5],
            "g2_a": [0.52, 0.48],
        },
        {"g1_a": "g1", "g1_b": "g1", "g2_a": "g2"},
    )

    selected = strategy.select(["g1_a", "g1_b", "g2_a"], 2, context)

    assert len(selected) == 2
    assert {context._groups[sample_id] for sample_id in selected} == {"g1", "g2"}


@pytest.mark.parametrize("strategy", [EntropyStrategy(), LeastConfidenceStrategy(), MarginStrategy()])
def test_uncertainty_strategies_avoid_already_labeled_groups_when_group_ids_are_available(strategy: Any) -> None:
    context = GroupedProbabilityContext(
        {
            "g1_labeled": [0.99, 0.01],
            "g1_a": [0.5, 0.5],
            "g2_a": [0.52, 0.48],
            "g3_a": [0.54, 0.46],
        },
        {"g1_labeled": "g1", "g1_a": "g1", "g2_a": "g2", "g3_a": "g3"},
        labeled_ids=["g1_labeled"],
    )

    selected = strategy.select(["g1_a", "g2_a", "g3_a"], 2, context)

    assert selected == ["g2_a", "g3_a"]


def test_entropy_blends_diversity_when_seed_set_is_tiny_and_embeddings_are_available() -> None:
    context = SmallSeedDiversityContext()

    selected = EntropyStrategy().select(["near_uncertain", "far_diverse", "other"], 2, context)

    assert selected == ["far_diverse", "other"]


def test_random_strategy_does_not_return_duplicate_ids_when_k_exceeds_unique_pool_size() -> None:
    selected = RandomStrategy().select(["dup", "dup", "other"], 3, object())

    _assert_unique_pool_selection(selected, ["dup", "other"], 3)


def test_single_strategy_scheduler_fills_after_deduplicating_strategy_output() -> None:
    context = FakeProbabilityContext(
        {
            "dup": [0.5, 0.5],
            "other": [0.99, 0.01],
        }
    )
    scheduler = StrategyScheduler(SchedulerConfig(mode="single", strategy="entropy"))

    selected, snapshot = scheduler.select_batch(["dup", "dup", "other"], 2, context, state={})

    assert snapshot == {"mode": "single", "strategy": "entropy"}
    _assert_unique_pool_selection(selected, ["dup", "other"], 2)


def test_single_strategy_scheduler_rejects_out_of_pool_strategy_ids() -> None:
    scheduler = StrategyScheduler(
        SchedulerConfig(mode="single", strategy="out_of_pool"),
        strategies=[OutOfPoolStrategy()],
    )

    with pytest.raises(ConfigurationError, match="outside the candidate pool"):
        scheduler.select_batch(["kept", "also-kept"], 2, object(), state={})


def test_custom_strategy_scheduler_rejects_out_of_pool_selector_ids() -> None:
    scheduler = StrategyScheduler(
        SchedulerConfig(
            mode="custom",
            custom_selector=lambda context, k: ["not-in-pool", "kept", "also-kept"][:k],
        )
    )

    with pytest.raises(StrategyError, match="outside the candidate pool"):
        scheduler.select_batch(["kept", "also-kept"], 2, object(), state={})


def test_custom_strategy_scheduler_rejects_duplicate_selector_ids() -> None:
    scheduler = StrategyScheduler(
        SchedulerConfig(
            mode="custom",
            custom_selector=lambda context, k: ["kept", "kept", "also-kept"][:k],
        )
    )

    with pytest.raises(StrategyError, match="duplicate sample IDs"):
        scheduler.select_batch(["kept", "also-kept"], 2, object(), state={})


def test_registered_custom_single_strategy_is_not_refilled_when_it_intentionally_underselects() -> None:
    scheduler = StrategyScheduler(
        SchedulerConfig(mode="single", strategy="short_custom"),
        strategies=[ShortCustomStrategy()],
    )

    selected, snapshot = scheduler.select_batch(["kept", "also-kept"], 2, object(), state={})

    assert selected == ["kept"]
    assert snapshot == {"mode": "single", "strategy": "short_custom"}


def test_custom_single_strategy_overriding_builtin_name_is_not_refilled() -> None:
    scheduler = StrategyScheduler(
        SchedulerConfig(mode="single", strategy="entropy"),
        strategies=[ShortEntropyOverrideStrategy()],
    )

    selected, snapshot = scheduler.select_batch(["kept", "also-kept"], 2, object(), state={})

    assert selected == ["kept"]
    assert snapshot == {"mode": "single", "strategy": "entropy"}


def test_strategy_capability_cache_distinguishes_runtime_override_objects() -> None:
    context = FakeRuntimeContext(
        {
            "kept": [0.5, 0.5],
            "also-kept": [0.99, 0.01],
        }
    )
    scheduler = StrategyScheduler(SchedulerConfig(mode="single", strategy="entropy"))
    scheduler.select_batch(["kept", "also-kept"], 1, context, state={})
    scheduler.register_strategy(EmbeddingRequiredEntropyOverrideStrategy())

    with pytest.raises(ConfigurationError, match="requires missing model capability: embed"):
        scheduler.select_batch(["kept", "also-kept"], 1, context, state={})
