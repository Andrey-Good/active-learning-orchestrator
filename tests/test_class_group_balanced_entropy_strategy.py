from __future__ import annotations

from typing import Sequence

import pytest

from active_learning_sdk.configs import SchedulerConfig
from active_learning_sdk.engine import StrategyScheduler
from active_learning_sdk.exceptions import ConfigurationError
from active_learning_sdk.strategies import ClassGroupBalancedEntropyStrategy
from active_learning_sdk.types import DataSample


class FakeSelectionContext:
    def __init__(
        self,
        *,
        probabilities: dict[str, list[float]],
        groups: dict[str, str | None] | None = None,
    ) -> None:
        self._probabilities = probabilities
        self._groups = groups or {}

    def model_id(self) -> str:
        return "fake-model"

    def predict_proba(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [self._probabilities[sample_id] for sample_id in sample_ids]

    def get_samples(self, sample_ids: Sequence[str]) -> list[DataSample]:
        return [
            DataSample(
                sample_id=sample_id,
                data={"text": sample_id},
                group_id=self._groups.get(sample_id),
            )
            for sample_id in sample_ids
        ]


def test_class_group_balanced_entropy_round_robins_across_predicted_classes() -> None:
    context = FakeSelectionContext(
        probabilities={
            "a1": [0.5, 0.5, 0.0],
            "a2": [0.51, 0.49, 0.0],
            "b1": [0.0, 0.5, 0.5],
            "b2": [0.0, 0.51, 0.49],
        },
        groups={"a1": "ga1", "a2": "ga2", "b1": "gb1", "b2": "gb2"},
    )

    selected = ClassGroupBalancedEntropyStrategy().select(["a1", "a2", "b1", "b2"], 4, context)

    assert selected == ["a1", "b1", "a2", "b2"]


def test_class_group_balanced_entropy_avoids_repeated_groups_with_class_alternative() -> None:
    context = FakeSelectionContext(
        probabilities={
            "a1": [0.5, 0.5, 0.0],
            "a2": [0.51, 0.49, 0.0],
            "b1": [0.0, 0.5, 0.5],
            "b2": [0.0, 0.52, 0.48],
        },
        groups={"a1": "shared", "a2": "group-a", "b1": "shared", "b2": "group-b"},
    )

    selected = ClassGroupBalancedEntropyStrategy().select(["a1", "a2", "b1", "b2"], 2, context)

    assert selected == ["a1", "b2"]


def test_class_group_balanced_entropy_fills_by_class_balanced_order_when_groups_exhaust() -> None:
    context = FakeSelectionContext(
        probabilities={
            "a1": [0.5, 0.5, 0.0],
            "a2": [0.51, 0.49, 0.0],
            "b1": [0.0, 0.5, 0.5],
            "b2": [0.0, 0.51, 0.49],
        },
        groups={"a1": "group-a", "a2": "group-a", "b1": "group-b", "b2": "group-b"},
    )

    selected = ClassGroupBalancedEntropyStrategy().select(["a1", "a2", "b1", "b2"], 4, context)

    assert selected == ["a1", "b1", "a2", "b2"]


def test_class_group_balanced_entropy_treats_missing_group_ids_as_isolated() -> None:
    context = FakeSelectionContext(
        probabilities={
            "u1": [0.5, 0.5, 0.0],
            "u2": [0.51, 0.49, 0.0],
            "g1": [0.0, 0.5, 0.5],
            "g2": [0.0, 0.51, 0.49],
        },
        groups={"u1": None, "u2": None, "g1": "group-g", "g2": "group-g"},
    )

    selected = ClassGroupBalancedEntropyStrategy().select(["u1", "u2", "g1", "g2"], 3, context)

    assert selected == ["u1", "g1", "u2"]


def test_class_group_balanced_entropy_handles_edge_cases_and_duplicate_pool_ids() -> None:
    context = FakeSelectionContext(
        probabilities={
            "s1": [0.5, 0.5],
            "s2": [0.6, 0.4],
        },
        groups={"s1": "group-a", "s2": "group-b"},
    )
    strategy = ClassGroupBalancedEntropyStrategy()

    assert strategy.select(["s1", "s2"], 0, context) == []
    assert strategy.select([], 2, context) == []
    assert strategy.select(["s1", "s1", "s2"], 5, context) == ["s1", "s2"]


def test_class_group_balanced_entropy_rejects_invalid_probability_rows() -> None:
    context = FakeSelectionContext(probabilities={"s1": [0.5, "bad"]})  # type: ignore[list-item]

    with pytest.raises(ConfigurationError, match="must be numeric"):
        ClassGroupBalancedEntropyStrategy().select(["s1"], 1, context)


def test_class_group_balanced_entropy_is_available_through_strategy_scheduler() -> None:
    context = FakeSelectionContext(
        probabilities={
            "a1": [0.5, 0.5, 0.0],
            "a2": [0.51, 0.49, 0.0],
            "b1": [0.0, 0.5, 0.5],
        },
        groups={"a1": "group-a", "a2": "group-a", "b1": "group-b"},
    )
    scheduler = StrategyScheduler(SchedulerConfig(mode="single", strategy="class_group_balanced_entropy"))

    selected, snapshot = scheduler.select_batch(["a1", "a2", "b1"], 2, context, state={})

    assert selected == ["a1", "b1"]
    assert snapshot == {"mode": "single", "strategy": "class_group_balanced_entropy"}
