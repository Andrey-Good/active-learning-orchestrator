from __future__ import annotations

from typing import Sequence

from active_learning_sdk.configs import LabelSchema, SchedulerConfig
from active_learning_sdk.engine import StrategyScheduler
from active_learning_sdk.strategies import AdaptiveUncertaintyDiversityStrategy, RandomStrategy
from active_learning_sdk.types import DataSample


class FakeAdaptiveContext:
    def __init__(self, *, labeled_count: int, label_count: int = 3) -> None:
        self.labeled_ids = [f"labeled-{index}" for index in range(labeled_count)]
        labels = ["a", "b", "c"] if label_count == 3 else [f"label-{index}" for index in range(label_count)]
        self.label_schema = LabelSchema(task="text_classification", labels=labels)
        self._probabilities = {
            "uncertain_near": [0.5, 0.5, 0.0],
            "confident_far": [0.98, 0.01, 0.01],
            "middle": [0.6, 0.4, 0.0],
        }
        self._embeddings = {
            "uncertain_near": [0.0, 0.0],
            "confident_far": [10.0, 0.0],
            "middle": [0.1, 0.0],
        }

    def model_id(self) -> str:
        return "fake-adaptive-model"

    def predict_proba(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [self._probabilities[sample_id] for sample_id in sample_ids]

    def embed(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [self._embeddings.get(sample_id, [0.0, 0.0]) for sample_id in sample_ids]

    def get_samples(self, sample_ids: Sequence[str]) -> list[DataSample]:
        return [
            DataSample(sample_id=sample_id, data={"text": sample_id}, group_id=sample_id)
            for sample_id in sample_ids
        ]


def test_adaptive_strategy_uses_guarded_diversity_in_early_phase() -> None:
    context = FakeAdaptiveContext(labeled_count=16)

    selected = AdaptiveUncertaintyDiversityStrategy().select(
        ["uncertain_near", "confident_far", "middle"],
        2,
        context,
    )

    assert selected == ["uncertain_near", "confident_far"]


def test_adaptive_strategy_switches_to_entropy_after_enough_labels() -> None:
    context = FakeAdaptiveContext(labeled_count=32)

    selected = AdaptiveUncertaintyDiversityStrategy().select(
        ["confident_far", "middle", "uncertain_near"],
        2,
        context,
    )

    assert selected == ["uncertain_near", "middle"]


def test_adaptive_strategy_uses_random_exploration_for_many_class_cold_start() -> None:
    context = FakeAdaptiveContext(labeled_count=10, label_count=25)
    pool_ids = ["confident_far", "middle", "uncertain_near"]

    selected = AdaptiveUncertaintyDiversityStrategy().select(pool_ids, 2, context)

    assert selected == RandomStrategy().select(pool_ids, 2, context)


def test_adaptive_strategy_is_available_through_scheduler() -> None:
    context = FakeAdaptiveContext(labeled_count=16)
    scheduler = StrategyScheduler(SchedulerConfig(strategy="adaptive_uncertainty_diversity"))

    selected, snapshot = scheduler.select_batch(
        ["uncertain_near", "confident_far", "middle"],
        2,
        context,
        state={},
    )

    assert selected == ["uncertain_near", "confident_far"]
    assert snapshot == {"mode": "single", "strategy": "adaptive_uncertainty_diversity"}
