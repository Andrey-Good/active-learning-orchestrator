from __future__ import annotations

from typing import Sequence

import pytest

from active_learning_sdk.configs import SchedulerConfig
from active_learning_sdk.engine import StrategyScheduler
from active_learning_sdk.exceptions import ConfigurationError
from active_learning_sdk.strategies import (
    ClassGroupBalancedEntropyStrategy,
    GroupDiverseEntropyStrategy,
)
from active_learning_sdk.types import DataSample


class _MisorderedGroupContext:
    def __init__(self) -> None:
        self._probabilities = {
            "same_a": [0.50, 0.50],
            "same_b": [0.51, 0.49],
            "other": [0.52, 0.48],
        }
        self._samples = {
            "same_a": DataSample(sample_id="same_a", data={"text": "same_a"}, group_id="shared"),
            "same_b": DataSample(sample_id="same_b", data={"text": "same_b"}, group_id="shared"),
            "other": DataSample(sample_id="other", data={"text": "other"}, group_id="other"),
        }

    def model_id(self) -> str:
        return "deep-audit-group-model"

    def predict_proba(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        return [self._probabilities[sample_id] for sample_id in sample_ids]

    def get_samples(self, sample_ids: Sequence[str]) -> list[DataSample]:
        del sample_ids
        return [self._samples["same_a"], self._samples["other"], self._samples["same_b"]]


class _HybridIncompleteGroupContext:
    def __init__(self) -> None:
        self._probabilities = {
            "a": [0.50, 0.50],
            "b": [0.51, 0.49],
            "c": [0.52, 0.48],
        }
        self._embeddings = {
            "a": [0.0],
            "b": [1.0],
            "c": [2.0],
        }

    def model_id(self) -> str:
        return "deep-audit-hybrid-group-model"

    def predict_proba(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        return [self._probabilities[sample_id] for sample_id in sample_ids]

    def embed(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        return [self._embeddings[sample_id] for sample_id in sample_ids]

    def get_samples(self, sample_ids: Sequence[str]) -> list[DataSample]:
        del sample_ids
        return [
            DataSample(sample_id="a", data={"text": "a"}, group_id="shared"),
            DataSample(sample_id="foreign", data={"text": "foreign"}, group_id="other"),
        ]


def test_group_diverse_entropy_rejects_misordered_group_lookup_results() -> None:
    context = _MisorderedGroupContext()

    with pytest.raises(ConfigurationError, match="get_samples|sample_id|order"):
        GroupDiverseEntropyStrategy().select(["same_a", "same_b", "other"], 2, context)


def test_class_group_balanced_entropy_rejects_misordered_group_lookup_results() -> None:
    context = _MisorderedGroupContext()

    with pytest.raises(ConfigurationError, match="get_samples|sample_id|order"):
        ClassGroupBalancedEntropyStrategy().select(["same_a", "same_b", "other"], 3, context)


def test_hybrid_group_balance_rejects_incomplete_or_foreign_group_lookup_results() -> None:
    scheduler = StrategyScheduler(
        SchedulerConfig(
            mode="hybrid",
            hybrid={
                "mode": "weighted",
                "uncertainty_weight": 1.0,
                "diversity_weight": 0.0,
                "group_balance": True,
            },
        )
    )

    with pytest.raises(ConfigurationError, match="group_balance|get_samples|sample_id"):
        scheduler.select_batch(["a", "b", "c"], 2, _HybridIncompleteGroupContext(), state={})
