from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from active_learning_sdk.configs import LabelSchema, SchedulerConfig
from active_learning_sdk.engine import StrategyScheduler
from active_learning_sdk.strategies import AdaptiveUncertaintyDiversityStrategy, HybridStrategy, RandomStrategy
from active_learning_sdk.types import DataSample


class FakeAdaptiveContext:
    def __init__(self, *, labeled_count: int, label_count: int = 3) -> None:
        self.labeled_ids = [f"labeled-{index}" for index in range(labeled_count)]
        labels = ["a", "b", "c"] if label_count == 3 else [f"label-{index}" for index in range(label_count)]
        self.label_schema = LabelSchema(task="text_classification", labels=labels)
        self._label_count = label_count
        self._strategy_diagnostics: list[dict[str, Any]] = []
        self.predict_proba_call_ids: list[list[str]] = []
        self._base_probabilities = {
            "uncertain_near": [0.45, 0.45, 0.10],
            "confident_far": [0.98, 0.01, 0.01],
            "middle": [0.6, 0.3, 0.1],
            "dense_uncertain_a": [0.34, 0.33, 0.33],
            "dense_uncertain_b": [0.33, 0.34, 0.33],
            "dense_uncertain_c": [0.33, 0.33, 0.34],
            "semantic_far": [0.98, 0.01, 0.01],
            "semantic_far_a": [0.98, 0.01, 0.01],
            "semantic_far_b": [0.97, 0.02, 0.01],
            "semantic_far_c": [0.96, 0.03, 0.01],
            "semantic_far_d": [0.95, 0.04, 0.01],
            "semantic_far_e": [0.94, 0.05, 0.01],
            "near_seed": [0.6, 0.3, 0.1],
        }
        self._embeddings = {
            "uncertain_near": [0.0, 0.0],
            "confident_far": [10.0, 0.0],
            "middle": [0.1, 0.0],
            "dense_uncertain_a": [0.0, 0.0],
            "dense_uncertain_b": [0.1, 0.0],
            "dense_uncertain_c": [0.2, 0.0],
            "semantic_far": [12.0, 0.0],
            "semantic_far_a": [8.0, 0.0],
            "semantic_far_b": [16.0, 0.0],
            "semantic_far_c": [24.0, 0.0],
            "semantic_far_d": [32.0, 0.0],
            "semantic_far_e": [40.0, 0.0],
            "near_seed": [0.2, 0.0],
        }

    def model_id(self) -> str:
        return "fake-adaptive-model"

    def predict_proba(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        self.predict_proba_call_ids.append(list(sample_ids))
        return [self._probability_row(sample_id) for sample_id in sample_ids]

    def embed(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [self._embeddings.get(sample_id, [0.0, 0.0]) for sample_id in sample_ids]

    def get_samples(self, sample_ids: Sequence[str]) -> list[DataSample]:
        return [
            DataSample(sample_id=sample_id, data={"text": sample_id}, group_id=sample_id)
            for sample_id in sample_ids
        ]

    def record_strategy_diagnostic(self, strategy_name: str, diagnostic: Mapping[str, Any]) -> None:
        self._strategy_diagnostics.append({"strategy": strategy_name, **dict(diagnostic)})

    def clear_strategy_diagnostics(self) -> None:
        self._strategy_diagnostics.clear()

    def consume_strategy_diagnostics(self) -> list[dict[str, Any]]:
        diagnostics = list(self._strategy_diagnostics)
        self._strategy_diagnostics.clear()
        return diagnostics

    def _probability_row(self, sample_id: str) -> list[float]:
        base_row = list(self._base_probabilities[sample_id])
        if self._label_count <= len(base_row):
            return base_row[: self._label_count]
        return [*base_row, *([0.0] * (self._label_count - len(base_row)))]


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


def test_adaptive_strategy_uses_diversity_prefilter_hybrid_for_many_class_pool() -> None:
    context = FakeAdaptiveContext(labeled_count=10, label_count=25)
    pool_ids = [
        "dense_uncertain_a",
        "dense_uncertain_b",
        "dense_uncertain_c",
        "semantic_far_a",
        "semantic_far_b",
        "semantic_far_c",
        "semantic_far_d",
        "semantic_far_e",
    ]

    selected = AdaptiveUncertaintyDiversityStrategy().select(pool_ids, 2, context)
    selected_again = AdaptiveUncertaintyDiversityStrategy().select(
        pool_ids,
        2,
        FakeAdaptiveContext(labeled_count=10, label_count=25),
    )
    uncertainty_only = HybridStrategy(
        {
            "mode": "weighted",
            "uncertainty": "entropy",
            "diversity": "coreset_kcenter",
            "uncertainty_weight": 1.0,
            "diversity_weight": 0.0,
        }
    ).select(pool_ids, 2, FakeAdaptiveContext(labeled_count=10, label_count=25)).selected
    random_selected = RandomStrategy().select(pool_ids, 2, FakeAdaptiveContext(labeled_count=10, label_count=25))

    assert len(selected) == 2
    assert selected == selected_again
    assert set(selected).issubset(pool_ids)
    assert len(set(selected)) == len(selected)
    assert any(sample_id.startswith("dense_uncertain") for sample_id in selected)
    assert any(sample_id.startswith("semantic_far") for sample_id in selected)
    assert selected != uncertainty_only
    assert selected != random_selected
    assert context.predict_proba_call_ids == [pool_ids]
    assert context.consume_strategy_diagnostics() == [
        {
            "strategy": "adaptive_uncertainty_diversity",
            "phase": "many_class_diversity_prefilter",
            "label_count": 25,
            "labeled_count": 10,
            "effective_strategy": "hybrid:diversity_prefilter_uncertainty:coreset_kcenter+entropy",
        }
    ]


def test_adaptive_strategy_scheduler_snapshot_includes_diagnostics() -> None:
    context = FakeAdaptiveContext(labeled_count=10, label_count=25)
    scheduler = StrategyScheduler(SchedulerConfig(strategy="adaptive_uncertainty_diversity"))
    pool_ids = [
        "dense_uncertain_a",
        "dense_uncertain_b",
        "dense_uncertain_c",
        "semantic_far_a",
        "semantic_far_b",
        "semantic_far_c",
        "semantic_far_d",
        "semantic_far_e",
    ]

    selected, snapshot = scheduler.select_batch(
        pool_ids,
        2,
        context,
        state={},
    )

    assert len(selected) == 2
    assert set(selected).issubset(pool_ids)
    assert context.predict_proba_call_ids == [pool_ids]
    assert snapshot == {
        "mode": "single",
        "strategy": "adaptive_uncertainty_diversity",
        "strategy_diagnostics": [
            {
                "strategy": "adaptive_uncertainty_diversity",
                "phase": "many_class_diversity_prefilter",
                "label_count": 25,
                "labeled_count": 10,
                "effective_strategy": "hybrid:diversity_prefilter_uncertainty:coreset_kcenter+entropy",
            }
        ],
    }
    assert context.consume_strategy_diagnostics() == []
    json.dumps(snapshot, allow_nan=False)
