from __future__ import annotations

from typing import Sequence

import pytest

from active_learning_sdk.configs import SchedulerConfig
from active_learning_sdk.engine import StrategyScheduler
from active_learning_sdk.exceptions import ConfigurationError
from active_learning_sdk.types import DataSample


class OrderedStrategy:
    def __init__(self, name: str, order: Sequence[str]) -> None:
        self.name = name
        self._order = list(order)

    def select(self, pool_ids: Sequence[str], k: int, context: object) -> list[str]:
        pool = set(pool_ids)
        return [sample_id for sample_id in self._order if sample_id in pool][:k]


class FakeSelectionContext:
    def __init__(self, groups: dict[str, str | None] | None = None, *, fail_groups: bool = False) -> None:
        self._groups = groups or {}
        self._fail_groups = fail_groups

    def get_samples(self, sample_ids: Sequence[str]) -> list[DataSample]:
        if self._fail_groups:
            raise RuntimeError("group provider unavailable")
        return [
            DataSample(sample_id=sample_id, data={"text": sample_id}, group_id=self._groups.get(sample_id))
            for sample_id in sample_ids
        ]


def test_scheduler_config_accepts_mix_interleaved_and_requires_positive_mix() -> None:
    SchedulerConfig(mode="mix_interleaved", mix={"alpha": 1.0}).validate()

    with pytest.raises(ConfigurationError):
        SchedulerConfig(mode="mix_interleaved").validate()

    with pytest.raises(ConfigurationError):
        SchedulerConfig(mode="mix_interleaved", mix={"alpha": 0.0}).validate()


def test_existing_mix_remains_sorted_block_based() -> None:
    scheduler = StrategyScheduler(
        SchedulerConfig(mode="mix", mix={"beta": 0.5, "alpha": 0.5}),
        strategies=[
            OrderedStrategy("alpha", ["a1", "a2", "b1", "b2"]),
            OrderedStrategy("beta", ["b1", "b2", "a1", "a2"]),
        ],
    )

    selected, snapshot = scheduler.select_batch(["a1", "a2", "b1", "b2"], 4, FakeSelectionContext(), state={})

    assert selected == ["a1", "a2", "b1", "b2"]
    assert snapshot["mode"] == "mix"
    assert snapshot["requested_allocations"] == {"beta": 2, "alpha": 2}


def test_mix_interleaved_rotates_arms_in_config_order() -> None:
    scheduler = StrategyScheduler(
        SchedulerConfig(mode="mix_interleaved", mix={"beta": 0.5, "alpha": 0.5}),
        strategies=[
            OrderedStrategy("alpha", ["a1", "a2", "b1", "b2"]),
            OrderedStrategy("beta", ["b1", "b2", "a1", "a2"]),
        ],
    )

    selected, snapshot = scheduler.select_batch(["a1", "a2", "b1", "b2"], 4, FakeSelectionContext(), state={})

    assert selected == ["b1", "a1", "b2", "a2"]
    assert snapshot["mode"] == "mix_interleaved"
    assert snapshot["arm_order"] == ["beta", "alpha"]
    assert snapshot["requested_allocations"] == {"beta": 2, "alpha": 2}
    assert snapshot["actual_allocations"] == {"beta": 2, "alpha": 2}
    assert snapshot["group_lookup_available"] is True
    assert snapshot["selected_group_count"] == 4
    assert snapshot["group_constrained_selected_count"] == 4
    assert snapshot["group_relaxed_fallback_count"] == 0
    assert "strategy_diagnostics" not in snapshot


def test_mix_interleaved_uses_largest_remainder_for_tiny_batches() -> None:
    scheduler = StrategyScheduler(
        SchedulerConfig(mode="mix_interleaved", mix={"class_group": 0.7, "random": 0.3}),
        strategies=[
            OrderedStrategy("class_group", ["c1", "c2", "c3", "r1"]),
            OrderedStrategy("random", ["r1", "r2", "c1", "c2"]),
        ],
    )

    selected, snapshot = scheduler.select_batch(["c1", "c2", "c3", "r1", "r2"], 3, FakeSelectionContext(), state={})

    assert snapshot["requested_allocations"] == {"class_group": 2, "random": 1}
    assert snapshot["actual_allocations"] == {"class_group": 2, "random": 1}
    assert selected == ["c1", "r1", "c2"]


def test_mix_interleaved_avoids_repeated_groups_before_fallback() -> None:
    scheduler = StrategyScheduler(
        SchedulerConfig(mode="mix_interleaved", mix={"beta": 0.5, "alpha": 0.5}),
        strategies=[
            OrderedStrategy("alpha", ["a1", "a2", "b1", "b2"]),
            OrderedStrategy("beta", ["b1", "b2", "a1", "a2"]),
        ],
    )
    context = FakeSelectionContext(
        {
            "b1": "shared",
            "a1": "shared",
            "b2": "beta-only",
            "a2": "alpha-only",
        }
    )

    selected, snapshot = scheduler.select_batch(["b1", "a1", "b2", "a2"], 4, context, state={})

    assert selected[:3] == ["b1", "a2", "b2"]
    assert selected == ["b1", "a2", "b2", "a1"]
    assert snapshot["fallback_requested"] == 1
    assert snapshot["fallback_actual"] == 1
    assert snapshot["group_lookup_available"] is True
    assert snapshot["selected_group_count"] == 3
    assert snapshot["group_constrained_selected_count"] == 3
    assert snapshot["group_relaxed_fallback_count"] == 1


def test_mix_interleaved_fallback_fills_when_unique_groups_are_exhausted() -> None:
    scheduler = StrategyScheduler(
        SchedulerConfig(mode="mix_interleaved", mix={"alpha": 1.0}),
        strategies=[OrderedStrategy("alpha", ["s1", "s2", "s3"])],
    )
    context = FakeSelectionContext({"s1": "same", "s2": "same", "s3": "same"})

    selected, snapshot = scheduler.select_batch(["s1", "s2", "s3"], 3, context, state={})

    assert selected == ["s1", "s2", "s3"]
    assert snapshot["requested_allocations"] == {"alpha": 3}
    assert snapshot["actual_allocations"] == {"alpha": 3}
    assert snapshot["fallback_requested"] == 2
    assert snapshot["fallback_actual"] == 2
    assert snapshot["group_lookup_available"] is True
    assert snapshot["selected_group_count"] == 1
    assert snapshot["group_constrained_selected_count"] == 1
    assert snapshot["group_relaxed_fallback_count"] == 2


def test_mix_interleaved_degrades_to_id_only_when_groups_are_unavailable() -> None:
    scheduler = StrategyScheduler(
        SchedulerConfig(mode="mix_interleaved", mix={"alpha": 1.0}),
        strategies=[OrderedStrategy("alpha", ["s1", "s2", "s3"])],
    )

    selected, snapshot = scheduler.select_batch(
        ["s1", "s2", "s3"],
        3,
        FakeSelectionContext(fail_groups=True),
        state={},
    )

    assert selected == ["s1", "s2", "s3"]
    assert snapshot["group_lookup_available"] is False
    assert snapshot["selected_group_count"] == 3
    assert snapshot["group_constrained_selected_count"] == 3
    assert snapshot["group_relaxed_fallback_count"] == 0
    assert snapshot["fallback_actual"] == 0
