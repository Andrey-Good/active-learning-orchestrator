from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from active_learning_sdk.configs import CacheConfig, LabelBackendConfig, LabelSchema, SchedulerConfig, SplitConfig
from active_learning_sdk.backends.base import RoundProgress, RoundPullResult, RoundPushResult
from active_learning_sdk.engine import ActiveLearningEngine, StrategyScheduler
from active_learning_sdk.strategies import GroupDiverseEntropyStrategy
from active_learning_sdk.types import DataSample


class FakeSelectionContext:
    def __init__(
        self,
        *,
        probabilities: dict[str, list[float]],
        groups: dict[str, str | None],
        labeled_ids: Sequence[str] = (),
    ) -> None:
        self._probabilities = probabilities
        self._groups = groups
        self.labeled_ids = list(labeled_ids)

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


class InMemoryDataset:
    def __init__(self, sample_ids: Sequence[str] = ("s1", "s2", "s3")) -> None:
        self._samples = {
            sample_id: DataSample(sample_id=sample_id, data={"text": f"text {sample_id}"})
            for sample_id in sample_ids
        }

    def iter_sample_ids(self):
        yield from self._samples.keys()

    def get_sample(self, sample_id: str) -> DataSample:
        return self._samples[sample_id]

    def get_samples(self, sample_ids: Sequence[str]) -> list[DataSample]:
        return [self.get_sample(sample_id) for sample_id in sample_ids]

    def schema(self) -> dict[str, str]:
        return {"sample_id": "str", "text": "str"}


class DummyModel:
    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [[0.5, 0.5] for _ in texts]

    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs) -> None:
        return None

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> dict[str, float]:
        return {"accuracy": 1.0}


class NoopBackend:
    def ensure_ready(self, label_schema: LabelSchema) -> dict[str, Any]:
        return {"backend": "noop"}

    def push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: dict[str, Any] | None = None,
    ) -> RoundPushResult:
        return RoundPushResult(task_ids={sample.sample_id: sample.sample_id for sample in samples})

    def poll_round(
        self,
        round_id: str,
        task_ids: Mapping[str, str],
        policy: Any,
    ) -> RoundProgress:
        return RoundProgress(total=len(task_ids), done=len(task_ids), ready_sample_ids=list(task_ids.keys()))

    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        return RoundPullResult(annotations={})


def _configure_engine(workdir: Path) -> ActiveLearningEngine:
    engine = ActiveLearningEngine("test-project", workdir, lock=False)
    engine.configure(
        dataset=InMemoryDataset(),
        model=DummyModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=NoopBackend(),
        scheduler_config=SchedulerConfig(strategy="random"),
        cache_config=CacheConfig(enable=False),
        split_config=SplitConfig(
            mode="explicit",
            explicit_splits={"train": ["s1", "s2"], "val": ["s3"], "test": []},
        ),
    )
    return engine


def test_group_diverse_entropy_prefers_different_groups_when_possible() -> None:
    context = FakeSelectionContext(
        probabilities={
            "a1": [0.5, 0.5],
            "a2": [0.51, 0.49],
            "b1": [0.6, 0.4],
        },
        groups={"a1": "group-a", "a2": "group-a", "b1": "group-b"},
    )

    selected = GroupDiverseEntropyStrategy().select(["a1", "a2", "b1"], 2, context)

    assert selected == ["a1", "b1"]


def test_group_diverse_entropy_fills_when_groups_are_fewer_than_k() -> None:
    context = FakeSelectionContext(
        probabilities={
            "a1": [0.5, 0.5],
            "a2": [0.51, 0.49],
            "b1": [0.7, 0.3],
        },
        groups={"a1": "group-a", "a2": "group-a", "b1": "group-b"},
    )

    selected = GroupDiverseEntropyStrategy().select(["a1", "a2", "b1"], 3, context)

    assert selected == ["a1", "b1", "a2"]


def test_group_diverse_entropy_avoids_already_labeled_groups_when_possible() -> None:
    context = FakeSelectionContext(
        probabilities={
            "a2": [0.5, 0.5],
            "b1": [0.55, 0.45],
            "c1": [0.6, 0.4],
            "labeled-a": [0.7, 0.3],
        },
        groups={"a2": "group-a", "b1": "group-b", "c1": "group-c", "labeled-a": "group-a"},
        labeled_ids=["labeled-a"],
    )

    selected = GroupDiverseEntropyStrategy().select(["a2", "b1", "c1"], 2, context)

    assert selected == ["b1", "c1"]


def test_group_diverse_entropy_treats_missing_group_ids_as_isolated() -> None:
    context = FakeSelectionContext(
        probabilities={
            "u1": [0.5, 0.5],
            "u2": [0.51, 0.49],
            "g1": [0.6, 0.4],
        },
        groups={"u1": None, "u2": None, "g1": "group-g"},
    )

    selected = GroupDiverseEntropyStrategy().select(["u1", "u2", "g1"], 2, context)

    assert selected == ["u1", "u2"]


def test_group_diverse_entropy_output_is_unique_and_clipped() -> None:
    context = FakeSelectionContext(
        probabilities={
            "a1": [0.5, 0.5],
            "b1": [0.6, 0.4],
        },
        groups={"a1": "group-a", "b1": "group-b"},
    )

    selected = GroupDiverseEntropyStrategy().select(["a1", "a1", "b1"], 5, context)

    assert selected == ["a1", "b1"]


def test_group_diverse_entropy_is_available_through_strategy_scheduler() -> None:
    context = FakeSelectionContext(
        probabilities={
            "a1": [0.5, 0.5],
            "a2": [0.51, 0.49],
            "b1": [0.6, 0.4],
        },
        groups={"a1": "group-a", "a2": "group-a", "b1": "group-b"},
    )
    scheduler = StrategyScheduler(SchedulerConfig(mode="single", strategy="group_diverse_entropy"))

    selected, snapshot = scheduler.select_batch(["a1", "a2", "b1"], 2, context, state={})

    assert selected == ["a1", "b1"]
    assert snapshot == {"mode": "single", "strategy": "group_diverse_entropy"}


def test_group_diverse_entropy_is_registered_for_configured_engine(tmp_path: Path) -> None:
    engine = _configure_engine(tmp_path)
    assert engine._scheduler is not None

    assert "group_diverse_entropy" in engine._scheduler.available_strategies()


def test_group_diverse_entropy_is_registered_after_attach_runtime(tmp_path: Path) -> None:
    engine = _configure_engine(tmp_path)
    engine.close()

    reopened = ActiveLearningEngine("test-project", tmp_path, lock=False)
    reopened.attach_runtime(
        dataset=InMemoryDataset(),
        model=DummyModel(),
        label_backend=NoopBackend(),
    )
    assert reopened._scheduler is not None

    assert "group_diverse_entropy" in reopened._scheduler.available_strategies()
