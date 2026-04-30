from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pytest

from active_learning_sdk import (
    ActiveLearningProject,
    CacheConfig,
    ConfigurationError,
    DatasetMismatchError,
    LabelBackendConfig,
    LabelSchema,
    SchedulerConfig,
    SplitConfig,
    StateCorruptedError,
)
from active_learning_sdk.backends.simulator import SimulatorLabelBackend
from active_learning_sdk.cache import InMemoryCacheStore
from active_learning_sdk.engine import SelectionContext
from active_learning_sdk.strategies import EntropyStrategy
from active_learning_sdk.types import DataSample, StepKind


class _NoopModel:
    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [[0.5, 0.5] for _ in texts]

    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs: Any) -> None:
        return None

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> dict[str, float]:
        return {"accuracy": 0.0}


class _Dataset:
    def __init__(self, samples: dict[str, DataSample]) -> None:
        self._samples = samples

    def iter_sample_ids(self):
        yield from self._samples.keys()

    def get_sample(self, sample_id: str) -> DataSample:
        return self._samples[sample_id]

    def get_samples(self, sample_ids: Sequence[str]) -> list[DataSample]:
        return [self.get_sample(sample_id) for sample_id in sample_ids]

    def get_texts(self, sample_ids: Sequence[str]) -> list[str]:
        return [str(self._samples[sample_id].data["text"]) for sample_id in sample_ids]

    def schema(self) -> dict[str, str]:
        return {"sample_id": "str", "text": "str"}


class _InvalidProbabilityModel(_NoopModel):
    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [[5.0, 5.0] for _ in texts]


class _TwoColumnProbabilityContext:
    label_schema = LabelSchema(task="text_classification", labels=["a", "b", "c"])

    def model_id(self) -> str:
        return "two-column-model"

    def predict_proba(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [[0.5, 0.5] for _ in sample_ids]


def _configure_column_split_project(workdir: Path, dataset: pd.DataFrame) -> None:
    project = ActiveLearningProject("hard-audit-column-split", workdir, lock=False)
    project.configure(
        dataset=dataset,
        model=_NoopModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="simulator"),
        label_backend=SimulatorLabelBackend(),
        scheduler_config=SchedulerConfig(mode="single", strategy="random"),
        cache_config=CacheConfig(enable=False),
        split_config=SplitConfig(mode="column", split_column="split"),
    )
    project.close()


def test_select_step_never_queries_validation_or_test_samples(tmp_path: Path) -> None:
    dataset = _Dataset(
        {
            "train-1": DataSample(sample_id="train-1", data={"text": "train"}),
            "val-1": DataSample(sample_id="val-1", data={"text": "validation"}),
            "test-1": DataSample(sample_id="test-1", data={"text": "test"}),
        }
    )
    project = ActiveLearningProject("hard-audit-selection-pool", tmp_path, lock=False)
    project.configure(
        dataset=dataset,
        model=_NoopModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="simulator"),
        label_backend=SimulatorLabelBackend(),
        scheduler_config=SchedulerConfig(mode="single", strategy="random"),
        cache_config=CacheConfig(enable=False),
        split_config=SplitConfig(
            mode="explicit",
            explicit_splits={"train": ["train-1"], "val": ["val-1"], "test": ["test-1"]},
        ),
    )

    result = project.run_step(batch_size=3)

    assert result.step == StepKind.SELECT
    assert set(project.get_state().rounds[-1].selected_sample_ids) <= {"train-1"}


def test_attach_runtime_rejects_column_split_drift_hidden_by_fast_fingerprint(tmp_path: Path) -> None:
    original = pd.DataFrame(
        [
            {"sample_id": "s1", "text": "same text one", "split": "train"},
            {"sample_id": "s2", "text": "same text two", "split": "val"},
        ]
    )
    drifted = pd.DataFrame(
        [
            {"sample_id": "s1", "text": "same text one", "split": "val"},
            {"sample_id": "s2", "text": "same text two", "split": "train"},
        ]
    )

    _configure_column_split_project(tmp_path, original)

    reopened = ActiveLearningProject("hard-audit-column-split", tmp_path, lock=False)
    with pytest.raises((ConfigurationError, DatasetMismatchError, StateCorruptedError)):
        reopened.attach_runtime(
            dataset=drifted,
            model=_NoopModel(),
            label_backend=SimulatorLabelBackend(),
        )


def test_cache_disabled_selection_context_still_rejects_invalid_probabilities() -> None:
    context = SelectionContext(
        provider=_Dataset({"s1": DataSample(sample_id="s1", data={"text": "one"})}),
        model=_InvalidProbabilityModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        prediction_cache=None,
        embedding_cache=None,
        labeled_ids=[],
        last_metrics={},
    )

    with pytest.raises(ConfigurationError, match="sum to 1.0"):
        context.predict_proba(["s1"])


def test_uncertainty_strategy_rejects_probability_width_mismatching_label_schema() -> None:
    with pytest.raises(ConfigurationError, match="label"):
        EntropyStrategy().select(["s1", "s2"], 1, _TwoColumnProbabilityContext())


def test_zero_sized_in_memory_cache_does_not_crash_with_internal_stop_iteration() -> None:
    store = InMemoryCacheStore(max_items=0)
    store.set("k", "v")
    assert store.stats()["items"] == 0


def test_status_active_round_is_none_after_round_update(tmp_path: Path) -> None:
    dataset = _Dataset({"s1": DataSample(sample_id="s1", data={"text": "one"})})
    backend = SimulatorLabelBackend()
    project = ActiveLearningProject("hard-audit-status", tmp_path, lock=False)
    project.configure(
        dataset=dataset,
        model=_NoopModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="simulator"),
        label_backend=backend,
        scheduler_config=SchedulerConfig(mode="single", strategy="random"),
        cache_config=CacheConfig(enable=False),
        split_config=SplitConfig(mode="explicit", explicit_splits={"train": ["s1"], "val": [], "test": []}),
    )

    assert project.run_step(batch_size=1).step == StepKind.SELECT
    assert project.run_step(batch_size=1).step == StepKind.PUSH
    round_id = project.get_state().rounds[-1].round_id
    backend.submit_annotation(round_id=round_id, sample_id="s1", value="positive")
    assert project.run_step(batch_size=1).step == StepKind.WAIT
    assert project.run_step(batch_size=1).step == StepKind.PULL
    assert project.run_step(batch_size=1).step == StepKind.TRAIN_EVAL
    assert project.run_step(batch_size=1).step == StepKind.UPDATE

    assert project.status()["active_round"] is None
