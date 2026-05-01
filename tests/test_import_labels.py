from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from active_learning_sdk import (
    ActiveLearningError,
    ActiveLearningProject,
    CacheConfig,
    ConfigurationError,
    LabelBackendConfig,
    LabelSchema,
    SchedulerConfig,
    SplitConfig,
)
from active_learning_sdk.backends.base import RoundProgress, RoundPullResult, RoundPushResult
from active_learning_sdk.types import DataSample, StepKind


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


class FitRequiredModel:
    def __init__(self) -> None:
        self.fit_count = 0
        self.predict_count = 0
        self.evaluate_count = 0
        self.fit_texts: list[str] = []
        self.fit_labels: list[Any] = []

    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        if self.fit_count == 0:
            raise RuntimeError("model is not fitted")
        self.predict_count += 1
        return [[0.8, 0.2] for _ in texts]

    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs) -> None:
        self.fit_count += 1
        self.fit_texts = list(texts)
        self.fit_labels = list(labels)

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> dict[str, float]:
        self.evaluate_count += 1
        return {"accuracy": 0.5}


class PredictReadyStableModel(FitRequiredModel):
    def get_model_id(self) -> str:
        return "seed-trained-model-v1"

    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        self.predict_count += 1
        return [[0.8, 0.2] for _ in texts]


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

    def close(self) -> None:
        return None


def _configured_project(
    workdir: Path,
    *,
    label_schema: LabelSchema | None = None,
    model: Any | None = None,
    scheduler_config: SchedulerConfig | None = None,
) -> ActiveLearningProject:
    project = ActiveLearningProject("test-project", workdir, lock=False)
    project.configure(
        dataset=InMemoryDataset(),
        model=model or DummyModel(),
        label_schema=label_schema or LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=NoopBackend(),
        scheduler_config=scheduler_config or SchedulerConfig(strategy="random"),
        cache_config=CacheConfig(enable=False),
        split_config=SplitConfig(
            mode="explicit",
            explicit_splits={"train": ["s1", "s2", "s3"], "val": [], "test": []},
        ),
    )
    return project


def test_import_labels_updates_status_and_persists(tmp_path: Path) -> None:
    project = _configured_project(tmp_path)

    summary = project.import_labels({"s1": "positive", "s2": "negative"}, source="seed")

    assert summary == {"imported": 2, "unchanged": 0, "overwritten": 0, "total": 2, "source": "seed"}
    assert project.status()["counts"] == {"labeled": 2, "unlabeled": 1, "needs_review": 0, "invalid": 0}
    assert project.get_state().sample_labels == {"s1": "positive", "s2": "negative"}

    reopened = ActiveLearningProject("test-project", tmp_path, lock=False)
    assert reopened.status()["counts"]["labeled"] == 2
    assert reopened.get_state().sample_labels == {"s1": "positive", "s2": "negative"}


def test_import_labels_is_idempotent_for_same_labels(tmp_path: Path) -> None:
    project = _configured_project(tmp_path)
    project.import_labels({"s1": "positive"})

    summary = project.import_labels({"s1": "positive"})

    assert summary == {"imported": 0, "unchanged": 1, "overwritten": 0, "total": 1}
    assert project.status()["counts"]["labeled"] == 1


def test_import_labels_rejects_unknown_sample_id(tmp_path: Path) -> None:
    project = _configured_project(tmp_path)

    with pytest.raises(ConfigurationError, match="Unknown sample_id='missing'"):
        project.import_labels({"missing": "positive"})

    assert project.status()["counts"]["labeled"] == 0


def test_import_labels_rejects_invalid_single_label(tmp_path: Path) -> None:
    project = _configured_project(tmp_path)

    with pytest.raises(ConfigurationError, match="not in LabelSchema.labels"):
        project.import_labels({"s1": "neutral"})

    assert project.status()["counts"]["labeled"] == 0


def test_import_labels_rejects_conflict_without_overwrite(tmp_path: Path) -> None:
    project = _configured_project(tmp_path)
    project.import_labels({"s1": "positive"})

    with pytest.raises(ActiveLearningError, match="Conflicting label"):
        project.import_labels({"s1": "negative"})

    assert project.get_state().sample_labels["s1"] == "positive"


def test_import_labels_conflict_rejects_entire_batch(tmp_path: Path) -> None:
    project = _configured_project(tmp_path)
    project.import_labels({"s1": "positive"})

    with pytest.raises(ActiveLearningError, match="Conflicting label"):
        project.import_labels({"s2": "negative", "s1": "negative"})

    assert project.get_state().sample_labels == {"s1": "positive"}
    assert project.status()["counts"]["labeled"] == 1


def test_import_labels_can_overwrite_existing_label(tmp_path: Path) -> None:
    project = _configured_project(tmp_path)
    project.import_labels({"s1": "positive"})

    summary = project.import_labels({"s1": "negative"}, overwrite=True)

    assert summary == {"imported": 0, "unchanged": 0, "overwritten": 1, "total": 1}
    assert project.get_state().sample_labels["s1"] == "negative"
    assert project.status()["counts"]["labeled"] == 1


def test_import_labels_validates_and_canonicalizes_multi_label_values(tmp_path: Path) -> None:
    project = _configured_project(
        tmp_path,
        label_schema=LabelSchema(
            task="text_classification",
            labels=["news", "sports", "finance"],
            multi_label=True,
        ),
    )

    summary = project.import_labels({"s1": ["finance", "news"]})

    assert summary == {"imported": 1, "unchanged": 0, "overwritten": 0, "total": 1}
    assert project.get_state().sample_labels["s1"] == ["news", "finance"]
    assert project.import_labels({"s1": ["news", "finance"]})["unchanged"] == 1


def test_import_labels_rejects_invalid_multi_label_value(tmp_path: Path) -> None:
    project = _configured_project(
        tmp_path,
        label_schema=LabelSchema(
            task="text_classification",
            labels=["news", "sports"],
            multi_label=True,
        ),
    )

    with pytest.raises(ConfigurationError, match="not in LabelSchema.labels"):
        project.import_labels({"s1": ["news", "finance"]})

    assert project.status()["counts"]["labeled"] == 0


def test_run_step_trains_on_seed_labels_before_first_entropy_select(tmp_path: Path) -> None:
    model = FitRequiredModel()
    project = _configured_project(
        tmp_path,
        model=model,
        scheduler_config=SchedulerConfig(strategy="entropy"),
    )
    project.import_labels({"s1": "positive", "s2": "negative"}, source="seed")

    seed_result = project.run_step(batch_size=1)

    assert seed_result.step == StepKind.TRAIN_EVAL
    assert seed_result.round_id is None
    assert seed_result.details["seed"] is True
    assert model.fit_count == 1
    assert model.predict_count == 0
    assert model.fit_texts == ["text s1", "text s2"]
    assert model.fit_labels == ["positive", "negative"]
    assert project.get_state().rounds == []
    assert project.get_state().metrics_history[0].step == "seed_eval"

    select_result = project.run_step(batch_size=1)

    assert select_result.step == StepKind.SELECT
    assert model.fit_count == 1
    assert model.predict_count == 1
    assert project.get_state().rounds[0].selected_sample_ids == ["s3"]


def test_run_step_does_not_seed_train_without_labeled_train_samples(tmp_path: Path) -> None:
    model = FitRequiredModel()
    project = _configured_project(
        tmp_path,
        model=model,
        scheduler_config=SchedulerConfig(strategy="entropy"),
    )

    with pytest.raises(ActiveLearningError, match="model.predict_proba failed: model is not fitted"):
        project.run_step(batch_size=1)

    assert model.fit_count == 0
    assert project.get_state().metrics_history == []


def test_seed_train_does_not_repeat_once_metrics_exist(tmp_path: Path) -> None:
    model = FitRequiredModel()
    project = _configured_project(
        tmp_path,
        model=model,
        scheduler_config=SchedulerConfig(strategy="entropy"),
    )
    project.import_labels({"s1": "positive", "s2": "negative"}, source="seed")

    assert project.run_step(batch_size=1).step == StepKind.TRAIN_EVAL
    assert project.run_step(batch_size=1).step == StepKind.SELECT

    assert model.fit_count == 1
    assert [record.step for record in project.get_state().metrics_history] == ["seed_eval"]


def test_seed_train_repeats_for_volatile_model_after_restart_before_first_select(tmp_path: Path) -> None:
    first_model = FitRequiredModel()
    project = _configured_project(
        tmp_path,
        model=first_model,
        scheduler_config=SchedulerConfig(strategy="entropy"),
    )
    project.import_labels({"s1": "positive", "s2": "negative"}, source="seed")

    assert project.run_step(batch_size=1).step == StepKind.TRAIN_EVAL
    assert first_model.fit_count == 1
    assert project.get_state().rounds == []
    assert [record.step for record in project.get_state().metrics_history] == ["seed_eval"]
    project.close()

    fresh_model = FitRequiredModel()
    reopened = ActiveLearningProject("test-project", tmp_path, lock=False)
    reopened.attach_runtime(
        dataset=InMemoryDataset(),
        model=fresh_model,
        label_backend=NoopBackend(),
    )

    restarted_seed_result = reopened.run_step(batch_size=1)

    assert restarted_seed_result.step == StepKind.TRAIN_EVAL
    assert restarted_seed_result.round_id is None
    assert restarted_seed_result.details["seed"] is True
    assert fresh_model.fit_count == 1
    assert fresh_model.predict_count == 0
    assert reopened.get_state().rounds == []
    assert [record.step for record in reopened.get_state().metrics_history] == ["seed_eval", "seed_eval"]

    select_result = reopened.run_step(batch_size=1)
    assert select_result.step == StepKind.SELECT
    assert fresh_model.fit_count == 1
    assert fresh_model.predict_count == 1
    assert reopened.get_state().rounds[0].selected_sample_ids == ["s3"]


def test_seed_train_repeats_for_volatile_model_after_same_engine_runtime_rebind_before_first_select(tmp_path: Path) -> None:
    first_model = FitRequiredModel()
    project = _configured_project(
        tmp_path,
        model=first_model,
        scheduler_config=SchedulerConfig(strategy="entropy"),
    )
    project.import_labels({"s1": "positive", "s2": "negative"}, source="seed")

    assert project.run_step(batch_size=1).step == StepKind.TRAIN_EVAL
    assert first_model.fit_count == 1
    assert project.get_state().rounds == []

    fresh_model = FitRequiredModel()
    project.attach_runtime(
        dataset=InMemoryDataset(),
        model=fresh_model,
        label_backend=NoopBackend(),
    )

    rebound_seed_result = project.run_step(batch_size=1)

    assert rebound_seed_result.step == StepKind.TRAIN_EVAL
    assert rebound_seed_result.round_id is None
    assert rebound_seed_result.details["seed"] is True
    assert fresh_model.fit_count == 1
    assert fresh_model.predict_count == 0
    assert project.get_state().rounds == []
    assert [record.step for record in project.get_state().metrics_history] == ["seed_eval", "seed_eval"]

    select_result = project.run_step(batch_size=1)
    assert select_result.step == StepKind.SELECT
    assert fresh_model.fit_count == 1
    assert fresh_model.predict_count == 1
    assert project.get_state().rounds[0].selected_sample_ids == ["s3"]


def test_seed_train_does_not_repeat_for_same_stable_model_id_after_restart(tmp_path: Path) -> None:
    first_model = PredictReadyStableModel()
    project = _configured_project(
        tmp_path,
        model=first_model,
        scheduler_config=SchedulerConfig(strategy="entropy"),
    )
    project.import_labels({"s1": "positive", "s2": "negative"}, source="seed")

    assert project.run_step(batch_size=1).step == StepKind.TRAIN_EVAL
    assert first_model.fit_count == 1
    project.close()

    fresh_model = PredictReadyStableModel()
    reopened = ActiveLearningProject("test-project", tmp_path, lock=False)
    reopened.attach_runtime(
        dataset=InMemoryDataset(),
        model=fresh_model,
        label_backend=NoopBackend(),
    )

    select_result = reopened.run_step(batch_size=1)

    assert select_result.step == StepKind.SELECT
    assert fresh_model.fit_count == 0
    assert fresh_model.predict_count == 1
    assert [record.step for record in reopened.get_state().metrics_history] == ["seed_eval"]
    assert reopened.get_state().rounds[0].selected_sample_ids == ["s3"]
