from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from active_learning_sdk import (
    ActiveLearningProject,
    CacheConfig,
    ConfigurationError,
    LabelBackendConfig,
    LabelSchema,
    SchedulerConfig,
    SplitConfig,
)
from active_learning_sdk.backends.base import RoundProgress, RoundPullResult, RoundPushResult
from active_learning_sdk.types import AnnotationRecord, DataSample, SampleStatus, StepKind


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


class DuplicateIdDataset:
    def __init__(self) -> None:
        self._sample = DataSample(sample_id="dup", data={"text": "duplicate text"})

    def iter_sample_ids(self):
        yield "dup"
        yield "dup"

    def get_sample(self, sample_id: str) -> DataSample:
        if sample_id != "dup":
            raise KeyError(sample_id)
        return self._sample

    def schema(self) -> dict[str, str]:
        return {"sample_id": "str", "text": "str"}


class DummyModel:
    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [[0.5, 0.5] for _ in texts]

    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs: Any) -> None:
        return None

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> dict[str, float]:
        return {"accuracy": 1.0}


class ShortCachedPredictionModel(DummyModel):
    def get_model_id(self) -> str:
        return "short-proba-model"

    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [[0.5, 0.5] for _ in list(texts)[:-1]]


class MalformedCachedPredictionModel(DummyModel):
    def __init__(self) -> None:
        self.calls = 0

    def get_model_id(self) -> str:
        return "malformed-proba-model"

    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[Any]:
        self.calls += 1
        return [[0.5, 0.5], ["bad"]][: len(list(texts))]


class ReadyBackend:
    def __init__(self, annotations: Mapping[str, Sequence[AnnotationRecord]] | None = None) -> None:
        self.annotations = {sample_id: list(records) for sample_id, records in (annotations or {}).items()}

    def ensure_ready(self, label_schema: LabelSchema) -> dict[str, Any]:
        return {"backend": "ready"}

    def push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: dict[str, Any] | None = None,
    ) -> RoundPushResult:
        return RoundPushResult(task_ids={sample.sample_id: sample.sample_id for sample in samples})

    def poll_round(self, round_id: str, task_ids: Mapping[str, str], policy: Any) -> RoundProgress:
        return RoundProgress(total=len(task_ids), done=len(task_ids), ready_sample_ids=list(task_ids.keys()))

    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        return RoundPullResult(
            annotations={sample_id: list(self.annotations.get(sample_id, [])) for sample_id in task_ids}
        )

    def close(self) -> None:
        return None


class ExtraAnnotationBackend(ReadyBackend):
    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        annotations = {sample_id: list(self.annotations.get(sample_id, [])) for sample_id in task_ids}
        annotations["ghost"] = [AnnotationRecord(annotator_id="annotator", created_at=1.0, value="positive")]
        return RoundPullResult(annotations=annotations)


def _configure_project(
    workdir: Path,
    *,
    dataset: Any | None = None,
    model: Any | None = None,
    backend: Any | None = None,
    scheduler_config: SchedulerConfig | None = None,
    cache_config: CacheConfig | None = None,
    split_config: SplitConfig | None = None,
) -> ActiveLearningProject:
    project = ActiveLearningProject("audit-runtime", workdir, lock=False)
    project.configure(
        dataset=dataset or InMemoryDataset(),
        model=model or DummyModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=backend or ReadyBackend(),
        scheduler_config=scheduler_config or SchedulerConfig(strategy="random"),
        cache_config=cache_config or CacheConfig(enable=False),
        split_config=split_config
        or SplitConfig(
            mode="explicit",
            explicit_splits={"train": ["s1"], "val": ["s2"], "test": ["s3"]},
        ),
    )
    return project


def test_configure_rejects_duplicate_ids_from_custom_dataset_provider(tmp_path: Path) -> None:
    project = ActiveLearningProject("audit-runtime", tmp_path, lock=False)

    with pytest.raises(ConfigurationError, match="Duplicate sample_id"):
        project.configure(
            dataset=DuplicateIdDataset(),
            model=DummyModel(),
            label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
            label_backend_config=LabelBackendConfig(backend="custom"),
            label_backend=ReadyBackend(),
            scheduler_config=SchedulerConfig(strategy="random"),
            cache_config=CacheConfig(enable=False),
            split_config=SplitConfig(mode="explicit", explicit_splits={"train": ["dup"], "val": [], "test": []}),
        )


def test_configure_rejects_explicit_split_ids_that_are_not_in_dataset(tmp_path: Path) -> None:
    project = ActiveLearningProject("audit-runtime", tmp_path, lock=False)

    with pytest.raises(ConfigurationError, match="Unknown split sample_id"):
        project.configure(
            dataset=InMemoryDataset(sample_ids=("s1",)),
            model=DummyModel(),
            label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
            label_backend_config=LabelBackendConfig(backend="custom"),
            label_backend=ReadyBackend(),
            scheduler_config=SchedulerConfig(strategy="random"),
            cache_config=CacheConfig(enable=False),
            split_config=SplitConfig(mode="explicit", explicit_splits={"train": ["s1", "missing"], "val": [], "test": []}),
        )


def test_backend_labels_are_validated_against_label_schema_before_persisting(tmp_path: Path) -> None:
    backend = ReadyBackend(
        annotations={"s1": [AnnotationRecord(annotator_id="annotator", created_at=1.0, value="not-a-schema-label")]}
    )
    project = _configure_project(
        tmp_path,
        dataset=InMemoryDataset(sample_ids=("s1",)),
        backend=backend,
        split_config=SplitConfig(mode="explicit", explicit_splits={"train": ["s1"], "val": [], "test": []}),
    )

    assert project.run_step(batch_size=1).step == StepKind.SELECT
    assert project.run_step(batch_size=1).step == StepKind.PUSH
    assert project.run_step(batch_size=1).step == StepKind.WAIT

    with pytest.raises(ConfigurationError, match="not in LabelSchema.labels"):
        project.run_step(batch_size=1)

    assert project.get_state().sample_status["s1"] != SampleStatus.LABELED.value
    assert "s1" not in project.get_state().sample_labels


def test_cached_predict_proba_row_count_mismatch_raises_configuration_error(tmp_path: Path) -> None:
    project = _configure_project(
        tmp_path,
        model=ShortCachedPredictionModel(),
        scheduler_config=SchedulerConfig(strategy="entropy"),
        cache_config=CacheConfig(enable=True, persist=False),
        split_config=SplitConfig(mode="explicit", explicit_splits={"train": ["s1", "s2", "s3"], "val": [], "test": []}),
    )

    with pytest.raises(ConfigurationError, match="predict_proba returned 2 rows for 3"):
        project.run_step(batch_size=3)


def test_backend_pull_rejects_annotations_for_unknown_sample_ids_before_persisting(tmp_path: Path) -> None:
    backend = ExtraAnnotationBackend(
        annotations={"s1": [AnnotationRecord(annotator_id="annotator", created_at=1.0, value="positive")]}
    )
    project = _configure_project(
        tmp_path,
        dataset=InMemoryDataset(sample_ids=("s1",)),
        backend=backend,
        split_config=SplitConfig(mode="explicit", explicit_splits={"train": ["s1"], "val": [], "test": []}),
    )

    assert project.run_step(batch_size=1).step == StepKind.SELECT
    assert project.run_step(batch_size=1).step == StepKind.PUSH
    assert project.run_step(batch_size=1).step == StepKind.WAIT

    with pytest.raises(ConfigurationError, match="Unknown annotation sample_id"):
        project.run_step(batch_size=1)

    assert "ghost" not in project.get_state().sample_status
    assert "ghost" not in project.get_state().sample_labels


def test_malformed_cached_predict_proba_rows_do_not_poison_cache(tmp_path: Path) -> None:
    model = MalformedCachedPredictionModel()
    project = _configure_project(
        tmp_path,
        dataset=InMemoryDataset(sample_ids=("s1", "s2")),
        model=model,
        scheduler_config=SchedulerConfig(strategy="entropy"),
        cache_config=CacheConfig(enable=True, persist=False),
        split_config=SplitConfig(mode="explicit", explicit_splits={"train": ["s1", "s2"], "val": [], "test": []}),
    )

    with pytest.raises(ConfigurationError, match="predict_proba row 1"):
        project.run_step(batch_size=2)
    with pytest.raises(ConfigurationError, match="predict_proba row 1"):
        project.run_step(batch_size=2)

    assert model.calls == 2
