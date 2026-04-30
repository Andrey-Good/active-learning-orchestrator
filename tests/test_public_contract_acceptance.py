from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from active_learning_sdk import (
    ActiveLearningProject,
    CacheConfig,
    LabelBackendConfig,
    LabelSchema,
    PrelabelConfig,
    SchedulerConfig,
    SplitConfig,
)
from active_learning_sdk.backends.base import RoundProgress, RoundPullResult, RoundPushResult
from active_learning_sdk.dataset.provider import DataFrameDatasetProvider
from active_learning_sdk.exceptions import ConfigurationError, DatasetMismatchError
from active_learning_sdk.types import AnnotationRecord, DataSample, StepKind


class SplitAwareDataset:
    def __init__(self, rows: dict[str, tuple[str, str | None]] | None = None) -> None:
        self._rows = rows or {
            "s1": ("train sample", "train"),
            "s2": ("validation sample", "val"),
            "s3": ("test sample", "test"),
        }

    def iter_sample_ids(self):
        yield from self._rows.keys()

    def get_sample(self, sample_id: str) -> DataSample:
        text, split = self._rows[sample_id]
        meta = {} if split is None else {"split": split}
        return DataSample(sample_id=sample_id, data={"text": text}, meta=meta)

    def get_samples(self, sample_ids: Sequence[str]) -> list[DataSample]:
        return [self.get_sample(sample_id) for sample_id in sample_ids]

    def schema(self) -> dict[str, str]:
        return {"sample_id": "str", "text": "str", "split": "str"}


class NonStringIdDataset:
    def iter_sample_ids(self):
        yield 1

    def get_sample(self, sample_id: str) -> DataSample:
        return DataSample(sample_id=str(sample_id), data={"text": "sample"})

    def schema(self) -> dict[str, str]:
        return {"sample_id": "int", "text": "str"}


class BrokenGetSampleDataset:
    def iter_sample_ids(self):
        yield from ["s1", "s2"]

    def get_sample(self, sample_id: str) -> DataSample:
        if sample_id == "s2":
            raise KeyError("provider lost s2")
        return DataSample(sample_id=sample_id, data={"text": "sample"})

    def schema(self) -> dict[str, str]:
        return {"sample_id": "str", "text": "str"}


class LowConfidenceModel:
    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        return [[0.51, 0.49] for _ in texts]

    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs: Any) -> None:
        return None

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> dict[str, float]:
        return {"accuracy": 1.0}

    def get_model_id(self) -> str:
        return "low-confidence-model"


class FitEvaluateOnlyModel:
    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs: Any) -> None:
        return None

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> dict[str, float]:
        return {"accuracy": 1.0}


class CapturingBackend:
    def __init__(self) -> None:
        self.prelabels_seen: dict[str, Any] | None = None

    def ensure_ready(self, label_schema: LabelSchema) -> dict[str, Any]:
        label_schema.validate()
        return {"backend": "capturing"}

    def push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: dict[str, Any] | None = None,
    ) -> RoundPushResult:
        self.prelabels_seen = dict(prelabels or {})
        return RoundPushResult(
            task_ids={sample.sample_id: f"task:{round_id}:{sample.sample_id}" for sample in samples}
        )

    def poll_round(self, round_id: str, task_ids: Mapping[str, str], policy: Any) -> RoundProgress:
        return RoundProgress(total=len(task_ids), done=len(task_ids), ready_sample_ids=list(task_ids.keys()))

    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        return RoundPullResult(
            annotations={
                sample_id: [AnnotationRecord(annotator_id="a", created_at=1.0, value="positive")]
                for sample_id in task_ids
            }
        )

    def close(self) -> None:
        return None


def test_column_split_mode_uses_sample_metadata_values(tmp_path: Path) -> None:
    project = ActiveLearningProject("acceptance-public-contract", tmp_path, lock=False)

    project.configure(
        dataset=SplitAwareDataset(),
        model=LowConfidenceModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=CapturingBackend(),
        scheduler_config=SchedulerConfig(strategy="random"),
        split_config=SplitConfig(mode="column", split_column="split"),
        cache_config=CacheConfig(enable=False),
    )

    assert project.get_state().splits == {"train": ["s1"], "val": ["s2"], "test": ["s3"]}


def test_get_state_returns_detached_snapshot_not_live_state(tmp_path: Path) -> None:
    project = ActiveLearningProject("acceptance-public-contract", tmp_path, lock=False)
    project.configure(
        dataset=SplitAwareDataset(),
        model=LowConfidenceModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=CapturingBackend(),
        scheduler_config=SchedulerConfig(strategy="random"),
        split_config=SplitConfig(mode="column", split_column="split"),
        cache_config=CacheConfig(enable=False),
    )

    first = project.get_state()
    second = project.get_state()
    first.sample_labels["s1"] = "mutated-outside-api"
    first.splits["train"].append("s2")

    fresh = project.get_state()
    assert first is not second
    assert "s1" not in fresh.sample_labels
    assert fresh.splits == {"train": ["s1"], "val": ["s2"], "test": ["s3"]}


def test_configure_rejects_non_string_dataset_sample_ids_with_sdk_error(tmp_path: Path) -> None:
    project = ActiveLearningProject("acceptance-public-contract", tmp_path, lock=False)

    with pytest.raises(ConfigurationError, match="iter_sample_ids.*string sample_id"):
        project.configure(
            dataset=NonStringIdDataset(),
            model=LowConfidenceModel(),
            label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
            label_backend_config=LabelBackendConfig(backend="custom"),
            label_backend=CapturingBackend(),
            scheduler_config=SchedulerConfig(strategy="random"),
            split_config=SplitConfig(mode="explicit", explicit_splits={"train": ["1"], "val": [], "test": []}),
            cache_config=CacheConfig(enable=False),
        )


def test_configure_wraps_provider_get_sample_failures_in_sdk_error(tmp_path: Path) -> None:
    project = ActiveLearningProject("acceptance-public-contract", tmp_path, lock=False)

    with pytest.raises(ConfigurationError, match="dataset provider"):
        project.configure(
            dataset=BrokenGetSampleDataset(),
            model=LowConfidenceModel(),
            label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
            label_backend_config=LabelBackendConfig(backend="custom"),
            label_backend=CapturingBackend(),
            scheduler_config=SchedulerConfig(strategy="random"),
            split_config=SplitConfig(mode="explicit", explicit_splits={"train": ["s1", "s2"], "val": [], "test": []}),
            cache_config=CacheConfig(enable=False),
        )


def test_configure_rejects_unsupported_label_schema_task(tmp_path: Path) -> None:
    project = ActiveLearningProject("acceptance-public-contract", tmp_path, lock=False)

    with pytest.raises(ConfigurationError, match="Unsupported label_schema.task"):
        project.configure(
            dataset=SplitAwareDataset(),
            model=LowConfidenceModel(),
            label_schema=LabelSchema(task="unsupported_task", labels=["negative", "positive"]),
            label_backend_config=LabelBackendConfig(backend="custom"),
            label_backend=CapturingBackend(),
            scheduler_config=SchedulerConfig(strategy="random"),
            split_config=SplitConfig(mode="column", split_column="split"),
            cache_config=CacheConfig(enable=False),
        )


def test_optional_sklearn_adapter_is_available_from_root_namespace() -> None:
    pytest.importorskip("sklearn")

    from active_learning_sdk import SklearnTextClassifierAdapter

    assert SklearnTextClassifierAdapter.__name__ == "SklearnTextClassifierAdapter"


def test_column_split_mode_uses_dataframe_top_level_columns(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    dataset = pd.DataFrame(
        [
            {"sample_id": "s1", "text": "train sample", "split": "train"},
            {"sample_id": "s2", "text": "validation sample", "split": "val"},
            {"sample_id": "s3", "text": "test sample", "split": "test"},
        ]
    )
    project = ActiveLearningProject("acceptance-public-contract", tmp_path, lock=False)

    project.configure(
        dataset=dataset,
        model=LowConfidenceModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=CapturingBackend(),
        scheduler_config=SchedulerConfig(strategy="random"),
        split_config=SplitConfig(mode="column", split_column="split"),
        cache_config=CacheConfig(enable=False),
    )

    assert project.get_state().splits == {"train": ["s1"], "val": ["s2"], "test": ["s3"]}


def test_dataframe_provider_schema_remains_compatible_with_extra_columns() -> None:
    pd = pytest.importorskip("pandas")
    dataset = pd.DataFrame(
        [
            {"sample_id": "s1", "text": "train sample", "split": "train", "score": 0.8},
            {"sample_id": "s2", "text": "validation sample", "split": "val", "score": 0.2},
        ]
    )

    provider = DataFrameDatasetProvider(dataset)

    assert provider.schema() == {"sample_id": "str", "text": "str"}
    assert provider.get_sample("s1").data["split"] == "train"
    assert provider.get_sample("s1").data["score"] == 0.8


def test_export_labels_rejects_directory_output_path_with_sdk_error(tmp_path: Path) -> None:
    project = ActiveLearningProject("acceptance-public-contract", tmp_path / "project", lock=False)
    project.configure(
        dataset=SplitAwareDataset({"s1": ("train sample", "train")}),
        model=LowConfidenceModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=CapturingBackend(),
        scheduler_config=SchedulerConfig(strategy="random"),
        split_config=SplitConfig(mode="explicit", explicit_splits={"train": ["s1"], "val": [], "test": []}),
        cache_config=CacheConfig(enable=False),
    )
    project.import_labels({"s1": "positive"}, source="seed")
    output_dir = tmp_path / "labels_directory"
    output_dir.mkdir()

    with pytest.raises(ConfigurationError, match="export_labels output_path must be a file path"):
        project.export_labels(output_dir, format="jsonl")


def test_column_split_reconfigure_rejects_changed_assignments(tmp_path: Path) -> None:
    project = ActiveLearningProject("acceptance-public-contract", tmp_path, lock=False)
    common_kwargs = {
        "model": LowConfidenceModel(),
        "label_schema": LabelSchema(task="text_classification", labels=["negative", "positive"]),
        "label_backend_config": LabelBackendConfig(backend="custom"),
        "label_backend": CapturingBackend(),
        "scheduler_config": SchedulerConfig(strategy="random"),
        "split_config": SplitConfig(mode="column", split_column="split"),
        "cache_config": CacheConfig(enable=False),
    }
    project.configure(dataset=SplitAwareDataset(), **common_kwargs)

    with pytest.raises(DatasetMismatchError, match="split assignments changed"):
        project.configure(
            dataset=SplitAwareDataset(
                {
                    "s1": ("train sample", "train"),
                    "s2": ("validation sample", "train"),
                    "s3": ("test sample", "test"),
                }
            ),
            **common_kwargs,
        )


def test_column_split_mode_rejects_missing_split_fields(tmp_path: Path) -> None:
    project = ActiveLearningProject("acceptance-public-contract", tmp_path, lock=False)

    with pytest.raises(ConfigurationError, match="Missing split column 'split'"):
        project.configure(
            dataset=SplitAwareDataset({"s1": ("train sample", "train"), "s2": ("missing split", None)}),
            model=LowConfidenceModel(),
            label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
            label_backend_config=LabelBackendConfig(backend="custom"),
            label_backend=CapturingBackend(),
            scheduler_config=SchedulerConfig(strategy="random"),
            split_config=SplitConfig(mode="column", split_column="split"),
            cache_config=CacheConfig(enable=False),
        )


def test_column_split_mode_rejects_unknown_split_values(tmp_path: Path) -> None:
    project = ActiveLearningProject("acceptance-public-contract", tmp_path, lock=False)

    with pytest.raises(ConfigurationError, match="Unknown split value 'dev'"):
        project.configure(
            dataset=SplitAwareDataset({"s1": ("train sample", "train"), "s2": ("dev sample", "dev")}),
            model=LowConfidenceModel(),
            label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
            label_backend_config=LabelBackendConfig(backend="custom"),
            label_backend=CapturingBackend(),
            scheduler_config=SchedulerConfig(strategy="random"),
            split_config=SplitConfig(mode="column", split_column="split"),
            cache_config=CacheConfig(enable=False),
        )


def test_prelabel_min_confidence_filters_low_confidence_suggestions(tmp_path: Path) -> None:
    backend = CapturingBackend()
    project = ActiveLearningProject("acceptance-prelabel-contract", tmp_path, lock=False)
    project.configure(
        dataset=SplitAwareDataset(),
        model=LowConfidenceModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=backend,
        scheduler_config=SchedulerConfig(strategy="random"),
        split_config=SplitConfig(mode="explicit", explicit_splits={"train": ["s1", "s2"], "val": [], "test": ["s3"]}),
        prelabel_config=PrelabelConfig(enable=True, min_confidence=0.99),
        cache_config=CacheConfig(enable=False),
    )

    assert project.run_step(batch_size=1).step == StepKind.SELECT
    assert project.run_step(batch_size=1).step == StepKind.PUSH

    assert backend.prelabels_seen == {}


def test_prelabel_enable_requires_predict_proba_at_configure_time(tmp_path: Path) -> None:
    project = ActiveLearningProject("acceptance-prelabel-contract", tmp_path, lock=False)

    with pytest.raises(ConfigurationError, match="prelabel_config.*predict_proba"):
        project.configure(
            dataset=SplitAwareDataset(),
            model=FitEvaluateOnlyModel(),
            label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
            label_backend_config=LabelBackendConfig(backend="custom"),
            label_backend=CapturingBackend(),
            scheduler_config=SchedulerConfig(strategy="random"),
            split_config=SplitConfig(mode="explicit", explicit_splits={"train": ["s1", "s2"], "val": [], "test": ["s3"]}),
            prelabel_config=PrelabelConfig(enable=True),
            cache_config=CacheConfig(enable=False),
        )
