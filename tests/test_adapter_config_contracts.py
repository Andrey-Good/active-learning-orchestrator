from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from active_learning_sdk import (
    ActiveLearningProject,
    AnnotationPolicy,
    CacheConfig,
    ConfigurationError,
    LabelBackendConfig,
    LabelSchema,
    PrelabelConfig,
    SchedulerConfig,
    SplitConfig,
    StopCriteria,
)
from active_learning_sdk.adapters.base import TextClassificationAdapter
from active_learning_sdk.adapters.huggingface import HFSequenceClassifierAdapter
from active_learning_sdk.backends.base import RoundProgress, RoundPullResult, RoundPushResult
from active_learning_sdk.types import DataSample


class MinimalRequiredAdapter:
    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [[0.5, 0.5] for _ in texts]

    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs: Any) -> None:
        return None

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> dict[str, float]:
        return {"accuracy": 0.0}


def test_runtime_protocol_accepts_documented_minimal_text_classifier_adapter() -> None:
    """The public protocol says only predict_proba/fit/evaluate are required for MVP use."""

    assert isinstance(MinimalRequiredAdapter(), TextClassificationAdapter)


@pytest.mark.parametrize(
    ("train_ratio", "val_ratio", "test_ratio"),
    [
        (-0.1, 1.1, 0.0),
        (1.1, 0.0, -0.1),
    ],
)
def test_split_config_rejects_negative_random_split_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    with pytest.raises(ConfigurationError, match="ratio"):
        SplitConfig(
            mode="random",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        ).validate()


def test_annotation_policy_type_errors_are_reported_as_configuration_errors() -> None:
    with pytest.raises(ConfigurationError, match="min_votes"):
        AnnotationPolicy(min_votes="2").validate()  # type: ignore[arg-type]


def test_scheduler_mix_type_errors_are_reported_as_configuration_errors() -> None:
    with pytest.raises(ConfigurationError, match="scheduler_config.mix"):
        SchedulerConfig(mode="mix", mix={"entropy": "bad"}).validate()  # type: ignore[dict-item]


def test_stop_criteria_type_errors_are_reported_as_configuration_errors() -> None:
    with pytest.raises(ConfigurationError, match="stop_criteria.max_labeled"):
        StopCriteria(max_labeled="10").validate()  # type: ignore[arg-type]


@pytest.mark.parametrize("managed_port", [0, -1, 65536])
def test_managed_label_backend_config_rejects_invalid_ports(managed_port: int) -> None:
    with pytest.raises(ConfigurationError, match="managed_port"):
        LabelBackendConfig(
            backend="label_studio",
            mode="managed_docker",
            managed_port=managed_port,
        ).validate()


class _FakeNoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class _FakeLogits:
    def __init__(self, row_count: int) -> None:
        self.row_count = row_count


class _FakeProbabilities:
    def __init__(self, row_count: int) -> None:
        self._rows = [[0.25, 0.75] for _ in range(row_count)]

    def cpu(self) -> "_FakeProbabilities":
        return self

    def tolist(self) -> list[list[float]]:
        return self._rows


class _FakeTokenizer:
    def __call__(self, texts: Sequence[str], **kwargs: Any) -> dict[str, Any]:
        return {"input_ids": list(texts)}


class _FakeHFModel:
    def eval(self) -> None:
        return None

    def __call__(self, **encoded: Any) -> SimpleNamespace:
        return SimpleNamespace(logits=_FakeLogits(len(encoded["input_ids"])))


def test_huggingface_adapter_normalizes_zero_batch_size_like_other_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = SimpleNamespace(
        no_grad=lambda: _FakeNoGrad(),
        softmax=lambda logits, dim: _FakeProbabilities(logits.row_count),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace())

    adapter = HFSequenceClassifierAdapter(model=_FakeHFModel(), tokenizer=_FakeTokenizer())

    assert adapter.predict_proba(["first", "second"], batch_size=0) == [
        [0.25, 0.75],
        [0.25, 0.75],
    ]


class _TwoSampleDataset:
    def __init__(self) -> None:
        self._samples = {
            "s1": DataSample(sample_id="s1", data={"text": "first"}),
            "s2": DataSample(sample_id="s2", data={"text": "second"}),
        }

    def iter_sample_ids(self):
        yield from self._samples

    def get_sample(self, sample_id: str) -> DataSample:
        return self._samples[sample_id]

    def schema(self) -> dict[str, str]:
        return {"sample_id": "str", "text": "str"}


class _MalformedPrelabelModel(MinimalRequiredAdapter):
    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [[1.0] for _ in texts]


class _NonUnitPrelabelModel(MinimalRequiredAdapter):
    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [[0.6, 0.6] for _ in texts]


class _RecordingBackend:
    def __init__(self) -> None:
        self.prelabels_seen: list[dict[str, Any] | None] = []

    def ensure_ready(self, label_schema: LabelSchema) -> dict[str, Any]:
        return {"backend": "recording"}

    def push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: dict[str, Any] | None = None,
    ) -> RoundPushResult:
        self.prelabels_seen.append(prelabels)
        return RoundPushResult(task_ids={sample.sample_id: sample.sample_id for sample in samples})

    def poll_round(
        self,
        round_id: str,
        task_ids: Mapping[str, str],
        policy: AnnotationPolicy,
    ) -> RoundProgress:
        return RoundProgress(total=len(task_ids), done=0, ready_sample_ids=[])

    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        return RoundPullResult(annotations={})

    def close(self) -> None:
        return None


def test_prelabeling_validates_probability_width_against_label_schema_before_backend_push(
    tmp_path,
) -> None:
    backend = _RecordingBackend()
    project = ActiveLearningProject("prelabel-width", tmp_path, lock=False)
    project.configure(
        dataset=_TwoSampleDataset(),
        model=_MalformedPrelabelModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=backend,
        scheduler_config=SchedulerConfig(mode="single", strategy="random"),
        cache_config=CacheConfig(enable=False),
        split_config=SplitConfig(
            mode="explicit",
            explicit_splits={"train": ["s1", "s2"], "val": [], "test": []},
        ),
        prelabel_config=PrelabelConfig(enable=True),
    )

    project.run_step(batch_size=1)
    with pytest.raises(ConfigurationError, match="label_schema"):
        project.run_step(batch_size=1)

    assert backend.prelabels_seen == []


def test_prelabeling_rejects_non_unit_probability_rows_before_backend_push(tmp_path) -> None:
    backend = _RecordingBackend()
    project = ActiveLearningProject("prelabel-non-unit", tmp_path, lock=False)
    project.configure(
        dataset=_TwoSampleDataset(),
        model=_NonUnitPrelabelModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=backend,
        scheduler_config=SchedulerConfig(mode="single", strategy="random"),
        cache_config=CacheConfig(enable=False),
        split_config=SplitConfig(
            mode="explicit",
            explicit_splits={"train": ["s1", "s2"], "val": [], "test": []},
        ),
        prelabel_config=PrelabelConfig(enable=True),
    )

    project.run_step(batch_size=1)
    with pytest.raises(ConfigurationError, match="sum to 1.0"):
        project.run_step(batch_size=1)

    assert backend.prelabels_seen == []
