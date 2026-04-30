from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from active_learning_sdk import (
    ActiveLearningProject,
    AnnotationPolicy,
    CacheConfig,
    LabelBackendConfig,
    LabelBackendError,
    LabelSchema,
    SchedulerConfig,
    SplitConfig,
)
import active_learning_sdk.adapters.huggingface as hf_adapter_module
from active_learning_sdk.adapters.huggingface import HFSequenceClassifierAdapter
from active_learning_sdk.backends.base import RoundProgress, RoundPullResult, RoundPushResult
from active_learning_sdk.types import AnnotationRecord, DataSample, StepKind


class TinyProvider:
    def __init__(self) -> None:
        self.rows = {
            "s1": "train seed positive",
            "s2": "train seed negative",
            "s3": "validation positive",
            "s4": "validation negative",
            "s5": "test positive",
            "s6": "test negative",
        }

    def iter_sample_ids(self):
        yield from self.rows

    def get_sample(self, sample_id: str) -> DataSample:
        return DataSample(sample_id=sample_id, data={"text": self.rows[sample_id]})

    def get_samples(self, sample_ids: Sequence[str]) -> list[DataSample]:
        return [self.get_sample(sample_id) for sample_id in sample_ids]

    def schema(self) -> dict[str, str]:
        return {"sample_id": "str", "text": "str"}


class GoodModel:
    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        return [[0.55, 0.45] for _ in texts]

    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs: Any) -> None:
        return None

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> dict[str, float]:
        return {"accuracy": 1.0}


class Backend:
    def __init__(self, *, malformed_step: str | None = None, payload: Any = None) -> None:
        self.malformed_step = malformed_step
        self.payload = payload

    def ensure_ready(self, label_schema: LabelSchema) -> dict[str, Any]:
        return {"ready": True}

    def push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: dict[str, Any] | None = None,
    ) -> RoundPushResult:
        del round_id, prelabels
        if self.malformed_step == "push":
            return self.payload
        return RoundPushResult(
            task_ids={sample.sample_id: f"task:{sample.sample_id}" for sample in samples},
            backend_round_ref={"backend": "tiny"},
        )

    def poll_round(self, round_id: str, task_ids: Mapping[str, str], policy: AnnotationPolicy) -> RoundProgress:
        del round_id, policy
        if self.malformed_step == "poll":
            return self.payload
        return RoundProgress(total=len(task_ids), done=len(task_ids), ready_sample_ids=list(task_ids))

    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        del round_id
        if self.malformed_step == "pull":
            return self.payload
        return RoundPullResult(
            annotations={
                sample_id: [AnnotationRecord(annotator_id="ann", created_at=1.0, value="positive")]
                for sample_id in task_ids
            }
        )


def _project(tmp_path: Path, backend: Any) -> ActiveLearningProject:
    project = ActiveLearningProject("wave8-contracts", tmp_path, lock=False)
    project.configure(
        dataset=TinyProvider(),
        model=GoodModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=backend,
        scheduler_config=SchedulerConfig(strategy="random"),
        annotation_policy=AnnotationPolicy(mode="latest", min_votes=1),
        cache_config=CacheConfig(enable=False),
        split_config=SplitConfig(
            mode="explicit",
            explicit_splits={"train": ["s1", "s2"], "val": ["s3", "s4"], "test": ["s5", "s6"]},
        ),
    )
    return project


@pytest.mark.parametrize(
    ("step", "expected_step"),
    [
        ("push", StepKind.PUSH),
        ("poll", StepKind.WAIT),
        ("pull", StepKind.PULL),
    ],
)
def test_backend_none_lifecycle_returns_are_label_backend_errors(
    tmp_path: Path,
    step: str,
    expected_step: StepKind,
) -> None:
    project = _project(tmp_path, Backend(malformed_step=step, payload=None))

    assert project.run_step(batch_size=2).step == StepKind.SELECT
    if expected_step in {StepKind.WAIT, StepKind.PULL}:
        assert project.run_step(batch_size=2).step == StepKind.PUSH
    if expected_step == StepKind.PULL:
        assert project.run_step(batch_size=2).step == StepKind.WAIT

    with pytest.raises(LabelBackendError, match=f"{step}_round|invalid payload"):
        project.run_step(batch_size=2)


@pytest.mark.parametrize("bad_task_id", [123, None, "", object()])
def test_backend_push_task_ids_must_be_non_empty_strings(tmp_path: Path, bad_task_id: Any) -> None:
    class BadTaskIdBackend(Backend):
        def push_round(
            self,
            round_id: str,
            samples: Sequence[DataSample],
            prelabels: dict[str, Any] | None = None,
        ) -> RoundPushResult:
            del round_id, prelabels
            return RoundPushResult(
                task_ids={samples[0].sample_id: bad_task_id, samples[1].sample_id: "task-ok"},
                backend_round_ref={},
            )

    project = _project(tmp_path, BadTaskIdBackend())

    assert project.run_step(batch_size=2).step == StepKind.SELECT
    with pytest.raises(LabelBackendError, match="task ids|task_ids"):
        project.run_step(batch_size=2)


def test_export_dataset_split_supports_configured_train_val_test_names(tmp_path: Path) -> None:
    project = _project(tmp_path, Backend())
    output_dir = tmp_path / "exports"

    for split_name in ("train", "val", "test"):
        project.export_dataset_split(output_dir, which=split_name, format="jsonl")
        path = output_dir / f"{split_name}.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        assert [record["sample_id"] for record in records] == project.get_state().splits[split_name]


def test_huggingface_adapter_construction_requires_huggingface_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    monkeypatch.setattr(hf_adapter_module, "find_spec", lambda name: None)

    with pytest.raises(ImportError, match=r"active-learning-sdk\[huggingface\]"):
        HFSequenceClassifierAdapter(model=object(), tokenizer=object())
