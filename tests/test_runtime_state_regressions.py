from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from active_learning_sdk import (
    ActiveLearningProject,
    CacheConfig,
    LabelBackendConfig,
    LabelSchema,
    SchedulerConfig,
    SplitConfig,
    StateCorruptedError,
)
from active_learning_sdk.backends.base import LLMLabelBackend, RoundProgress, RoundPullResult, RoundPushResult
from active_learning_sdk.types import AnnotationRecord, DataSample, StepKind


class InMemoryDataset:
    def __init__(self, samples: Mapping[str, str]) -> None:
        self._samples = {
            sample_id: DataSample(sample_id=sample_id, data={"text": text})
            for sample_id, text in samples.items()
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

    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs: Any) -> None:
        return None

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> dict[str, float]:
        return {"accuracy": 1.0}


class ReadyBackend:
    def ensure_ready(self, label_schema: LabelSchema) -> dict[str, Any]:
        return {"backend": "ready"}

    def push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: dict[str, Any] | None = None,
    ) -> RoundPushResult:
        return RoundPushResult(task_ids={sample.sample_id: f"task:{sample.sample_id}" for sample in samples})

    def poll_round(self, round_id: str, task_ids: Mapping[str, str], policy: Any) -> RoundProgress:
        return RoundProgress(total=len(task_ids), done=0, ready_sample_ids=[])

    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        return RoundPullResult(annotations={})

    def close(self) -> None:
        return None


def _configure_project(
    workdir: Path,
    *,
    dataset: InMemoryDataset | None = None,
    backend: Any | None = None,
) -> ActiveLearningProject:
    project = ActiveLearningProject("senior-runtime-audit", workdir, lock=False)
    project.configure(
        dataset=dataset or InMemoryDataset({"s1": "one", "s2": "two"}),
        model=DummyModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=backend or ReadyBackend(),
        scheduler_config=SchedulerConfig(strategy="random"),
        cache_config=CacheConfig(enable=False),
        split_config=SplitConfig(
            mode="explicit",
            explicit_splits={"train": ["s1", "s2"], "val": [], "test": []},
        ),
    )
    return project


def _live_project_state(project: ActiveLearningProject):
    return project._engine.get_state()  # type: ignore[attr-defined]


def test_state_load_rejects_unknown_sample_status_values(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "state_version": 1,
                "project_name": "senior-runtime-audit",
                "created_at": 1.0,
                "updated_at": 2.0,
                "sample_status": {"s1": "not-a-valid-status"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(StateCorruptedError, match="sample_status"):
        ActiveLearningProject("senior-runtime-audit", tmp_path, lock=False)


def test_validate_reports_task_ids_that_do_not_match_selected_samples(tmp_path: Path) -> None:
    project = _configure_project(tmp_path)

    assert project.run_step(batch_size=2).step == StepKind.SELECT
    assert project.run_step(batch_size=2).step == StepKind.PUSH

    active_round = _live_project_state(project).rounds[-1]
    active_round.task_ids.pop(active_round.selected_sample_ids[0])

    validation = project.validate()

    assert validation["ok"] is False
    assert any("task_ids" in issue and "selected" in issue for issue in validation["issues"])


def test_llm_backend_label_function_receives_original_sample_payload() -> None:
    observed_texts: list[str] = []

    def label_fn(sample: DataSample) -> AnnotationRecord:
        observed_texts.append(str(sample.data.get("text", "")))
        value = "positive" if sample.data.get("text") == "real text" else "negative"
        return AnnotationRecord(annotator_id="llm", created_at=1.0, value=value)

    backend = LLMLabelBackend(label_fn)
    backend.ensure_ready(LabelSchema(task="text_classification", labels=["negative", "positive"]))
    push_result = backend.push_round("r1", [DataSample(sample_id="s1", data={"text": "real text"})])

    pulled = backend.pull_round("r1", push_result.task_ids)

    assert observed_texts == ["real text"]
    assert pulled.annotations["s1"][0].value == "positive"
