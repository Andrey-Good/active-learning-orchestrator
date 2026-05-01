from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from active_learning_sdk import (
    ActiveLearningProject,
    CacheConfig,
    LabelBackendConfig,
    LabelSchema,
    SchedulerConfig,
    SplitConfig,
)
from active_learning_sdk.cache import JsonlDiskCacheStore
from active_learning_sdk.backends.base import RoundProgress, RoundPullResult, RoundPushResult
from active_learning_sdk.state.store import JsonFileStateStore
from active_learning_sdk.types import AnnotationRecord, DataSample, SampleStatus, StepKind


class InMemoryDataset:
    def __init__(self, samples: Mapping[str, str] | None = None) -> None:
        samples = samples or {"s1": "one", "s2": "two"}
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


class CountingReadyBackend:
    def __init__(self) -> None:
        self.push_calls = 0
        self.poll_calls = 0
        self.pull_calls = 0

    def ensure_ready(self, label_schema: LabelSchema) -> dict[str, Any]:
        return {"backend": "counting-ready"}

    def push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: dict[str, Any] | None = None,
    ) -> RoundPushResult:
        self.push_calls += 1
        return RoundPushResult(task_ids={sample.sample_id: f"task:{round_id}:{sample.sample_id}" for sample in samples})

    def poll_round(self, round_id: str, task_ids: Mapping[str, str], policy: Any) -> RoundProgress:
        self.poll_calls += 1
        return RoundProgress(total=len(task_ids), done=len(task_ids), ready_sample_ids=list(task_ids.keys()))

    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        self.pull_calls += 1
        return RoundPullResult(
            annotations={
                sample_id: [AnnotationRecord(annotator_id="a", created_at=1.0, value="positive")]
                for sample_id in task_ids
            }
        )

    def close(self) -> None:
        return None


def _configure_project(
    workdir: Path,
    *,
    dataset: InMemoryDataset | None = None,
    backend: CountingReadyBackend | None = None,
) -> ActiveLearningProject:
    project = ActiveLearningProject("w98-runtime-state", workdir, lock=False)
    project.configure(
        dataset=dataset or InMemoryDataset(),
        model=DummyModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=backend or CountingReadyBackend(),
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


def test_resume_selected_round_with_persisted_task_ids_does_not_push_duplicates(tmp_path: Path) -> None:
    dataset = InMemoryDataset()
    first_backend = CountingReadyBackend()
    project = _configure_project(tmp_path, dataset=dataset, backend=first_backend)

    assert project.run_step(batch_size=2).step == StepKind.SELECT
    round_state = _live_project_state(project).rounds[-1]
    round_state.task_ids = {sample_id: f"task:{round_state.round_id}:{sample_id}" for sample_id in round_state.selected_sample_ids}
    JsonFileStateStore(tmp_path / "state.json").save_atomic(_live_project_state(project))
    project.close()

    resumed_backend = CountingReadyBackend()
    resumed = ActiveLearningProject("w98-runtime-state", tmp_path, lock=False)
    resumed.attach_runtime(dataset=dataset, model=DummyModel(), label_backend=resumed_backend)

    result = resumed.run_step(batch_size=2)

    assert result.step == StepKind.PUSH
    assert resumed_backend.push_calls == 0
    assert resumed.get_state().rounds[-1].task_ids == round_state.task_ids


def test_resume_after_pushed_round_continues_to_wait_without_recreating_tasks(tmp_path: Path) -> None:
    dataset = InMemoryDataset()
    first_backend = CountingReadyBackend()
    project = _configure_project(tmp_path, dataset=dataset, backend=first_backend)

    assert project.run_step(batch_size=2).step == StepKind.SELECT
    assert project.run_step(batch_size=2).step == StepKind.PUSH
    assert first_backend.push_calls == 1
    persisted_task_ids = dict(project.get_state().rounds[-1].task_ids)
    project.close()

    resumed_backend = CountingReadyBackend()
    resumed = ActiveLearningProject("w98-runtime-state", tmp_path, lock=False)
    resumed.attach_runtime(dataset=dataset, model=DummyModel(), label_backend=resumed_backend)

    result = resumed.run_step(batch_size=2)

    assert result.step == StepKind.WAIT
    assert resumed_backend.push_calls == 0
    assert resumed_backend.poll_calls == 1
    assert resumed.get_state().rounds[-1].task_ids == persisted_task_ids


def test_validate_rejects_labeled_status_without_persisted_label(tmp_path: Path) -> None:
    project = _configure_project(tmp_path)
    state = _live_project_state(project)
    state.sample_status["s1"] = SampleStatus.LABELED.value
    state.sample_labels.pop("s1", None)

    validation = project.validate()

    assert validation["ok"] is False
    assert any("labeled" in issue.lower() and "label" in issue.lower() for issue in validation["issues"])


def test_jsonl_cache_store_rejects_index_entry_that_points_at_a_different_key(tmp_path: Path) -> None:
    store = JsonlDiskCacheStore(tmp_path / "cache", "predictions")
    store.set("sample:s1", {"proba": [0.9, 0.1]})
    store.set("sample:s2", {"proba": [0.1, 0.9]})

    index_path = tmp_path / "cache" / "predictions.index.json"
    corrupt_index = json.loads(index_path.read_text(encoding="utf-8"))
    corrupt_index["sample:s1"] = corrupt_index["sample:s2"]
    index_path.write_text(json.dumps(corrupt_index, sort_keys=True), encoding="utf-8")

    reopened = JsonlDiskCacheStore(tmp_path / "cache", "predictions")

    assert reopened.get("sample:s1") is None
    repaired_index = json.loads(index_path.read_text(encoding="utf-8"))
    assert "sample:s1" not in repaired_index
