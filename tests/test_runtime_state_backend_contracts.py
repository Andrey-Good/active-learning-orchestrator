from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from active_learning_sdk import (
    ActiveLearningProject,
    AnnotationPolicy,
    CacheConfig,
    ConfigurationError,
    LabelBackendConfig,
    LabelBackendError,
    LabelSchema,
    SchedulerConfig,
    SplitConfig,
)
from active_learning_sdk.backends.base import LLMLabelBackend, RoundProgress, RoundPullResult, RoundPushResult
from active_learning_sdk.backends.simulator import SimulatorLabelBackend
from active_learning_sdk.state.lock import ProjectLock
from active_learning_sdk.state.store import RoundState, state_from_json_dict
from active_learning_sdk.types import AnnotationRecord, DataSample, RoundStatus, StepKind


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


class ConflictingVotesBackend:
    def ensure_ready(self, label_schema: LabelSchema) -> dict[str, Any]:
        return {"backend": "conflicting"}

    def push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: dict[str, Any] | None = None,
    ) -> RoundPushResult:
        return RoundPushResult(task_ids={sample.sample_id: f"task:{sample.sample_id}" for sample in samples})

    def poll_round(self, round_id: str, task_ids: Mapping[str, str], policy: AnnotationPolicy) -> RoundProgress:
        return RoundProgress(total=len(task_ids), done=len(task_ids), ready_sample_ids=list(task_ids))

    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        return RoundPullResult(
            annotations={
                sample_id: [
                    AnnotationRecord(annotator_id="a", created_at=1.0, value="negative"),
                    AnnotationRecord(annotator_id="b", created_at=2.0, value="positive"),
                ]
                for sample_id in task_ids
            }
        )

    def close(self) -> None:
        return None


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

    def poll_round(self, round_id: str, task_ids: Mapping[str, str], policy: AnnotationPolicy) -> RoundProgress:
        return RoundProgress(total=len(task_ids), done=0, ready_sample_ids=[])

    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        return RoundPullResult(annotations={})

    def close(self) -> None:
        return None


class TypeErroringRestoreBackend(ReadyBackend):
    def __init__(self) -> None:
        self.restore_calls = 0

    def restore_round_samples(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        task_ids: Mapping[str, str] | None = None,
    ) -> None:
        del round_id, samples
        self.restore_calls += 1
        if task_ids is not None:
            raise TypeError("internal restore bug")


class LegacyRestoreBackend(ReadyBackend):
    def __init__(self) -> None:
        self.restored_payloads: list[tuple[str, list[str]]] = []

    def restore_round_samples(self, round_id: str, samples: Sequence[DataSample]) -> None:
        self.restored_payloads.append((round_id, [sample.sample_id for sample in samples]))


class PollRecordingBackend(ReadyBackend):
    def __init__(self) -> None:
        self.poll_calls = 0

    def poll_round(self, round_id: str, task_ids: Mapping[str, str], policy: AnnotationPolicy) -> RoundProgress:
        self.poll_calls += 1
        return super().poll_round(round_id, task_ids, policy)


def _configure_project(
    workdir: Path,
    *,
    dataset: InMemoryDataset | None = None,
    backend: Any | None = None,
    labels: list[str] | None = None,
    annotation_policy: AnnotationPolicy | None = None,
) -> ActiveLearningProject:
    project = ActiveLearningProject("w97-runtime", workdir, lock=False)
    project.configure(
        dataset=dataset or InMemoryDataset(),
        model=DummyModel(),
        label_schema=LabelSchema(task="text_classification", labels=labels or ["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=backend or ReadyBackend(),
        scheduler_config=SchedulerConfig(strategy="random"),
        annotation_policy=annotation_policy or AnnotationPolicy(),
        cache_config=CacheConfig(enable=False),
        split_config=SplitConfig(
            mode="explicit",
            explicit_splits={"train": ["s1", "s2"], "val": [], "test": []},
        ),
    )
    return project


def _live_project_state(project: ActiveLearningProject):
    return project._engine.get_state()  # type: ignore[attr-defined]


def test_restore_round_samples_internal_type_error_propagates(tmp_path: Path) -> None:
    backend = TypeErroringRestoreBackend()
    project = _configure_project(tmp_path, backend=backend)

    assert project.run_step(batch_size=2).step == StepKind.SELECT
    assert project.run_step(batch_size=2).step == StepKind.PUSH

    with pytest.raises(TypeError, match="internal restore bug"):
        project.run_step(batch_size=2)
    assert backend.restore_calls == 1


def test_legacy_two_argument_restore_round_samples_still_supported(tmp_path: Path) -> None:
    backend = LegacyRestoreBackend()
    project = _configure_project(tmp_path, backend=backend)

    assert project.run_step(batch_size=2).step == StepKind.SELECT
    assert project.run_step(batch_size=2).step == StepKind.PUSH
    round_state = _live_project_state(project).rounds[-1]

    assert project.run_step(batch_size=2).step == StepKind.WAIT
    assert backend.restored_payloads == [(round_state.round_id, list(round_state.selected_sample_ids))]


def test_wait_rejects_corrupted_task_id_mapping_before_poll(tmp_path: Path) -> None:
    backend = PollRecordingBackend()
    project = _configure_project(tmp_path, backend=backend)

    assert project.run_step(batch_size=2).step == StepKind.SELECT
    assert project.run_step(batch_size=2).step == StepKind.PUSH
    round_state = _live_project_state(project).rounds[-1]
    original_task_ids = dict(round_state.task_ids)
    selected_ids = list(round_state.selected_sample_ids)
    round_state.task_ids = {
        selected_ids[0]: original_task_ids[selected_ids[0]],
        "outside-round": "task:outside-round",
    }

    with pytest.raises(ConfigurationError, match="missing selected sample ids|unexpected sample ids"):
        project.run_step(batch_size=2)
    assert backend.poll_calls == 0


def test_simulator_rejects_swapped_task_id_sample_bindings() -> None:
    backend = SimulatorLabelBackend(label_by_sample_id={"s1": "negative", "s2": "positive"})
    backend.ensure_ready(LabelSchema(task="text_classification", labels=["negative", "positive"]))
    push = backend.push_round(
        "r1",
        [
            DataSample(sample_id="s1", data={"text": "one"}),
            DataSample(sample_id="s2", data={"text": "two"}),
        ],
    )
    swapped = {"s1": push.task_ids["s2"], "s2": push.task_ids["s1"]}

    with pytest.raises(LabelBackendError, match="does not belong to sample_id"):
        backend.poll_round("r1", swapped, AnnotationPolicy())
    with pytest.raises(LabelBackendError, match="does not belong to sample_id"):
        backend.pull_round("r1", swapped)


def test_v1_state_without_review_metadata_loads_empty_review_metadata() -> None:
    state = state_from_json_dict(
        {
            "state_version": 1,
            "project_name": "legacy",
            "created_at": 1.0,
            "updated_at": 2.0,
            "sample_status": {"s1": "unlabeled"},
            "sample_labels": {},
            "rounds": [],
            "metrics_history": [],
        }
    )

    assert state.sample_review_metadata == {}


@pytest.mark.parametrize(
    ("task_ids", "match"),
    [
        ({"s1": "sim:r1:s2", "s2": "sim:r1:s1"}, "expected deterministic binding"),
        ({"s1": "sim:r2:s1", "s2": "sim:r1:s2"}, "expected deterministic binding"),
        ({"s1": "not-a-simulator-task", "s2": "sim:r1:s2"}, "expected deterministic binding"),
        ({"s1": "sim:r1:s1", "s2": "sim:r1:s1"}, "Duplicate simulator task ids"),
        ({"s1": "sim:r1:s1"}, "Missing simulator task id"),
        ({"s1": "sim:r1:s1", "s2": "sim:r1:s2", "s3": "sim:r1:s3"}, "Unexpected simulator task ids"),
    ],
)
def test_simulator_restore_rejects_corrupt_task_ids(task_ids: Mapping[str, str], match: str) -> None:
    backend = SimulatorLabelBackend(label_by_sample_id={"s1": "negative", "s2": "positive"})
    backend.ensure_ready(LabelSchema(task="text_classification", labels=["negative", "positive"]))
    samples = [
        DataSample(sample_id="s1", data={"text": "one"}),
        DataSample(sample_id="s2", data={"text": "two"}),
    ]

    with pytest.raises(LabelBackendError, match=match):
        backend.restore_round_samples("r1", samples, task_ids)


def test_simulator_resume_rejects_corrupted_persisted_task_ids(tmp_path: Path) -> None:
    dataset = InMemoryDataset({"s1": "one", "s2": "two"})
    project = _configure_project(
        tmp_path,
        dataset=dataset,
        backend=SimulatorLabelBackend(label_by_sample_id={"s1": "negative", "s2": "positive"}),
    )
    assert project.run_step(batch_size=2).step == StepKind.SELECT
    assert project.run_step(batch_size=2).step == StepKind.PUSH
    project.close()

    state_path = tmp_path / "state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    active_round = payload["rounds"][-1]
    original_task_ids = dict(active_round["task_ids"])
    active_round["task_ids"] = {"s1": original_task_ids["s2"], "s2": original_task_ids["s1"]}
    state_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    resumed = ActiveLearningProject("w97-runtime", tmp_path, lock=False)
    resumed.attach_runtime(
        dataset=dataset,
        model=DummyModel(),
        label_backend=SimulatorLabelBackend(label_by_sample_id={"s1": "negative", "s2": "positive"}),
    )

    with pytest.raises(LabelBackendError, match="expected deterministic binding"):
        resumed.run_step(batch_size=2)


def test_all_needs_review_pull_marks_round_done_without_training(tmp_path: Path) -> None:
    project = _configure_project(
        tmp_path,
        backend=ConflictingVotesBackend(),
        annotation_policy=AnnotationPolicy(mode="majority", min_votes=2, min_agreement=0.75),
    )

    assert project.run_step(batch_size=2).step == StepKind.SELECT
    assert project.run_step(batch_size=2).step == StepKind.PUSH
    assert project.run_step(batch_size=2).step == StepKind.WAIT
    assert project.run_step(batch_size=2).step == StepKind.PULL

    state = project.get_state()
    assert state.rounds[-1].status == RoundStatus.DONE
    assert state.rounds[-1].resolved == {}
    assert set(state.sample_status.values()) == {"needs_review"}


def test_pull_persists_review_metadata_and_import_clears_stale_metadata(tmp_path: Path) -> None:
    project = _configure_project(
        tmp_path,
        backend=ConflictingVotesBackend(),
        annotation_policy=AnnotationPolicy(mode="majority", min_votes=2, min_agreement=0.75),
    )

    assert project.run_step(batch_size=2).step == StepKind.SELECT
    assert project.run_step(batch_size=2).step == StepKind.PUSH
    assert project.run_step(batch_size=2).step == StepKind.WAIT
    assert project.run_step(batch_size=2).step == StepKind.PULL

    state = project.get_state()
    assert set(state.sample_review_metadata) == {"s1", "s2"}
    assert state.sample_review_metadata["s1"]["reason"] == "majority_tie"
    assert state.sample_review_metadata["s1"]["agreement"] == 0.5
    assert state.sample_review_metadata["s1"]["annotation_count"] == 2
    assert state.sample_review_metadata["s1"]["eligible_vote_count"] == 2
    assert state.sample_review_metadata["s1"]["details"]["counts"] == {"negative": 1, "positive": 1}
    assert state.sample_review_metadata["s1"]["policy"]["mode"] == "majority"

    persisted = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert persisted["sample_review_metadata"]["s1"]["reason"] == "majority_tie"
    assert project.status()["review_metadata"]["by_reason"] == {"majority_tie": 2}

    project.import_labels({"s1": "negative"}, source="review")

    assert project.get_state().sample_status["s1"] == "labeled"
    assert "s1" not in project.get_state().sample_review_metadata
    assert "s2" in project.get_state().sample_review_metadata


def test_validate_rejects_multiple_active_rounds(tmp_path: Path) -> None:
    project = _configure_project(tmp_path)
    state = _live_project_state(project)
    state.rounds.append(
        RoundState(round_id="r1", status=RoundStatus.PUSHED, created_at=1.0, updated_at=1.0, selected_sample_ids=["s1"], task_ids={"s1": "task:s1"})
    )
    state.rounds.append(RoundState(round_id="r2", status=RoundStatus.SELECTING, created_at=2.0, updated_at=2.0))

    validation = project.validate()

    assert validation["ok"] is False
    assert any("active round" in issue.lower() for issue in validation["issues"])


def test_validate_rejects_sample_status_dataset_coverage_mismatch(tmp_path: Path) -> None:
    project = _configure_project(tmp_path)
    _live_project_state(project).sample_status.pop("s2")

    validation = project.validate()

    assert validation["ok"] is False
    assert any("sample_status" in issue and "dataset" in issue for issue in validation["issues"])


def test_reconfigure_rejects_label_schema_change_with_existing_labels(tmp_path: Path) -> None:
    dataset = InMemoryDataset()
    project = _configure_project(tmp_path, dataset=dataset, labels=["old", "other"])
    project.import_labels({"s1": "old"})

    with pytest.raises(ConfigurationError, match="label_schema"):
        project.configure(
            dataset=dataset,
            model=DummyModel(),
            label_schema=LabelSchema(task="text_classification", labels=["new", "other"]),
            label_backend_config=LabelBackendConfig(backend="custom"),
            label_backend=ReadyBackend(),
            scheduler_config=SchedulerConfig(strategy="random"),
            cache_config=CacheConfig(enable=False),
            split_config=SplitConfig(
                mode="explicit",
                explicit_splits={"train": ["s1", "s2"], "val": [], "test": []},
            ),
        )


def test_llm_backend_can_resume_pushed_round_after_restart(tmp_path: Path) -> None:
    dataset = InMemoryDataset({"s1": "real text"})

    def label_fn(sample: DataSample) -> AnnotationRecord:
        return AnnotationRecord(annotator_id="llm", created_at=1.0, value="positive" if sample.data["text"] == "real text" else "negative")

    project = ActiveLearningProject("w97-llm-resume", tmp_path, lock=False)
    project.configure(
        dataset=dataset,
        model=DummyModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=LLMLabelBackend(label_fn),
        scheduler_config=SchedulerConfig(strategy="random"),
        cache_config=CacheConfig(enable=False),
        split_config=SplitConfig(mode="explicit", explicit_splits={"train": ["s1"], "val": [], "test": []}),
    )
    assert project.run_step(batch_size=1).step == StepKind.SELECT
    assert project.run_step(batch_size=1).step == StepKind.PUSH
    project.close()

    resumed = ActiveLearningProject("w97-llm-resume", tmp_path, lock=False)
    resumed.attach_runtime(dataset=dataset, model=DummyModel(), label_backend=LLMLabelBackend(label_fn))

    assert resumed.run_step(batch_size=1).step == StepKind.WAIT
    assert resumed.run_step(batch_size=1).step == StepKind.PULL
    assert resumed.get_state().sample_labels == {"s1": "positive"}


def test_stale_project_lock_from_dead_process_does_not_block_resume(tmp_path: Path) -> None:
    lock_path = tmp_path / "project.lock"
    lock_path.write_text('{"pid": 999999999, "created_at": 1.0}', encoding="utf-8")

    lock = ProjectLock(lock_path)
    lock.acquire()
    try:
        assert lock_path.exists()
    finally:
        lock.release()


def test_run_resume_false_rejects_existing_active_round(tmp_path: Path) -> None:
    project = _configure_project(tmp_path)
    assert project.run_step(batch_size=1).step == StepKind.SELECT

    with pytest.raises(ConfigurationError, match="resume=False"):
        project.run(batch_size=1, resume=False, poll_interval_seconds=0)
