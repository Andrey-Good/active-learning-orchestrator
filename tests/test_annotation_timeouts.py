from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from active_learning_sdk import (
    ActiveLearningError,
    ActiveLearningProject,
    AnnotationPolicy,
    CacheConfig,
    ConfigurationError,
    LabelBackendConfig,
    LabelSchema,
    SchedulerConfig,
    SplitConfig,
)
from active_learning_sdk.backends.base import RoundProgress, RoundPullResult, RoundPushResult
from active_learning_sdk.types import AnnotationRecord, DataSample, RoundStatus, SampleStatus, StepKind


class InMemoryDataset:
    def __init__(self, sample_ids: Sequence[str] = ("s1", "s2")) -> None:
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

    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs: Any) -> None:
        return None

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> dict[str, float]:
        return {"accuracy": 1.0}


class TimeoutBackend:
    def __init__(
        self,
        *,
        progress: RoundProgress,
        annotations: Mapping[str, Sequence[AnnotationRecord]] | None = None,
    ) -> None:
        self.progress = progress
        self.annotations = {sample_id: list(records) for sample_id, records in (annotations or {}).items()}
        self.pull_calls = 0

    def ensure_ready(self, label_schema: LabelSchema) -> dict[str, Any]:
        return {"backend": "timeout-test"}

    def push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: dict[str, Any] | None = None,
    ) -> RoundPushResult:
        return RoundPushResult(task_ids={sample.sample_id: sample.sample_id for sample in samples})

    def poll_round(self, round_id: str, task_ids: Mapping[str, str], policy: Any) -> RoundProgress:
        return self.progress

    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        self.pull_calls += 1
        return RoundPullResult(
            annotations={sample_id: list(self.annotations.get(sample_id, [])) for sample_id in task_ids},
            backend_payload={"pull_calls": self.pull_calls},
        )

    def close(self) -> None:
        return None


class MutableProgressBackend(TimeoutBackend):
    def __init__(self, *, annotations: Mapping[str, Sequence[AnnotationRecord]] | None = None) -> None:
        super().__init__(progress=RoundProgress(total=2, done=0), annotations=annotations)


class NonFiniteDiagnosticsBackend(TimeoutBackend):
    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        self.pull_calls += 1
        return RoundPullResult(
            annotations={sample_id: list(self.annotations.get(sample_id, [])) for sample_id in task_ids},
            backend_payload={"bad_float": float("nan"), "nested": {"bad_float": float("inf")}},
        )


def _configured_project(
    tmp_path: Path,
    *,
    backend: TimeoutBackend,
    policy: AnnotationPolicy,
) -> ActiveLearningProject:
    project = ActiveLearningProject("timeout-project", tmp_path, lock=False)
    project.configure(
        dataset=InMemoryDataset(),
        model=DummyModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=backend,
        scheduler_config=SchedulerConfig(strategy="random"),
        annotation_policy=policy,
        cache_config=CacheConfig(enable=False),
        split_config=SplitConfig(
            mode="explicit",
            explicit_splits={"train": ["s1", "s2"], "val": [], "test": []},
        ),
    )
    assert project.run_step(batch_size=2).step == StepKind.SELECT
    assert project.run_step(batch_size=2).step == StepKind.PUSH
    active_round = _live_project_state(project).rounds[-1]
    active_round.status = RoundStatus.WAITING
    active_round.scheduler_snapshot["annotation_timeout"] = {"wait_started_at": 1.0}
    return project


def _live_project_state(project: ActiveLearningProject):
    return project._engine.get_state()  # type: ignore[attr-defined]


def _timeout_trace(project: ActiveLearningProject) -> dict[str, Any]:
    return project.get_state().rounds[-1].scheduler_snapshot["annotation_timeout"]


def test_wait_timeout_raise_persists_trace_and_keeps_round_waiting(tmp_path: Path) -> None:
    backend = TimeoutBackend(progress=RoundProgress(total=2, done=0, details={"poll": "slow"}))
    project = _configured_project(
        tmp_path,
        backend=backend,
        policy=AnnotationPolicy(timeout_seconds=1, on_timeout="raise"),
    )

    with pytest.raises(ActiveLearningError, match="Annotation wait timed out"):
        project.run_step(batch_size=2)

    state = project.get_state()
    assert state.rounds[-1].status == RoundStatus.WAITING
    assert state.rounds[-1].error is not None
    trace = _timeout_trace(project)
    assert trace["timed_out"] is True
    assert trace["action"] == "raise"
    assert trace["progress"]["details"] == {"poll": "slow"}
    assert json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))["rounds"][-1]["scheduler_snapshot"][
        "annotation_timeout"
    ]["action"] == "raise"


def test_wait_timeout_needs_review_marks_unready_samples_and_keeps_ready_labels(tmp_path: Path) -> None:
    backend = TimeoutBackend(
        progress=RoundProgress(total=2, done=1, ready_sample_ids=["s1"]),
        annotations={
            "s1": [
                AnnotationRecord(annotator_id="a", created_at=10.0, value="positive"),
                AnnotationRecord(annotator_id="b", created_at=11.0, value="positive"),
            ],
            "s2": [AnnotationRecord(annotator_id="a", created_at=10.0, value="negative")],
        },
    )
    project = _configured_project(
        tmp_path,
        backend=backend,
        policy=AnnotationPolicy(mode="majority", min_votes=2, timeout_seconds=1, on_timeout="needs_review"),
    )

    result = project.run_step(batch_size=2)

    state = project.get_state()
    assert result.step == StepKind.WAIT
    assert state.sample_status["s1"] == SampleStatus.LABELED.value
    assert state.sample_labels["s1"] == "positive"
    assert state.sample_status["s2"] == SampleStatus.NEEDS_REVIEW.value
    assert state.sample_review_metadata["s2"]["reason"] == "annotation_timeout_not_ready"
    assert state.sample_review_metadata["s2"]["annotation_count"] == 1
    assert state.sample_review_metadata["s2"]["eligible_vote_count"] == 1
    assert state.rounds[-1].status == RoundStatus.PULLED
    trace = _timeout_trace(project)
    assert trace["action"] == "needs_review"
    assert trace["accepted_sample_ids"] == ["s1"]
    assert trace["needs_review_sample_ids"] == ["s2"]
    assert trace["annotation_counts"] == {"s1": 2, "s2": 1}


def test_wait_timeout_payload_validation_is_atomic(tmp_path: Path) -> None:
    backend = TimeoutBackend(
        progress=RoundProgress(total=2, done=1, ready_sample_ids=["s1", "s2"]),
        annotations={
            "s1": [AnnotationRecord(annotator_id="a", created_at=10.0, value="positive")],
            "s2": [AnnotationRecord(annotator_id="a", created_at=11.0, value="outside-schema")],
        },
    )
    project = _configured_project(
        tmp_path,
        backend=backend,
        policy=AnnotationPolicy(mode="latest", min_votes=1, timeout_seconds=1, on_timeout="needs_review"),
    )

    with pytest.raises(ConfigurationError, match="LabelSchema"):
        project.run_step(batch_size=2)

    state = project.get_state()
    assert state.sample_status == {
        "s1": SampleStatus.UNLABELED.value,
        "s2": SampleStatus.UNLABELED.value,
    }
    assert state.sample_labels == {}
    assert state.rounds[-1].status == RoundStatus.WAITING


def test_wait_timeout_sanitizes_non_finite_diagnostics_before_persisting(tmp_path: Path) -> None:
    backend = NonFiniteDiagnosticsBackend(
        progress=RoundProgress(total=2, done=0, details={"bad_float": float("nan")}),
        annotations={},
    )
    project = _configured_project(
        tmp_path,
        backend=backend,
        policy=AnnotationPolicy(timeout_seconds=1, on_timeout="needs_review"),
    )

    project.run_step(batch_size=2)

    trace = _timeout_trace(project)
    assert trace["progress"]["details"]["bad_float"] is None
    assert trace["backend_payload"]["bad_float"] is None
    assert trace["backend_payload"]["nested"]["bad_float"] is None
    persisted = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    persisted_trace = persisted["rounds"][-1]["scheduler_snapshot"]["annotation_timeout"]
    assert persisted_trace["progress"]["details"]["bad_float"] is None
    assert persisted["sample_status"] == {"s1": SampleStatus.NEEDS_REVIEW.value, "s2": SampleStatus.NEEDS_REVIEW.value}


def test_wait_timeout_accept_latest_uses_available_annotation_without_synthetic_labels(tmp_path: Path) -> None:
    backend = TimeoutBackend(
        progress=RoundProgress(total=2, done=0),
        annotations={"s1": [AnnotationRecord(annotator_id="a", created_at=10.0, value="negative")]},
    )
    project = _configured_project(
        tmp_path,
        backend=backend,
        policy=AnnotationPolicy(mode="majority", min_votes=2, timeout_seconds=1, on_timeout="accept_latest"),
    )

    result = project.run_step(batch_size=2)

    state = project.get_state()
    assert result.details["details"]["annotation_timeout"]["action"] == "accept_latest"
    assert state.sample_status["s1"] == SampleStatus.LABELED.value
    assert state.sample_labels["s1"] == "negative"
    assert state.sample_status["s2"] == SampleStatus.NEEDS_REVIEW.value
    assert "s2" not in state.sample_labels
    assert state.sample_review_metadata["s2"]["reason"] == "no_annotations"
    assert state.sample_review_metadata["s2"]["annotation_count"] == 0
    assert state.rounds[-1].resolved == {"s1": "negative"}
    assert state.rounds[-1].status == RoundStatus.PULLED


def test_wait_timeout_without_annotations_completes_round_as_needs_review(tmp_path: Path) -> None:
    backend = TimeoutBackend(progress=RoundProgress(total=2, done=0), annotations={})
    project = _configured_project(
        tmp_path,
        backend=backend,
        policy=AnnotationPolicy(timeout_seconds=1, on_timeout="needs_review"),
    )

    project.run_step(batch_size=2)

    state = project.get_state()
    assert state.sample_status == {
        "s1": SampleStatus.NEEDS_REVIEW.value,
        "s2": SampleStatus.NEEDS_REVIEW.value,
    }
    assert state.sample_review_metadata["s1"]["reason"] == "annotation_timeout_not_ready"
    assert state.sample_review_metadata["s1"]["policy"]["on_timeout"] == "needs_review"
    assert state.rounds[-1].status == RoundStatus.DONE
    assert _timeout_trace(project)["needs_review_sample_ids"] == ["s1", "s2"]


def test_wait_timeout_is_enforced_after_restart_from_persisted_waiting_round(tmp_path: Path) -> None:
    first_backend = TimeoutBackend(progress=RoundProgress(total=2, done=0), annotations={})
    project = _configured_project(
        tmp_path,
        backend=first_backend,
        policy=AnnotationPolicy(timeout_seconds=1, on_timeout="raise"),
    )
    active_round = _live_project_state(project).rounds[-1]
    active_round.status = RoundStatus.WAITING
    active_round.created_at = 1.0
    active_round.scheduler_snapshot.pop("annotation_timeout", None)
    project._engine._save_state()
    project.close()

    reopened_backend = TimeoutBackend(progress=RoundProgress(total=2, done=0), annotations={})
    reopened = ActiveLearningProject("timeout-project", tmp_path, lock=False)
    reopened.attach_runtime(dataset=InMemoryDataset(), model=DummyModel(), label_backend=reopened_backend)

    with pytest.raises(ActiveLearningError, match="Annotation wait timed out"):
        reopened.run_step(batch_size=2)

    trace = _timeout_trace(reopened)
    assert trace["wait_started_at"] == 1.0
    assert trace["action"] == "raise"


def test_wait_without_timeout_does_not_persist_timeout_trace(tmp_path: Path) -> None:
    backend = TimeoutBackend(progress=RoundProgress(total=2, done=0))
    project = _configured_project(
        tmp_path,
        backend=backend,
        policy=AnnotationPolicy(timeout_seconds=None),
    )
    _live_project_state(project).rounds[-1].scheduler_snapshot.pop("annotation_timeout", None)

    project.run_step(batch_size=2)

    assert "annotation_timeout" not in project.get_state().rounds[-1].scheduler_snapshot


def test_successful_retry_after_timeout_raise_clears_stale_round_error(tmp_path: Path) -> None:
    backend = MutableProgressBackend()
    project = _configured_project(
        tmp_path,
        backend=backend,
        policy=AnnotationPolicy(timeout_seconds=1, on_timeout="raise"),
    )

    with pytest.raises(ActiveLearningError, match="Annotation wait timed out"):
        project.run_step(batch_size=2)

    state = project.get_state()
    assert state.rounds[-1].error is not None

    backend.progress = RoundProgress(total=2, done=2, ready_sample_ids=["s1", "s2"])
    project.run_step(batch_size=2)

    active_round = _live_project_state(project).rounds[-1]
    assert active_round.status == RoundStatus.READY_TO_PULL
    assert active_round.error is None
