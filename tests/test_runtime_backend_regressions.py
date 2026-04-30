from __future__ import annotations

import json
import os
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
from active_learning_sdk.state.lock import ProjectLock
from active_learning_sdk.types import DataSample, RoundStatus, StepKind


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


class InconsistentProgressBackend:
    def ensure_ready(self, label_schema: LabelSchema) -> dict[str, Any]:
        label_schema.validate()
        return {"backend": "inconsistent-progress"}

    def push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: dict[str, Any] | None = None,
    ) -> RoundPushResult:
        return RoundPushResult(task_ids={sample.sample_id: f"task:{sample.sample_id}" for sample in samples})

    def poll_round(self, round_id: str, task_ids: Mapping[str, str], policy: Any) -> RoundProgress:
        return RoundProgress(total=1, done=1, ready_sample_ids=["s1"])

    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        return RoundPullResult(annotations={})

    def close(self) -> None:
        return None


def _configure_project(workdir: Path, *, backend: Any) -> ActiveLearningProject:
    project = ActiveLearningProject("w02-runtime-state-backends", workdir, lock=False)
    project.configure(
        dataset=InMemoryDataset(),
        model=DummyModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=backend,
        scheduler_config=SchedulerConfig(strategy="random"),
        cache_config=CacheConfig(enable=False),
        split_config=SplitConfig(
            mode="explicit",
            explicit_splits={"train": ["s1", "s2"], "val": [], "test": []},
        ),
    )
    return project


def test_project_lock_release_from_non_owner_does_not_remove_or_break_active_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / "project.lock"
    owner = ProjectLock(lock_path)
    owner.acquire()
    third_party = ProjectLock(lock_path)
    contender: ProjectLock | None = None
    release_error: BaseException | None = None
    lock_exists_after_non_owner_release = False
    contender_acquired = False

    try:
        try:
            third_party.release()
        except BaseException as error:
            release_error = error

        lock_exists_after_non_owner_release = lock_path.exists()
        contender = ProjectLock(lock_path)
        try:
            contender.acquire()
        except Exception:
            contender_acquired = False
        else:
            contender_acquired = True
    finally:
        if contender_acquired and contender is not None:
            contender.release()
        owner.release()

    assert release_error is None, "release() on a lock object that never acquired the file should be a no-op."
    assert lock_exists_after_non_owner_release is True
    assert contender_acquired is False


def test_project_lock_release_does_not_remove_replaced_lock_file(tmp_path: Path) -> None:
    lock_path = tmp_path / "project.lock"
    owner = ProjectLock(lock_path)
    owner.acquire()
    replacement = {"pid": 999999, "created_at": 1.0, "owner_token": "someone-else"}
    try:
        lock_path.write_text(json.dumps(replacement), encoding="utf-8")
        owner.release()

        assert json.loads(lock_path.read_text(encoding="utf-8")) == replacement
    finally:
        lock_path.unlink(missing_ok=True)


def test_project_lock_stale_cleanup_does_not_remove_replaced_lock_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / "project.lock"
    stale = {"pid": 999999999, "created_at": 1.0, "owner_token": "stale"}
    replacement = {"pid": os.getpid(), "created_at": 2.0, "owner_token": "fresh"}
    lock_path.write_text(json.dumps(stale), encoding="utf-8")
    lock = ProjectLock(lock_path)
    original_read = lock._read_lock_payload_with_identity
    calls = 0

    def racing_read():
        nonlocal calls
        calls += 1
        result = original_read()
        if calls == 1:
            lock_path.write_text(json.dumps(replacement), encoding="utf-8")
        return result

    monkeypatch.setattr(lock, "_read_lock_payload_with_identity", racing_read)

    assert lock._remove_stale_lock() is False
    assert json.loads(lock_path.read_text(encoding="utf-8")) == replacement


def test_project_lock_acquire_recovers_stale_malformed_acquisition_gate(tmp_path: Path) -> None:
    lock_path = tmp_path / "project.lock"
    gate_path = tmp_path / "project.lock.acquire"
    gate_path.write_text("", encoding="utf-8")
    os.utime(gate_path, (1.0, 1.0))

    lock = ProjectLock(lock_path)
    lock.acquire()
    try:
        assert lock_path.exists()
        assert not gate_path.exists()
    finally:
        lock.release()


def test_wait_rejects_backend_progress_total_that_does_not_match_tracked_tasks(tmp_path: Path) -> None:
    project = _configure_project(tmp_path, backend=InconsistentProgressBackend())

    assert project.run_step(batch_size=2).step == StepKind.SELECT
    assert project.run_step(batch_size=2).step == StepKind.PUSH

    with pytest.raises(ConfigurationError, match="progress|total|task"):
        project.run_step(batch_size=2)

    assert project.get_state().rounds[-1].status == RoundStatus.WAITING
