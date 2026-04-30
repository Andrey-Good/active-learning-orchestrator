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
from active_learning_sdk.cache import JsonlDiskCacheStore
from active_learning_sdk.exceptions import StateCorruptedError
from active_learning_sdk.state.store import JsonFileStateStore
from active_learning_sdk.types import DataSample, SampleStatus
from active_learning_sdk import utils as sdk_utils


class InMemoryDataset:
    def __init__(self) -> None:
        self._samples = {
            sample_id: DataSample(sample_id=sample_id, data={"text": f"text {sample_id}"})
            for sample_id in ("s1", "s2", "s3")
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


def _configured_project(workdir: Path) -> ActiveLearningProject:
    project = ActiveLearningProject("w06-objection-sweep", workdir, lock=False)
    project.configure(
        dataset=InMemoryDataset(),
        model=DummyModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=NoopBackend(),
        scheduler_config=SchedulerConfig(strategy="random"),
        cache_config=CacheConfig(enable=False),
        split_config=SplitConfig(
            mode="explicit",
            explicit_splits={"train": ["s1", "s2"], "val": ["s3"], "test": []},
        ),
    )
    return project


def _live_project_state(project: ActiveLearningProject):
    # These tests intentionally inject corrupt internal state. Public
    # get_state() returns a detached snapshot and must not be used for mutation.
    return project._engine._state  # type: ignore[attr-defined]


def test_state_store_rejects_boolean_state_version(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "state_version": True,
                "project_name": "w06-objection-sweep",
                "created_at": 1.0,
                "updated_at": 2.0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(StateCorruptedError, match="Invalid state_version"):
        JsonFileStateStore(state_path).load()


def test_validate_flags_corrupt_persisted_split_membership(tmp_path: Path) -> None:
    project = _configured_project(tmp_path)
    state = _live_project_state(project)
    state.splits = {
        "train": ["s1", "s1", "s2"],
        "val": ["missing"],
        "test": ["s2"],
    }

    validation = project.validate()

    assert validation["ok"] is False
    assert any("split" in issue.lower() and "duplicate" in issue.lower() for issue in validation["issues"])
    assert any("split" in issue.lower() and "unknown" in issue.lower() for issue in validation["issues"])
    assert any("split" in issue.lower() and "overlap" in issue.lower() for issue in validation["issues"])


def test_export_dataset_split_rejects_unknown_subset_name(tmp_path: Path) -> None:
    project = _configured_project(tmp_path)

    with pytest.raises(ConfigurationError, match="Unsupported dataset split export subset"):
        project.export_dataset_split(tmp_path / "exports", which="labled", format="jsonl")

    assert not (tmp_path / "exports" / "labled.jsonl").exists()


def test_export_labels_refuses_labeled_status_without_label(tmp_path: Path) -> None:
    project = _configured_project(tmp_path)
    state = _live_project_state(project)
    state.sample_status["s1"] = SampleStatus.LABELED.value
    state.sample_labels.pop("s1", None)
    assert project.validate()["ok"] is False

    with pytest.raises(ConfigurationError, match="labeled status but no persisted label"):
        project.export_labels(tmp_path / "labels.jsonl", format="jsonl")

    assert not (tmp_path / "labels.jsonl").exists()


def test_jsonl_disk_cache_rejects_non_finite_values_before_append(tmp_path: Path) -> None:
    store = JsonlDiskCacheStore(tmp_path / "cache", "predictions")

    with pytest.raises(ValueError, match="Out of range float values"):
        store.set("sample:s1", {"proba": [float("nan"), 1.0]})

    data_path = tmp_path / "cache" / "predictions.jsonl"
    if data_path.exists():
        raw = data_path.read_text(encoding="utf-8")
        assert "NaN" not in raw
        assert "Infinity" not in raw


def test_jsonl_disk_cache_uses_byte_offset_index_and_seeks_on_get(tmp_path: Path) -> None:
    store = JsonlDiskCacheStore(tmp_path / "cache", "predictions")

    store.set("sample:s1", {"proba": [0.9, 0.1]})
    store.set("sample:s2", {"proba": [0.1, 0.9]})

    index_path = tmp_path / "cache" / "predictions.index.json"
    raw_index = json.loads(index_path.read_text(encoding="utf-8"))
    assert raw_index["sample:s1"]["offset"] == 0
    assert raw_index["sample:s2"]["offset"] > raw_index["sample:s1"]["offset"]

    reopened = JsonlDiskCacheStore(tmp_path / "cache", "predictions")
    assert reopened.get("sample:s2") == {"proba": [0.1, 0.9]}


def test_jsonl_disk_cache_retries_transient_permission_error_during_index_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_replace = os.replace
    calls = {"count": 0}

    def flaky_replace(source: str, target: str) -> None:
        if target.endswith("predictions.index.json") and calls["count"] < 2:
            calls["count"] += 1
            raise PermissionError("simulated transient Windows file lock")
        real_replace(source, target)

    monkeypatch.setattr(sdk_utils.os, "replace", flaky_replace)
    monkeypatch.setattr(sdk_utils.time, "sleep", lambda seconds: None)
    store = JsonlDiskCacheStore(tmp_path / "cache", "predictions")

    store.set("sample:s1", {"proba": [0.9, 0.1]})

    assert calls["count"] == 2
    assert store.get("sample:s1") == {"proba": [0.9, 0.1]}
    assert not list((tmp_path / "cache").glob("*.tmp"))


def test_jsonl_disk_cache_rebuilds_legacy_line_number_index(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    data_path = cache_dir / "predictions.jsonl"
    data_path.write_text(
        "\n".join(
            [
                json.dumps({"key": "sample:s1", "value": {"proba": [0.9, 0.1]}}),
                json.dumps({"key": "sample:s2", "value": {"proba": [0.1, 0.9]}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (cache_dir / "predictions.index.json").write_text(
        json.dumps({"sample:s1": 0, "sample:s2": 1}),
        encoding="utf-8",
    )

    store = JsonlDiskCacheStore(cache_dir, "predictions")

    assert store.get("sample:s2") == {"proba": [0.1, 0.9]}
    rebuilt_index = json.loads((cache_dir / "predictions.index.json").read_text(encoding="utf-8"))
    assert rebuilt_index["sample:s1"]["offset"] == 0
    assert rebuilt_index["sample:s2"]["offset"] > 1
