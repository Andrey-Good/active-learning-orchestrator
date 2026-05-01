from __future__ import annotations

import json
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
from active_learning_sdk.backends.base import RoundProgress, RoundPullResult, RoundPushResult
from active_learning_sdk.types import AnnotationRecord, DataSample, StepKind


class InMemoryDataset:
    def __init__(self) -> None:
        self._samples = {
            "s1": DataSample(sample_id="s1", data={"text": "one"}),
            "s2": DataSample(sample_id="s2", data={"text": "two"}),
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


class AuditedBackend:
    def ensure_ready(self, label_schema: LabelSchema) -> dict[str, Any]:
        return {"backend": "audited"}

    def push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: dict[str, Any] | None = None,
    ) -> RoundPushResult:
        return RoundPushResult(
            task_ids={sample.sample_id: f"task:{sample.sample_id}" for sample in samples},
            backend_round_ref={"project_id": 7, "api_token": "secret-token"},
        )

    def poll_round(self, round_id: str, task_ids: Mapping[str, str], policy: AnnotationPolicy) -> RoundProgress:
        return RoundProgress(
            total=len(task_ids),
            done=len(task_ids),
            ready_sample_ids=list(task_ids),
            details={"token": "poll-secret", "counts": {"ready": len(task_ids)}, "bad": float("nan")},
        )

    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        return RoundPullResult(
            annotations={
                sample_id: [AnnotationRecord(annotator_id="a", created_at=1.0, value="positive")]
                for sample_id in task_ids
            },
            backend_payload={"password": "pull-secret", "payload_count": len(task_ids)},
        )

    def close(self) -> None:
        return None


class RecoveringPushBackend(AuditedBackend):
    def __init__(self) -> None:
        self.push_calls = 0
        self.recovery_calls = 0

    def push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: dict[str, Any] | None = None,
    ) -> RoundPushResult:
        self.push_calls += 1
        raise LabelBackendError("ambiguous push failure")

    def recover_push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> RoundPushResult:
        self.recovery_calls += 1
        return RoundPushResult(
            task_ids={sample.sample_id: f"recovered:{sample.sample_id}" for sample in samples},
            backend_round_ref={"project_id": "recovered"},
        )


class SecretFailingPushBackend(AuditedBackend):
    def push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: dict[str, Any] | None = None,
    ) -> RoundPushResult:
        raise LabelBackendError("backend failed with api token secret-token-123 and password=p4ss")


def _project(tmp_path: Path, backend: Any) -> ActiveLearningProject:
    project = ActiveLearningProject("backend-hardening", tmp_path, lock=False)
    project.configure(
        dataset=InMemoryDataset(),
        model=DummyModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=backend,
        scheduler_config=SchedulerConfig(strategy="random"),
        annotation_policy=AnnotationPolicy(mode="latest", min_votes=1),
        cache_config=CacheConfig(enable=False),
        split_config=SplitConfig(
            mode="explicit",
            explicit_splits={"train": ["s1", "s2"], "val": [], "test": []},
        ),
    )
    return project


def test_backend_audit_summaries_are_persisted_and_redacted(tmp_path: Path) -> None:
    project = _project(tmp_path, AuditedBackend())

    assert project.run_step(batch_size=2).step == StepKind.SELECT
    assert project.run_step(batch_size=2).step == StepKind.PUSH
    assert project.run_step(batch_size=2).step == StepKind.WAIT
    assert project.run_step(batch_size=2).step == StepKind.PULL

    round_state = project.get_state().rounds[-1]
    assert round_state.backend_ref["backend_round_ref"]["project_id"] == 7
    assert round_state.backend_ref["backend_round_ref"]["api_token"] == "<redacted>"
    assert round_state.last_poll_progress["details"]["token"] == "<redacted>"
    assert round_state.last_poll_progress["details"]["bad"] is None
    assert round_state.pull_summary["backend_payload"]["password"] == "<redacted>"
    assert round_state.pull_summary["annotation_count"] == 2

    persisted = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    persisted_round = persisted["rounds"][-1]
    assert persisted_round["backend_ref"]["backend_round_ref"]["api_token"] == "<redacted>"
    assert "secret-token" not in json.dumps(persisted)
    assert "poll-secret" not in json.dumps(persisted)
    assert "pull-secret" not in json.dumps(persisted)


def test_engine_uses_same_round_push_recovery_before_failing(tmp_path: Path) -> None:
    backend = RecoveringPushBackend()
    project = _project(tmp_path, backend)

    assert project.run_step(batch_size=2).step == StepKind.SELECT
    assert project.run_step(batch_size=2).step == StepKind.PUSH

    round_state = project.get_state().rounds[-1]
    assert backend.push_calls == 1
    assert backend.recovery_calls == 1
    assert round_state.task_ids == {"s1": "recovered:s1", "s2": "recovered:s2"}
    assert round_state.backend_error_history == []


def test_backend_error_history_and_round_error_redact_secret_text(tmp_path: Path) -> None:
    project = _project(tmp_path, SecretFailingPushBackend())

    assert project.run_step(batch_size=2).step == StepKind.SELECT
    with pytest.raises(LabelBackendError):
        project.run_step(batch_size=2)

    state_json = (tmp_path / "state.json").read_text(encoding="utf-8")
    round_state = project.get_state().rounds[-1]
    assert "secret-token-123" not in state_json
    assert "p4ss" not in state_json
    assert "<redacted>" in state_json
    assert round_state.error is not None
    assert "secret-token-123" not in round_state.error
    assert round_state.backend_error_history
    assert "secret-token-123" not in round_state.backend_error_history[-1]["message"]
