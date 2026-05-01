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
from active_learning_sdk.backends.base import RoundProgress, RoundPullResult, RoundPushResult
from active_learning_sdk.state.store import JsonFileStateStore, ProjectState, RoundState
from active_learning_sdk.types import AnnotationRecord, DataSample, RoundStatus, StepKind
from active_learning_sdk.utils import sha256_file


class AuditDataset:
    def __init__(self, sample_ids: Sequence[str] = ("s1", "s2", "s3", "s4", "s5")) -> None:
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


class VolatileModel:
    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [[0.55, 0.45] for _ in texts]

    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs: Any) -> None:
        return None

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> dict[str, float]:
        return {"accuracy": 1.0}


class AuditBackend:
    def ensure_ready(self, label_schema: LabelSchema) -> dict[str, Any]:
        return {"backend": "audit", "ready": True}

    def push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: dict[str, Any] | None = None,
    ) -> RoundPushResult:
        return RoundPushResult(
            task_ids={sample.sample_id: f"task-{sample.sample_id}" for sample in samples},
            backend_round_ref={"round_id": round_id, "backend": "audit"},
        )

    def poll_round(self, round_id: str, task_ids: Mapping[str, str], policy: Any) -> RoundProgress:
        return RoundProgress(
            total=len(task_ids),
            done=len(task_ids),
            ready_sample_ids=list(task_ids.keys()),
            details={"poll": "complete"},
        )

    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        return RoundPullResult(
            annotations={
                sample_id: [AnnotationRecord(annotator_id="tester", created_at=1.0, value="positive")]
                for sample_id in task_ids
            },
            backend_payload={"source": "audit-test"},
        )

    def close(self) -> None:
        return None


def _configured_project(tmp_path: Path, *, cache_config: CacheConfig | None = None) -> ActiveLearningProject:
    project = ActiveLearningProject("audit-project", tmp_path, lock=False)
    project.configure(
        dataset=AuditDataset(),
        model=VolatileModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=AuditBackend(),
        scheduler_config=SchedulerConfig(strategy="random"),
        cache_config=cache_config or CacheConfig(enable=False),
        split_config=SplitConfig(
            mode="explicit",
            explicit_splits={"train": ["s1", "s2", "s3", "s4", "s5"], "val": [], "test": []},
        ),
    )
    return project


def _events(workdir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (workdir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_audit_event_log_records_ordered_round_lifecycle(tmp_path: Path) -> None:
    project = _configured_project(tmp_path)

    assert project.run_step(batch_size=2).step == StepKind.SELECT
    assert project.run_step(batch_size=2).step == StepKind.PUSH
    assert project.run_step(batch_size=2).step == StepKind.WAIT
    assert project.run_step(batch_size=2).step == StepKind.PULL
    assert project.run_step(batch_size=2).step == StepKind.TRAIN_EVAL
    assert project.run_step(batch_size=2).step == StepKind.UPDATE

    events = _events(tmp_path)
    assert [event["index"] for event in events] == list(range(1, len(events) + 1))
    event_types = [event["event_type"] for event in events]
    for expected in (
        "project.configure",
        "backend.ensure_ready",
        "round.created",
        "round.select",
        "backend.push",
        "round.push",
        "round.wait",
        "backend.poll",
        "round.pull_ready",
        "backend.pull",
        "round.pull",
        "cache.invalidated",
        "round.train",
        "round.update",
    ):
        assert expected in event_types
    assert event_types.index("round.created") < event_types.index("round.select") < event_types.index("round.push")
    assert event_types.index("round.pull") < event_types.index("round.train") < event_types.index("round.update")
    json.dumps(events, allow_nan=False)


def test_selection_audit_artifact_records_pool_selected_and_unselected_hashes(tmp_path: Path) -> None:
    project = _configured_project(tmp_path)

    project.run_step(batch_size=2)

    round_state = project.get_state().rounds[0]
    reference = round_state.selection_audit
    artifact_path = tmp_path / reference["path"]
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert reference["sha256"] == sha256_file(artifact_path)
    assert payload["schema_version"] == 1
    assert payload["round_id"] == round_state.round_id
    assert payload["eligible_pool"]["count"] == 5
    assert payload["selected"]["ids"] == round_state.selected_sample_ids
    assert payload["selected"]["count"] == 2
    assert payload["unselected"]["count"] == 3
    assert payload["unselected"]["ids"]
    assert payload["scheduler_snapshot"]["mode"] == "single"
    assert payload["scheduler_snapshot"]["strategy"] == "random"


def test_report_summary_and_manifest_reference_audit_artifacts(tmp_path: Path) -> None:
    project = _configured_project(tmp_path)
    project.run_step(batch_size=2)

    project.generate_report("reports")

    summary = json.loads((tmp_path / "reports" / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "reports" / "manifest.json").read_text(encoding="utf-8"))
    state_ref = manifest["audit_artifacts"]["state"]
    event_ref = manifest["audit_artifacts"]["event_log"]
    selection_refs = manifest["audit_artifacts"]["selection_audits"]

    assert summary["rounds"][0]["selection_audit"]["path"].endswith(".selection.json")
    assert summary["audit"]["event_log"]["path"] == "events.jsonl"
    assert state_ref == {"path": "state.json", "sha256": sha256_file(tmp_path / "state.json")}
    assert event_ref == {"path": "events.jsonl", "sha256": sha256_file(tmp_path / "events.jsonl")}
    assert selection_refs[0]["path"] == summary["rounds"][0]["selection_audit"]["path"]
    assert selection_refs[0]["sha256"] == sha256_file(tmp_path / selection_refs[0]["path"])


def test_report_manifest_rehashes_selection_audit_artifacts_instead_of_trusting_state(tmp_path: Path) -> None:
    project = _configured_project(tmp_path)
    project.run_step(batch_size=2)
    state = project.get_state()
    stale_hash = "0" * 64
    state.rounds[0].selection_audit["sha256"] = stale_hash
    state.audit_artifacts["selection_audits"][0]["sha256"] = stale_hash
    JsonFileStateStore(tmp_path / "state.json").save_atomic(state)

    project.generate_report("reports")

    summary = json.loads((tmp_path / "reports" / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "reports" / "manifest.json").read_text(encoding="utf-8"))
    selection_ref = manifest["audit_artifacts"]["selection_audits"][0]
    assert selection_ref["sha256"] != stale_hash
    assert selection_ref["sha256"] == sha256_file(tmp_path / selection_ref["path"])
    assert summary["audit"]["selection_audits"][0]["sha256"] == selection_ref["sha256"]
    assert summary["rounds"][0]["selection_audit"]["sha256"] == selection_ref["sha256"]


def test_manual_and_automatic_cache_invalidation_events_are_recorded(tmp_path: Path) -> None:
    project = _configured_project(tmp_path, cache_config=CacheConfig(enable=True, persist=False))

    project.clear_cache(kind="predictions")
    project.run_step(batch_size=1)
    project.run_step(batch_size=1)
    project.run_step(batch_size=1)
    project.run_step(batch_size=1)
    project.run_step(batch_size=1)

    events = _events(tmp_path)
    cache_events = [event for event in events if event["event_type"] in {"cache.clear", "cache.invalidated"}]
    assert cache_events[0]["event_type"] == "cache.clear"
    assert cache_events[0]["metadata"]["kind"] == "predictions"
    assert cache_events[0]["metadata"]["reason"] == "manual"
    assert any(
        event["event_type"] == "cache.invalidated"
        and event["metadata"]["reason"] == "automatic_model_id_unstable"
        for event in cache_events
    )


def test_legacy_state_without_audit_fields_still_loads(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "state_version": 1,
                "project_name": "legacy-project",
                "created_at": 1.0,
                "updated_at": 2.0,
                "rounds": [
                    {
                        "round_id": "r0001",
                        "status": "selected",
                        "created_at": 1.0,
                        "updated_at": 2.0,
                        "selected_sample_ids": ["s1"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = JsonFileStateStore(state_path).load()

    assert isinstance(loaded, ProjectState)
    assert loaded.audit_artifacts == {}
    assert isinstance(loaded.rounds[0], RoundState)
    assert loaded.rounds[0].selection_audit == {}
    assert loaded.rounds[0].status == RoundStatus.SELECTED
