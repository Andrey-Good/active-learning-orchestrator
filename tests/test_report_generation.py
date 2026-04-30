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
from active_learning_sdk.report import ReportGenerator
from active_learning_sdk.state.store import DatasetRef, ProjectState, RoundState
from active_learning_sdk.types import DataSample, MetricRecord, RoundStatus, SampleStatus


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

    def poll_round(self, round_id: str, task_ids: Mapping[str, str], policy: Any) -> RoundProgress:
        return RoundProgress(total=len(task_ids), done=len(task_ids), ready_sample_ids=list(task_ids.keys()))

    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        return RoundPullResult(annotations={})

    def close(self) -> None:
        return None


def _state() -> ProjectState:
    return ProjectState(
        state_version=1,
        project_name="report-project",
        created_at=1.0,
        updated_at=2.0,
        dataset_ref=DatasetRef(
            source_type="provider",
            source_path=None,
            schema={"sample_id": "str", "text": "str"},
            fingerprint="abc123",
            fingerprint_config={"include_schema": True},
        ),
        label_schema={"task": "text_classification", "labels": ["negative", "positive"]},
        scheduler_config={"strategy": "entropy", "mode": "single"},
        sample_status={
            "s1": SampleStatus.LABELED.value,
            "s2": SampleStatus.LABELED.value,
            "s3": SampleStatus.NEEDS_REVIEW.value,
            "s4": SampleStatus.UNLABELED.value,
        },
        sample_labels={"s1": "positive", "s2": "negative"},
        sample_review_metadata={
            "s3": {
                "reason": "majority_tie",
                "agreement": 0.5,
                "annotation_count": 2,
                "eligible_vote_count": 2,
                "details": {"counts": {"negative": 1, "positive": 1}},
                "policy": {"mode": "majority", "min_votes": 2},
            }
        },
        metrics_history=[
            MetricRecord(
                step="eval",
                created_at=10.0,
                metrics={"accuracy": float("nan"), "loss": float("inf"), "f1": 0.75},
            )
        ],
        rounds=[
            RoundState(
                round_id="r0001",
                status=RoundStatus.DONE,
                created_at=3.0,
                updated_at=4.0,
                selected_sample_ids=["s1", "s2"],
                task_ids={"s1": "t1", "s2": "t2"},
                resolved={"s1": "positive"},
                metrics_before={"accuracy": 0.5},
                metrics_after={"accuracy": float("-inf")},
                reward=float("nan"),
                scheduler_snapshot={
                    "mode": "single",
                    "strategy": "entropy",
                    "score_mean": 0.42,
                    "annotation_timeout": {
                        "timed_out": True,
                        "action": "needs_review",
                        "timeout_seconds": 1,
                    },
                },
            )
        ],
        scheduler_state={
            "stop_trace": {
                "stopped": True,
                "reason": "metric_plateau",
                "observed_values": {"recent_values": [0.75, float("nan")]},
            }
        },
    )


def test_strict_json_report_sanitizes_non_finite_metrics(tmp_path: Path) -> None:
    output_path = tmp_path / "audit.json"

    ReportGenerator().generate_report(_state(), output_path)

    raw = output_path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw
    parsed = json.loads(raw)
    assert parsed["metrics_history"][0]["metrics"] == {"accuracy": None, "f1": 0.75, "loss": None}
    assert parsed["review"]["by_reason"] == {"majority_tie": 1}
    assert parsed["review"]["samples"]["s3"]["eligible_vote_count"] == 2
    assert parsed["rounds"][0]["metrics_after"]["accuracy"] is None
    assert parsed["rounds"][0]["reward"] is None
    json.dumps(parsed, allow_nan=False)


def test_markdown_and_html_reports_are_created_with_key_sections(tmp_path: Path) -> None:
    ReportGenerator().generate_report(_state(), tmp_path / "report.html")

    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    html = (tmp_path / "report.html").read_text(encoding="utf-8")

    assert "## Project Metadata" in markdown
    assert "## Dataset" in markdown
    assert "## Sample Counts" in markdown
    assert "## Annotation Review Metadata" in markdown
    assert "## Metrics History" in markdown
    assert "## Round Timeline" in markdown
    assert "## Scheduler And Acquisition Snapshots" in markdown
    assert "<h2>Project Metadata</h2>" in html
    assert "<h2>Annotation Review Metadata</h2>" in html
    assert "<h2>Round Timeline</h2>" in html
    assert "<h2>Annotation Timeout Traces</h2>" in html


def test_timeout_and_stop_traces_are_included(tmp_path: Path) -> None:
    ReportGenerator().generate_report(_state(), tmp_path)

    parsed = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))

    assert parsed["traces"]["stop_trace"]["reason"] == "metric_plateau"
    assert parsed["traces"]["stop_trace"]["observed_values"]["recent_values"] == [0.75, None]
    assert parsed["traces"]["annotation_timeouts"] == [
        {
            "round_id": "r0001",
            "trace": {"action": "needs_review", "timed_out": True, "timeout_seconds": 1},
        }
    ]


def test_report_manifest_exists_and_includes_reproducibility_fields(tmp_path: Path) -> None:
    paths = ReportGenerator().generate_report(_state(), tmp_path)

    assert paths["manifest"] == tmp_path / "manifest.json"
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    raw = (tmp_path / "manifest.json").read_text(encoding="utf-8")

    assert "NaN" not in raw
    assert "Infinity" not in raw
    assert manifest["artifact_schema_version"] == 1
    assert manifest["sdk"]["package_name"] == "active-learning-sdk"
    assert "package_version" in manifest["sdk"]
    assert manifest["runtime"]["python_version_info"]["major"] >= 3
    assert manifest["project"]["project_name"] == "report-project"
    assert manifest["project"]["state_version"] == 1
    assert manifest["dataset"]["fingerprint"] == "abc123"
    assert manifest["dataset"]["fingerprint_config"] == {"include_schema": True}
    assert manifest["configuration"]["scheduler_config"] == {"mode": "single", "strategy": "entropy"}
    assert "annotation_policy" in manifest["configuration"]
    assert "label_backend_config" in manifest["configuration"]
    assert "cache_config" in manifest["configuration"]
    assert manifest["counts"] == {"labels": 2, "metrics": 1, "review_metadata": 1, "rounds": 1, "samples": 4}
    assert manifest["artifacts"] == {
        "html": "report.html",
        "json": "summary.json",
        "manifest": "manifest.json",
        "markdown": "report.md",
    }
    json.dumps(manifest, allow_nan=False)


def test_markdown_and_html_list_manifest_artifact(tmp_path: Path) -> None:
    ReportGenerator().generate_report(_state(), tmp_path)

    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    html = (tmp_path / "report.html").read_text(encoding="utf-8")

    assert "## Reproducibility Manifest" in markdown
    assert "manifest\\.json" in markdown
    assert "<h2>Reproducibility Manifest</h2>" in html
    assert "manifest.json" in html


def test_markdown_report_escapes_user_controlled_html_like_values(tmp_path: Path) -> None:
    state = _state()
    state.project_name = '<img src=x onerror="alert(1)">'
    state.sample_labels = {"s1": "<script>alert(1)</script>"}
    state.rounds[0].scheduler_snapshot["payload"] = {"raw": "<svg onload=alert(1)>"}

    ReportGenerator().generate_report(state, tmp_path)

    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "<img" not in markdown
    assert "<script>" not in markdown
    assert "<svg" not in markdown
    assert "&lt;img" in markdown
    assert "&lt;script&gt;" in markdown
    assert "&lt;svg" in markdown


def test_markdown_report_escapes_markdown_link_and_image_syntax(tmp_path: Path) -> None:
    state = _state()
    state.project_name = "[click](javascript:alert(1))"
    state.sample_labels = {"s1": "![x](javascript:alert(1))"}
    state.rounds[0].scheduler_snapshot["payload"] = {"raw": "[ref](javascript:alert(1))"}

    ReportGenerator().generate_report(state, tmp_path)

    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "[click](javascript:alert(1))" not in markdown
    assert "![x](javascript:alert(1))" not in markdown
    assert "[ref](javascript:alert(1))" not in markdown
    assert "\\[click\\]\\(javascript:alert\\(1\\)\\)" in markdown
    assert "!\\[x\\]\\(javascript:alert\\(1\\)\\)" in markdown


def test_markdown_report_normalizes_line_separators_in_user_values(tmp_path: Path) -> None:
    state = _state()
    state.project_name = "safe\r## injected"
    state.sample_labels = {"s1": "label\u2028## injected", "s2": "other\u2029---"}

    ReportGenerator().generate_report(state, tmp_path)

    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "\r" not in markdown
    assert "\u2028" not in markdown
    assert "\u2029" not in markdown
    assert "safe \\#\\# injected" in markdown
    assert "label \\#\\# injected" in markdown
    assert "other \\-\\-\\-" in markdown


def test_public_project_generate_report_writes_artifacts(tmp_path: Path) -> None:
    project = ActiveLearningProject("report-project", tmp_path, lock=False)
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
    project.import_labels({"s1": "positive", "s2": "negative"})

    paths = project.generate_report("reports")

    summary_path = tmp_path / "reports" / "summary.json"
    assert paths == {
        "json": summary_path,
        "manifest": tmp_path / "reports" / "manifest.json",
        "markdown": tmp_path / "reports" / "report.md",
        "html": tmp_path / "reports" / "report.html",
    }
    assert summary_path.exists()
    assert (tmp_path / "reports" / "manifest.json").exists()
    assert (tmp_path / "reports" / "report.md").exists()
    assert (tmp_path / "reports" / "report.html").exists()
    parsed = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "reports" / "manifest.json").read_text(encoding="utf-8"))
    assert parsed["metadata"]["project_name"] == "report-project"
    assert parsed["counts"]["labeled"] == 2
    assert parsed["labels"]["distribution"] == {"negative": 1, "positive": 1}
    assert manifest["project"]["project_name"] == "report-project"
    assert manifest["artifacts"]["manifest"] == "manifest.json"
