from __future__ import annotations

import json
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
    StopCriteria,
)
from active_learning_sdk.backends.base import RoundProgress, RoundPullResult, RoundPushResult
from active_learning_sdk.engine import ActiveLearningEngine
from active_learning_sdk.state.store import RoundState
from active_learning_sdk.types import DataSample, MetricRecord, RoundStatus, SampleStatus


class InMemoryDataset:
    def __init__(self, sample_ids: Sequence[str] = ("s1", "s2", "s3")) -> None:
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

    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs) -> None:
        return None

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> dict[str, float]:
        return {"accuracy": 0.5, "ece": 0.1}


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


def _engine(tmp_path: Path) -> ActiveLearningEngine:
    engine = ActiveLearningEngine("test-project", tmp_path, lock=False)
    state = engine.get_state()
    state.sample_status = {
        "s1": SampleStatus.LABELED.value,
        "s2": SampleStatus.LABELED.value,
        "s3": SampleStatus.UNLABELED.value,
    }
    state.sample_labels = {"s1": "positive", "s2": "negative"}
    return engine


def _metric(step: int, **metrics: float) -> MetricRecord:
    return MetricRecord(step=f"eval-{step}", created_at=float(step), metrics=dict(metrics))


def _done_round(
    index: int,
    *,
    snapshot: dict[str, Any] | None = None,
    resolved: dict[str, Any] | None = None,
) -> RoundState:
    return RoundState(
        round_id=f"r{index:04d}",
        status=RoundStatus.DONE,
        created_at=float(index),
        updated_at=float(index),
        selected_sample_ids=[f"s{index}"],
        scheduler_snapshot=snapshot or {},
        resolved=resolved or {},
    )


def _round(
    index: int,
    *,
    status: RoundStatus,
    snapshot: dict[str, Any] | None = None,
    resolved: dict[str, Any] | None = None,
) -> RoundState:
    return RoundState(
        round_id=f"r{index:04d}",
        status=status,
        created_at=float(index),
        updated_at=float(index),
        selected_sample_ids=[f"s{index}"],
        scheduler_snapshot=snapshot or {},
        resolved=resolved or {},
    )


def _configured_project(tmp_path: Path) -> ActiveLearningProject:
    project = ActiveLearningProject("test-project", tmp_path, lock=False)
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


def test_min_labeled_prevents_plateau_stop_before_enough_labels(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.get_state().metrics_history = [_metric(1, accuracy=0.8), _metric(2, accuracy=0.8)]

    stopped = engine._should_stop(
        StopCriteria(min_labeled=3, plateau_rounds=1, min_improvement=0.01)
    )

    assert stopped is False
    assert engine.get_state().scheduler_state["stop_trace"]["reason"] == "minimums_not_met"


def test_min_rounds_prevents_plateau_stop_before_enough_rounds(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    state = engine.get_state()
    state.metrics_history = [_metric(1, accuracy=0.8), _metric(2, accuracy=0.8)]
    state.rounds = [_done_round(1)]

    stopped = engine._should_stop(
        StopCriteria(min_rounds=2, plateau_rounds=1, min_improvement=0.01)
    )

    assert stopped is False
    assert state.scheduler_state["stop_trace"]["completed_round_count"] == 1


def test_metric_plateau_stop_writes_trace(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.get_state().metrics_history = [_metric(1, accuracy=0.8), _metric(2, accuracy=0.8)]

    assert engine._should_stop(StopCriteria(plateau_rounds=1, min_improvement=0.01)) is True

    trace = engine.get_state().scheduler_state["stop_trace"]
    assert trace["stopped"] is True
    assert trace["reason"] == "metric_plateau"
    assert trace["criteria"]["metric_name"] == "accuracy"
    assert trace["observed_values"]["recent_values"] == [0.8, 0.8]


def test_acquisition_score_convergence_stop_writes_trace(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.get_state().rounds = [
        _done_round(1, snapshot={"score_mean": 0.42}),
        _done_round(2, snapshot={"score_mean": 0.421}),
        _done_round(3, snapshot={"score_mean": 0.4215}),
    ]

    stopped = engine._should_stop(
        StopCriteria(acquisition_score_rounds=2, acquisition_score_min_delta=0.002)
    )

    assert stopped is True
    trace = engine.get_state().scheduler_state["stop_trace"]
    assert trace["reason"] == "acquisition_score_convergence"
    assert trace["observed_values"]["scores"] == [0.42, 0.421, 0.4215]


def test_acquisition_score_convergence_ignores_incomplete_and_failed_rounds(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.get_state().rounds = [
        _done_round(1, snapshot={"score_mean": 0.42}),
        _round(2, status=RoundStatus.FAILED, snapshot={"score_mean": 9.0}),
        _done_round(3, snapshot={"score_mean": 0.421}),
        _round(4, status=RoundStatus.PUSHED, snapshot={"score_mean": 9.0}),
        _done_round(5, snapshot={"score_mean": 0.4215}),
    ]

    stopped = engine._should_stop(
        StopCriteria(acquisition_score_rounds=2, acquisition_score_min_delta=0.002)
    )

    assert stopped is True
    trace = engine.get_state().scheduler_state["stop_trace"]
    assert trace["observed_values"]["round_ids"] == ["r0001", "r0003", "r0005"]
    assert trace["observed_values"]["scores"] == [0.42, 0.421, 0.4215]


def test_acquisition_score_convergence_requires_recent_completed_round_scores(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.get_state().rounds = [
        _done_round(1, snapshot={"score_mean": 0.42}),
        _done_round(2, snapshot={"score_mean": 0.421}),
        _done_round(3),
        _done_round(4, snapshot={"score_mean": 0.4215}),
    ]

    stopped = engine._should_stop(
        StopCriteria(acquisition_score_rounds=2, acquisition_score_min_delta=0.002)
    )

    assert stopped is False
    trace = engine.get_state().scheduler_state["stop_trace"]
    observed = trace["observed_values"]["acquisition_score_convergence"]
    assert trace["reason"] == "criteria_not_met"
    assert observed["recent_completed_round_ids"] == ["r0002", "r0003", "r0004"]
    assert observed["missing_score_round_ids"] == ["r0003"]


def test_no_acquisition_score_data_means_no_stop(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.get_state().rounds = [_done_round(1), _done_round(2), _done_round(3)]

    assert engine._should_stop(StopCriteria(acquisition_score_rounds=2)) is False

    assert engine.get_state().scheduler_state["stop_trace"]["reason"] == "criteria_not_met"


def test_non_finite_acquisition_scores_are_skipped_and_trace_is_strict_json(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.get_state().rounds = [
        _done_round(1, snapshot={"score_mean": 0.5}),
        _done_round(2, snapshot={"score_mean": float("nan")}),
        _done_round(3, snapshot={"scores": [float("inf"), 0.51]}),
    ]

    stopped = engine._should_stop(
        StopCriteria(acquisition_score_rounds=2, acquisition_score_min_delta=0.02)
    )

    assert stopped is False
    trace = engine.get_state().scheduler_state["stop_trace"]
    observed = trace["observed_values"]["acquisition_score_convergence"]
    assert observed["scores"] == [0.5, 0.51]
    assert observed["missing_score_round_ids"] == ["r0002"]
    json.dumps(trace, allow_nan=False)


def test_label_distribution_stabilization_stop(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.get_state().rounds = [
        _done_round(1, resolved={"s1": "positive", "s2": "negative"}),
        _done_round(2, resolved={"s3": "positive", "s4": "negative"}),
        _done_round(3, resolved={"s5": "positive", "s6": "negative"}),
    ]

    stopped = engine._should_stop(
        StopCriteria(label_distribution_rounds=2, label_distribution_max_delta=0.0)
    )

    assert stopped is True
    trace = engine.get_state().scheduler_state["stop_trace"]
    assert trace["reason"] == "label_distribution_stabilization"
    assert trace["observed_values"]["l1_deltas"] == [0.0, 0.0]


def test_label_distribution_insufficient_data_does_not_stop(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.get_state().rounds = [
        _done_round(1, resolved={"s1": "positive", "s2": "negative"}),
        _done_round(2, resolved={}),
    ]

    assert (
        engine._should_stop(
            StopCriteria(label_distribution_rounds=1, label_distribution_max_delta=0.0)
        )
        is False
    )

    assert engine.get_state().scheduler_state["stop_trace"]["reason"] == "criteria_not_met"


def test_calibration_stabilization_stop(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.get_state().metrics_history = [
        _metric(1, accuracy=0.7, ece=0.1),
        _metric(2, accuracy=0.7, ece=0.101),
        _metric(3, accuracy=0.7, ece=0.1005),
    ]

    stopped = engine._should_stop(
        StopCriteria(calibration_rounds=2, calibration_min_delta=0.002)
    )

    assert stopped is True
    assert engine.get_state().scheduler_state["stop_trace"]["reason"] == "calibration_stabilization"


def test_malformed_calibration_metrics_do_not_crash_stop_evaluation(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.get_state().metrics_history = [
        _metric(1, accuracy=0.7, ece=0.1),
        _metric(2, accuracy=0.7, ece="bad"),
        _metric(3, accuracy=0.7, ece=None),
    ]

    stopped = engine._should_stop(
        StopCriteria(calibration_rounds=2, calibration_min_delta=0.002)
    )

    assert stopped is False
    trace = engine.get_state().scheduler_state["stop_trace"]
    assert trace["reason"] == "criteria_not_met"
    observed = trace["observed_values"]["calibration_stabilization"]
    assert observed["values"] == [0.1]
    assert observed["skipped_values"] == [
        {"record_index": 1, "step": "eval-2", "value": "'bad'", "reason": "not_numeric"},
        {"record_index": 2, "step": "eval-3", "value": "None", "reason": "not_numeric"},
    ]


def test_malformed_plateau_metrics_do_not_crash_stop_evaluation(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.get_state().metrics_history = [
        _metric(1, accuracy=0.7),
        _metric(2, accuracy=float("nan")),
        _metric(3, accuracy="bad"),
    ]

    stopped = engine._should_stop(StopCriteria(plateau_rounds=1, min_improvement=0.01))

    assert stopped is False
    trace = engine.get_state().scheduler_state["stop_trace"]
    observed = trace["observed_values"]["metric_plateau"]
    assert observed["values"] == [0.7]
    assert observed["skipped_values"] == [
        {"record_index": 1, "step": "eval-2", "value": "nan", "reason": "not_finite"},
        {"record_index": 2, "step": "eval-3", "value": "'bad'", "reason": "not_numeric"},
    ]


def test_non_stop_trace_includes_configured_criterion_observations(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    state = engine.get_state()
    state.metrics_history = [_metric(1, accuracy=0.8, ece=0.1)]
    state.rounds = [
        _done_round(1, snapshot={"score_mean": 0.42}, resolved={"s1": "positive", "s2": "negative"})
    ]

    stopped = engine._should_stop(
        StopCriteria(
            plateau_rounds=1,
            min_improvement=0.01,
            acquisition_score_rounds=1,
            label_distribution_rounds=1,
            label_distribution_max_delta=0.0,
            calibration_rounds=1,
        )
    )

    assert stopped is False
    trace = state.scheduler_state["stop_trace"]
    assert trace["reason"] == "criteria_not_met"
    assert trace["observed_values"]["metric_plateau"]["values"] == [0.8]
    assert trace["observed_values"]["acquisition_score_convergence"]["scores"] == [0.42]
    assert trace["observed_values"]["label_distribution_stabilization"]["distributions"] == [
        {"negative": 0.5, "positive": 0.5}
    ]
    assert trace["observed_values"]["calibration_stabilization"]["values"] == [0.1]


def test_max_labeled_still_clips_effective_batch_size(tmp_path: Path) -> None:
    engine = _engine(tmp_path)

    assert engine._effective_batch_size(10, StopCriteria(max_labeled=3)) == 1


@pytest.mark.parametrize(
    "criteria",
    [
        StopCriteria(max_labeled=-1),
        StopCriteria(max_rounds=-1),
        StopCriteria(min_labeled=-1),
        StopCriteria(min_rounds=-1),
        StopCriteria(min_labeled=3, max_labeled=2),
        StopCriteria(min_rounds=3, max_rounds=2),
        StopCriteria(plateau_rounds=0),
        StopCriteria(min_improvement=-0.1),
        StopCriteria(acquisition_score_key=""),
        StopCriteria(acquisition_score_rounds=0),
        StopCriteria(acquisition_score_min_delta=-0.1),
        StopCriteria(label_distribution_rounds=0, label_distribution_max_delta=0.0),
        StopCriteria(label_distribution_rounds=1),
        StopCriteria(label_distribution_max_delta=-0.1),
        StopCriteria(calibration_metric_name=""),
        StopCriteria(calibration_rounds=0),
        StopCriteria(calibration_min_delta=-0.1),
    ],
)
def test_stop_criteria_validation_errors(criteria: StopCriteria) -> None:
    with pytest.raises(ConfigurationError):
        criteria.validate()


def test_trace_persists_in_project_state_after_run_stops(tmp_path: Path) -> None:
    project = _configured_project(tmp_path)
    project.import_labels({"s1": "positive", "s2": "negative"})

    project.run(batch_size=1, stop_criteria=StopCriteria(max_labeled=2))

    reopened = ActiveLearningProject("test-project", tmp_path, lock=False)
    trace = reopened.get_state().scheduler_state["stop_trace"]
    assert trace["stopped"] is True
    assert trace["reason"] == "max_labeled"
    assert trace["labeled_count"] == 2


def test_exhausted_pool_run_stop_writes_stopped_trace(tmp_path: Path) -> None:
    project = ActiveLearningProject("test-project", tmp_path, lock=False)
    project.configure(
        dataset=InMemoryDataset(sample_ids=("s1", "s2")),
        model=DummyModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=NoopBackend(),
        scheduler_config=SchedulerConfig(strategy="random"),
        cache_config=CacheConfig(enable=False),
        split_config=SplitConfig(
            mode="explicit",
            explicit_splits={"train": ["s1", "s2"], "val": [], "test": []},
        ),
    )
    project.import_labels({"s1": "positive", "s2": "negative"})

    project.run(batch_size=1)

    reopened = ActiveLearningProject("test-project", tmp_path, lock=False)
    trace = reopened.get_state().scheduler_state["stop_trace"]
    assert trace["stopped"] is True
    assert trace["reason"] == "no_unlabeled_samples"
    assert trace["labeled_count"] == 2
