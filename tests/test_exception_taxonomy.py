from __future__ import annotations

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
    ModelAdapterError,
    PrelabelConfig,
    SchedulerConfig,
    SplitConfig,
)
from active_learning_sdk.backends.base import RoundProgress, RoundPullResult, RoundPushResult
from active_learning_sdk.engine import SelectionContext
from active_learning_sdk.types import AnnotationRecord, DataSample, StepKind


class TinyProvider:
    def __init__(self) -> None:
        self.rows = {
            "s1": "seed positive",
            "s2": "seed negative",
            "s3": "candidate positive",
            "s4": "candidate negative",
        }

    def iter_sample_ids(self):
        yield from self.rows

    def get_sample(self, sample_id: str) -> DataSample:
        return DataSample(sample_id=sample_id, data={"text": self.rows[sample_id]})

    def get_samples(self, sample_ids: Sequence[str]) -> list[DataSample]:
        return [self.get_sample(sample_id) for sample_id in sample_ids]

    def schema(self) -> dict[str, str]:
        return {"sample_id": "str", "text": "str"}


class GoodModel:
    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        return [[0.6, 0.4] for _ in texts]

    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs: Any) -> None:
        return None

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> dict[str, float]:
        return {"accuracy": 1.0}


class BadProbaModel(GoodModel):
    def __init__(self, output: Any) -> None:
        self.output = output

    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> Any:
        del texts, batch_size
        return self.output


class ReadyBackend:
    def ensure_ready(self, label_schema: LabelSchema) -> dict[str, Any]:
        return {"ready": True}

    def push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: dict[str, Any] | None = None,
    ) -> RoundPushResult:
        del round_id, prelabels
        return RoundPushResult(
            task_ids={sample.sample_id: f"task:{sample.sample_id}" for sample in samples},
            backend_round_ref={"backend": "ready"},
        )

    def poll_round(self, round_id: str, task_ids: Mapping[str, str], policy: AnnotationPolicy) -> RoundProgress:
        del round_id, policy
        return RoundProgress(total=len(task_ids), done=len(task_ids), ready_sample_ids=list(task_ids))

    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        del round_id
        return RoundPullResult(
            annotations={
                sample_id: [AnnotationRecord(annotator_id="ann", created_at=1.0, value="positive")]
                for sample_id in task_ids
            }
        )


class FailingEnsureReadyBackend(ReadyBackend):
    def ensure_ready(self, label_schema: LabelSchema) -> dict[str, Any]:
        del label_schema
        raise RuntimeError("intentional backend failure in ensure_ready")


class FailingPushBackend(ReadyBackend):
    def push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: dict[str, Any] | None = None,
    ) -> RoundPushResult:
        del round_id, samples, prelabels
        raise RuntimeError("intentional backend failure in push_round")


class FailingPollBackend(ReadyBackend):
    def poll_round(self, round_id: str, task_ids: Mapping[str, str], policy: AnnotationPolicy) -> RoundProgress:
        del round_id, task_ids, policy
        raise RuntimeError("intentional backend failure in poll_round")


class FailingPullBackend(ReadyBackend):
    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        del round_id, task_ids
        raise RuntimeError("intentional backend failure in pull_round")


def _configure_project(
    tmp_path: Path,
    *,
    backend: Any,
    model: Any | None = None,
    strategy: str = "random",
    prelabel_config: PrelabelConfig | None = None,
) -> ActiveLearningProject:
    project = ActiveLearningProject("wave7-taxonomy", tmp_path, lock=False)
    project.configure(
        dataset=TinyProvider(),
        model=model or GoodModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=backend,
        scheduler_config=SchedulerConfig(strategy=strategy),
        annotation_policy=AnnotationPolicy(mode="latest", min_votes=1),
        cache_config=CacheConfig(enable=False),
        split_config=SplitConfig(
            mode="explicit",
            explicit_splits={"train": ["s1", "s2", "s3", "s4"], "val": [], "test": []},
        ),
        prelabel_config=prelabel_config or PrelabelConfig(enable=False),
    )
    return project


def test_backend_ensure_ready_runtime_error_is_label_backend_error(tmp_path: Path) -> None:
    with pytest.raises(LabelBackendError, match="ensure_ready"):
        _configure_project(tmp_path, backend=FailingEnsureReadyBackend())


def test_backend_push_runtime_error_is_label_backend_error(tmp_path: Path) -> None:
    project = _configure_project(tmp_path, backend=FailingPushBackend())

    assert project.run_step(batch_size=1).step == StepKind.SELECT
    with pytest.raises(LabelBackendError, match="push_round"):
        project.run_step(batch_size=1)


def test_backend_poll_runtime_error_is_label_backend_error(tmp_path: Path) -> None:
    project = _configure_project(tmp_path, backend=FailingPollBackend())

    assert project.run_step(batch_size=1).step == StepKind.SELECT
    assert project.run_step(batch_size=1).step == StepKind.PUSH
    with pytest.raises(LabelBackendError, match="poll_round"):
        project.run_step(batch_size=1)


def test_backend_pull_runtime_error_is_label_backend_error(tmp_path: Path) -> None:
    project = _configure_project(tmp_path, backend=FailingPullBackend())

    assert project.run_step(batch_size=1).step == StepKind.SELECT
    assert project.run_step(batch_size=1).step == StepKind.PUSH
    assert project.run_step(batch_size=1).step == StepKind.WAIT
    with pytest.raises(LabelBackendError, match="pull_round"):
        project.run_step(batch_size=1)


@pytest.mark.parametrize("bad_output", [None, 0.25])
def test_selection_context_predict_proba_top_level_contract_errors_use_sdk_taxonomy(bad_output: Any) -> None:
    context = SelectionContext(
        provider=TinyProvider(),
        model=BadProbaModel(bad_output),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        prediction_cache=None,
        embedding_cache=None,
        labeled_ids=[],
        last_metrics={},
    )

    with pytest.raises(ModelAdapterError, match="model\\.predict_proba"):
        context.predict_proba(["s3", "s4"])


@pytest.mark.parametrize("bad_output", [None, 0.25])
def test_prelabel_predict_proba_top_level_contract_errors_use_sdk_taxonomy(
    tmp_path: Path,
    bad_output: Any,
) -> None:
    project = _configure_project(
        tmp_path,
        backend=ReadyBackend(),
        model=BadProbaModel(bad_output),
        prelabel_config=PrelabelConfig(enable=True, min_confidence=0.0),
    )

    assert project.run_step(batch_size=1).step == StepKind.SELECT
    with pytest.raises(ModelAdapterError, match="model\\.predict_proba"):
        project.run_step(batch_size=1)
