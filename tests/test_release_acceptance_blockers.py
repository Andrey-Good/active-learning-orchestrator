from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from active_learning_sdk.cache import InMemoryCacheStore, PredictionCache
from active_learning_sdk.engine import SelectionContext, StrategyScheduler
from active_learning_sdk.strategies import EntropyStrategy
from active_learning_sdk.types import AnnotationRecord, DataSample, RoundStatus, SampleStatus, StepKind
from benchmarks import manual_strategy_benchmark


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

    def get_texts(self, sample_ids: Sequence[str]) -> list[str]:
        return [self._samples[sample_id].data["text"] for sample_id in sample_ids]

    def schema(self) -> dict[str, str]:
        return {"sample_id": "str", "text": "str"}


class DummyModel:
    def get_model_id(self) -> str:
        return "senior-acceptance-model"

    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        return [[0.5, 0.5] for _ in texts]

    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs: Any) -> None:
        return None

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> dict[str, float]:
        return {"accuracy": 1.0}


class AllTaskIdsBackend:
    def __init__(
        self,
        all_sample_ids: Sequence[str],
        annotations: Mapping[str, Sequence[AnnotationRecord]] | None = None,
    ) -> None:
        self.all_sample_ids = list(all_sample_ids)
        self.annotations = {sample_id: list(records) for sample_id, records in (annotations or {}).items()}

    def ensure_ready(self, label_schema: LabelSchema) -> dict[str, Any]:
        return {"backend": "ready"}

    def push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: dict[str, Any] | None = None,
    ) -> RoundPushResult:
        del round_id, samples, prelabels
        return RoundPushResult(task_ids={sample_id: sample_id for sample_id in self.all_sample_ids})

    def poll_round(self, round_id: str, task_ids: Mapping[str, str], policy: Any) -> RoundProgress:
        del round_id, policy
        return RoundProgress(total=len(task_ids), done=len(task_ids), ready_sample_ids=list(task_ids))

    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        del round_id
        return RoundPullResult(
            annotations={sample_id: list(self.annotations.get(sample_id, [])) for sample_id in task_ids}
        )

    def close(self) -> None:
        return None


class PartialInvalidPullBackend(AllTaskIdsBackend):
    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        del round_id, task_ids
        return RoundPullResult(
            annotations={
                "s1": [AnnotationRecord(annotator_id="a", created_at=1.0, value="alpha")],
                "s2": [AnnotationRecord(annotator_id="a", created_at=2.0, value="outside-schema")],
            }
        )


class EmptyPullBackend(AllTaskIdsBackend):
    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        del round_id, task_ids
        return RoundPullResult(annotations={})


class NamedStrategy:
    required_capabilities = frozenset()

    def __init__(self, name: str, selected: str) -> None:
        self.name = name
        self.selected = selected

    def select(self, pool_ids: Sequence[str], k: int, context: object) -> list[str]:
        del pool_ids, context
        return [self.selected][:k]


class ZeroSumModel(DummyModel):
    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        return [[0.0, 0.0] for _ in texts]


def _configure_project(
    workdir: Path,
    *,
    dataset: InMemoryDataset | None = None,
    backend: Any | None = None,
    split_config: SplitConfig | None = None,
) -> ActiveLearningProject:
    project = ActiveLearningProject("senior-acceptance", workdir, lock=False)
    project.configure(
        dataset=dataset or InMemoryDataset(),
        model=DummyModel(),
        label_schema=LabelSchema(task="text_classification", labels=["alpha", "beta"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=backend
        or AllTaskIdsBackend(
            ["s1", "s2"],
            annotations={
                "s1": [AnnotationRecord(annotator_id="a", created_at=1.0, value="alpha")],
                "s2": [AnnotationRecord(annotator_id="a", created_at=2.0, value="alpha")],
            },
        ),
        scheduler_config=SchedulerConfig(strategy="random"),
        cache_config=CacheConfig(enable=False),
        split_config=split_config
        or SplitConfig(mode="explicit", explicit_splits={"train": ["s1", "s2"], "val": [], "test": []}),
    )
    return project


def test_backend_push_task_ids_must_match_selected_samples(tmp_path: Path) -> None:
    project = _configure_project(tmp_path)

    assert project.run_step(batch_size=1).step == StepKind.SELECT
    selected = set(project.get_state().rounds[-1].selected_sample_ids)
    assert len(selected) == 1

    with pytest.raises(ConfigurationError, match="task_ids"):
        project.run_step(batch_size=1)


def test_pull_with_later_invalid_label_is_atomic(tmp_path: Path) -> None:
    project = _configure_project(tmp_path, backend=PartialInvalidPullBackend(["s1", "s2"]))

    assert project.run_step(batch_size=2).step == StepKind.SELECT
    assert project.run_step(batch_size=2).step == StepKind.PUSH
    assert project.run_step(batch_size=2).step == StepKind.WAIT

    with pytest.raises(ConfigurationError, match="LabelSchema"):
        project.run_step(batch_size=2)

    state = project.get_state()
    assert state.sample_status == {"s1": SampleStatus.UNLABELED.value, "s2": SampleStatus.UNLABELED.value}
    assert state.sample_labels == {}


def test_pull_must_not_complete_when_selected_task_annotations_are_missing(tmp_path: Path) -> None:
    project = _configure_project(tmp_path, backend=EmptyPullBackend(["s1"]))

    assert project.run_step(batch_size=1).step == StepKind.SELECT
    assert project.run_step(batch_size=1).step == StepKind.PUSH
    assert project.run_step(batch_size=1).step == StepKind.WAIT

    with pytest.raises(ConfigurationError, match="missing"):
        project.run_step(batch_size=1)

    assert project.get_state().rounds[-1].status != RoundStatus.PULLED


def test_explicit_splits_must_reject_overlap_across_splits(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="overlap|duplicate"):
        _configure_project(
            tmp_path,
            dataset=InMemoryDataset(("s1", "s2")),
            split_config=SplitConfig(
                mode="explicit",
                explicit_splits={"train": ["s1"], "val": ["s1"], "test": ["s2"]},
            ),
        )


def test_bandit_scheduler_uses_reward_state_instead_of_arm_order() -> None:
    scheduler = StrategyScheduler(
        SchedulerConfig(mode="bandit", bandit_arms=["bad_arm", "good_arm"], bandit_algo="ucb1"),
        strategies=[NamedStrategy("bad_arm", "bad"), NamedStrategy("good_arm", "good")],
    )
    state = {
        "bandit": {
            "bad_arm": {"n": 100, "reward_sum": 0.0},
            "good_arm": {"n": 100, "reward_sum": 100.0},
        }
    }

    selected, snapshot = scheduler.select_batch(["bad", "good"], 1, object(), state=state)

    assert snapshot["chosen_arm"] == "good_arm"
    assert selected == ["good"]


def test_invalid_predict_proba_rows_do_not_poison_prediction_cache() -> None:
    store = InMemoryCacheStore()
    context = SelectionContext(
        provider=InMemoryDataset(("s1",)),
        model=ZeroSumModel(),
        label_schema=LabelSchema(task="text_classification", labels=["alpha", "beta"]),
        prediction_cache=PredictionCache(store),
        embedding_cache=None,
        labeled_ids=[],
        last_metrics={},
    )

    with pytest.raises(ConfigurationError, match="positive sum"):
        EntropyStrategy().select(["s1"], 1, context)

    assert store.stats()["items"] == 0


def test_manual_benchmark_fixture_must_distinguish_uncertainty_formulas() -> None:
    pool_ids = [candidate.sample_id for candidate in manual_strategy_benchmark.AUDIT_CANDIDATES]
    probabilities_by_id = manual_strategy_benchmark.probability_map()

    selections = {
        strategy: tuple(
            manual_strategy_benchmark.manual_select(pool_ids, 5, probabilities_by_id, strategy)
        )
        for strategy in manual_strategy_benchmark.SUPPORTED_STRATEGIES
    }

    assert len(set(selections.values())) == len(selections)
