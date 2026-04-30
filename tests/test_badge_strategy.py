from __future__ import annotations

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
from active_learning_sdk.engine import StrategyScheduler
from active_learning_sdk.strategies import BadgeStrategy
from active_learning_sdk.types import DataSample


class FakeGradientContext:
    def __init__(self, embeddings: Mapping[str, Any]) -> None:
        self._embeddings = dict(embeddings)
        self.gradient_embed_calls: list[list[str]] = []
        self._strategy_diagnostics: list[dict[str, Any]] = []

    def gradient_embed(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[Any]:
        ids = [str(sample_id) for sample_id in sample_ids]
        self.gradient_embed_calls.append(ids)
        return [self._embeddings[sample_id] for sample_id in ids]

    def embed(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[Any]:
        raise AssertionError("BADGE must use gradient_embed, not embed")

    def record_strategy_diagnostic(self, strategy_name: str, diagnostic: Mapping[str, Any]) -> None:
        self._strategy_diagnostics.append({"strategy": strategy_name, **dict(diagnostic)})

    def clear_strategy_diagnostics(self) -> None:
        self._strategy_diagnostics.clear()

    def consume_strategy_diagnostics(self) -> list[dict[str, Any]]:
        diagnostics = list(self._strategy_diagnostics)
        self._strategy_diagnostics.clear()
        return diagnostics


class GroupedGradientContext(FakeGradientContext):
    def __init__(
        self,
        embeddings: Mapping[str, Any],
        *,
        groups: Mapping[str, str | None],
        labeled_ids: Sequence[str] = (),
    ) -> None:
        super().__init__(embeddings)
        self._groups = dict(groups)
        self.labeled_ids = list(labeled_ids)

    def get_samples(self, sample_ids: Sequence[str]) -> list[DataSample]:
        return [
            DataSample(sample_id=str(sample_id), data={"text": str(sample_id)}, group_id=self._groups.get(str(sample_id)))
            for sample_id in sample_ids
        ]


class FakeColdStartBadgeContext(FakeGradientContext):
    def __init__(
        self,
        *,
        probabilities: Mapping[str, list[float]],
        embeddings: Mapping[str, Any],
        gradient_embeddings: Mapping[str, Any],
    ) -> None:
        super().__init__(gradient_embeddings)
        self._probabilities = dict(probabilities)
        self._regular_embeddings = dict(embeddings)
        self.label_schema = LabelSchema(
            task="text_classification",
            labels=["known_a", "known_b", "unseen_c", "unseen_d", "unseen_e"],
        )
        self.predict_proba_calls: list[list[str]] = []
        self.embed_calls: list[list[str]] = []

    def predict_proba(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        ids = [str(sample_id) for sample_id in sample_ids]
        self.predict_proba_calls.append(ids)
        return [self._probabilities[sample_id] for sample_id in ids]

    def embed(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[Any]:
        ids = [str(sample_id) for sample_id in sample_ids]
        self.embed_calls.append(ids)
        return [self._regular_embeddings[sample_id] for sample_id in ids]


class InMemoryDataset:
    def __init__(self) -> None:
        self._samples = {
            "s1": DataSample(sample_id="s1", data={"text": "one"}),
            "s2": DataSample(sample_id="s2", data={"text": "two"}),
            "s3": DataSample(sample_id="s3", data={"text": "three"}),
        }

    def iter_sample_ids(self):
        yield from self._samples

    def get_sample(self, sample_id: str) -> DataSample:
        return self._samples[sample_id]

    def get_samples(self, sample_ids: Sequence[str]) -> list[DataSample]:
        return [self.get_sample(sample_id) for sample_id in sample_ids]

    def schema(self) -> dict[str, str]:
        return {"sample_id": "str", "text": "str"}


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
        return RoundProgress(total=len(task_ids), done=len(task_ids), ready_sample_ids=list(task_ids))

    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        return RoundPullResult(annotations={})


class FitEvaluateOnlyModel:
    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs) -> None:
        return None

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> dict[str, float]:
        return {"accuracy": 0.0}


class GradientEmbeddingModel(FitEvaluateOnlyModel):
    def gradient_embed(
        self,
        texts: Sequence[str],
        labels: Sequence[Any] | None = None,
        batch_size: int = 32,
    ) -> list[list[float]]:
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


def _configure_project(tmp_path: Path, *, model: Any) -> ActiveLearningProject:
    project = ActiveLearningProject("badge-test", tmp_path, lock=False)
    project.configure(
        dataset=InMemoryDataset(),
        model=model,
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=NoopBackend(),
        scheduler_config=SchedulerConfig(strategy="badge"),
        cache_config=CacheConfig(enable=False),
        split_config=SplitConfig(
            mode="explicit",
            explicit_splits={"train": ["s1"], "val": ["s2"], "test": ["s3"]},
        ),
    )
    return project


def _assert_valid_unique_selection(selected: Sequence[str], pool_ids: Sequence[str], k: int) -> None:
    assert len(selected) == min(k, len(set(pool_ids)))
    assert len(selected) == len(set(selected))
    assert set(selected).issubset(set(pool_ids))


def test_badge_selects_diverse_endpoints_on_simple_gradient_geometry() -> None:
    context = FakeGradientContext(
        {
            "left": [-10.0, 0.0],
            "middle": [0.0, 0.0],
            "right": [10.0, 0.0],
        }
    )

    selected = BadgeStrategy().select(["left", "middle", "right"], 2, context)

    assert selected == ["left", "right"]
    assert context.gradient_embed_calls == [["left", "middle", "right"]]


def test_badge_uses_representative_embedding_exploration_when_label_support_is_sparse() -> None:
    context = FakeColdStartBadgeContext(
        probabilities={
            "known_1": [0.5, 0.5, 0.0, 0.0, 0.0],
            "known_2": [0.51, 0.49, 0.0, 0.0, 0.0],
            "novel_far": [0.99, 0.01, 0.0, 0.0, 0.0],
        },
        embeddings={
            "known_1": [0.0, 0.0],
            "known_2": [0.1, 0.0],
            "novel_far": [10.0, 0.0],
        },
        gradient_embeddings={
            "known_1": [100.0, 0.0],
            "known_2": [90.0, 0.0],
            "novel_far": [1.0, 0.0],
        },
    )

    selected = BadgeStrategy().select(["known_1", "known_2", "novel_far"], 2, context)

    assert selected == ["novel_far", "known_1"]
    assert context.predict_proba_calls == [["known_1", "known_2", "novel_far"]]
    assert context.embed_calls == [["known_1", "known_2", "novel_far"]]
    assert context.gradient_embed_calls == [["known_1", "known_2", "novel_far"]]


def test_badge_continues_with_gradient_embeddings_when_predict_proba_is_absent() -> None:
    class NoProbabilityContext(FakeGradientContext):
        label_schema = LabelSchema(
            task="text_classification",
            labels=["known_a", "known_b", "unseen_c", "unseen_d", "unseen_e"],
        )

    context = NoProbabilityContext(
        {
            "left": [-10.0, 0.0],
            "middle": [0.0, 0.0],
            "right": [10.0, 0.0],
        }
    )

    selected = BadgeStrategy().select(["left", "middle", "right"], 2, context)

    assert selected == ["left", "right"]
    assert context.gradient_embed_calls == [["left", "middle", "right"]]


def test_badge_avoids_already_labeled_groups_when_group_ids_are_available() -> None:
    context = GroupedGradientContext(
        {
            "labeled-a": [0.0, 0.0],
            "a2": [100.0, 0.0],
            "b1": [10.0, 0.0],
            "c1": [-10.0, 0.0],
        },
        groups={"labeled-a": "group-a", "a2": "group-a", "b1": "group-b", "c1": "group-c"},
        labeled_ids=["labeled-a"],
    )

    selected = BadgeStrategy().select(["a2", "b1", "c1"], 2, context)

    assert set(selected) == {"b1", "c1"}


@pytest.mark.parametrize(
    ("probabilities", "match"),
    [
        ([["wrong-count"]], "badge\\.predict_proba returned 1 rows for 2 pool ids"),
        ([[0.5, 0.5], [0.5]], "badge\\.predict_proba row 1.*must have at least 2 probability columns"),
        ([[0.5, 0.5], [True, 0.0]], "badge\\.predict_proba value at row 1, column 0 must be numeric"),
        ([[0.5, 0.5], [0.6, 0.6]], "badge\\.predict_proba row 1.*must sum to 1.0"),
    ],
)
def test_badge_malformed_cold_start_probability_output_raises_configuration_error(
    probabilities: list[Any],
    match: str,
) -> None:
    class MalformedProbabilityContext(FakeGradientContext):
        label_schema = LabelSchema(
            task="text_classification",
            labels=["negative", "positive"],
        )

        def predict_proba(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[Any]:
            return probabilities

    context = MalformedProbabilityContext({"a": [1.0, 0.0], "b": [0.0, 1.0]})

    with pytest.raises(ConfigurationError, match=match):
        BadgeStrategy().select(["a", "b"], 1, context)
    assert context.gradient_embed_calls == []


def test_badge_is_deterministic_and_duplicate_free_with_identical_embeddings_and_duplicate_pool_ids() -> None:
    context = FakeGradientContext({sample_id: [1.0, 1.0] for sample_id in ("a", "b", "c")})
    pool = ["b", "a", "b", "c", "a"]

    first = BadgeStrategy().select(pool, 5, context)
    second = BadgeStrategy().select(pool, 5, context)

    assert first == second
    assert first == ["b", "a", "c"]
    _assert_valid_unique_selection(first, pool, 5)


def test_badge_returns_empty_for_empty_pool_and_non_positive_k() -> None:
    assert BadgeStrategy().select([], 3, FakeGradientContext({})) == []
    assert BadgeStrategy().select(["a"], 0, FakeGradientContext({"a": [1.0]})) == []
    assert BadgeStrategy().select(["a"], -1, FakeGradientContext({"a": [1.0]})) == []


@pytest.mark.parametrize(
    ("embeddings", "match"),
    [
        ([["wrong-count"]], "returned 1 rows for 2 sample ids"),
        ([1.0, 2.0], "must be a sequence"),
        ([[1.0], [1.0, 2.0]], "expected 1"),
        ([[], []], "must not be empty"),
        ([[1.0], [True]], "must be numeric"),
        ([[1.0], ["bad"]], "must be numeric"),
        ([[1.0], [float("nan")]], "must be finite"),
        ([[1.0], [[2.0]]], "must be numeric"),
    ],
)
def test_malformed_gradient_embeddings_raise_configuration_error(embeddings: list[Any], match: str) -> None:
    class MalformedContext(FakeGradientContext):
        def gradient_embed(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[Any]:
            return embeddings

    with pytest.raises(ConfigurationError, match=match):
        BadgeStrategy().select(["a", "b"], 1, MalformedContext({}))


def test_configuring_badge_without_gradient_embed_fails(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="badge.*gradient_embed"):
        _configure_project(tmp_path, model=FitEvaluateOnlyModel())


def test_configuring_badge_with_gradient_capable_model_succeeds(tmp_path: Path) -> None:
    project = _configure_project(tmp_path, model=GradientEmbeddingModel())

    assert project.get_state().scheduler_config["strategy"] == "badge"


def test_attach_runtime_requires_gradient_embed_for_persisted_badge_project(tmp_path: Path) -> None:
    project = _configure_project(tmp_path, model=GradientEmbeddingModel())
    project.close()

    reopened = ActiveLearningProject("badge-test", tmp_path, lock=False)
    with pytest.raises(ConfigurationError, match="badge.*gradient_embed"):
        reopened.attach_runtime(
            dataset=InMemoryDataset(),
            model=FitEvaluateOnlyModel(),
            label_backend=NoopBackend(),
        )

    reopened.attach_runtime(
        dataset=InMemoryDataset(),
        model=GradientEmbeddingModel(),
        label_backend=NoopBackend(),
    )


def test_strategy_scheduler_can_select_with_badge() -> None:
    context = FakeGradientContext(
        {
            "a": [0.0, 0.0],
            "b": [3.0, 0.0],
            "c": [9.0, 0.0],
        }
    )
    scheduler = StrategyScheduler(SchedulerConfig(strategy="badge"))

    selected, snapshot = scheduler.select_batch(["a", "b", "c"], 2, context, state={})

    _assert_valid_unique_selection(selected, ["a", "b", "c"], 2)
    assert selected == ["c", "a"]
    assert snapshot == {"mode": "single", "strategy": "badge"}


def test_strategy_scheduler_records_badge_cold_start_fallback_diagnostics() -> None:
    context = FakeColdStartBadgeContext(
        probabilities={
            "known_1": [0.5, 0.5, 0.0, 0.0, 0.0],
            "known_2": [0.51, 0.49, 0.0, 0.0, 0.0],
            "novel_far": [0.99, 0.01, 0.0, 0.0, 0.0],
        },
        embeddings={
            "known_1": [0.0, 0.0],
            "known_2": [0.1, 0.0],
            "novel_far": [10.0, 0.0],
        },
        gradient_embeddings={
            "known_1": [100.0, 0.0],
            "known_2": [90.0, 0.0],
            "novel_far": [1.0, 0.0],
        },
    )
    scheduler = StrategyScheduler(SchedulerConfig(strategy="badge"))

    selected, snapshot = scheduler.select_batch(["known_1", "known_2", "novel_far"], 2, context, state={})

    assert selected == ["novel_far", "known_1"]
    assert context.gradient_embed_calls == [["known_1", "known_2", "novel_far"]]
    assert snapshot == {
        "mode": "single",
        "strategy": "badge",
        "strategy_diagnostics": [
            {
                "strategy": "badge",
                "effective_strategy": "cold_start_blend:max_min_embedding+badge",
                "fallback_reason": "cold_start_sparse_probability_support",
                "label_count": 5,
                "support_count": 2,
                "support_fraction": 0.4,
                "missing_label_count": 3,
                "fallback_mode": "blend",
                "exploration_count": 1,
                "exploitation_count": 1,
            }
        ],
    }
