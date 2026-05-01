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
from active_learning_sdk.strategies import hybrid as hybrid_module
from active_learning_sdk.strategies.hybrid import normalize_scores
from active_learning_sdk.types import DataSample


class FakeSelectionContext:
    def __init__(
        self,
        probabilities: Mapping[str, Sequence[float]],
        embeddings: Mapping[str, Sequence[float]],
        *,
        groups: Mapping[str, str | None] | None = None,
        labeled_ids: Sequence[str] = (),
    ) -> None:
        self._probabilities = dict(probabilities)
        self._embeddings = dict(embeddings)
        self._groups = dict(groups or {})
        self.labeled_ids = list(labeled_ids)

    def model_id(self) -> str:
        return "fake-hybrid-model"

    def predict_proba(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [list(self._probabilities[sample_id]) for sample_id in sample_ids]

    def embed(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [list(self._embeddings[sample_id]) for sample_id in sample_ids]

    def get_samples(self, sample_ids: Sequence[str]) -> list[DataSample]:
        return [
            DataSample(sample_id=sample_id, data={"text": sample_id}, group_id=self._groups.get(sample_id))
            for sample_id in sample_ids
        ]


class InMemoryDataset:
    def __init__(self) -> None:
        self._samples = {
            "s1": DataSample(sample_id="s1", data={"text": "one"}),
            "s2": DataSample(sample_id="s2", data={"text": "two"}),
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

    def close(self) -> None:
        return None


class FitEvaluateOnlyModel:
    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs) -> None:
        return None

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> dict[str, float]:
        return {"accuracy": 0.0}


class ProbabilityModel(FitEvaluateOnlyModel):
    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [[0.5, 0.5] for _ in texts]


class EmbeddingOnlyModel(FitEvaluateOnlyModel):
    def embed(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [[0.0, 1.0] for _ in texts]


class HybridCapableModel(ProbabilityModel):
    def embed(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


def _configure_project(tmp_path: Path, *, model: Any, scheduler_config: SchedulerConfig) -> ActiveLearningProject:
    project = ActiveLearningProject("hybrid-test", tmp_path, lock=False)
    project.configure(
        dataset=InMemoryDataset(),
        model=model,
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=NoopBackend(),
        scheduler_config=scheduler_config,
        cache_config=CacheConfig(enable=False),
        split_config=SplitConfig(
            mode="explicit",
            explicit_splits={"train": ["s1"], "val": ["s2"], "test": []},
        ),
    )
    return project


def test_scheduler_config_accepts_hybrid_and_validates_mapping() -> None:
    SchedulerConfig(mode="hybrid", hybrid={}).validate()

    with pytest.raises(ConfigurationError, match="hybrid"):
        SchedulerConfig(mode="hybrid").validate()

    with pytest.raises(ConfigurationError, match="Unsupported hybrid.uncertainty"):
        SchedulerConfig(mode="hybrid", hybrid={"uncertainty": "unknown"}).validate()

    with pytest.raises(ConfigurationError, match="uncertainty_weight"):
        SchedulerConfig(mode="hybrid", hybrid={"uncertainty_weight": -0.1}).validate()


def test_hybrid_configure_requires_predict_proba_and_embed(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="entropy.*predict_proba"):
        _configure_project(
            tmp_path / "missing-proba",
            model=EmbeddingOnlyModel(),
            scheduler_config=SchedulerConfig(mode="hybrid", hybrid={}),
        )

    with pytest.raises(ConfigurationError, match="coreset_kcenter.*embed"):
        _configure_project(
            tmp_path / "missing-embed",
            model=ProbabilityModel(),
            scheduler_config=SchedulerConfig(mode="hybrid", hybrid={}),
        )

    project = _configure_project(
        tmp_path / "ok",
        model=HybridCapableModel(),
        scheduler_config=SchedulerConfig(mode="hybrid", hybrid={}),
    )
    assert project.get_state().scheduler_config["mode"] == "hybrid"


def test_weighted_hybrid_returns_deterministic_valid_ids() -> None:
    pool = ["a", "b", "c", "d", "a"]
    context = FakeSelectionContext(
        probabilities={
            "a": [0.50, 0.50],
            "b": [0.55, 0.45],
            "c": [0.90, 0.10],
            "d": [0.80, 0.20],
        },
        embeddings={
            "a": [0.0],
            "b": [1.0],
            "c": [5.0],
            "d": [9.0],
        },
    )
    scheduler = StrategyScheduler(
        SchedulerConfig(
            mode="hybrid",
            hybrid={
                "mode": "weighted",
                "uncertainty_weight": 0.7,
                "diversity_weight": 0.3,
            },
        )
    )

    first, snapshot = scheduler.select_batch(pool, 3, context, state={})
    second, _ = scheduler.select_batch(pool, 3, context, state={})

    assert first == second
    assert len(first) == 3
    assert len(set(first)) == 3
    assert set(first) <= {"a", "b", "c", "d"}
    assert snapshot["mode"] == "hybrid"
    assert snapshot["hybrid"]["mode"] == "weighted"
    assert snapshot["uncertainty"] == "entropy"
    assert snapshot["diversity"] == "coreset_kcenter"


def test_weighted_hybrid_snapshot_reports_actual_fallback_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    context = FakeSelectionContext(
        probabilities={
            "a": [0.50, 0.50],
            "b": [0.55, 0.45],
            "c": [0.90, 0.10],
        },
        embeddings={
            "a": [0.0],
            "b": [1.0],
            "c": [5.0],
        },
    )
    scheduler = StrategyScheduler(SchedulerConfig(mode="hybrid", hybrid={"mode": "weighted"}))

    def constrained_guardrails(ordered_ids, **kwargs):
        del kwargs
        return list(ordered_ids)[:1]

    monkeypatch.setattr(hybrid_module, "_apply_guardrails", constrained_guardrails)

    selected, snapshot = scheduler.select_batch(["a", "b", "c"], 3, context, state={})

    assert len(selected) == 3
    assert len(set(selected)) == 3
    assert snapshot["fallback_count"] == 2


def test_prefilter_modes_behave_differently_on_designed_geometry() -> None:
    context = FakeSelectionContext(
        probabilities={
            "a": [0.50, 0.50],
            "b": [0.55, 0.45],
            "c": [0.60, 0.40],
            "d": [0.99, 0.01],
        },
        embeddings={
            "a": [0.0],
            "b": [1.0],
            "c": [2.0],
            "d": [100.0],
        },
    )
    pool = ["a", "b", "c", "d"]

    uncertainty_first = StrategyScheduler(
        SchedulerConfig(
            mode="hybrid",
            hybrid={
                "mode": "uncertainty_prefilter_diversity",
                "prefilter_multiplier": 1.0,
            },
        )
    )
    diversity_first = StrategyScheduler(
        SchedulerConfig(
            mode="hybrid",
            hybrid={
                "mode": "diversity_prefilter_uncertainty",
                "prefilter_multiplier": 1.0,
            },
        )
    )

    uncertainty_selected, uncertainty_snapshot = uncertainty_first.select_batch(pool, 2, context, state={})
    diversity_selected, diversity_snapshot = diversity_first.select_batch(pool, 2, context, state={})

    assert set(uncertainty_selected) == {"a", "b"}
    assert set(diversity_selected) == {"a", "d"}
    assert uncertainty_selected != diversity_selected
    assert uncertainty_snapshot["prefilter_count"] == 2
    assert diversity_snapshot["prefilter_count"] == 2


def test_hybrid_diversity_components_make_distinct_weighted_and_prefilter_choices() -> None:
    context = FakeSelectionContext(
        probabilities={
            "a": [0.50, 0.50],
            "b": [0.50, 0.50],
            "c": [0.50, 0.50],
            "d": [0.50, 0.50],
        },
        embeddings={
            "labeled": [10.0],
            "a": [0.0],
            "b": [3.0],
            "c": [4.0],
            "d": [8.0],
        },
        labeled_ids=["labeled"],
    )
    pool = ["a", "b", "c", "d"]

    expected = {
        "coreset_kcenter": ["a"],
        "embedding_kmeans_pp": ["c"],
        "max_min_embedding": ["d"],
    }
    mode_configs = (
        {
            "mode": "weighted",
            "uncertainty_weight": 0.0,
            "diversity_weight": 1.0,
        },
        {
            "mode": "diversity_prefilter_uncertainty",
            "prefilter_multiplier": 1.0,
        },
    )

    for mode_config in mode_configs:
        selections = {}
        for component in ("coreset_kcenter", "embedding_kmeans_pp", "max_min_embedding"):
            scheduler = StrategyScheduler(
                SchedulerConfig(
                    mode="hybrid",
                    hybrid={
                        **mode_config,
                        "diversity": component,
                    },
                )
            )

            selected, _ = scheduler.select_batch(pool, 1, context, state={})
            selections[component] = selected

        assert selections == expected


def test_weighted_embedding_kmeans_pp_uses_batch_greedy_order_for_k_gt_one() -> None:
    context = FakeSelectionContext(
        probabilities={
            "center": [0.50, 0.50],
            "near": [0.50, 0.50],
            "far_left": [0.50, 0.50],
            "far_right": [0.50, 0.50],
        },
        embeddings={
            "center": [0.0],
            "near": [1.0],
            "far_left": [-10.0],
            "far_right": [9.0],
        },
    )
    scheduler = StrategyScheduler(
        SchedulerConfig(
            mode="hybrid",
            hybrid={
                "mode": "weighted",
                "uncertainty_weight": 0.0,
                "diversity_weight": 1.0,
                "diversity": "embedding_kmeans_pp",
            },
        )
    )

    selected, _ = scheduler.select_batch(["center", "near", "far_left", "far_right"], 2, context, state={})

    assert selected == ["center", "far_left"]


def test_normalization_handles_constant_and_extreme_finite_values() -> None:
    assert normalize_scores([3.0, 3.0, 3.0]) == [0.0, 0.0, 0.0]

    normalized = normalize_scores([-1e308, 0.0, 1e308])

    assert normalized[0] == pytest.approx(0.0)
    assert normalized[1] == pytest.approx(0.5)
    assert normalized[2] == pytest.approx(1.0)


def test_non_finite_scores_and_model_outputs_raise_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="finite"):
        normalize_scores([0.0, float("nan")])

    scheduler = StrategyScheduler(SchedulerConfig(mode="hybrid", hybrid={}))
    context = FakeSelectionContext(
        probabilities={"a": [float("inf"), 0.5]},
        embeddings={"a": [0.0]},
    )

    with pytest.raises(ConfigurationError, match="finite"):
        scheduler.select_batch(["a"], 1, context, state={})


def test_class_and_group_balance_guardrails_reduce_monopolization() -> None:
    scheduler = StrategyScheduler(
        SchedulerConfig(
            mode="hybrid",
            hybrid={
                "mode": "weighted",
                "uncertainty_weight": 1.0,
                "diversity_weight": 0.0,
                "class_balance": True,
                "group_balance": True,
            },
        )
    )
    context = FakeSelectionContext(
        probabilities={
            "c0a": [0.51, 0.49],
            "c0b": [0.52, 0.48],
            "c0c": [0.53, 0.47],
            "c1a": [0.10, 0.90],
        },
        embeddings={
            "c0a": [0.0],
            "c0b": [0.0],
            "c0c": [0.0],
            "c1a": [0.0],
        },
        groups={
            "c0a": "shared",
            "c0b": "shared",
            "c0c": "shared",
            "c1a": "other",
        },
    )

    selected, _ = scheduler.select_batch(["c0a", "c0b", "c0c", "c1a"], 3, context, state={})

    assert "c1a" in selected
    assert len(selected) == 3
    assert len(set(selected)) == 3


def test_hybrid_group_balance_fails_closed_when_group_lookup_fails() -> None:
    class BrokenGroupContext(FakeSelectionContext):
        def get_samples(self, sample_ids: Sequence[str]) -> list[DataSample]:
            raise RuntimeError("provider unavailable")

    scheduler = StrategyScheduler(
        SchedulerConfig(
            mode="hybrid",
            hybrid={"mode": "weighted", "uncertainty_weight": 1.0, "diversity_weight": 0.0, "group_balance": True},
        )
    )
    context = BrokenGroupContext(
        probabilities={"a": [0.5, 0.5], "b": [0.6, 0.4]},
        embeddings={"a": [0.0], "b": [1.0]},
    )

    with pytest.raises(ConfigurationError, match="group_balance"):
        scheduler.select_batch(["a", "b"], 1, context, state={})
