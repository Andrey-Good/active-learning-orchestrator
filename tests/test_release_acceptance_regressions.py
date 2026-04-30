from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pytest

from active_learning_sdk import (
    ActiveLearningProject,
    CacheConfig,
    ConfigurationError,
    DatasetMismatchError,
    FingerprintConfig,
    LabelBackendConfig,
    LabelSchema,
    SchedulerConfig,
    SplitConfig,
    StateCorruptedError,
)
from active_learning_sdk.backends.simulator import SimulatorLabelBackend
from active_learning_sdk.cache import InMemoryCacheStore, PredictionCache
from active_learning_sdk.engine import SelectionContext
from active_learning_sdk.engine import StrategyScheduler
from active_learning_sdk.types import DataSample, RoundStatus
from active_learning_sdk.strategies.stochastic import CommitteeVoteEntropyStrategy, McDropoutEntropyStrategy


class _EmbeddingModel:
    def get_model_id(self) -> str:
        return "embedding-model-v1"

    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [[0.5, 0.5] for _ in texts]

    def embed(self, texts: Sequence[str], batch_size: int = 32) -> list[list[Any]]:
        return [[1.0, float("nan")] for _ in texts]

    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs: Any) -> None:
        return None

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> dict[str, float]:
        return {"accuracy": 0.0}


class _FailingPredictModel(_EmbeddingModel):
    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        raise RuntimeError("synthetic predict failure")


class _Provider:
    def __init__(self) -> None:
        self._samples = {"s1": DataSample(sample_id="s1", data={"text": "one"}, group_id="g1")}

    def iter_sample_ids(self):
        yield from self._samples.keys()

    def get_sample(self, sample_id: str) -> DataSample:
        return self._samples[sample_id]

    def get_texts(self, sample_ids: Sequence[str]) -> list[str]:
        return [str(self._samples[sample_id].data["text"]) for sample_id in sample_ids]

    def schema(self) -> dict[str, str]:
        return {"sample_id": "str", "text": "str", "group_id": "str"}


class _ThreeColumnModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def get_model_id(self) -> str:
        return "stable-model-id"

    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        self.calls.append([str(text) for text in texts])
        return [[0.2, 0.3, 0.5] for _ in texts]


class _HybridWidthContext:
    label_schema = LabelSchema(task="text_classification", labels=["negative", "neutral", "positive"])

    def model_id(self) -> str:
        return "hybrid-width-model"

    def predict_proba(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [[0.5, 0.5] for _ in sample_ids]

    def embed(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [[float(index), 0.0] for index, _ in enumerate(sample_ids)]


class _StochasticWidthContext:
    label_schema = LabelSchema(task="text_classification", labels=["negative", "neutral", "positive"])

    def predict_stochastic(self, sample_ids: Sequence[str], n: int = 10, batch_size: int = 32) -> list[list[list[float]]]:
        return [[[0.4, 0.6] for _ in range(n)] for _ in sample_ids]

    def predict_committee(self, sample_ids: Sequence[str], batch_size: int = 32) -> list[list[list[float]]]:
        return [[[0.4, 0.6], [0.7, 0.3]] for _ in sample_ids]


def test_context_embed_without_cache_rejects_invalid_embedding_rows() -> None:
    context = SelectionContext(
        provider=_Provider(),
        model=_EmbeddingModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        prediction_cache=None,
        embedding_cache=None,
        labeled_ids=[],
        last_metrics={},
    )

    with pytest.raises(ConfigurationError, match="finite"):
        context.embed(["s1"])


def test_hybrid_strategy_rejects_probability_width_mismatching_label_schema() -> None:
    scheduler = StrategyScheduler(
        SchedulerConfig(
            mode="hybrid",
            hybrid={"mode": "weighted", "uncertainty_weight": 1.0, "diversity_weight": 0.0},
        )
    )

    with pytest.raises(ConfigurationError, match="label_schema"):
        scheduler.select_batch(["s1", "s2", "s3"], 2, _HybridWidthContext(), state={})


def test_stochastic_strategy_rejects_probability_width_mismatching_label_schema() -> None:
    with pytest.raises(ConfigurationError, match="label_schema width"):
        McDropoutEntropyStrategy().select(["s1", "s2"], 1, _StochasticWidthContext())


def test_committee_strategy_rejects_probability_width_mismatching_label_schema() -> None:
    with pytest.raises(ConfigurationError, match="label_schema width"):
        CommitteeVoteEntropyStrategy().select(["s1", "s2"], 1, _StochasticWidthContext())


def test_multilabel_projects_reject_builtin_softmax_acquisition(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        [
            {"sample_id": "s1", "text": "one"},
            {"sample_id": "s2", "text": "two"},
        ]
    )
    project = ActiveLearningProject("multi-label-acquisition-audit", tmp_path, lock=False)

    with pytest.raises(ConfigurationError, match="multi_label=True.*single-label softmax"):
        project.configure(
            dataset=dataset,
            model=_EmbeddingModel(),
            label_schema=LabelSchema(task="text_classification", labels=["news", "sports"], multi_label=True),
            label_backend_config=LabelBackendConfig(backend="simulator"),
            label_backend=SimulatorLabelBackend(),
            scheduler_config=SchedulerConfig(strategy="entropy"),
        )


def test_round_failure_is_persisted_with_failed_status_and_error(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        [
            {"sample_id": "s1", "text": "one"},
            {"sample_id": "s2", "text": "two"},
        ]
    )
    project = ActiveLearningProject("failed-round-audit", tmp_path, lock=False)
    project.configure(
        dataset=dataset,
        model=_FailingPredictModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="simulator"),
        label_backend=SimulatorLabelBackend(),
        scheduler_config=SchedulerConfig(strategy="entropy"),
    )

    with pytest.raises(Exception, match="synthetic predict failure"):
        project.run_step(batch_size=1)

    failed_round = project.get_state().rounds[-1]
    assert failed_round.status == RoundStatus.FAILED
    assert "synthetic predict failure" in (failed_round.error or "")


def test_stale_prediction_cache_width_mismatch_is_discarded_before_model_call() -> None:
    model = _ThreeColumnModel()
    prediction_cache = PredictionCache(InMemoryCacheStore())
    prediction_cache.set("stable-model-id", "s1", [0.5, 0.5])
    context = SelectionContext(
        provider=_Provider(),
        model=model,
        label_schema=LabelSchema(task="text_classification", labels=["negative", "neutral", "positive"]),
        prediction_cache=prediction_cache,
        embedding_cache=None,
        labeled_ids=[],
        last_metrics={},
    )

    assert context.predict_proba(["s1"]) == [[0.2, 0.3, 0.5]]
    assert model.calls == [["one"]]


def test_attach_runtime_rejects_group_id_drift_for_group_aware_strategy(tmp_path: Path) -> None:
    original = pd.DataFrame(
        [
            {"sample_id": "s1", "text": "same one", "group_id": "group-a"},
            {"sample_id": "s2", "text": "same two", "group_id": "group-b"},
            {"sample_id": "s3", "text": "same three", "group_id": "group-c"},
        ]
    )
    drifted = pd.DataFrame(
        [
            {"sample_id": "s1", "text": "same one", "group_id": "group-z"},
            {"sample_id": "s2", "text": "same two", "group_id": "group-z"},
            {"sample_id": "s3", "text": "same three", "group_id": "group-z"},
        ]
    )

    project = ActiveLearningProject("group-drift-audit", tmp_path, lock=False)
    project.configure(
        dataset=original,
        model=_EmbeddingModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="simulator"),
        label_backend=SimulatorLabelBackend(),
        scheduler_config=SchedulerConfig(mode="single", strategy="group_diverse_entropy"),
        cache_config=CacheConfig(enable=False),
        fingerprint_config=FingerprintConfig(mode="strict"),
        split_config=SplitConfig(
            mode="explicit",
            explicit_splits={"train": ["s1", "s2", "s3"], "val": [], "test": []},
        ),
    )
    project.close()

    reopened = ActiveLearningProject("group-drift-audit", tmp_path, lock=False)
    with pytest.raises((ConfigurationError, DatasetMismatchError, StateCorruptedError)):
        reopened.attach_runtime(
            dataset=drifted,
            model=_EmbeddingModel(),
            label_backend=SimulatorLabelBackend(),
        )


def test_attach_runtime_rejects_group_id_drift_for_adaptive_group_aware_strategy(tmp_path: Path) -> None:
    original = pd.DataFrame(
        [
            {"sample_id": "s1", "text": "same one", "group_id": "group-a"},
            {"sample_id": "s2", "text": "same two", "group_id": "group-b"},
            {"sample_id": "s3", "text": "same three", "group_id": "group-c"},
        ]
    )
    drifted = pd.DataFrame(
        [
            {"sample_id": "s1", "text": "same one", "group_id": "group-z"},
            {"sample_id": "s2", "text": "same two", "group_id": "group-z"},
            {"sample_id": "s3", "text": "same three", "group_id": "group-z"},
        ]
    )

    project = ActiveLearningProject("adaptive-group-drift-audit", tmp_path, lock=False)
    project.configure(
        dataset=original,
        model=_EmbeddingModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="simulator"),
        label_backend=SimulatorLabelBackend(),
        scheduler_config=SchedulerConfig(mode="single", strategy="adaptive_uncertainty_diversity"),
        cache_config=CacheConfig(enable=False),
        fingerprint_config=FingerprintConfig(mode="strict"),
        split_config=SplitConfig(
            mode="explicit",
            explicit_splits={"train": ["s1", "s2", "s3"], "val": [], "test": []},
        ),
    )
    project.close()

    reopened = ActiveLearningProject("adaptive-group-drift-audit", tmp_path, lock=False)
    with pytest.raises((ConfigurationError, DatasetMismatchError, StateCorruptedError)):
        reopened.attach_runtime(
            dataset=drifted,
            model=_EmbeddingModel(),
            label_backend=SimulatorLabelBackend(),
        )


def test_reconfigure_after_active_round_is_rejected(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        [
            {"sample_id": "s1", "text": "one"},
            {"sample_id": "s2", "text": "two"},
        ]
    )
    project = ActiveLearningProject("reconfigure-audit", tmp_path, lock=False)
    project.configure(
        dataset=dataset,
        model=_EmbeddingModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="simulator"),
        label_backend=SimulatorLabelBackend(),
        scheduler_config=SchedulerConfig(mode="single", strategy="random"),
        cache_config=CacheConfig(enable=False),
        split_config=SplitConfig(
            mode="explicit",
            explicit_splits={"train": ["s1", "s2"], "val": [], "test": []},
        ),
    )
    project.run_step(batch_size=1)

    with pytest.raises(ConfigurationError, match="reconfigure|round"):
        project.configure(
            dataset=dataset,
            model=_EmbeddingModel(),
            label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
            label_backend_config=LabelBackendConfig(backend="simulator"),
            label_backend=SimulatorLabelBackend(),
            scheduler_config=SchedulerConfig(mode="single", strategy="entropy"),
            cache_config=CacheConfig(enable=False),
            split_config=SplitConfig(
                mode="explicit",
                explicit_splits={"train": ["s2"], "val": ["s1"], "test": []},
            ),
        )


def test_custom_selector_reconfigure_after_active_round_is_rejected_even_when_persisted_payload_matches(
    tmp_path: Path,
) -> None:
    dataset = pd.DataFrame(
        [
            {"sample_id": "s1", "text": "one"},
            {"sample_id": "s2", "text": "two"},
        ]
    )

    def first_selector(context: Any, k: int, pool_ids: Sequence[str]) -> list[str]:
        del context, k
        return [pool_ids[0]]

    def second_selector(context: Any, k: int, pool_ids: Sequence[str]) -> list[str]:
        del context, k
        return [pool_ids[-1]]

    project = ActiveLearningProject("custom-reconfigure-audit", tmp_path, lock=False)
    project.configure(
        dataset=dataset,
        model=_EmbeddingModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="simulator"),
        label_backend=SimulatorLabelBackend(),
        scheduler_config=SchedulerConfig(mode="custom", custom_selector=first_selector),
        cache_config=CacheConfig(enable=False),
        split_config=SplitConfig(
            mode="explicit",
            explicit_splits={"train": ["s1", "s2"], "val": [], "test": []},
        ),
    )
    project.run_step(batch_size=1)

    with pytest.raises(ConfigurationError, match="custom scheduler|reconfigure"):
        project.configure(
            dataset=dataset,
            model=_EmbeddingModel(),
            label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
            label_backend_config=LabelBackendConfig(backend="simulator"),
            label_backend=SimulatorLabelBackend(),
            scheduler_config=SchedulerConfig(mode="custom", custom_selector=second_selector),
            cache_config=CacheConfig(enable=False),
            split_config=SplitConfig(
                mode="explicit",
                explicit_splits={"train": ["s1", "s2"], "val": [], "test": []},
            ),
        )
