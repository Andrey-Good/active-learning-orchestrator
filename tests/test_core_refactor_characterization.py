from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

from active_learning_sdk import (
    ActiveLearningProject,
    CacheConfig,
    LabelBackendConfig,
    LabelSchema,
    SchedulerConfig,
    SplitConfig,
)
from active_learning_sdk.backends.base import RoundPullResult, RoundPushResult
from active_learning_sdk.cache import EmbeddingCache, InMemoryCacheStore, PredictionCache
from active_learning_sdk.engine import SelectionContext, StrategyScheduler
from active_learning_sdk.exceptions import ConfigurationError
from active_learning_sdk.types import DataSample


class TinyDataset:
    def __init__(self, rows: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        self._rows = dict(
            rows
            or {
                "s1": {"text": "sample one", "split": "train"},
                "s2": {"text": "sample two", "split": "val"},
                "s3": {"text": "sample three", "split": "test"},
            }
        )

    def iter_sample_ids(self):
        yield from self._rows.keys()

    def get_sample(self, sample_id: str) -> DataSample:
        row = self._rows[sample_id]
        data = {"text": str(row["text"])}
        if "split" in row:
            data["split"] = row["split"]
        return DataSample(sample_id=sample_id, data=data)

    def get_samples(self, sample_ids: Sequence[str]) -> list[DataSample]:
        return [self.get_sample(sample_id) for sample_id in sample_ids]

    def schema(self) -> dict[str, str]:
        return {"sample_id": "str", "text": "str"}


class RecordingModel:
    def __init__(self, *, model_id: str = "recording-model", embedding_config: Any = "emb-v1") -> None:
        self._model_id = model_id
        self.embedding_config = embedding_config
        self.predict_calls: list[list[str]] = []
        self.embed_calls: list[list[str]] = []

    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        self.predict_calls.append(list(texts))
        return [[0.75, 0.25] for _ in texts]

    def embed(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        self.embed_calls.append(list(texts))
        return [[float(len(text)), float(index + 1)] for index, text in enumerate(texts)]

    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs: Any) -> None:
        return None

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> dict[str, float]:
        return {"accuracy": 1.0}

    def get_model_id(self) -> str:
        return self._model_id


class ColdStartDiagnosticsModel:
    def __init__(
        self,
        *,
        probabilities: Mapping[str, list[float]],
        embeddings: Mapping[str, list[float]] | None = None,
        model_id: str = "cold-start-diagnostics-model",
    ) -> None:
        self._probabilities = dict(probabilities)
        self._embeddings = dict(embeddings or {})
        self._model_id = model_id

    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        return [self._probabilities[text] for text in texts]

    def embed(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        return [self._embeddings[text] for text in texts]

    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs: Any) -> None:
        return None

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> dict[str, float]:
        return {"accuracy": 1.0}

    def get_model_id(self) -> str:
        return self._model_id


class NoopBackend:
    def ensure_ready(self, label_schema: LabelSchema) -> dict[str, Any]:
        label_schema.validate()
        return {"backend": "noop"}

    def push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: dict[str, Any] | None = None,
    ) -> RoundPushResult:
        del prelabels
        return RoundPushResult(task_ids={sample.sample_id: f"{round_id}:{sample.sample_id}" for sample in samples})

    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        del round_id
        return RoundPullResult(annotations={sample_id: [] for sample_id in task_ids})

    def close(self) -> None:
        return None


class ReturningStrategy:
    def __init__(self, returned: Sequence[str]) -> None:
        self.name = "returning"
        self._returned = list(returned)

    def select(self, pool_ids: Sequence[str], k: int, context: object) -> list[str]:
        del pool_ids, k, context
        return list(self._returned)


class DiagnosticStrategy:
    name = "diagnostic"
    required_capabilities = frozenset()

    def __init__(self, diagnostic: Mapping[str, Any]) -> None:
        self._diagnostic = dict(diagnostic)

    def select(self, pool_ids: Sequence[str], k: int, context: object) -> list[str]:
        recorder = getattr(context, "record_strategy_diagnostic", None)
        if callable(recorder):
            recorder(self.name, self._diagnostic)
        return list(pool_ids)[:k]


def _label_schema() -> LabelSchema:
    return LabelSchema(task="text_classification", labels=["negative", "positive"])


def _configured_project(tmp_path: Path, *, split_config: SplitConfig | None = None) -> ActiveLearningProject:
    project = ActiveLearningProject("stage3a-characterization", tmp_path, lock=False)
    project.configure(
        dataset=TinyDataset(),
        model=RecordingModel(),
        label_schema=_label_schema(),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=NoopBackend(),
        scheduler_config=SchedulerConfig(strategy="random"),
        split_config=split_config
        or SplitConfig(mode="explicit", explicit_splits={"train": ["s1"], "val": ["s2"], "test": ["s3"]}),
        cache_config=CacheConfig(enable=False),
    )
    return project


def _live_project_state(project: ActiveLearningProject):
    return project._engine.get_state()  # type: ignore[attr-defined]


def _selection_context(
    *,
    model: RecordingModel,
    prediction_cache: PredictionCache | None = None,
    embedding_cache: EmbeddingCache | None = None,
    dataset_fingerprint: str = "fp-a",
) -> SelectionContext:
    return SelectionContext(
        provider=TinyDataset(),
        model=model,
        label_schema=_label_schema(),
        prediction_cache=prediction_cache,
        embedding_cache=embedding_cache,
        labeled_ids=[],
        last_metrics={},
        dataset_fingerprint=dataset_fingerprint,
    )


def _mapped_text_context(
    *,
    model: ColdStartDiagnosticsModel,
    label_schema: LabelSchema,
    sample_ids: Sequence[str],
) -> SelectionContext:
    return SelectionContext(
        provider=TinyDataset({sample_id: {"text": sample_id, "split": "train"} for sample_id in sample_ids}),
        model=model,
        label_schema=label_schema,
        prediction_cache=None,
        embedding_cache=None,
        labeled_ids=[],
        last_metrics={},
        dataset_fingerprint="fp-diagnostics",
    )


@pytest.mark.parametrize(
    ("explicit_splits", "message"),
    [
        ({"train": ["s1"], "val": ["s2"], "test": []}, "does not cover every dataset sample_id"),
        ({"train": ["s1", "s2"], "val": ["s2"], "test": ["s3"]}, "Explicit split overlap detected"),
        ({"train": ["s1"], "val": ["missing"], "test": ["s2", "s3"]}, "Unknown split sample_id"),
    ],
)
def test_explicit_split_resolution_rejects_missing_overlap_and_unknown_ids(
    tmp_path: Path,
    explicit_splits: dict[str, list[str]],
    message: str,
) -> None:
    project = ActiveLearningProject("stage3a-explicit-splits", tmp_path, lock=False)

    with pytest.raises(ConfigurationError, match=message):
        project.configure(
            dataset=TinyDataset(),
            model=RecordingModel(),
            label_schema=_label_schema(),
            label_backend_config=LabelBackendConfig(backend="custom"),
            label_backend=NoopBackend(),
            scheduler_config=SchedulerConfig(strategy="random"),
            split_config=SplitConfig(mode="explicit", explicit_splits=explicit_splits),
            cache_config=CacheConfig(enable=False),
        )


def test_validate_reports_persisted_split_coverage_overlap_and_unknown_ids(tmp_path: Path) -> None:
    project = _configured_project(tmp_path)
    state = _live_project_state(project)
    state.splits = {"train": ["s1", "s2"], "val": ["s2", "unknown"], "test": []}

    report = project.validate()

    assert report["ok"] is False
    assert any("Persisted split overlap" in issue and "'s2'" in issue for issue in report["issues"])
    assert any("contains unknown sample ids" in issue and "unknown" in issue for issue in report["issues"])
    assert any("Persisted split coverage is invalid" in issue and "missing ids: ['s3']" in issue for issue in report["issues"])


def test_column_split_assignments_are_stable_across_reconfigure_order_changes(tmp_path: Path) -> None:
    project = _configured_project(tmp_path, split_config=SplitConfig(mode="column", split_column="split"))

    project.configure(
        dataset=TinyDataset(
            {
                "s3": {"text": "sample three", "split": "test"},
                "s1": {"text": "sample one", "split": "train"},
                "s2": {"text": "sample two", "split": "val"},
            }
        ),
        model=RecordingModel(),
        label_schema=_label_schema(),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=NoopBackend(),
        scheduler_config=SchedulerConfig(strategy="random"),
        split_config=SplitConfig(mode="column", split_column="split"),
        cache_config=CacheConfig(enable=False),
    )

    assert project.get_state().splits == {"train": ["s1"], "val": ["s2"], "test": ["s3"]}


def test_prediction_cache_is_scoped_and_invalid_cached_rows_are_recomputed() -> None:
    cache = PredictionCache(InMemoryCacheStore())
    model = RecordingModel()
    cache.set("recording-model", "s1", [1.0], dataset_fingerprint="fp-a")
    cache.set("recording-model", "s1", [0.6, 0.4], dataset_fingerprint="fp-other")

    fp_a_context = _selection_context(model=model, prediction_cache=cache, dataset_fingerprint="fp-a")
    assert fp_a_context.predict_proba(["s1"]) == [[0.75, 0.25]]

    fp_b_context = _selection_context(model=model, prediction_cache=cache, dataset_fingerprint="fp-b")
    assert fp_b_context.predict_proba(["s1"]) == [[0.75, 0.25]]

    assert model.predict_calls == [["sample one"], ["sample one"]]
    assert cache.get("recording-model", "s1", dataset_fingerprint="fp-a") == [0.75, 0.25]
    assert cache.get("recording-model", "s1", dataset_fingerprint="fp-other") == [0.6, 0.4]


def test_prediction_context_revalidates_and_evicts_invalid_last_result_cache() -> None:
    cache = PredictionCache(InMemoryCacheStore())
    model = RecordingModel()
    context = _selection_context(model=model, prediction_cache=cache, dataset_fingerprint="fp-a")

    assert context.predict_proba(["s1"]) == [[0.75, 0.25]]
    context._last_predict_proba_rows = [[0.5]]

    assert context.predict_proba(["s1"]) == [[0.75, 0.25]]
    assert model.predict_calls == [["sample one"]]


def test_embedding_cache_is_scoped_and_invalid_cached_rows_are_recomputed() -> None:
    cache = EmbeddingCache(InMemoryCacheStore())
    model = RecordingModel(embedding_config={"name": "emb-v1"})
    cache.set(
        "recording-model",
        "s1",
        ["invalid"],
        dataset_fingerprint="fp-a",
        embedding_config={"name": "emb-v1"},
    )
    cache.set(
        "recording-model",
        "s1",
        [9.0, 9.0],
        dataset_fingerprint="fp-a",
        embedding_config={"name": "emb-other"},
    )

    context = _selection_context(model=model, embedding_cache=cache, dataset_fingerprint="fp-a")

    assert context.embed(["s1"]) == [[10.0, 1.0]]
    assert model.embed_calls == [["sample one"]]
    assert cache.get(
        "recording-model",
        "s1",
        dataset_fingerprint="fp-a",
        embedding_config={"name": "emb-v1"},
    ) == [10.0, 1.0]
    assert cache.get(
        "recording-model",
        "s1",
        dataset_fingerprint="fp-a",
        embedding_config={"name": "emb-other"},
    ) == [9.0, 9.0]


def test_scheduler_deduplicates_custom_strategy_selection_without_refill() -> None:
    scheduler = StrategyScheduler(
        SchedulerConfig(mode="single", strategy="returning"),
        strategies=[ReturningStrategy(["s1", "s1", "s2", "s2"])],
    )

    selected, snapshot = scheduler.select_batch(["s1", "s2", "s3"], 3, object(), state={})

    assert selected == ["s1", "s2"]
    assert snapshot == {"mode": "single", "strategy": "returning"}


def test_single_strategy_snapshot_records_entropy_cold_start_fallback_diagnostics() -> None:
    sample_ids = ["known_1", "known_2", "novel_far"]
    context = _mapped_text_context(
        sample_ids=sample_ids,
        label_schema=LabelSchema(task="text_classification", labels=["a", "b", "c", "d", "e"]),
        model=ColdStartDiagnosticsModel(
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
        ),
    )
    scheduler = StrategyScheduler(SchedulerConfig(mode="single", strategy="entropy"))

    selected, snapshot = scheduler.select_batch(sample_ids, 2, context, state={})

    assert selected == ["novel_far", "known_1"]
    assert snapshot == {
        "mode": "single",
        "strategy": "entropy",
        "strategy_diagnostics": [
            {
                "strategy": "entropy",
                "effective_strategy": "cold_start_blend:coreset_kcenter+entropy",
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


def test_single_strategy_snapshot_remains_minimal_without_fallback() -> None:
    sample_ids = ["confident", "middle", "uncertain"]
    context = _mapped_text_context(
        sample_ids=sample_ids,
        label_schema=_label_schema(),
        model=ColdStartDiagnosticsModel(
            probabilities={
                "confident": [0.99, 0.01],
                "middle": [0.7, 0.3],
                "uncertain": [0.5, 0.5],
            },
            embeddings={
                "confident": [10.0, 0.0],
                "middle": [5.0, 0.0],
                "uncertain": [0.0, 0.0],
            },
        ),
    )
    scheduler = StrategyScheduler(SchedulerConfig(mode="single", strategy="entropy"))

    selected, snapshot = scheduler.select_batch(sample_ids, 2, context, state={})

    assert selected == ["uncertain", "middle"]
    assert snapshot == {"mode": "single", "strategy": "entropy"}


def test_strategy_diagnostics_are_json_safe_and_cannot_overwrite_single_snapshot_fields() -> None:
    context = _selection_context(model=RecordingModel())
    scheduler = StrategyScheduler(
        SchedulerConfig(mode="single", strategy="diagnostic"),
        strategies=[
            DiagnosticStrategy(
                {
                    "mode": "corrupted",
                    "strategy": "corrupted",
                    "non_finite_real": np.float32("nan"),
                    "infinite_real": np.float64("inf"),
                }
            )
        ],
    )

    selected, snapshot = scheduler.select_batch(["s1"], 1, context, state={})

    assert selected == ["s1"]
    assert snapshot["mode"] == "single"
    assert snapshot["strategy"] == "diagnostic"
    assert snapshot["strategy_diagnostics"] == [
        {
            "mode": "corrupted",
            "strategy": "diagnostic",
            "non_finite_real": None,
            "infinite_real": None,
        }
    ]
    json.dumps(snapshot, allow_nan=False)


def test_mix_strategy_snapshot_records_and_consumes_fallback_diagnostics() -> None:
    sample_ids = ["known_1", "known_2", "novel_far"]
    context = _mapped_text_context(
        sample_ids=sample_ids,
        label_schema=LabelSchema(task="text_classification", labels=["a", "b", "c", "d", "e"]),
        model=ColdStartDiagnosticsModel(
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
        ),
    )
    context.record_strategy_diagnostic("entropy", {"fallback_reason": "stale"})
    scheduler = StrategyScheduler(SchedulerConfig(mode="mix", mix={"entropy": 1.0}))

    selected, snapshot = scheduler.select_batch(sample_ids, 2, context, state={})

    assert selected == ["novel_far", "known_1"]
    assert snapshot["mode"] == "mix"
    assert snapshot["strategy_diagnostics"] == [
        {
            "strategy": "entropy",
            "effective_strategy": "cold_start_blend:coreset_kcenter+entropy",
            "fallback_reason": "cold_start_sparse_probability_support",
            "label_count": 5,
            "support_count": 2,
            "support_fraction": 0.4,
            "missing_label_count": 3,
            "fallback_mode": "blend",
            "exploration_count": 1,
            "exploitation_count": 1,
        }
    ]
    assert context.consume_strategy_diagnostics() == []


def test_bandit_strategy_snapshot_records_and_consumes_fallback_diagnostics() -> None:
    sample_ids = ["known_1", "known_2", "novel_far"]
    context = _mapped_text_context(
        sample_ids=sample_ids,
        label_schema=LabelSchema(task="text_classification", labels=["a", "b", "c", "d", "e"]),
        model=ColdStartDiagnosticsModel(
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
        ),
    )
    scheduler = StrategyScheduler(SchedulerConfig(mode="bandit", bandit_arms=["entropy"]))

    selected, snapshot = scheduler.select_batch(sample_ids, 2, context, state={})

    assert selected == ["novel_far", "known_1"]
    assert snapshot["mode"] == "bandit"
    assert snapshot["chosen_arm"] == "entropy"
    assert snapshot["strategy_diagnostics"] == [
        {
            "strategy": "entropy",
            "effective_strategy": "cold_start_blend:coreset_kcenter+entropy",
            "fallback_reason": "cold_start_sparse_probability_support",
            "label_count": 5,
            "support_count": 2,
            "support_fraction": 0.4,
            "missing_label_count": 3,
            "fallback_mode": "blend",
            "exploration_count": 1,
            "exploitation_count": 1,
        }
    ]
    assert context.consume_strategy_diagnostics() == []


def test_scheduler_rejects_custom_strategy_selection_outside_pool() -> None:
    scheduler = StrategyScheduler(
        SchedulerConfig(mode="single", strategy="returning"),
        strategies=[ReturningStrategy(["s1", "outside"])],
    )

    with pytest.raises(ConfigurationError, match="outside the candidate pool: 'outside'"):
        scheduler.select_batch(["s1", "s2"], 2, object(), state={})


def test_validate_returns_structured_ok_report_for_healthy_configured_project(tmp_path: Path) -> None:
    project = _configured_project(tmp_path)

    report = project.validate()

    assert report == {"ok": True, "issues": []}
