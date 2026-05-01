from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from active_learning_sdk import (
    ActiveLearningProject,
    CacheConfig,
    DataSample,
    LabelBackendConfig,
    LabelSchema,
    SchedulerConfig,
    SplitConfig,
    StopCriteria,
)
from active_learning_sdk.backends.base import RoundProgress, RoundPullResult, RoundPushResult
from active_learning_sdk.cache import EmbeddingCache, InMemoryCacheStore, JsonlDiskCacheStore, PredictionCache
from active_learning_sdk.types import AnnotationRecord


class CacheStatsProvider:
    def __init__(self, count: int = 40) -> None:
        self.ids = [f"s{i:03d}" for i in range(count)]

    def iter_sample_ids(self):
        yield from self.ids

    def get_sample(self, sample_id: str) -> DataSample:
        index = int(sample_id[1:])
        return DataSample(sample_id=sample_id, data={"text": f"text {index} topic {index % 7}"})

    def get_samples(self, sample_ids: Sequence[str]) -> list[DataSample]:
        return [self.get_sample(sample_id) for sample_id in sample_ids]

    def schema(self) -> dict[str, str]:
        return {"sample_id": "str", "text": "str"}


class StableIdProbabilityModel:
    def __init__(self) -> None:
        self.predicted_rows = 0
        self.fit_calls = 0

    def get_model_id(self) -> str:
        return "stable-but-not-versioned"

    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        self.predicted_rows += len(texts)
        rows: list[list[float]] = []
        for text in texts:
            p_pos = 0.2 + ((sum(ord(char) for char in text) % 60) / 100.0)
            rows.append([1.0 - p_pos, p_pos])
        return rows

    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs: Any) -> None:
        del texts, labels, kwargs
        self.fit_calls += 1

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> dict[str, float]:
        del texts, labels
        return {"accuracy": 1.0}

    def get_embedding_config(self) -> str:
        return "stable-test-embeddings"

    def embed(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        return [[float(len(text)), float(sum(ord(char) for char in text) % 17)] for text in texts]


class ImmediateBackend:
    def ensure_ready(self, label_schema: LabelSchema) -> dict[str, Any]:
        del label_schema
        return {"backend": "immediate"}

    def push_round(
        self,
        round_id: str,
        samples: Sequence[DataSample],
        prelabels: dict[str, Any] | None = None,
    ) -> RoundPushResult:
        del prelabels
        return RoundPushResult(task_ids={sample.sample_id: f"{round_id}:{sample.sample_id}" for sample in samples})

    def poll_round(self, round_id: str, task_ids: Mapping[str, str], policy: Any) -> RoundProgress:
        del round_id, policy
        return RoundProgress(total=len(task_ids), done=len(task_ids), ready_sample_ids=list(task_ids))

    def pull_round(self, round_id: str, task_ids: Mapping[str, str]) -> RoundPullResult:
        del round_id
        now = time.time()
        return RoundPullResult(
            annotations={
                sample_id: [
                    AnnotationRecord(
                        annotator_id="backend",
                        created_at=now,
                        value="positive" if index % 2 else "negative",
                    )
                ]
                for index, sample_id in enumerate(task_ids)
            }
        )


def _configured_project(workdir: Path, provider: CacheStatsProvider, model: StableIdProbabilityModel) -> ActiveLearningProject:
    project = ActiveLearningProject("cache-stats-semantics", workdir, lock=False)
    project.configure(
        dataset=provider,
        model=model,
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=ImmediateBackend(),
        scheduler_config=SchedulerConfig(strategy="entropy"),
        cache_config=CacheConfig(enable=True, persist=True),
        split_config=SplitConfig(mode="explicit", explicit_splits={"train": provider.ids, "val": [], "test": []}),
    )
    return project


def _seed(project: ActiveLearningProject, provider: CacheStatsProvider, count: int = 6) -> None:
    project.import_labels(
        {sample_id: ("positive" if index % 2 else "negative") for index, sample_id in enumerate(provider.ids[:count])}
    )


def test_jsonl_disk_cache_lifetime_stats_survive_reopen_after_clear(tmp_path: Path) -> None:
    store = JsonlDiskCacheStore(tmp_path / "cache", "predictions")

    store.set("k1", {"value": 1})
    store.set("k2", {"value": 2})
    store.clear(reason="automatic_model_id_unstable", kind="predictions")

    reopened = JsonlDiskCacheStore(tmp_path / "cache", "predictions")
    stats = reopened.stats()

    assert stats["stored_items"] == 0
    assert stats["items"] == 0
    assert stats["lifetime_writes"] == 2
    assert stats["lifetime_clears"] == 1
    assert stats["lifetime_invalidations"] >= 1
    assert stats["session_writes"] == 0
    assert stats["last_clear_reason"] == "automatic_model_id_unstable"
    assert stats["last_clear_kind"] == "predictions"
    assert stats["last_cleared_items"] == 2
    assert stats["metadata_bytes"] > 0


def test_engine_cache_stats_reports_epoch_invalidation_without_erasing_entries(tmp_path: Path) -> None:
    provider = CacheStatsProvider(50)
    model = StableIdProbabilityModel()
    project = _configured_project(tmp_path, provider, model)
    _seed(project, provider)

    project.run(batch_size=10, stop_criteria=StopCriteria(max_rounds=1), poll_interval_seconds=0)

    stats = project.cache_stats()["prediction_cache"]
    assert model.predicted_rows > 0
    assert stats["stored_items"] == model.predicted_rows
    assert stats["items"] == model.predicted_rows
    assert stats["current_reusable_items"] == 0
    assert stats["lifetime_writes"] == model.predicted_rows
    assert stats["session_writes"] == model.predicted_rows
    assert stats["lifetime_invalidations"] >= 1
    assert stats["last_invalidation_reason"] == "automatic_model_id_epoch_advanced"
    assert stats["last_invalidation_kind"] == "all"
    assert stats["data_bytes"] > 0
    assert stats["index_bytes"] > 0


def test_reopened_project_cache_stats_preserve_lifetime_but_reset_session_counters(tmp_path: Path) -> None:
    provider = CacheStatsProvider(45)
    model = StableIdProbabilityModel()
    project = _configured_project(tmp_path, provider, model)
    _seed(project, provider)
    project.run(batch_size=8, stop_criteria=StopCriteria(max_rounds=1), poll_interval_seconds=0)
    before_close = project.cache_stats()["prediction_cache"]
    project.close()

    reopened = ActiveLearningProject("cache-stats-semantics", tmp_path, lock=False)
    reopened.attach_runtime(dataset=provider, model=StableIdProbabilityModel(), label_backend=ImmediateBackend())
    after_reopen = reopened.cache_stats()["prediction_cache"]

    assert after_reopen["stored_items"] == before_close["stored_items"]
    assert after_reopen["lifetime_writes"] == before_close["lifetime_writes"]
    assert after_reopen["session_writes"] == 0
    assert after_reopen["current_reusable_items"] == 0


def test_manual_clear_records_reason_without_erasing_lifetime_stats(tmp_path: Path) -> None:
    provider = CacheStatsProvider(35)
    model = StableIdProbabilityModel()
    project = _configured_project(tmp_path, provider, model)
    _seed(project, provider)
    project.run(batch_size=5, stop_criteria=StopCriteria(max_rounds=1), poll_interval_seconds=0)

    project.clear_cache(kind="predictions")

    stats = project.cache_stats()["prediction_cache"]
    assert stats["stored_items"] == 0
    assert stats["lifetime_writes"] == model.predicted_rows
    assert stats["lifetime_clears"] >= 1
    assert stats["last_clear_reason"] == "manual"
    assert stats["last_clear_kind"] == "predictions"


def test_in_memory_cache_items_report_stored_entries_even_when_scope_is_not_reusable() -> None:
    cache = PredictionCache(InMemoryCacheStore())
    cache.set("old-model", "s1", [0.4, 0.6], dataset_fingerprint="dataset")

    stats = cache.stats(model_id="new-model", dataset_fingerprint="dataset")

    assert stats["items"] == 1
    assert stats["stored_items"] == 1
    assert stats["current_reusable_items"] == 0


def test_engine_embedding_cache_stats_use_current_embedding_config(tmp_path: Path) -> None:
    provider = CacheStatsProvider(30)
    model = StableIdProbabilityModel()
    project = ActiveLearningProject("embedding-cache-stats", tmp_path, lock=False)
    project.configure(
        dataset=provider,
        model=model,
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        label_backend_config=LabelBackendConfig(backend="custom"),
        label_backend=ImmediateBackend(),
        scheduler_config=SchedulerConfig(strategy="coreset_kcenter"),
        cache_config=CacheConfig(enable=True, persist=True),
        split_config=SplitConfig(mode="explicit", explicit_splits={"train": provider.ids, "val": [], "test": []}),
    )
    _seed(project, provider)

    project.run(batch_size=6, stop_criteria=StopCriteria(max_rounds=1), poll_interval_seconds=0)

    stats = project.cache_stats()["embedding_cache"]
    assert stats["stored_items"] > 0
    assert stats["current_reusable_items"] == 0
    assert stats["lifetime_invalidations"] >= 1
    assert stats["last_invalidation_reason"] == "automatic_model_id_epoch_advanced"


def test_cache_wrappers_tolerate_legacy_custom_store_stats_and_clear_signatures() -> None:
    class LegacyStore:
        def __init__(self) -> None:
            self.cleared = False

        def get(self, key: str) -> Any | None:
            del key
            return None

        def set(self, key: str, value: Any) -> None:
            del key, value

        def delete(self, key: str) -> None:
            del key

        def stats(self) -> dict[str, Any]:
            return {"items": 0}

        def clear(self) -> None:
            self.cleared = True

    store = LegacyStore()
    prediction_cache = PredictionCache(store)  # type: ignore[arg-type]
    embedding_cache = EmbeddingCache(store)  # type: ignore[arg-type]

    assert prediction_cache.stats(model_id="m")["current_reusable_items"] == 0
    assert embedding_cache.stats(model_id="m", embedding_config="e")["current_reusable_items"] == 0
    prediction_cache.clear(reason="automatic_model_id_unstable", kind="predictions")
    assert store.cleared is True
