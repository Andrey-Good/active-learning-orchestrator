from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import pytest

from active_learning_sdk import (
    ActiveLearningProject,
    CacheConfig,
    ConfigurationError,
    FingerprintConfig,
    SchedulerConfig,
    SplitConfig,
)
from active_learning_sdk.backends.base import build_label_backend
from active_learning_sdk.backends.label_studio import LabelStudioBackend
from active_learning_sdk.backends.simulator import SimulatorLabelBackend
from active_learning_sdk.cache import InMemoryCacheStore, JsonlDiskCacheStore, PredictionCache
from active_learning_sdk.configs import LabelBackendConfig, LabelSchema
from active_learning_sdk.dataset.fingerprint import DatasetFingerprinter
from active_learning_sdk.dataset.provider import DataFrameDatasetProvider
from active_learning_sdk.exceptions import LabelBackendError
from active_learning_sdk.engine import SelectionContext
from active_learning_sdk.state.store import ProjectState
from active_learning_sdk.types import DataSample

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.quality_gate_report import build_quality_gate_report


def test_quality_gate_report_accepts_reference_benchmark_string_metadata(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    metrics_path.write_text(
        "\n".join(
            [
                "dataset,strategy,strategy_family,seed,budget,macro_f1,runtime_seconds",
                "toy,random,sdk,13,12,0.40,0.01",
                "toy,entropy,sdk,13,12,0.45,0.02",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_quality_gate_report(tmp_path)

    assert report["row_count"] == 2
    assert report["quality_gates"]["checks"]


def _configured_label_studio_backend() -> LabelStudioBackend:
    backend = LabelStudioBackend(
        LabelBackendConfig(
            backend="label_studio",
            mode="external",
            url="http://localhost:8080",
            api_token="not-used-by-this-unit-test",
        )
    )
    backend._label_schema = LabelSchema(task="text_classification", labels=["negative", "positive"])
    return backend


def test_label_studio_mapping_prelabels_reject_bool_and_negative_scores() -> None:
    backend = _configured_label_studio_backend()

    with pytest.raises(LabelBackendError):
        backend._build_prediction({"negative": True, "positive": False})

    with pytest.raises(LabelBackendError):
        backend._build_prediction({"negative": -0.2, "positive": -0.8})


class _ListResponseLabelStudioClient:
    def request(self, method: str, path: str, **kwargs: Any) -> list[dict[str, Any]]:
        del method, path, kwargs
        return [{"id": 1, "title": "Existing Project"}]


def test_label_studio_iter_projects_accepts_bare_list_responses() -> None:
    backend = _configured_label_studio_backend()
    backend._http_client = _ListResponseLabelStudioClient()  # type: ignore[assignment]

    assert list(backend._iter_projects()) == [{"id": 1, "title": "Existing Project"}]


class _MemoryStateStore:
    def __init__(self) -> None:
        self.load_calls = 0
        self.saved = ProjectState(
            state_version=1,
            project_name="stored-project",
            created_at=1.0,
            updated_at=1.0,
        )

    def load(self) -> ProjectState:
        self.load_calls += 1
        return self.saved

    def save_atomic(self, state: ProjectState) -> None:
        self.saved = state


def test_custom_state_store_loads_without_default_state_json(tmp_path: Path) -> None:
    store = _MemoryStateStore()
    project = ActiveLearningProject("requested-project", tmp_path, state_store=store, lock=False)

    assert store.load_calls == 1
    assert project.get_state().project_name == "stored-project"


class _TextProvider:
    def __init__(self, text_by_id: dict[str, str]) -> None:
        self._text_by_id = text_by_id

    def get_texts(self, sample_ids: Sequence[str]) -> list[str]:
        return [self._text_by_id[sample_id] for sample_id in sample_ids]

    def get_sample(self, sample_id: str) -> DataSample:
        return DataSample(sample_id=sample_id, data={"text": self._text_by_id[sample_id]})

    def get_samples(self, sample_ids: Sequence[str]) -> list[DataSample]:
        return [self.get_sample(sample_id) for sample_id in sample_ids]

    def schema(self) -> dict[str, str]:
        return {"sample_id": "str", "text": "str"}


class _TextSensitiveModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def get_model_id(self) -> str:
        return "same-model-id"

    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        del batch_size
        text_list = list(texts)
        self.calls.append(text_list)
        return [[0.9, 0.1] if text == "old text" else [0.1, 0.9] for text in text_list]


def test_prediction_cache_is_isolated_by_dataset_fingerprint() -> None:
    cache = PredictionCache(InMemoryCacheStore())
    model = _TextSensitiveModel()
    label_schema = LabelSchema(task="text_classification", labels=["negative", "positive"])
    first_context = SelectionContext(
        provider=_TextProvider({"s1": "old text"}),
        model=model,
        label_schema=label_schema,
        prediction_cache=cache,
        embedding_cache=None,
        labeled_ids=[],
        last_metrics={},
        dataset_fingerprint="dataset-old",
    )
    second_context = SelectionContext(
        provider=_TextProvider({"s1": "new text"}),
        model=model,
        label_schema=label_schema,
        prediction_cache=cache,
        embedding_cache=None,
        labeled_ids=[],
        last_metrics={},
        dataset_fingerprint="dataset-new",
    )

    assert first_context.predict_proba(["s1"]) == [[0.9, 0.1]]
    assert second_context.predict_proba(["s1"]) == [[0.1, 0.9]]
    assert model.calls == [["old text"], ["new text"]]


def test_selection_context_rejects_samples_missing_text_payload() -> None:
    class MissingTextProvider:
        def get_sample(self, sample_id: str) -> DataSample:
            return DataSample(sample_id=sample_id, data={"body": "not text"})

        def schema(self) -> dict[str, str]:
            return {"sample_id": "str", "body": "str"}

    context = SelectionContext(
        provider=MissingTextProvider(),
        model=_TextSensitiveModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        prediction_cache=None,
        embedding_cache=None,
        labeled_ids=[],
        last_metrics={},
    )

    with pytest.raises(ConfigurationError, match=r"missing required data\['text'\]"):
        context.predict_proba(["s1"])


def test_selection_context_rejects_non_string_provider_text_rows() -> None:
    class BadTextProvider:
        def get_texts(self, sample_ids: Sequence[str]) -> list[object]:
            return [None for _ in sample_ids]

        def get_sample(self, sample_id: str) -> DataSample:
            return DataSample(sample_id=sample_id, data={"text": "unused"})

        def schema(self) -> dict[str, str]:
            return {"sample_id": "str", "text": "str"}

    context = SelectionContext(
        provider=BadTextProvider(),
        model=_TextSensitiveModel(),
        label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
        prediction_cache=None,
        embedding_cache=None,
        labeled_ids=[],
        last_metrics={},
    )

    with pytest.raises(ConfigurationError, match="must be a string"):
        context.predict_proba(["s1"])


def test_disk_cache_stats_preserve_write_and_clear_observability(tmp_path: Path) -> None:
    store = JsonlDiskCacheStore(tmp_path, "predictions")
    cache = PredictionCache(store)

    cache.set("model", "s1", [0.9, 0.1], dataset_fingerprint="dataset")
    cache.set("model", "s2", [0.1, 0.9], dataset_fingerprint="dataset")
    cache.clear()

    stats = cache.stats()
    assert stats["items"] == 0
    assert stats["writes"] == 2
    assert stats["clears"] == 1
    assert stats["data_bytes"] == 0
    assert stats["index_bytes"] == 0


def test_dataframe_strict_fingerprint_includes_payload_columns_seen_by_backends() -> None:
    pd = pytest.importorskip("pandas")
    df_first = pd.DataFrame(
        [
            {"sample_id": "s1", "text": "same text", "priority": "low"},
            {"sample_id": "s2", "text": "same text too", "priority": "low"},
        ]
    )
    df_second = pd.DataFrame(
        [
            {"sample_id": "s1", "text": "same text", "priority": "high"},
            {"sample_id": "s2", "text": "same text too", "priority": "low"},
        ]
    )
    fingerprinter = DatasetFingerprinter(FingerprintConfig(mode="strict", hash_algo="sha256"))

    assert fingerprinter.fingerprint(DataFrameDatasetProvider(df_first)) != fingerprinter.fingerprint(
        DataFrameDatasetProvider(df_second)
    )


def test_strict_fingerprint_includes_backend_visible_meta_and_group_id() -> None:
    class Provider:
        def __init__(self, *, priority: str, group_id: str) -> None:
            self._sample = DataSample(
                sample_id="s1",
                data={"text": "same text"},
                meta={"priority": priority},
                group_id=group_id,
            )

        def iter_sample_ids(self):
            yield "s1"

        def get_sample(self, sample_id: str) -> DataSample:
            assert sample_id == "s1"
            return self._sample

        def schema(self) -> dict[str, str]:
            return {"sample_id": "str", "text": "str", "priority": "str", "group_id": "str"}

    fingerprinter = DatasetFingerprinter(FingerprintConfig(mode="strict", hash_algo="sha256"))

    baseline = fingerprinter.fingerprint(Provider(priority="low", group_id="a"))

    assert baseline != fingerprinter.fingerprint(Provider(priority="high", group_id="a"))
    assert baseline != fingerprinter.fingerprint(Provider(priority="low", group_id="b"))


class _ThreeSampleDataset:
    def __init__(self) -> None:
        self.samples = {
            sample_id: DataSample(sample_id=sample_id, data={"text": sample_id})
            for sample_id in ("s1", "s2", "s3")
        }

    def iter_sample_ids(self):
        yield from self.samples

    def get_sample(self, sample_id: str) -> DataSample:
        return self.samples[sample_id]

    def schema(self) -> dict[str, str]:
        return {"sample_id": "str", "text": "str"}


class _NoopModel:
    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        return [[0.5, 0.5] for _ in texts]

    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs: Any) -> None:
        return None

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> dict[str, float]:
        return {}


def test_explicit_splits_must_cover_every_dataset_sample_at_configure_time(tmp_path: Path) -> None:
    project = ActiveLearningProject("explicit-split-gap", tmp_path, lock=False)

    with pytest.raises(ConfigurationError, match="split"):
        project.configure(
            dataset=_ThreeSampleDataset(),
            model=_NoopModel(),
            label_schema=LabelSchema(task="text_classification", labels=["negative", "positive"]),
            label_backend_config=LabelBackendConfig(backend="custom"),
            label_backend=SimulatorLabelBackend(),
            scheduler_config=SchedulerConfig(strategy="random"),
            cache_config=CacheConfig(enable=False),
            split_config=SplitConfig(
                mode="explicit",
                explicit_splits={"train": ["s1"], "val": [], "test": ["s3"]},
            ),
        )


def test_custom_backend_config_reports_missing_injected_backend_contract() -> None:
    with pytest.raises(ConfigurationError, match="label_backend"):
        build_label_backend(LabelBackendConfig(backend="custom"))
