"""
Core engine of the Active Learning SDK.

If you are new to this codebase, read this file after `project.py`.

High-level idea:
- The engine runs active learning as a simple state machine over "rounds".
- After each step it checkpoints state to disk (`workdir/state.json`).
- On restart, it resumes from the last saved status without duplicating work.

Round state machine (MVP):
SELECT -> PUSH -> WAIT -> PULL -> TRAIN_EVAL -> UPDATE
"""

from __future__ import annotations


import dataclasses
import inspect
import json
import math
import re
import time
import uuid
from collections import Counter
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from .adapters.base import ModelCapabilities, TextClassificationAdapter, inspect_model_capabilities
from .annotation import AnnotationAggregator
from .backends.base import LabelBackend, RoundProgress, RoundPullResult, RoundPushResult, build_label_backend
from .cache import CacheStore, EmbeddingCache, InMemoryCacheStore, JsonlDiskCacheStore, PredictionCache
from .configs import (
    AnnotationPolicy,
    CacheConfig,
    FingerprintConfig,
    LabelBackendConfig,
    LabelSchema,
    PrelabelConfig,
    SchedulerConfig,
    SplitConfig,
    StopCriteria,
)
from .dataset.fingerprint import DatasetFingerprinter
from .dataset.provider import DataFrameDatasetProvider, DatasetProvider
from .exceptions import (
    ActiveLearningError,
    ConfigurationError,
    DatasetMismatchError,
    LabelBackendError,
    ModelAdapterError,
    StateCorruptedError,
    StopCriteriaReached,
    StrategyError,
)
from .report import ReportGenerator
from .runtime.selection_context_validation import validate_embedding_rows, validate_predict_proba_rows
from .runtime.split_resolution import (
    resolve_splits as resolve_runtime_splits,
    validate_persisted_splits as validate_runtime_persisted_splits,
    validate_resolved_split_stability as validate_runtime_resolved_split_stability,
)
from .state.lock import ProjectLock
from .state.store import (
    CURRENT_STATE_VERSION,
    DatasetRef,
    JsonFileStateStore,
    ProjectState,
    RoundState,
    StateStore,
    validate_state_version,
)
from .strategies.base import SamplingStrategy
from .strategies.adaptive import AdaptiveUncertaintyDiversityStrategy
from .strategies.badge import BadgeStrategy
from .strategies.embedding import (
    DeduplicateNearNeighborsStrategy,
    DensityWeightedDiversityStrategy,
    EmbeddingKMeansPPStrategy,
    KCenterGreedyStrategy,
    MaxMinEmbeddingStrategy,
)
from .strategies.hybrid import HybridStrategy, validate_hybrid_config
from .strategies.stochastic import (
    BaldStrategy,
    CommitteeKLDivergenceStrategy,
    CommitteeMarginStrategy,
    CommitteePairwiseDisagreementStrategy,
    CommitteeVoteEntropyStrategy,
    McDropoutEntropyStrategy,
    PredictionVarianceStrategy,
    VariationRatioStrategy,
    _normalize_probability_cube,
)
from .strategies.uncertainty import (
    ClassBalancedEntropyStrategy,
    ClassGroupBalancedEntropyStrategy,
    EntropyStrategy,
    GroupDiverseEntropyStrategy,
    LeastConfidenceStrategy,
    MarginStrategy,
    RandomStrategy,
)
from .types import DataSample, MetricRecord, ResolvedLabel, RoundStatus, SampleStatus, StepKind
from .utils import atomic_write_text, dataclass_to_dict, sha256_file, sha256_json

_ANNOTATION_TIMEOUT_TRACE_KEY = "annotation_timeout"
_SEED_TRAIN_COMPLETED_STATE_KEY = "_seed_train_completed"
_MODEL_CACHE_EPOCH_STATE_KEY = "_model_cache_epoch"
_GROUP_ID_SNAPSHOT_CACHE_KEY = "group_id_snapshot"
_AUDIT_EVENT_SCHEMA_VERSION = 1
_AUDIT_EVENT_LOG_NAME = "events.jsonl"
_SELECTION_AUDIT_SCHEMA_VERSION = 1
_SELECTION_AUDIT_MAX_UNSELECTED_IDS = 100
_BACKEND_AUDIT_HISTORY_LIMIT = 10
_BACKEND_AUDIT_MAX_ITEMS = 50
_BACKEND_AUDIT_MAX_DEPTH = 6
_BACKEND_AUDIT_MAX_STRING = 1000
_SECRET_KEY_MARKERS = ("token", "password", "secret", "authorization", "api_key", "apikey", "credential")
_SECRET_TEXT_PATTERNS = (
    re.compile(
        r"(?i)\b(token|password|secret|authorization|credential|api[_\s-]?key)\b\s*[:=]\s*[^\s,;]+"
    ),
    re.compile(r"(?i)\b(bearer|token)\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)\b[^\s,;]*?(token|password|secret|credential|api[_-]?key)[^\s,;]*\b"),
)
_GROUP_AWARE_STRATEGIES = frozenset(
    {
        "adaptive_uncertainty_diversity",
        "group_diverse_entropy",
        "class_group_balanced_entropy",
    }
)


def _pandas_for_csv_export() -> Any:
    try:
        import pandas as pd  # type: ignore
    except Exception as error:
        raise ConfigurationError("CSV export requires pandas.") from error
    return pd


class _MaterializedIdDatasetProvider:
    """Delegate provider access while reusing a validated, stable sample-id list."""

    def __init__(self, provider: DatasetProvider, sample_ids: Sequence[str]) -> None:
        self._provider = provider
        self._sample_ids = list(sample_ids)

    def iter_sample_ids(self) -> Iterator[str]:
        yield from self._sample_ids

    def get_sample(self, sample_id: str) -> DataSample:
        try:
            return self._provider.get_sample(sample_id)
        except ActiveLearningError:
            raise
        except Exception as error:
            raise ConfigurationError(f"dataset provider get_sample failed for sample_id={sample_id!r}: {error}") from error

    def get_samples(self, sample_ids: Sequence[str]) -> List[DataSample]:
        getter = getattr(self._provider, "get_samples", None)
        try:
            if callable(getter):
                return list(getter(sample_ids))
            return [self.get_sample(sample_id) for sample_id in sample_ids]
        except ActiveLearningError:
            raise
        except Exception as error:
            raise ConfigurationError(f"dataset provider get_samples failed for {len(sample_ids)} sample ids: {error}") from error

    def schema(self) -> Dict[str, str]:
        return self._provider.schema()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)


def _built_in_strategies() -> List[SamplingStrategy]:
    return [
        RandomStrategy(),
        AdaptiveUncertaintyDiversityStrategy(),
        EntropyStrategy(),
        ClassBalancedEntropyStrategy(),
        ClassGroupBalancedEntropyStrategy(),
        GroupDiverseEntropyStrategy(),
        LeastConfidenceStrategy(),
        MarginStrategy(),
        KCenterGreedyStrategy(),
        EmbeddingKMeansPPStrategy(),
        MaxMinEmbeddingStrategy(),
        DeduplicateNearNeighborsStrategy(),
        DensityWeightedDiversityStrategy(),
        BadgeStrategy(),
        McDropoutEntropyStrategy(),
        BaldStrategy(),
        VariationRatioStrategy(),
        PredictionVarianceStrategy(),
        CommitteeVoteEntropyStrategy(),
        CommitteeKLDivergenceStrategy(),
        CommitteePairwiseDisagreementStrategy(),
        CommitteeMarginStrategy(),
    ]


def _get_samples_from_provider(provider: DatasetProvider, sample_ids: Sequence[str]) -> List[DataSample]:
    getter = getattr(provider, "get_samples", None)
    try:
        if callable(getter):
            return list(getter(sample_ids))
        return [provider.get_sample(sample_id) for sample_id in sample_ids]
    except ActiveLearningError:
        raise
    except Exception as error:
        raise ConfigurationError(f"dataset provider failed to return {len(sample_ids)} sample(s): {error}") from error


def _validate_strategy_capabilities(
    strategy_name: str,
    strategy: SamplingStrategy,
    caps: ModelCapabilities,
    *,
    enforce_missing_capabilities: bool,
) -> None:
    required_capabilities = frozenset(getattr(strategy, "required_capabilities", frozenset()))
    missing = [name for name in sorted(required_capabilities) if not getattr(caps, name, False)]
    unsupported_reason = getattr(strategy, "unsupported_reason", None)

    if missing:
        if not enforce_missing_capabilities:
            return
        details = ", ".join(
            f"{name} ({caps.unsupported_methods.get(name, 'missing or not callable')})" for name in missing
        )
        if unsupported_reason:
            raise ConfigurationError(
                f"Strategy {strategy_name!r} requires missing model capability: {details}. "
                f"Strategy is unsupported: {unsupported_reason}"
            )
        raise ConfigurationError(
            f"Strategy {strategy_name!r} requires missing model capability: {details}."
        )

    if unsupported_reason:
        required = ", ".join(sorted(required_capabilities)) or "none"
        raise ConfigurationError(
            f"Strategy {strategy_name!r} is unsupported: {unsupported_reason} "
            f"Required capabilities: {required}."
        )


def _coerce_predict_proba_rows(
    probabilities: Any,
    sample_ids: Sequence[str],
    *,
    source: str = "model.predict_proba",
) -> List[Any]:
    if probabilities is None:
        raise ModelAdapterError(f"{source} must return one probability row per sample; got NoneType.")
    if isinstance(probabilities, (str, bytes)) or isinstance(probabilities, Real):
        raise ModelAdapterError(
            f"{source} must return an iterable of probability rows for {len(sample_ids)} sample ids; "
            f"got {type(probabilities).__name__}."
        )
    try:
        rows = list(probabilities)
    except TypeError as exc:
        raise ModelAdapterError(
            f"{source} must return an iterable of probability rows for {len(sample_ids)} sample ids; "
            f"got {type(probabilities).__name__}."
        ) from exc
    if len(rows) != len(sample_ids):
        raise ConfigurationError(f"{source} returned {len(rows)} rows for {len(sample_ids)} sample ids.")
    return rows


def _backend_lifecycle_error(
    backend: Any,
    method_name: str,
    error: Exception,
    *,
    round_id: Optional[str] = None,
) -> LabelBackendError:
    location = f" for round {round_id!r}" if round_id is not None else ""
    backend_name = type(backend).__name__
    return LabelBackendError(
        f"Label backend {backend_name}.{method_name} failed{location}: "
        f"{type(error).__name__}: {error}"
    )


def _backend_return_error(
    backend: Any,
    method_name: str,
    expected_type: type,
    value: Any,
    *,
    round_id: Optional[str] = None,
) -> LabelBackendError:
    location = f" for round {round_id!r}" if round_id is not None else ""
    backend_name = type(backend).__name__
    return LabelBackendError(
        f"Label backend {backend_name}.{method_name} returned invalid payload{location}: "
        f"expected {expected_type.__name__}, got {type(value).__name__}."
    )


class SelectionContext:
    """
    Context object passed to sampling strategies.

    Exposes:
    - dataset access (samples, texts)
    - model inference hooks (predict_proba, embed)
    - caches
    - project history (labeled IDs, metrics, round history)

    Strategies should not access the project state directly; they should use this context
    to avoid entangling strategy logic with persistence concerns.

    Attributes:
        provider (DatasetProvider):
            Where: used by `get_samples()`/`get_texts()` and indirectly by all strategies.
            What: runtime dataset access object (loads DataSample by id).
            Why: strategies work with ids, and only the provider can turn ids into real data.
        model (TextClassificationAdapter):
            Where: used by `predict_proba()`/`embed()` through the adapter contract.
            What: user-provided model wrapper (scikit-learn, transformers, etc).
            Why: lets strategies score samples without depending on a specific ML library.
        label_schema (LabelSchema):
            Where: available to custom strategies (and custom selectors) for validation/logging.
            What: description of label space and task type.
            Why: some strategies may need to know number of classes or label names.
        prediction_cache (Optional[PredictionCache]):
            Where: used inside `predict_proba()` to reuse model outputs.
            What: cache object keyed by (model_id, sample_id).
            Why: prevents recomputing predictions when restarting or re-running selection.
        embedding_cache (Optional[EmbeddingCache]):
            Where: used inside `embed()` to reuse embeddings.
            What: cache object keyed by (model_id, sample_id).
            Why: embeddings can be expensive; caching makes diversity strategies practical.
        labeled_ids (List[str]):
            Where: consumed by some strategies (e.g., diversity) and helpful for custom logic.
            What: ids that already have a resolved label in the project state.
            Why: selection should focus on unlabeled pool and avoid duplicates.
        last_metrics (Dict[str, float]):
            Where: can be used by adaptive strategies and bandits as "reward signal".
            What: metrics dict from the last TRAIN_EVAL step (e.g., accuracy).
            Why: enables learning-to-select approaches (choose strategies based on performance).
    """

    def __init__(
        self,
        *,
        provider: DatasetProvider,
        model: TextClassificationAdapter,
        label_schema: LabelSchema,
        prediction_cache: Optional[PredictionCache],
        embedding_cache: Optional[EmbeddingCache],
        labeled_ids: Sequence[str],
        last_metrics: Dict[str, float],
        dataset_fingerprint: Optional[str] = None,
        model_id_override: Optional[str] = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.label_schema = label_schema
        self.prediction_cache = prediction_cache
        self.embedding_cache = embedding_cache
        self.labeled_ids = list(labeled_ids)
        self.last_metrics = dict(last_metrics)
        self.dataset_fingerprint = dataset_fingerprint or "default-dataset"
        self._model_id_override = model_id_override
        self._model_id_cache: Optional[str] = None
        self._runtime_model_id = f"runtime:{uuid.uuid4().hex}"
        self._last_predict_proba_key: Optional[tuple[str, str, Tuple[str, ...]]] = None
        self._last_predict_proba_rows: Optional[List[Any]] = None
        self._strategy_diagnostics: List[Dict[str, Any]] = []

    def model_id(self) -> str:
        if self._model_id_override is not None:
            return self._model_id_override
        if self._model_id_cache is not None:
            return self._model_id_cache
        if callable(getattr(self.model, "get_model_id", None)):
            try:
                model_id = self.model.get_model_id()  # type: ignore[attr-defined]
            except Exception as e:
                raise ModelAdapterError(f"model.get_model_id() failed: {e}") from e
            if model_id is not None:
                model_id_str = str(model_id).strip()
                if model_id_str:
                    self._model_id_cache = model_id_str
                    return self._model_id_cache
        # Fallback is intentionally runtime-unique so persistent caches cannot
        # alias different adapters that lack a stable model id.
        self._model_id_cache = self._runtime_model_id
        return self._model_id_cache

    def get_samples(self, sample_ids: Sequence[str]) -> List[DataSample]:
        return _get_samples_from_provider(self.provider, sample_ids)

    def get_texts(self, sample_ids: Sequence[str]) -> List[str]:
        get_texts = getattr(self.provider, "get_texts", None)
        if callable(get_texts):
            texts = list(get_texts(sample_ids))
            if len(texts) != len(sample_ids):
                raise ConfigurationError(f"provider.get_texts returned {len(texts)} rows for {len(sample_ids)} sample ids.")
            return self._validate_text_rows(texts, sample_ids, source="provider.get_texts")
        samples = self.get_samples(sample_ids)
        sample_texts: List[Any] = []
        for sample_id, sample in zip(sample_ids, samples):
            if "text" not in sample.data:
                raise ConfigurationError(f"Sample {sample_id!r} is missing required data['text'] for text classification.")
            sample_texts.append(sample.data["text"])
        return self._validate_text_rows(sample_texts, sample_ids, source="sample.data['text']")

    def _validate_text_rows(self, texts: Sequence[Any], sample_ids: Sequence[str], *, source: str) -> List[str]:
        validated: List[str] = []
        for sample_id, text in zip(sample_ids, texts):
            if not isinstance(text, str):
                raise ConfigurationError(
                    f"{source} for sample {sample_id!r} must be a string, got {type(text).__name__}."
                )
            validated.append(text)
        return validated

    def record_strategy_diagnostic(self, strategy_name: str, diagnostic: Mapping[str, Any]) -> None:
        """Record JSON-serializable strategy diagnostics for the current selection call."""

        sanitized = self._json_safe_strategy_diagnostic(diagnostic)
        sanitized["strategy"] = str(strategy_name)
        self._strategy_diagnostics.append(sanitized)

    def clear_strategy_diagnostics(self) -> None:
        self._strategy_diagnostics.clear()

    def consume_strategy_diagnostics(self) -> List[Dict[str, Any]]:
        diagnostics = list(self._strategy_diagnostics)
        self._strategy_diagnostics.clear()
        return diagnostics

    def predict_proba(self, sample_ids: Sequence[str], batch_size: int = 32) -> Any:
        mid = self.model_id()
        sample_id_tuple = tuple(sample_ids)
        if self.prediction_cache is None:
            return self._predict_proba_no_cache(sample_ids, batch_size=batch_size)
        dataset_scope = self.dataset_fingerprint
        cached_result = self._last_predict_proba_result(mid, dataset_scope, sample_id_tuple)
        if cached_result is not None:
            return cached_result

        missing, out, cached_hits = self._collect_predict_proba_cache_rows(mid, dataset_scope, sample_id_tuple)

        if missing:
            self._fill_missing_predict_proba_rows(missing, out, mid, dataset_scope, batch_size=batch_size)

        result = self._ordered_predict_proba_rows(
            sample_id_tuple,
            out,
            cached_hits,
            mid,
            dataset_scope,
            batch_size=batch_size,
        )
        self._last_predict_proba_key = (mid, dataset_scope, sample_id_tuple)
        self._last_predict_proba_rows = result
        return list(result)

    def embed(self, sample_ids: Sequence[str], batch_size: int = 32) -> Any:
        if not callable(getattr(self.model, "embed", None)):
            raise ConfigurationError("Model does not support embeddings (embed method missing).")

        mid = self.model_id()
        embedding_config = self._embedding_config_key()
        if self.embedding_cache is None:
            return self._embed_no_cache(sample_ids, batch_size=batch_size)

        missing, out = self._collect_embedding_cache_rows(sample_ids, mid, embedding_config)

        if missing:
            self._fill_missing_embedding_rows(sample_ids, missing, out, mid, embedding_config, batch_size=batch_size)

        return self._validated_embedding_result_or_recompute(sample_ids, out, mid, embedding_config, batch_size=batch_size)

    def _last_predict_proba_result(
        self,
        model_id: str,
        dataset_scope: str,
        sample_ids: Tuple[str, ...],
    ) -> Optional[List[List[float]]]:
        if (
            self._last_predict_proba_key != (model_id, dataset_scope, sample_ids)
            or self._last_predict_proba_rows is None
        ):
            return None
        try:
            cleaned_rows = self._validate_predict_proba_cache_rows(self._last_predict_proba_rows, sample_ids)
        except ConfigurationError:
            self._last_predict_proba_key = None
            self._last_predict_proba_rows = None
            return None
        self._last_predict_proba_rows = cleaned_rows
        return list(cleaned_rows)

    def _collect_predict_proba_cache_rows(
        self,
        model_id: str,
        dataset_scope: str,
        sample_ids: Sequence[str],
    ) -> tuple[List[str], Dict[str, Any], List[str]]:
        assert self.prediction_cache is not None
        missing: List[str] = []
        out: Dict[str, Any] = {}
        cached_hits: List[str] = []
        for sid in sample_ids:
            cached = self.prediction_cache.get(model_id, sid, dataset_fingerprint=dataset_scope)
            if cached is None:
                missing.append(sid)
                continue
            try:
                out[sid] = self._validate_predict_proba_cache_rows([cached], [sid])[0]
            except ConfigurationError:
                self.prediction_cache.delete(model_id, sid, dataset_fingerprint=dataset_scope)
                missing.append(sid)
            else:
                cached_hits.append(sid)
        return missing, out, cached_hits

    def _fill_missing_predict_proba_rows(
        self,
        missing: Sequence[str],
        out: Dict[str, Any],
        model_id: str,
        dataset_scope: str,
        *,
        batch_size: int,
    ) -> None:
        assert self.prediction_cache is not None
        texts = self.get_texts(missing)
        try:
            proba = self.model.predict_proba(texts, batch_size=batch_size)
        except Exception as e:
            raise ModelAdapterError(f"model.predict_proba failed: {e}") from e

        proba_rows = _coerce_predict_proba_rows(proba, missing)
        for sid, row in zip(missing, self._validate_predict_proba_cache_rows(proba_rows, missing)):
            self.prediction_cache.set(model_id, sid, row, dataset_fingerprint=dataset_scope)
            out[sid] = row

    def _ordered_predict_proba_rows(
        self,
        sample_ids: Tuple[str, ...],
        out: Dict[str, Any],
        cached_hits: Sequence[str],
        model_id: str,
        dataset_scope: str,
        *,
        batch_size: int,
    ) -> List[List[float]]:
        missing_outputs = [sid for sid in sample_ids if sid not in out]
        if missing_outputs:
            preview = ", ".join(repr(sid) for sid in missing_outputs[:5])
            raise ConfigurationError(f"Missing predict_proba rows for sample ids: {preview}")

        result = [out[sid] for sid in sample_ids]
        try:
            return self._validate_predict_proba_cache_rows(result, sample_ids)
        except ConfigurationError:
            if not cached_hits:
                raise
        assert self.prediction_cache is not None
        for sid in cached_hits:
            self.prediction_cache.delete(model_id, sid, dataset_fingerprint=dataset_scope)
        return self._compute_and_cache_predict_proba(
            sample_ids,
            model_id,
            batch_size=batch_size,
            dataset_fingerprint=dataset_scope,
        )

    def _collect_embedding_cache_rows(
        self,
        sample_ids: Sequence[str],
        model_id: str,
        embedding_config: Any,
    ) -> tuple[List[str], Dict[str, Any]]:
        assert self.embedding_cache is not None
        missing: List[str] = []
        out: Dict[str, Any] = {}
        for sid in sample_ids:
            cached = self.embedding_cache.get(
                model_id,
                sid,
                dataset_fingerprint=self.dataset_fingerprint,
                embedding_config=embedding_config,
            )
            if cached is None:
                missing.append(sid)
                continue
            try:
                out[sid] = self._validate_embedding_cache_rows([cached], [sid])[0]
            except ConfigurationError:
                self.embedding_cache.delete(
                    model_id,
                    sid,
                    dataset_fingerprint=self.dataset_fingerprint,
                    embedding_config=embedding_config,
                )
                missing.append(sid)
        return missing, out

    def _fill_missing_embedding_rows(
        self,
        sample_ids: Sequence[str],
        missing: Sequence[str],
        out: Dict[str, Any],
        model_id: str,
        embedding_config: Any,
        *,
        batch_size: int,
    ) -> None:
        embs = self._compute_embedding_rows(missing, batch_size=batch_size)
        if out:
            embs, missing = self._embedding_rows_compatible_with_cache(
                sample_ids,
                missing,
                embs,
                out,
                model_id,
                embedding_config,
                batch_size,
            )
        assert self.embedding_cache is not None
        for sid, emb in zip(missing, embs):
            self.embedding_cache.set(
                model_id,
                sid,
                emb,
                dataset_fingerprint=self.dataset_fingerprint,
                embedding_config=embedding_config,
            )
            out[sid] = emb

    def _embedding_rows_compatible_with_cache(
        self,
        sample_ids: Sequence[str],
        missing: Sequence[str],
        embs: Sequence[Any],
        out: Dict[str, Any],
        model_id: str,
        embedding_config: Any,
        batch_size: int,
    ) -> tuple[List[List[float]], Sequence[str]]:
        tentative_out = dict(out)
        for sid, emb in zip(missing, embs):
            tentative_out[sid] = emb
        try:
            self._validate_embedding_cache_rows([tentative_out[sid] for sid in sample_ids], sample_ids)
            return list(embs), missing
        except ConfigurationError:
            self._delete_embedding_cache_rows(list(out.keys()), model_id, embedding_config)
            out.clear()
        return self._compute_embedding_rows(sample_ids, batch_size=batch_size), list(sample_ids)

    def _validated_embedding_result_or_recompute(
        self,
        sample_ids: Sequence[str],
        out: Dict[str, Any],
        model_id: str,
        embedding_config: Any,
        *,
        batch_size: int,
    ) -> List[List[float]]:
        result = [out[sid] for sid in sample_ids]
        try:
            return self._validate_embedding_cache_rows(result, sample_ids)
        except ConfigurationError:
            self._delete_embedding_cache_rows(sample_ids, model_id, embedding_config)
        result = self._compute_embedding_rows(sample_ids, batch_size=batch_size)
        assert self.embedding_cache is not None
        for sid, emb in zip(sample_ids, result):
            self.embedding_cache.set(
                model_id,
                sid,
                emb,
                dataset_fingerprint=self.dataset_fingerprint,
                embedding_config=embedding_config,
            )
        return result

    def _compute_embedding_rows(self, sample_ids: Sequence[str], *, batch_size: int) -> List[List[float]]:
        texts = self.get_texts(sample_ids)
        try:
            result = list(self.model.embed(texts, batch_size=batch_size))  # type: ignore[attr-defined]
        except Exception as e:
            raise ModelAdapterError(f"model.embed failed: {e}") from e
        if len(result) != len(sample_ids):
            raise ConfigurationError(f"model.embed returned {len(result)} rows for {len(sample_ids)} sample ids.")
        return self._validate_embedding_cache_rows(result, sample_ids)

    def _delete_embedding_cache_rows(
        self,
        sample_ids: Sequence[str],
        model_id: str,
        embedding_config: Any,
    ) -> None:
        assert self.embedding_cache is not None
        for sid in sample_ids:
            self.embedding_cache.delete(
                model_id,
                sid,
                dataset_fingerprint=self.dataset_fingerprint,
                embedding_config=embedding_config,
            )

    def gradient_embed(self, sample_ids: Sequence[str], labels: Sequence[Any] | None = None, batch_size: int = 32) -> Any:
        if not callable(getattr(self.model, "gradient_embed", None)):
            raise ConfigurationError("Model does not support gradient embeddings (gradient_embed method missing).")
        return self._gradient_embed_no_cache(sample_ids, labels=labels, batch_size=batch_size)

    def predict_stochastic(self, sample_ids: Sequence[str], n: int = 10, batch_size: int = 32) -> Any:
        caps = inspect_model_capabilities(self.model)
        if not caps.predict_stochastic:
            reason = caps.unsupported_methods.get("predict_stochastic", "missing or not callable")
            raise ConfigurationError(f"Model does not support stochastic prediction (predict_stochastic: {reason}).")
        texts = self.get_texts(sample_ids)
        try:
            predictions = self.model.predict_stochastic(texts, n=n, batch_size=batch_size)  # type: ignore[attr-defined]
        except Exception as e:
            raise ModelAdapterError(f"model.predict_stochastic failed: {e}") from e
        return _normalize_probability_cube(
            predictions,
            sample_ids,
            strategy_name="model",
            method_name="predict_stochastic",
            expected_member_count=n,
            expected_width=self._label_schema_width(),
        )

    def predict_committee(self, sample_ids: Sequence[str], batch_size: int = 32) -> Any:
        caps = inspect_model_capabilities(self.model)
        if not caps.predict_committee:
            reason = caps.unsupported_methods.get("predict_committee", "missing or not callable")
            raise ConfigurationError(f"Model does not support committee prediction (predict_committee: {reason}).")
        texts = self.get_texts(sample_ids)
        try:
            predictions = self.model.predict_committee(texts, batch_size=batch_size)  # type: ignore[attr-defined]
        except Exception as e:
            raise ModelAdapterError(f"model.predict_committee failed: {e}") from e
        return _normalize_probability_cube(
            predictions,
            sample_ids,
            strategy_name="model",
            method_name="predict_committee",
            min_member_count=2,
            expected_width=self._label_schema_width(),
        )

    def _embedding_config_key(self) -> Any:
        for name in ("get_embedding_config", "get_embedding_config_id", "get_embedding_version"):
            getter = getattr(self.model, name, None)
            if callable(getter):
                try:
                    value = getter()
                except Exception as e:
                    raise ModelAdapterError(f"model.{name}() failed: {e}") from e
                if value is not None:
                    return value
        for name in ("embedding_config", "embedding_version"):
            if hasattr(self.model, name):
                value = getattr(self.model, name)
                if value is not None:
                    return value
        return "default-embedding"

    def _predict_proba_no_cache(self, sample_ids: Sequence[str], batch_size: int) -> Any:
        texts = self.get_texts(sample_ids)
        try:
            proba = self.model.predict_proba(texts, batch_size=batch_size)
        except Exception as e:
            raise ModelAdapterError(f"model.predict_proba failed: {e}") from e
        proba_rows = _coerce_predict_proba_rows(proba, sample_ids)
        return self._validate_predict_proba_cache_rows(proba_rows, sample_ids)

    def _compute_and_cache_predict_proba(
        self,
        sample_ids: Sequence[str],
        model_id: str,
        *,
        batch_size: int,
        dataset_fingerprint: Optional[str] = None,
    ) -> List[Any]:
        assert self.prediction_cache is not None
        texts = self.get_texts(sample_ids)
        try:
            proba = self.model.predict_proba(texts, batch_size=batch_size)
        except Exception as e:
            raise ModelAdapterError(f"model.predict_proba failed: {e}") from e
        proba_rows = _coerce_predict_proba_rows(proba, sample_ids)
        cleaned_rows = self._validate_predict_proba_cache_rows(proba_rows, sample_ids)
        for sid, row in zip(sample_ids, cleaned_rows):
            self.prediction_cache.set(model_id, sid, row, dataset_fingerprint=dataset_fingerprint)
        return cleaned_rows

    def _validate_predict_proba_cache_rows(self, rows: Sequence[Any], sample_ids: Sequence[str]) -> List[List[float]]:
        return validate_predict_proba_rows(rows, sample_ids, label_width=self._label_schema_width())

    def _label_schema_width(self) -> Optional[int]:
        labels = getattr(self.label_schema, "labels", None)
        if labels is None:
            return None
        try:
            return len(list(labels))
        except TypeError:
            return None

    def _embed_no_cache(self, sample_ids: Sequence[str], batch_size: int) -> Any:
        texts = self.get_texts(sample_ids)
        try:
            embeddings = list(self.model.embed(texts, batch_size=batch_size))  # type: ignore[attr-defined]
        except Exception as e:
            raise ModelAdapterError(f"model.embed failed: {e}") from e
        if len(embeddings) != len(sample_ids):
            raise ConfigurationError(f"model.embed returned {len(embeddings)} rows for {len(sample_ids)} sample ids.")
        return self._validate_embedding_cache_rows(embeddings, sample_ids)

    def _validate_embedding_cache_rows(self, rows: Sequence[Any], sample_ids: Sequence[str]) -> List[List[float]]:
        return validate_embedding_rows(rows, sample_ids)

    def _gradient_embed_no_cache(self, sample_ids: Sequence[str], labels: Sequence[Any] | None, batch_size: int) -> Any:
        texts = self.get_texts(sample_ids)
        try:
            return self.model.gradient_embed(texts, labels=labels, batch_size=batch_size)  # type: ignore[attr-defined]
        except Exception as e:
            raise ModelAdapterError(f"model.gradient_embed failed: {e}") from e

    def _json_safe_strategy_diagnostic(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, bool)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                return None
            return value
        if isinstance(value, Real):
            number = float(value)
            if not math.isfinite(number):
                return None
            return number
        if isinstance(value, Mapping):
            return {
                str(key): self._json_safe_strategy_diagnostic(nested_value)
                for key, nested_value in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._json_safe_strategy_diagnostic(item) for item in value]
        return str(value)


class StrategyScheduler:
    """
    Strategy scheduler selects samples each round.

    This object owns:
    - available strategies registry
    - scheduling mode (single/mix/bandit/custom)
    - optional stateful logic (bandit stats)

    Public methods:
    - register_strategy()
    - available_strategies()
    - select_batch()
    - update_reward()  (bandit mode)

    Attributes:
        config (SchedulerConfig):
            Where: checked inside `select_batch()` to decide which mode to run.
            What: user configuration (single/mix/bandit/custom).
            Why: keeps selection policy in config instead of hardcoding it in the engine.
        _strategies (Dict[str, SamplingStrategy]):
            Where: populated by `register_strategy()` and read by `_get_strategy()`.
            What: registry mapping strategy name -> implementation.
            Why: lets users plug in custom strategies without editing engine code.
    """

    def __init__(self, config: SchedulerConfig, strategies: Optional[Sequence[SamplingStrategy]] = None) -> None:
        config.validate()
        self.config = config
        self._strategies: Dict[str, SamplingStrategy] = {}
        self._built_in_strategy_ids: set[int] = set()
        self._capabilities_by_model_id: Dict[int, ModelCapabilities] = {}
        self._validated_strategy_model_pairs: set[tuple[str, int, int]] = set()
        for strategy in _built_in_strategies():
            self.register_strategy(strategy)
            self._built_in_strategy_ids.add(id(strategy))
        if strategies:
            for s in strategies:
                self.register_strategy(s)

    def register_strategy(self, strategy: SamplingStrategy) -> None:
        """Register a sampling strategy by its unique name."""
        if not getattr(strategy, "name", None):
            raise ConfigurationError("Strategy must have a non-empty 'name' attribute.")
        self._strategies[strategy.name] = strategy

    def available_strategies(self) -> List[str]:
        """Return a list of registered strategy names."""
        return sorted(self._strategies.keys())

    def select_batch(self, pool_ids: Sequence[str], k: int, context: SelectionContext, state: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        """
        Select k sample_ids from the pool.

        Returns:
        - selected_ids
        - scheduler_snapshot (persist into RoundState for reproducibility)
        """
        # Developer notes:
        # - This function must be deterministic given:
        #     - pool_ids order
        #     - model outputs
        #     - scheduler state
        #   For true determinism, store RNG seeds and strategy internal randomness.
        if k <= 0:
            return [], {"mode": self.config.mode, "k": k}

        unique_pool = self._dedup_and_clip(pool_ids, len(pool_ids))
        target_k = min(k, len(unique_pool))

        if self.config.mode == "custom":
            return self._select_custom(unique_pool, target_k, context)

        if self.config.mode == "single":
            return self._select_single(unique_pool, target_k, context)

        if self.config.mode == "mix":
            return self._select_mix(unique_pool, target_k, context, requested_k=k)

        if self.config.mode == "mix_interleaved":
            return self._select_mix_interleaved(unique_pool, target_k, context, requested_k=k)

        if self.config.mode == "hybrid":
            return self._select_hybrid(unique_pool, target_k, context)

        if self.config.mode == "bandit":
            return self._select_bandit(unique_pool, target_k, context, state)

        raise ConfigurationError(f"Unsupported scheduler mode: {self.config.mode}")

    def _select_custom(
        self,
        pool_ids: Sequence[str],
        target_k: int,
        context: SelectionContext,
    ) -> Tuple[List[str], Dict[str, Any]]:
        assert self.config.custom_selector is not None
        selected = self._call_custom_selector(context, target_k, pool_ids)
        selected_ids = self._normalize_selection(
            selected,
            pool_ids,
            target_k,
            source="custom selector",
            refill=False,
            error_type=StrategyError,
            reject_duplicates=True,
        )
        return selected_ids, {"mode": "custom"}

    def _select_single(
        self,
        pool_ids: Sequence[str],
        target_k: int,
        context: SelectionContext,
    ) -> Tuple[List[str], Dict[str, Any]]:
        strat = self._get_strategy(self.config.strategy)
        self._validate_strategy_runtime_support(self.config.strategy, strat, context)
        selected, diagnostics = self._select_strategy_with_diagnostics(
            self.config.strategy,
            strat,
            pool_ids,
            target_k,
            context,
        )
        selected_ids = self._normalize_selection(
            selected,
            pool_ids,
            target_k,
            source=f"strategy {strat.name!r}",
            refill=id(strat) in self._built_in_strategy_ids,
        )
        snapshot: Dict[str, Any] = {"mode": "single", "strategy": strat.name}
        self._append_strategy_diagnostics(snapshot, diagnostics)
        return selected_ids, snapshot

    def _select_mix(
        self,
        pool_ids: Sequence[str],
        target_k: int,
        context: SelectionContext,
        *,
        requested_k: int,
    ) -> Tuple[List[str], Dict[str, Any]]:
        normalized_weights = self._normalized_mix_weights()
        alloc = self._allocate_sorted_mix_quotas(normalized_weights, target_k)
        actual_alloc: Dict[str, int] = {name: 0 for name in alloc}
        selected: List[str] = []
        remaining_pool = list(pool_ids)
        snapshot = self._mix_snapshot(
            requested_k=requested_k,
            target_k=target_k,
            pool_size=len(remaining_pool),
            weights=normalized_weights,
            alloc=alloc,
            actual_alloc=actual_alloc,
        )

        diagnostics = self._append_mix_strategy_selections(
            selected,
            remaining_pool,
            alloc,
            actual_alloc,
            context,
            target_k=target_k,
        )
        fallback_requested, fallback_actual, fallback_diagnostics = self._append_mix_fallback(
            selected,
            remaining_pool,
            context,
            target_k=target_k,
        )
        snapshot["fallback_requested"] = fallback_requested
        snapshot["fallback_actual"] = fallback_actual
        self._append_strategy_diagnostics(snapshot, [*diagnostics, *fallback_diagnostics])

        return self._normalize_selection(
            selected,
            pool_ids,
            target_k,
            source="mix scheduler",
            refill=True,
        ), snapshot

    def _select_hybrid(
        self,
        pool_ids: Sequence[str],
        target_k: int,
        context: SelectionContext,
    ) -> Tuple[List[str], Dict[str, Any]]:
        if not isinstance(self.config.hybrid, dict):
            raise ConfigurationError("hybrid config missing")
        hybrid = validate_hybrid_config(self.config.hybrid)
        for component_name in (hybrid["uncertainty"], hybrid["diversity"]):
            component = self._get_strategy(component_name)
            self._validate_strategy_runtime_support(component_name, component, context)
        result = HybridStrategy(self.config.hybrid).select(pool_ids, target_k, context)
        return self._normalize_selection(
            result.selected,
            pool_ids,
            target_k,
            source="hybrid scheduler",
            refill=True,
        ), result.snapshot

    def _select_bandit(
        self,
        pool_ids: Sequence[str],
        target_k: int,
        context: SelectionContext,
        state: Dict[str, Any],
    ) -> Tuple[List[str], Dict[str, Any]]:
        # Store arm stats in `state`, update via update_reward().
        arms = self.config.bandit_arms or []
        if not arms:
            raise ConfigurationError("bandit_arms is empty")
        chosen = self._choose_bandit_arm(arms, state)
        strat = self._get_strategy(chosen)
        self._validate_strategy_runtime_support(chosen, strat, context)
        selected, diagnostics = self._select_strategy_with_diagnostics(chosen, strat, pool_ids, target_k, context)
        snapshot = {"mode": "bandit", "algo": self.config.bandit_algo, "chosen_arm": chosen, "arms": list(arms)}
        self._append_strategy_diagnostics(snapshot, diagnostics)
        return self._normalize_selection(
            selected,
            pool_ids,
            target_k,
            source=f"strategy {chosen!r}",
            refill=True,
        ), snapshot

    def update_reward(self, reward: float, snapshot: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update internal scheduler state given a reward.

        In bandit mode, reward is used to update arm statistics.
        For non-bandit modes, this can be a no-op.

        Returns updated scheduler_state to persist in ProjectState.
        """
        # Developer notes:
        # - Reward should be normalized/clamped before updating bandit stats.
        # - Persist everything required to reproduce decisions later.
        if snapshot.get("mode") != "bandit":
            return state

        arm = snapshot.get("chosen_arm")
        if not arm:
            return state

        # Minimal placeholder stats.
        stats = dict(state.get("bandit", {}))
        arm_stat = dict(stats.get(arm, {"n": 0, "reward_sum": 0.0}))
        arm_stat["n"] = int(arm_stat["n"]) + 1
        arm_stat["reward_sum"] = float(arm_stat["reward_sum"]) + float(reward)
        stats[arm] = arm_stat
        state["bandit"] = stats
        return state

    def _get_strategy(self, name: str) -> SamplingStrategy:
        if name in self._strategies:
            return self._strategies[name]
        raise ConfigurationError(f"Unknown strategy: {name!r}. Available: {self.available_strategies()}")

    def _call_custom_selector(
        self,
        context: SelectionContext,
        k: int,
        pool_ids: Sequence[str],
    ) -> Sequence[str]:
        assert self.config.custom_selector is not None
        selector = self.config.custom_selector
        try:
            signature = inspect.signature(selector)
        except (TypeError, ValueError):
            try:
                return selector(context, k, list(pool_ids))  # type: ignore[misc]
            except TypeError:
                return selector(context, k)

        accepts_varargs = any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL
            for parameter in signature.parameters.values()
        )
        positional_capacity = sum(
            1
            for parameter in signature.parameters.values()
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
        if accepts_varargs or positional_capacity >= 3:
            return selector(context, k, list(pool_ids))  # type: ignore[misc]
        return selector(context, k)

    def _validate_strategy_runtime_support(
        self,
        strategy_name: str,
        strategy: SamplingStrategy,
        context: SelectionContext,
    ) -> None:
        model = getattr(context, "model", None)
        if model is None:
            return
        model_key = id(model)
        validation_key = (strategy_name, id(strategy), model_key)
        if validation_key in self._validated_strategy_model_pairs:
            return
        caps = self._capabilities_by_model_id.get(model_key)
        if caps is None:
            caps = inspect_model_capabilities(model)
            self._capabilities_by_model_id[model_key] = caps
        _validate_strategy_capabilities(
            strategy_name,
            strategy,
            caps,
            enforce_missing_capabilities=True,
        )
        self._validated_strategy_model_pairs.add(validation_key)

    def _clear_context_strategy_diagnostics(self, context: SelectionContext) -> None:
        clear = getattr(context, "clear_strategy_diagnostics", None)
        if callable(clear):
            clear()

    def _consume_context_strategy_diagnostics(
        self,
        context: SelectionContext,
        *,
        strategy_name: str,
    ) -> List[Dict[str, Any]]:
        consume = getattr(context, "consume_strategy_diagnostics", None)
        if not callable(consume):
            return []
        diagnostics = consume()
        if not isinstance(diagnostics, list):
            return []
        return [
            dict(diagnostic)
            for diagnostic in diagnostics
            if isinstance(diagnostic, dict) and diagnostic.get("strategy") == strategy_name
        ]

    def _select_strategy_with_diagnostics(
        self,
        strategy_name: str,
        strategy: SamplingStrategy,
        pool_ids: Sequence[str],
        target_k: int,
        context: SelectionContext,
    ) -> Tuple[Sequence[str], List[Dict[str, Any]]]:
        self._clear_context_strategy_diagnostics(context)
        try:
            selected = strategy.select(pool_ids, target_k, context)
        except Exception:
            self._consume_context_strategy_diagnostics(context, strategy_name=strategy_name)
            raise
        diagnostics = self._consume_context_strategy_diagnostics(context, strategy_name=strategy_name)
        return selected, diagnostics

    def _append_strategy_diagnostics(
        self,
        snapshot: Dict[str, Any],
        diagnostics: Sequence[Mapping[str, Any]],
    ) -> None:
        if diagnostics:
            existing = snapshot.setdefault("strategy_diagnostics", [])
            if isinstance(existing, list):
                existing.extend(dict(diagnostic) for diagnostic in diagnostics)

    def _dedup_and_clip(self, ids: Sequence[str], k: int) -> List[str]:
        seen = set()
        out: List[str] = []
        for sid in ids:
            if sid in seen:
                continue
            seen.add(sid)
            out.append(sid)
            if len(out) >= k:
                break
        return out

    def _validate_selection_in_pool(
        self,
        ids: Sequence[str],
        pool_ids: Sequence[str],
        *,
        source: str,
        error_type: type[Exception] = ConfigurationError,
    ) -> None:
        pool = set(pool_ids)
        outside = [sid for sid in ids if sid not in pool]
        if outside:
            preview = ", ".join(repr(sid) for sid in outside[:5])
            if len(outside) > 5:
                preview += ", ..."
            raise error_type(f"{source} returned sample IDs outside the candidate pool: {preview}")

    def _validate_no_duplicate_selection_ids(
        self,
        ids: Sequence[str],
        *,
        source: str,
        error_type: type[Exception],
    ) -> None:
        seen: set[str] = set()
        duplicates: List[str] = []
        for sid in ids:
            if sid in seen and sid not in duplicates:
                duplicates.append(sid)
            seen.add(sid)
        if duplicates:
            preview = ", ".join(repr(sid) for sid in duplicates[:5])
            if len(duplicates) > 5:
                preview += ", ..."
            raise error_type(f"{source} returned duplicate sample IDs: {preview}")

    def _normalize_selection(
        self,
        ids: Sequence[str],
        pool_ids: Sequence[str],
        k: int,
        *,
        source: str,
        refill: bool,
        error_type: type[Exception] = ConfigurationError,
        reject_duplicates: bool = False,
    ) -> List[str]:
        selected = list(ids)
        self._validate_selection_in_pool(selected, pool_ids, source=source, error_type=error_type)
        if reject_duplicates:
            self._validate_no_duplicate_selection_ids(selected, source=source, error_type=error_type)
        normalized = self._dedup_and_clip(selected, k)
        if refill:
            selected_set = set(normalized)
            for sid in pool_ids:
                if len(normalized) >= k:
                    break
                if sid in selected_set:
                    continue
                normalized.append(sid)
                selected_set.add(sid)
        return normalized

    def _append_remaining_selection(
        self,
        selected: List[str],
        remaining_pool: List[str],
        ids: Sequence[str],
        *,
        source: str,
    ) -> int:
        self._validate_selection_in_pool(ids, remaining_pool, source=source)
        remaining = set(remaining_pool)
        added_ids: List[str] = []
        for sid in ids:
            if sid not in remaining:
                continue
            remaining.remove(sid)
            added_ids.append(sid)
        if not added_ids:
            return 0
        selected.extend(added_ids)
        selected_set = set(selected)
        remaining_pool[:] = [sid for sid in remaining_pool if sid not in selected_set]
        return len(added_ids)

    def _normalized_mix_weights(self) -> Dict[str, float]:
        if not self.config.mix:
            raise ConfigurationError("mix config missing")
        weights = {name: weight for name, weight in self.config.mix.items() if weight > 0}
        total = sum(weights.values())
        return {name: weight / total for name, weight in weights.items()}

    def _allocate_sorted_mix_quotas(self, weights: Mapping[str, float], target_k: int) -> Dict[str, int]:
        return self._allocate_largest_remainder_quotas(weights, target_k, tie_order=sorted(weights.keys()))

    def _mix_snapshot(
        self,
        *,
        requested_k: int,
        target_k: int,
        pool_size: int,
        weights: Mapping[str, float],
        alloc: Mapping[str, int],
        actual_alloc: Dict[str, int],
    ) -> Dict[str, Any]:
        return {
            "mode": "mix",
            "k": requested_k,
            "target_k": target_k,
            "pool_size": pool_size,
            "mix": dict(weights),
            "requested_allocations": dict(alloc),
            "actual_allocations": actual_alloc,
        }

    def _append_mix_strategy_selections(
        self,
        selected: List[str],
        remaining_pool: List[str],
        alloc: Mapping[str, int],
        actual_alloc: Dict[str, int],
        context: SelectionContext,
        *,
        target_k: int,
    ) -> List[Dict[str, Any]]:
        diagnostics: List[Dict[str, Any]] = []
        for name in sorted(alloc.keys()):
            if len(selected) >= target_k or not remaining_pool:
                break
            part_k = min(target_k - len(selected), alloc[name], len(remaining_pool))
            if part_k <= 0:
                continue
            strat = self._get_strategy(name)
            self._validate_strategy_runtime_support(name, strat, context)
            part, part_diagnostics = self._select_strategy_with_diagnostics(name, strat, remaining_pool, part_k, context)
            diagnostics.extend(part_diagnostics)
            actual_alloc[name] += self._append_remaining_selection(
                selected,
                remaining_pool,
                part,
                source=f"strategy {name!r}",
            )
        return diagnostics

    def _append_mix_fallback(
        self,
        selected: List[str],
        remaining_pool: List[str],
        context: SelectionContext,
        *,
        target_k: int,
    ) -> Tuple[int, int, List[Dict[str, Any]]]:
        fallback_requested = min(target_k - len(selected), len(remaining_pool))
        if fallback_requested <= 0:
            return fallback_requested, 0, []
        fallback = self._get_strategy("random") if "random" in self._strategies else RandomStrategy()
        fallback_part, diagnostics = self._select_strategy_with_diagnostics(
            fallback.name,
            fallback,
            remaining_pool,
            fallback_requested,
            context,
        )
        fallback_actual = self._append_remaining_selection(
            selected,
            remaining_pool,
            fallback_part,
            source="strategy 'random'",
        )
        return fallback_requested, fallback_actual, diagnostics

    def _select_mix_interleaved(
        self,
        pool_ids: Sequence[str],
        k: int,
        context: SelectionContext,
        *,
        requested_k: Optional[int] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        if not self.config.mix:
            raise ConfigurationError("mix config missing")

        remaining_pool = list(pool_ids)
        target_k = min(k, len(remaining_pool))
        normalized_weights = self._normalized_mix_weights()
        alloc = self._allocate_mix_quotas(normalized_weights, target_k)
        actual_alloc: Dict[str, int] = {name: 0 for name in alloc}
        arm_order = list(alloc.keys())
        group_keys, group_lookup_available = self._group_keys_for_pool(remaining_pool, context)
        arm_candidates, diagnostics = self._mix_arm_candidates(remaining_pool, arm_order, context)
        selected: List[str] = []
        selected_groups: set[Tuple[str, str]] = set()

        self._select_interleaved_group_pass(
            selected,
            selected_groups,
            arm_order,
            alloc,
            actual_alloc,
            arm_candidates,
            remaining_pool,
            group_keys,
            target_k=target_k,
        )
        group_constrained_selected_count = len(selected)
        fallback_requested = target_k - len(selected)
        if fallback_requested > 0:
            fallback_actual = self._select_interleaved_relaxed_pass(
                selected,
                arm_order,
                alloc,
                actual_alloc,
                arm_candidates,
                remaining_pool,
                group_keys,
                target_k=target_k,
            )
        else:
            fallback_actual = 0

        selected_group_count = len({group_keys[sid] for sid in selected})
        snapshot = {
            "mode": "mix_interleaved",
            "k": k if requested_k is None else requested_k,
            "target_k": target_k,
            "pool_size": len(remaining_pool),
            "mix": dict(normalized_weights),
            "arm_order": arm_order,
            "requested_allocations": dict(alloc),
            "actual_allocations": actual_alloc,
            "fallback_requested": fallback_requested,
            "fallback_actual": fallback_actual,
            "group_lookup_available": group_lookup_available,
            "selected_group_count": selected_group_count,
            "group_constrained_selected_count": group_constrained_selected_count,
            "group_relaxed_fallback_count": fallback_actual,
        }
        self._append_strategy_diagnostics(snapshot, diagnostics)
        return self._normalize_selection(
            selected,
            remaining_pool,
            target_k,
            source="mix_interleaved scheduler",
            refill=True,
        ), snapshot

    def _select_interleaved_group_pass(
        self,
        selected: List[str],
        selected_groups: set[Tuple[str, str]],
        arm_order: Sequence[str],
        alloc: Mapping[str, int],
        actual_alloc: Dict[str, int],
        arm_candidates: Dict[str, List[str]],
        pool_ids: Sequence[str],
        group_keys: Mapping[str, Tuple[str, str]],
        *,
        target_k: int,
    ) -> None:
        while len(selected) < target_k:
            progressed = self._try_interleaved_group_round(
                selected,
                selected_groups,
                arm_order,
                alloc,
                actual_alloc,
                arm_candidates,
                pool_ids,
                group_keys,
                target_k=target_k,
            )
            if not progressed:
                break

    def _try_interleaved_group_round(
        self,
        selected: List[str],
        selected_groups: set[Tuple[str, str]],
        arm_order: Sequence[str],
        alloc: Mapping[str, int],
        actual_alloc: Dict[str, int],
        arm_candidates: Dict[str, List[str]],
        pool_ids: Sequence[str],
        group_keys: Mapping[str, Tuple[str, str]],
        *,
        target_k: int,
    ) -> bool:
        progressed = False
        for name in arm_order:
            if len(selected) >= target_k:
                break
            if actual_alloc[name] >= alloc[name]:
                continue
            candidate = self._pop_next_candidate(
                arm_candidates[name],
                selected,
                pool_ids,
                forbidden_groups=selected_groups,
                group_keys=group_keys,
            )
            if candidate is None:
                continue
            selected.append(candidate)
            selected_groups.add(group_keys[candidate])
            actual_alloc[name] += 1
            progressed = True
        return progressed

    def _select_interleaved_relaxed_pass(
        self,
        selected: List[str],
        arm_order: Sequence[str],
        alloc: Mapping[str, int],
        actual_alloc: Dict[str, int],
        arm_candidates: Dict[str, List[str]],
        pool_ids: Sequence[str],
        group_keys: Mapping[str, Tuple[str, str]],
        *,
        target_k: int,
    ) -> int:
        fallback_actual = 0
        while len(selected) < target_k:
            added = self._try_interleaved_relaxed_round(
                selected,
                arm_order,
                alloc,
                actual_alloc,
                arm_candidates,
                pool_ids,
                group_keys,
                target_k=target_k,
            )
            if added <= 0:
                break
            fallback_actual += added
        return fallback_actual

    def _try_interleaved_relaxed_round(
        self,
        selected: List[str],
        arm_order: Sequence[str],
        alloc: Mapping[str, int],
        actual_alloc: Dict[str, int],
        arm_candidates: Dict[str, List[str]],
        pool_ids: Sequence[str],
        group_keys: Mapping[str, Tuple[str, str]],
        *,
        target_k: int,
    ) -> int:
        added = 0
        for name in arm_order:
            if len(selected) >= target_k:
                break
            if actual_alloc[name] >= alloc[name]:
                continue
            candidate = self._pop_next_candidate(
                arm_candidates[name],
                selected,
                pool_ids,
                forbidden_groups=None,
                group_keys=group_keys,
            )
            if candidate is None:
                candidate = self._first_unselected_id(pool_ids, selected)
            if candidate is None:
                continue
            selected.append(candidate)
            actual_alloc[name] += 1
            added += 1
        return added

    def _allocate_mix_quotas(self, weights: Mapping[str, float], target_k: int) -> Dict[str, int]:
        return self._allocate_largest_remainder_quotas(weights, target_k, tie_order=list(weights.keys()))

    def _allocate_largest_remainder_quotas(
        self,
        weights: Mapping[str, float],
        target_k: int,
        *,
        tie_order: Sequence[str],
    ) -> Dict[str, int]:
        alloc: Dict[str, int] = {}
        remainders: Dict[str, float] = {}
        for name, weight in weights.items():
            exact = weight * target_k
            floor = int(exact)
            alloc[name] = floor
            remainders[name] = exact - floor
        deficit = target_k - sum(alloc.values())
        tie_rank = {name: index for index, name in enumerate(tie_order)}
        for name in sorted(weights.keys(), key=lambda item: (-remainders[item], tie_rank.get(item, len(tie_rank)), item)):
            if deficit <= 0:
                break
            alloc[name] += 1
            deficit -= 1
        return alloc

    def _mix_arm_candidates(
        self,
        pool_ids: Sequence[str],
        arm_order: Sequence[str],
        context: SelectionContext,
    ) -> Tuple[Dict[str, List[str]], List[Dict[str, Any]]]:
        candidates: Dict[str, List[str]] = {}
        diagnostics: List[Dict[str, Any]] = []
        for name in arm_order:
            strat = self._get_strategy(name)
            self._validate_strategy_runtime_support(name, strat, context)
            selected, arm_diagnostics = self._select_strategy_with_diagnostics(
                name,
                strat,
                pool_ids,
                len(pool_ids),
                context,
            )
            diagnostics.extend(arm_diagnostics)
            candidates[name] = self._normalize_selection(
                selected,
                pool_ids,
                len(pool_ids),
                source=f"strategy {name!r}",
                refill=False,
            )
        return candidates, diagnostics

    def _group_keys_for_pool(
        self,
        pool_ids: Sequence[str],
        context: SelectionContext,
    ) -> Tuple[Dict[str, Tuple[str, str]], bool]:
        keys = {sid: ("sample", sid) for sid in pool_ids}
        try:
            samples = context.get_samples(pool_ids)
        except Exception:
            return keys, False
        for sample in samples:
            sid = str(getattr(sample, "sample_id", ""))
            if sid not in keys:
                continue
            group_id = getattr(sample, "group_id", None)
            if group_id is not None:
                keys[sid] = ("group", str(group_id))
        return keys, True

    def _pop_next_candidate(
        self,
        candidates: List[str],
        selected: Sequence[str],
        pool_ids: Sequence[str],
        *,
        forbidden_groups: Optional[set[Tuple[str, str]]],
        group_keys: Mapping[str, Tuple[str, str]],
    ) -> Optional[str]:
        selected_set = set(selected)
        pool_set = set(pool_ids)
        for index, sid in enumerate(candidates):
            if sid in selected_set or sid not in pool_set:
                continue
            if forbidden_groups is not None and group_keys[sid] in forbidden_groups:
                continue
            return candidates.pop(index)
        return None

    def _first_unselected_id(self, pool_ids: Sequence[str], selected: Sequence[str]) -> Optional[str]:
        selected_set = set(selected)
        for sid in pool_ids:
            if sid not in selected_set:
                return sid
        return None

    def _choose_bandit_arm(self, arms: Sequence[str], state: Dict[str, Any]) -> str:
        arm_list = list(arms)
        if not arm_list:
            raise ConfigurationError("bandit_arms is empty")

        algo = (self.config.bandit_algo or "ucb1").lower()
        if algo != "ucb1":
            raise ConfigurationError(f"Unsupported bandit_algo={self.config.bandit_algo!r}; supported: 'ucb1'")

        stats = state.get("bandit", {}) if isinstance(state, dict) else {}
        if not isinstance(stats, dict):
            stats = {}

        parsed_stats: Dict[str, Tuple[int, float]] = {}
        total_pulls = 0
        for arm in arm_list:
            raw = stats.get(arm, {})
            if not isinstance(raw, Mapping):
                raw = {}
            n = max(0, int(raw.get("n", 0) or 0))
            reward_sum = float(raw.get("reward_sum", 0.0) or 0.0)
            parsed_stats[arm] = (n, reward_sum)
            total_pulls += n

        for arm in arm_list:
            if parsed_stats[arm][0] == 0:
                return arm

        exploration_denominator = max(1, total_pulls)
        best_arm = arm_list[0]
        best_score = -math.inf
        for arm in arm_list:
            n, reward_sum = parsed_stats[arm]
            average_reward = reward_sum / n
            score = average_reward + math.sqrt((2.0 * math.log(exploration_denominator)) / n)
            if score > best_score:
                best_arm = arm
                best_score = score
        return best_arm


@dataclass
class StepResult:
    """
    Result of executing a single step via run_step().

    Provides enough information for:
    - logging
    - UI integration
    - debugging and monitoring

    Attributes:
        step (StepKind):
            Where: returned from `ActiveLearningEngine.run_step()` to the caller.
            What: which state-machine step was executed (SELECT/PUSH/WAIT/...).
            Why: the caller can decide what to do next (sleep, refresh UI, etc.).
        round_id (Optional[str]):
            Where: returned to the caller and used in logs.
            What: active round identifier (or None if there is no active round yet).
            Why: helps correlate step results with a specific round.
        message (str):
            Where: user-friendly summary shown in logs/CLI/monitoring.
            What: short text description of what happened.
            Why: makes the step-by-step API easy to observe.
        details (Dict[str, Any]):
            Where: optional extra payload, e.g. `RoundProgress` for WAIT.
            What: JSON-serializable details (best-effort).
            Why: debugging/monitoring without exposing internal engine objects.
    """
    step: StepKind
    round_id: Optional[str]
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class ActiveLearningEngine:
    """
    ActiveLearningProject orchestrates the full Active Learning lifecycle.

    Public responsibilities:
    - manage persisted project state (idempotent, resumable)
    - protect dataset integrity via fingerprinting
    - run active learning rounds end-to-end or step-by-step
    - coordinate model training, evaluation, caching
    - integrate with a labeling backend (Label Studio by default)

    Public API overview:
    - configure(...)
    - run(...)
    - run_step(...)
    - status()
    - get_state()
    - list_rounds()
    - get_round(round_id)
    - validate()
    - generate_report(...)
    - export_labels(...)
    - export_dataset_split(...)
    - cache_stats()
    - clear_cache(...)
    - close()

    Extension points:
    - custom DatasetProvider
    - custom LabelBackend (human UI, LLM labeler, etc.)
    - custom sampling strategies and schedulers (bandit, user callbacks)
    - additional modalities via DataSample.data structure

    Attributes:
        project_name (str):
            Name of the project.
            Where: written into state and shown in status.
            What: user-provided identifier.
            Why: helps avoid mixing different projects in the same folder.
        workdir (Path):
            Directory where the engine stores persistent artifacts.
            Where: `state.json`, caches, lock file live here.
            What: filesystem path.
            Why: makes runs resumable and crash-safe.
        _state_store (StateStore):
            How the engine loads/saves `ProjectState`.
            Where: used by `_ensure_state_loaded()` and `_save_state()`.
            What: JsonFileStateStore by default.
            Why: can be swapped for SQLite or other storage later.
        _state (Optional[ProjectState]):
            In-memory copy of the persisted state.
            Where: read/written by every engine step.
            What: ProjectState dataclass.
            Why: central source of truth for resume and idempotency.
        _provider (Optional[DatasetProvider]):
            Live dataset provider (runtime object).
            Where: used to fetch samples/texts for strategies and training.
            Why: state.json stores only ids, provider gives the actual data.
        _model (Optional[TextClassificationAdapter]):
            Live model adapter (runtime object).
            Where: used in selection (predict_proba) and training (fit/evaluate).
            Why: engine stays ML-library-agnostic.
        _label_backend (Optional[LabelBackend]):
            Live backend connection (runtime object).
            Where: used in PUSH/WAIT/PULL steps.
            Why: integrates with Label Studio or other systems.
        _scheduler (Optional[StrategyScheduler]):
            The component that chooses which strategy to run and returns selected ids.
            Where: used in SELECT step.
            Why: centralizes selection policy.
        _prediction_cache/_embedding_cache (Optional[PredictionCache]/Optional[EmbeddingCache]):
            Caches used by SelectionContext to avoid recomputing predictions/embeddings.
            Where: selection step and any strategy calling `context.predict_proba()`/`context.embed()`.
            Why: performance and resume speed.
        _state_path (Path):
            Where: points to `workdir/state.json` for the default JsonFileStateStore.
            What: filesystem path to the persisted state file.
            Why: centralizes where state is stored so the engine can load/save it consistently.
        _lock_enabled (bool):
            Where: checked in `__init__` and `close()` to acquire/release a lock.
            What: whether locking is enabled for this engine instance.
            Why: prevents two processes from writing to the same state file at once.
        _lock (Optional[ProjectLock]):
            Where: created in `__init__` when locking is enabled.
            What: small filesystem lock helper for `project.lock`.
            Why: protects state.json and caches from concurrent corruption.
        _label_schema (Optional[LabelSchema]):
            Where: set in `configure()`/`attach_runtime()` and used by backends/validation.
            What: runtime object version of label schema.
            Why: needed for backend setup and for strategies that want schema info.
        _annotation_policy (Optional[AnnotationPolicy]):
            Where: set in `configure()`/`attach_runtime()` and used by the aggregator.
            What: runtime policy object (mode, min_votes, agreements).
            Why: determines how raw annotations become final labels.
        _aggregator (Optional[AnnotationAggregator]):
            Where: used in PULL step to turn many annotations into one label.
            What: helper that implements `AnnotationPolicy`.
            Why: keeps aggregation logic out of the engine step code.
        _reporter (Optional[ReportGenerator]):
            Where: created lazily in `generate_report()`.
            What: optional reporting component.
            Why: keeps heavy report dependencies separate from the core loop.
    """

    STATE_VERSION = CURRENT_STATE_VERSION

    def __init__(
        self,
        project_name: str,
        workdir: Union[str, Path],
        *,
        state_store: Optional[StateStore] = None,
        lock: bool = True,
    ) -> None:
        """
        Create or open an Active Learning project.

        Parameters
        ----------
        project_name:
            Human-readable project identifier. Stored in the state.
        workdir:
            Directory containing the project's persistent artifacts (state, caches, logs).
        state_store:
            Optional custom state store. Defaults to JSON file store in workdir/state.json.
        lock:
            If True, acquire an exclusive lock for the project to prevent concurrent writes.
        """
        self.project_name = project_name
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)

        self._lock_enabled = lock
        self._lock = ProjectLock(self.workdir / "project.lock") if lock else None

        self._state_path = self.workdir / "state.json"
        self._event_log_path = self.workdir / _AUDIT_EVENT_LOG_NAME
        self._selection_audit_dir = self.workdir / "audit" / "selections"
        self._audit_event_index: Optional[int] = None
        state_store_provided = state_store is not None
        self._state_store: StateStore = state_store or JsonFileStateStore(self._state_path)

        self._state: Optional[ProjectState] = None

        # Configured runtime components (set by configure()).
        self._provider: Optional[DatasetProvider] = None
        self._model: Optional[TextClassificationAdapter] = None
        self._label_schema: Optional[LabelSchema] = None
        self._annotation_policy: Optional[AnnotationPolicy] = None
        self._scheduler: Optional[StrategyScheduler] = None
        self._label_backend: Optional[LabelBackend] = None
        self._aggregator: Optional[AnnotationAggregator] = None
        self._reporter: Optional[ReportGenerator] = None

        self._prediction_cache: Optional[PredictionCache] = None
        self._embedding_cache: Optional[EmbeddingCache] = None
        self._seed_train_completed_in_runtime = False

        lock_acquired = False
        if self._lock_enabled and self._lock is not None:
            self._lock.acquire()
            lock_acquired = True

        try:
            # Custom stores may not be file-backed; let the abstraction decide.
            if state_store_provided:
                if self._provided_file_state_store_is_missing():
                    self._state = self._new_state()
                else:
                    self._state = self._state_store.load()
                    if isinstance(self._state_store, JsonFileStateStore):
                        self._validate_loaded_state_basic()
            elif self._state_path.exists():
                self._state = self._state_store.load()
                self._validate_loaded_state_basic()
            else:
                self._state = self._new_state()
        except Exception:
            if lock_acquired and self._lock is not None:
                self._lock.release()
            raise

    # ---------------------------------------------------------------------
    # Context manager helpers
    # ---------------------------------------------------------------------

    def __enter__(self) -> "ActiveLearningEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _provided_file_state_store_is_missing(self) -> bool:
        if not isinstance(self._state_store, JsonFileStateStore):
            return False
        return not self._state_store.state_path.exists()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def configure(
        self,
        *,
        dataset: Union[DatasetProvider, Any, str, Path],
        model: TextClassificationAdapter,
        label_schema: LabelSchema,
        label_backend_config: LabelBackendConfig,
        scheduler_config: SchedulerConfig,
        label_backend: Optional[LabelBackend] = None,
        annotation_policy: AnnotationPolicy = AnnotationPolicy(),
        cache_config: CacheConfig = CacheConfig(),
        fingerprint_config: FingerprintConfig = FingerprintConfig(),
        split_config: SplitConfig = SplitConfig(),
        prelabel_config: PrelabelConfig = PrelabelConfig(),
        strategies: Optional[Sequence[SamplingStrategy]] = None,
    ) -> None:
        """
        Configure the project with dataset, model, backend, and policies.

        This must be called before run()/run_step() for a fresh project.

        The configuration is persisted into the project state, enabling deterministic resume.

        Parameters
        ----------
        dataset:
            A DatasetProvider instance, or a pandas.DataFrame, or a path to CSV/Parquet.
        model:
            User-provided model adapter implementing the TextClassificationAdapter contract.
        label_schema:
            LabelSchema describing task and label set.
        label_backend_config:
            Configuration of the labeling backend (Label Studio, LLM, custom).
        scheduler_config:
            Configuration for selecting samples (single/mix/bandit/custom).
        label_backend:
            Optional backend instance to use instead of building one from label_backend_config.
            Required for 'llm' and 'custom' backend modes.
        annotation_policy:
            How to aggregate multiple annotations into a final label.
        cache_config:
            Prediction/embedding caching settings.
        fingerprint_config:
            Dataset fingerprint settings; ensures dataset integrity on resume.
        split_config:
            Train/val/test split settings; resolved split IDs are persisted.
        prelabel_config:
            Prelabeling settings (model suggestions to labelers).
        strategies:
            Optional runtime-only strategy instances to register before strict
            capability validation. Use this when scheduler_config refers to a
            custom strategy name.

        Raises
        ------
        ConfigurationError:
            If configuration is invalid or incompatible.
        DatasetMismatchError:
            If dataset does not match existing project fingerprint.
        """
        # Developer notes:
        # - Must be idempotent: re-configure should be either forbidden or explicit.
        # - Persist configs and resolved artifacts (fingerprint, splits) into state.
        # - Validate model capabilities before enabling strategies that require them.
        self._ensure_state_loaded()
        assert self._state is not None

        label_schema.validate()
        annotation_policy.validate()
        scheduler_config.validate()
        cache_config  # validated implicitly
        fingerprint_config.validate()
        split_config.validate()
        prelabel_config.validate()
        label_backend_config.validate()

        raw_provider = self._coerce_dataset(dataset)
        sample_ids = self._materialize_provider_sample_ids(raw_provider)
        provider = _MaterializedIdDatasetProvider(raw_provider, sample_ids)

        # Compute dataset fingerprint and validate against existing state.
        fingerprinter = DatasetFingerprinter(fingerprint_config)
        fp = fingerprinter.fingerprint(provider)

        if self._state and self._state.dataset_ref is not None:
            if self._state.dataset_ref.fingerprint != fp:
                raise DatasetMismatchError(
                    f"Dataset fingerprint mismatch. expected={self._state.dataset_ref.fingerprint} got={fp}"
                )
        self._validate_reconfigure_safety(
            label_schema=label_schema,
            annotation_policy=annotation_policy,
            scheduler_config=scheduler_config,
            label_backend_config=label_backend_config,
            split_config=split_config,
            prelabel_config=prelabel_config,
        )
        self._validate_reconfigure_schema_change(label_schema)

        # Validate model contract and scheduler/strategy compatibility.
        caps = inspect_model_capabilities(model)
        self._validate_model_capabilities(caps)
        self._validate_label_schema_scheduler_support(label_schema, scheduler_config)
        self._validate_scheduler_support(scheduler_config, caps, strategies=strategies)
        self._validate_prelabel_support(prelabel_config, caps)

        # Initialize caches.
        self._init_caches(cache_config)

        # Initialize label backend and aggregator.
        # Developer notes:
        # - The backend can be injected directly (recommended for custom/LLM backends).
        # - Otherwise we build one of the registered backends from config.
        if label_backend is not None:
            backend = label_backend
        else:
            if label_backend_config.backend in {"llm", "custom"}:
                raise ConfigurationError(
                    "label_backend must be provided when label_backend_config.backend is 'llm' or 'custom'."
                )
            backend = build_label_backend(label_backend_config)
        try:
            backend_ready = backend.ensure_ready(label_schema)
        except LabelBackendError:
            raise
        except Exception as error:
            raise _backend_lifecycle_error(backend, "ensure_ready", error) from error
        self._label_backend = backend

        self._provider = provider
        self._model = model
        self._label_schema = label_schema
        self._annotation_policy = annotation_policy
        self._aggregator = AnnotationAggregator(annotation_policy)

        # Scheduler owns built-in registration; provided strategies are runtime-only user overrides.
        scheduler = StrategyScheduler(scheduler_config, strategies=strategies)
        self._scheduler = scheduler

        # Persist configuration into state.
        self._state.dataset_ref = DatasetRef(
            source_type=self._infer_source_type(dataset),
            source_path=str(dataset) if isinstance(dataset, (str, Path)) else None,
            schema=provider.schema(),
            fingerprint=fp,
            fingerprint_config=dataclasses.asdict(fingerprint_config),
        )
        self._state.label_schema = dataclasses.asdict(label_schema)
        self._state.annotation_policy = dataclasses.asdict(annotation_policy)
        self._state.scheduler_config = self._scheduler_config_state_dict(scheduler_config)
        self._state.label_backend_config = dataclasses.asdict(label_backend_config)
        self._state.cache_config = dataclasses.asdict(cache_config)
        self._state.split_config = dataclasses.asdict(split_config)
        self._state.prelabel_config = dataclasses.asdict(prelabel_config)
        self._record_or_validate_group_id_snapshot(provider, sample_ids, scheduler_config)

        # Resolve and persist splits deterministically.
        resolved_splits = self._resolve_splits(provider, split_config, sample_ids=sample_ids)
        self._validate_resolved_split_stability(resolved_splits, split_config)
        self._state.splits = resolved_splits

        # Initialize sample statuses if first configure.
        if not self._state.sample_status:
            self._state.sample_status = {sid: SampleStatus.UNLABELED.value for sid in sample_ids}
        self._assert_sample_status_matches_dataset_ids(sample_ids)

        self._touch_state()
        self._append_audit_event(
            "project.configure",
            metadata={
                "dataset_fingerprint": fp,
                "sample_count": len(sample_ids),
                "scheduler_config": self._state.scheduler_config,
                "cache_config": self._state.cache_config,
            },
        )
        self._append_audit_event(
            "backend.ensure_ready",
            metadata={
                "backend": label_backend_config.backend,
                "summary": backend_ready,
            },
        )
        self._refresh_audit_artifact_refs()
        self._save_state()
        self._seed_train_completed_in_runtime = False

    def attach_runtime(
        self,
        *,
        dataset: Union[DatasetProvider, Any, str, Path],
        model: TextClassificationAdapter,
        label_backend: Optional[LabelBackend] = None,
        strategies: Optional[Sequence[SamplingStrategy]] = None,
    ) -> None:
        """
        Attach runtime components to an already-configured project.

        Use this when you open an existing workdir in a new Python process.
        The persisted configuration (label schema, policies, scheduler config, etc.)
        is loaded from state.json; this method only binds the live objects:
        - dataset provider (or DataFrame/path)
        - model adapter
        - optional label backend instance

        This method does NOT change persisted configuration unless you explicitly
        call configure() again.

        Parameters
        ----------
        dataset:
            DatasetProvider, pandas.DataFrame, or a path to CSV/Parquet.
        model:
            Model adapter instance.
        label_backend:
            Optional backend instance. If not provided, the backend is built from
            the persisted label_backend_config.
        strategies:
            Optional runtime-only strategy instances to register before strict
            capability validation. Required when persisted scheduler config uses
            a custom strategy name.

        Raises
        ------
        DatasetMismatchError:
            If the attached dataset does not match the stored dataset fingerprint.
        ConfigurationError:
            If the project is not configured or runtime binding fails.
        """
        # Developer notes:
        # - Load config objects from persisted dicts.
        # - Validate dataset fingerprint matches dataset_ref.
        # - Re-create scheduler and caches based on persisted configs.
        self._ensure_state_loaded()
        assert self._state is not None
        if self._state.dataset_ref is None or self._state.label_schema is None:
            raise ConfigurationError("Project is not configured; cannot attach runtime.")

        raw_provider = self._coerce_dataset(dataset)
        sample_ids = self._materialize_provider_sample_ids(raw_provider)
        provider = _MaterializedIdDatasetProvider(raw_provider, sample_ids)
        fp_cfg = FingerprintConfig(**self._state.dataset_ref.fingerprint_config)
        fp = DatasetFingerprinter(fp_cfg).fingerprint(provider)
        if fp != self._state.dataset_ref.fingerprint:
            raise DatasetMismatchError(
                f"Dataset fingerprint mismatch. expected={self._state.dataset_ref.fingerprint} got={fp}"
            )
        self._assert_sample_status_matches_dataset_ids(sample_ids)
        split_config = SplitConfig(**self._state.split_config) if self._state.split_config else SplitConfig()
        split_config.validate()
        if split_config.mode == "column":
            resolved_splits = self._resolve_splits(provider, split_config, sample_ids=sample_ids)
            self._validate_resolved_split_stability(resolved_splits, split_config)

        sc = SchedulerConfig(**self._state.scheduler_config) if self._state.scheduler_config else SchedulerConfig()
        if sc.mode == "custom" and sc.custom_selector is None:
            raise ConfigurationError(
                "Persisted scheduler mode is 'custom', but custom_selector is a runtime-only callable and cannot be "
                "restored from state.json. Call configure(...) again with SchedulerConfig(mode='custom', "
                "custom_selector=...) or use a serializable built-in strategy before attach_runtime()."
            )
        self._validate_group_id_snapshot(provider, sample_ids, sc)
        caps = inspect_model_capabilities(model)
        self._validate_model_capabilities(caps)
        self._validate_label_schema_scheduler_support(LabelSchema(**self._state.label_schema), sc)
        self._validate_scheduler_support(sc, caps, strategies=strategies)
        pc = PrelabelConfig(**self._state.prelabel_config) if self._state.prelabel_config else PrelabelConfig()
        self._validate_prelabel_support(pc, caps)

        # Bind runtime objects.
        self._provider = provider
        self._model = model
        self._label_schema = LabelSchema(**self._state.label_schema)
        self._annotation_policy = AnnotationPolicy(**self._state.annotation_policy)
        self._aggregator = AnnotationAggregator(self._annotation_policy)

        # Caches
        cc = CacheConfig(**self._state.cache_config) if self._state.cache_config else CacheConfig()
        self._init_caches(cc)

        # Scheduler owns built-in registration; provided strategies are runtime-only user overrides.
        self._scheduler = StrategyScheduler(sc, strategies=strategies)

        # Backend
        if label_backend is not None:
            backend = label_backend
        else:
            bc = LabelBackendConfig(**self._state.label_backend_config) if self._state.label_backend_config else LabelBackendConfig()
            if bc.backend in {"llm", "custom"}:
                raise ConfigurationError(
                    "Persisted backend is 'llm' or 'custom'. Provide label_backend explicitly in attach_runtime()."
                )
            backend = build_label_backend(bc)
        try:
            backend_ready = backend.ensure_ready(self._label_schema)
        except LabelBackendError:
            raise
        except Exception as error:
            raise _backend_lifecycle_error(backend, "ensure_ready", error) from error
        self._label_backend = backend
        self._seed_train_completed_in_runtime = False
        self._append_audit_event(
            "runtime.attach",
            metadata={
                "dataset_fingerprint": fp,
                "scheduler_config": self._state.scheduler_config,
            },
        )
        self._append_audit_event(
            "backend.ensure_ready",
            metadata={
                "backend": self._state.label_backend_config.get("backend"),
                "summary": backend_ready,
            },
        )
        self._refresh_audit_artifact_refs()
        self._touch_state()
        self._save_state()

    def register_strategy(self, strategy: SamplingStrategy) -> None:
        """
        Register a custom sampling strategy.

        This enables user-defined selection heuristics without modifying SDK internals.
        The strategy becomes available by its `strategy.name`.

        Notes
        -----
        You must call this after configure() or attach_runtime(), because the scheduler
        is created during those steps.
        """
        # Developer notes:
        # - Consider persisting the fact that a custom strategy was registered,
        #   if reproducibility across processes is required.
        self._ensure_configured()
        assert self._scheduler is not None
        strategy_name = getattr(strategy, "name", None)
        if not strategy_name:
            raise ConfigurationError("Strategy must have a non-empty 'name' attribute.")
        if self._configured_strict_scheduler_uses_strategy(strategy_name):
            assert self._model is not None
            caps = inspect_model_capabilities(self._model)
            _validate_strategy_capabilities(
                strategy_name,
                strategy,
                caps,
                enforce_missing_capabilities=True,
            )
        self._scheduler.register_strategy(strategy)

    def import_labels(
        self,
        labels: Mapping[str, Any],
        *,
        overwrite: bool = False,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Import externally known labels into project state.

        Imported labels are treated as resolved user/oracle labels. They update
        sample_status and sample_labels, but do not create synthetic rounds or
        backend tasks.
        """
        self._ensure_configured()
        assert self._state is not None
        assert self._label_schema is not None

        normalized_labels = self._normalize_import_labels(labels, self._label_schema)

        imported = 0
        unchanged = 0
        overwritten = 0
        metadata_cleared = False
        updates: List[Tuple[str, Any]] = []

        for sample_id, label in normalized_labels.items():
            current_label_exists = sample_id in self._state.sample_labels
            current_label = self._state.sample_labels.get(sample_id)
            current_status = self._state.sample_status.get(sample_id)

            if current_label_exists and self._labels_equal(current_label, label, self._label_schema):
                if current_status == SampleStatus.LABELED.value:
                    if self._state.sample_review_metadata.pop(sample_id, None) is not None:
                        metadata_cleared = True
                    unchanged += 1
                    continue
                imported += 1
            elif current_label_exists:
                if not overwrite:
                    raise ActiveLearningError(
                        f"Conflicting label for sample_id={sample_id!r}. "
                        "Pass overwrite=True to replace existing labels."
                    )
                overwritten += 1
            else:
                imported += 1

            updates.append((sample_id, label))

        for sample_id, label in updates:
            self._state.sample_labels[sample_id] = label
            self._state.sample_status[sample_id] = SampleStatus.LABELED.value
            self._state.sample_review_metadata.pop(sample_id, None)

        if imported or overwritten or metadata_cleared:
            if not any(r.selected_sample_ids or r.task_ids for r in self._state.rounds):
                self._clear_seed_train_completed_marker()
            self._touch_state()
            self._append_audit_event(
                "labels.import",
                metadata={
                    "imported": imported,
                    "unchanged": unchanged,
                    "overwritten": overwritten,
                    "total": len(normalized_labels),
                    "source": source,
                    "metadata_cleared": metadata_cleared,
                },
            )
            self._refresh_audit_artifact_refs()
            self._save_state()

        summary: Dict[str, Any] = {
            "imported": imported,
            "unchanged": unchanged,
            "overwritten": overwritten,
            "total": len(normalized_labels),
        }
        if source is not None:
            summary["source"] = source
        return summary

    def run(
        self,
        *,
        budget: Optional[int] = None,
        batch_size: int = 50,
        stop_criteria: StopCriteria = StopCriteria(),
        resume: bool = True,
        poll_interval_seconds: int = 10,
    ) -> None:
        """
        Run the Active Learning loop in blocking mode.

        The loop proceeds round-by-round until stop criteria is reached.

        Parameters
        ----------
        budget:
            Convenience alias for stop_criteria.max_labeled if provided.
        batch_size:
            Number of samples selected per round.
        stop_criteria:
            StopCriteria object defining termination conditions.
        resume:
            If True, continue unfinished rounds based on persisted state.
        poll_interval_seconds:
            Sleep interval between polling iterations during WAIT step.

        Raises
        ------
        ActiveLearningError:
            For configuration, backend, model, or state errors.
        """
        # Developer notes:
        # - This should be a thin wrapper around repeated run_step().
        # - Ensure we checkpoint state after every status transition.
        self._validate_batch_size(batch_size)
        self._ensure_configured()
        if not resume and self._has_active_round():
            raise ConfigurationError(
                "resume=False cannot continue an existing active round. Finish or repair the active round before "
                "starting a non-resume run."
            )

        if budget is not None:
            stop_criteria = dataclasses.replace(stop_criteria, max_labeled=budget)
        stop_criteria.validate()

        while True:
            # Enforce stop criteria before running steps.
            if self._should_stop(stop_criteria):
                self._save_state()
                break

            effective_batch_size = self._effective_batch_size(batch_size, stop_criteria)

            try:
                result = self.run_step(batch_size=effective_batch_size, poll_interval_seconds=poll_interval_seconds)
            except StopCriteriaReached as error:
                self._write_stop_criteria_reached_trace(stop_criteria, error)
                self._save_state()
                break
            if result.step == StepKind.NOOP:
                # No work to do (e.g., waiting but no polling in this call).
                # In blocking mode, we sleep and continue.
                time.sleep(poll_interval_seconds)
                continue

    def run_step(self, *, batch_size: int = 50, poll_interval_seconds: int = 0) -> StepResult:
        """
        Execute the next required step based on the current state.

        This method is the recommended integration point for:
        - notebooks,
        - workflow engines,
        - services where blocking behavior is undesirable.

        Parameters
        ----------
        batch_size:
            Target batch size for selection in the next round (if selecting).
        poll_interval_seconds:
            If > 0 and the next step is WAIT, the method performs one poll and then sleeps
            up to poll_interval_seconds (implementation-defined), returning progress.

        Returns
        -------
        StepResult:
            The executed step and context information.

        Raises
        ------
        ActiveLearningError:
            If the step cannot be executed due to misconfiguration or runtime failure.
        """
        # Developer notes:
        # - Determine next step from current round status or create a new round.
        # - Each step must be idempotent (safe to retry).
        self._validate_batch_size(batch_size)
        self._ensure_configured()

        state = self._state  # after _ensure_configured(), state is loaded
        assert state is not None

        if self._should_seed_train_before_select():
            metrics = self._step_seed_train_eval()
            return StepResult(
                step=StepKind.TRAIN_EVAL,
                round_id=None,
                message="Trained model on seed labels and evaluated metrics.",
                details={"seed": True, "metrics": metrics},
            )

        round_state = self._get_or_create_active_round()

        next_step = self._next_step(round_state)
        try:
            if next_step == StepKind.SELECT:
                self._step_select(round_state, batch_size=batch_size)
                return StepResult(step=StepKind.SELECT, round_id=round_state.round_id, message="Selected samples.")

            if next_step == StepKind.PUSH:
                self._step_push(round_state)
                return StepResult(step=StepKind.PUSH, round_id=round_state.round_id, message="Pushed tasks to backend.")

            if next_step == StepKind.WAIT:
                progress = self._step_wait(round_state)
                if poll_interval_seconds > 0:
                    time.sleep(poll_interval_seconds)
                return StepResult(step=StepKind.WAIT, round_id=round_state.round_id, message="Waiting for annotations.", details=dataclass_to_dict(progress))

            if next_step == StepKind.PULL:
                self._step_pull(round_state)
                return StepResult(step=StepKind.PULL, round_id=round_state.round_id, message="Pulled annotations and resolved labels.")

            if next_step == StepKind.TRAIN_EVAL:
                self._step_train_eval(round_state)
                return StepResult(step=StepKind.TRAIN_EVAL, round_id=round_state.round_id, message="Trained model and evaluated metrics.")

            if next_step == StepKind.UPDATE:
                self._step_update(round_state)
                return StepResult(step=StepKind.UPDATE, round_id=round_state.round_id, message="Updated scheduler state and checkpointed.")
        except Exception as error:
            if next_step in {StepKind.PUSH, StepKind.WAIT, StepKind.PULL}:
                self._record_backend_error(round_state, operation=next_step.value, error=error)
            if next_step != StepKind.WAIT:
                self._mark_round_failed(round_state, error)
            raise

        return StepResult(step=StepKind.NOOP, round_id=round_state.round_id if round_state else None, message="No action performed.")

    def status(self) -> Dict[str, Any]:
        """
        Return a concise status snapshot for monitoring and UI.

        Includes:
        - labeled/unlabeled counts
        - active round status
        - last metrics
        """
        self._ensure_state_loaded()
        assert self._state is not None
        labeled = sum(1 for s in self._state.sample_status.values() if s == SampleStatus.LABELED.value)
        unlabeled = sum(1 for s in self._state.sample_status.values() if s == SampleStatus.UNLABELED.value)
        needs_review = sum(1 for s in self._state.sample_status.values() if s == SampleStatus.NEEDS_REVIEW.value)
        invalid = sum(1 for s in self._state.sample_status.values() if s == SampleStatus.INVALID.value)

        active_rounds = self._active_rounds()
        last_round = active_rounds[-1] if active_rounds else None
        last_metrics = self._state.metrics_history[-1].metrics if self._state.metrics_history else {}

        return {
            "project_name": self._state.project_name,
            "counts": {"labeled": labeled, "unlabeled": unlabeled, "needs_review": needs_review, "invalid": invalid},
            "review_metadata": self._review_metadata_summary(self._state.sample_review_metadata),
            "active_round": dataclass_to_dict(last_round) if last_round else None,
            "last_metrics": dict(last_metrics),
            "state_version": self._state.state_version,
            "updated_at": self._state.updated_at,
        }

    def get_state(self) -> ProjectState:
        """
        Return the live ProjectState object for internal engine-level workflows.

        The public `ActiveLearningProject.get_state()` facade returns a detached
        snapshot. Engine-level callers and white-box tests use this live object
        intentionally.
        """
        self._ensure_state_loaded()
        assert self._state is not None
        return self._state

    def list_rounds(self) -> List[Dict[str, Any]]:
        """Return a list of round summaries."""
        self._ensure_state_loaded()
        assert self._state is not None
        return [
            {"round_id": r.round_id, "status": r.status.value, "created_at": r.created_at, "updated_at": r.updated_at}
            for r in self._state.rounds
        ]

    def get_round(self, round_id: str) -> Dict[str, Any]:
        """Return a specific round's detailed state."""
        self._ensure_state_loaded()
        assert self._state is not None
        for r in self._state.rounds:
            if r.round_id == round_id:
                return dataclass_to_dict(r)
        raise ConfigurationError(f"Unknown round_id={round_id!r}")

    def validate(self) -> Dict[str, Any]:
        """
        Validate internal consistency of the project.

        Checks:
        - dataset_ref present
        - fingerprint matches current dataset (if configured)
        - round status transitions are valid
        - task_ids mapping exists for pushed rounds
        - caches configuration is consistent

        Returns a ValidationReport as a dict.
        """
        self._ensure_state_loaded()
        assert self._state is not None
        report: Dict[str, Any] = {"ok": True, "issues": []}
        provider_ids: Optional[List[str]] = None

        if self._state.dataset_ref is None:
            report["ok"] = False
            report["issues"].append("dataset_ref is missing (project not configured).")
        elif self._provider is not None:
            try:
                fp_cfg = FingerprintConfig(**self._state.dataset_ref.fingerprint_config)
                current_fp = DatasetFingerprinter(fp_cfg).fingerprint(self._provider)
            except Exception as error:
                report["ok"] = False
                report["issues"].append(f"dataset fingerprint validation failed: {error}")
            else:
                if current_fp != self._state.dataset_ref.fingerprint:
                    report["ok"] = False
                    report["issues"].append(
                        "dataset fingerprint mismatch between persisted state and the attached runtime dataset."
                    )
            try:
                provider_ids = list(self._provider.iter_sample_ids())
                self._assert_sample_status_matches_dataset_ids(provider_ids)
            except Exception as error:
                report["ok"] = False
                report["issues"].append(f"sample_status/dataset coverage mismatch: {error}")

        split_reference_ids = provider_ids if provider_ids is not None else list(self._state.sample_status.keys())
        for issue in self._validate_persisted_splits(split_reference_ids):
            report["ok"] = False
            report["issues"].append(issue)

        active_rounds = self._active_rounds()
        if len(active_rounds) > 1:
            report["ok"] = False
            report["issues"].append(
                f"Project state contains multiple active rounds: {[round_state.round_id for round_state in active_rounds]}"
            )

        if self._state.label_schema:
            label_schema = LabelSchema(**self._state.label_schema)
            for sample_id, status in self._state.sample_status.items():
                if status == SampleStatus.LABELED.value and sample_id not in self._state.sample_labels:
                    report["ok"] = False
                    report["issues"].append(
                        f"Sample {sample_id!r} has labeled status but no persisted label in sample_labels."
                    )
            for sample_id, label in self._state.sample_labels.items():
                label_status = self._state.sample_status.get(sample_id)
                if label_status != SampleStatus.LABELED.value:
                    report["ok"] = False
                    if label_status is None:
                        report["issues"].append(
                            f"sample_labels entry for {sample_id!r} has no corresponding sample_status entry."
                        )
                    else:
                        report["issues"].append(
                            f"sample_labels entry for {sample_id!r} requires labeled status, found {label_status!r}."
                        )
                try:
                    self._normalize_import_label(label, label_schema, sample_id)
                except ConfigurationError as error:
                    report["ok"] = False
                    report["issues"].append(f"Invalid sample_labels entry for {sample_id!r}: {error}")
                if sample_id in self._state.sample_review_metadata:
                    report["ok"] = False
                    report["issues"].append(
                        f"sample_review_metadata for {sample_id!r} is stale because the sample is labeled."
                    )

        for sample_id in self._state.sample_review_metadata:
            review_status = self._state.sample_status.get(sample_id)
            if review_status is None:
                report["ok"] = False
                report["issues"].append(
                    f"sample_review_metadata entry for {sample_id!r} has no corresponding sample_status entry."
                )
            elif review_status == SampleStatus.LABELED.value:
                report["ok"] = False
                report["issues"].append(
                    f"sample_review_metadata entry for {sample_id!r} requires non-labeled status, found {review_status!r}."
                )

        # Validate round invariants.
        for r in self._state.rounds:
            if r.status in {RoundStatus.PUSHED, RoundStatus.WAITING, RoundStatus.READY_TO_PULL, RoundStatus.PULLED, RoundStatus.TRAINED, RoundStatus.DONE}:
                if not r.selected_sample_ids:
                    report["ok"] = False
                    report["issues"].append(f"Round {r.round_id} has status {r.status} but selected_sample_ids is empty.")
            if r.status in {RoundStatus.PUSHED, RoundStatus.WAITING, RoundStatus.READY_TO_PULL, RoundStatus.PULLED, RoundStatus.TRAINED, RoundStatus.DONE}:
                if not r.task_ids:
                    report["ok"] = False
                    report["issues"].append(f"Round {r.round_id} has status {r.status} but task_ids is empty (idempotency anchor missing).")
                else:
                    try:
                        self._validate_backend_task_ids(r, r.task_ids)
                    except ConfigurationError as error:
                        report["ok"] = False
                        report["issues"].append(
                            f"Round {r.round_id} has inconsistent task_ids and selected_sample_ids: {error}"
                        )

        return report

    def generate_report(self, output_path: Union[str, Path] = "report.html") -> Dict[str, Path]:
        """
        Generate strict JSON, Markdown, and HTML report artifacts for the project.
        """
        self._ensure_state_loaded()
        assert self._state is not None
        if self._reporter is None:
            self._reporter = ReportGenerator()
        self._append_audit_event(
            "report.generate",
            metadata={"output_path": str(output_path)},
        )
        self._refresh_audit_artifact_refs()
        self._touch_state()
        self._save_state()
        return self._reporter.generate_report(
            self._state,
            self.workdir / output_path,
            workdir=self.workdir,
            state_path=self._state_path,
        )

    def export_labels(self, output_path: Union[str, Path], *, format: str = "jsonl") -> None:
        """
        Export resolved labels for labeled samples.

        Parameters
        ----------
        output_path:
            Destination path.
        format:
            'jsonl' or 'csv' (csv may require pandas).

        Notes
        -----
        Export is read-only and does not modify project state.
        """
        self._ensure_state_loaded()
        assert self._state is not None
        if format not in {"jsonl", "csv"}:
            raise ConfigurationError(f"Unsupported export format: {format!r}")
        self._validate_label_export_invariants()
        out_path = Path(output_path)
        self._validate_export_file_path(out_path, api_name="export_labels")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        labeled_items = [
            {"sample_id": sid, "label": self._state.sample_labels.get(sid)}
            for sid, st in self._state.sample_status.items()
            if st == SampleStatus.LABELED.value
        ]

        if format == "jsonl":
            with out_path.open("w", encoding="utf-8") as f:
                for rec in labeled_items:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return

        if format == "csv":
            pd = _pandas_for_csv_export()
            df = pd.DataFrame(labeled_items)
            df.to_csv(out_path, index=False)
            return

        raise AssertionError("unreachable export format branch")

    def export_dataset_split(
        self,
        output_dir: Union[str, Path],
        *,
        which: str = "labeled",
        format: str = "jsonl",
    ) -> None:
        """
        Export a subset of the dataset by split/status.

        Parameters
        ----------
        output_dir:
            Output directory.
        which:
            'labeled' | 'unlabeled' | 'needs_review' | 'invalid' | 'all'
        format:
            'jsonl' | 'csv'
        """
        self._ensure_configured()
        assert self._state is not None
        assert self._provider is not None
        status_subsets = {
            SampleStatus.LABELED.value,
            SampleStatus.UNLABELED.value,
            SampleStatus.NEEDS_REVIEW.value,
            SampleStatus.INVALID.value,
        }
        split_subsets = set(self._state.splits)
        supported_subsets = status_subsets | split_subsets | {"all"}
        if which not in supported_subsets:
            raise ConfigurationError(f"Unsupported dataset split export subset: {which!r}")
        if format not in {"jsonl", "csv"}:
            raise ConfigurationError(f"Unsupported export format: {format!r}")
        self._validate_label_export_invariants()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if which == "all":
            selected_ids = list(self._state.sample_status)
        elif which in split_subsets:
            selected_ids = list(self._state.splits[which])
        else:
            selected_ids = [sid for sid, st in self._state.sample_status.items() if st == which]

        samples = _get_samples_from_provider(self._provider, selected_ids)
        records = []
        for sample in samples:
            record = {
                "sample_id": sample.sample_id,
                "status": self._state.sample_status.get(sample.sample_id),
                **sample.data,
            }
            if sample.meta:
                record["meta"] = sample.meta
            if sample.sample_id in self._state.sample_labels:
                record["label"] = self._state.sample_labels[sample.sample_id]
            records.append(record)

        out_path = out_dir / f"{which}.{format}"
        self._validate_export_file_path(out_path, api_name="export_dataset_split")
        if format == "jsonl":
            with out_path.open("w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return
        if format == "csv":
            pd = _pandas_for_csv_export()
            df = pd.DataFrame(records)
            df.to_csv(out_path, index=False)
            return
        raise AssertionError("unreachable export format branch")

    def _validate_export_file_path(self, output_path: Path, *, api_name: str) -> None:
        if output_path.exists() and output_path.is_dir():
            raise ConfigurationError(f"{api_name} output_path must be a file path, got directory: {output_path}")

    def _validate_label_export_invariants(self) -> None:
        assert self._state is not None
        label_schema = LabelSchema(**self._state.label_schema) if self._state.label_schema else None

        for sample_id, status in self._state.sample_status.items():
            if status == SampleStatus.LABELED.value and sample_id not in self._state.sample_labels:
                raise ConfigurationError(
                    f"Sample {sample_id!r} has labeled status but no persisted label in sample_labels."
                )

        for sample_id, label in self._state.sample_labels.items():
            label_status = self._state.sample_status.get(sample_id)
            if label_status != SampleStatus.LABELED.value:
                if label_status is None:
                    raise ConfigurationError(
                        f"sample_labels entry for {sample_id!r} has no corresponding sample_status entry."
                    )
                raise ConfigurationError(
                    f"sample_labels entry for {sample_id!r} requires labeled status, found {label_status!r}."
                )
            if label_schema is not None:
                self._normalize_import_label(label, label_schema, sample_id)

    def cache_stats(self) -> Dict[str, Any]:
        """
        Return cache statistics.

        Returns separate stats for prediction and embedding caches.
        """
        model_id = self._cache_model_id_for_cache() if self._model is not None else None
        dataset_fingerprint = (
            self._state.dataset_ref.fingerprint
            if self._state is not None and self._state.dataset_ref is not None
            else None
        )
        pred = (
            self._prediction_cache.stats(model_id=model_id, dataset_fingerprint=dataset_fingerprint)
            if self._prediction_cache
            else {"enabled": False}
        )
        embedding_config = self._current_embedding_config_key() if self._model is not None and self._embedding_cache else None
        emb = (
            self._embedding_cache.stats(
                model_id=model_id,
                dataset_fingerprint=dataset_fingerprint,
                embedding_config=embedding_config,
            )
            if self._embedding_cache
            else {"enabled": False}
        )
        return {"prediction_cache": pred, "embedding_cache": emb}

    def clear_cache(self, *, kind: str = "all", reason: str = "manual") -> None:
        """
        Clear caches.

        Parameters
        ----------
        kind:
            'predictions' | 'embeddings' | 'all'
        """
        if kind not in {"predictions", "embeddings", "all"}:
            raise ConfigurationError(f"Unsupported cache kind: {kind!r}")
        if kind in {"predictions", "all"} and self._prediction_cache:
            self._prediction_cache.clear(reason=reason, kind=kind)
        if kind in {"embeddings", "all"} and self._embedding_cache:
            self._embedding_cache.clear(reason=reason, kind=kind)
        self._append_audit_event(
            "cache.clear" if reason == "manual" else "cache.invalidated",
            metadata={
                "kind": kind,
                "reason": reason,
                "cache_stats": self.cache_stats(),
            },
        )
        if self._state is not None:
            self._refresh_audit_artifact_refs()
            self._touch_state()
            self._save_state()

    def _record_cache_invalidation(self, *, kind: str, reason: str) -> None:
        if kind in {"predictions", "all"} and self._prediction_cache:
            self._prediction_cache.record_invalidation(reason=reason, kind=kind)
        if kind in {"embeddings", "all"} and self._embedding_cache:
            self._embedding_cache.record_invalidation(reason=reason, kind=kind)

    def _current_embedding_config_key(self) -> Any:
        assert self._model is not None
        for name in ("get_embedding_config", "get_embedding_config_id", "get_embedding_version"):
            getter = getattr(self._model, name, None)
            if callable(getter):
                try:
                    value = getter()
                except Exception as e:
                    raise ModelAdapterError(f"model.{name}() failed: {e}") from e
                if value is not None:
                    return value
        for name in ("embedding_config", "embedding_version"):
            if hasattr(self._model, name):
                value = getattr(self._model, name)
                if value is not None:
                    return value
        return "default-embedding"

    def _text_rows_for_samples(self, samples: Sequence[DataSample], *, source: str) -> List[str]:
        texts: List[str] = []
        for sample in samples:
            if "text" not in sample.data:
                raise ConfigurationError(
                    f"{source} sample {sample.sample_id!r} is missing required data['text'] for text classification."
                )
            text = sample.data["text"]
            if not isinstance(text, str):
                raise ConfigurationError(
                    f"{source} sample {sample.sample_id!r} data['text'] must be a string, got {type(text).__name__}."
                )
            texts.append(text)
        return texts

    def close(self) -> None:
        """
        Close the project and release resources.

        - releases project lock
        - closes label backend if initialized
        """
        # Developer notes:
        # - Must be safe to call multiple times.
        if self._label_backend is not None:
            try:
                self._label_backend.close()
            except Exception:
                pass
        if self._lock_enabled and self._lock is not None:
            try:
                self._lock.release()
            except Exception:
                pass

    # ---------------------------------------------------------------------
    # Private helpers: state and configuration
    # ---------------------------------------------------------------------

    def _new_state(self) -> ProjectState:
        now = time.time()
        return ProjectState(
            state_version=self.STATE_VERSION,
            project_name=self.project_name,
            created_at=now,
            updated_at=now,
        )

    def _touch_state(self) -> None:
        assert self._state is not None
        self._state.updated_at = time.time()

    def _save_state(self) -> None:
        assert self._state is not None
        self._state_store.save_atomic(self._state)

    def _next_audit_event_index(self) -> int:
        if self._audit_event_index is not None:
            self._audit_event_index += 1
            return self._audit_event_index
        last_index = 0
        if self._event_log_path.exists():
            for raw_line in self._event_log_path.read_text(encoding="utf-8").splitlines():
                if not raw_line.strip():
                    continue
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise StateCorruptedError(f"Invalid JSON in audit event log: {error}") from error
                raw_index = payload.get("index") if isinstance(payload, Mapping) else None
                if isinstance(raw_index, int) and not isinstance(raw_index, bool):
                    last_index = max(last_index, raw_index)
        self._audit_event_index = last_index + 1
        return self._audit_event_index

    def _append_audit_event(
        self,
        event_type: str,
        *,
        round_id: Optional[str] = None,
        previous_status: Any = None,
        new_status: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        record = {
            "schema_version": _AUDIT_EVENT_SCHEMA_VERSION,
            "index": self._next_audit_event_index(),
            "timestamp": time.time(),
            "event_type": str(event_type),
            "project_name": self.project_name,
            "round_id": str(round_id) if round_id is not None else None,
            "previous_status": self._status_value(previous_status),
            "new_status": self._status_value(new_status),
            "metadata": self._diagnostic_json_safe(dict(metadata or {})),
        }
        serialized = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
        self._event_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._event_log_path.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
        return record

    def _status_value(self, status: Any) -> Optional[str]:
        if status is None:
            return None
        if isinstance(status, RoundStatus):
            return status.value
        return str(status)

    def _relative_workdir_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.workdir).as_posix()
        except ValueError:
            return str(path)

    def _audit_file_reference(self, path: Path) -> Dict[str, Any]:
        reference: Dict[str, Any] = {"path": self._relative_workdir_path(path)}
        if path.exists():
            reference["sha256"] = sha256_file(path)
        else:
            reference["sha256"] = None
            reference["missing"] = True
        return reference

    def _refresh_audit_artifact_refs(self) -> None:
        assert self._state is not None
        artifacts: Dict[str, Any] = dict(self._state.audit_artifacts)
        artifacts["event_log"] = self._audit_file_reference(self._event_log_path)
        selection_artifacts = []
        for round_state in self._state.rounds:
            if round_state.selection_audit:
                selection_artifacts.append(
                    {
                        "round_id": round_state.round_id,
                        **dict(round_state.selection_audit),
                    }
                )
        artifacts["selection_audits"] = selection_artifacts
        self._state.audit_artifacts = self._diagnostic_json_safe(artifacts)

    def _write_selection_audit_artifact(
        self,
        round_state: RoundState,
        *,
        pool_ids: Sequence[str],
        selected_ids: Sequence[str],
        scheduler_snapshot: Mapping[str, Any],
    ) -> Dict[str, Any]:
        selected_set = set(selected_ids)
        unselected_ids = [sample_id for sample_id in pool_ids if sample_id not in selected_set]
        unselected: Dict[str, Any] = {
            "count": len(unselected_ids),
            "hash": sha256_json(list(unselected_ids)),
        }
        if len(unselected_ids) <= _SELECTION_AUDIT_MAX_UNSELECTED_IDS:
            unselected["ids"] = list(unselected_ids)
        else:
            unselected["ids_preview"] = list(unselected_ids[:_SELECTION_AUDIT_MAX_UNSELECTED_IDS])
            unselected["ids_truncated"] = True

        payload = {
            "schema_version": _SELECTION_AUDIT_SCHEMA_VERSION,
            "project_name": self.project_name,
            "round_id": round_state.round_id,
            "created_at": time.time(),
            "eligible_pool": {
                "count": len(pool_ids),
                "hash": sha256_json(list(pool_ids)),
            },
            "selected": {
                "count": len(selected_ids),
                "ids": list(selected_ids),
                "hash": sha256_json(list(selected_ids)),
            },
            "unselected": unselected,
            "scheduler_snapshot": dict(scheduler_snapshot),
            "diagnostics": dict(scheduler_snapshot).get("strategy_diagnostics", []),
            "selection_metadata": self._selection_audit_metadata(scheduler_snapshot),
        }
        safe_payload = self._diagnostic_json_safe(payload)
        if not isinstance(safe_payload, dict):
            raise ConfigurationError("Selection audit artifact could not be normalized to a JSON object.")

        path = self._selection_audit_dir / f"{round_state.round_id}.selection.json"
        serialized = json.dumps(safe_payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        atomic_write_text(path, serialized + "\n")
        reference = {
            "path": self._relative_workdir_path(path),
            "sha256": sha256_file(path),
            "schema_version": _SELECTION_AUDIT_SCHEMA_VERSION,
            "eligible_count": len(pool_ids),
            "eligible_hash": safe_payload["eligible_pool"]["hash"],
            "selected_count": len(selected_ids),
            "selected_hash": safe_payload["selected"]["hash"],
            "unselected_count": len(unselected_ids),
            "unselected_hash": safe_payload["unselected"]["hash"],
        }
        return self._diagnostic_json_safe(reference)

    def _selection_audit_metadata(self, scheduler_snapshot: Mapping[str, Any]) -> Dict[str, Any]:
        keys = (
            "mode",
            "strategy",
            "chosen_arm",
            "fallback_requested",
            "fallback_actual",
            "requested_k",
            "target_k",
            "pool_size",
            "alloc",
            "actual_alloc",
            "score_key",
            "score_mean",
            "scores",
        )
        return {key: scheduler_snapshot[key] for key in keys if key in scheduler_snapshot}

    def _ensure_state_loaded(self) -> None:
        if self._state is None:
            self._state = self._state_store.load()

    def _ensure_configured(self) -> None:
        """
        Ensure the project is configured and runtime objects are attached.

        This checks two things:
        1. The persisted state indicates the project was configured at least once.
        2. In this Python process we have live objects attached (dataset/model/backend/etc).

        If you open an existing `workdir` in a new Python process, you usually need
        to call `attach_runtime(...)` before you can run steps.
        """
        self._ensure_state_loaded()
        assert self._state is not None
        if self._state.dataset_ref is None or self._state.label_schema is None:
            raise ConfigurationError("Project is not configured. Call configure() first.")
        if (
            self._provider is None
            or self._model is None
            or self._label_schema is None
            or self._annotation_policy is None
            or self._scheduler is None
            or self._label_backend is None
        ):
            raise ConfigurationError(
                "Runtime components are not initialized. Call configure() or attach_runtime() in this process."
            )
        if self._aggregator is None:
            self._aggregator = AnnotationAggregator(AnnotationPolicy(**self._state.annotation_policy))
        if self._provider is not None:
            self._assert_sample_status_matches_dataset_ids(list(self._provider.iter_sample_ids()))

    def _validate_loaded_state_basic(self) -> None:
        """
        Basic sanity checks for `state.json`.

        This intentionally stays small for the scaffold:
        - project name must match
        - state schema version must match
        """
        assert self._state is not None
        if self._state.project_name != self.project_name:
            # Allow opening same workdir with different name? Better to be strict.
            raise StateCorruptedError(
                f"Project name mismatch: state={self._state.project_name!r} init={self.project_name!r}"
            )
        validate_state_version(self._state.state_version)

    def _active_rounds(self) -> List[RoundState]:
        assert self._state is not None
        return [round_state for round_state in self._state.rounds if round_state.status not in {RoundStatus.DONE, RoundStatus.FAILED}]

    def _mark_round_failed(self, round_state: RoundState, error: Exception) -> None:
        assert self._state is not None
        previous_status = round_state.status
        round_state.status = RoundStatus.FAILED
        round_state.error = self._redact_secret_text(f"{type(error).__name__}: {error}")
        round_state.updated_at = time.time()
        self._touch_state()
        self._append_audit_event(
            "round.failed",
            round_id=round_state.round_id,
            previous_status=previous_status,
            new_status=round_state.status,
            metadata={"error": round_state.error},
        )
        self._refresh_audit_artifact_refs()
        self._save_state()

    def _has_active_round(self) -> bool:
        return bool(self._active_rounds())

    def _assert_single_active_round(self) -> None:
        active_rounds = self._active_rounds()
        if len(active_rounds) > 1:
            round_ids = [round_state.round_id for round_state in active_rounds]
            raise StateCorruptedError(f"Project state contains multiple active rounds: {round_ids}")

    def _assert_sample_status_matches_dataset_ids(self, sample_ids: Sequence[str]) -> None:
        assert self._state is not None
        expected = {str(sample_id) for sample_id in sample_ids}
        actual = set(self._state.sample_status.keys())
        if expected == actual:
            return
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details: List[str] = []
        if missing:
            details.append(f"missing dataset sample ids in sample_status: {missing[:10]}")
        if extra:
            details.append(f"sample_status contains ids not present in dataset: {extra[:10]}")
        raise StateCorruptedError("sample_status does not match attached dataset ids; " + "; ".join(details))

    def _validate_persisted_splits(self, reference_sample_ids: Sequence[str]) -> List[str]:
        assert self._state is not None
        return validate_runtime_persisted_splits(self._state.splits, reference_sample_ids)

    def _validate_reconfigure_schema_change(self, label_schema: LabelSchema) -> None:
        assert self._state is not None
        if self._state.label_schema is None:
            return
        next_schema = dataclasses.asdict(label_schema)
        if self._state.label_schema == next_schema:
            return
        if self._state.sample_labels or self._state.rounds:
            raise ConfigurationError(
                "Cannot change label_schema on a project that already contains labels or rounds. "
                "Create a new workdir or clear project state first."
            )

    def _validate_reconfigure_safety(
        self,
        *,
        label_schema: LabelSchema,
        annotation_policy: AnnotationPolicy,
        scheduler_config: SchedulerConfig,
        label_backend_config: LabelBackendConfig,
        split_config: SplitConfig,
        prelabel_config: PrelabelConfig,
    ) -> None:
        assert self._state is not None
        if self._state.dataset_ref is None:
            return
        next_split_config = dataclasses.asdict(split_config)
        if self._state.split_config is not None and self._state.split_config != next_split_config:
            raise ConfigurationError(
                "Cannot reconfigure split_config for an existing project workdir. "
                "Split assignments are part of persisted project identity; create a new workdir for different splits."
            )
        if not self._state.sample_labels and not self._state.rounds:
            return
        if self._state.scheduler_config.get("mode") == "custom" or scheduler_config.mode == "custom":
            raise ConfigurationError(
                "Cannot reconfigure a custom scheduler after project contains labels or rounds. "
                "Custom selector callables are runtime-only and cannot be proven equivalent. "
                "Create a new workdir or clear project state first."
            )

        next_values = {
            "label_schema": dataclasses.asdict(label_schema),
            "annotation_policy": dataclasses.asdict(annotation_policy),
            "scheduler_config": self._scheduler_config_state_dict(scheduler_config),
            "label_backend_config": dataclasses.asdict(label_backend_config),
            "split_config": next_split_config,
            "prelabel_config": dataclasses.asdict(prelabel_config),
        }
        current_values = {
            "label_schema": self._state.label_schema,
            "annotation_policy": self._state.annotation_policy,
            "scheduler_config": self._state.scheduler_config,
            "label_backend_config": self._state.label_backend_config,
            "split_config": self._state.split_config,
            "prelabel_config": self._state.prelabel_config,
        }
        for field_name, next_value in next_values.items():
            if current_values[field_name] != next_value:
                raise ConfigurationError(
                    f"Cannot reconfigure {field_name} after project contains labels or rounds. "
                    "Create a new workdir or clear project state first."
                )

    def _record_or_validate_group_id_snapshot(
        self,
        provider: DatasetProvider,
        sample_ids: Sequence[str],
        scheduler_config: SchedulerConfig,
    ) -> None:
        assert self._state is not None
        if not self._scheduler_depends_on_group_ids(scheduler_config):
            if not self._state.sample_labels and not self._state.rounds:
                self._state.caches_index.pop(_GROUP_ID_SNAPSHOT_CACHE_KEY, None)
            return

        current = self._group_id_snapshot(provider, sample_ids)
        existing = self._state.caches_index.get(_GROUP_ID_SNAPSHOT_CACHE_KEY)
        if existing is None or (not self._state.sample_labels and not self._state.rounds):
            self._state.caches_index[_GROUP_ID_SNAPSHOT_CACHE_KEY] = current
            return
        self._compare_group_id_snapshot(existing, current)

    def _validate_group_id_snapshot(
        self,
        provider: DatasetProvider,
        sample_ids: Sequence[str],
        scheduler_config: SchedulerConfig,
    ) -> None:
        assert self._state is not None
        if not self._scheduler_depends_on_group_ids(scheduler_config):
            return
        existing = self._state.caches_index.get(_GROUP_ID_SNAPSHOT_CACHE_KEY)
        if existing is None:
            raise ConfigurationError(
                "Missing persisted group_id snapshot for group-aware scheduler configuration; "
                "cannot safely attach runtime."
            )
        self._compare_group_id_snapshot(existing, self._group_id_snapshot(provider, sample_ids))

    def _scheduler_depends_on_group_ids(self, scheduler_config: SchedulerConfig) -> bool:
        if scheduler_config.mode == "mix_interleaved":
            return True
        if scheduler_config.mode == "hybrid" and isinstance(scheduler_config.hybrid, dict):
            hybrid = validate_hybrid_config(scheduler_config.hybrid)
            if hybrid.get("group_balance"):
                return True
        return any(strategy_name in _GROUP_AWARE_STRATEGIES for strategy_name in self._configured_strategy_names(scheduler_config))

    def _group_id_snapshot(self, provider: DatasetProvider, sample_ids: Sequence[str]) -> Dict[str, List[str]]:
        samples = _get_samples_from_provider(provider, sample_ids)
        if len(samples) != len(sample_ids):
            raise ConfigurationError(
                f"provider.get_samples returned {len(samples)} samples for {len(sample_ids)} sample ids."
            )
        snapshot: Dict[str, List[str]] = {}
        for expected_id, sample in zip(sample_ids, samples):
            sample_id = str(getattr(sample, "sample_id", expected_id))
            if sample_id != expected_id:
                raise ConfigurationError(
                    f"provider returned sample_id={sample_id!r} while reading expected sample_id={expected_id!r}."
                )
            group_id = getattr(sample, "group_id", None)
            if group_id is None:
                snapshot[expected_id] = ["sample", expected_id]
            else:
                snapshot[expected_id] = ["group", str(group_id)]
        return snapshot

    def _compare_group_id_snapshot(self, existing: Any, current: Mapping[str, Sequence[str]]) -> None:
        if not isinstance(existing, Mapping):
            raise ConfigurationError("Invalid persisted group_id snapshot; cannot safely attach runtime.")
        normalized_existing = {
            str(sample_id): [str(parts[0]), str(parts[1])]
            for sample_id, parts in existing.items()
            if isinstance(parts, SequenceABC) and not isinstance(parts, (str, bytes)) and len(parts) == 2
        }
        normalized_current = {str(sample_id): [str(parts[0]), str(parts[1])] for sample_id, parts in current.items()}
        if normalized_existing != normalized_current:
            raise DatasetMismatchError(
                "Dataset group_id assignments changed for a group-aware scheduler configuration."
            )

    def _scheduler_config_state_dict(self, scheduler_config: SchedulerConfig) -> Dict[str, Any]:
        payload = dataclasses.asdict(scheduler_config)
        if callable(payload.get("custom_selector")):
            payload["custom_selector"] = None
        return payload

    def _coerce_dataset(self, dataset: Union[DatasetProvider, Any, str, Path]) -> DatasetProvider:
        """
        Convert different dataset inputs into a `DatasetProvider`.

        Accepted forms:
        - a `DatasetProvider` instance (used as-is)
        - a `pandas.DataFrame` (wrapped by `DataFrameDatasetProvider`)
        - a path to CSV/Parquet (loaded with pandas, then wrapped)

        This keeps the rest of the engine simple: it always talks to a provider.
        """
        # Developer notes:
        # - Support DatasetProvider directly.
        # - Support pandas.DataFrame as a convenience.
        # - Support file paths (CSV/Parquet) through the provider's lazy pandas import.
        if isinstance(dataset, DatasetProvider):  # type: ignore[arg-type]
            return dataset
        if hasattr(dataset, "columns"):
            return DataFrameDatasetProvider(dataset)
        if isinstance(dataset, (str, Path)):
            path = Path(dataset)
            if not path.exists():
                raise ConfigurationError(f"Dataset path does not exist: {path}")
            if path.suffix.lower() == ".csv":
                return DataFrameDatasetProvider.from_path(path)
            if path.suffix.lower() in {".parquet", ".pq"}:
                return DataFrameDatasetProvider.from_path(path)
            raise ConfigurationError(f"Unsupported dataset file type: {path.suffix}")
        raise ConfigurationError("Unsupported dataset type. Provide a DatasetProvider, DataFrame, or file path.")

    def _materialize_provider_sample_ids(self, provider: DatasetProvider) -> List[str]:
        sample_ids: List[str] = []
        seen = set()
        for raw_sample_id in provider.iter_sample_ids():
            if not isinstance(raw_sample_id, str):
                raise ConfigurationError(
                    "dataset.iter_sample_ids() must yield string sample_id values; "
                    f"got {type(raw_sample_id).__name__} value {raw_sample_id!r}."
                )
            sample_id = raw_sample_id
            if not sample_id:
                raise ConfigurationError("dataset.iter_sample_ids() must not yield empty sample_id values.")
            if sample_id in seen:
                raise ConfigurationError(f"Duplicate sample_id in dataset: {sample_id!r}")
            seen.add(sample_id)
            sample_ids.append(sample_id)
        return sample_ids

    def _infer_source_type(self, dataset: Any) -> str:
        if isinstance(dataset, (str, Path)):
            return "path"
        if hasattr(dataset, "columns"):
            return "dataframe"
        return "provider"

    def _validate_model_capabilities(self, caps: ModelCapabilities) -> None:
        required_methods = ("fit", "evaluate")
        missing = [name for name in required_methods if not getattr(caps, name)]
        if missing:
            details = ", ".join(
                f"{name} ({caps.unsupported_methods.get(name, 'missing or not callable')})" for name in missing
            )
            raise ConfigurationError(f"Model adapter does not support required methods: {details}.")

    def _validate_scheduler_support(
        self,
        scheduler_config: SchedulerConfig,
        caps: ModelCapabilities,
        *,
        strategies: Optional[Sequence[SamplingStrategy]] = None,
    ) -> None:
        available_strategies = self._strategies_for_capability_validation(strategies)
        for strategy_name in self._configured_strategy_names(scheduler_config):
            strategy = self._strategy_for_capability_validation(
                strategy_name,
                strategies=available_strategies,
                allow_unknown=not scheduler_config.strict_capabilities,
            )
            if strategy is None:
                continue
            _validate_strategy_capabilities(
                strategy_name,
                strategy,
                caps,
                enforce_missing_capabilities=scheduler_config.strict_capabilities,
            )

    def _validate_prelabel_support(self, prelabel_config: PrelabelConfig, caps: ModelCapabilities) -> None:
        if not prelabel_config.enable:
            return
        if caps.predict_proba:
            return
        reason = caps.unsupported_methods.get("predict_proba", "missing or not callable")
        raise ConfigurationError(f"prelabel_config.enable=True requires model.predict_proba ({reason}).")

    def _validate_label_schema_scheduler_support(
        self,
        label_schema: LabelSchema,
        scheduler_config: SchedulerConfig,
    ) -> None:
        if not label_schema.multi_label:
            return

        unsupported: List[str] = []
        if scheduler_config.mode == "single":
            if scheduler_config.strategy != "random":
                unsupported.append(str(scheduler_config.strategy))
        elif scheduler_config.mode in {"mix", "mix_interleaved"}:
            unsupported.extend(sorted(name for name in (scheduler_config.mix or {}) if name != "random"))
        elif scheduler_config.mode in {"hybrid", "bandit"}:
            unsupported.append(scheduler_config.mode)

        if unsupported:
            joined = ", ".join(unsupported)
            raise ConfigurationError(
                "multi_label=True is supported for label import/normalization, but current built-in "
                f"acquisition strategies require single-label softmax probabilities. Unsupported strategy/mode: {joined}."
            )

    def _strategy_for_capability_validation(
        self,
        strategy_name: str,
        *,
        strategies: Optional[Mapping[str, SamplingStrategy]] = None,
        allow_unknown: bool = False,
    ) -> Optional[SamplingStrategy]:
        available_strategies = dict(strategies) if strategies is not None else self._strategies_for_capability_validation()
        if strategy_name in available_strategies:
            return available_strategies[strategy_name]
        if allow_unknown:
            return None
        available = sorted(available_strategies)
        raise ConfigurationError(
            f"Unknown strategy: {strategy_name!r}. Available strategies: {available}"
        )

    def _strategies_for_capability_validation(
        self,
        strategies: Optional[Sequence[SamplingStrategy]] = None,
    ) -> Dict[str, SamplingStrategy]:
        available = {strategy.name: strategy for strategy in _built_in_strategies()}
        if strategies:
            for strategy in strategies:
                if not getattr(strategy, "name", None):
                    raise ConfigurationError("Strategy must have a non-empty 'name' attribute.")
                available[strategy.name] = strategy
        return available

    def _configured_strict_scheduler_uses_strategy(self, strategy_name: str) -> bool:
        if self._state is None or not self._state.scheduler_config:
            return False
        scheduler_config = SchedulerConfig(**self._state.scheduler_config)
        if not scheduler_config.strict_capabilities:
            return False
        return strategy_name in self._configured_strategy_names(scheduler_config)

    def _configured_strategy_names(self, scheduler_config: SchedulerConfig) -> List[str]:
        if scheduler_config.mode == "single":
            return [scheduler_config.strategy]
        if scheduler_config.mode in {"mix", "mix_interleaved"} and scheduler_config.mix:
            return list(scheduler_config.mix.keys())
        if scheduler_config.mode == "hybrid" and isinstance(scheduler_config.hybrid, dict):
            hybrid = validate_hybrid_config(scheduler_config.hybrid)
            return [hybrid["uncertainty"], hybrid["diversity"]]
        if scheduler_config.mode == "bandit" and scheduler_config.bandit_arms:
            return list(scheduler_config.bandit_arms)
        return []

    def _normalize_import_labels(self, labels: Mapping[str, Any], label_schema: LabelSchema) -> Dict[str, Any]:
        assert self._state is not None
        if not isinstance(labels, Mapping):
            raise ConfigurationError("labels must be a mapping of sample_id to label.")

        normalized: Dict[str, Any] = {}
        for raw_sample_id, raw_label in labels.items():
            sample_id = str(raw_sample_id)
            if sample_id in normalized:
                raise ConfigurationError(f"Duplicate sample_id after string normalization: {sample_id!r}")
            if sample_id not in self._state.sample_status:
                raise ConfigurationError(f"Unknown sample_id={sample_id!r}; label import aborted.")
            normalized[sample_id] = self._normalize_import_label(raw_label, label_schema, sample_id)
        return normalized

    def _normalize_import_label(self, label: Any, label_schema: LabelSchema, sample_id: str) -> Any:
        if label_schema.multi_label:
            return self._normalize_multi_label_import(label, label_schema, sample_id)
        return self._normalize_single_label_import(label, label_schema, sample_id)

    def _normalize_backend_label(self, label: Any, sample_id: str) -> Any:
        assert self._label_schema is not None
        return self._normalize_import_label(label, self._label_schema, sample_id)

    def _normalize_single_label_import(self, label: Any, label_schema: LabelSchema, sample_id: str) -> str:
        if isinstance(label, (list, tuple, set, dict)):
            raise ConfigurationError(
                f"Invalid label for sample_id={sample_id!r}: single-label projects require one label value."
            )
        if label not in label_schema.labels:
            raise ConfigurationError(
                f"Invalid label for sample_id={sample_id!r}: {label!r} is not in LabelSchema.labels."
            )
        return str(label)

    def _normalize_multi_label_import(self, label: Any, label_schema: LabelSchema, sample_id: str) -> List[str]:
        if isinstance(label, (str, bytes)) or not isinstance(label, SequenceABC):
            raise ConfigurationError(
                f"Invalid label for sample_id={sample_id!r}: multi-label projects require a sequence of labels."
            )

        label_order = {value: index for index, value in enumerate(label_schema.labels)}
        seen = set()
        normalized: List[str] = []
        for value in label:
            if value not in label_order:
                raise ConfigurationError(
                    f"Invalid label for sample_id={sample_id!r}: {value!r} is not in LabelSchema.labels."
                )
            if value in seen:
                raise ConfigurationError(f"Duplicate label for sample_id={sample_id!r}: {value!r}")
            seen.add(value)
            normalized.append(str(value))

        normalized.sort(key=lambda value: label_order[value])
        return normalized

    def _labels_equal(self, left: Any, right: Any, label_schema: LabelSchema) -> bool:
        if not label_schema.multi_label:
            return left == right
        if not isinstance(left, SequenceABC) or isinstance(left, (str, bytes)):
            return False
        return list(left) == list(right)

    def _build_review_metadata(
        self,
        *,
        sample_id: str,
        status: SampleStatus,
        annotations: Sequence[Any],
        policy: AnnotationPolicy,
        round_id: str,
        source: str,
        resolution: Optional[ResolvedLabel] = None,
        reason: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        resolution_details = dict(resolution.details) if resolution is not None else {}
        if details:
            resolution_details.update(dict(details))
        metadata_reason = reason or str(resolution_details.get("reason") or status.value)
        metadata = {
            "sample_id": sample_id,
            "round_id": round_id,
            "source": source,
            "status": status.value,
            "reason": metadata_reason,
            "agreement": resolution.agreement if resolution is not None else None,
            "annotation_count": len(annotations),
            "eligible_vote_count": self._eligible_vote_count(annotations, policy),
            "details": resolution_details,
            "policy": dataclasses.asdict(policy),
        }
        safe_metadata = self._review_json_safe(metadata, path=("sample_review_metadata", sample_id))
        if not isinstance(safe_metadata, dict):
            raise ConfigurationError(f"Invalid review metadata for sample_id={sample_id!r}.")
        return safe_metadata

    def _eligible_vote_count(self, annotations: Sequence[Any], policy: AnnotationPolicy) -> int:
        if policy.allow_single_annotator:
            return len(annotations)
        annotator_ids = set()
        for annotation in annotations:
            annotator_ids.add(str(getattr(annotation, "annotator_id", "")))
        return len(annotator_ids)

    def _review_metadata_summary(self, sample_review_metadata: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
        by_reason = Counter(str(metadata.get("reason", "unknown")) for metadata in sample_review_metadata.values())
        return {
            "total": len(sample_review_metadata),
            "by_reason": {reason: by_reason[reason] for reason in sorted(by_reason)},
            "samples": self._review_json_safe(sample_review_metadata, path=("sample_review_metadata",)),
        }

    def _review_json_safe(self, value: Any, *, path: Sequence[str]) -> Any:
        if value is None or isinstance(value, (str, bool)):
            return value
        if isinstance(value, int):
            return int(value)
        if isinstance(value, Real):
            number = float(value)
            if not math.isfinite(number):
                raise ConfigurationError(f"Review metadata contains non-finite number at {self._review_json_path(path)}.")
            return number
        if isinstance(value, Mapping):
            result: Dict[str, Any] = {}
            for key, item in value.items():
                safe_key = self._review_json_key(key, path=(*path, "<key>"))
                result[safe_key] = self._review_json_safe(item, path=(*path, safe_key))
            return result
        if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes)):
            return [self._review_json_safe(item, path=(*path, f"[{index}]")) for index, item in enumerate(value)]
        raise ConfigurationError(
            f"Review metadata contains unsupported value at {self._review_json_path(path)}: {type(value).__name__}."
        )

    def _review_json_key(self, key: Any, *, path: Sequence[str]) -> str:
        safe_key = self._review_json_safe(key, path=path)
        if isinstance(safe_key, str):
            return safe_key
        return json.dumps(safe_key, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _review_json_path(self, path: Sequence[str]) -> str:
        rendered = "$"
        for part in path:
            if part.startswith("["):
                rendered += part
            else:
                rendered += f".{part}"
        return rendered

    def _diagnostic_json_safe(self, value: Any) -> Any:
        """Best-effort sanitizer for persisted diagnostic payloads from external systems."""
        if not isinstance(value, type) and dataclasses.is_dataclass(value):
            return self._diagnostic_json_safe(dataclass_to_dict(value))
        if value is None or isinstance(value, (str, bool)):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
        if isinstance(value, Real):
            number = float(value)
            return number if math.isfinite(number) else None
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): self._diagnostic_json_safe(item) for key, item in value.items()}
        if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
            return [self._diagnostic_json_safe(item) for item in value]
        if isinstance(value, (set, frozenset)):
            sanitized_items = [self._diagnostic_json_safe(item) for item in value]
            return sorted(sanitized_items, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, default=str))
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            try:
                return self._diagnostic_json_safe(tolist())
            except Exception:
                pass
        item = getattr(value, "item", None)
        if callable(item):
            try:
                return self._diagnostic_json_safe(item())
            except Exception:
                pass
        return repr(value)

    def _resolve_model_id_for_cache(self) -> Optional[str]:
        assert self._model is not None
        getter = getattr(self._model, "get_model_id", None)
        if not callable(getter):
            return None
        try:
            model_id = getter()
        except Exception:
            return None
        if model_id is None:
            return None
        model_id_str = str(model_id).strip()
        if not model_id_str:
            return None
        return model_id_str

    def _model_cache_epoch(self) -> int:
        assert self._state is not None
        raw = self._state.scheduler_state.get(_MODEL_CACHE_EPOCH_STATE_KEY, 0)
        if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
            return raw
        return 0

    def _cache_model_id_for_cache(self) -> Optional[str]:
        base_model_id = self._resolve_model_id_for_cache()
        if base_model_id is None:
            return None
        epoch = self._model_cache_epoch()
        return f"{base_model_id}::cache_epoch={epoch}"

    def _advance_model_cache_epoch(self) -> None:
        assert self._state is not None
        self._state.scheduler_state[_MODEL_CACHE_EPOCH_STATE_KEY] = self._model_cache_epoch() + 1

    def _init_caches(self, cache_config: CacheConfig) -> None:
        """
        Initialize prediction and embedding caches based on CacheConfig.

        Notes:
        - If `persist=True`, caches live under `workdir/caches/` and survive restarts.
        - If the model does not provide a stable `get_model_id()`, caches may be cleared
          after training to avoid mixing predictions across model versions.
        """
        if not cache_config.enable:
            self._prediction_cache = None
            self._embedding_cache = None
            self._append_audit_event(
                "cache.configure",
                metadata={"enabled": False, "persist": cache_config.persist, "max_items": cache_config.max_items},
            )
            return

        cache_dir = self.workdir / "caches"
        if cache_config.persist:
            pred_store: CacheStore = JsonlDiskCacheStore(cache_dir, "predictions")
            emb_store: CacheStore = JsonlDiskCacheStore(cache_dir, "embeddings")
        else:
            pred_store = InMemoryCacheStore(max_items=cache_config.max_items)
            emb_store = InMemoryCacheStore(max_items=cache_config.max_items)

        self._prediction_cache = PredictionCache(pred_store)
        self._embedding_cache = EmbeddingCache(emb_store)
        self._append_audit_event(
            "cache.configure",
            metadata={"enabled": True, "persist": cache_config.persist, "max_items": cache_config.max_items},
        )

    def _resolve_splits(
        self,
        provider: DatasetProvider,
        split_config: SplitConfig,
        *,
        sample_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, List[str]]:
        """
        Create train/val/test splits and return explicit sample_id lists.

        The engine stores the resulting split IDs into project state so that:
        - resume is deterministic
        - train/val sets stay the same across process restarts
        """
        # Developer notes:
        # - Persist explicit IDs for determinism.
        # - For extremely large datasets, consider storing split rules + seed instead.
        ids = list(sample_ids) if sample_ids is not None else self._materialize_provider_sample_ids(provider)
        return resolve_runtime_splits(provider, split_config, ids)

    def _validate_resolved_split_stability(
        self,
        resolved_splits: Mapping[str, Sequence[str]],
        split_config: SplitConfig,
    ) -> None:
        assert self._state is not None
        validate_runtime_resolved_split_stability(
            existing_splits=self._state.splits,
            has_dataset_ref=self._state.dataset_ref is not None,
            resolved_splits=resolved_splits,
            split_config=split_config,
        )

    def _should_stop(self, criteria: StopCriteria) -> bool:
        assert self._state is not None
        labeled_count = self._labeled_count()
        completed_round_count = self._completed_round_count()
        observed_checks: Dict[str, Any] = {}

        if not self._next_step_would_select():
            self._write_stop_trace(
                criteria,
                stopped=False,
                reason="not_at_selection_boundary",
                observed_values={},
                labeled_count=labeled_count,
                completed_round_count=completed_round_count,
            )
            return False

        if criteria.max_labeled is not None and labeled_count >= criteria.max_labeled:
            self._write_stop_trace(
                criteria,
                stopped=True,
                reason="max_labeled",
                observed_values={"max_labeled": criteria.max_labeled},
                labeled_count=labeled_count,
                completed_round_count=completed_round_count,
            )
            return True
        if criteria.max_rounds is not None and completed_round_count >= criteria.max_rounds:
            self._write_stop_trace(
                criteria,
                stopped=True,
                reason="max_rounds",
                observed_values={"max_rounds": criteria.max_rounds},
                labeled_count=labeled_count,
                completed_round_count=completed_round_count,
            )
            return True

        if not self._stop_minimums_met(criteria, labeled_count, completed_round_count):
            self._write_stop_trace(
                criteria,
                stopped=False,
                reason="minimums_not_met",
                observed_values={
                    "min_labeled": criteria.min_labeled,
                    "min_rounds": criteria.min_rounds,
                },
                labeled_count=labeled_count,
                completed_round_count=completed_round_count,
            )
            return False

        if criteria.plateau_rounds is not None:
            stopped, observed = self._metric_plateau_observation(
                criteria.metric_name,
                criteria.plateau_rounds,
                criteria.min_improvement,
            )
            observed_checks["metric_plateau"] = observed
            if stopped:
                self._write_stop_trace(
                    criteria,
                    stopped=True,
                    reason="metric_plateau",
                    observed_values=observed,
                    labeled_count=labeled_count,
                    completed_round_count=completed_round_count,
                )
                return True

        if criteria.acquisition_score_rounds is not None:
            stopped, observed = self._acquisition_score_converged(criteria)
            observed_checks["acquisition_score_convergence"] = observed
            if stopped:
                self._write_stop_trace(
                    criteria,
                    stopped=True,
                    reason="acquisition_score_convergence",
                    observed_values=observed,
                    labeled_count=labeled_count,
                    completed_round_count=completed_round_count,
                )
                return True

        if criteria.label_distribution_rounds is not None:
            stopped, observed = self._label_distribution_stabilized(criteria)
            observed_checks["label_distribution_stabilization"] = observed
            if stopped:
                self._write_stop_trace(
                    criteria,
                    stopped=True,
                    reason="label_distribution_stabilization",
                    observed_values=observed,
                    labeled_count=labeled_count,
                    completed_round_count=completed_round_count,
                )
                return True

        if criteria.calibration_rounds is not None:
            stopped, observed = self._calibration_stabilized(criteria)
            observed_checks["calibration_stabilization"] = observed
            if stopped:
                self._write_stop_trace(
                    criteria,
                    stopped=True,
                    reason="calibration_stabilization",
                    observed_values=observed,
                    labeled_count=labeled_count,
                    completed_round_count=completed_round_count,
                )
                return True

        self._write_stop_trace(
            criteria,
            stopped=False,
            reason="criteria_not_met",
            observed_values=observed_checks,
            labeled_count=labeled_count,
            completed_round_count=completed_round_count,
        )
        return False

    def _effective_batch_size(self, batch_size: int, criteria: StopCriteria) -> int:
        if criteria.max_labeled is None or not self._next_step_would_select():
            return batch_size
        remaining = criteria.max_labeled - self._labeled_count()
        return min(batch_size, max(remaining, 0))

    def _validate_batch_size(self, batch_size: int) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ConfigurationError("batch_size must be a positive integer.")

    def _labeled_count(self) -> int:
        assert self._state is not None
        return sum(1 for s in self._state.sample_status.values() if s == SampleStatus.LABELED.value)

    def _completed_round_count(self) -> int:
        assert self._state is not None
        return sum(1 for r in self._state.rounds if r.status == RoundStatus.DONE and r.selected_sample_ids)

    def _has_unlabeled_samples(self) -> bool:
        assert self._state is not None
        return bool(self._unlabeled_train_ids())

    def _next_step_would_select(self) -> bool:
        assert self._state is not None
        if not self._state.rounds:
            return True
        last_round = self._state.rounds[-1]
        if last_round.status in {RoundStatus.DONE, RoundStatus.FAILED}:
            return True
        return self._next_step(last_round) == StepKind.SELECT

    def _should_seed_train_before_select(self) -> bool:
        assert self._state is not None
        if not self._next_step_would_select():
            return False
        if self._seed_train_completed_in_runtime:
            return False
        if self._seed_train_completed_in_state():
            return False
        if any(r.selected_sample_ids or r.task_ids for r in self._state.rounds):
            return False
        if any(r.status in {RoundStatus.TRAINED, RoundStatus.DONE} for r in self._state.rounds):
            return False
        return bool(self._labeled_train_ids())

    def _labeled_train_ids(self) -> List[str]:
        assert self._state is not None
        train_ids = self._state.splits.get("train", [])
        return [sid for sid in train_ids if self._state.sample_status.get(sid) == SampleStatus.LABELED.value]

    def _unlabeled_train_ids(self) -> List[str]:
        assert self._state is not None
        train_ids = self._state.splits.get("train", [])
        return [sid for sid in train_ids if self._state.sample_status.get(sid) == SampleStatus.UNLABELED.value]

    def _seed_train_completed_in_state(self) -> bool:
        assert self._state is not None
        marker = self._state.scheduler_state.get(_SEED_TRAIN_COMPLETED_STATE_KEY)
        if not isinstance(marker, Mapping):
            return False
        marker_model_id = marker.get("model_id")
        current_model_id = self._resolve_model_id_for_cache() if self._model is not None else None
        return bool(marker.get("completed")) and marker_model_id is not None and marker_model_id == current_model_id

    def _mark_seed_train_completed(self) -> None:
        assert self._state is not None
        self._state.scheduler_state[_SEED_TRAIN_COMPLETED_STATE_KEY] = {
            "completed": True,
            "model_id": self._resolve_model_id_for_cache(),
        }

    def _clear_seed_train_completed_marker(self) -> None:
        assert self._state is not None
        self._seed_train_completed_in_runtime = False
        self._state.scheduler_state.pop(_SEED_TRAIN_COMPLETED_STATE_KEY, None)

    def _metric_plateau(self, metric_name: str, plateau_rounds: int, min_improvement: float) -> bool:
        stopped, _ = self._metric_plateau_observation(metric_name, plateau_rounds, min_improvement)
        return stopped

    def _metric_plateau_observation(
        self,
        metric_name: str,
        plateau_rounds: int,
        min_improvement: float,
    ) -> Tuple[bool, Dict[str, Any]]:
        assert self._state is not None
        # Developer notes:
        # - Plateau is computed on metrics_history. Ensure each TRAIN_EVAL logs a record.
        vals, skipped = self._metric_values_with_skipped(metric_name)
        observed: Dict[str, Any] = {
            "metric_name": metric_name,
            "values": vals,
            "skipped_values": skipped,
            "required_values": plateau_rounds + 1,
            "min_improvement": min_improvement,
        }
        if len(vals) < plateau_rounds + 1:
            return False, observed
        recent = vals[-(plateau_rounds + 1):]
        best_before = max(recent[:-1])
        last = recent[-1]
        improvement = last - best_before
        observed.update(
            {
                "recent_values": recent,
                "best_before": best_before,
                "last": last,
                "improvement": improvement,
            }
        )
        return improvement < min_improvement, observed

    def _calibration_stabilized(self, criteria: StopCriteria) -> Tuple[bool, Dict[str, Any]]:
        assert criteria.calibration_rounds is not None
        values, skipped = self._metric_values_with_skipped(criteria.calibration_metric_name)
        return self._series_stabilized(
            values,
            rounds=criteria.calibration_rounds,
            max_delta=criteria.calibration_min_delta,
            observed_prefix={
                "metric_name": criteria.calibration_metric_name,
                "values": values,
                "skipped_values": skipped,
                "min_delta": criteria.calibration_min_delta,
            },
        )

    def _acquisition_score_converged(self, criteria: StopCriteria) -> Tuple[bool, Dict[str, Any]]:
        assert self._state is not None
        assert criteria.acquisition_score_rounds is not None
        required = criteria.acquisition_score_rounds + 1
        scores: List[float] = []
        round_ids: List[str] = []
        completed_rounds = [round_state for round_state in self._state.rounds if round_state.status == RoundStatus.DONE]
        recent_completed_rounds = completed_rounds[-required:]
        missing_score_round_ids: List[str] = []
        for round_state in recent_completed_rounds:
            value = self._snapshot_numeric_score(round_state.scheduler_snapshot, criteria.acquisition_score_key)
            if value is None:
                missing_score_round_ids.append(round_state.round_id)
                continue
            scores.append(value)
            round_ids.append(round_state.round_id)
        observed_prefix: Dict[str, Any] = {
            "score_key": criteria.acquisition_score_key,
            "scores": scores,
            "round_ids": round_ids,
            "completed_round_ids": [round_state.round_id for round_state in completed_rounds],
            "recent_completed_round_ids": [round_state.round_id for round_state in recent_completed_rounds],
            "missing_score_round_ids": missing_score_round_ids,
            "required_completed_rounds": required,
            "min_delta": criteria.acquisition_score_min_delta,
        }
        if len(recent_completed_rounds) < required or missing_score_round_ids:
            observed_prefix["required_values"] = required
            return False, observed_prefix
        return self._series_stabilized(
            scores,
            rounds=criteria.acquisition_score_rounds,
            max_delta=criteria.acquisition_score_min_delta,
            observed_prefix=observed_prefix,
        )

    def _label_distribution_stabilized(self, criteria: StopCriteria) -> Tuple[bool, Dict[str, Any]]:
        assert self._state is not None
        assert criteria.label_distribution_rounds is not None
        assert criteria.label_distribution_max_delta is not None

        distributions: List[Dict[str, float]] = []
        round_ids: List[str] = []
        for round_state in self._state.rounds:
            if round_state.status != RoundStatus.DONE or not round_state.resolved:
                continue
            distributions.append(self._label_distribution(list(round_state.resolved.values())))
            round_ids.append(round_state.round_id)

        required = criteria.label_distribution_rounds + 1
        observed: Dict[str, Any] = {
            "round_ids": round_ids,
            "distributions": distributions,
            "required_distributions": required,
            "max_delta": criteria.label_distribution_max_delta,
        }
        if len(distributions) < required:
            return False, observed

        recent = distributions[-required:]
        deltas = [
            self._distribution_l1_delta(previous, current)
            for previous, current in zip(recent, recent[1:])
        ]
        observed.update({"recent_distributions": recent, "l1_deltas": deltas, "max_observed_delta": max(deltas)})
        return max(deltas) <= criteria.label_distribution_max_delta, observed

    def _metric_values(self, metric_name: str) -> List[float]:
        values, _ = self._metric_values_with_skipped(metric_name)
        return values

    def _metric_values_with_skipped(self, metric_name: str) -> Tuple[List[float], List[Dict[str, Any]]]:
        assert self._state is not None
        values: List[float] = []
        skipped: List[Dict[str, Any]] = []
        for record_index, rec in enumerate(self._state.metrics_history):
            if metric_name in rec.metrics:
                raw_value = rec.metrics[metric_name]
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    skipped.append(
                        {
                            "record_index": record_index,
                            "step": rec.step,
                            "value": repr(raw_value),
                            "reason": "not_numeric",
                        }
                    )
                    continue
                if not math.isfinite(value):
                    skipped.append(
                        {
                            "record_index": record_index,
                            "step": rec.step,
                            "value": repr(raw_value),
                            "reason": "not_finite",
                        }
                    )
                    continue
                values.append(value)
        return values, skipped

    def _series_stabilized(
        self,
        values: Sequence[float],
        *,
        rounds: int,
        max_delta: float,
        observed_prefix: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        required = rounds + 1
        observed = dict(observed_prefix)
        observed["required_values"] = required
        if len(values) < required:
            return False, observed
        recent = [float(value) for value in values[-required:]]
        deltas = [abs(current - previous) for previous, current in zip(recent, recent[1:])]
        observed.update({"recent_values": recent, "deltas": deltas, "max_observed_delta": max(deltas)})
        return max(deltas) <= max_delta, observed

    def _snapshot_numeric_score(self, snapshot: Mapping[str, Any], key: str) -> Optional[float]:
        if not snapshot:
            return None
        value = snapshot.get(key)
        if value is None and key == "score_mean" and isinstance(snapshot.get("scores"), SequenceABC):
            raw_scores = [
                float(item)
                for item in snapshot["scores"]
                if not isinstance(item, bool) and isinstance(item, (int, float)) and math.isfinite(float(item))
            ]
            if raw_scores:
                value = sum(raw_scores) / len(raw_scores)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None

    def _label_distribution(self, labels: Sequence[Any]) -> Dict[str, float]:
        counts: Counter[str] = Counter()
        total = 0
        for label in labels:
            label_values = label if isinstance(label, (list, tuple, set)) else [label]
            for value in label_values:
                counts[self._label_key(value)] += 1
                total += 1
        if total == 0:
            return {}
        return {label: count / total for label, count in sorted(counts.items())}

    def _label_key(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, sort_keys=True)
        except TypeError:
            return str(value)

    def _distribution_l1_delta(self, left: Mapping[str, float], right: Mapping[str, float]) -> float:
        labels = set(left) | set(right)
        return sum(abs(float(left.get(label, 0.0)) - float(right.get(label, 0.0))) for label in labels)

    def _stop_minimums_met(self, criteria: StopCriteria, labeled_count: int, completed_round_count: int) -> bool:
        if criteria.min_labeled is not None and labeled_count < criteria.min_labeled:
            return False
        if criteria.min_rounds is not None and completed_round_count < criteria.min_rounds:
            return False
        return True

    def _write_stop_trace(
        self,
        criteria: StopCriteria,
        *,
        stopped: bool,
        reason: str,
        observed_values: Dict[str, Any],
        labeled_count: int,
        completed_round_count: int,
    ) -> None:
        assert self._state is not None
        self._state.scheduler_state["stop_trace"] = {
            "timestamp": time.time(),
            "stopped": stopped,
            "reason": reason,
            "criteria": dataclasses.asdict(criteria),
            "observed_values": observed_values,
            "labeled_count": labeled_count,
            "completed_round_count": completed_round_count,
        }
        self._append_audit_event(
            "stop.decision",
            metadata={
                "stopped": stopped,
                "reason": reason,
                "observed_values": observed_values,
                "labeled_count": labeled_count,
                "completed_round_count": completed_round_count,
            },
        )
        self._refresh_audit_artifact_refs()

    def _write_stop_criteria_reached_trace(self, criteria: StopCriteria, error: StopCriteriaReached) -> None:
        assert self._state is not None
        labeled_count = self._labeled_count()
        completed_round_count = self._completed_round_count()
        message = str(error)
        reason = "stop_criteria_reached"
        if "no unlabeled samples" in message.lower():
            reason = "no_unlabeled_samples"
        self._write_stop_trace(
            criteria,
            stopped=True,
            reason=reason,
            observed_values={"message": message},
            labeled_count=labeled_count,
            completed_round_count=completed_round_count,
        )
        self._touch_state()

    # ---------------------------------------------------------------------
    # Private helpers: round management
    # ---------------------------------------------------------------------

    def _get_or_create_active_round(self) -> RoundState:
        """
        Return the current in-progress round, or create a new one.

        Invariant:
        - The last round in `state.rounds` is considered the active one unless it is DONE/FAILED.

        Side effects:
        - May append a new `RoundState` and checkpoint state immediately.
        """
        assert self._state is not None
        self._assert_single_active_round()
        if self._state.rounds and self._state.rounds[-1].status not in {RoundStatus.DONE, RoundStatus.FAILED}:
            return self._state.rounds[-1]
        if not self._has_unlabeled_samples():
            raise StopCriteriaReached("No unlabeled samples left.")
        # Create a new round
        rid = self._new_round_id()
        now = time.time()
        r = RoundState(
            round_id=rid,
            status=RoundStatus.SELECTING,
            created_at=now,
            updated_at=now,
        )
        self._state.rounds.append(r)
        self._touch_state()
        self._append_audit_event(
            "round.created",
            round_id=rid,
            previous_status=None,
            new_status=r.status,
            metadata={"round_index": len(self._state.rounds)},
        )
        self._refresh_audit_artifact_refs()
        self._save_state()
        return r

    def _new_round_id(self) -> str:
        # Developer notes:
        # - Make it sortable and unique.
        return f"r{len(self._state.rounds)+1:04d}_{uuid.uuid4().hex[:8]}"  # type: ignore[union-attr]

    def _next_step(self, round_state: RoundState) -> StepKind:
        """Map a persisted `RoundStatus` to the next executable step."""
        if round_state.status == RoundStatus.SELECTING:
            return StepKind.SELECT
        if round_state.status == RoundStatus.SELECTED:
            return StepKind.PUSH
        if round_state.status in {RoundStatus.PUSHED, RoundStatus.WAITING}:
            return StepKind.WAIT
        if round_state.status == RoundStatus.READY_TO_PULL:
            return StepKind.PULL
        if round_state.status == RoundStatus.PULLED:
            return StepKind.TRAIN_EVAL
        if round_state.status == RoundStatus.TRAINED:
            return StepKind.UPDATE
        return StepKind.NOOP

    # ---------------------------------------------------------------------
    # Private helpers: step implementations
    # ---------------------------------------------------------------------

    def _step_select(self, round_state: RoundState, *, batch_size: int) -> None:
        """
        SELECT step: choose which unlabeled samples to send for labeling next.

        The selection decision is delegated to `StrategyScheduler`, using `SelectionContext`.
        The chosen sample IDs are persisted into `round_state.selected_sample_ids`.

        Idempotency:
        - If this round already has selected IDs and status is not SELECTING, this is a no-op.
        """
        assert self._state is not None
        assert self._provider is not None
        assert self._model is not None
        assert self._label_schema is not None
        assert self._scheduler is not None

        # Idempotency: if already selected, do nothing.
        if round_state.selected_sample_ids and round_state.status != RoundStatus.SELECTING:
            return

        pool_ids = self._unlabeled_train_ids()
        if not pool_ids:
            # No more unlabeled data; stop gracefully.
            raise StopCriteriaReached("No unlabeled samples left.")

        last_metrics = self._state.metrics_history[-1].metrics if self._state.metrics_history else {}
        labeled_ids = [sid for sid, st in self._state.sample_status.items() if st == SampleStatus.LABELED.value]

        context = SelectionContext(
            provider=self._provider,
            model=self._model,
            label_schema=self._label_schema,
            prediction_cache=self._prediction_cache,
            embedding_cache=self._embedding_cache,
            labeled_ids=labeled_ids,
            last_metrics=last_metrics,
            dataset_fingerprint=self._state.dataset_ref.fingerprint if self._state.dataset_ref is not None else None,
            model_id_override=self._cache_model_id_for_cache(),
        )

        selected, snapshot = self._scheduler.select_batch(pool_ids, batch_size, context, state=self._state.scheduler_state)
        if not selected:
            raise StrategyError("Scheduler returned an empty selection batch.")

        previous_status = round_state.status
        selection_audit = self._write_selection_audit_artifact(
            round_state,
            pool_ids=pool_ids,
            selected_ids=selected,
            scheduler_snapshot=snapshot,
        )
        round_state.selected_sample_ids = selected
        round_state.scheduler_snapshot = snapshot
        round_state.selection_audit = selection_audit
        round_state.status = RoundStatus.SELECTED
        round_state.updated_at = time.time()
        self._touch_state()
        self._append_audit_event(
            "round.select",
            round_id=round_state.round_id,
            previous_status=previous_status,
            new_status=round_state.status,
            metadata={
                "eligible_count": len(pool_ids),
                "selected_count": len(selected),
                "selection_audit": selection_audit,
                "scheduler_snapshot": snapshot,
            },
        )
        self._refresh_audit_artifact_refs()
        self._save_state()

    def _step_push(self, round_state: RoundState) -> None:
        """
        PUSH step: create labeling tasks in the backend.

        The backend returns `task_ids` which become the idempotency anchors for resume:
        they are persisted in `round_state.task_ids`.

        Idempotency:
        - If `round_state.task_ids` is already populated, the method must not push again.
        """
        assert self._state is not None
        assert self._provider is not None
        assert self._model is not None
        assert self._label_schema is not None

        if round_state.task_ids:
            self._validate_backend_task_ids(round_state, round_state.task_ids)
            # Already pushed (idempotent).
            if round_state.status != RoundStatus.PUSHED:
                idempotent_previous_status = round_state.status
                round_state.status = RoundStatus.PUSHED
                round_state.updated_at = time.time()
                self._touch_state()
                self._append_audit_event(
                    "round.push",
                    round_id=round_state.round_id,
                    previous_status=idempotent_previous_status,
                    new_status=round_state.status,
                    metadata={"idempotent": True, "task_count": len(round_state.task_ids)},
                )
                self._refresh_audit_artifact_refs()
                self._save_state()
            return

        if self._label_backend is None:
            raise ConfigurationError("Label backend is not initialized in this process.")

        # Build samples payload
        samples = _get_samples_from_provider(self._provider, round_state.selected_sample_ids)

        prelabels: Optional[Dict[str, Any]] = None
        if self._state.prelabel_config.get("enable", False):
            # Developer notes:
            # - Prelabels must be formatted per backend requirements.
            # - Store confidence/proba in state for analytics if needed.
            prelabels = self._make_prelabels(samples)

        try:
            res = self._label_backend.push_round(round_state.round_id, samples, prelabels=prelabels)
        except Exception as error:
            res = self._recover_backend_push(round_state, samples, prelabels=prelabels, error=error)
        if not isinstance(res, RoundPushResult):
            raise _backend_return_error(self._label_backend, "push_round", RoundPushResult, res, round_id=round_state.round_id)
        self._validate_backend_task_ids(round_state, res.task_ids)
        push_previous_status = round_state.status
        round_state.task_ids = dict(res.task_ids)
        round_state.backend_ref = self._backend_ref_summary(res.backend_round_ref, task_ids=res.task_ids)

        round_state.status = RoundStatus.PUSHED
        round_state.updated_at = time.time()
        self._touch_state()
        self._append_audit_event(
            "backend.push",
            round_id=round_state.round_id,
            metadata=round_state.backend_ref,
        )
        self._append_audit_event(
            "round.push",
            round_id=round_state.round_id,
            previous_status=push_previous_status,
            new_status=round_state.status,
            metadata={"task_count": len(round_state.task_ids)},
        )
        self._refresh_audit_artifact_refs()
        self._save_state()

    def _recover_backend_push(
        self,
        round_state: RoundState,
        samples: Sequence[DataSample],
        *,
        prelabels: Optional[Dict[str, Any]],
        error: Exception,
    ) -> RoundPushResult:
        if self._label_backend is None:
            if isinstance(error, LabelBackendError):
                raise error
            raise LabelBackendError(f"Label backend push_round failed: {type(error).__name__}: {error}") from error
        recover = getattr(self._label_backend, "recover_push_round", None)
        if not callable(recover):
            if isinstance(error, LabelBackendError):
                raise error
            raise _backend_lifecycle_error(self._label_backend, "push_round", error, round_id=round_state.round_id) from error
        try:
            recovered = recover(round_state.round_id, samples, prelabels=prelabels, error=error)
        except Exception as recovery_error:
            raise LabelBackendError(
                "Backend push failed and same-round idempotent recovery also failed: "
                f"push_error={type(error).__name__}: {error}; "
                f"recovery_error={type(recovery_error).__name__}: {recovery_error}"
            ) from error
        if not isinstance(recovered, RoundPushResult):
            raise ConfigurationError("Backend recover_push_round() must return RoundPushResult.")
        return recovered

    def _backend_ref_summary(self, backend_round_ref: Mapping[str, Any], *, task_ids: Mapping[str, Any]) -> Dict[str, Any]:
        summary = {
            "backend_round_ref": backend_round_ref,
            "task_count": len(task_ids),
            "task_ids": dict(task_ids),
        }
        safe = self._backend_audit_json_safe(summary)
        if not isinstance(safe, dict):
            return {"summary": safe}
        return safe

    def _poll_progress_summary(self, progress: RoundProgress) -> Dict[str, Any]:
        summary = {
            "total": progress.total,
            "done": progress.done,
            "ready_count": len(progress.ready_sample_ids),
            "ready_sample_ids": list(progress.ready_sample_ids),
            "details": progress.details,
            "updated_at": time.time(),
        }
        safe = self._backend_audit_json_safe(summary)
        if not isinstance(safe, dict):
            return {"summary": safe}
        return safe

    def _pull_payload_summary(self, pull: RoundPullResult) -> Dict[str, Any]:
        annotation_counts = {
            str(sample_id): len(records)
            for sample_id, records in pull.annotations.items()
        }
        summary = {
            "sample_count": len(pull.annotations),
            "annotation_count": sum(annotation_counts.values()),
            "annotation_counts": annotation_counts,
            "backend_payload": pull.backend_payload,
            "updated_at": time.time(),
        }
        safe = self._backend_audit_json_safe(summary)
        if not isinstance(safe, dict):
            return {"summary": safe}
        return safe

    def _record_backend_error(self, round_state: RoundState, *, operation: str, error: Exception) -> None:
        assert self._state is not None
        summary = {
            "operation": operation,
            "error_type": type(error).__name__,
            "message": self._redact_secret_text(str(error)),
            "timestamp": time.time(),
        }
        safe = self._backend_audit_json_safe(summary)
        if not isinstance(safe, dict):
            safe = {"summary": safe}
        round_state.backend_error_history.append(safe)
        round_state.backend_error_history = round_state.backend_error_history[-_BACKEND_AUDIT_HISTORY_LIMIT:]
        round_state.updated_at = time.time()
        self._touch_state()
        self._append_audit_event(
            "backend.error",
            round_id=round_state.round_id,
            metadata=safe,
        )
        self._refresh_audit_artifact_refs()
        self._save_state()

    def _backend_audit_json_safe(self, value: Any, *, depth: int = 0) -> Any:
        if depth > _BACKEND_AUDIT_MAX_DEPTH:
            return "<omitted>"
        if isinstance(value, Mapping):
            result: Dict[str, Any] = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= _BACKEND_AUDIT_MAX_ITEMS:
                    result["<truncated>"] = True
                    break
                key_str = str(key)
                if self._is_secret_key(key_str):
                    result[key_str] = "<redacted>"
                else:
                    result[key_str] = self._backend_audit_json_safe(item, depth=depth + 1)
            return result
        if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
            items = [
                self._backend_audit_json_safe(item, depth=depth + 1)
                for item in list(value)[:_BACKEND_AUDIT_MAX_ITEMS]
            ]
            if len(value) > _BACKEND_AUDIT_MAX_ITEMS:
                items.append("<truncated>")
            return items
        if isinstance(value, str):
            redacted = self._redact_secret_text(value)
            return redacted if len(redacted) <= _BACKEND_AUDIT_MAX_STRING else redacted[:_BACKEND_AUDIT_MAX_STRING] + "...<truncated>"
        return self._diagnostic_json_safe(value)

    def _is_secret_key(self, key: str) -> bool:
        lowered = key.lower()
        return any(marker in lowered for marker in _SECRET_KEY_MARKERS)

    def _redact_secret_text(self, text: str) -> str:
        redacted = text
        for pattern in _SECRET_TEXT_PATTERNS:
            redacted = pattern.sub("<redacted>", redacted)
        return redacted

    def _validate_backend_task_ids(self, round_state: RoundState, task_ids: Mapping[str, Any]) -> None:
        selected_ids = set(round_state.selected_sample_ids)
        returned_ids = set(task_ids.keys())
        non_string_task_ids = [
            sample_id
            for sample_id, task_id in task_ids.items()
            if not isinstance(task_id, str)
        ]
        task_id_values = [task_id for task_id in task_ids.values() if isinstance(task_id, str)]
        empty_task_ids = [sample_id for sample_id, task_id in task_ids.items() if isinstance(task_id, str) and task_id.strip() == ""]
        duplicate_task_ids = [
            task_id
            for task_id, count in Counter(task_id_values).items()
            if count > 1
        ]
        if (
            returned_ids == selected_ids
            and len(task_ids) == len(round_state.selected_sample_ids)
            and not non_string_task_ids
            and not empty_task_ids
            and not duplicate_task_ids
        ):
            return

        if non_string_task_ids or empty_task_ids or duplicate_task_ids:
            protocol_details: List[str] = []
            if non_string_task_ids:
                protocol_details.append(f"non-string backend task ids for sample ids: {non_string_task_ids}")
            if empty_task_ids:
                protocol_details.append(f"empty backend task ids for sample ids: {empty_task_ids}")
            if duplicate_task_ids:
                protocol_details.append(f"duplicate backend task ids: {duplicate_task_ids}")
            raise LabelBackendError("Backend task_ids violate the label-backend protocol; " + "; ".join(protocol_details))

        missing = sorted(selected_ids - returned_ids)
        extra = sorted(returned_ids - selected_ids)
        details: List[str] = []
        if missing:
            details.append(f"missing selected sample ids: {missing}")
        if extra:
            details.append(f"unexpected sample ids: {extra}")
        if len(selected_ids) != len(round_state.selected_sample_ids):
            details.append("selected sample ids contain duplicates")
        raise ConfigurationError(
            "Backend task_ids must exactly match selected sample ids; " + "; ".join(details)
        )

    def _restore_backend_round_payloads(self, round_state: RoundState) -> None:
        if self._label_backend is None or self._provider is None:
            return
        restore = getattr(self._label_backend, "restore_round_samples", None)
        if not callable(restore):
            return
        samples = _get_samples_from_provider(self._provider, round_state.selected_sample_ids)
        restore_call_style = self._restore_round_samples_call_style(restore)
        if restore_call_style == "positional_task_ids":
            restore(round_state.round_id, samples, round_state.task_ids)
        elif restore_call_style == "keyword_task_ids":
            restore(round_state.round_id, samples, task_ids=round_state.task_ids)
        else:
            restore(round_state.round_id, samples)

    @staticmethod
    def _restore_round_samples_call_style(restore: Any) -> str:
        try:
            signature = inspect.signature(restore)
        except (TypeError, ValueError):
            return "positional_task_ids"

        probe_round_id = "__round_id__"
        probe_samples: List[DataSample] = []
        probe_task_ids: Dict[str, str] = {}
        try:
            signature.bind(probe_round_id, probe_samples, probe_task_ids)
        except TypeError:
            pass
        else:
            return "positional_task_ids"

        try:
            signature.bind(probe_round_id, probe_samples, task_ids=probe_task_ids)
        except TypeError:
            pass
        else:
            return "keyword_task_ids"

        try:
            signature.bind(probe_round_id, probe_samples)
        except TypeError:
            return "positional_task_ids"
        return "legacy"

    def _step_wait(self, round_state: RoundState) -> RoundProgress:
        """
        WAIT step: poll the labeling backend until enough annotations are ready.

        The backend decides "readiness" using `AnnotationPolicy` (e.g. min_votes).
        When the backend reports everything ready, the round transitions to READY_TO_PULL.
        """
        assert self._state is not None
        if self._label_backend is None:
            raise ConfigurationError("Label backend is not initialized in this process.")
        if self._annotation_policy is None:
            raise ConfigurationError("Annotation policy is not initialized in this process.")
        self._validate_backend_task_ids(round_state, round_state.task_ids)
        self._restore_backend_round_payloads(round_state)

        # Transition to WAITING if needed
        if round_state.status == RoundStatus.PUSHED:
            now = time.time()
            previous_status = round_state.status
            round_state.status = RoundStatus.WAITING
            round_state.updated_at = now
            if self._annotation_timeout_enabled():
                self._ensure_annotation_wait_started(round_state, wait_started_at=now)
            self._touch_state()
            self._append_audit_event(
                "round.wait",
                round_id=round_state.round_id,
                previous_status=previous_status,
                new_status=round_state.status,
                metadata={"task_count": len(round_state.task_ids)},
            )
            self._refresh_audit_artifact_refs()
            self._save_state()
        elif self._annotation_timeout_enabled():
            if self._ensure_annotation_wait_started(round_state):
                self._touch_state()
                self._append_audit_event(
                    "round.wait_timeout_started",
                    round_id=round_state.round_id,
                    previous_status=round_state.status,
                    new_status=round_state.status,
                    metadata={"task_count": len(round_state.task_ids)},
                )
                self._refresh_audit_artifact_refs()
                self._save_state()

        try:
            progress = self._label_backend.poll_round(round_state.round_id, round_state.task_ids, self._annotation_policy)
        except LabelBackendError:
            raise
        except Exception as error:
            raise _backend_lifecycle_error(self._label_backend, "poll_round", error, round_id=round_state.round_id) from error
        if not isinstance(progress, RoundProgress):
            raise _backend_return_error(self._label_backend, "poll_round", RoundProgress, progress, round_id=round_state.round_id)
        self._validate_round_progress(round_state, progress)
        round_state.last_poll_progress = self._poll_progress_summary(progress)
        self._append_audit_event(
            "backend.poll",
            round_id=round_state.round_id,
            metadata=round_state.last_poll_progress,
        )
        if progress.done >= progress.total and progress.total > 0:
            ready_previous_status = round_state.status
            round_state.status = RoundStatus.READY_TO_PULL
            round_state.error = None
            round_state.updated_at = time.time()
            self._touch_state()
            self._append_audit_event(
                "round.pull_ready",
                round_id=round_state.round_id,
                previous_status=ready_previous_status,
                new_status=round_state.status,
                metadata={"progress": round_state.last_poll_progress},
            )
            self._refresh_audit_artifact_refs()
            self._save_state()
            return progress
        if self._annotation_wait_timed_out(round_state):
            return self._handle_annotation_timeout(round_state, progress)
        round_state.updated_at = time.time()
        self._touch_state()
        self._refresh_audit_artifact_refs()
        self._save_state()
        return progress

    def _validate_round_progress(self, round_state: RoundState, progress: RoundProgress) -> None:
        tracked_ids = set(round_state.task_ids.keys())
        if isinstance(progress.total, bool) or not isinstance(progress.total, int):
            raise ConfigurationError("Backend progress total must be an integer matching tracked task count.")
        if isinstance(progress.done, bool) or not isinstance(progress.done, int):
            raise ConfigurationError("Backend progress done must be an integer.")
        if progress.total != len(tracked_ids):
            raise ConfigurationError(
                f"Backend progress total {progress.total} does not match tracked task count {len(tracked_ids)}."
            )
        if progress.done < 0 or progress.done > progress.total:
            raise ConfigurationError(
                f"Backend progress done {progress.done} is outside valid bounds 0..{progress.total}."
            )

        ready_sample_ids = [str(sample_id) for sample_id in progress.ready_sample_ids]
        duplicate_ready_ids = sorted(sample_id for sample_id, count in Counter(ready_sample_ids).items() if count > 1)
        if duplicate_ready_ids:
            raise ConfigurationError(f"Backend progress ready_sample_ids contains duplicates: {duplicate_ready_ids[:10]}.")
        unknown_ready_ids = sorted(set(ready_sample_ids) - tracked_ids)
        if unknown_ready_ids:
            raise ConfigurationError(f"Backend progress ready_sample_ids contains unknown task sample ids: {unknown_ready_ids[:10]}.")
        if progress.done == progress.total and set(ready_sample_ids) != tracked_ids:
            raise ConfigurationError("Backend progress is complete but ready_sample_ids do not match tracked task sample ids.")

    def _ensure_annotation_wait_started(self, round_state: RoundState, *, wait_started_at: Optional[float] = None) -> bool:
        trace = self._annotation_timeout_trace(round_state)
        if "wait_started_at" in trace:
            return False
        trace["wait_started_at"] = float(wait_started_at if wait_started_at is not None else round_state.created_at)
        self._set_annotation_timeout_trace(round_state, trace)
        return True

    def _annotation_timeout_enabled(self) -> bool:
        assert self._annotation_policy is not None
        return self._annotation_policy.timeout_seconds is not None

    def _annotation_wait_timed_out(self, round_state: RoundState, *, now: Optional[float] = None) -> bool:
        assert self._annotation_policy is not None
        if self._annotation_policy.timeout_seconds is None:
            return False
        trace = self._annotation_timeout_trace(round_state)
        wait_started_at = float(trace.get("wait_started_at", round_state.created_at))
        return float(now if now is not None else time.time()) >= wait_started_at + self._annotation_policy.timeout_seconds

    def _handle_annotation_timeout(self, round_state: RoundState, progress: RoundProgress) -> RoundProgress:
        assert self._annotation_policy is not None
        mode = self._annotation_policy.on_timeout
        if mode == "raise":
            trace = self._write_annotation_timeout_trace(round_state, progress, action="raise")
            message = (
                f"Annotation wait timed out for round {round_state.round_id} after "
                f"{trace['elapsed_seconds']:.3f}s (timeout_seconds={self._annotation_policy.timeout_seconds})."
            )
            round_state.error = message
            round_state.updated_at = time.time()
            self._touch_state()
            self._save_state()
            raise ActiveLearningError(message)

        return self._finalize_annotation_timeout(round_state, progress, mode=mode)

    def _finalize_annotation_timeout(
        self,
        round_state: RoundState,
        progress: RoundProgress,
        *,
        mode: str,
    ) -> RoundProgress:
        assert self._state is not None
        assert self._annotation_policy is not None
        assert self._aggregator is not None
        assert self._label_schema is not None
        if self._label_backend is None:
            raise ConfigurationError("Label backend is not initialized in this process.")

        self._validate_backend_task_ids(round_state, round_state.task_ids)
        self._restore_backend_round_payloads(round_state)
        try:
            pull = self._label_backend.pull_round(round_state.round_id, round_state.task_ids)
        except LabelBackendError:
            raise
        except Exception as error:
            raise _backend_lifecycle_error(self._label_backend, "pull_round", error, round_id=round_state.round_id) from error
        if not isinstance(pull, RoundPullResult):
            raise _backend_return_error(self._label_backend, "pull_round", RoundPullResult, pull, round_id=round_state.round_id)
        round_state.pull_summary = self._pull_payload_summary(pull)
        tracked_sample_ids = list(round_state.task_ids.keys()) or list(round_state.selected_sample_ids)
        unknown_ids = sorted(set(pull.annotations.keys()) - set(tracked_sample_ids))
        if unknown_ids:
            raise ConfigurationError(f"Unknown annotation sample_id from backend: {unknown_ids[0]!r}")
        ready_sample_ids = set(progress.ready_sample_ids)
        accepted_sample_ids: List[str] = []
        needs_review_sample_ids: List[str] = []
        resolved: Dict[str, Any] = {}
        status_updates: Dict[str, str] = {}
        labels_to_clear: List[str] = []
        review_metadata_updates: Dict[str, Dict[str, Any]] = {}
        annotation_counts: Dict[str, int] = {}

        accept_latest_aggregator: Optional[AnnotationAggregator] = None
        accept_latest_policy: Optional[AnnotationPolicy] = None
        if mode == "accept_latest":
            accept_latest_policy = dataclasses.replace(
                self._annotation_policy,
                mode="latest",
                min_votes=1,
                allow_single_annotator=True,
            )
            accept_latest_aggregator = AnnotationAggregator(accept_latest_policy)

        for sample_id in tracked_sample_ids:
            if sample_id not in self._state.sample_status:
                raise ConfigurationError(f"Unknown annotation sample_id from backend: {sample_id!r}")
            annotations = list(pull.annotations.get(sample_id, []))
            annotation_counts[sample_id] = len(annotations)
            if mode == "needs_review" and sample_id not in ready_sample_ids:
                status_updates[sample_id] = SampleStatus.NEEDS_REVIEW.value
                labels_to_clear.append(sample_id)
                review_metadata_updates[sample_id] = self._build_review_metadata(
                    sample_id=sample_id,
                    status=SampleStatus.NEEDS_REVIEW,
                    annotations=annotations,
                    policy=self._annotation_policy,
                    round_id=round_state.round_id,
                    source="annotation_timeout",
                    reason="annotation_timeout_not_ready",
                    details={"timeout_action": mode},
                )
                needs_review_sample_ids.append(sample_id)
                continue

            aggregator = accept_latest_aggregator if accept_latest_aggregator is not None else self._aggregator
            resolution = aggregator.resolve(sample_id, annotations)
            if resolution.status == SampleStatus.LABELED:
                label = self._normalize_backend_label(resolution.label, sample_id)
                status_updates[sample_id] = SampleStatus.LABELED.value
                resolved[sample_id] = label
                accepted_sample_ids.append(sample_id)
            else:
                status_updates[sample_id] = SampleStatus.NEEDS_REVIEW.value
                labels_to_clear.append(sample_id)
                review_metadata_updates[sample_id] = self._build_review_metadata(
                    sample_id=sample_id,
                    status=SampleStatus.NEEDS_REVIEW,
                    annotations=annotations,
                    policy=accept_latest_policy or self._annotation_policy,
                    round_id=round_state.round_id,
                    source="annotation_timeout",
                    resolution=resolution,
                )
                needs_review_sample_ids.append(sample_id)

        for sample_id, status in status_updates.items():
            self._state.sample_status[sample_id] = status
        for sample_id in labels_to_clear:
            self._state.sample_labels.pop(sample_id, None)
        for sample_id in resolved:
            self._state.sample_review_metadata.pop(sample_id, None)
        self._state.sample_review_metadata.update(review_metadata_updates)
        self._state.sample_labels.update(resolved)
        previous_status = round_state.status
        round_state.resolved = resolved
        round_state.status = RoundStatus.PULLED if resolved else RoundStatus.DONE
        round_state.updated_at = time.time()
        trace = self._write_annotation_timeout_trace(
            round_state,
            progress,
            action=mode,
            accepted_sample_ids=accepted_sample_ids,
            needs_review_sample_ids=needs_review_sample_ids,
            annotation_counts=annotation_counts,
            backend_payload=pull.backend_payload,
        )
        self._touch_state()
        self._append_audit_event(
            "backend.pull",
            round_id=round_state.round_id,
            metadata=round_state.pull_summary,
        )
        self._append_audit_event(
            "round.pull",
            round_id=round_state.round_id,
            previous_status=previous_status,
            new_status=round_state.status,
            metadata={
                "resolved_count": len(resolved),
                "needs_review_count": len(needs_review_sample_ids),
                "timeout_action": mode,
            },
        )
        self._refresh_audit_artifact_refs()
        self._save_state()
        details = dict(progress.details)
        details[_ANNOTATION_TIMEOUT_TRACE_KEY] = trace
        return RoundProgress(
            total=progress.total,
            done=progress.total,
            ready_sample_ids=accepted_sample_ids,
            details=details,
        )

    def _annotation_timeout_trace(self, round_state: RoundState) -> Dict[str, Any]:
        raw = round_state.scheduler_snapshot.get(_ANNOTATION_TIMEOUT_TRACE_KEY)
        if isinstance(raw, dict):
            return dict(raw)
        return {}

    def _set_annotation_timeout_trace(self, round_state: RoundState, trace: Mapping[str, Any]) -> None:
        round_state.scheduler_snapshot[_ANNOTATION_TIMEOUT_TRACE_KEY] = dict(trace)

    def _write_annotation_timeout_trace(
        self,
        round_state: RoundState,
        progress: RoundProgress,
        *,
        action: str,
        accepted_sample_ids: Optional[Sequence[str]] = None,
        needs_review_sample_ids: Optional[Sequence[str]] = None,
        annotation_counts: Optional[Mapping[str, int]] = None,
        backend_payload: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        assert self._annotation_policy is not None
        now = time.time()
        trace = self._annotation_timeout_trace(round_state)
        wait_started_at = float(trace.get("wait_started_at", round_state.created_at))
        trace.update(
            {
                "timed_out": True,
                "action": action,
                "timestamp": now,
                "wait_started_at": wait_started_at,
                "timeout_seconds": self._annotation_policy.timeout_seconds,
                "elapsed_seconds": now - wait_started_at,
                "progress": dataclass_to_dict(progress),
            }
        )
        if accepted_sample_ids is not None:
            trace["accepted_sample_ids"] = list(accepted_sample_ids)
        if needs_review_sample_ids is not None:
            trace["needs_review_sample_ids"] = list(needs_review_sample_ids)
        if annotation_counts is not None:
            trace["annotation_counts"] = dict(annotation_counts)
        if backend_payload is not None:
            trace["backend_payload"] = dict(backend_payload)
        trace = self._diagnostic_json_safe(trace)
        if not isinstance(trace, dict):
            raise ConfigurationError("Annotation timeout trace could not be normalized to a JSON object.")
        self._set_annotation_timeout_trace(round_state, trace)
        return trace

    def _step_pull(self, round_state: RoundState) -> None:
        """
        PULL step: fetch annotations and resolve them into final labels.

        Flow:
        1. Backend returns raw annotations as `AnnotationRecord` objects.
        2. `AnnotationAggregator` resolves them into a single label per sample.
        3. Project state is updated: sample_status + sample_labels.
        """
        assert self._state is not None
        assert self._aggregator is not None
        assert self._label_schema is not None

        if self._label_backend is None:
            raise ConfigurationError("Label backend is not initialized in this process.")

        self._validate_backend_task_ids(round_state, round_state.task_ids)
        self._restore_backend_round_payloads(round_state)
        try:
            pull = self._label_backend.pull_round(round_state.round_id, round_state.task_ids)
        except LabelBackendError:
            raise
        except Exception as error:
            raise _backend_lifecycle_error(self._label_backend, "pull_round", error, round_id=round_state.round_id) from error
        if not isinstance(pull, RoundPullResult):
            raise _backend_return_error(self._label_backend, "pull_round", RoundPullResult, pull, round_id=round_state.round_id)
        round_state.pull_summary = self._pull_payload_summary(pull)
        resolved, sample_status_updates, labels_to_clear, review_metadata_updates = self._validate_pull_payload(round_state, pull.annotations)

        for sid, status in sample_status_updates.items():
            self._state.sample_status[sid] = status
        for sid in labels_to_clear:
            self._state.sample_labels.pop(sid, None)
        for sid in resolved:
            self._state.sample_review_metadata.pop(sid, None)
        self._state.sample_review_metadata.update(review_metadata_updates)
        self._state.sample_labels.update(resolved)

        previous_status = round_state.status
        round_state.resolved = resolved
        round_state.status = RoundStatus.PULLED if resolved else RoundStatus.DONE
        round_state.updated_at = time.time()
        self._touch_state()
        self._append_audit_event(
            "backend.pull",
            round_id=round_state.round_id,
            metadata=round_state.pull_summary,
        )
        self._append_audit_event(
            "round.pull",
            round_id=round_state.round_id,
            previous_status=previous_status,
            new_status=round_state.status,
            metadata={
                "resolved_count": len(resolved),
                "needs_review_count": sum(1 for status in sample_status_updates.values() if status == SampleStatus.NEEDS_REVIEW.value),
            },
        )
        self._refresh_audit_artifact_refs()
        self._save_state()

    def _validate_pull_payload(
        self,
        round_state: RoundState,
        annotations_by_sample: Mapping[str, Sequence[Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, str], List[str], Dict[str, Dict[str, Any]]]:
        assert self._state is not None
        assert self._aggregator is not None
        assert self._annotation_policy is not None

        expected_ids = set(round_state.task_ids.keys())
        returned_ids = set(annotations_by_sample.keys())
        unknown_ids = sorted(returned_ids - expected_ids)
        if unknown_ids:
            raise ConfigurationError(f"Unknown annotation sample_id from backend: {unknown_ids[0]!r}")

        missing_ids = sorted(expected_ids - returned_ids)
        empty_ids = sorted(sid for sid in expected_ids & returned_ids if not annotations_by_sample[sid])
        if missing_ids or empty_ids:
            affected = missing_ids + [sid for sid in empty_ids if sid not in missing_ids]
            raise ConfigurationError(f"Backend pull_round missing annotations for selected sample ids: {affected}")

        resolved: Dict[str, Any] = {}
        status_updates: Dict[str, str] = {}
        labels_to_clear: List[str] = []
        review_metadata_updates: Dict[str, Dict[str, Any]] = {}
        for sid in round_state.task_ids:
            if sid not in self._state.sample_status:
                raise ConfigurationError(f"Unknown annotation sample_id from backend: {sid!r}")

            annotations = list(annotations_by_sample[sid])
            resolution = self._aggregator.resolve(sid, annotations)
            if resolution.status == SampleStatus.LABELED:
                resolved[sid] = self._normalize_backend_label(resolution.label, sid)
                status_updates[sid] = SampleStatus.LABELED.value
            elif resolution.status == SampleStatus.NEEDS_REVIEW:
                status_updates[sid] = SampleStatus.NEEDS_REVIEW.value
                labels_to_clear.append(sid)
                review_metadata_updates[sid] = self._build_review_metadata(
                    sample_id=sid,
                    status=SampleStatus.NEEDS_REVIEW,
                    annotations=annotations,
                    policy=self._annotation_policy,
                    round_id=round_state.round_id,
                    source="annotation_pull",
                    resolution=resolution,
                )
            else:
                status_updates[sid] = SampleStatus.UNLABELED.value
                labels_to_clear.append(sid)
                review_metadata_updates[sid] = self._build_review_metadata(
                    sample_id=sid,
                    status=SampleStatus.UNLABELED,
                    annotations=annotations,
                    policy=self._annotation_policy,
                    round_id=round_state.round_id,
                    source="annotation_pull",
                    resolution=resolution,
                )

        return resolved, status_updates, labels_to_clear, review_metadata_updates

    def _step_seed_train_eval(self) -> Dict[str, Any]:
        """
        Initial TRAIN_EVAL before the first active SELECT.

        Seed labels come from public import_labels() and should warm the model
        without creating synthetic backend rounds or tasks.
        """
        metrics_before, metrics_after, train_count, val_count = self._train_eval_labeled_data(metric_step="seed_eval")
        self._seed_train_completed_in_runtime = True
        self._mark_seed_train_completed()
        self._touch_state()
        self._append_audit_event(
            "round.train",
            metadata={
                "seed": True,
                "metrics_before": metrics_before,
                "metrics_after": metrics_after,
                "train_count": train_count,
                "val_count": val_count,
            },
        )
        self._refresh_audit_artifact_refs()
        self._save_state()
        return {
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "train_count": train_count,
            "val_count": val_count,
        }

    def _step_train_eval(self, round_state: RoundState) -> None:
        """
        TRAIN_EVAL step: train the model on labeled training data and evaluate on validation set.

        Notes:
        - The engine expects the model adapter to implement `fit()` and `evaluate()`.
        - Metrics are appended to `state.metrics_history`.
        """
        assert self._state is not None
        metrics_before, metrics_after, _, _ = self._train_eval_labeled_data(metric_step="eval")
        round_state.metrics_before = metrics_before
        round_state.metrics_after = metrics_after

        previous_status = round_state.status
        round_state.status = RoundStatus.TRAINED
        round_state.updated_at = time.time()
        self._touch_state()
        self._append_audit_event(
            "round.train",
            round_id=round_state.round_id,
            previous_status=previous_status,
            new_status=round_state.status,
            metadata={
                "metrics_before": metrics_before,
                "metrics_after": metrics_after,
            },
        )
        self._refresh_audit_artifact_refs()
        self._save_state()

    def _train_eval_labeled_data(self, *, metric_step: str) -> Tuple[Dict[str, float], Dict[str, float], int, int]:
        assert self._state is not None
        assert self._provider is not None
        assert self._model is not None

        labeled_train = self._labeled_train_ids()
        val_ids = self._state.splits.get("val", [])
        labeled_val = [sid for sid in val_ids if self._state.sample_status.get(sid) == SampleStatus.LABELED.value]

        if not labeled_train:
            raise ActiveLearningError("No labeled training samples available. Cannot train.")

        train_samples = _get_samples_from_provider(self._provider, labeled_train)
        train_texts = self._text_rows_for_samples(train_samples, source="training")
        train_labels = [self._state.sample_labels[s.sample_id] for s in train_samples]

        metrics_before = dict(self._state.metrics_history[-1].metrics) if self._state.metrics_history else {}
        model_id_before_fit = self._resolve_model_id_for_cache()

        try:
            self._model.fit(train_texts, train_labels)
        except Exception as e:
            raise ModelAdapterError(f"model.fit failed: {e}") from e

        if labeled_val:
            val_samples = _get_samples_from_provider(self._provider, labeled_val)
            val_texts = self._text_rows_for_samples(val_samples, source="validation")
            val_labels = [self._state.sample_labels[s.sample_id] for s in val_samples]
            try:
                raw_metrics = self._model.evaluate(val_texts, val_labels)
            except Exception as e:
                raise ModelAdapterError(f"model.evaluate failed: {e}") from e
            metrics_after = self._validate_model_metrics(raw_metrics, source="model.evaluate")
        else:
            metrics_after = {}

        self._state.metrics_history.append(MetricRecord(step=metric_step, created_at=time.time(), metrics=dict(metrics_after)))

        # Training changes the model; caches may need invalidation depending on get_model_id.
        # If the adapter does not expose a usable versioned model id, clear caches
        # so later rounds cannot reuse stale predictions or embeddings.
        model_id_after_fit = self._resolve_model_id_for_cache()
        if model_id_before_fit is None or model_id_after_fit is None:
            self.clear_cache(kind="all", reason="automatic_model_id_unstable")
        elif model_id_before_fit == model_id_after_fit:
            self._advance_model_cache_epoch()
            self._record_cache_invalidation(kind="all", reason="automatic_model_id_epoch_advanced")
            self._append_audit_event(
                "cache.invalidated",
                metadata={
                    "kind": "all",
                    "reason": "automatic_model_id_epoch_advanced",
                    "model_id": model_id_after_fit,
                    "cache_epoch": self._model_cache_epoch(),
                    "cache_stats": self.cache_stats(),
                },
            )

        return metrics_before, metrics_after, len(labeled_train), len(labeled_val)

    def _validate_model_metrics(self, metrics: Any, *, source: str) -> Dict[str, float]:
        if not isinstance(metrics, Mapping):
            raise ModelAdapterError(f"{source} must return a mapping of metric name to finite numeric value.")

        validated: Dict[str, float] = {}
        for metric_name, raw_value in metrics.items():
            if not isinstance(metric_name, str) or not metric_name:
                raise ModelAdapterError(f"{source} metric names must be non-empty strings.")
            if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                raise ModelAdapterError(
                    f"{source} metric {metric_name!r} must be a finite numeric value, "
                    f"got {type(raw_value).__name__}."
                )
            value = float(raw_value)
            if not math.isfinite(value):
                raise ModelAdapterError(f"{source} metric {metric_name!r} must be finite.")
            validated[metric_name] = value
        return validated

    def _step_update(self, round_state: RoundState) -> None:
        """
        UPDATE step: compute reward and update scheduler state (bandit mode).

        In MVP this is mostly a hook for future scheduling algorithms.
        The round is marked DONE at the end.
        """
        assert self._state is not None
        assert self._scheduler is not None

        # Compute reward if possible
        reward = self._compute_reward(round_state)
        round_state.reward = reward

        # Update scheduler state (bandit)
        self._state.scheduler_state = self._scheduler.update_reward(
            reward=reward,
            snapshot=round_state.scheduler_snapshot,
            state=self._state.scheduler_state,
        )

        previous_status = round_state.status
        round_state.status = RoundStatus.DONE
        round_state.updated_at = time.time()
        self._touch_state()
        self._append_audit_event(
            "round.update",
            round_id=round_state.round_id,
            previous_status=previous_status,
            new_status=round_state.status,
            metadata={"reward": reward, "scheduler_state": self._state.scheduler_state},
        )
        self._refresh_audit_artifact_refs()
        self._save_state()

    # ---------------------------------------------------------------------
    # Private helpers: prelabeling and reward
    # ---------------------------------------------------------------------

    def _make_prelabels(self, samples: Sequence[DataSample]) -> Dict[str, Any]:
        assert self._model is not None
        assert self._state is not None
        texts = self._text_rows_for_samples(samples, source="prelabel")
        sample_ids = [sample.sample_id for sample in samples]
        try:
            proba = self._model.predict_proba(texts)
        except Exception as error:
            raise ModelAdapterError(f"model.predict_proba failed: {error}") from error
        proba_rows = _coerce_predict_proba_rows(proba, sample_ids)
        min_confidence = float(self._state.prelabel_config.get("min_confidence", 0.0))
        prelabels: Dict[str, Any] = {}
        for sample, row in zip(samples, proba_rows):
            confidence = max(self._prelabel_probability_values(row, sample.sample_id))
            if confidence >= min_confidence:
                prelabels[sample.sample_id] = row
        return prelabels

    def _prelabel_probability_values(self, row: Any, sample_id: str) -> List[float]:
        if isinstance(row, (str, bytes)):
            raise ConfigurationError(
                f"model.predict_proba row for sample {sample_id!r} must be a sequence of probabilities."
            )
        try:
            values = list(row)
        except TypeError as exc:
            raise ConfigurationError(
                f"model.predict_proba row for sample {sample_id!r} must be a sequence."
            ) from exc
        if not values:
            raise ConfigurationError(f"model.predict_proba row for sample {sample_id!r} must not be empty.")
        if len(values) < 2:
            raise ConfigurationError(
                f"model.predict_proba row for sample {sample_id!r} must have at least 2 probability columns; "
                "label_schema probabilities require a full class distribution."
            )
        label_width = len(self._label_schema.labels) if self._label_schema is not None else None
        if label_width is not None and len(values) != label_width:
            raise ConfigurationError(
                f"model.predict_proba row for sample {sample_id!r} has width {len(values)}; "
                f"label_schema.labels defines {label_width} labels."
            )

        probabilities: List[float] = []
        for column_index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, (int, float, Real)):
                raise ConfigurationError(
                    f"model.predict_proba row, column {column_index} for sample {sample_id!r} must be numeric."
                )
            probability = float(value)
            if not math.isfinite(probability):
                raise ConfigurationError(
                    f"model.predict_proba row, column {column_index} for sample {sample_id!r} must be finite."
                )
            if probability < 0:
                raise ConfigurationError(
                    f"model.predict_proba row, column {column_index} for sample {sample_id!r} must be non-negative."
                )
            probabilities.append(probability)
        row_sum = sum(probabilities)
        if row_sum <= 0:
            raise ConfigurationError(f"model.predict_proba row for sample {sample_id!r} must have a positive sum.")
        if not math.isclose(row_sum, 1.0, rel_tol=1e-9, abs_tol=1e-12):
            raise ConfigurationError(
                f"model.predict_proba row for sample {sample_id!r} must sum to 1.0; got {row_sum}."
            )
        return probabilities

    def _compute_reward(self, round_state: RoundState) -> float:
        assert self._state is not None
        # Developer notes:
        # - Default reward: delta of configured metric (new - old)
        # - Clamp/normalize to stabilize bandit updates.
        metric_name = (self._state.scheduler_config or {}).get("reward_metric", "accuracy")
        before = float(round_state.metrics_before.get(metric_name, 0.0))
        after = float(round_state.metrics_after.get(metric_name, 0.0))
        reward = after - before
        # Simple clamp
        if reward > 1.0:
            reward = 1.0
        if reward < -1.0:
            reward = -1.0
        return reward
