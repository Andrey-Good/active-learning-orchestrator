from __future__ import annotations

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

import dataclasses
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .adapters.base import ModelCapabilities, TextClassificationAdapter, inspect_model_capabilities
from .annotation import AnnotationAggregator
from .backends.base import LabelBackend, RoundProgress, build_label_backend
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
    ModelAdapterError,
    StateCorruptedError,
    StopCriteriaReached,
    StrategyError,
)
from .report import ReportGenerator
from .state.lock import ProjectLock
from .state.store import DatasetRef, JsonFileStateStore, ProjectState, RoundState, StateStore
from .strategies.base import SamplingStrategy
from .strategies.uncertainty import (
    EntropyStrategy,
    KCenterGreedyStrategy,
    LeastConfidenceStrategy,
    MarginStrategy,
    RandomStrategy,
)
from .types import DataSample, MetricRecord, RoundStatus, SampleStatus, StepKind
from .utils import dataclass_to_dict

try:
    import pandas as _pd  # type: ignore
except Exception:
    _pd = None  # type: ignore

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
    ) -> None:
        self.provider = provider
        self.model = model
        self.label_schema = label_schema
        self.prediction_cache = prediction_cache
        self.embedding_cache = embedding_cache
        self.labeled_ids = list(labeled_ids)
        self.last_metrics = dict(last_metrics)

    def model_id(self) -> str:
        if callable(getattr(self.model, "get_model_id", None)):
            try:
                return str(self.model.get_model_id())  # type: ignore[attr-defined]
            except Exception as e:
                raise ModelAdapterError(f"model.get_model_id() failed: {e}") from e
        # Fallback: model id unknown; treat as volatile.
        return "unknown"

    def get_samples(self, sample_ids: Sequence[str]) -> List[DataSample]:
        return self.provider.get_samples(sample_ids)

    def get_texts(self, sample_ids: Sequence[str]) -> List[str]:
        samples = self.get_samples(sample_ids)
        return [str(s.data.get("text", "")) for s in samples]

    def predict_proba(self, sample_ids: Sequence[str], batch_size: int = 32) -> Any:
        mid = self.model_id()
        if self.prediction_cache is None:
            return self._predict_proba_no_cache(sample_ids, batch_size=batch_size)

        # Developer notes:
        # - For speed, batch compute missing items only.
        missing: List[str] = []
        out: Dict[str, Any] = {}
        for sid in sample_ids:
            cached = self.prediction_cache.get(mid, sid)
            if cached is None:
                missing.append(sid)
            else:
                out[sid] = cached

        if missing:
            texts = self.get_texts(missing)
            try:
                proba = self.model.predict_proba(texts, batch_size=batch_size)
            except Exception as e:
                raise ModelAdapterError(f"model.predict_proba failed: {e}") from e

            # We don't assume numpy; proba can be list-like. Map by order.
            for sid, p in zip(missing, list(proba)):
                self.prediction_cache.set(mid, sid, p)
                out[sid] = p

        # Preserve input order
        return [out[sid] for sid in sample_ids]

    def embed(self, sample_ids: Sequence[str], batch_size: int = 32) -> Any:
        if not callable(getattr(self.model, "embed", None)):
            raise ConfigurationError("Model does not support embeddings (embed method missing).")

        mid = self.model_id()
        if self.embedding_cache is None:
            return self._embed_no_cache(sample_ids, batch_size=batch_size)

        missing: List[str] = []
        out: Dict[str, Any] = {}
        for sid in sample_ids:
            cached = self.embedding_cache.get(mid, sid)
            if cached is None:
                missing.append(sid)
            else:
                out[sid] = cached

        if missing:
            texts = self.get_texts(missing)
            try:
                embs = self.model.embed(texts, batch_size=batch_size)  # type: ignore[attr-defined]
            except Exception as e:
                raise ModelAdapterError(f"model.embed failed: {e}") from e
            for sid, emb in zip(missing, list(embs)):
                self.embedding_cache.set(mid, sid, emb)
                out[sid] = emb

        return [out[sid] for sid in sample_ids]

    def _predict_proba_no_cache(self, sample_ids: Sequence[str], batch_size: int) -> Any:
        texts = self.get_texts(sample_ids)
        try:
            return self.model.predict_proba(texts, batch_size=batch_size)
        except Exception as e:
            raise ModelAdapterError(f"model.predict_proba failed: {e}") from e

    def _embed_no_cache(self, sample_ids: Sequence[str], batch_size: int) -> Any:
        texts = self.get_texts(sample_ids)
        try:
            return self.model.embed(texts, batch_size=batch_size)  # type: ignore[attr-defined]
        except Exception as e:
            raise ModelAdapterError(f"model.embed failed: {e}") from e


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

        if self.config.mode == "custom":
            assert self.config.custom_selector is not None
            selected = self.config.custom_selector(context, k)
            return self._dedup_and_clip(selected, k), {"mode": "custom"}

        if self.config.mode == "single":
            strat = self._get_strategy(self.config.strategy)
            selected = strat.select(pool_ids, k, context)
            return self._dedup_and_clip(selected, k), {"mode": "single", "strategy": strat.name}

        if self.config.mode == "mix":
            if not self.config.mix:
                raise ConfigurationError("mix config missing")
            selected: List[str] = []
            weights = dict(self.config.mix)
            total = sum(weights.values())
            # Normalize weights
            weights = {k: v / total for k, v in weights.items() if v > 0}
            remaining = k
            snapshot = {"mode": "mix", "mix": dict(weights)}

            # Allocate counts by weight (deterministic rounding).
            alloc: Dict[str, int] = {}
            for name, w in weights.items():
                alloc[name] = int(w * k)
            # Fix rounding deficit.
            deficit = k - sum(alloc.values())
            # Distribute deficit deterministically by sorted name.
            for name in sorted(weights.keys()):
                if deficit <= 0:
                    break
                alloc[name] += 1
                deficit -= 1

            for name in sorted(alloc.keys()):
                if remaining <= 0:
                    break
                part_k = min(remaining, alloc[name])
                if part_k <= 0:
                    continue
                strat = self._get_strategy(name)
                part = strat.select(pool_ids, part_k, context)
                selected.extend(part)
                remaining = k - len(set(selected))

            # Fallback if strategies returned too few
            if len(set(selected)) < k:
                fallback = self._get_strategy("random") if "random" in self._strategies else RandomStrategy()
                selected.extend(fallback.select(pool_ids, k - len(set(selected)), context))

            return self._dedup_and_clip(selected, k), snapshot

        if self.config.mode == "bandit":
            # Placeholder: implement bandit arm selection (e.g., UCB1).
            # Store arm stats in `state`, update via update_reward().
            arms = self.config.bandit_arms or []
            if not arms:
                raise ConfigurationError("bandit_arms is empty")
            chosen = self._choose_bandit_arm(arms, state)
            strat = self._get_strategy(chosen)
            selected = strat.select(pool_ids, k, context)
            snapshot = {"mode": "bandit", "algo": self.config.bandit_algo, "chosen_arm": chosen, "arms": list(arms)}
            return self._dedup_and_clip(selected, k), snapshot

        raise ConfigurationError(f"Unsupported scheduler mode: {self.config.mode}")

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
        # Allow built-ins if not registered explicitly
        if name == "random":
            return RandomStrategy()
        if name == "entropy":
            return EntropyStrategy()
        if name == "least_confidence":
            return LeastConfidenceStrategy()
        if name == "margin":
            return MarginStrategy()
        if name == "coreset_kcenter":
            return KCenterGreedyStrategy()
        raise ConfigurationError(f"Unknown strategy: {name!r}. Available: {self.available_strategies()}")

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

    def _choose_bandit_arm(self, arms: Sequence[str], state: Dict[str, Any]) -> str:
        # Developer notes:
        # - Implement UCB1/Thompson Sampling/etc.
        # - Deterministic tie-breaking is important for reproducibility.
        # Placeholder: choose the first arm.
        return list(arms)[0]


@dataclass(frozen=True)
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

    STATE_VERSION = 1

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

        if self._lock_enabled and self._lock is not None:
            self._lock.acquire()

        # Load existing state if present.
        if state_store is not None:
            try:
                self._state = self._state_store.load()
                self._validate_loaded_state_basic()
            except StateCorruptedError as error:
                # A fresh custom store (for example a new SQLite database) may not
                # have a saved ProjectState yet. Treat only that case as empty;
                # real corruption should still fail loudly.
                message = str(error)
                if "No project state found" not in message and "does not exist" not in message:
                    raise
                self._state = self._new_state()
        elif self._state_path.exists():
            self._state = self._state_store.load()
            self._validate_loaded_state_basic()
        else:
            self._state = self._new_state()

    # ---------------------------------------------------------------------
    # Context manager helpers
    # ---------------------------------------------------------------------

    def __enter__(self) -> "ActiveLearningEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

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

        label_schema.validate()
        annotation_policy.validate()
        scheduler_config.validate()
        cache_config  # validated implicitly
        fingerprint_config.validate()
        split_config.validate()
        prelabel_config.validate()
        label_backend_config.validate()

        provider = self._coerce_dataset(dataset)

        # Compute dataset fingerprint and validate against existing state.
        fingerprinter = DatasetFingerprinter(fingerprint_config)
        fp = fingerprinter.fingerprint(provider)

        if self._state and self._state.dataset_ref is not None:
            if self._state.dataset_ref.fingerprint != fp:
                raise DatasetMismatchError(
                    f"Dataset fingerprint mismatch. expected={self._state.dataset_ref.fingerprint} got={fp}"
                )

        # Validate model contract and scheduler/strategy compatibility.
        caps = inspect_model_capabilities(model)
        self._validate_model_capabilities(caps)

        # Initialize caches.
        self._init_caches(cache_config)

        # Initialize label backend and aggregator.
        # Developer notes:
        # - The backend can be injected directly (recommended for custom/LLM backends).
        # - Otherwise we build a backend from config (Label Studio supported in this scaffold).
        if label_backend is not None:
            backend = label_backend
        else:
            if label_backend_config.backend in {"llm", "custom"}:
                raise ConfigurationError(
                    "label_backend must be provided when label_backend_config.backend is 'llm' or 'custom'."
                )
            backend = build_label_backend(label_backend_config)
        backend.ensure_ready(label_schema)
        self._label_backend = backend

        self._provider = provider
        self._model = model
        self._label_schema = label_schema
        self._annotation_policy = annotation_policy
        self._aggregator = AnnotationAggregator(annotation_policy)

        # Scheduler with built-in strategies.
        scheduler = StrategyScheduler(
            scheduler_config,
            strategies=[RandomStrategy(), EntropyStrategy(), LeastConfidenceStrategy(), MarginStrategy(), KCenterGreedyStrategy()],
        )
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
        self._state.scheduler_config = dataclasses.asdict(scheduler_config)
        self._state.label_backend_config = dataclasses.asdict(label_backend_config)
        self._state.cache_config = dataclasses.asdict(cache_config)
        self._state.split_config = dataclasses.asdict(split_config)
        self._state.prelabel_config = dataclasses.asdict(prelabel_config)

        # Resolve and persist splits deterministically.
        self._state.splits = self._resolve_splits(provider, split_config)

        # Initialize sample statuses if first configure.
        if not self._state.sample_status:
            self._state.sample_status = {sid: SampleStatus.UNLABELED.value for sid in provider.iter_sample_ids()}

        self._touch_state()
        self._save_state()

    def attach_runtime(
        self,
        *,
        dataset: Union[DatasetProvider, Any, str, Path],
        model: TextClassificationAdapter,
        label_backend: Optional[LabelBackend] = None,
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
            the persisted label_backend_config (Label Studio supported in this scaffold).

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

        provider = self._coerce_dataset(dataset)
        fp_cfg = FingerprintConfig(**self._state.dataset_ref.fingerprint_config)
        fp = DatasetFingerprinter(fp_cfg).fingerprint(provider)
        if fp != self._state.dataset_ref.fingerprint:
            raise DatasetMismatchError(
                f"Dataset fingerprint mismatch. expected={self._state.dataset_ref.fingerprint} got={fp}"
            )

        # Bind runtime objects.
        self._provider = provider
        self._model = model
        self._label_schema = LabelSchema(**self._state.label_schema)
        self._annotation_policy = AnnotationPolicy(**self._state.annotation_policy)
        self._aggregator = AnnotationAggregator(self._annotation_policy)

        # Caches
        cc = CacheConfig(**self._state.cache_config) if self._state.cache_config else CacheConfig()
        self._init_caches(cc)

        # Scheduler
        sc = SchedulerConfig(**self._state.scheduler_config) if self._state.scheduler_config else SchedulerConfig()
        self._scheduler = StrategyScheduler(
            sc,
            strategies=[RandomStrategy(), EntropyStrategy(), LeastConfidenceStrategy(), MarginStrategy(), KCenterGreedyStrategy()],
        )

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
        backend.ensure_ready(self._label_schema)
        self._label_backend = backend

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
        self._scheduler.register_strategy(strategy)

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
        self._ensure_configured()

        if budget is not None:
            stop_criteria = dataclasses.replace(stop_criteria, max_labeled=budget)
        stop_criteria.validate()

        while True:
            # Enforce stop criteria before running steps.
            if self._should_stop(stop_criteria):
                break

            try:
                result = self.run_step(batch_size=batch_size, poll_interval_seconds=poll_interval_seconds)
            except StopCriteriaReached:
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
        self._ensure_configured()

        state = self._state  # after _ensure_configured(), state is loaded
        assert state is not None

        round_state = self._get_or_create_active_round()

        next_step = self._next_step(round_state)
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
        counts = {status.value: 0 for status in SampleStatus}
        for status_value in self._state.sample_status.values():
            counts[status_value] = counts.get(status_value, 0) + 1
        counts["training_ready"] = sum(
            1 for status_value in self._state.sample_status.values()
            if status_value in SampleStatus.training_ready_values()
        )

        last_round = self._state.rounds[-1] if self._state.rounds else None
        last_metrics = self._state.metrics_history[-1].metrics if self._state.metrics_history else {}

        return {
            "project_name": self._state.project_name,
            "counts": counts,
            "active_round": dataclass_to_dict(last_round) if last_round else None,
            "last_metrics": dict(last_metrics),
            "state_version": self._state.state_version,
            "updated_at": self._state.updated_at,
        }

    def get_state(self) -> ProjectState:
        """
        Return the full ProjectState object.

        The returned object is a live snapshot; modifying it directly is unsupported.
        Use public methods to change project state.
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
        raise KeyError(f"Unknown round_id={round_id!r}")

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

        if self._state.dataset_ref is None:
            report["ok"] = False
            report["issues"].append("dataset_ref is missing (project not configured).")

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

        return report

    def generate_report(self, output_path: Union[str, Path] = "report.html") -> None:
        """
        Generate an HTML report for the project.

        The report typically includes:
        - metrics over time vs labeled sample count
        - class distribution
        - hardest examples (optional)

        This method may require optional dependencies (e.g., jinja2, matplotlib).
        """
        self._ensure_state_loaded()
        assert self._state is not None
        if self._reporter is None:
            self._reporter = ReportGenerator()
        self._reporter.generate_html(self._state, self.workdir / output_path)

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
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        labeled_items = [
            {"sample_id": sid, "label": self._state.sample_labels.get(sid)}
            for sid, st in self._state.sample_status.items()
            if st in SampleStatus.training_ready_values()
        ]

        if format == "jsonl":
            with out_path.open("w", encoding="utf-8") as f:
                for rec in labeled_items:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return

        if format == "csv":
            if _pd is None:
                raise ConfigurationError("CSV export requires pandas.")
            df = _pd.DataFrame(labeled_items)
            df.to_csv(out_path, index=False)
            return

        raise ConfigurationError(f"Unsupported export format: {format!r}")

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

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def match(status: str) -> bool:
            if which == "all":
                return True
            if which == SampleStatus.LABELED.value:
                return status in SampleStatus.training_ready_values()
            return status == which

        selected_ids = [sid for sid, st in self._state.sample_status.items() if match(st)]

        samples = self._provider.get_samples(selected_ids)
        records = [{"sample_id": s.sample_id, **s.data, **({"meta": s.meta} if s.meta else {})} for s in samples]

        out_path = out_dir / f"{which}.{format}"
        if format == "jsonl":
            with out_path.open("w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return
        if format == "csv":
            if _pd is None:
                raise ConfigurationError("CSV export requires pandas.")
            df = _pd.DataFrame(records)
            df.to_csv(out_path, index=False)
            return
        raise ConfigurationError(f"Unsupported export format: {format!r}")

    def cache_stats(self) -> Dict[str, Any]:
        """
        Return cache statistics.

        Returns separate stats for prediction and embedding caches.
        """
        pred = self._prediction_cache.stats() if self._prediction_cache else {"enabled": False}
        emb = self._embedding_cache.stats() if self._embedding_cache else {"enabled": False}
        return {"prediction_cache": pred, "embedding_cache": emb}

    def clear_cache(self, *, kind: str = "all") -> None:
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
            self._prediction_cache.clear()
        if kind in {"embeddings", "all"} and self._embedding_cache:
            self._embedding_cache.clear()

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
        if self._state.state_version != self.STATE_VERSION:
            # Developer notes:
            # - In production, implement state migrations.
            raise StateCorruptedError(
                f"Unsupported state_version: {self._state.state_version}. Expected {self.STATE_VERSION}."
            )

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
        # - Support file paths (CSV/Parquet) when pandas is available.
        if isinstance(dataset, DatasetProvider):  # type: ignore[arg-type]
            return dataset
        if _pd is not None and hasattr(dataset, "columns"):
            return DataFrameDatasetProvider(dataset)
        if isinstance(dataset, (str, Path)):
            path = Path(dataset)
            if _pd is None:
                raise ConfigurationError("Reading dataset from path requires pandas.")
            if not path.exists():
                raise ConfigurationError(f"Dataset path does not exist: {path}")
            if path.suffix.lower() == ".csv":
                return DataFrameDatasetProvider.from_path(path)
            if path.suffix.lower() in {".parquet", ".pq"}:
                return DataFrameDatasetProvider.from_path(path)
            raise ConfigurationError(f"Unsupported dataset file type: {path.suffix}")
        raise ConfigurationError("Unsupported dataset type. Provide a DatasetProvider, DataFrame, or file path.")

    def _infer_source_type(self, dataset: Any) -> str:
        if isinstance(dataset, (str, Path)):
            return "path"
        if _pd is not None and hasattr(dataset, "columns"):
            return "dataframe"
        return "provider"

    def _validate_model_capabilities(self, caps: ModelCapabilities) -> None:
        if not (caps.predict_proba and caps.fit and caps.evaluate):
            raise ConfigurationError(
                "Model adapter must implement predict_proba(), fit(), and evaluate() for MVP."
            )

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

    def _resolve_splits(self, provider: DatasetProvider, split_config: SplitConfig) -> Dict[str, List[str]]:
        """
        Create train/val/test splits and return explicit sample_id lists.

        The engine stores the resulting split IDs into project state so that:
        - resume is deterministic
        - train/val sets stay the same across process restarts
        """
        # Developer notes:
        # - Persist explicit IDs for determinism.
        # - For extremely large datasets, consider storing split rules + seed instead.
        ids = list(provider.iter_sample_ids())
        if split_config.mode == "explicit":
            assert split_config.explicit_splits is not None
            return {k: list(v) for k, v in split_config.explicit_splits.items()}
        if split_config.mode == "column":
            # Column-based split requires DataFrame provider or custom provider exposing split values.
            # This scaffold does not implement it.
            raise NotImplementedError("SplitConfig.mode='column' is not implemented in the scaffold.")
        # random split
        import random
        rng = random.Random(split_config.seed)
        ids_shuffled = list(ids)
        rng.shuffle(ids_shuffled)

        n = len(ids_shuffled)
        n_train = int(n * split_config.train_ratio)
        n_val = int(n * split_config.val_ratio)
        train = ids_shuffled[:n_train]
        val = ids_shuffled[n_train:n_train + n_val]
        test = ids_shuffled[n_train + n_val:]
        return {"train": train, "val": val, "test": test}

    def _should_stop(self, criteria: StopCriteria) -> bool:
        assert self._state is not None
        labeled = sum(1 for s in self._state.sample_status.values() if s in SampleStatus.training_ready_values())
        if criteria.max_labeled is not None and labeled >= criteria.max_labeled:
            return True
        if criteria.max_rounds is not None and len(self._state.rounds) >= criteria.max_rounds:
            return True
        if criteria.plateau_rounds is not None:
            return self._metric_plateau(criteria.metric_name, criteria.plateau_rounds, criteria.min_improvement)
        return False

    def _metric_plateau(self, metric_name: str, plateau_rounds: int, min_improvement: float) -> bool:
        assert self._state is not None
        # Developer notes:
        # - Plateau is computed on metrics_history. Ensure each TRAIN_EVAL logs a record.
        vals: List[float] = []
        for rec in self._state.metrics_history:
            if metric_name in rec.metrics:
                vals.append(float(rec.metrics[metric_name]))
        if len(vals) < plateau_rounds + 1:
            return False
        recent = vals[-(plateau_rounds + 1):]
        best_before = max(recent[:-1])
        last = recent[-1]
        return (last - best_before) < min_improvement

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
        if self._state.rounds and self._state.rounds[-1].status not in {RoundStatus.DONE, RoundStatus.FAILED}:
            return self._state.rounds[-1]
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

        pool_ids = [sid for sid, st in self._state.sample_status.items() if st in SampleStatus.selectable_values()]
        if not pool_ids:
            # No more unlabeled data; stop gracefully.
            round_state.status = RoundStatus.DONE
            round_state.updated_at = time.time()
            self._touch_state()
            self._save_state()
            raise StopCriteriaReached("No unlabeled samples left.")

        last_metrics = self._state.metrics_history[-1].metrics if self._state.metrics_history else {}
        labeled_ids = [sid for sid, st in self._state.sample_status.items() if st in SampleStatus.training_ready_values()]

        context = SelectionContext(
            provider=self._provider,
            model=self._model,
            label_schema=self._label_schema,
            prediction_cache=self._prediction_cache,
            embedding_cache=self._embedding_cache,
            labeled_ids=labeled_ids,
            last_metrics=last_metrics,
        )

        selected, snapshot = self._scheduler.select_batch(pool_ids, batch_size, context, state=self._state.scheduler_state)
        if not selected:
            raise StrategyError("Scheduler returned an empty selection batch.")

        round_state.selected_sample_ids = selected
        round_state.scheduler_snapshot = snapshot
        for sample_id in selected:
            self._state.sample_status[sample_id] = SampleStatus.PENDING.value
        round_state.status = RoundStatus.SELECTED
        round_state.updated_at = time.time()
        self._touch_state()
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
            # Already pushed (idempotent). Keep selected samples in the PRD SENT state.
            for sample_id in round_state.task_ids.keys():
                self._state.sample_status[sample_id] = SampleStatus.SENT.value
            if round_state.status != RoundStatus.PUSHED:
                round_state.status = RoundStatus.PUSHED
                round_state.updated_at = time.time()
                self._touch_state()
                self._save_state()
            return

        if self._label_backend is None:
            raise ConfigurationError("Label backend is not initialized in this process.")

        # Build samples payload
        samples = self._provider.get_samples(round_state.selected_sample_ids)

        prelabels: Optional[Dict[str, Any]] = None
        if self._state.prelabel_config.get("enable", False):
            # Developer notes:
            # - Prelabels must be formatted per backend requirements.
            # - Store confidence/proba in state for analytics if needed.
            prelabels = self._make_prelabels(samples)

        res = self._label_backend.push_round(round_state.round_id, samples, prelabels=prelabels)
        round_state.task_ids = dict(res.task_ids)
        for sample_id in round_state.task_ids.keys():
            self._state.sample_status[sample_id] = SampleStatus.SENT.value

        round_state.status = RoundStatus.PUSHED
        round_state.updated_at = time.time()
        self._touch_state()
        self._save_state()

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

        # Transition to WAITING if needed
        if round_state.status == RoundStatus.PUSHED:
            round_state.status = RoundStatus.WAITING
            round_state.updated_at = time.time()
            self._touch_state()
            self._save_state()

        progress = self._label_backend.poll_round(round_state.round_id, round_state.task_ids, self._annotation_policy)
        if progress.done >= progress.total and progress.total > 0:
            round_state.status = RoundStatus.READY_TO_PULL
            round_state.updated_at = time.time()
            self._touch_state()
            self._save_state()
        return progress

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

        if self._label_backend is None:
            raise ConfigurationError("Label backend is not initialized in this process.")

        pull = self._label_backend.pull_round(round_state.round_id, round_state.task_ids)

        resolved: Dict[str, Any] = {}
        for sid, anns in pull.annotations.items():
            r = self._aggregator.resolve(sid, anns)
            if r.status == SampleStatus.LABELED:
                self._state.sample_status[sid] = SampleStatus.LABELED.value
                self._state.sample_labels[sid] = r.label
                resolved[sid] = r.label
            elif r.status == SampleStatus.NEEDS_REVIEW:
                self._state.sample_status[sid] = SampleStatus.NEEDS_REVIEW.value
            else:
                self._state.sample_status[sid] = SampleStatus.UNLABELED.value

        round_state.resolved = resolved
        round_state.status = RoundStatus.PULLED
        round_state.updated_at = time.time()
        self._touch_state()
        self._save_state()

    def _step_train_eval(self, round_state: RoundState) -> None:
        """
        TRAIN_EVAL step: train the model on labeled training data and evaluate on validation set.

        Notes:
        - The engine expects the model adapter to implement `fit()` and `evaluate()`.
        - Metrics are appended to `state.metrics_history`.
        """
        assert self._state is not None
        assert self._provider is not None
        assert self._model is not None

        # Collect training data from labeled samples in train split
        train_ids = self._state.splits.get("train", [])
        val_ids = self._state.splits.get("val", [])

        labeled_train = [sid for sid in train_ids if self._state.sample_status.get(sid) in SampleStatus.training_ready_values()]
        labeled_val = [sid for sid in val_ids if self._state.sample_status.get(sid) in SampleStatus.training_ready_values()]

        if not labeled_train:
            raise ActiveLearningError("No labeled training samples available. Cannot train.")

        train_samples = self._provider.get_samples(labeled_train)
        train_texts = [str(s.data.get("text", "")) for s in train_samples]
        train_labels = [self._state.sample_labels[s.sample_id] for s in train_samples]

        # Metrics before training
        last_metrics = self._state.metrics_history[-1].metrics if self._state.metrics_history else {}
        round_state.metrics_before = dict(last_metrics)

        try:
            self._model.fit(train_texts, train_labels)
        except Exception as e:
            raise ModelAdapterError(f"model.fit failed: {e}") from e

        # Evaluate
        if labeled_val:
            val_samples = self._provider.get_samples(labeled_val)
            val_texts = [str(s.data.get("text", "")) for s in val_samples]
            val_labels = [self._state.sample_labels[s.sample_id] for s in val_samples]
            try:
                metrics = self._model.evaluate(val_texts, val_labels)
            except Exception as e:
                raise ModelAdapterError(f"model.evaluate failed: {e}") from e
        else:
            metrics = {}

        round_state.metrics_after = dict(metrics)
        self._state.metrics_history.append(MetricRecord(step="eval", created_at=time.time(), metrics=dict(metrics)))

        # Training changes the model; caches may need invalidation depending on get_model_id.
        # Developer notes:
        # - If model_id changes, caches naturally segregate by key.
        # - If model_id is unknown, you may want to clear caches here to avoid stale reuse.
        if not callable(getattr(self._model, "get_model_id", None)):
            self.clear_cache(kind="all")

        for sample_id in round_state.resolved.keys():
            if self._state.sample_status.get(sample_id) == SampleStatus.LABELED.value:
                self._state.sample_status[sample_id] = SampleStatus.IN_TRAINING.value

        round_state.status = RoundStatus.TRAINED
        round_state.updated_at = time.time()
        self._touch_state()
        self._save_state()

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

        round_state.status = RoundStatus.DONE
        round_state.updated_at = time.time()
        self._touch_state()
        self._save_state()

    # ---------------------------------------------------------------------
    # Private helpers: prelabeling and reward
    # ---------------------------------------------------------------------

    def _make_prelabels(self, samples: Sequence[DataSample]) -> Dict[str, Any]:
        assert self._model is not None
        # Developer notes:
        # - Compute model predictions and format them to backend-specific predictions format.
        # - Store per-sample confidence if needed.
        texts = [str(s.data.get("text", "")) for s in samples]
        proba = self._model.predict_proba(texts)
        # Placeholder: return raw probabilities by sample_id
        return {s.sample_id: p for s, p in zip(samples, list(proba))}

    def _compute_reward(self, round_state: RoundState) -> float:
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
