from __future__ import annotations

"""
Configuration objects for the SDK.

These are small dataclasses that you pass into `ActiveLearningProject.configure(...)`.

Why dataclasses instead of dicts?
- They make the expected fields explicit.
- They are easier to validate.
- IDE autocompletion works better.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from .exceptions import ConfigurationError


def _has_xxhash() -> bool:
    """Return True if `xxhash` is importable (optional dependency)."""
    try:
        import xxhash  # noqa: F401

        return True
    except Exception:
        return False


@dataclass(frozen=True)
class LabelSchema:
    """
    Describes what the labeling task looks like.

    For the MVP we assume text classification, so:
    - `task` is typically "text_classification"
    - `labels` is the list of allowed class names
    - `multi_label` controls whether a sample can have more than one label

    Attributes:
        task (str):
            What kind of labeling task this is.
            Where: passed to backends in `LabelBackend.ensure_ready(...)`.
            What: a short string like "text_classification".
            Why: a backend may need to create a project/config depending on task type.
        labels (List[str]):
            List of allowed class names.
            Where: used by UI/backends and by your model adapter to interpret probabilities.
            What: ordered list like ["positive", "negative"].
            Why: makes the label space explicit and stable across runs.
        multi_label (bool):
            Whether one sample can have multiple labels (e.g. ["sports", "news"]).
            Where: used mainly by annotation aggregation and future multi-label support.
            What: False for single-label classification (MVP default).
            Why: affects how labels are normalized and validated.
    """
    task: str
    labels: List[str]
    multi_label: bool = False

    def validate(self) -> None:
        if not self.task:
            raise ConfigurationError("label_schema.task must be a non-empty string.")
        if not self.labels:
            raise ConfigurationError("label_schema.labels must be a non-empty list.")
        if len(set(self.labels)) != len(self.labels):
            raise ConfigurationError("label_schema.labels must not contain duplicates.")


@dataclass(frozen=True)
class AnnotationPolicy:
    """
    How to turn multiple annotations into one training label.

    Examples:
    - mode="latest": last annotation wins (simple and works well for MVP)
    - mode="majority": vote over annotators

    Notes for juniors:
    - This policy is applied in `AnnotationAggregator.resolve(...)`.
    - If you add new modes, update both the validator and aggregator.

    Attributes:
        mode (str):
            How to pick the final label when multiple annotations exist.
            Where: used in `AnnotationAggregator.resolve(...)`.
            What: one of {"latest", "first", "majority", "consensus"}.
            Why: you may want speed (latest) or quality (majority/consensus).
        min_votes (int):
            Minimum number of annotations required before a sample is considered "ready".
            Where: used in aggregator and backend polling logic.
            What: 1 means "one annotation is enough".
            Why: prevents training on too few / unreliable labels.
        min_agreement (float):
            Minimum agreement ratio for majority/consensus modes.
            Where: used when mode is "majority" or "consensus".
            What: number in [0, 1], e.g. 0.67 means 2 out of 3 annotators.
            Why: allows you to route uncertain items to review.
        allow_single_annotator (bool):
            Placeholder setting for future multi-annotator workflows.
            Where: not strongly enforced in the scaffold yet.
            What: True means one annotator is allowed to produce labels.
            Why: in some projects you may require multiple independent votes.
        timeout_seconds (Optional[int]):
            Optional time limit for waiting on annotations.
            Where: intended to be used by `WAIT` step / backend polling.
            What: seconds, e.g. 86400 for 24h.
            Why: prevents the pipeline from waiting forever.
        on_timeout (str):
            What to do when timeout is reached.
            Where: intended to be used by engine/backends.
            What: one of {"needs_review", "accept_latest", "raise"}.
            Why: defines a safe fallback for long-running labeling.
    """
    mode: str = "latest"
    min_votes: int = 1
    min_agreement: float = 0.67
    allow_single_annotator: bool = True
    timeout_seconds: Optional[int] = None
    on_timeout: str = "needs_review"

    def validate(self) -> None:
        if self.mode not in {"latest", "first", "majority", "consensus"}:
            raise ConfigurationError(f"Unsupported annotation_policy.mode={self.mode!r}")
        if self.min_votes < 1:
            raise ConfigurationError("annotation_policy.min_votes must be >= 1")
        if not (0.0 <= self.min_agreement <= 1.0):
            raise ConfigurationError("annotation_policy.min_agreement must be in [0, 1]")
        if self.on_timeout not in {"needs_review", "accept_latest", "raise"}:
            raise ConfigurationError(f"Unsupported annotation_policy.on_timeout={self.on_timeout!r}")


@dataclass(frozen=True)
class SchedulerConfig:
    """
    How to choose the next batch of samples.

    Common MVP setup:
    - mode="single", strategy="entropy"

    Advanced:
    - mode="mix": combine strategies with weights
    - mode="bandit": pick strategies based on reward (placeholder)
    - mode="custom": provide a Python callback

    Attributes:
        mode (str):
            Scheduler mode.
            Where: used inside `StrategyScheduler.select_batch(...)`.
            What: "single", "mix", "bandit", or "custom".
            Why: lets you start simple and later add more advanced scheduling.
        strategy (str):
            Strategy name when mode="single".
            Where: used to look up a strategy object by name.
            What: e.g. "entropy", "margin", "random".
            Why: selects the algorithm that picks the next samples.
        mix (Optional[Dict[str, float]]):
            Strategy weights when mode="mix".
            Where: used to allocate k samples across strategies.
            What: {"entropy": 0.7, "random": 0.3}.
            Why: mixing can stabilize selection early in training.
        bandit_arms (Optional[List[str]]):
            Available strategies when mode="bandit".
            Where: used by the bandit placeholder logic.
            What: list of strategy names.
            Why: bandits choose arms based on observed reward over time.
        bandit_algo (str):
            Bandit algorithm name (placeholder).
            Where: stored in snapshots/state for future implementation.
            What: e.g. "ucb1".
            Why: makes it explicit which algorithm is used.
        reward_metric (str):
            Which metric to use as "reward" after training.
            Where: used by engine in `_compute_reward`.
            What: "accuracy" by default.
            Why: bandit scheduling needs a numeric reward signal.
        strict_capabilities (bool):
            Placeholder: how strict to be about requiring model capabilities.
            Where: not strongly enforced in the scaffold yet.
            What: True means "fail fast" if a strategy needs unsupported model features.
            Why: avoids silently running a different strategy than you expected.
        custom_selector (Optional[Callable[[Any, int], List[str]]]):
            Custom selection function when mode="custom".
            Where: called by scheduler with `(context, k)`.
            What: returns a list of sample_ids.
            Why: lets you implement your own selection logic quickly.
    """
    mode: str = "single"
    strategy: str = "random"
    mix: Optional[Dict[str, float]] = None
    bandit_arms: Optional[List[str]] = None
    bandit_algo: str = "ucb1"
    reward_metric: str = "accuracy"
    strict_capabilities: bool = True
    custom_selector: Optional[Callable[[Any, int], List[str]]] = None

    def validate(self) -> None:
        if self.mode not in {"single", "mix", "bandit", "custom"}:
            raise ConfigurationError(f"Unsupported scheduler_config.mode={self.mode!r}")
        if self.mode == "single" and not self.strategy:
            raise ConfigurationError("scheduler_config.strategy must be set for mode='single'")
        if self.mode == "mix":
            if not self.mix:
                raise ConfigurationError("scheduler_config.mix must be provided for mode='mix'")
            if any(weight < 0 for weight in self.mix.values()):
                raise ConfigurationError("scheduler_config.mix weights must be >= 0")
            if sum(self.mix.values()) <= 0:
                raise ConfigurationError("scheduler_config.mix weights sum must be > 0")
        if self.mode == "bandit" and not self.bandit_arms:
            raise ConfigurationError("scheduler_config.bandit_arms must be provided for mode='bandit'")
        if self.mode == "custom" and self.custom_selector is None:
            raise ConfigurationError("scheduler_config.custom_selector must be provided for mode='custom'")


@dataclass(frozen=True)
class CacheConfig:
    """
    Prediction/embedding caching settings.

    Caching matters a lot in active learning because strategies may call the model
    many times. If you disable caching, selection rounds can get very slow.

    Attributes:
        enable (bool):
            Master on/off switch for caching.
            Where: used by engine in `_init_caches`.
            What: if False, caches are disabled entirely.
            Why: useful for debugging or tiny datasets.
        persist (bool):
            Whether to store caches on disk.
            Where: if True, engine uses `JsonlDiskCacheStore`.
            What: True means caches survive restarts.
            Why: saves time when you resume a project or rerun selection.
        max_items (Optional[int]):
            Optional size cap for in-memory cache.
            Where: used by `InMemoryCacheStore`.
            What: None means unlimited (within RAM).
            Why: prevents memory blowups in long sessions.
    """
    enable: bool = True
    persist: bool = True
    max_items: Optional[int] = None


@dataclass(frozen=True)
class FingerprintConfig:
    """
    Dataset fingerprint settings (safety feature).

    A fingerprint is a deterministic hash of the dataset. It is stored in state and
    checked on resume to prevent accidentally mixing different datasets in one project.

    Attributes:
        mode (str):
            Fingerprint strictness.
            Where: used in `DatasetFingerprinter.fingerprint`.
            What: "fast", "strict", or "file".
            Why: trade speed vs accuracy of change detection.
        text_prefix_chars (int):
            How many text characters to include in fast mode.
            Where: used in fast/file fingerprint modes.
            What: 100 by default.
            Why: faster than hashing full text for every sample.
        normalize_text (bool):
            Whether to normalize whitespace before hashing.
            Where: used by fingerprinter.
            What: True collapses whitespace so trivial formatting changes are ignored.
            Why: avoids false mismatches from spacing differences.
        hash_algo (str):
            Hash algorithm to use.
            Where: used by fingerprinter to create hasher.
            What: "blake2b" (default), "sha256", or "xxhash64" (optional dep).
            Why: allows choosing stdlib vs optional faster hashing.
    """
    mode: str = "fast"
    text_prefix_chars: int = 100
    normalize_text: bool = True
    hash_algo: str = "blake2b"

    def validate(self) -> None:
        if self.mode not in {"fast", "strict", "file"}:
            raise ConfigurationError(f"Unsupported fingerprint_config.mode={self.mode!r}")
        if self.text_prefix_chars < 0:
            raise ConfigurationError("fingerprint_config.text_prefix_chars must be >= 0")
        if self.hash_algo not in {"blake2b", "sha256", "xxhash64"}:
            raise ConfigurationError(f"Unsupported fingerprint_config.hash_algo={self.hash_algo!r}")
        if self.hash_algo == "xxhash64" and not _has_xxhash():
            raise ConfigurationError("xxhash is not installed. Install active-learning-sdk[xxhash] or use blake2b.")


@dataclass(frozen=True)
class SplitConfig:
    """
    How to create train/val/test splits.

    MVP uses a deterministic random split. The resulting split IDs are persisted into state,
    so later runs use the exact same split even if your Python process restarts.

    Attributes:
        mode (str):
            How the split is created.
            Where: used in engine `_resolve_splits`.
            What: "random" (MVP), "explicit", or "column" (not implemented in scaffold).
            Why: different teams have different splitting requirements.
        seed (int):
            Random seed for deterministic splitting.
            Where: used in random split mode.
            What: any integer.
            Why: ensures reproducible train/val/test partitions.
        train_ratio (float):
            Fraction of ids assigned to train split.
            Where: random split mode.
            What: defaults to 0.8.
            Why: controls training set size.
        val_ratio (float):
            Fraction of ids assigned to validation split.
            Where: random split mode.
            What: defaults to 0.2.
            Why: controls evaluation set size.
        test_ratio (float):
            Fraction of ids assigned to test split (not used by engine in MVP).
            Where: random split mode.
            What: defaults to 0.0.
            Why: reserved for future/test workflows.
        split_column (Optional[str]):
            Column name for column-based split (not implemented in scaffold).
            Where: placeholder for future.
            What: e.g. "split" column with values "train"/"val".
            Why: lets dataset itself define split.
        explicit_splits (Optional[Dict[str, List[str]]]):
            Exact split ids when mode="explicit".
            Where: used by engine to store splits into state.
            What: {"train": [...], "val": [...], "test": [...]}.
            Why: gives full control and avoids randomness.
    """
    mode: str = "random"
    seed: int = 1337
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    test_ratio: float = 0.0
    split_column: Optional[str] = None
    explicit_splits: Optional[Dict[str, List[str]]] = None

    def validate(self) -> None:
        if self.mode not in {"random", "column", "explicit"}:
            raise ConfigurationError(f"Unsupported split_config.mode={self.mode!r}")
        if self.mode == "random":
            total = self.train_ratio + self.val_ratio + self.test_ratio
            if abs(total - 1.0) > 1e-6:
                raise ConfigurationError("split_config ratios must sum to 1.0")
        if self.mode == "column" and not self.split_column:
            raise ConfigurationError("split_config.split_column must be set for mode='column'")
        if self.mode == "explicit" and not self.explicit_splits:
            raise ConfigurationError("split_config.explicit_splits must be provided for mode='explicit'")


@dataclass(frozen=True)
class LabelBackendConfig:
    """
    Settings for the labeling backend (Label Studio, LLM, custom).

    For Label Studio in external mode you usually set:
    - url (e.g. "http://localhost:8080")
    - api_token
    - optionally project_id

    Attributes:
        backend (str):
            Which backend implementation to use.
            Where: used by `build_label_backend`.
            What: "label_studio" (scaffold), "llm", or "custom".
            Why: allows supporting different labeling systems.
        mode (str):
            How the backend is run/connected.
            Where: used by backend implementations.
            What: for label_studio: "external" or "managed_docker" (placeholder).
            Why: separates "you run LS yourself" from "SDK manages LS".
        url (Optional[str]):
            Base URL to the labeling system.
            Where: used by Label Studio backend.
            What: "http://localhost:8080".
            Why: tells the SDK where to call HTTP APIs.
        api_token (Optional[str]):
            Authentication token for the labeling system.
            Where: used by Label Studio backend.
            What: a secret string from Label Studio settings.
            Why: required to authenticate API calls.
        project_id (Optional[Union[str, int]]):
            Existing project id in the backend.
            Where: used by backend to reuse a project.
            What: an integer id or string.
            Why: avoids creating a new project each run.
        managed_port (int):
            Port for managed-docker mode (placeholder).
            Where: reserved for future.
            What: 8080 by default.
            Why: defines where the managed service would be exposed.
        managed_compose_path (Optional[str]):
            Path to a docker-compose file (placeholder).
            Where: reserved for future.
            What: filesystem path.
            Why: would allow customizing managed deployment.
        llm_model_name (Optional[str]):
            Placeholder for LLM backend settings.
            Where: reserved for future.
            What: model name like "gpt-4.1-mini" etc.
            Why: would configure automated labeling.
    """
    backend: str = "label_studio"
    mode: str = "external"
    url: Optional[str] = None
    api_token: Optional[str] = None
    project_id: Optional[Union[str, int]] = None
    managed_port: int = 8080
    managed_compose_path: Optional[str] = None
    llm_model_name: Optional[str] = None

    def validate(self) -> None:
        if self.backend not in {"label_studio", "llm", "custom"}:
            raise ConfigurationError(f"Unsupported label_backend_config.backend={self.backend!r}")
        if self.backend == "label_studio":
            if self.mode not in {"external", "managed_docker"}:
                raise ConfigurationError(f"Unsupported label_backend_config.mode={self.mode!r}")
            if self.mode == "external" and (not self.url or not self.api_token):
                raise ConfigurationError("External Label Studio requires url and api_token.")


@dataclass(frozen=True)
class PrelabelConfig:
    """
    Prelabeling settings.

    If enabled, the SDK sends model suggestions to the backend when pushing tasks.
    This can speed up human labeling, but requires backend support.

    Attributes:
        enable (bool):
            Turn prelabeling on/off.
            Where: used in engine `_step_push`.
            What: True means engine calls `_make_prelabels(...)` and passes result to backend.
            Why: gives annotators a suggested label to speed up work.
        min_confidence (float):
            Minimum confidence threshold for sending suggestions (placeholder).
            Where: not enforced in scaffold yet.
            What: in [0, 1].
            Why: lets you avoid sending low-quality suggestions.
    """
    enable: bool = False
    min_confidence: float = 0.0

    def validate(self) -> None:
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ConfigurationError("prelabel_config.min_confidence must be in [0, 1]")


@dataclass(frozen=True)
class StopCriteria:
    """
    When `project.run()` should stop.

    You can stop by:
    - number of labeled samples (budget)
    - number of rounds
    - metric plateau (no improvement for N evals)

    Attributes:
        max_labeled (Optional[int]):
            Stop after this many samples become LABELED.
            Where: checked by engine `_should_stop`.
            What: like a labeling budget.
            Why: active learning is usually limited by annotation cost.
        max_rounds (Optional[int]):
            Stop after this many rounds.
            Where: checked by engine `_should_stop`.
            What: integer.
            Why: limits iterations even if labels remain.
        metric_name (str):
            Which metric to inspect for plateau checks.
            Where: used in `_metric_plateau`.
            What: "accuracy" by default.
            Why: you might care about f1/auc instead.
        plateau_rounds (Optional[int]):
            How many recent evaluations to compare for plateau.
            Where: used in `_metric_plateau`.
            What: e.g. 3 means "no improvement over last 3 evals".
            Why: stops when the model stops improving.
        min_improvement (float):
            Minimum required improvement to count as progress.
            Where: used in `_metric_plateau`.
            What: 0.0 by default.
            Why: avoids stopping due to tiny random fluctuations.
    """
    max_labeled: Optional[int] = None
    max_rounds: Optional[int] = None
    metric_name: str = "accuracy"
    plateau_rounds: Optional[int] = None
    min_improvement: float = 0.0

    def validate(self) -> None:
        if self.max_labeled is not None and self.max_labeled < 0:
            raise ConfigurationError("stop_criteria.max_labeled must be >= 0")
        if self.max_rounds is not None and self.max_rounds < 0:
            raise ConfigurationError("stop_criteria.max_rounds must be >= 0")
        if self.plateau_rounds is not None and self.plateau_rounds < 1:
            raise ConfigurationError("stop_criteria.plateau_rounds must be >= 1")
