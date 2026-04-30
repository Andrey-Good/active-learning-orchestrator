"""
Model adapter contracts.

The SDK does not require a specific ML library. Instead you provide an "adapter"
object that implements the methods in `TextClassificationAdapter`.

For juniors:
- If you have a scikit-learn model, you can write a small adapter around it.
- If you have a HuggingFace model, you can use/extend `HFSequenceClassifierAdapter`.
"""

from __future__ import annotations


from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, Sequence, runtime_checkable

_UNSUPPORTED_METHOD_ATTR = "__active_learning_sdk_unsupported_reason__"


def normalize_adapter_batch_size(batch_size: int, *, default: int = 32) -> int:
    try:
        return max(1, int(batch_size))
    except Exception:
        return default


@dataclass(frozen=True)
class ModelCapabilities:
    """
    A computed view of model adapter capabilities.

    This dataclass is created by `inspect_model_capabilities()` and used by the
    engine to decide which features can be enabled safely.

    Attributes:
        predict_proba (bool):
            Where: checked during `engine.configure`.
            What: True if adapter has a callable `predict_proba`.
            Why: uncertainty strategies need class probabilities.
        predict_logits (bool):
            Where: checked by strategies that need raw model logits.
            What: True if adapter has a callable `predict_logits`.
            Why: some advanced selectors operate before probability normalization.
        embed (bool):
            Where: checked by embedding-based strategies.
            What: True if adapter has a callable `embed`.
            Why: diversity strategies need embeddings.
        gradient_embed (bool):
            Where: checked by gradient-embedding strategies.
            What: True if adapter has a callable `gradient_embed`.
            Why: BADGE-style strategies need gradient embeddings.
        predict_stochastic (bool):
            Where: checked by stochastic prediction strategies.
            What: True if adapter has a callable `predict_stochastic`.
            Why: MC-dropout or ensemble uncertainty needs repeated predictions.
        predict_committee (bool):
            Where: checked by committee disagreement strategies.
            What: True if adapter has a callable `predict_committee`.
            Why: committee strategies need per-member probabilities.
        fit (bool):
            Where: checked during `engine.configure`.
            What: True if adapter has a callable `fit`.
            Why: engine needs to train/update the model after new labels.
        evaluate (bool):
            Where: checked during `engine.configure`.
            What: True if adapter has a callable `evaluate`.
            Why: engine records metrics and computes reward.
        get_model_id (bool):
            Where: used for cache correctness.
            What: True if adapter has a callable `get_model_id`.
            Why: separates cache entries between model versions.
        save_load (bool):
            Where: reserved for future reproducibility features.
            What: True if adapter has both `save` and `load`.
            Why: allows snapshotting models between rounds.
    """
    predict_proba: bool
    predict_logits: bool
    embed: bool
    gradient_embed: bool
    predict_stochastic: bool
    predict_committee: bool
    fit: bool
    evaluate: bool
    get_model_id: bool
    save_load: bool
    unsupported_methods: Dict[str, str] = field(default_factory=dict)


@runtime_checkable
class TextClassificationAdapter(Protocol):
    """
    Model adapter contract for text classification.

    Minimal engine methods:
    - fit(texts, labels): trains/updates the model
    - evaluate(texts, labels): returns a dict of metrics (e.g. {"accuracy": 0.9})

    Optional methods:
    - predict_proba(texts): required by probability strategies and prelabeling
    - get_model_id(): stable id used to version caches across training iterations
    - embed(): required by embedding-based strategies (not required for uncertainty)

    Attributes:
        (adapter-specific):
            Where: your adapter class stores the underlying model/tokenizer/featurizer.
            What: for example a scikit-learn estimator, a HuggingFace model, or any custom pipeline.
            Why: the engine and strategies only call the adapter methods and do not depend on ML libraries.
    """

    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> Any:
        ...

    def fit(self, texts: Sequence[str], labels: Sequence[Any], **kwargs) -> None:
        ...

    def evaluate(self, texts: Sequence[str], labels: Sequence[Any]) -> Dict[str, float]:
        ...


def unsupported_adapter_method(reason: str):
    """
    Mark an adapter method as an intentional scaffold placeholder.

    `inspect_model_capabilities()` treats decorated methods as unsupported so the
    engine can fail fast during configuration instead of crashing later at runtime.
    """

    def decorator(func):
        setattr(func, _UNSUPPORTED_METHOD_ATTR, reason)
        return func

    return decorator


def _get_method(model: Any, method_name: str) -> Any:
    try:
        return getattr(model, method_name, None)
    except Exception as error:
        return _CapabilityAccessError(method_name=method_name, error=error)


@dataclass(frozen=True)
class _CapabilityAccessError:
    method_name: str
    error: Exception


def _unsupported_reason(model: Any, method_name: str) -> str | None:
    method = _get_method(model, method_name)
    if isinstance(method, _CapabilityAccessError):
        return f"attribute access failed: {method.error.__class__.__name__}: {method.error}"
    if not callable(method):
        if isinstance(model, TextClassificationAdapter):
            return "not implemented on adapter"
        return "missing or not callable"

    func = getattr(method, "__func__", method)
    reason = getattr(func, _UNSUPPORTED_METHOD_ATTR, None)
    if reason is not None:
        return reason

    if func is TextClassificationAdapter.__dict__.get(method_name):
        return "not implemented on adapter"

    return None


def inspect_model_capabilities(model: Any) -> ModelCapabilities:
    """
    Infer model adapter capabilities by checking which methods exist.

    The engine uses this to validate that the adapter supports MVP requirements.
    """
    unsupported_methods: Dict[str, str] = {}

    def is_supported(method_name: str) -> bool:
        reason = _unsupported_reason(model, method_name)
        if reason is None:
            return True
        unsupported_methods[method_name] = reason
        return False

    save_reason = _unsupported_reason(model, "save")
    load_reason = _unsupported_reason(model, "load")
    if save_reason is not None:
        unsupported_methods["save"] = save_reason
    if load_reason is not None:
        unsupported_methods["load"] = load_reason

    return ModelCapabilities(
        predict_proba=is_supported("predict_proba"),
        predict_logits=is_supported("predict_logits"),
        embed=is_supported("embed"),
        gradient_embed=is_supported("gradient_embed"),
        predict_stochastic=is_supported("predict_stochastic"),
        predict_committee=is_supported("predict_committee"),
        fit=is_supported("fit"),
        evaluate=is_supported("evaluate"),
        get_model_id=is_supported("get_model_id"),
        save_load=save_reason is None and load_reason is None,
        unsupported_methods=unsupported_methods,
    )
