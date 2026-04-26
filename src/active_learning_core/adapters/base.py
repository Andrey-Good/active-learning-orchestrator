from __future__ import annotations

"""
Model adapter contracts.

The SDK does not require a specific ML library. Instead you provide an "adapter"
object that implements the methods in `TextClassificationAdapter`.

For juniors:
- If you have a scikit-learn model, you can write a small adapter around it.
- If you have a HuggingFace model, you can use/extend `HFSequenceClassifierAdapter`.
"""

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Sequence, runtime_checkable


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
        fit (bool):
            Where: checked during `engine.configure`.
            What: True if adapter has a callable `fit`.
            Why: engine needs to train/update the model after new labels.
        evaluate (bool):
            Where: checked during `engine.configure`.
            What: True if adapter has a callable `evaluate`.
            Why: engine records metrics and computes reward.
        embed (bool):
            Where: relevant for embedding-based strategies (not required for MVP uncertainty).
            What: True if adapter has a callable `embed`.
            Why: diversity strategies need embeddings.
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
    fit: bool
    evaluate: bool
    embed: bool
    get_model_id: bool
    save_load: bool


@runtime_checkable
class TextClassificationAdapter(Protocol):
    """
    Model adapter contract for text classification.

    MVP required methods:
    - predict_proba(texts): returns per-text class probabilities
    - fit(texts, labels): trains/updates the model
    - evaluate(texts, labels): returns a dict of metrics (e.g. {"accuracy": 0.9})

    Optional methods:
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

    def embed(self, texts: Sequence[str], batch_size: int = 32) -> Any:
        ...

    def get_model_id(self) -> str:
        ...

    def save(self, path: str) -> None:
        ...

    def load(self, path: str) -> None:
        ...


def inspect_model_capabilities(model: Any) -> ModelCapabilities:
    """
    Infer model adapter capabilities by checking which methods exist.

    The engine uses this to validate that the adapter supports MVP requirements.
    """

    return ModelCapabilities(
        predict_proba=callable(getattr(model, "predict_proba", None)),
        fit=callable(getattr(model, "fit", None)),
        evaluate=callable(getattr(model, "evaluate", None)),
        embed=callable(getattr(model, "embed", None)),
        get_model_id=callable(getattr(model, "get_model_id", None)),
        save_load=callable(getattr(model, "save", None)) and callable(getattr(model, "load", None)),
    )
