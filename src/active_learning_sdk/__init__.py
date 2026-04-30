"""Active Learning SDK public package surface."""

from importlib import import_module
from typing import Any

from .adapters.base import ModelCapabilities, TextClassificationAdapter, inspect_model_capabilities
from .backends.base import LabelBackend, RoundProgress, RoundPullResult, RoundPushResult
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
from .engine import SelectionContext, StepResult, StrategyScheduler
from .exceptions import (
    ActiveLearningError,
    ConfigurationError,
    DatasetMismatchError,
    InfrastructureError,
    LabelBackendError,
    ModelAdapterError,
    ProjectLockedError,
    StateCorruptedError,
    StopCriteriaReached,
    StrategyError,
)
from .project import ActiveLearningProject
from .types import AnnotationRecord, DataSample, MetricRecord, ResolvedLabel, RoundStatus, SampleStatus, StepKind

_OPTIONAL_ROOT_EXPORTS = {
    "HFSequenceClassifierAdapter": ".adapters.huggingface",
    "SklearnTextClassifierAdapter": ".adapters.sklearn",
}


def __getattr__(name: str) -> Any:
    module_name = _OPTIONAL_ROOT_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = import_module(module_name, __name__)
    except ModuleNotFoundError as error:
        if name == "SklearnTextClassifierAdapter":
            raise ImportError(
                "SklearnTextClassifierAdapter requires the optional scikit-learn dependency. "
                "Install it with active-learning-sdk[sklearn]."
            ) from error
        if name == "HFSequenceClassifierAdapter":
            raise ImportError(
                "HFSequenceClassifierAdapter requires optional Hugging Face dependencies. "
                "Install them with active-learning-sdk[huggingface]."
            ) from error
        raise
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "ActiveLearningProject",
    "SelectionContext",
    "StepResult",
    "StrategyScheduler",
    "ActiveLearningError",
    "ConfigurationError",
    "DatasetMismatchError",
    "ProjectLockedError",
    "StateCorruptedError",
    "InfrastructureError",
    "LabelBackendError",
    "StrategyError",
    "ModelAdapterError",
    "StopCriteriaReached",
    "LabelSchema",
    "AnnotationPolicy",
    "SchedulerConfig",
    "CacheConfig",
    "FingerprintConfig",
    "SplitConfig",
    "LabelBackendConfig",
    "PrelabelConfig",
    "StopCriteria",
    "DataSample",
    "AnnotationRecord",
    "ResolvedLabel",
    "MetricRecord",
    "SampleStatus",
    "RoundStatus",
    "StepKind",
    "LabelBackend",
    "RoundPushResult",
    "RoundProgress",
    "RoundPullResult",
    "TextClassificationAdapter",
    "ModelCapabilities",
    "inspect_model_capabilities",
    "CacheStore",
    "InMemoryCacheStore",
    "JsonlDiskCacheStore",
    "PredictionCache",
    "EmbeddingCache",
    "SklearnTextClassifierAdapter",
    "HFSequenceClassifierAdapter",
]
