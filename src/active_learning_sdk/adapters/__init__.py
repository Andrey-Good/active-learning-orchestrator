"""Model adapter protocols and implementations."""

from typing import Any

from .base import ModelCapabilities, TextClassificationAdapter, inspect_model_capabilities

__all__ = [
    "ModelCapabilities",
    "TextClassificationAdapter",
    "inspect_model_capabilities",
    "HFSequenceClassifierAdapter",
    "SklearnTextClassifierAdapter",
]

_OPTIONAL_ADAPTERS = {
    "HFSequenceClassifierAdapter": ".huggingface",
    "SklearnTextClassifierAdapter": ".sklearn",
}


def __getattr__(name: str) -> Any:
    module_name = _OPTIONAL_ADAPTERS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module
    from importlib.util import find_spec

    if name == "SklearnTextClassifierAdapter":
        try:
            sklearn_available = find_spec("sklearn") is not None
        except ModuleNotFoundError as error:
            raise ImportError(
                "SklearnTextClassifierAdapter requires the optional scikit-learn dependency. "
                "Install it with active-learning-sdk[sklearn]."
            ) from error
        if not sklearn_available:
            raise ImportError(
                "SklearnTextClassifierAdapter requires the optional scikit-learn dependency. "
                "Install it with active-learning-sdk[sklearn]."
            )

    try:
        module = import_module(module_name, __name__)
    except ModuleNotFoundError as error:
        if name == "SklearnTextClassifierAdapter":
            raise ImportError(
                "SklearnTextClassifierAdapter requires the optional scikit-learn dependency. "
                "Install it with active-learning-sdk[sklearn]."
            ) from error
        raise
    value = getattr(module, name)
    globals()[name] = value
    return value
