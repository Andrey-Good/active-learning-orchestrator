"""Model adapter protocols and implementations."""

from .base import ModelCapabilities, TextClassificationAdapter, inspect_model_capabilities
from .huggingface import HFSequenceClassifierAdapter

__all__ = [
    "ModelCapabilities",
    "TextClassificationAdapter",
    "inspect_model_capabilities",
    "HFSequenceClassifierAdapter",
]
