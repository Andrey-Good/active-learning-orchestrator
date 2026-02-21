"""
SDK exception hierarchy.

As a rule of thumb:
- Errors here are meant to be raised by the SDK and handled by the user.
- Use these types instead of raising bare `ValueError`/`RuntimeError` in SDK code.
"""


class ActiveLearningError(RuntimeError):
    """Base error type for the SDK."""


class ConfigurationError(ActiveLearningError):
    """Raised when configuration is invalid or incompatible."""


class DatasetMismatchError(ActiveLearningError):
    """Raised when an input dataset does not match the project's fingerprint."""


class ProjectLockedError(ActiveLearningError):
    """Raised when a project cannot acquire an exclusive lock."""


class StateCorruptedError(ActiveLearningError):
    """Raised when the project state cannot be loaded or validated."""


class InfrastructureError(ActiveLearningError):
    """Raised when local infrastructure is unavailable or misconfigured."""


class LabelBackendError(ActiveLearningError):
    """Raised for label backend communication or protocol errors."""


class StrategyError(ActiveLearningError):
    """Raised when a sampling strategy fails."""


class ModelAdapterError(ActiveLearningError):
    """Raised when a model adapter violates contract or fails at runtime."""


class StopCriteriaReached(ActiveLearningError):
    """Internal control-flow exception to stop the run loop cleanly."""
