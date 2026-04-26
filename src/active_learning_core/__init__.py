"""
Active Learning Core SDK (Phase 1)
Mathematical core for Human-in-the-Loop ML processes.
"""

from .core.session import ALSession
from .core.pool import Pool
from .label_studio.client import LabelStudioClient

# Экспортируем стратегии и критерии остановки, чтобы клиент мог конфигурировать раунды
from .strategies.typiclust import TypiClustStrategy
from .strategies.badge import BADGEStrategy
from .stopping.pac_bayesian import PACBayesStopping

# Оставляем полезные исключения, адаптированные под наши реалии
from .exceptions import (
    ActiveLearningError,
    ConfigurationError,
    StateCorruptedError,      # Ошибка целостности SQLite (Crash recovery fail)
    LabelStudioClientError,   # Ошибка REST API или несовпадение Idempotency Key
    StopCriteriaReached,      # Успешный сигнал автостопа от PAC-Bayes
    ModelInferenceError,      # Ошибка при батчинге или OOM
    ScaleMismatchError        # Если клиент попытается использовать BADGE некорректно
)

__version__ = "1.4.0-dev"

__all__ = [
    # Core
    "ALSession",
    "Pool",
    "LabelStudioClient",

    # Strategies & Math
    "TypiClustStrategy",
    "BADGEStrategy",
    "PACBayesStopping",

    # Exceptions
    "ActiveLearningError",
    "ConfigurationError",
    "StateCorruptedError",
    "LabelStudioClientError",
    "StopCriteriaReached",
    "ModelInferenceError",
    "ScaleMismatchError"
]