from typing import List, Dict, Any, Optional
from active_learning_core.state.sqlite_store import SQLiteStore

class Pool:
    """Управление пулом данных с привязкой к SQLite."""
    def __init__(self, store: SQLiteStore):
        self.store = store

    def add_samples(self, samples: List[Dict[str, Any]]):
        """Добавляет новые неразмеченные данные со статусом new_unseen."""
        self.store.add_samples(samples, initial_status="new_unseen")

    def get_samples_for_scoring(self) -> List[Dict[str, Any]]:
        """Возвращает данные, требующие инференса (new_unseen)."""
        return self.store.get_samples_by_status("new_unseen")

    def get_training_set(self, replay_ratio: float = 0.3) -> List[Dict[str, Any]]:
        """
        Формирует датасет: размеченные данные + Coreset из старых батчей[cite: 87].
        Решает проблему Catastrophic Forgetting[cite: 85].
        """
        new_labeled = self.store.get_samples_by_status("labeled")
        old_coreset = self.store.get_coreset(ratio=replay_ratio)
        return new_labeled + old_coreset